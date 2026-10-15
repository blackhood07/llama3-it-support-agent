import os
import gc
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import PeftModel, prepare_model_for_kbit_training

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B"
MAX_SEQ_LENGTH = 1024

def locate_path(name, is_dir=False, search_root="/kaggle"):
    for root, dirs, files in os.walk(search_root):
        target = dirs if is_dir else files
        if name in target:
            return os.path.join(root, name)
    return None

train_file = locate_path("sft_train.jsonl", is_dir=False)
val_file = locate_path("sft_val.jsonl", is_dir=False)
cpt_adapter_dir = locate_path("cpt_adapter", is_dir=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def tokenize_dialogue(sample):
    messages = sample["messages"]
    sys_content, user_content, asst_content = "", "", ""
    for m in messages:
        role = m.get("role", "").lower().strip()
        content = m.get("content", "").strip()
        if role == "system": sys_content = content
        elif role == "user": user_content = content
        elif role == "assistant": asst_content = content

    prompt_str = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_content}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    asst_str = f"{asst_content}<|eot_id|>"

    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    asst_ids = tokenizer.encode(asst_str, add_special_tokens=False)

    if len(prompt_ids) + len(asst_ids) > MAX_SEQ_LENGTH:
        asst_ids = asst_ids[:max(MAX_SEQ_LENGTH - len(prompt_ids), 64)]

    full_ids = (prompt_ids + asst_ids)[:MAX_SEQ_LENGTH]
    labels = ([-100] * len(prompt_ids) + asst_ids)[:MAX_SEQ_LENGTH]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

raw_train = Dataset.from_list(load_jsonl(train_file))
raw_val = Dataset.from_list(load_jsonl(val_file))

train_ds = raw_train.map(tokenize_dialogue, remove_columns=["messages"]).filter(lambda s: any(l != -100 for l in s["labels"]))
val_ds = raw_val.map(tokenize_dialogue, remove_columns=["messages"]).filter(lambda s: any(l != -100 for l in s["labels"]))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.float16,
    attn_implementation="sdpa"
)

base_model = prepare_model_for_kbit_training(base_model)
base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

model = PeftModel.from_pretrained(base_model, cpt_adapter_dir, is_trainable=True)

training_args = TrainingArguments(
    output_dir="/kaggle/working/checkpoints_sft",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,    # PREVENTS 4.40 GiB EVAL OOM
    eval_accumulation_steps=1,       # PREVENTS VRAM BUILDUP DURING EVAL
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=40,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=40,
    save_strategy="steps",
    save_steps=40,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
)

# Automated Checkpoint Resumption
checkpoints = [
    os.path.join("/kaggle/working/checkpoints_sft", d)
    for d in os.listdir("/kaggle/working/checkpoints_sft")
    if d.startswith("checkpoint-")
] if os.path.exists("/kaggle/working/checkpoints_sft") else []

latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1] if checkpoints else None
if latest_checkpoint:
    print(f"[INFO] Resuming from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("[INFO] Starting clean training run from step 0...")
    trainer.train()

FINAL_OUTPUT = "/kaggle/working/final_it_support_agent"
model.save_pretrained(FINAL_OUTPUT)
tokenizer.save_pretrained(FINAL_OUTPUT)
print(f"[SUCCESS] Export complete: {FINAL_OUTPUT}")