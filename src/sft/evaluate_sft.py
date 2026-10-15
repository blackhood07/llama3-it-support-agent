import os
import gc
import json
import math
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import PeftModel
from torch.utils.data import DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("=" * 65)
print("[STEP 1] Evaluating Held-Out Validation Dialogues (167 Samples)")
print("=" * 65)

# 1. Discover Artifacts across /kaggle/working and /kaggle/input
def locate_path(name, is_dir=False, search_roots=["/kaggle/working", "/kaggle/input"]):
    for root_dir in search_roots:
        if not os.path.exists(root_dir):
            continue
        for root, dirs, files in os.walk(root_dir):
            target = dirs if is_dir else files
            if name in target:
                return os.path.join(root, name)
    return None

val_file = locate_path("sft_val.jsonl", is_dir=False)
if not val_file:
    raise FileNotFoundError("Could not locate sft_val.jsonl in /kaggle.")

model_dir = locate_path("final_it_support_agent", is_dir=True)
if not model_dir:
    chk_root = locate_path("checkpoints_sft", is_dir=True)
    if chk_root:
        checkpoints = [os.path.join(chk_root, d) for d in os.listdir(chk_root) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            model_dir = checkpoints[-1]

if not model_dir:
    raise FileNotFoundError("Could not locate final_it_support_agent or checkpoint directory.")

print(f"[FOUND] Model Weights:    {model_dir}")
print(f"[FOUND] Validation Data:  {val_file}")

# 2. Tokenizer & ChatML Formatting
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B"
MAX_SEQ_LENGTH = 1024

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

raw_val = Dataset.from_list(load_jsonl(val_file))
val_ds = raw_val.map(tokenize_dialogue, remove_columns=["messages"]).filter(lambda s: any(l != -100 for l in s["labels"]))

# 3. Load Model in 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("\n[INFO] Loading base model on GPU 0...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.float16,
    attn_implementation="sdpa"
)

print(f"[INFO] Attaching trained SFT adapter from: {model_dir}")
model = PeftModel.from_pretrained(base_model, model_dir)
model.eval()

# 4. Evaluate Next-Token Accuracy on Assistant Tokens (labels != -100)
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")
val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collator, shuffle=False)

total_loss = 0.0
total_active_tokens = 0
correct_predictions = 0

print(f"[INFO] Evaluating {len(val_loader)} validation dialogues...")
with torch.no_grad():
    for step, batch in enumerate(val_loader, 1):
        input_ids = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")
        labels = batch["labels"].to("cuda:0")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        active_mask = (shift_labels != -100)
        preds = torch.argmax(shift_logits, dim=-1)

        correct = (preds[active_mask] == shift_labels[active_mask]).sum().item()
        active_count = active_mask.sum().item()

        correct_predictions += correct
        total_active_tokens += active_count

        if step % 35 == 0 or step == len(val_loader):
            print(f"       Evaluated [{step:3d}/{len(val_loader)}] dialogues...")

avg_loss = total_loss / len(val_loader)
perplexity = math.exp(min(avg_loss, 20.0))
accuracy = (correct_predictions / total_active_tokens) * 100

metrics = {
    "model_path": model_dir,
    "evaluated_samples": len(val_loader),
    "total_active_tokens": total_active_tokens,
    "correct_tokens": correct_predictions,
    "sft_accuracy": round(accuracy, 2),
    "sft_val_loss": round(avg_loss, 4),
    "sft_val_perplexity": round(perplexity, 2),
    "cpt_accuracy_baseline": 73.12,
    "cpt_perplexity_baseline": 2.81,
    "gain_vs_cpt_pct": round(accuracy - 73.12, 2)
}

metrics_out = "/kaggle/working/sft_benchmark_metrics.json"
with open(metrics_out, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\n" + "=" * 65)
print("[BENCHMARK REPORT] SFT Final Generalization Metrics")
print("=" * 65)
print(f"  • Total Evaluated Assistant Tokens: {total_active_tokens:,}")
print(f"  • Correctly Predicted Tokens:       {correct_predictions:,}")
print(f"  • SFT Next-Token Target Accuracy:   {accuracy:.2f}%")
print(f"  • Final SFT Validation Loss:        {avg_loss:.4f}")
print(f"  • Final SFT Validation Perplexity:  {perplexity:.2f}")
print(f"  • Stage 4 CPT Baseline:             73.12% Accuracy | 2.81 Perplexity")
print(f"  • Net Performance Gain:             +{accuracy - 73.12:.2f}%")
print(f"[SUCCESS] Metrics exported to: {metrics_out}")
print("=" * 65)