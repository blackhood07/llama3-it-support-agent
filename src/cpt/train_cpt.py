"""
train_cpt.py - Production Continuous Pre-Training Engine for IT Domain Llama-3.1-8B
Engineered for 1x NVIDIA T4 GPU (16GB VRAM) on Google Colab with 100GB Drive storage.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import glob
import math
import json
import torch
import numpy as np

# Purge any residual VRAM from prior notebook executions
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# Workspace directories
BASE_DIR = "/content/drive/MyDrive/IT_Support_Ticket_Automation_LLM"
DATA_DIR = os.path.join(BASE_DIR, "data")
TOKENIZED_DIR = os.path.join(DATA_DIR, "tokenized_cpt_dataset")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
FINAL_ADAPTER_DIR = os.path.join(BASE_DIR, "final_adapter")
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "training_history.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
FALLBACK_MODEL_ID = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

# 731 train blocks / 16 per step = ~46 steps/epoch -> 100 steps covers ~2.18 full epochs
MAX_STEPS = 100
PER_DEVICE_BATCH_SIZE = 1          # Micro-batch size 1 minimizes peak activation VRAM
GRADIENT_ACCUMULATION_STEPS = 16   # 1 * 16 = 16 sequences (16,384 tokens per optimizer step)
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 5
EVAL_STEPS = 25
SAVE_STEPS = 25
SAVE_TOTAL_LIMIT = 5

def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    return token

def find_latest_checkpoint(checkpoint_dir):
    ckpt_dirs = [
        d for d in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
        if os.path.isdir(d)
    ]
    if not ckpt_dirs:
        return None
    try:
        latest = max(ckpt_dirs, key=lambda x: int(os.path.basename(x).split("-")[-1]))
        return latest
    except Exception:
        return None

class MetricsHistoryCallback(TrainerCallback):
    """Persists training and evaluation metrics to JSON on Google Drive."""
    def __init__(self, history_file):
        self.history_file = history_file
        self.history = {
            "train_steps": [],
            "train_loss": [],
            "eval_steps": [],
            "eval_loss": [],
            "eval_perplexity": [],
            "eval_accuracy": [],
            "learning_rate": []
        }
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    self.history = json.load(f)
            except Exception:
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.history["train_steps"].append(step)
            self.history["train_loss"].append(float(logs["loss"]))
            if "learning_rate" in logs:
                self.history["learning_rate"].append(float(logs["learning_rate"]))
        if "eval_loss" in logs:
            self.history["eval_steps"].append(step)
            eval_loss = float(logs["eval_loss"])
            self.history["eval_loss"].append(eval_loss)
            self.history["eval_perplexity"].append(float(math.exp(min(eval_loss, 20.0))))
            if "eval_accuracy" in logs:
                self.history["eval_accuracy"].append(float(logs["eval_accuracy"]))
        
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

def preprocess_logits_for_metrics(logits, labels):
    """Extracts argmax predictions per batch to compute accuracy without VRAM bloat."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    shift_preds = predictions[:, :-1]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if np.sum(mask) == 0:
        return {"accuracy": 0.0}
    accuracy = np.sum((shift_preds == shift_labels) & mask) / np.sum(mask)
    return {"accuracy": float(accuracy)}

def run_training():
    print("=" * 65, flush=True)
    print("[PIPELINE] Stage 4: Resilient QLoRA Continuous Pre-Training", flush=True)
    print("=" * 65, flush=True)

    token = get_hf_token()

    # 1. Load Tokenized Dataset from Drive
    print(f"[INFO] Loading pre-tokenized dataset from: {TOKENIZED_DIR}", flush=True)
    if not os.path.exists(TOKENIZED_DIR):
        print(f"[ERROR] Tokenized dataset not found at {TOKENIZED_DIR}. Run data_prep.py first!", flush=True)
        return
    dataset = DatasetDict.load_from_disk(TOKENIZED_DIR)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    print(f"[INFO] Train blocks: {len(train_ds):,} | Validation blocks: {len(val_ds):,}", flush=True)

    # 2. Tokenizer Initialization
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID, token=token, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. 4-bit Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 4. Model Loading
    print(f"[INFO] Loading base model architecture...", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16,
            token=token
        )
    except Exception as e:
        print(f"[WARN] Accessing {MODEL_ID} failed ({e}). Loading non-gated mirror {FALLBACK_MODEL_ID}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16,
            token=token
        )

    # 5. Enable Gradient Checkpointing with non-reentrant mode
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # 6. LoRA Adapter Setup
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        dataloader_num_workers=2,
        load_best_model_at_end=False
    )

    history_callback = MetricsHistoryCallback(HISTORY_PATH)

    # 8. Trainer Initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        callbacks=[history_callback],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # 9. Auto-Resume from Checkpoint
    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"[INFO] Found checkpoint on Drive: {latest_checkpoint}. Resuming training...", flush=True)
        resume_arg = latest_checkpoint
    else:
        print("[INFO] No existing checkpoint found. Starting fresh CPT run...", flush=True)
        resume_arg = None

    # 10. Start Training
    trainer.train(resume_from_checkpoint=resume_arg)

    # 11. Save Final Adapter
    print(f"\n[INFO] Saving final adapter weights to: {FINAL_ADAPTER_DIR}...", flush=True)
    trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    eval_metrics = trainer.evaluate()
    final_ppl = math.exp(min(eval_metrics.get("eval_loss", 0.0), 20.0))
    final_acc = eval_metrics.get("eval_accuracy", 0.0)

    print("=" * 65, flush=True)
    print("[SUCCESS] Continuous Pre-Training Complete", flush=True)
    print(f"  - Final Evaluation Loss:       {eval_metrics.get('eval_loss', 0.0):.4f}", flush=True)
    print(f"  - Final Perplexity:            {final_ppl:.2f}", flush=True)
    print(f"  - Next-Token Accuracy:         {final_acc * 100:.2f}%", flush=True)
    print(f"  - LoRA Adapter Output:         {FINAL_ADAPTER_DIR}", flush=True)
    print(f"  - Metrics History Log:         {HISTORY_PATH}", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    run_training()
