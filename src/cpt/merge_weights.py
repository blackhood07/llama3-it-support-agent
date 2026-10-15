"""
merge_weights.py - Standalone Production Weight Merging Pipeline
Consolidates 4-bit trained LoRA adapter into full 16-bit FP16 base model weights.
Outputs standard ~16 GB model.safetensors directly to Google Drive.
"""

import os
import gc
import torch
import shutil

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = "/content/drive/MyDrive/IT_Support_Ticket_Automation_LLM"
ADAPTER_PATH = os.path.join(BASE_DIR, "final_adapter")
MERGED_DIR = os.path.join(BASE_DIR, "merged_production_model")
OFFLOAD_DIR = "/content/offload"

# Public, ungated 16-bit base model repository
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B"

def run_merge():
    print("=" * 65, flush=True)
    print("[PIPELINE] Stage 6: Standalone Production Weight Merging", flush=True)
    print("=" * 65, flush=True)

    os.makedirs(MERGED_DIR, exist_ok=True)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    # Purge VRAM before merging
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[INFO] Source Base Model:  {BASE_MODEL}", flush=True)
    print(f"[INFO] Source LoRA Adapter: {ADAPTER_PATH}", flush=True)
    print(f"[INFO] Destination Target:  {MERGED_DIR}", flush=True)
    print(f"[INFO] Loading base model in 16-bit FP16 with hybrid memory map...", flush=True)

    # Load 16-bit FP16 base model with memory partitioning to prevent OOM
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "13.0GiB", "cpu": "10.0GiB"},
        offload_folder=OFFLOAD_DIR,
        low_cpu_mem_usage=True
    )

    print("[INFO] Loading trained PEFT LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        offload_folder=OFFLOAD_DIR
    )

    print("[INFO] Executing mathematical weight merge (merge_and_unload)...", flush=True)
    merged_model = model.merge_and_unload()

    print(f"[INFO] Serializing consolidated weights to: {MERGED_DIR}...", flush=True)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)

    print("[INFO] Saving tokenizer configuration...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(MERGED_DIR)

    # Clean temporary offload cache
    if os.path.exists(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR, ignore_errors=True)

    # Calculate final disk size on Drive
    total_size_gb = sum(
        os.path.getsize(os.path.join(MERGED_DIR, f))
        for f in os.listdir(MERGED_DIR)
        if os.path.isfile(os.path.join(MERGED_DIR, f))
    ) / (1024 ** 3)

    print("=" * 65, flush=True)
    print("[SUCCESS] Production Model Merged and Serialized Successfully", flush=True)
    print(f"  - Target Directory:       {MERGED_DIR}", flush=True)
    print(f"  - Total Standalone Size:  {total_size_gb:.2f} GB", flush=True)
    print(f"  - Weight Format:          Standard Hugging Face safetensors (FP16)", flush=True)
    print(f"  - Zero-Dependency Status: Standalone model (No PEFT/LoRA needed)", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    run_merge()
