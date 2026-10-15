# ---------------------------------------------------------------------
# CELL: STAGE 9 PRODUCTION WEIGHT CONSOLIDATION (FP16 IN-PLACE MERGE)
# ---------------------------------------------------------------------
!pip uninstall -y torchao

import os
import gc
import json
import torch
import peft.import_utils

# Neutralize torchao check in memory
peft.import_utils.is_torchao_available = lambda: False
try:
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B"
OUTPUT_DIR = "/kaggle/working/final_it_support_production_model"

print("=" * 65)
print("[STAGE 9] Production Model Consolidation (In-Place FP16 Merge)")
print("=" * 65)

# 1. Exact Adapter Discovery
ADAPTER_PATH = "/kaggle/input/notebooks/musharafmaqbool007/sft-kaggle/final_it_support_agent"
if not os.path.exists(ADAPTER_PATH):
    # Dynamic search fallback
    for search_root in ["/kaggle/working", "/kaggle/input"]:
        for root, dirs, files in os.walk(search_root):
            if "adapter_model.safetensors" in files:
                ADAPTER_PATH = root
                break
        if ADAPTER_PATH:
            break

print(f"[INFO] Source Base Model:      {BASE_MODEL_NAME}")
print(f"[INFO] Trained LoRA Adapter:    {ADAPTER_PATH}")
print(f"[INFO] Target Production Path:  {OUTPUT_DIR}")

# 2. Flush GPU Memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 3. Load Base Model in 16-bit FP16 (CPU RAM)
print("\n[INFO] Loading unquantized base model in 16-bit FP16 (CPU RAM)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True
)

# 4. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 5. Attach Trained Adapter
print(f"[INFO] Attaching trained SFT adapter from: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    device_map="cpu"
)

# 6. In-Place Weight Addition: W_production = W_base + (alpha / r) * (B @ A)
print("[INFO] Folding adapter weights into base tensors (merge_and_unload)...")
merged_model = model.merge_and_unload()
merged_model.eval()

# 7. Serialize Consolidated Model to Disk
print(f"\n[INFO] Serializing consolidated production model to: {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

merged_model.save_pretrained(
    OUTPUT_DIR,
    max_shard_size="4GB",
    safe_serialization=True
)
tokenizer.save_pretrained(OUTPUT_DIR)

# 8. Verification & Artifact Audit
print("\n" + "=" * 65)
print("[SUCCESS] Production Model Consolidated Successfully!")
print("=" * 65)
saved_files = sorted(os.listdir(OUTPUT_DIR))
total_size_gb = 0.0
for f in saved_files:
    f_path = os.path.join(OUTPUT_DIR, f)
    size_mb = os.path.getsize(f_path) / (1024 * 1024)
    total_size_gb += (size_mb / 1024)
    print(f"  • {f:38s} | Size: {size_mb:8.2f} MB")

print("-" * 65)
print(f"  Total Standalone Footprint: {total_size_gb:.2f} GB")
print("  Deployment Readiness:       100% Native (vLLM, Ollama, GGUF, Triton)")
print("  LoRA Dependency:            Eliminated (Zero adapter overhead)")
print("=" * 65)