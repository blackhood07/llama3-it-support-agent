# =====================================================================
# STEP 1: PRODUCTION MODEL INCIDENT RESPONSE VERIFICATION
# =====================================================================
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "/kaggle/working/final_it_support_production_model"

print("=" * 65)
print("[STEP 1] Initializing Standalone Production Model for Inference")
print("=" * 65)

# 1. Flush GPU Memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Load Model with 8-bit Quantization (Consumes ~8.5 GiB VRAM on GPU 0)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

print("[INFO] Loading production model on GPU 0 (~8.5 GiB VRAM)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.float16
)
model.eval()

# 4. Formulate Mission-Critical Enterprise Incident Prompt (ChatML)
system_prompt = (
    "You are an enterprise Level-3 IT Incident Response and SRE Specialist. "
    "Diagnose infrastructure failures systematically using a deterministic 5-point triage schema: "
    "1. Incident Severity & Blast Radius, 2. Root Cause Analysis (RCA), "
    "3. Actionable Remediation Runbook, 4. Verification Probes, and 5. User Communication."
)

incident_input = """[INCIDENT TELEMETRY]
Application: production-payment-gateway-service
Environment: Kubernetes Cluster (us-east-1)
Status: CrashLoopBackOff
Pod Events & Diagnostic Log:
  Warning  BackOff    3m   kubelet   Back-off restarting failed container payment-worker in pod payment-gateway-7f89d4-c9x2
  Normal   Killing    3m   kubelet   Container payment-worker failed liveness probe, will be restarted
  Warning  OOMKilled  3m   kernel    Task 'java' pid 1 killed by cgroup OOM killer: memory cgroup limit 512Mi exceeded (usage 512Mi).
Current Resource Spec:
  resources:
    limits:
      memory: "512Mi"
      cpu: "1000m"
    requests:
      memory: "256Mi"
      cpu: "500m"
"""

formatted_prompt = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    f"<|start_header_id|>user<|end_header_id|>\n\n{incident_input}<|eot_id|>"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# 5. Execute Generation
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")

print("[INFO] Generating Level-3 SRE Triage Runbook...\n")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.2,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.encode("<|eot_id|>")[-1]
    )

# Decode response tokens
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print("=" * 65)
print("[AUTONOMOUS INCIDENT RESPONSE OUTPUT]")
print("=" * 65)
print(response_text)
print("=" * 65)