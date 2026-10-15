# Local Deployment with Ollama

This directory contains the operational deployment artifacts for running the fine-tuned **Enterprise IT Support & SRE Triage Agent** locally or on edge infrastructure using [Ollama](https://ollama.com).

## 1. Quickstart

### Step 1: Place or Link Adapter / Merged Model
Ensure your trained LoRA adapter (or merged standalone weights) are placed in the directory referenced by `Modelfile`:
```bash
# If using LoRA adapter:
ADAPTER ./final_it_support_agent
```

### Step 2: Build the Ollama Model
From this directory, run:
```bash
ollama create llama3-it-support -f Modelfile
```

### Step 3: Run Interactive Triage Session
```bash
ollama run llama3-it-support
```

## 2. Example Prompt

```text
>>> [INCIDENT REPORT]
Host: k8s-worker-node-04
Error: OOMKilled on payments-gateway-pod. Exit code 137. Container restart count: 14 in last 1 hour.
Investigate and remediate.
```

The model will respond following the strict **5-Point SRE Incident Triage Schema**:
1. **Severity & Categorization** (e.g., P1 - Critical / Pod Infrastructure)
2. **Root Cause Analysis (RCA)** (Failure mechanism & memory limits analysis)
3. **Actionable Remediation Runbook** (kubectl / docker commands to inspect & patch resource limits)
4. **Post-Remediation Verification** (Diagnostic probes to confirm container health)
5. **User-Facing Resolution Summary** (Executive summary for stakeholders)
