"""
data_prep.py - Tokenization and Sequence Packing Engine for Continuous Pre-Training
Converts cleaned ticket data into packed 1024-token sequences for Llama-3.1-8B.
Saves DatasetDict directly to Google Drive for zero-redundancy training.
"""

import os
import random
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

BASE_DIR = "/content/drive/MyDrive/IT_Support_Ticket_Automation_LLM"
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_CSV_PATH = os.path.join(DATA_DIR, "cleaned_it_tickets_master.csv")
CORPUS_TXT_PATH = os.path.join(DATA_DIR, "cleaned_it_corpus.txt")
TOKENIZED_DIR = os.path.join(DATA_DIR, "tokenized_cpt_dataset")

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
FALLBACK_MODEL_ID = "unsloth/Meta-Llama-3.1-8B"
BLOCK_SIZE = 1024
TRAIN_RATIO = 0.95
RANDOM_SEED = 42

def get_hf_token():
    """Retrieves Hugging Face token from environment or Colab userdata."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    return token

def load_documents():
    """Loads and formats individual ticket records from clean CSV or text corpus."""
    if os.path.exists(CLEAN_CSV_PATH):
        df = pd.read_csv(CLEAN_CSV_PATH)
        documents = []
        for _, row in df.iterrows():
            doc = f"""================================================================================
ENTERPRISE INCIDENT MANAGEMENT RECORD: {row['ticket_id']}
CATEGORY: {row['category']} | SOURCE: {row['source']}
SUBJECT: {row['subject_title']}
TAGS: {row['tags']}
================================================================================

[1. USER INCIDENT REPORT]
{row['user_problem_description']}

[2. SYSTEM TELEMETRY / ERROR LOGS]
{row['error_logs_and_telemetry']}

[3. VERIFIED RESOLUTION & REMEDIATION PLAN]
{row['accepted_resolution']}
================================================================================
"""
            documents.append(doc.strip())
        return documents

    if os.path.exists(CORPUS_TXT_PATH):
        with open(CORPUS_TXT_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()
        import re
        pattern = r'=+\s+ENTERPRISE INCIDENT MANAGEMENT RECORD:.*?(?==+\s+ENTERPRISE INCIDENT MANAGEMENT RECORD:|\Z)'
        return [doc.strip() for doc in re.findall(pattern, raw_text, re.DOTALL)]

    raise FileNotFoundError("Neither cleaned CSV nor corpus text file found in data directory.")

def pack_sequences(tokenized_docs, block_size, eos_token_id):
    """
    Concatenates tokenized documents separated by EOS and chunks into
    contiguous block_size sequences. Guarantees 100% compute density.
    """
    all_tokens = []
    for doc_ids in tokenized_docs:
        all_tokens.extend(doc_ids)
        all_tokens.append(eos_token_id)
        
    total_tokens = len(all_tokens)
    num_blocks = total_tokens // block_size
    
    if num_blocks == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}
        
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        chunk = all_tokens[start_idx:end_idx]
        
        input_ids_list.append(chunk)
        attention_mask_list.append([1] * block_size)
        labels_list.append(chunk.copy())
        
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

def run_data_prep():
    print("=" * 65, flush=True)
    print("[PIPELINE] Stage 3: Data Tokenization & Sequence Packing", flush=True)
    print("=" * 65, flush=True)

    token = get_hf_token()
    
    # Load tokenizer with graceful fallback
    print(f"[INFO] Initializing tokenizer: {MODEL_ID}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, use_fast=True)
    except Exception as e:
        print(f"[WARN] Accessing {MODEL_ID} failed ({e}). Attempting fallback {FALLBACK_MODEL_ID}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID, token=token, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    eos_id = tokenizer.eos_token_id

    # Ingest discrete documents
    print("[INFO] Ingesting enterprise incident records...", flush=True)
    documents = load_documents()
    print(f"[INFO] Loaded {len(documents):,} structured incident records.", flush=True)

    # 95% Train / 5% Validation split
    random.seed(RANDOM_SEED)
    random.shuffle(documents)
    split_idx = int(len(documents) * TRAIN_RATIO)
    train_docs = documents[:split_idx]
    val_docs = documents[split_idx:]
    print(f"[INFO] Dataset partition: {len(train_docs):,} train records | {len(val_docs):,} validation records", flush=True)

    # Tokenize
    print("[INFO] Tokenizing documents...", flush=True)
    train_tokenized = [tokenizer(doc, add_special_tokens=False)["input_ids"] for doc in train_docs]
    val_tokenized = [tokenizer(doc, add_special_tokens=False)["input_ids"] for doc in val_docs]

    # Pack sequences into contiguous 1024-token blocks
    print(f"[INFO] Packing tokens into contiguous {BLOCK_SIZE}-token sequences...", flush=True)
    train_packed = pack_sequences(train_tokenized, BLOCK_SIZE, eos_id)
    val_packed = pack_sequences(val_tokenized, BLOCK_SIZE, eos_id)

    train_ds = Dataset.from_dict(train_packed)
    val_ds = Dataset.from_dict(val_packed)
    cpt_dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    # Save to Drive
    print(f"[INFO] Saving pre-tokenized DatasetDict to: {TOKENIZED_DIR}...", flush=True)
    os.makedirs(TOKENIZED_DIR, exist_ok=True)
    cpt_dataset.save_to_disk(TOKENIZED_DIR)

    total_packed_tokens = (len(train_ds) + len(val_ds)) * BLOCK_SIZE
    print("=" * 65, flush=True)
    print("[SUCCESS] Sequence Packing Complete", flush=True)
    print(f"  - Target Directory:       {TOKENIZED_DIR}", flush=True)
    print(f"  - Sequence Length (L):    {BLOCK_SIZE} tokens", flush=True)
    print(f"  - Train Blocks:           {len(train_ds):,} ({len(train_ds) * BLOCK_SIZE:,} tokens)", flush=True)
    print(f"  - Validation Blocks:      {len(val_ds):,} ({len(val_ds) * BLOCK_SIZE:,} tokens)", flush=True)
    print(f"  - Total Packed Tokens:    {total_packed_tokens:,} tokens", flush=True)
    print(f"  - Packing Efficiency:     100.0% (Zero pad tokens)", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    run_data_prep()
