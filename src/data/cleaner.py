"""
cleaner.py - Production Cleaning and Normalization Pipeline
Cleans raw_it_tickets.csv, isolates code/telemetry, deduplicates,
and exports cleaned_it_tickets_master.csv and cleaned_it_corpus.txt.
"""

import os
import re
import html
import hashlib
import pandas as pd
from bs4 import BeautifulSoup

BASE_DIR = "/content/drive/MyDrive/IT_Support_Ticket_Automation_LLM"
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
RAW_CSV_PATH = os.path.join(DATA_DIR, "raw_it_tickets.csv")
CLEAN_CSV_PATH = os.path.join(DATA_DIR, "cleaned_it_tickets_master.csv")
CORPUS_TXT_PATH = os.path.join(DATA_DIR, "cleaned_it_corpus.txt")

def clean_html_and_extract_logs(raw_html):
    """
    Extracts multi-line logs, configs, and shell traces into telemetry,
    preserves inline terms, and unescapes HTML entities.
    """
    if not raw_html or not isinstance(raw_html, str):
        return "", ""
        
    soup = BeautifulSoup(raw_html, "html.parser")
    
    # 1. Extract <pre> blocks (stack traces, configs, shell outputs)
    logs_and_code = []
    for pre in soup.find_all("pre"):
        snippet = pre.get_text().strip()
        if len(snippet) > 10:
            logs_and_code.append(snippet)
        pre.decompose()

    # 2. Extract multi-line code blocks
    for code in soup.find_all("code"):
        snippet = code.get_text().strip()
        if "\n" in snippet or len(snippet) > 80:
            logs_and_code.append(snippet)
            code.decompose()
        
    extracted_logs = "\n---\n".join(logs_and_code)
    
    # 3. Extract prose
    prose = soup.get_text(separator=" ")
    prose = html.unescape(prose)
    
    # 4. Strip trailing email/forum sign-offs
    prose = re.sub(r'(?i)\n\s*(?:thanks in advance|regards|best regards|cheers|sincerely)[\s\S]*$', '', prose)
    prose = re.sub(r'\s+', ' ', prose).strip()
    
    return prose, extracted_logs

def hash_content(text):
    return hashlib.md5(text.lower().encode("utf-8")).hexdigest()

def run_cleaner():
    print("=" * 65, flush=True)
    print("[PIPELINE] Stage 2: Data Cleaning and Normalization", flush=True)
    print("=" * 65, flush=True)
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"[ERROR] Source file not found: {RAW_CSV_PATH}", flush=True)
        return

    raw_df = pd.read_csv(RAW_CSV_PATH)
    print(f"[INFO] Loaded {len(raw_df):,} raw records from {RAW_CSV_PATH}", flush=True)

    cleaned_rows = []
    seen_hashes = set()
    dropped_short = 0
    dropped_duplicates = 0

    for idx, row in raw_df.iterrows():
        title = html.unescape(str(row.get("raw_title", "") if not pd.isna(row.get("raw_title")) else "")).strip()
        q_html = str(row.get("raw_question_html", "") if not pd.isna(row.get("raw_question_html")) else "")
        a_html = str(row.get("raw_accepted_answer_html", "") if not pd.isna(row.get("raw_accepted_answer_html")) else "")
        category = str(row.get("target_category", "General IT Support") if not pd.isna(row.get("target_category")) else "General IT Support")
        source = str(row.get("source_community", "IT Community") if not pd.isna(row.get("source_community")) else "IT Community")
        tags = str(row.get("tags", "") if not pd.isna(row.get("tags")) else "")

        # Separate explanation prose from telemetry/code
        clean_problem, error_logs = clean_html_and_extract_logs(q_html)
        clean_solution, fix_scripts = clean_html_and_extract_logs(a_html)

        # Quality Gate 1: Length threshold
        if len(clean_problem) < 30 or len(clean_solution) < 25:
            dropped_short += 1
            continue

        # Quality Gate 2: Deduplication
        content_hash = hash_content(title + clean_problem[:100])
        if content_hash in seen_hashes:
            dropped_duplicates += 1
            continue
        seen_hashes.add(content_hash)

        # Consolidate resolution
        final_resolution = clean_solution
        if fix_scripts:
            final_resolution += "\n\n[Remediation Script / Config]:\n" + fix_scripts

        # Estimate tokens (~1.3 tokens per word)
        combined_text = f"{title} {clean_problem} {error_logs} {final_resolution}"
        token_count = int(len(combined_text.split()) * 1.3)

        cleaned_rows.append({
            "ticket_id": f"IT-{len(cleaned_rows)+1:06d}",
            "source": source,
            "category": category,
            "subject_title": title,
            "user_problem_description": clean_problem,
            "error_logs_and_telemetry": error_logs if error_logs else "No explicit stack trace attached.",
            "accepted_resolution": final_resolution,
            "tags": tags,
            "token_estimate": token_count
        })

    clean_df = pd.DataFrame(cleaned_rows)
    print("\n[REPORT] Data Quality and Filtering Summary:", flush=True)
    print(f"  - Discarded (insufficient length): {dropped_short:,}", flush=True)
    print(f"  - Discarded (duplicate records):   {dropped_duplicates:,}", flush=True)
    print(f"  - Verified Clean Records:          {len(clean_df):,}", flush=True)

    # 1. Save Clean Master CSV
    clean_df.to_csv(CLEAN_CSV_PATH, index=False, encoding="utf-8")
    csv_size_mb = os.path.getsize(CLEAN_CSV_PATH) / (1024 * 1024)
    print(f"\n[OUTPUT] Master CSV saved: {CLEAN_CSV_PATH} ({csv_size_mb:.2f} MB)", flush=True)

    # 2. Build Continuous Text Corpus for Phase 1 CPT
    print(f"[INFO] Compiling continuous domain corpus...", flush=True)
    with open(CORPUS_TXT_PATH, "w", encoding="utf-8") as f_txt:
        for _, row in clean_df.iterrows():
            formatted_entry = f"""================================================================================
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
            f_txt.write(formatted_entry + "\n\n")

    corpus_size_mb = os.path.getsize(CORPUS_TXT_PATH) / (1024 * 1024)
    total_tokens = clean_df["token_estimate"].sum()

    print("=" * 65, flush=True)
    print("[SUCCESS] Data Cleaning Complete", flush=True)
    print(f"  - Clean Dataset:  {CLEAN_CSV_PATH}", flush=True)
    print(f"  - Text Corpus:    {CORPUS_TXT_PATH} ({corpus_size_mb:.2f} MB)", flush=True)
    print(f"  - Domain Tokens:  ~{total_tokens:,} tokens", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    run_cleaner()
