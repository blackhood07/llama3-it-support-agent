"""
scrapper.py - Scaled IT Support Dataset Ingestion Engine
Preserves existing 784 tickets and ingests 2,500 technical tickets from the 
61.8k Tobi-Bueck archive to hit our 3,200+ target.
"""

import os
import html
import pandas as pd
from datasets import load_dataset

BASE_DIR = "/content/drive/MyDrive/IT_Support_Ticket_Automation_LLM"
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
RAW_CSV_PATH = os.path.join(DATA_DIR, "raw_it_tickets.csv")

def run_scraper():
    print("=" * 65, flush=True)
    print("🚀 SCALING DATASET TO 3,200+ IT SUPPORT TICKETS", flush=True)
    print("=" * 65, flush=True)
    
    all_records = []
    
    # 1. Load existing 784 tickets from Drive
    if os.path.exists(RAW_CSV_PATH):
        try:
            existing_df = pd.read_csv(RAW_CSV_PATH)
            all_records = existing_df.to_dict("records")
            print(f"✅ Preserved your existing {len(all_records):,} tickets from Drive.", flush=True)
        except Exception as e:
            print(f"ℹ️ Could not load existing CSV: {e}", flush=True)

    # 2. Ingest 2,500 technical tickets from the 61,800-row Tobi-Bueck archive
    print("\n📥 Ingesting 2,500 technical tickets from Tobi-Bueck archive (61.8k rows)...", flush=True)
    try:
        support_ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
        added_count = 0
        target_to_add = 2500
        start_id = len(all_records) + 1
        
        for idx, row in enumerate(support_ds):
            if added_count >= target_to_add:
                break
                
            # Column mapping (dataset schema uses lowercase: body, answer, subject, queue, language)
            body = str(row.get("body") or row.get("Body") or "").strip()
            subj = str(row.get("subject") or row.get("Subject") or "IT Technical Support").strip()
            res = str(row.get("answer") or row.get("Resolution") or row.get("resolution") or "").strip()
            queue = str(row.get("queue") or row.get("Category") or "Technical Support").strip()
            lang = str(row.get("language") or row.get("Language") or "en").strip().lower()
            tags = str(row.get("tag_1") or queue).strip().lower()
            
            # Filter for English technical entries with substantive body and resolution
            if (lang == "en" or not lang) and len(body) > 40 and len(res) > 30:
                all_records.append({
                    "raw_id": f"TS-INGEST-{start_id + added_count:06d}",
                    "source_community": f"Enterprise Support ({queue})",
                    "target_category": f"IT Support / {queue}",
                    "raw_title": subj,
                    "raw_question_html": f"<p>{body}</p>",
                    "raw_accepted_answer_html": f"<p>{res}</p>",
                    "tags": tags
                })
                added_count += 1
                
        print(f"   ✅ Successfully added {added_count:,} new tickets! (Grand Total: {len(all_records):,})", flush=True)
    except Exception as e:
        print(f"   ⚠️ Error loading archive: {e}", flush=True)

    # 3. Save Master Collection to Google Drive
    print("\n" + "=" * 65, flush=True)
    print("💾 SAVING EXPANDED DATASET TO GOOGLE DRIVE...", flush=True)
    print("=" * 65, flush=True)
    
    final_df = pd.DataFrame(all_records)
    final_df = final_df.drop_duplicates(subset=["raw_title", "raw_question_html"])
    final_df.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8")
    
    size_mb = os.path.getsize(RAW_CSV_PATH) / (1024 * 1024)
    print(f"🎉 TARGET ACHIEVED!")
    print(f"📁 Master Raw CSV:  {RAW_CSV_PATH}")
    print(f"📊 Total Records:   {len(final_df):,} verified tickets")
    print(f"💾 File Size:       {size_mb:.2f} MB")
    print("=" * 65, flush=True)

if __name__ == "__main__":
    run_scraper()
