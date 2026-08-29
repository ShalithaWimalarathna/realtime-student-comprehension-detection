"""
KIU Research — SUS Response Merger
====================================
Merges all individual participant response CSV files
into one master file ready for statistical analysis.

Usage:
    1. Put all sus_response_*.csv files into a folder called  sus_responses/
    2. Run:  python merge_sus_responses.py
    3. Output: sus_responses/MASTER_responses.csv

Then use MASTER_responses.csv for your statistical analysis.
"""

import os
import csv
import glob
from pathlib import Path

INPUT_FOLDER  = "sus_responses"
OUTPUT_FILE   = "sus_responses/MASTER_responses.csv"

def merge():
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    files = sorted(glob.glob(f"{INPUT_FOLDER}/sus_response_*.csv"))

    if not files:
        print(f"[INFO] No response files found in {INPUT_FOLDER}/")
        print("       Place your sus_response_*.csv files there and run again.")
        return

    all_rows   = []
    all_headers = None

    for fpath in files:
        with open(fpath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                continue
            if all_headers is None:
                all_headers = list(rows[0].keys())
            all_rows.extend(rows)

    if not all_rows:
        print("[WARN] All files were empty.")
        return

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*50}")
    print(f"  SUS Response Merger")
    print(f"{'='*50}")
    print(f"  Files merged    : {len(files)}")
    print(f"  Total responses : {len(all_rows)}")
    print(f"  Output file     : {OUTPUT_FILE}")

    # Quick SUS summary
    scores = []
    for row in all_rows:
        s = row.get('sus_score','')
        if s:
            try: scores.append(float(s))
            except: pass

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n  SUS Score Summary:")
        print(f"    Average  : {avg:.1f}")
        print(f"    Min      : {min(scores):.1f}")
        print(f"    Max      : {max(scores):.1f}")
        print(f"    Count    : {len(scores)}")

        if avg >= 85:   grade = "Excellent (A)"
        elif avg >= 72: grade = "Good (B)"
        elif avg >= 52: grade = "Acceptable (C)"
        elif avg >= 38: grade = "Poor (D)"
        else:           grade = "Unacceptable (F)"
        print(f"    Grade    : {grade}")

    print(f"{'='*50}\n")

if __name__ == "__main__":
    merge()
