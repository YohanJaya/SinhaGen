"""
STEP 5 — Quality filter and dedupe (no split, JSONL only).

- Drops exact duplicate documents (common when combining multiple
  source datasets).
- Drops documents that are too short, or where too much of the content
  isn't actually Sinhala script (leftover junk/garbage records).
- Writes ONE final JSONL file with everything kept -- no train/val
  split, no .txt output.

Just run:  python3 05_quality_filter_and_split.py
"""

import json
import os

INPUT_FILE = os.path.expanduser("~/pipeline_step4_boilerplate_removed.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step5_quality_filtered.jsonl")

MIN_DOC_LENGTH = 50          # characters
MIN_SINHALA_RATIO = 0.6      # fraction of non-space chars that must be Sinhala


def sinhala_ratio(text: str) -> float:
    non_space_chars = [c for c in text if not c.isspace()]
    if not non_space_chars:
        return 0.0
    sinhala_count = sum(1 for c in non_space_chars if 0x0D80 <= ord(c) <= 0x0DFF)
    return sinhala_count / len(non_space_chars)


def main():
    seen = set()
    kept_records = []
    dropped_short = 0
    dropped_low_ratio = 0
    dropped_dupe = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get('text', '')
            if not isinstance(text, str):
                continue

            if text in seen:
                dropped_dupe += 1
                continue

            if len(text) < MIN_DOC_LENGTH:
                dropped_short += 1
                continue

            if sinhala_ratio(text) < MIN_SINHALA_RATIO:
                dropped_low_ratio += 1
                continue

            seen.add(text)
            kept_records.append(record)

    print(f"[Step 5] Kept {len(kept_records)} records.")
    print(f"[Step 5] Dropped: {dropped_dupe} duplicates, "
          f"{dropped_short} too short, {dropped_low_ratio} low Sinhala ratio.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in kept_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"[Step 5] Wrote {len(kept_records)} records -> {OUTPUT_FILE}")


if __name__ == '__main__':
    main()