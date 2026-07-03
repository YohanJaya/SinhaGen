"""
STEP 5 — Quality filter, dedupe, and train/val split.

- Drops exact duplicate documents (common when combining multiple
  source datasets).
- Drops documents that are too short, or where too much of the content
  isn't actually Sinhala script (leftover junk/garbage records).
- Shuffles and splits into train/validation sets.
- Writes both JSONL (with metadata preserved) and plain .txt versions
  (one document per block, blank-line separated) -- the .txt files are
  what you'll typically feed into tokenizer/decoder pretraining.

Just run:  python3 05_quality_filter_and_split.py
"""

import json
import os
import random
import unicodedata as ud

INPUT_FILE = os.path.expanduser("~/pipeline_step4_boilerplate_removed.jsonl")

TRAIN_JSONL = os.path.expanduser("~/final_train.jsonl")
VAL_JSONL = os.path.expanduser("~/final_val.jsonl")
TRAIN_TXT = os.path.expanduser("~/final_train.txt")
VAL_TXT = os.path.expanduser("~/final_val.txt")

MIN_DOC_LENGTH = 50          # characters
MIN_SINHALA_RATIO = 0.6      # fraction of non-space chars that must be Sinhala
VAL_FRACTION = 0.05          # 5% held out for validation
RANDOM_SEED = 42


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

    random.seed(RANDOM_SEED)
    random.shuffle(kept_records)

    val_count = max(1, int(len(kept_records) * VAL_FRACTION))
    val_records = kept_records[:val_count]
    train_records = kept_records[val_count:]

    with open(TRAIN_JSONL, 'w', encoding='utf-8') as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(VAL_JSONL, 'w', encoding='utf-8') as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(TRAIN_TXT, 'w', encoding='utf-8') as f:
        for r in train_records:
            f.write(r['text'].strip() + '\n\n')

    with open(VAL_TXT, 'w', encoding='utf-8') as f:
        for r in val_records:
            f.write(r['text'].strip() + '\n\n')

    print(f"[Step 5] Train: {len(train_records)} docs -> {TRAIN_JSONL}, {TRAIN_TXT}")
    print(f"[Step 5] Val:   {len(val_records)} docs -> {VAL_JSONL}, {VAL_TXT}")


if __name__ == '__main__':
    main()