"""
Memory-efficient JSONL dataset combiner with lazy loading.

- Reads one line at a time (never loads full file into RAM)
- Dedup uses a hash of each text (just 16 bytes per record) instead of full text
- Streams output directly to disk

Usage:
    python combine_datasets.py ../CulturaX ../MADLAD_400 --output ../combined.jsonl
    python combine_datasets.py ../CulturaX ../MADLAD_400 --output ../combined.jsonl --dedup
"""

import sys
import json
import glob
import os
import argparse
import hashlib

# Field names to look for in each record (checked in order)
TEXT_FIELD_CANDIDATES = ["text", "content", "raw_content", "sentence"]


def extract_text(record: dict):
    for field in TEXT_FIELD_CANDIDATES:
        if field in record and record[field]:
            return record[field]
    return None


def hash_text(text: str) -> bytes:
    """Return a compact 16-byte MD5 hash of text — much cheaper than storing full strings."""
    return hashlib.md5(text.strip().encode("utf-8")).digest()


def find_jsonl_files(folders):
    files = []
    for folder in folders:
        found = glob.glob(os.path.join(folder, "**", "*.jsonl"), recursive=True)
        files.extend(found)
    return sorted(set(files))


def combine(folders, output_path, dedup=False, log_every=10000):
    jsonl_files = find_jsonl_files(folders)

    if not jsonl_files:
        print("No .jsonl files found in the given folder(s).")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL file(s):")
    for f in jsonl_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {f}  ({size_mb:.1f} MB)")

    # For dedup: store only 16-byte hashes (not full text strings)
    # 1M records = ~16 MB RAM — very manageable
    seen_hashes = set() if dedup else None

    total_written = 0
    total_skipped_duplicate = 0
    total_skipped_no_text = 0

    with open(output_path, "w", encoding="utf-8") as out_f:

        for file_path in jsonl_files:
            source_name = os.path.basename(os.path.dirname(file_path)) or os.path.basename(file_path)
            file_written = 0
            file_skipped = 0

            print(f"\nProcessing: {file_path}")

            with open(file_path, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(in_f):  # lazy: reads one line at a time
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)   # parse just this one line
                    except json.JSONDecodeError:
                        continue

                    text = extract_text(record)
                    if not text:
                        total_skipped_no_text += 1
                        continue

                    if dedup:
                        h = hash_text(text)
                        if h in seen_hashes:
                            total_skipped_duplicate += 1
                            file_skipped += 1
                            continue
                        seen_hashes.add(h)

                    # Write one line at a time — never accumulate in memory
                    out_record = {"text": text, "source": source_name}
                    out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

                    file_written += 1
                    total_written += 1

                    # Progress log every N records
                    if total_written % log_every == 0:
                        print(f"  [{total_written} records written so far...]")

            print(f"  Done: wrote {file_written} | skipped duplicates {file_skipped}")

    print(f"\n{'='*50}")
    print(f"Combined dataset saved to: {output_path}")
    print(f"Total records written    : {total_written}")
    print(f"Skipped (no text field)  : {total_skipped_no_text}")
    if dedup:
        print(f"Skipped (duplicates)     : {total_skipped_duplicate}")
        print(f"Unique hash set size     : ~{len(seen_hashes) * 16 / (1024*1024):.1f} MB RAM used for dedup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-efficient JSONL dataset combiner.")
    parser.add_argument("folders", nargs="+", help="Folders to search for .jsonl files")
    parser.add_argument("--output", default="../combined.jsonl", help="Output file path")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate using hash (low RAM)")
    args = parser.parse_args()

    combine(args.folders, args.output, dedup=args.dedup)