"""
POST-STEP — Collapse multiple spaces and strip trailing whitespace.

Runs on your final_dataset_no_emoji.jsonl file (the output of the
emoji-removal step). For each line of text:
  - Collapses 2+ consecutive spaces/tabs into a single space.
  - Also normalizes non-breaking spaces (U+00A0) to regular spaces
    before collapsing, since these often sneak in from scraped HTML.
  - Strips leading/trailing whitespace on each line.
  - Strips leading/trailing whitespace on the document as a whole.

Real newline characters (\\n) are never touched -- only spaces/tabs
within each line are affected.

Writes a NEW file (doesn't overwrite your existing input).

Just run:  python3 clean_whitespace_from_final.py
"""

import json
import os
import re

INPUT_FILE = os.path.expanduser("~/pipeline_step6_emojis_removed.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step7_clean_whitespace.jsonl")

MULTI_SPACE_RE = re.compile(r'[ \t]{2,}')


def clean_whitespace(text: str) -> str:
    # Normalize non-breaking spaces to regular spaces first.
    text = text.replace('\xa0', ' ')

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = MULTI_SPACE_RE.sub(' ', line)
        line = line.strip()  # remove leading + trailing whitespace on this line
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def main():
    total = 0
    changed = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout_json:

        for line in fin:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(record.get('text'), str):
                original = record['text']
                cleaned = clean_whitespace(original)
                if cleaned != original:
                    changed += 1
                record['text'] = cleaned

            fout_json.write(json.dumps(record, ensure_ascii=False) + '\n')
            total += 1

    print(f"Processed {total} records, cleaned whitespace in {changed}.")
    print(f"  -> {OUTPUT_FILE}")


if __name__ == '__main__':
    main()