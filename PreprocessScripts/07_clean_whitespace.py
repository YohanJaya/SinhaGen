"""
POST-STEP — Collapse multiple spaces and strip trailing whitespace.

Runs on your latest final_train_no_emoji / final_val_no_emoji files
(the output of the emoji-removal step). For each line of text:
  - Collapses 2+ consecutive spaces/tabs into a single space.
  - Also normalizes non-breaking spaces (U+00A0) to regular spaces
    before collapsing, since these often sneak in from scraped HTML.
  - Strips leading/trailing whitespace on each line.
  - Strips leading/trailing whitespace on the document as a whole.

Real newline characters (\\n) are never touched -- only spaces/tabs
within each line are affected.

Writes NEW files (doesn't overwrite your existing ones), and
regenerates matching .txt versions so both formats stay in sync.

Just run:  python3 clean_whitespace_from_final.py
"""

import json
import os
import re

FILES = [
    # (input_jsonl, output_jsonl, output_txt)
    (os.path.expanduser("~/final_train_no_emoji.jsonl"),
     os.path.expanduser("~/final_train_clean.jsonl"),
     os.path.expanduser("~/final_train_clean.txt")),
    (os.path.expanduser("~/final_val_no_emoji.jsonl"),
     os.path.expanduser("~/final_val_clean.jsonl"),
     os.path.expanduser("~/final_val_clean.txt")),
]

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


def process(input_jsonl, output_jsonl, output_txt):
    total = 0
    changed = 0

    with open(input_jsonl, 'r', encoding='utf-8') as fin, \
         open(output_jsonl, 'w', encoding='utf-8') as fout_json, \
         open(output_txt, 'w', encoding='utf-8') as fout_txt:

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
            fout_txt.write(record['text'].strip() + '\n\n')
            total += 1

    print(f"Processed {total} records from {input_jsonl}, "
          f"cleaned whitespace in {changed}.")
    print(f"  -> {output_jsonl}")
    print(f"  -> {output_txt}")


def main():
    for input_jsonl, output_jsonl, output_txt in FILES:
        if not os.path.exists(input_jsonl):
            print(f"Skipping {input_jsonl} (not found).")
            continue
        process(input_jsonl, output_jsonl, output_txt)


if __name__ == '__main__':
    main()