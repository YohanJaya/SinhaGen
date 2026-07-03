"""
STEP 10 — Remove numbers fused directly onto Sinhala words.

Targets garbage like "කරන0456" (word + digits with no space) and
strips just the digit part, keeping the real word: "කරන0456" -> "කරන".

Only removes a digit run when it TOUCHES a Sinhala letter/mark with
zero space in between (on either side). A normal number that's simply
part of a sentence, like "1990 වසරේ" (digits separated by a space from
the word), is left completely untouched.

Just run:  python3 10_remove_fused_numbers.py
"""

import json
import os
import re

INPUT_FILE = os.path.expanduser("~/pipeline_step9_dates_removed.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step10_fused_numbers_removed.jsonl")

FUSED_DIGITS_RE = re.compile(
    r'(?<=[\u0D80-\u0DFF])\d+|\d+(?=[\u0D80-\u0DFF])'
)


def remove_fused_numbers(text: str) -> str:
    return FUSED_DIGITS_RE.sub('', text)


def main():
    total = 0
    changed = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:

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
                cleaned = remove_fused_numbers(original)
                if cleaned != original:
                    changed += 1
                record['text'] = cleaned

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            total += 1

    print(f"[Step 10] Processed {total} records, removed fused numbers in {changed}.")
    print(f"[Step 10] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()