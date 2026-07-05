"""
STEP 0 — Unicode normalization (NFC). RUNS FIRST.

Sinhala conjunct/vowel-sign sequences can sometimes be represented by
different (but visually identical) sequences of Unicode code points
depending on the source website/scraper. If left inconsistent, your
tokenizer will learn the "same" character as multiple different token
sequences, hurting training quality.

NFC (Canonical Composition) normalization makes sure every string uses
one consistent representation. This runs FIRST, before any other
cleaning step, so every later regex/character-class check in this
pipeline operates on one consistent character representation from the
very start. It's safe to run before newline-marker repair too --
normalization only touches Unicode combining sequences, never the
plain ASCII backslash/n characters used in the "\\n" line-break marker.

Just run:  python3 00_normalize_unicode.py
"""

import json
import os
import unicodedata

INPUT_FILE = os.path.expanduser("~/combined_sinhala_dataset.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step0_unicode_normalized.jsonl")


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
                normalized = unicodedata.normalize('NFC', original)
                if normalized != original:
                    changed += 1
                record['text'] = normalized

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            total += 1

    print(f"[Step 0] Processed {total} records, normalized {changed}.")
    print(f"[Step 0] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()