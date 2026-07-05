"""
STEP 2 — Remove English words and alphanumeric junk tokens.

Removes:
  - Pure English words (e.g. "Hello", "read", "AI")
  - Mixed letter+digit junk tokens (e.g. "234FGH", "2H4", "ERT234",
    "iPhone13") -- these are almost always garbled scraper artifacts,
    not real content.

Keeps:
  - Pure Sinhala text (untouched -- outside the ASCII range entirely).
  - Pure numbers (e.g. "1990", "234") -- these are often meaningful
    (years, quantities) so they are NOT removed by default. Change
    KEEP_PURE_NUMBERS below if you want those gone too.

WHY THIS RUNS AFTER NEWLINE-FIXING (steps 0-1), NOT BEFORE:
This step deletes any standalone run of ASCII letters, including a
single orphaned letter like "n". If this ran before the newline-marker
repair step, it would silently delete any already-broken "\\n" marker
(one that lost its backslash from earlier corruption) as if it were
ordinary English text -- permanently destroying that paragraph break.
Running this AFTER steps 0-1 guarantees every line break is already a
real newline character by the time this step's word-removal regex
runs, so it can never be mistaken for text content.

The defensive `text.replace('\\n', '\n')` below is kept as a no-op
safeguard in case this script is ever run standalone out of order --
it has nothing to do if step 1 already ran first, as intended.

Just run:  python3 02_remove_english_alphanumeric.py
"""

import json
import os
import re

INPUT_FILE = os.path.expanduser("~/pipeline_step1_newlines_fixed.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step2_english_removed.jsonl")

KEEP_PURE_NUMBERS = True  # set False to also strip standalone digit runs

# A "token" is a contiguous run of ASCII letters/digits, optionally with
# an apostrophe-joined suffix (e.g. "don't").
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z]+)?")


def remove_english_and_junk(text: str) -> str:
    # 1. Protect the newline marker FIRST, before any word removal.
    text = text.replace('\\n', '\n')

    # 2. Remove English words / alphanumeric junk tokens.
    def _replace(m):
        tok = m.group(0)
        if tok.isdigit():
            return tok if KEEP_PURE_NUMBERS else ''
        return ''  # pure English word or mixed alphanumeric junk

    text = TOKEN_RE.sub(_replace, text)

    # 3. Tidy up leftover whitespace left behind by removed tokens
    #    (does not touch newline characters).
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = '\n'.join(line.rstrip(' \t') for line in text.split('\n'))

    return text


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
                cleaned = remove_english_and_junk(original)
                if cleaned != original:
                    changed += 1
                record['text'] = cleaned

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            total += 1

    print(f"[Step 2] Processed {total} records, modified {changed}.")
    print(f"[Step 2] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()