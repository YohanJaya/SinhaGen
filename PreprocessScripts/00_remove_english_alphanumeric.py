"""
STEP 0 — Remove English words and alphanumeric junk tokens.

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

IMPORTANT: Your dataset uses a literal "\\n" (backslash + n) marker for
line breaks inside the "text" field. A naive word-removal regex would
treat the "n" in that marker as a standalone English word and delete
it, corrupting the marker (this is exactly the bug we hit before).
To prevent that, this script first converts any intact "\\n" marker
into a real newline character -- before doing any word removal -- so
the "n" is never exposed to the word-removal step in the first place.

Just run:  python3 00_remove_english_alphanumeric.py
"""

import json
import os
import re

INPUT_FILE = os.path.expanduser("~/combined_sinhala_dataset.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step0_english_removed.jsonl")

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

    print(f"[Step 0] Processed {total} records, modified {changed}.")
    print(f"[Step 0] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()