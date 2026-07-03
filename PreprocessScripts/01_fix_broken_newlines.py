"""
STEP 1 — Fix broken newline markers.

Your raw dataset uses a literal two-character marker "\\n" (backslash + n)
inside the "text" field to represent line breaks. An earlier cleaning
pass stripped stray backslashes as junk symbols, which turned some of
these markers into an orphan "n" (and "\\n\\n" paragraph breaks into
"nn"). Since English words/letters were already removed in an earlier
step, any standalone "n" left in otherwise-Sinhala text is almost
certainly one of these broken markers, not real content.

This script:
  1. Converts any INTACT "\\n" / "\\n\\n" markers into real newline
     characters.
  2. Finds ORPHANED standalone "n" / "nn" (not touching other Latin
     letters) and converts those into real newline(s) too.

Just run:  python3 01_fix_broken_newlines.py
(edit INPUT_FILE / OUTPUT_FILE below if your paths differ)
"""

import json
import os
import re

INPUT_FILE = os.path.expanduser("~/pipeline_step0b_emojis_removed.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step1_newlines_fixed.jsonl")

# Orphaned "n" or "nn" not adjacent to any other Latin letter.
ORPHAN_N_RE = re.compile(r'(?<![A-Za-z])(n{1,2})(?![A-Za-z])')


def fix_newlines(text: str) -> str:
    # 1. Restore any still-intact literal "\n" markers to real newlines.
    text = text.replace('\\n', '\n')

    # 2. Restore orphaned n/nn (missing their backslash) to newline(s).
    def _replace(m):
        return '\n' * len(m.group(1))

    text = ORPHAN_N_RE.sub(_replace, text)
    return text


def main():
    total = 0
    fixed_count = 0

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
                cleaned = fix_newlines(original)
                if cleaned != original:
                    fixed_count += 1
                record['text'] = cleaned

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            total += 1

    print(f"[Step 1] Processed {total} records, fixed newlines in {fixed_count}.")
    print(f"[Step 1] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()