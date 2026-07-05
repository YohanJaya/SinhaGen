"""
======================================================================
 STEP 14 — Alphabet verification (QA check, not a cleaning step)
======================================================================

PURPOSE
-------
Reads the final cleaned corpus and reports every unique character that
appears in it -- codepoint, Unicode category, and official name -- so
you can visually confirm the dataset is genuinely clean BEFORE handing
it to tokenizer training.

WHAT "GOOD" LOOKS LIKE
------------------------
For genuinely clean Sinhala text, expect roughly 100-140 unique
characters total: Sinhala letters, vowel signs, virama, ZWJ/ZWNJ,
digits, and the small punctuation/whitespace set from step 13.

Checklist when reading the printed table:
  - No "Cn" (unassigned) category entries -- indicates corruption
  - No stray "Cc" (control character) entries other than '\\n' itself
  - Only ONE space-like entry (category "Zs")
  - '\\u200c' (ZWNJ) and '\\u200d' (ZWJ) SHOULD be present -- required
    for Sinhala conjuncts, not junk
  - No unexpected non-Sinhala letters (Tamil, Arabic, Latin, etc.)

If the count comes back in the thousands, something upstream (steps
00-13) isn't running as expected on your actual data -- re-check the
pipeline chain before trusting the corpus for tokenizer training.

Just run:  python3 14_inspect_alphabet.py
"""

import json
import os
import unicodedata
from collections import Counter

INPUT_FILE = os.path.expanduser("~/pipeline_step13_sinhala_only.jsonl")


def main():
    char_counts = Counter()
    total_records = 0
    bad_json = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            total_records += 1
            text = record.get("text", "")
            if isinstance(text, str):
                char_counts.update(text)

    print(f"[Step 14] Records scanned: {total_records} ({bad_json} malformed JSON lines skipped)")
    print(f"[Step 14] Total unique characters: {len(char_counts)}\n")

    print(f"{'char':<8}{'codepoint':<10}{'category':<8}{'count':<12}{'name'}")
    print("-" * 80)
    for ch in sorted(char_counts, key=ord):
        count = char_counts[ch]
        display = ch if ch.isprintable() and not ch.isspace() else repr(ch)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "UNNAMED"
        category = unicodedata.category(ch)
        print(f"{display:<8}U+{ord(ch):04X}  {category:<8}{count:<12}{name}")


if __name__ == "__main__":
    main()
