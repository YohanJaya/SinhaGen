"""
======================================================================
 STEP 13 — Final Sinhala-only character whitelist (safety net)
======================================================================

WHY THIS STEP EXISTS
---------------------
Steps 00-12 handle targeted, content-aware cleaning: fixing broken
newline markers, stripping English/junk tokens, removing dates,
boilerplate, navigation widgets, fused numbers, stray backslashes, etc.
None of those steps guarantee that ONLY Sinhala-valid characters remain
afterward -- they remove specific known patterns, not everything that
could possibly be wrong.

When we inspected the actual output of step 12 on this dataset, the
corpus still contained over 5,000 unique characters -- an impossible
number for real Sinhala (which has roughly 60-90 base characters).
Investigating that revealed:
  - Null bytes and other control characters (encoding corruption)
  - Unassigned Unicode codepoints that merely fall inside the Sinhala
    block's numeric range but were never given meaning (more
    corruption, not real text)
  - Leftover Tamil, Arabic, and Latin script characters (code-mixed
    web text from the source datasets)
  - Symbols, emoji, currency signs, and other web-scraping noise

This step is the final safety net: it keeps ONLY characters that are
legitimately part of clean Sinhala text and removes everything else,
regardless of how it got there.

THREE IMPORTANT LESSONS BAKED INTO THE LOGIC BELOW
-----------------------------------------------------
LESSON 1 — Zero Width Joiner / Non-Joiner (U+200C, U+200D) MUST be kept.
These invisible characters are structurally required to form certain
Sinhala conjunct/ligature clusters (e.g. the "ශ්‍ර" cluster used in
words like "ශ්‍රී ලංකාව"). They fall outside the main Sinhala Unicode
block, so a naive range-check would delete them -- which would silently
fragment a large number of real Sinhala words (confirmed: 14+ million
occurrences in this corpus).

LESSON 2 — Not every codepoint "inside" the Sinhala Unicode block range
is a real character. Unicode has unassigned gaps within that range
(e.g. U+0DB2, U+0DBC, U+0DBE...). Their presence is itself a symptom of
encoding corruption, so we additionally require the codepoint to be an
*assigned* Unicode character (category != "Cn"), not just numerically
in range.

LESSON 3 — Python's str.isspace() is not a safe test for "this is
harmless whitespace." It returns True for some ASCII control characters
(e.g. \\x1f, the "Unit Separator") that are not real whitespace. We
check Unicode category directly (category == "Zs") instead.

WHAT IS KEPT
------------
  1. Sinhala Unicode block (U+0D80-U+0DFF), excluding unassigned codepoints
  2. Zero Width Joiner / Non-Joiner (U+200C, U+200D)
  3. ASCII digits 0-9
  4. A small, unambiguous punctuation set: . , ! ? ; : ' " ( ) - /
  5. Whitespace: real newlines, plus any Unicode space-separator (Zs)
     character -- all such variants are normalized to one regular space.

HOW REMOVAL IS APPLIED
------------------------
  - Invisible/control categories (Cf, Cc, Cn) are deleted outright --
    inserting a visible space where an invisible character used to be
    would incorrectly split a word in two.
  - Visible disallowed characters (foreign letters, symbols, disallowed
    punctuation) are replaced with a single space, so two words that
    were separated by that character don't fuse into one.

Just run:  python3 13_finalize_sinhala_only.py
"""

import json
import os
import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# File paths -- wired to continue the pipeline chain from step 12's output.
# ---------------------------------------------------------------------------
INPUT_FILE = os.path.expanduser("~/pipeline_step12_backslashes_cleaned.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step13_sinhala_only.jsonl")
REPORT_TOP_N = 40  # how many most-frequently-removed characters to print in the summary

# Sinhala Unicode block: covers all Sinhala consonants, independent vowels,
# dependent vowel signs, the virama (al-lakuna), anusvaraya/visargaya, and
# the native Sinhala Lith digit set.
SINHALA_RANGE = (0x0D80, 0x0DFF)

# Zero Width Joiner (U+200D) / Zero Width Non-Joiner (U+200C).
# NOT junk -- required to correctly form certain Sinhala consonant
# conjuncts/ligatures. See LESSON 1 above.
ZERO_WIDTH_JOINERS = {"\u200c", "\u200d"}

# A deliberately small, unambiguous punctuation whitelist.
ALLOWED_PUNCTUATION = set(".,!?;:'\"()-/")


def is_allowed_char(ch: str) -> bool:
    """Decide whether a single character should survive cleaning."""
    code = ord(ch)
    category = unicodedata.category(ch)

    if SINHALA_RANGE[0] <= code <= SINHALA_RANGE[1]:
        # Reject unassigned codepoints inside the numeric range. See LESSON 2.
        return category != "Cn"

    if ch in ZERO_WIDTH_JOINERS:
        return True

    if ch.isdigit() and code < 128:  # ASCII digits only
        return True

    if ch in ALLOWED_PUNCTUATION:
        return True

    # Check category == "Zs" directly rather than ch.isspace(). See LESSON 3.
    if ch == "\n" or category == "Zs":
        return True

    return False


def clean_text(text: str, removed_counter: Counter) -> str:
    cleaned_chars = []

    for ch in text:
        if is_allowed_char(ch):
            category = unicodedata.category(ch)
            # Normalize every Unicode space-separator variant down to one
            # regular ASCII space.
            if ch != "\n" and category == "Zs":
                cleaned_chars.append(" ")
            else:
                cleaned_chars.append(ch)
        else:
            removed_counter[ch] += 1
            category = unicodedata.category(ch)
            if category in ("Cf", "Cc", "Cn"):
                # Invisible/control/unassigned -- delete outright, don't
                # insert a space that would fracture a word.
                cleaned_chars.append("")
            else:
                # Visible disallowed character -- replace with a space so
                # adjacent words don't fuse together.
                cleaned_chars.append(" ")

    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return cleaned


def main():
    removed_counter = Counter()
    total_records = 0
    bad_json = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for raw_line in fin:
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
                record["text"] = clean_text(text, removed_counter)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_removed = sum(removed_counter.values())
    print(f"[Step 13] Processed {total_records} records ({bad_json} malformed JSON lines skipped)")
    print(f"[Step 13] Total disallowed characters removed: {total_removed}")
    print(f"[Step 13] Unique disallowed character types found: {len(removed_counter)}")
    print(f"\n[Step 13] Top {REPORT_TOP_N} most frequent removed characters:")
    print(f"{'char':<8}{'codepoint':<12}{'category':<10}{'count':<10}")
    print("-" * 40)
    for ch, count in removed_counter.most_common(REPORT_TOP_N):
        display = ch if ch.isprintable() and not ch.isspace() else repr(ch)
        print(f"{display:<8}{'U+%04X' % ord(ch):<12}{unicodedata.category(ch):<10}{count:<10}")

    print(f"\n[Step 13] Wrote: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
