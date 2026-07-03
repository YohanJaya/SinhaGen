"""
STEP 8 — Clean punctuation.

- Removes empty bracket leftovers like "()" / "[ ]" (artifacts from
  stripped links/citations).
- Collapses runs of repeated/mixed punctuation ("...", "?!", "--", '""')
  down to just the first mark, since the first mark is usually the
  meaningful one.
- Carefully protects: real newlines, Sinhala letters, Sinhala
  combining vowel signs/virama (these are Unicode "marks", not
  "letters", so a naive regex would delete them -- this one doesn't),
  and the zero-width joiner used inside Sinhala conjunct letters.

Just run:  python3 08_clean_punctuation2.py
"""

import json
import os
import re
import unicodedata as ud

INPUT_FILE = os.path.expanduser("~/pipeline_step7_unicode_normalized.jsonl")
OUTPUT_FILE = os.path.expanduser("~clear")

ZWJ = '\u200d'
ZWNJ = '\u200c'


def _build_combining_mark_class():
    marks = []
    for cp in range(0x0D80, 0x0E00):  # Sinhala Unicode block
        ch = chr(cp)
        if ud.category(ch) in ('Mn', 'Mc'):
            marks.append(ch)
    return ''.join(marks)


SINHALA_MARKS = _build_combining_mark_class()
PROTECTED = r'\w\s' + re.escape(ZWJ + ZWNJ + SINHALA_MARKS)

REPEATED_PUNCT_RE = re.compile(
    r'([^' + PROTECTED + r'])[^' + PROTECTED + r']+', re.UNICODE
)
EMPTY_BRACKETS_RE = re.compile(r'[\(\[\{]\s*[\)\]\}]')


def remove_empty_brackets(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = EMPTY_BRACKETS_RE.sub('', text)
    return text


def remove_punctuation_only_lines(text: str) -> str:
    """If an entire line contains no letters/digits at all (just
    punctuation/symbols and whitespace, e.g. ": : -" or ".. //"),
    wipe that line's content. The newline itself is preserved --
    only the junk content on that line is removed."""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not any(ch.isalnum() for ch in stripped):
            cleaned_lines.append('')
        else:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def clean_text(text: str) -> str:
    text = remove_empty_brackets(text)
    text = REPEATED_PUNCT_RE.sub(r'\1', text)
    text = remove_punctuation_only_lines(text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
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
                cleaned = clean_text(original)
                if cleaned != original:
                    changed += 1
                record['text'] = cleaned

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            total += 1

    print(f"[Step 8] Processed {total} records, cleaned punctuation in {changed}.")
    print(f"[Step 8] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()