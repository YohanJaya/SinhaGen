"""
STEP 11 — Remove prev/next post navigation snippets.

Many scraped blog templates show a "previous post / next post" widget
using guillemet marks, e.g.:

    « Previous Post Title Next Post Title »
    Author Name: « Previous Post Title Next Post Title »

These are page navigation, not real article content, and the titles
inside are unrelated to whatever document they got glued onto -- so
the whole "« ... »" span (plus an optional leading colon) is removed.

ASSUMPTION: this treats every "« ... »" span in the corpus as
navigation junk. If your data legitimately uses guillemets as quote
marks elsewhere, this would remove that too -- based on the examples
seen so far, this pattern only showed up as nav widgets, but flagging
this in case you spot a false positive.

After removal, this also blanks out any line left with no letters or
digits (e.g. a dangling ":"), the same safety check used in the
punctuation-cleaning step, so this works correctly regardless of
where you slot it into your pipeline.

Just run:  python3 11_remove_nav_titles.py
"""

import json
import os
import re

INPUT_FILE = os.path.expanduser("~/pipeline_step10_fused_numbers_removed.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step11_nav_titles_removed.jsonl")

# Optional leading colon + whitespace, then the whole «...» span
# (character class [^»] matches newlines too, so multi-line spans are
# also caught).
NAV_RE = re.compile(r':?\s*«[^»]*»')


def remove_junk_only_lines(text: str) -> str:
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
    text = NAV_RE.sub('', text)
    text = remove_junk_only_lines(text)
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

    print(f"[Step 11] Processed {total} records, removed nav titles in {changed}.")
    print(f"[Step 11] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()