"""
STEP 12 — Clean stray backslash-escaping.

Some scraped text ended up double-escaped, so quote marks that should be
plain `"` characters are showing up as `\"`, e.g.:

    \"ලොකු දුවේ මේ වයසේ ළමමයින්ට අසනීපයක් හැදෙනවා.ඔයාටත් දැන් ඒ අසසනීපෙ
    හැදිලා. ඒ නිසා බැදුම් තෙල් කන්න එපා \"

We want to keep the quote characters (they mark the start/end of speech)
but drop the backslash in front of them. The literal `\n` sequence
(backslash + n) must be left exactly as-is -- it is NOT converted to a
real newline and NOT touched at all.

Also, stray ellipses ("..." or the single-character "…") are removed the
same way -- the literal \n sequence is the ONLY thing that must survive
untouched.

Rule: remove every backslash in the text EXCEPT a backslash that is
immediately followed by the letter 'n' (i.e. leave \n alone). Then remove
every run of ellipsis dots/characters.

So:
    \"text\"   ->  "text"
    \n         ->  \n   (unchanged)
    \'         ->  '
    \\         ->  (each backslash not followed by n is stripped)
    ...        ->  (removed)
    …          ->  (removed)

Just run:  python3 10_clean_backslashes.py
"""
import json
import os
import re

INPUT_FILE = os.path.expanduser("~/pipeline_step11_nav_titles_removed.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step12_backslashes_cleaned.jsonl")

# Remove any backslash NOT immediately followed by 'n', so literal \n stays
# untouched while \" , \' , \\, etc. lose their backslash.
BACKSLASH_RE = re.compile(r'\\(?!n)')

# Remove ellipses: two-or-more literal dots ("...", "..") or the single
# unicode ellipsis character ("…"). \n is never matched by this pattern.
ELLIPSIS_RE = re.compile(r'\.{2,}|…')


def clean_backslashes(text: str) -> str:
    text = BACKSLASH_RE.sub('', text)
    text = ELLIPSIS_RE.sub('', text)
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
                cleaned = clean_backslashes(original)
                if cleaned != original:
                    changed += 1
                record['text'] = cleaned

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            total += 1

    print(f"[Step 12] Processed {total} records, cleaned backslashes in {changed}.")
    print(f"[Step 12] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()