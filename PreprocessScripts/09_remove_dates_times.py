"""
STEP 09 — Remove date/time stamps.

Scraped blog comments often carry timestamps like:
    ජූනි 3, 2013 6:56 පෙ.ව.
    මැයි 31, 2013 9:58 පෙ.ව.

These are page metadata, not real content, so they're removed
entirely. Matches: <Sinhala month> <day>, <year> [<hour>:<minute>
[am/pm marker]] -- the time and am/pm parts are optional, so a bare
date like "මැයි 31, 2013" is also caught.

Just run:  python3 09_remove_dates_times.py
"""

import json
import os
import re

INPUT_FILE = os.path.expanduser("~/pipeline_step8_punctuation_cleaned2.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step9_dates_removed.jsonl")

SINHALA_MONTHS = [
    'ජනවාරි', 'පෙබරවාරි', 'මාර්තු', 'අප්‍රේල්', 'මැයි', 'ජූනි', 'ජූලි',
    'අගෝස්තු', 'සැප්තැම්බර්', 'ඔක්තෝබර්', 'නොවැම්බර්', 'දෙසැම්බර්',
]

_month_pattern = '|'.join(re.escape(m) for m in SINHALA_MONTHS)

DATE_TIME_RE = re.compile(
    r'(?:' + _month_pattern + r')\s+\d{1,2},?\s*\d{4}'
    r'(?:\s+\d{1,2}:\d{2}\s*(?:පෙ\.ව\.|ප\.ව\.)?)?'
)


def remove_dates(text: str) -> str:
    text = DATE_TIME_RE.sub('', text)
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
                cleaned = remove_dates(original)
                if cleaned != original:
                    changed += 1
                record['text'] = cleaned

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            total += 1

    print(f"[Step 09] Processed {total} records, removed dates/times in {changed}.")
    print(f"[Step 09] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()