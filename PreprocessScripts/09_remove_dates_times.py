"""
STEP 09 — Remove date/time stamps.

Scraped blog comments often carry timestamps in two families of formats:

  1) Sinhala date/time, e.g.:
       ජූනි 3, 2013 6:56 පෙ.ව.
       මැයි 31, 2013 9:58 පෙ.ව.
       2016 ඔක්තෝබර් 14 වැනි සිකුරාදා

  2) Numeric date/time, e.g.:
       29, 2017 2:51
       9:16
       2019-12-13
       (2020)
       03-10-2013 09:15
       3/25/2017 05:49:00
       12-08-2019, 11:24

These are page metadata, not real content, so they're removed entirely.
The time/seconds/AM-PM portions are optional wherever it makes sense, so a
bare date is also caught.

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
SINHALA_WEEKDAYS = [
    'ඉරිදා', 'සඳුදා', 'අඟහරුවාදා', 'බදාදා', 'බ්‍රහස්පතින්දා', 'සිකුරාදා',
    'සෙනසුරාදා',
]

_month_pattern = '|'.join(re.escape(m) for m in SINHALA_MONTHS)
_weekday_pattern = '|'.join(re.escape(w) for w in SINHALA_WEEKDAYS)

# ---------------------------------------------------------------------------
# Sinhala formats
# ---------------------------------------------------------------------------

# Format A: "ජූනි 3, 2013 6:56 පෙ.ව." -- Month Day, Year [Time [AM/PM]]
_FORMAT_1 = (
    r'(?:' + _month_pattern + r')\s+\d{1,2},?\s*\d{4}'
    r'(?:\s+\d{1,2}:\d{2}\s*(?:පෙ\.ව\.|ප\.ව\.)?)?'
)

# Format B: "2016 ඔක්තෝබර් 14 වැනි සිකුරාදා" -- Year Month Day වැනි Weekday
_FORMAT_2 = (
    r'\d{4}\s+(?:' + _month_pattern + r')\s+\d{1,2}\s+වැනි\s+'
    r'(?:' + _weekday_pattern + r')\s*,?'
)

# Format C: bare "15 2019 12:05" -- Day Year Time, no month word (usually a
# continuation/fragment of another date mention nearby).
_FORMAT_3 = r'\b\d{1,2}\s+\d{4}\s+\d{1,2}:\d{2}\b'

_FORMAT_4 = r'\b\d{1,2}\s+\d{4}\b'  # bare "15 2019" -- Day Year, no month or time

# ---------------------------------------------------------------------------
# Numeric formats
#   29, 2017 2:51 | 9:16 | 2019-12-13 | (2020) | 03-10-2013 09:15
#   3/25/2017 05:49:00 | 12-08-2019, 11:24
# ---------------------------------------------------------------------------

# Format E: "29, 2017 2:51" -- Day, Year Time (month word dropped/fragment)
_FORMAT_5 = r'\b\d{1,2},\s*\d{4}\s+\d{1,2}:\d{2}\b'

# Format F: ISO date, optionally with time -- "2019-12-13" / "2019-12-13, 09:15"
_FORMAT_6 = r'\b\d{4}-\d{2}-\d{2}(?:,?\s*\d{1,2}:\d{2}(?::\d{2})?)?\b'


# Format H: DD-MM-YYYY or MM/DD/YYYY, optionally with time (and optional
# seconds) -- "03-10-2013 09:15", "3/25/2017 05:49:00", "12-08-2019, 11:24"
_FORMAT_7 = (
    r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}'
    r'(?:,?\s*\d{1,2}:\d{2}(?::\d{2})?)?\b'
)

# Format I: bare time -- "9:16", optionally with seconds -- "05:49:00"
_FORMAT_8 = r'\b\d{1,2}:\d{2}(?::\d{2})?\b'

# Order matters: longer/more specific patterns must be tried before the
# shorter bare-time / bare-year patterns so they aren't partially consumed.
DATE_TIME_RE = re.compile(
    '|'.join([
        _FORMAT_2,
        _FORMAT_7,
        _FORMAT_6,
        _FORMAT_5,
        _FORMAT_1,
        _FORMAT_3,
        _FORMAT_7,
        _FORMAT_8,
    ])
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