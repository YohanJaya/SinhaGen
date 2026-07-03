"""
POST-STEP — Remove emojis from the final quality-filtered dataset.

Use this after running the full pipeline (00 -> 05) to strip emojis
from the final combined dataset, without re-running everything else.

Protects Sinhala's zero-width joiner (ZWJ) the same way as before: a
ZWJ is only removed when it sits directly between two emoji
characters, never when it's part of a Sinhala conjunct letter.

Writes a NEW file (doesn't overwrite your existing input), plus a
matching .txt version.

Just run:  python3 remove_emojis_from_final.py
"""

import json
import os
import re

INPUT_FILE = os.path.expanduser("~/pipeline_step5_quality_filtered.jsonl")
OUTPUT_JSONL = os.path.expanduser("~/pipeline_step6_emojis_removed.jsonl")


ZWJ = '\u200d'

EMOJI_RANGES = [
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FA6F),
    (0x1FA70, 0x1FAFF),
    (0x1F1E6, 0x1F1FF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
    (0xFE00, 0xFE0F),
    (0x20E3, 0x20E3),
]

_class_chars = ''.join(f'{chr(s)}-{chr(e)}' for s, e in EMOJI_RANGES)

EMOJI_SEQUENCE_RE = re.compile(
    r'(?:[' + _class_chars + r'](?:' + ZWJ + r'[' + _class_chars + r'])*)+'
)


def remove_emojis(text: str) -> str:
    text = EMOJI_SEQUENCE_RE.sub('', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = '\n'.join(line.rstrip(' \t') for line in text.split('\n'))
    return text


def main():
    total = 0
    changed = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout_json, \
         open(OUTPUT_TXT, 'w', encoding='utf-8') as fout_txt:

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
                cleaned = remove_emojis(original)
                if cleaned != original:
                    changed += 1
                record['text'] = cleaned

            fout_json.write(json.dumps(record, ensure_ascii=False) + '\n')
            fout_txt.write(record['text'].strip() + '\n\n')
            total += 1

    print(f"Processed {total} records, removed emojis from {changed}.")
    print(f"  -> {OUTPUT_JSONL}")
    print(f"  -> {OUTPUT_TXT}")


if __name__ == '__main__':
    main()