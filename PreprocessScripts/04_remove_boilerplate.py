"""
STEP 4 — Strip repeated boilerplate lines.

Scraped blog/news datasets like yours typically have repeated
navigation, footer, "related articles", and comment-section text that
appears near-verbatim across many different documents (you can see
examples of this in your own sample data -- lines like site nav links,
"read more" lists, etc.). This is noise for language-model pretraining:
it teaches the model to reproduce boilerplate rather than natural song
lyrics / prose.

This script:
  1. First pass: counts how many *different documents* each line
     appears in.
  2. Second pass: removes lines that appear in an unusually large
     number of documents (likely boilerplate), and drops any resulting
     empty/near-empty records.
  3. Prints the top repeated lines it removed, so you can sanity check
     before trusting the output.

Tune MIN_REPEAT_DOCS and MIN_LINE_LENGTH below if it removes too much
or too little.

Just run:  python3 04_remove_boilerplate.py
"""

import json
import os
from collections import Counter

INPUT_FILE = os.path.expanduser("~/pipeline_step3_punctuation_cleaned.jsonl")
OUTPUT_FILE = os.path.expanduser("~/pipeline_step4_boilerplate_removed.jsonl")

# A line must appear in at least this many DIFFERENT documents to be
# considered boilerplate (not just "a common short phrase").
MIN_REPEAT_DOCS = 15

# Lines shorter than this are never removed as boilerplate -- short
# common phrases/words are normal and shouldn't be stripped.
MIN_LINE_LENGTH = 15

# Minimum length for a cleaned record to be kept at all.
MIN_RECORD_LENGTH = 30


def load_lines_per_doc(path):
    """Yield (record, list_of_lines) for each valid record."""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get('text', '')
            if not isinstance(text, str):
                continue
            yield record, text.split('\n')


def main():
    # Pass 1: count doc-frequency per line.
    doc_freq = Counter()
    total_docs = 0
    for _, lines in load_lines_per_doc(INPUT_FILE):
        total_docs += 1
        for uniq_line in set(l.strip() for l in lines if l.strip()):
            doc_freq[uniq_line] += 1

    boilerplate = {
        line for line, count in doc_freq.items()
        if count >= MIN_REPEAT_DOCS and len(line) >= MIN_LINE_LENGTH
    }

    print(f"[Step 4] Scanned {total_docs} documents.")
    print(f"[Step 4] Identified {len(boilerplate)} boilerplate lines "
          f"(appearing in >= {MIN_REPEAT_DOCS} docs).")
    print("[Step 4] Top repeated lines being removed:")
    top = sorted(boilerplate, key=lambda l: -doc_freq[l])[:15]
    for line in top:
        preview = line if len(line) <= 80 else line[:77] + "..."
        print(f"    ({doc_freq[line]}x) {preview}")
    print()

    # Pass 2: rewrite records with boilerplate lines stripped out.
    kept = 0
    dropped_empty = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        for record, lines in load_lines_per_doc(INPUT_FILE):
            filtered_lines = [l for l in lines if l.strip() not in boilerplate]
            cleaned_text = '\n'.join(filtered_lines).strip()

            if len(cleaned_text) < MIN_RECORD_LENGTH:
                dropped_empty += 1
                continue

            record['text'] = cleaned_text
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            kept += 1

    print(f"[Step 4] Kept {kept} records, dropped {dropped_empty} "
          f"that became too short after boilerplate removal.")
    print(f"[Step 4] Wrote: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()