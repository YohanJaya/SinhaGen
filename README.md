# Sinhala Corpus Preprocessing Pipeline — Reference

Run scripts **in numeric order, 00 → 14**. Each step reads the previous
step's output and writes a new file — nothing is overwritten in place,
so you can always inspect any intermediate stage if something looks off.

Run the whole thing in one command with `run_pipeline.sh` (see bottom).

# Sinhala Corpus Preprocessing Pipeline — Reference

Run scripts **in numeric order, 00 → 14**. Each step reads the previous
step's output and writes a new file — nothing is overwritten in place,
so you can always inspect any intermediate stage if something looks off.

Run the whole thing in one command with `run_pipeline.sh` (see bottom).

## A note on step order (00-02 were reordered from the original scripts)

The original script order ran English-word removal *before* newline-marker
repair. That's backwards: English-word removal deletes any standalone run
of ASCII letters, including a single orphaned `n` — and some documents
have a broken `\n` marker (missing its backslash from earlier corruption)
where that orphan `n` is the *only* trace of where a line break used to
be. Running word-removal first would silently destroy that structural
information before the repair step ever got a chance to fix it.

The order below runs Unicode normalization first (order-independent,
foundational), then newline-marker repair, then English-word removal —
so every line break is guaranteed to already be a real `\n` character
before any word-removal regex can misinterpret it as text content.
Verified with a test document containing a pre-broken marker: the repair
now correctly happens before removal, and the paragraph break survives.

## Pipeline steps

| # | Script | Reads | Writes | Purpose |
|---|--------|-------|--------|---------|
| 00 | `00_normalize_unicode.py` | `combined_sinhala_dataset.jsonl` | `pipeline_step0_unicode_normalized.jsonl` | NFC Unicode normalization, so visually identical Sinhala sequences aren't treated as different characters. Runs first since it's foundational and doesn't interact with the newline-marker system at all. |
| 01 | `01_fix_broken_newlines.py` | step 0 output | `pipeline_step1_newlines_fixed.jsonl` | Converts literal `\n` markers into real newline characters; repairs orphaned `n`/`nn` left over from earlier corruption. **Runs before word-removal** so broken markers are fixed before they can be mistaken for English text. |
| 02 | `02_remove_english_alphanumeric.py` | step 1 output | `pipeline_step2_english_removed.jsonl` | Strips English words and garbled alphanumeric tokens (e.g. `iPhone13`). Safe to run now since every line break is already a real newline character, not exposed text. |
| 03 | `03_clean_punctuation.py` | step 2 output | `pipeline_step3_punctuation_cleaned.jsonl` | Removes empty bracket leftovers (`()`, `[ ]`) and collapses repeated punctuation (`...`, `?!`). Protects Sinhala combining marks and ZWJ. |
| 04 | `04_remove_boilerplate.py` | step 3 output | `pipeline_step4_boilerplate_removed.jsonl` | Detects and strips lines repeated across many different documents (nav/footer junk), drops records that become too short afterward. |
| 05 | `05_quality_filter_and_split.py` | step 4 output | `pipeline_step5_quality_filtered.jsonl` | Drops exact-duplicate documents, too-short documents, and documents with too low a Sinhala-character ratio. |
| 06 | `06_remove_emojis.py` | step 5 output | `pipeline_step6_emojis_removed.jsonl` | Removes emoji sequences, correctly distinguishing emoji-joining ZWJ from Sinhala-conjunct ZWJ. |
| 07 | `07_clean_whitespace.py` | step 6 output | `pipeline_step7_clean_whitespace.jsonl` | Collapses repeated spaces/tabs, normalizes non-breaking spaces, trims line/document whitespace. |
| 08 | `08_clean_punctuation2.py` | step 7 output | `pipeline_step8_punctuation_cleaned2.jsonl` | Second punctuation cleanup pass (catches new empty-bracket/repeated-punctuation cases created by earlier steps) and blanks any line left with no letters/digits at all. |
| 09 | `09_remove_dates_times.py` | step 8 output | `pipeline_step9_dates_removed.jsonl` | Strips Sinhala and numeric date/time stamps (comment metadata, not real content). |
| 10 | `10_remove_fused_numbers.py` | step 9 output | `pipeline_step10_fused_numbers_removed.jsonl` | Removes digit runs fused directly onto Sinhala words with no space (e.g. `කරන0456` → `කරන`). Leaves genuine standalone numbers untouched. |
| 11 | `11_remove_nav_titles.py` | step 10 output | `pipeline_step11_nav_titles_removed.jsonl` | Removes `« prev/next post »` navigation widget text. |
| 12 | `12_remove_back_slashes.py` | step 11 output | `pipeline_step12_backslashes_cleaned.jsonl` | Removes stray escaping backslashes (`\"` → `"`) and stray ellipses, while never touching the literal `\n` marker. |
| 13 | `13_finalize_sinhala_only.py` | step 12 output | `pipeline_step13_sinhala_only.jsonl` | **Final safety net.** Whitelists only genuinely valid Sinhala-text characters (Sinhala block minus unassigned codepoints, ZWJ/ZWNJ, ASCII digits, a small punctuation set, normalized whitespace) — removes anything steps 00–12 didn't already catch (stray control characters, other scripts, symbols). |
| 14 | `14_inspect_alphabet.py` | step 13 output | *(prints report only, no file)* | QA check — lists every unique character left in the final corpus with its Unicode name/category/count, so you can visually confirm the dataset is clean before tokenizer training. |

## Key correctness rules baked into steps 13–14 (learned through earlier debugging)

1. **ZWJ / ZWNJ (`U+200C`, `U+200D`) must never be deleted.** They're structurally required for Sinhala conjunct clusters (e.g. `ශ්‍රී`). An early mistake deleted them because they fall outside the main Sinhala Unicode block — this silently fragmented millions of words. Fixed by explicitly whitelisting them.
2. **Not every codepoint "inside" the Sinhala Unicode range is real.** Unicode has unassigned gaps in that range; their presence indicates upstream corruption. Step 13 checks `unicodedata.category(ch) != "Cn"`, not just the numeric range.
3. **`str.isspace()` is not a safe whitespace test.** It incorrectly returns `True` for some control characters (e.g. `\x1f`). Step 13 checks `category == "Zs"` directly instead.

## Running everything at once

```bash
chmod +x run_pipeline.sh
tmux new -s preprocess
./run_pipeline.sh 2>&1 | tee pipeline_run_log.txt
```
Detach with `Ctrl+B`, then `D`. Reattach anytime with `tmux attach -t preprocess`.

The script stops immediately if any individual step fails (via `set -e`), so a broken step won't silently corrupt everything downstream of it.

## What to check after running

1. Read the **step 14 output** — expect ~100–140 unique characters total, no `Cn` category entries, only one `Zs` (space) entry, and confirm `U+200C`/`U+200D` are present.
2. Spot-check a few real records from `pipeline_step13_sinhala_only.jsonl` for readability.
3. Confirm the record count didn't drop unexpectedly at any step — the per-step console output prints `Processed N records` / `Kept N records` at each stage, useful for spotting a step that's dropping far more than expected.
