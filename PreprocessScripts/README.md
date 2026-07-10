# Sinhala Corpus Preprocessing Pipeline

A 15-stage preprocessing pipeline that converts raw, web-scraped Sinhala
text (sourced from MADLAD-400 and CulturaX) into a clean corpus suitable
for training a SentencePiece tokenizer.

## Table of Contents

- [Background](#background)
- [Why Sinhala Text Needs Special Handling](#why-sinhala-text-needs-special-handling)
- [Pipeline Architecture](#pipeline-architecture)
- [Step-by-Step Reference](#step-by-step-reference)
- [Design Decisions and Rationale](#design-decisions-and-rationale)
- [Usage](#usage)
- [Verifying the Output](#verifying-the-output)
- [Known Limitations](#known-limitations)
- [File Naming Convention](#file-naming-convention)

---

## Background

The source data is a combination of the Sinhala subsets of two large
web-crawled multilingual datasets: MADLAD-400 and CulturaX. Both are
known to contain the typical noise profile of web-scraped text:

- Boilerplate (navigation menus, footers, "related articles" widgets)
- Code-mixed content (English words, other South Asian scripts)
- Encoding corruption (from inconsistent source-page character encodings)
- Structural artifacts introduced during dataset assembly (e.g. a literal
  `\n` marker used in place of real newline characters within the `text`
  field of each JSON record)

Each record in the raw dataset is a JSON object of the form:
```json
{"text": "sentence one...\nsentence two...\nsentence three..."}
```
one line per document, `.jsonl` format throughout the pipeline.

The goal of this pipeline is to progressively strip all of that noise
down to clean, structurally-intact Sinhala text, ending in a final
character-level whitelist pass that guarantees nothing but valid Sinhala
content remains before it's used for tokenizer training.

## Why Sinhala Text Needs Special Handling

Sinhala (Unicode block `U+0D80`–`U+0DFF`) is an abugida script with
several properties that make naive text-cleaning approaches (built with
Latin-script assumptions) actively harmful if applied carelessly:

**Combining marks are separate Unicode characters.**
Vowel signs, the virama (*hal kirīma*), and the anusvaraya/visargaya are
not part of the base letter, they are distinct Unicode codepoints in
categories `Mn` (non-spacing mark) and `Mc` (spacing combining mark)
layered onto a base consonant. A cleaning step that only protects
"letters" (Unicode category `Lo`) and treats everything else as
disposable punctuation will silently mangle every word containing a
vowel sign.

**Conjunct consonants rely on an invisible joining character.**
Certain consonant clusters — for example the ligature in `ශ්‍රී` (as in
`ශ්‍රී ලංකාව`) are formed using the Zero Width Joiner (`U+200D`)
between two letters. This is a real structural character, not a
formatting artifact, even though it produces no visible glyph of its
own and falls *outside* the main Sinhala Unicode block. Any cleaning
step that removes "invisible" or "non-Sinhala-range" characters without
an explicit exception will fragment these words.

**The Sinhala Unicode block contains unassigned gaps.**
Not every codepoint between `U+0D80` and `U+0DFF` has been assigned a
meaning by the Unicode Standard (e.g. `U+0DB2`, `U+0DBC`, `U+0DBE`).
Their presence in text is not a legitimate rare character — it is a
symptom of encoding corruption (e.g. a page declared as UTF-8 but
actually served in a different encoding, decoded incorrectly). A
whitelist based purely on numeric range, without also checking that the
codepoint is an *assigned* character, will let corruption through.

**Python's `str.isspace()` is not a reliable whitespace test.**
It returns `True` for a handful of ASCII control characters (e.g.
`U+001F`, the "Unit Separator") that are not visual whitespace at all —
they are leftover control codes, typically from corrupted source data.
Any step relying on `isspace()` to decide "this is safe to keep" can let
genuine control-character corruption through undetected.

These four points directly motivated the design of the final whitelist
stage (step 13) and the general character-handling philosophy used
throughout the pipeline.

## Pipeline Architecture

The pipeline is organized into three phases:

**Phase 1 — Structural repair (steps 00–02)**
Fixes the two things that must be correct before any content-based
cleaning can safely run: consistent Unicode representation, and intact
newline/paragraph structure.

**Phase 2 — Content-aware cleaning (steps 03–12)**
Targeted removal of specific known noise patterns: punctuation
artifacts, repeated boilerplate, low-quality/duplicate documents,
emoji, date/timestamp metadata, fused numbers, navigation widgets, and
escaping artifacts. Each of these requires pattern-specific logic that
a generic character filter cannot express.

**Phase 3 — Final whitelist and verification (steps 13–14)**
A character-level safety net that removes anything the targeted steps
didn't anticipate (leftover foreign scripts, symbols, control
characters), followed by a QA report confirming the result.

Each step reads the previous step's output file and writes a brand-new
output file — no step modifies its input in place. This makes it
straightforward to inspect any intermediate stage in isolation if a
downstream step produces unexpected results.

## Step-by-Step Reference

| # | Script | Purpose |
|---|--------|---------|
| 00  | `00_normalize_unicode.py` | Applies NFC (Canonical Composition) Unicode normalization. Ensures every visually-identical Sinhala character sequence is represented by the same sequence of codepoints, regardless of which source website/scraper originally produced it. Runs first because it is independent of every other step and establishes a consistent baseline for all subsequent regex/character-class logic. | 
| 01 | `01_fix_broken_newlines.py` | Converts the literal two-character `\n` marker into a real newline character. Also repairs documents where that marker was already damaged by earlier processing (backslash stripped, leaving an orphan `n`/`nn`). Runs before any word-removal step — see [Design Decisions](#design-decisions-and-rationale) for why order matters here. |
| 02 | `02_remove_english_alphanumeric.py` | Removes standalone English words and garbled alphanumeric tokens (e.g. `iPhone13`, `234FGH`) that appear mixed into otherwise-Sinhala text. Pure numeric tokens are preserved by default, since years/quantities are often meaningful content. |
| 03 | `03_clean_punctuation.py` | Removes empty bracket leftovers (`()`, `[ ]`) created by earlier link/citation stripping, and collapses runs of repeated or mixed punctuation (`...`, `?!`, `--`) down to a single mark. Explicitly protects Sinhala combining marks and the zero-width joiner from being treated as removable punctuation. |
| 04 | `04_remove_boilerplate.py` | Identifies lines that repeat across an unusually large number of distinct documents (site navigation, footers, comment-widget text) and strips them. Uses a document-frequency (If a sentence occurs in many documents than the threshold it will be removed) threshold rather than raw occurrence count, so genuinely common short phrases aren't mistaken for template boilerplate. |
| 05 | `05_quality_filter_and_split.py` | Drops exact-duplicate documents, documents below a minimum length, and documents where the proportion of Sinhala-script characters falls below a quality threshold (a proxy for "this record is mostly non-Sinhala junk"). |
| 06 | `06_remove_emojis.py` | Removes emoji character sequences, including multi-codepoint emoji joined with ZWJ (e.g. family/flag emoji). Distinguishes emoji-joining ZWJ from Sinhala-conjunct ZWJ by only stripping a ZWJ when it sits directly between two emoji codepoints. |
| 07 | `07_clean_whitespace.py` | Collapses repeated spaces/tabs, converts non-breaking spaces to regular spaces, and trims leading/trailing whitespace at both the line and document level. |
| 08 | `08_clean_punctuation2.py` | A second punctuation pass, run after several other steps have already altered the text (which can create new empty-bracket or repeated-punctuation cases the first pass couldn't have seen). Also blanks any line that, after cleaning, contains no letters or digits at all — a sign the entire line was junk. |
| 09 | `09_remove_dates_times.py` | Strips date/time stamps in both Sinhala (month names, weekday names) and numeric formats. These are page/comment metadata (e.g. "Posted on [date]"), not article content. |
| 10 | `10_remove_fused_numbers.py` | Removes digit sequences fused directly onto a Sinhala word with no separating space (e.g. `කරන0456` → `කරන`) — a common scraping artifact. Numbers that are genuinely part of a sentence (space-separated) are left untouched. |
| 11 | `11_remove_nav_titles.py` | Removes "previous post / next post" navigation snippets, identified by the guillemet-quote pattern (`« ... »`) commonly used by blog templates for this widget. |
| 12 | `12_remove_back_slashes.py` | Removes stray escaping backslashes left over from double-encoded JSON (e.g. `\"` → `"`) and stray ellipsis characters/sequences, while leaving the literal `\n` line-break marker completely untouched (it is not present anymore by this point, since step 01 already converted it — this step's exclusion rule is a defensive safeguard). |
| 13 | `13_finalize_sinhala_only.py` | The final character-level safety net. Whitelists only: the Sinhala Unicode block (excluding unassigned codepoints), the zero-width joiner/non-joiner, ASCII digits, a small fixed punctuation set, and normalized whitespace. Everything else — regardless of how it got there — is removed. |
| 14 | `14_inspect_alphabet.py` | A verification/QA script (produces a report, not a new data file). Lists every unique character remaining in the final corpus with its Unicode codepoint, category, and name, so the corpus can be visually confirmed clean before it's used for tokenizer training. |

## Design Decisions and Rationale

### Why Unicode normalization runs before everything else

NFC normalization only affects how Unicode combining sequences are
represented — it never touches plain ASCII characters like `\` or `n`.
This makes it safe, order-independent, and foundational: every later
step's regex patterns and character-class checks are written assuming a
single canonical representation, so establishing that representation
first prevents subtle mismatches later in the pipeline.

### Why newline-marker repair runs before English-word removal

This is the one true ordering dependency in the pipeline. The
English-word-removal step deletes any standalone run of ASCII letters —
which includes a lone orphaned `n`. If a document's `\n` marker had
already lost its backslash from some earlier, separate processing pass
(leaving a bare `n` as the only surviving trace of a line break),
running word-removal *before* the repair step would treat that orphan
`n` as ordinary English text and delete it — destroying the paragraph
break permanently, with no way to recover it afterward.

Running the repair step first guarantees that every line break — intact
or already-damaged — becomes a real newline character before any
word-removal regex ever examines the text. At that point, the character
`n` no longer needs special-case protection, because it's no longer
present in the text at all (it has already become `\n`, a control
character, not a letter).

### Why boilerplate removal runs before deduplication

Two scraped documents can have identical actual article content but
different surrounding site-template text (different nav widget state,
different footer). Removing boilerplate first, then deduplicating,
correctly identifies these as duplicates. Deduplicating first would
treat them as distinct documents purely because of template noise.

### Why the final whitelist step exists at all, given 13 prior cleaning steps

Steps 00–12 are all *targeted*: each one recognizes and removes a
specific, known noise pattern. None of them provide a guarantee about
what character set survives at the end — they only guarantee that the
patterns they were built for are gone. In practice, running only steps
00–12 on this dataset still left several thousand unique characters in
the corpus: leftover Tamil and Arabic script fragments, private-use-area
and unassigned Unicode codepoints (encoding corruption), and assorted
symbols. A genuinely clean Sinhala corpus should contain on the order of
100–140 unique characters (the base alphabet, vowel signs, virama,
ZWJ/ZWNJ, digits, and a small punctuation set) — not thousands. Step 13
exists to close that gap categorically, rather than adding an
indefinite sequence of one-off patches for each new contamination
pattern discovered.

### Why disallowed characters are handled differently depending on visibility

In step 13, a disallowed character's *category* determines how it's
removed:
- **Invisible/control categories** (`Cf` format characters, `Cc` control
  characters, `Cn` unassigned codepoints) are deleted outright, with
  nothing inserted in their place. Inserting a visible space where an
  invisible character used to be would incorrectly introduce a word
  break that never existed in the original text.
- **Visible disallowed characters** (foreign-script letters, symbols,
  punctuation outside the whitelist) are replaced with a single space.
  This prevents two words that were separated only by the removed
  character from accidentally fusing into one.

## Usage

Run scripts in numeric order. Each script's `INPUT_FILE` and
`OUTPUT_FILE` constants are defined near the top of the file — edit
these if your directory layout differs from the default (`~/` for all
intermediate files).

Run the entire pipeline in one command:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh 2>&1 | tee pipeline_run_log.txt
```

The runner stops immediately if any individual step fails, so a broken
step cannot silently corrupt everything downstream of it. For a corpus
of non-trivial size, running this inside a persistent terminal
multiplexer session (e.g. `tmux` or `screen`) is recommended, since the
full run can take a significant amount of time and should not be tied
to an active SSH connection.

Each script can also be run individually:
```bash
python3 04_remove_boilerplate.py
```

## Verifying the Output

After running the full pipeline, review the report printed by step 14
(`14_inspect_alphabet.py`). A healthy result should show:

- **Roughly 100–140 unique characters total.** A count in the
  thousands indicates upstream contamination that steps 00–13 did not
  fully catch, and warrants re-checking the input data or an
  intermediate stage.
- **No `Cn` (unassigned) category entries.** Their presence indicates
  encoding corruption slipping through.
- **No stray `Cc` (control character) entries other than `\n` itself.**
- **Exactly one whitespace/space-like entry** (category `Zs`). Multiple
  distinct space variants indicate the whitespace-normalization logic
  isn't being applied consistently.
- **`U+200C` (ZWNJ) and `U+200D` (ZWJ) present.** Their absence would
  indicate the conjunct-preservation logic has regressed — these
  characters are required, not optional.
- **No unexpected non-Sinhala letters** (Tamil, Arabic, Latin, etc.)
  beyond the explicitly-allowed ASCII digits and punctuation.

It is also worth manually reading a handful of records from the final
output file (`pipeline_step13_sinhala_only.jsonl`), specifically looking
for words containing consonant conjuncts (e.g. `ශ්‍රී`), to visually
confirm they render correctly and were not fragmented by any step in
the pipeline.

## Known Limitations

- **Boilerplate detection (step 04) is threshold-based**
  (`MIN_REPEAT_DOCS`, `MIN_LINE_LENGTH`). If the corpus draws from many
  different site templates rather than a few common ones, this
  threshold may need adjustment — check the "top repeated lines being
  removed" printout on first run against real data to confirm it is
  catching genuine boilerplate rather than common Sinhala phrases.
- **The date/time removal patterns (step 09) are built from observed
  formats in this specific dataset.** New source data with different
  date formatting conventions may require additional patterns.
- **The punctuation whitelist (step 13) is deliberately small** (`.
  , ! ? ; : ' " ( ) - /`). Any punctuation mark not in this list —
  including some marks that may be legitimate in specific contexts — is
  removed. This was a deliberate simplicity/safety tradeoff; expand the
  `ALLOWED_PUNCTUATION` set in step 13 if a specific mark turns out to
  be needed.
- **Memory usage scales with corpus size** for any step that loads
  full-corpus statistics into memory at once (notably step 04's
  document-frequency counting pass, and step 05's in-memory
  deduplication set). Very large corpora may require processing in
  chunks or streaming variants of these steps.

## File Naming Convention

All intermediate files follow the pattern:
```
pipeline_step<N>_<description>.jsonl
```
where `<N>` matches the step number that produced it. This makes it
possible to identify which script generated any given intermediate file
purely from its filename, and to resume the pipeline from any point by
pointing the next script's `INPUT_FILE` at the desired intermediate
file.