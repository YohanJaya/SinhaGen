"""
Tune SentencePiece BPE hyperparameters for Sinhala.

Reads a JSONL dataset (one JSON object per line, each with a "text" field),
defines a grid of parameter combinations, loops over each one, trains a
tokenizer with those params, evaluates it against a held-out validation
split, then prints a summary of all candidates and picks the best one.

NOTE ON PARAMETERS vs. THE UNIGRAM SCRIPT:
BPE does not use `num_sub_iterations` — that parameter is specific to
Unigram's EM-based training algorithm. BPE instead builds its vocabulary
through greedy pairwise merges of the most frequent adjacent symbol pairs,
so there's no EM refinement step to tune. The grid below only searches
`vocab_size` and `max_sentencepiece_length`, matching what's actually
tunable for this algorithm.

Expected folder layout:
    project_root/
        pipeline_step14_sinhala_only.jsonl
        tokenizers/
            bpe.py   <- this script

Usage (run from inside the tokenizers/ folder):
    python bpe.py

Requires:
    pip install sentencepiece
"""

import argparse
import csv
import itertools
import json
import os
import random
import threading
import time
import unicodedata
from datetime import datetime

import sentencepiece as spm


# ---------------------------------------------------------------------------
# STEP 0: Define the parameter grid here.
# Every combination of these values will be trained and evaluated.
# Add/remove values from these lists to change what gets searched.
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "vocab_size": [4000, 8000, 16000, 32000, 48000, 64000],
    "max_sentencepiece_length": [4, 8, 16],
}

"""
Tune SentencePiece BPE hyperparameters for Sinhala.

Reads a JSONL dataset (one JSON object per line, each with a "text" field),
defines a grid of parameter combinations, loops over each one, trains a
tokenizer with those params, evaluates it against a held-out validation
split, then prints a summary of all candidates and picks the best one.

NOTE ON PARAMETERS vs. THE UNIGRAM SCRIPT:
BPE does not use `num_sub_iterations` — that parameter is specific to
Unigram's EM-based training algorithm. BPE instead builds its vocabulary
through greedy pairwise merges of the most frequent adjacent symbol pairs,
so there's no EM refinement step to tune. The grid below only searches
`vocab_size` and `max_sentencepiece_length`, matching what's actually
tunable for this algorithm.

Expected folder layout:
    project_root/
        pipeline_step14_sinhala_only.jsonl
        tokenizers/
            bpe.py   <- this script

Usage (run from inside the tokenizers/ folder):
    python bpe.py

Requires:
    pip install sentencepiece
"""

import argparse
import csv
import itertools
import json
import os
import random
import threading
import time
import unicodedata
from datetime import datetime

import sentencepiece as spm


# ---------------------------------------------------------------------------
# STEP 0: Define the parameter grid here.
# Every combination of these values will be trained and evaluated.
# Add/remove values from these lists to change what gets searched.
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "vocab_size": [4000, 8000, 16000, 32000, 48000, 64000],
    "max_sentencepiece_length": [4, 8, 16]
}



def build_param_combinations(grid: dict) -> list:
    """Turns the PARAM_GRID dict into a list of individual param dicts,
    one per combination, e.g.:
    [{"vocab_size": 8000, "max_sentencepiece_length": 16}, ...]
    """
    keys = list(grid.keys())
    value_combinations = itertools.product(*grid.values())
    return [dict(zip(keys, values)) for values in value_combinations]


def normalize_corpus(input_path: str, normalized_path: str) -> str:
    """
    Reads a JSONL file where each line looks like: {"text": "...sentence1...\\nsentence2..."}
    Extracts the 'text' field, splits it on internal \\n breaks (each JSON record
    holds multiple sentences/paragraphs joined by \\n), strips each resulting piece,
    NFC-normalizes it, and writes one clean sentence per line to a plain text file.
    """
    lines_written = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(normalized_path, "w", encoding="utf-8") as fout:
        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            text = record.get("text", "")
            if not text:
                skipped += 1
                continue

            # Each record's text may contain multiple sentences joined by \n
            for piece in text.split("\n"):
                piece = unicodedata.normalize("NFC", piece.strip())
                if piece:
                    fout.write(piece + "\n")
                    lines_written += 1

    print(f"Loaded JSONL: wrote {lines_written} normalized lines, skipped {skipped} bad/empty records")
    return normalized_path


def split_train_val(normalized_path: str, out_dir: str, val_fraction: float = 0.02, seed: int = 42):
    with open(normalized_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    n_val = max(1000, int(len(lines) * val_fraction))
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    train_path = os.path.join(out_dir, "train_split.txt")
    val_path = os.path.join(out_dir, "val_split.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"Split corpus: {len(train_lines)} train lines, {len(val_lines)} val lines")
    return train_path, val_path


def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Heartbeat:
    """Prints a 'still running' line every `interval` seconds in a background
    thread, so long silent SentencePiece training doesn't look stuck."""

    def __init__(self, label: str, interval: int = 30):
        self.label = label
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        start = time.time()
        while not self._stop_event.wait(self.interval):
            elapsed = round(time.time() - start)
            print(f"    ...[{now_str()}] still training {self.label} "
                  f"({elapsed}s elapsed) ...", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join()


# ---------------------------------------------------------------------------
# STEP 1: Train one BPE tokenizer given a single set of param values.
# ---------------------------------------------------------------------------
def train_candidate(train_path: str, params: dict, out_prefix: str):
    spm.SentencePieceTrainer.train(
        input=train_path,
        model_prefix=out_prefix,
        model_type="bpe",                # <-- the only algorithmic change vs. unigram.py
        vocab_size=params["vocab_size"],
        max_sentencepiece_length=params["max_sentencepiece_length"],
        character_coverage=1.0,          # fixed: Sinhala needs full char coverage
        byte_fallback=True,              # fixed: no information loss on unseen chars
        normalization_rule_name="identity",  # fixed: corpus already NFC-normalized
        input_sentence_size=2000000,     # cap training sample to avoid OOM on huge corpora
        shuffle_input_sentence=True,     # randomly sample instead of just taking the first N lines
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
    )


# ---------------------------------------------------------------------------
# STEP 2: Evaluate a trained tokenizer against the held-out validation set.
# Identical metrics to the unigram script, so the two are directly comparable.
# ---------------------------------------------------------------------------
def evaluate_candidate(model_path: str, val_path: str) -> dict:
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    with open(val_path, "r", encoding="utf-8") as f:
        val_lines = [l.strip() for l in f if l.strip()]

    total_tokens = 0
    total_chars = 0
    unk_count = 0
    roundtrip_failures = 0
    used_token_ids = set()
    unk_id = sp.unk_id()

    for line in val_lines:
        ids = sp.encode(line, out_type=int)
        total_tokens += len(ids)
        total_chars += len(line)
        unk_count += sum(1 for i in ids if i == unk_id)
        used_token_ids.update(ids)
        if sp.decode(ids) != line:
            roundtrip_failures += 1

    actual_vocab_size = sp.get_piece_size()

    return {
        "avg_tokens_per_sentence": round(total_tokens / len(val_lines), 2),
        "chars_per_token": round(total_chars / total_tokens, 3),   # higher = more compression
        "unk_rate_pct": round(100 * unk_count / total_tokens, 4),  # should be ~0
        "vocab_utilization_pct": round(100 * len(used_token_ids) / actual_vocab_size, 2),
        "roundtrip_failure_pct": round(100 * roundtrip_failures / len(val_lines), 4),  # should be 0
    }


# ---------------------------------------------------------------------------
# STEP 3: Pick the best candidate from all evaluated results.
#
# Selection rule (identical to the unigram script):
#   1. Reject any candidate with roundtrip failures or non-zero unk rate
#      (correctness gate — these indicate a bug, not a quality tradeoff).
#   2. Among the rest, reject vocab_utilization_pct below MIN_UTILIZATION
#      (means vocab_size is too big for the corpus — wasted capacity).
#   3. Among survivors, pick the one with the highest chars_per_token
#      (best compression) as the winner.
# ---------------------------------------------------------------------------
MIN_UTILIZATION_PCT = 75.0


def select_best(results: list) -> dict:
    valid = [r for r in results if r["unk_rate_pct"] == 0.0 and r["roundtrip_failure_pct"] == 0.0]
    if not valid:
        print("\nWARNING: no candidate passed the correctness gate (unk/roundtrip). "
              "Picking the least-bad option, but investigate preprocessing.")
        valid = results

    well_utilized = [r for r in valid if r["vocab_utilization_pct"] >= MIN_UTILIZATION_PCT]
    pool = well_utilized if well_utilized else valid

    return max(pool, key=lambda r: r["chars_per_token"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../pipeline_step14_sinhala_only.jsonl",
                         help="Path to the JSONL dataset (default: cleaned file in project root)")
    parser.add_argument("--val_input", default=None, help="Optional separate held-out file; if omitted, auto-splits --input")
    parser.add_argument("--out_dir", default="tuning_bpe")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[{now_str()}] Loading and normalizing corpus from {args.input} ...", flush=True)
    normalized_path = normalize_corpus(args.input, os.path.join(args.out_dir, "normalized.txt"))

    if args.val_input:
        train_path = normalized_path
        print(f"[{now_str()}] Loading and normalizing validation file from {args.val_input} ...", flush=True)
        val_path = normalize_corpus(args.val_input, os.path.join(args.out_dir, "normalized_val.txt"))
    else:
        print(f"[{now_str()}] No --val_input given, auto-splitting train/val from normalized corpus ...", flush=True)
        train_path, val_path = split_train_val(normalized_path, args.out_dir)

    print(f"[{now_str()}] Data ready. Train: {train_path} | Val: {val_path}", flush=True)

    param_combinations = build_param_combinations(PARAM_GRID)
    print(f"\n[{now_str()}] {len(param_combinations)} parameter combinations to try:", flush=True)
    for p in param_combinations:
        print(f"  {p}", flush=True)

    # -----------------------------------------------------------------
    # Main loop: substitute each param combination in, train, evaluate.
    # -----------------------------------------------------------------
    results = []
    for i, params in enumerate(param_combinations, 1):
        tag = f"v{params['vocab_size']}_l{params['max_sentencepiece_length']}"
        out_prefix = os.path.join(args.out_dir, f"bpe_{tag}")

        print("\n" + "#" * 70, flush=True)
        print(f"# CANDIDATE {i}/{len(param_combinations)}  |  config: {tag}", flush=True)
        print(f"# params: {params}", flush=True)
        print(f"# started: {now_str()}", flush=True)
        print("#" * 70, flush=True)

        start = time.time()
        with Heartbeat(label=tag, interval=30):
            train_candidate(train_path, params, out_prefix)
        train_seconds = round(time.time() - start, 1)

        print(f"[{now_str()}] Finished training {tag} in {train_seconds}s. Evaluating on held-out set...", flush=True)

        metrics = evaluate_candidate(out_prefix + ".model", val_path)
        metrics.update(params)
        metrics["config"] = tag
        metrics["train_seconds"] = train_seconds
        metrics["model_path"] = out_prefix + ".model"
        results.append(metrics)

        print(f"[{now_str()}] RESULT [{i}/{len(param_combinations)}] {tag}:", flush=True)
        for k, v in metrics.items():
            print(f"    {k}: {v}", flush=True)

    # -----------------------------------------------------------------
    # Summary of every candidate tried.
    # -----------------------------------------------------------------
    print("\n" + "=" * 105)
    print(f"{'config':<20}{'vocab':<8}{'max_len':<9}{'tok/sent':<10}"
          f"{'chars/tok':<11}{'unk_%':<9}{'vocab_util_%':<14}{'roundtrip_%':<12}{'train_s':<8}")
    print("-" * 105)
    for r in results:
        print(f"{r['config']:<20}{r['vocab_size']:<8}{r['max_sentencepiece_length']:<9}"
              f"{r['avg_tokens_per_sentence']:<10}{r['chars_per_token']:<11}{r['unk_rate_pct']:<9}"
              f"{r['vocab_utilization_pct']:<14}{r['roundtrip_failure_pct']:<12}{r['train_seconds']:<8}")

    # Save full results to CSV for later reference
    csv_path = os.path.join(args.out_dir, "tuning_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nFull results saved to: {csv_path}")

    # -----------------------------------------------------------------
    # Pick and report the best candidate.
    # -----------------------------------------------------------------
    best = select_best(results)
    print("\n" + "=" * 60)
    print("BEST BPE CONFIG")
    print("=" * 60)
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"\nSelection rule: highest compression (chars_per_token) among candidates "
          f"with 0% unk/roundtrip failures and >= {MIN_UTILIZATION_PCT}% vocab utilization.")
    print(f"Recommended model file: {best['model_path']}")


if __name__ == "__main__":
    main()


def build_param_combinations(grid: dict) -> list:
    """Turns the PARAM_GRID dict into a list of individual param dicts,
    one per combination, e.g.:
    [{"vocab_size": 8000, "max_sentencepiece_length": 16}, ...]
    """
    keys = list(grid.keys())
    value_combinations = itertools.product(*grid.values())
    return [dict(zip(keys, values)) for values in value_combinations]


def normalize_corpus(input_path: str, normalized_path: str) -> str:
    """
    Reads a JSONL file where each line looks like: {"text": "...sentence1...\\nsentence2..."}
    Extracts the 'text' field, splits it on internal \\n breaks (each JSON record
    holds multiple sentences/paragraphs joined by \\n), strips each resulting piece,
    NFC-normalizes it, and writes one clean sentence per line to a plain text file.
    """
    lines_written = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(normalized_path, "w", encoding="utf-8") as fout:
        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            text = record.get("text", "")
            if not text:
                skipped += 1
                continue

            # Each record's text may contain multiple sentences joined by \n
            for piece in text.split("\n"):
                piece = unicodedata.normalize("NFC", piece.strip())
                if piece:
                    fout.write(piece + "\n")
                    lines_written += 1

    print(f"Loaded JSONL: wrote {lines_written} normalized lines, skipped {skipped} bad/empty records")
    return normalized_path


def split_train_val(normalized_path: str, out_dir: str, val_fraction: float = 0.02, seed: int = 42):
    with open(normalized_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    n_val = max(1000, int(len(lines) * val_fraction))
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    train_path = os.path.join(out_dir, "train_split.txt")
    val_path = os.path.join(out_dir, "val_split.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"Split corpus: {len(train_lines)} train lines, {len(val_lines)} val lines")
    return train_path, val_path


def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Heartbeat:
    """Prints a 'still running' line every `interval` seconds in a background
    thread, so long silent SentencePiece training doesn't look stuck."""

    def __init__(self, label: str, interval: int = 30):
        self.label = label
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        start = time.time()
        while not self._stop_event.wait(self.interval):
            elapsed = round(time.time() - start)
            print(f"    ...[{now_str()}] still training {self.label} "
                  f"({elapsed}s elapsed) ...", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join()


# ---------------------------------------------------------------------------
# STEP 1: Train one BPE tokenizer given a single set of param values.
# ---------------------------------------------------------------------------
def train_candidate(train_path: str, params: dict, out_prefix: str):
    spm.SentencePieceTrainer.train(
        input=train_path,
        model_prefix=out_prefix,
        model_type="bpe",                # <-- the only algorithmic change vs. unigram.py
        vocab_size=params["vocab_size"],
        max_sentencepiece_length=params["max_sentencepiece_length"],
        character_coverage=1.0,          # fixed: Sinhala needs full char coverage
        byte_fallback=True,              # fixed: no information loss on unseen chars
        normalization_rule_name="identity",  # fixed: corpus already NFC-normalized
        input_sentence_size=2000000,     # cap training sample to avoid OOM on huge corpora
        shuffle_input_sentence=True,     # randomly sample instead of just taking the first N lines
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
    )


# ---------------------------------------------------------------------------
# STEP 2: Evaluate a trained tokenizer against the held-out validation set.
# Identical metrics to the unigram script, so the two are directly comparable.
# ---------------------------------------------------------------------------
def evaluate_candidate(model_path: str, val_path: str) -> dict:
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    with open(val_path, "r", encoding="utf-8") as f:
        val_lines = [l.strip() for l in f if l.strip()]

    total_tokens = 0
    total_chars = 0
    unk_count = 0
    roundtrip_failures = 0
    used_token_ids = set()
    unk_id = sp.unk_id()

    for line in val_lines:
        ids = sp.encode(line, out_type=int)
        total_tokens += len(ids)
        total_chars += len(line)
        unk_count += sum(1 for i in ids if i == unk_id)
        used_token_ids.update(ids)
        if sp.decode(ids) != line:
            roundtrip_failures += 1

    actual_vocab_size = sp.get_piece_size()

    return {
        "avg_tokens_per_sentence": round(total_tokens / len(val_lines), 2),
        "chars_per_token": round(total_chars / total_tokens, 3),   # higher = more compression
        "unk_rate_pct": round(100 * unk_count / total_tokens, 4),  # should be ~0
        "vocab_utilization_pct": round(100 * len(used_token_ids) / actual_vocab_size, 2),
        "roundtrip_failure_pct": round(100 * roundtrip_failures / len(val_lines), 4),  # should be 0
    }


# ---------------------------------------------------------------------------
# STEP 3: Pick the best candidate from all evaluated results.
#
# Selection rule (identical to the unigram script):
#   1. Reject any candidate with roundtrip failures or non-zero unk rate
#      (correctness gate — these indicate a bug, not a quality tradeoff).
#   2. Among the rest, reject vocab_utilization_pct below MIN_UTILIZATION
#      (means vocab_size is too big for the corpus — wasted capacity).
#   3. Among survivors, pick the one with the highest chars_per_token
#      (best compression) as the winner.
# ---------------------------------------------------------------------------
MIN_UTILIZATION_PCT = 75.0


def select_best(results: list) -> dict:
    valid = [r for r in results if r["unk_rate_pct"] == 0.0 and r["roundtrip_failure_pct"] == 0.0]
    if not valid:
        print("\nWARNING: no candidate passed the correctness gate (unk/roundtrip). "
              "Picking the least-bad option, but investigate preprocessing.")
        valid = results

    well_utilized = [r for r in valid if r["vocab_utilization_pct"] >= MIN_UTILIZATION_PCT]
    pool = well_utilized if well_utilized else valid

    return max(pool, key=lambda r: r["chars_per_token"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../pipeline_step14_sinhala_only.jsonl",
                         help="Path to the JSONL dataset (default: cleaned file in project root)")
    parser.add_argument("--val_input", default=None, help="Optional separate held-out file; if omitted, auto-splits --input")
    parser.add_argument("--out_dir", default="tuning_bpe")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[{now_str()}] Loading and normalizing corpus from {args.input} ...", flush=True)
    normalized_path = normalize_corpus(args.input, os.path.join(args.out_dir, "normalized.txt"))

    if args.val_input:
        train_path = normalized_path
        print(f"[{now_str()}] Loading and normalizing validation file from {args.val_input} ...", flush=True)
        val_path = normalize_corpus(args.val_input, os.path.join(args.out_dir, "normalized_val.txt"))
    else:
        print(f"[{now_str()}] No --val_input given, auto-splitting train/val from normalized corpus ...", flush=True)
        train_path, val_path = split_train_val(normalized_path, args.out_dir)

    print(f"[{now_str()}] Data ready. Train: {train_path} | Val: {val_path}", flush=True)

    param_combinations = build_param_combinations(PARAM_GRID)
    print(f"\n[{now_str()}] {len(param_combinations)} parameter combinations to try:", flush=True)
    for p in param_combinations:
        print(f"  {p}", flush=True)

    # -----------------------------------------------------------------
    # Main loop: substitute each param combination in, train, evaluate.
    # -----------------------------------------------------------------
    results = []
    for i, params in enumerate(param_combinations, 1):
        tag = f"v{params['vocab_size']}_l{params['max_sentencepiece_length']}"
        out_prefix = os.path.join(args.out_dir, f"bpe_{tag}")

        print("\n" + "#" * 70, flush=True)
        print(f"# CANDIDATE {i}/{len(param_combinations)}  |  config: {tag}", flush=True)
        print(f"# params: {params}", flush=True)
        print(f"# started: {now_str()}", flush=True)
        print("#" * 70, flush=True)

        start = time.time()
        with Heartbeat(label=tag, interval=30):
            train_candidate(train_path, params, out_prefix)
        train_seconds = round(time.time() - start, 1)

        print(f"[{now_str()}] Finished training {tag} in {train_seconds}s. Evaluating on held-out set...", flush=True)

        metrics = evaluate_candidate(out_prefix + ".model", val_path)
        metrics.update(params)
        metrics["config"] = tag
        metrics["train_seconds"] = train_seconds
        metrics["model_path"] = out_prefix + ".model"
        results.append(metrics)

        print(f"[{now_str()}] RESULT [{i}/{len(param_combinations)}] {tag}:", flush=True)
        for k, v in metrics.items():
            print(f"    {k}: {v}", flush=True)

    # -----------------------------------------------------------------
    # Summary of every candidate tried.
    # -----------------------------------------------------------------
    print("\n" + "=" * 105)
    print(f"{'config':<20}{'vocab':<8}{'max_len':<9}{'tok/sent':<10}"
          f"{'chars/tok':<11}{'unk_%':<9}{'vocab_util_%':<14}{'roundtrip_%':<12}{'train_s':<8}")
    print("-" * 105)
    for r in results:
        print(f"{r['config']:<20}{r['vocab_size']:<8}{r['max_sentencepiece_length']:<9}"
              f"{r['avg_tokens_per_sentence']:<10}{r['chars_per_token']:<11}{r['unk_rate_pct']:<9}"
              f"{r['vocab_utilization_pct']:<14}{r['roundtrip_failure_pct']:<12}{r['train_seconds']:<8}")

    # Save full results to CSV for later reference
    csv_path = os.path.join(args.out_dir, "tuning_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nFull results saved to: {csv_path}")

    # -----------------------------------------------------------------
    # Pick and report the best candidate.
    # -----------------------------------------------------------------
    best = select_best(results)
    print("\n" + "=" * 60)
    print("BEST BPE CONFIG")
    print("=" * 60)
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"\nSelection rule: highest compression (chars_per_token) among candidates "
          f"with 0% unk/roundtrip failures and >= {MIN_UTILIZATION_PCT}% vocab utilization.")
    print(f"Recommended model file: {best['model_path']}")


if __name__ == "__main__":
    main()