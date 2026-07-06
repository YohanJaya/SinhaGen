"""
Tokenize your full Sinhala dataset using the winning SentencePiece model
selected by select_best_tokenizer.py.

Produces one of two output formats:

  "packed_bin"  -> One big binary file of token IDs (uint16/uint32), all
                   documents concatenated with EOS separators. This is the
                   standard format for pretraining causal LMs (same idea as
                   nanoGPT's train.bin). Fast to load with np.memmap, no
                   per-example overhead, ideal for training from scratch.

  "jsonl"       -> One JSON object per line: {"input_ids": [...]}
                   Easier to inspect/debug, works directly with HuggingFace
                   `datasets.load_dataset("json", ...)`. Bigger on disk and
                   slower to load than packed_bin.

Usage:
    python tokenize_dataset.py
"""

import json
import os
import unicodedata

import numpy as np
import sentencepiece as spm


# =============================================================================
# CONFIG -- edit this block for your setup
# =============================================================================

# --- Path to the tokenizer model you selected as "best" ---
TOKENIZER_MODEL_PATH = "tuning_unigram/unigram_v16000_l16_i3.model"

# --- Input dataset(s): JSONL files with a "text" field, same format as
#     what bpe.py / unigram.py expect. Add train/val split paths here. ---
INPUT_FILES = {
    "train": "../pipeline_step14_sinhala_only.jsonl",
    # "val": "path/to/val.jsonl",   # add a val split if you have one held out
}

# --- Output format: "packed_bin" (recommended for pretraining) or "jsonl" ---
OUTPUT_FORMAT = "packed_bin"

# --- Where to write the tokenized output ---
OUTPUT_DIR = "tokenized_dataset"

# --- Add BOS/EOS tokens around each document? ---
ADD_BOS = False   # nanoGPT-style pretraining usually skips BOS
ADD_EOS = True    # EOS between documents is important so the model learns
                  # where one document ends and the next begins

# --- Print a progress update every N lines ---
LOG_EVERY = 50_000

# =============================================================================
# END CONFIG
# =============================================================================


def iter_texts(jsonl_path: str):
    """Yields normalized text lines from a JSONL file (same normalization
    logic as the tuning scripts, so tokenization matches what was tuned)."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            if not text:
                continue
            for piece in text.split("\n"):
                piece = unicodedata.normalize("NFC", piece.strip())
                if piece:
                    yield piece


def tokenize_to_packed_bin(sp: spm.SentencePieceProcessor, input_path: str, output_path: str):
    vocab_size = sp.get_piece_size()
    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    bos_id, eos_id = sp.bos_id(), sp.eos_id()

    all_ids = []  # accumulate then write once; switch to incremental memmap writes if RAM-constrained
    n_docs = 0
    n_tokens = 0

    for text in iter_texts(input_path):
        ids = sp.encode(text, out_type=int)
        if ADD_BOS and bos_id >= 0:
            ids = [bos_id] + ids
        if ADD_EOS and eos_id >= 0:
            ids = ids + [eos_id]
        all_ids.extend(ids)
        n_docs += 1
        n_tokens += len(ids)
        if n_docs % LOG_EVERY == 0:
            print(f"  ...{n_docs:,} docs tokenized, {n_tokens:,} tokens so far", flush=True)

    arr = np.array(all_ids, dtype=dtype)
    arr.tofile(output_path)
    print(f"Wrote {output_path}: {n_docs:,} docs, {n_tokens:,} tokens, dtype={dtype.__name__}")
    return n_docs, n_tokens


def tokenize_to_jsonl(sp: spm.SentencePieceProcessor, input_path: str, output_path: str):
    bos_id, eos_id = sp.bos_id(), sp.eos_id()
    n_docs = 0
    n_tokens = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for text in iter_texts(input_path):
            ids = sp.encode(text, out_type=int)
            if ADD_BOS and bos_id >= 0:
                ids = [bos_id] + ids
            if ADD_EOS and eos_id >= 0:
                ids = ids + [eos_id]
            fout.write(json.dumps({"input_ids": ids}) + "\n")
            n_docs += 1
            n_tokens += len(ids)
            if n_docs % LOG_EVERY == 0:
                print(f"  ...{n_docs:,} docs tokenized, {n_tokens:,} tokens so far", flush=True)

    print(f"Wrote {output_path}: {n_docs:,} docs, {n_tokens:,} tokens")
    return n_docs, n_tokens


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL_PATH)
    print(f"Loaded tokenizer: {TOKENIZER_MODEL_PATH} (vocab_size={sp.get_piece_size()})")

    summary = {}
    for split_name, input_path in INPUT_FILES.items():
        print(f"\nTokenizing '{split_name}' split from {input_path} ...")
        ext = "bin" if OUTPUT_FORMAT == "packed_bin" else "jsonl"
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.{ext}")

        if OUTPUT_FORMAT == "packed_bin":
            n_docs, n_tokens = tokenize_to_packed_bin(sp, input_path, output_path)
        elif OUTPUT_FORMAT == "jsonl":
            n_docs, n_tokens = tokenize_to_jsonl(sp, input_path, output_path)
        else:
            raise ValueError(f"Unknown OUTPUT_FORMAT: {OUTPUT_FORMAT}")

        summary[split_name] = {"docs": n_docs, "tokens": n_tokens, "path": output_path}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split_name, info in summary.items():
        print(f"  {split_name}: {info['docs']:,} docs, {info['tokens']:,} tokens -> {info['path']}")


if __name__ == "__main__":
    main()