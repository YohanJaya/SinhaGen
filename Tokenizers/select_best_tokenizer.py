"""
Select the best tokenizer (BPE vs Unigram, across all vocab_size /
max_sentencepiece_length configs) for a specific target LLM architecture.

HOW TO USE
----------
1. Run bpe.py and unigram.py as before (they already write tuning_results.csv
   into tuning_bpe/ and tuning_unigram/).
2. Edit the CONFIG block below to match your model and paths.
3. Run this script:  python select_best_tokenizer.py
4. It prints a ranked table and tells you which .model file to load.

WHAT IT DOES (plain-language)
------------------------------
Your tuning scripts already measured, per tokenizer config:
  - chars_per_tok       : compression (higher = fewer tokens for same text)
  - vocab_utilization_pct: how much of the vocab actually gets used
  - avg_tokens_per_sentence: proxy for sequence length / compute cost

This script adds what those CSVs can't know on their own: how expensive
each vocab_size is for the ACTUAL model you're about to train. It combines
compression + embedding cost + sequence cost + vocab utilization into a
single score, per candidate, so you can compare BPE and Unigram configs
on equal footing.
"""

import os
import pandas as pd


# =============================================================================
# CONFIG -- edit this block for your setup. Nothing else needs to change.
# =============================================================================

# --- Paths to your two tuning results CSVs ---
BPE_CSV     = "tuning_bpe/tuning_results.csv"
UNIGRAM_CSV = "tuning_unigram/tuning_results.csv"

# --- Your target LLM architecture ---
HIDDEN_DIM        = 768          # embedding/hidden size of the model you'll train
NON_EMBED_PARAMS  = 80_000_000  # approx. params in everything EXCEPT token embeddings
                                  # (attention + MLP layers). Rough estimate is fine.
TIE_EMBEDDINGS    = False        # True if input embedding and output head SHARE weights
                                  # (halves the embedding parameter cost if True)

# --- Minimum acceptable vocab utilization (matches your tuning scripts' gate) ---
MIN_VOCAB_UTILIZATION_PCT = 75.0

# --- How much each factor matters in the final score (raise/lower to taste) ---
WEIGHT_COMPRESSION      = 1.0   # reward higher chars_per_token
WEIGHT_PARAM_OVERHEAD   = 1.0   # penalize vocab eating into the parameter budget
WEIGHT_SEQ_COST         = 0.5   # penalize longer sequences (attention cost ~ n^2)
WEIGHT_VOCAB_UTIL       = 1.0   # reward vocab actually being used

# How many top candidates to show
TOP_N = 8

# =============================================================================
# END CONFIG -- nothing below this line needs editing for normal use
# =============================================================================


def load_results(csv_path: str, algo_name: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found, skipping {algo_name}.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["algo"] = algo_name
    return df


def normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Rescale a column to 0-1 so different metrics (e.g. chars_per_token vs
    embedding_overhead_pct) can be fairly added together."""
    s = series.astype(float)
    span = s.max() - s.min()
    if span == 0:
        return s * 0.0
    n = (s - s.min()) / span
    return n if higher_is_better else (1.0 - n)


def score_tokenizers(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # --- correctness gate: same rule as your tuning scripts ---
    d = d[(d["unk_rate_pct"] == 0.0) & (d["roundtrip_failure_pct"] == 0.0)]
    d = d[d["vocab_utilization_pct"] >= MIN_VOCAB_UTILIZATION_PCT]

    if d.empty:
        raise ValueError("No candidates passed the correctness/utilization gates. "
                          "Check your CSVs or lower MIN_VOCAB_UTILIZATION_PCT.")

    # --- embedding parameter cost for THIS model ---
    embed_multiplier = 1 if TIE_EMBEDDINGS else 2
    d["embed_params"] = embed_multiplier * d["vocab_size"] * HIDDEN_DIM
    d["total_params"] = d["embed_params"] + NON_EMBED_PARAMS
    d["embedding_overhead_pct"] = 100 * d["embed_params"] / d["total_params"]

    # --- sequence-length compute proxy (attention cost ~ n^2) ---
    d["seq_cost_proxy"] = d["avg_tokens_per_sentence"] ** 2

    # --- normalize each factor and combine into one score ---
    compression_n    = normalize(d["chars_per_token"], higher_is_better=True)
    overhead_n        = normalize(d["embedding_overhead_pct"], higher_is_better=False)
    seq_cost_n        = normalize(d["seq_cost_proxy"], higher_is_better=False)
    utilization_n     = normalize(d["vocab_utilization_pct"], higher_is_better=True)

    d["composite_score"] = (
        WEIGHT_COMPRESSION    * compression_n +
        WEIGHT_PARAM_OVERHEAD * overhead_n +
        WEIGHT_SEQ_COST       * seq_cost_n +
        WEIGHT_VOCAB_UTIL     * utilization_n
    )

    return d.sort_values("composite_score", ascending=False)


def main():
    bpe_df = load_results(BPE_CSV, "bpe")
    uni_df = load_results(UNIGRAM_CSV, "unigram")
    combined = pd.concat([bpe_df, uni_df], ignore_index=True)

    if combined.empty:
        print("No tuning result CSVs found. Check BPE_CSV / UNIGRAM_CSV paths.")
        return

    ranked = score_tokenizers(combined)

    print("=" * 100)
    print(f"Model assumptions: hidden_dim={HIDDEN_DIM}, "
          f"non_embed_params={NON_EMBED_PARAMS:,}, tied_embeddings={TIE_EMBEDDINGS}")
    print("=" * 100)

    display_cols = ["config", "algo", "vocab_size", "max_sentencepiece_length",
                     "chars_per_token", "vocab_utilization_pct", "embedding_overhead_pct",
                     "avg_tokens_per_sentence", "composite_score", "model_path"]
    display_cols = [c for c in display_cols if c in ranked.columns]

    print(ranked[display_cols].head(TOP_N).to_string(index=False))

    best = ranked.iloc[0]
    print("\n" + "=" * 100)
    print("RECOMMENDED TOKENIZER")
    print("=" * 100)
    print(f"  algo:                     {best['algo']}")
    print(f"  config:                   {best['config']}")
    print(f"  vocab_size:               {best['vocab_size']}")
    print(f"  embedding_overhead_pct:   {best['embedding_overhead_pct']:.2f}%")
    print(f"  model_path:               {best['model_path']}")
    print("\nLoad it in SentencePiece with:")
    print(f"    sp = spm.SentencePieceProcessor()")
    print(f"    sp.load('{best['model_path']}')")


if __name__ == "__main__":
    main()
