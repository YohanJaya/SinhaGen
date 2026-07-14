# Sinhala LLM Tokenizers

This folder contains implementation scripts, results, and resources for tuning and selecting the optimal tokenizer for a Sinhala-focused Large Language Model.

---

## Tokenizer Algorithms Used

We evaluated two core subword tokenization algorithms using Google's **SentencePiece** library:

### 1. Byte Pair Encoding (BPE) — `bpe.py`
- **Methodology**: BPE is a bottom-up, greedy compression algorithm. It begins with a base vocabulary containing all individual characters (bytes) in the corpus. It then iteratively scans the corpus and merges the most frequently occurring adjacent pair of tokens to create a new token, terminating once the target `vocab_size` is reached.
- **Tuned Hyperparameters**:
  - `vocab_size`: `[4000, 8000, 16000, 32000, 48000, 64000]`
  - `max_sentencepiece_length`: `[4, 8, 16]` (controls the maximum character length for any single subword token).
- **Core characteristic**: Purely frequency-driven greedy merges. No probabilistic modeling is performed during training or inference.

### 2. Unigram Language Model — `unigram.py`
- **Methodology**: Unigram is a top-down, probabilistic vocabulary reduction algorithm. It starts by initializing a very large vocabulary (containing all characters and frequent substrings). It then calculates the probability of each token and iteratively prunes the vocabulary to maximize the likelihood of the training corpus.
- **Tuned Hyperparameters**:
  - `vocab_size`: `[4000, 8000, 16000, 32000, 48000, 64000]`
  - `max_sentencepiece_length`: `[4, 8, 16]`
  - `num_sub_iterations`: `3` (the number of EM sub-iterations run per pruning step to refine token probabilities).
- **Core characteristic**: Probabilistic and grammar-independent. It allows for multiple tokenizations of a single string during training (probabilistic sampling), which serves as a form of regularization (subword regularization).

---

## Tokenizer Evaluation Metrics

Candidates are trained on the training split and evaluated on a held-out validation split. Five key metrics are monitored:

1. **Compression Ratio (`chars_per_token`)**: 
   - *Definition*: `Total Characters / Total Tokens` on the validation set.
   - *Interpretation*: Higher is better. A higher metric means each token represents more raw characters on average, compressing the sequence length.
2. **Mean Path Length (`avg_tokens_per_sentence`)**:
   - *Definition*: Average number of tokens produced per sentence.
   - *Interpretation*: Lower is better. Since self-attention computational cost scales quadratically `O(N^2)` with token sequence length (`N`), minimising the number of average tokens directly reduces training and inference compute costs.
3. **Out-of-Vocabulary / Unknown Rate (`unk_rate_pct`)**:
   - *Definition*: Percentage of `<unk>` tokens generated on the validation split.
   - *Requirement*: Must be exactly `0.0%` (Correctness gate).
   - *Enforcement*: Both scripts use the **Byte Fallback** mechanism (`byte_fallback=True`). Any unseen or out-of-vocabulary character is split into its raw UTF-8 bytes (e.g. `<0xEF>`) rather than failing or falling back to a generic `<unk>` token.
4. **Vocabulary Utilization (`vocab_utilization_pct`)**:
   - *Definition*: `(Number of Unique Tokens Used in Validation / Total Vocabulary Size) * 100`.
   - *Interpretation*: Higher is better. A low utilization percentage indicates that the vocabulary is too large for the corpus, leading to "dead" tokens that waste parameters.
5. **Roundtrip Success Rate (`roundtrip_failure_pct`)**:
   - *Definition*: The percentage of sentences that fail to reconstruct exactly when encoded and decoded: `decode(encode(sentence)) != sentence`.
   - *Requirement*: Must be exactly `0.0%` (Correctness gate).

---

## The Selection Logic (`select_best_tokenizer.py`)

A larger vocabulary size gives higher compression (`chars_per_token`), but it introduces a major trade-off for smaller transformer models: it dramatically inflates the size of the embedding vocabulary matrices.

`select_best_tokenizer.py` evaluates all candidate BPE and Unigram tokenizers on equal footing for a specific LLM architecture by calculating a **Composite Score**:

### 1. Correctness Gate
Any tokenizer candidate is immediately rejected if:
- `unk_rate_pct > 0`
- `roundtrip_failure_pct > 0`
- `vocab_utilization_pct < 75.0%` (to prevent oversized, bloated vocabularies).

### 2. Embedding Parameter Overhead (`embedding_overhead_pct`)
If weights are not tied (`TIE_EMBEDDINGS = False`), the model needs two separate embedding matrices (an input lookup embedding and an output language modeling projection head):
`Embedding Parameters = 2 * vocab_size * HIDDEN_DIM`

For a hypothetical search model with `HIDDEN_DIM = 768` and `NON_EMBED_PARAMS = 80,000,000` (representing attention + MLP parameters):

| Vocab Size (V) | Embedding Params | Total Model Params | Embedding Overhead % |
|---|---|---|---|
| **4,000** | 6.14M | 86.14M | 7.13% |
| **8,000** | 12.29M | 92.29M | 13.31% |
| **16,000** | 24.58M | 104.58M | **23.50%** |
| **32,000** | 49.15M | 129.15M | 38.06% |
| **48,000** | 73.73M | 153.73M | 47.96% |
| **64,000** | 98.30M | 178.30M | 55.13% |

> [!WARNING]
> At a 64,000 vocabulary size, **55.13%** of the model parameters are spent on static lookup tables rather than active model reasoning capacity (attention layers, feedforward layers). This is a massive parameter bottleneck.

> [!NOTE]
> **Actual Training Configuration**: The settings above represent the parameters used in the search script `select_best_tokenizer.py`. When we trained the actual Llama weights (Runs 1-5), the model size was scaled down to a hidden size of `384` and weight embeddings were tied (`tie_word_embeddings: true`, multiplier 1). This reduced the actual 16,000-vocabulary model size to `~22.3M` total parameters (with `6.14M` parameters for embeddings, representing a `27.5%` overhead), making it much faster to train on local GPUs.

### 3. Quadratic Sequence Cost Proxy (`seq_cost_proxy`)
Because self-attention scales quadratically, the compute cost is modeled as:
`seq_cost_proxy = (avg_tokens_per_sentence) ** 2`

### 4. Normalization and Weights
Each metric is min-max normalized to `[0, 1]` (where 1 is the best candidate and 0 is the worst). The composite score is then calculated as:

```
Composite Score = (1.0 * chars_per_token_norm) 
                  + (1.0 * (1.0 - embedding_overhead_pct_norm)) 
                  + (0.5 * (1.0 - seq_cost_proxy_norm)) 
                  + (1.0 * vocab_utilization_pct_norm)
```

---

## Ranked Results

Running `select_best_tokenizer.py` on the BPE and Unigram tuning grids yields the following top 10 ranked candidates:

| Rank | Config | Algo | Vocab Size | Max Len | Chars/Token | Vocab Util % | Embedding Overhead % | Avg Tokens/Sent | Composite Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **v16000_l16_i3** | **unigram** | Run Settings | 16 | 4.309 | 98.32% | 23.50% | 41.26 | 2.732 |
| 2 | v16000_l16 | bpe | 16,000 | 16 | 4.256 | 98.28% | 23.50% | 41.77 | 2.700 |
| 3 | v32000_l16_i3 | unigram | 32,000 | 16 | 4.682 | 99.15% | 38.06% | 37.98 | 2.687 |
| 4 | v32000_l16 | bpe | 32,000 | 16 | 4.646 | 99.05% | 38.06% | 38.27 | 2.662 |
| 5 | v16000_l8 | bpe | 16,000 | 8 | 4.122 | 98.31% | 23.50% | 43.14 | 2.626 |
| 6 | v16000_l8_i3 | unigram | 16,000 | 8 | 4.092 | 98.26% | 23.50% | 43.45 | 2.605 |
| 7 | v48000_l16_i3 | unigram | 48,000 | 16 | 4.857 | 99.43% | 47.96% | 36.60 | 2.592 |
| 8 | v8000_l16_i3 | unigram | 8,000 | 16 | 3.856 | 96.65% | 13.31% | 46.11 | 2.570 |
| 9 | v48000_l16 | bpe | 48,000 | 16 | 4.834 | 99.17% | 47.96% | 36.78 | 2.562 |
| 10 | v32000_l8 | bpe | 32,000 | 8 | 4.412 | 99.11% | 38.06% | 40.30 | 2.539 |

---

## The Selected Tokenizer

Based on the composite score, the selected tokenizer is:

- **Model type**: `unigram`
- **Config tag**: `v16000_l16_i3`
- **Vocabulary Size**: 16,000
- **Max Sentencepiece Length**: 16
- **Sub-iterations**: 3
- **Model Path**: `Tokenizers/tokenizer/unigram_v16000_l16_i3.model`

### Architectural Rationale for Selection:
1. **High Compression at Zero Parameter Overkill**: At 16,000 vocabulary entries, the tokenizer achieves 4.309 characters per token, which is 31.5% higher than the 4,000 vocab model (3.275).
2. **Minimal Embedding Footprint**: It restricts embedding parameters to 23.5% of the hypothetical search model's total parameter budget, keeping 76.5% of the model capacity for attention layers and feeds. Moving to 32,000 or 48,000 vocabulary size increases the embedding footprint to 38% and 48% respectively, causing diminishing returns in training efficiency.
3. **Capture of Long Morphological Structures**: The maximum subword length of 16 characters performs significantly better than shorter lengths (4 or 8) because Sinhala is an agglutinative language. Long words can be chunked into single morphemic blocks rather than fragmented into individual characters.
4. **Unigram Advantage**: Probabilistic segmentation during training (regularization) and slightly higher compression relative to BPE at the same vocabulary size.
