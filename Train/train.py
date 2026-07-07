"""
train.py

Trains the Llama-style decoder model FROM SCRATCH (random initial weights —
we are not fine-tuning an existing model) on your tokenized dataset, and
logs everything to Weights & Biases (wandb) so you can watch loss curves
and compare hyperparameter runs in a browser dashboard.

This version reads pre-tokenized, packed binary files (produced by your own
packing step) directly via numpy.memmap, instead of loading a HuggingFace
`datasets` folder. File paths are fixed constants below (edit them directly
if your layout changes) — only hyperparameters are exposed as CLI flags.

BEFORE RUNNING
--------------
1. pip install transformers accelerate wandb torch numpy
2. wandb login          (one-time; it will ask for an API key from wandb.ai)
3. Make sure train_split.bin and validation.bin already exist at the paths
   below (produced by your tokenizing/packing script).

HOW TO USE
----------
    python train.py \
        --wandb_project my-small-llm \
        --run_name run1-lr3e-4 \
        --learning_rate 3e-4 \
        --batch_size 32 \
        --num_train_epochs 3

Change --run_name and any hyperparameter flags to start a new, separately
tracked experiment in wandb without touching this file.
"""
import argparse
import os

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# ---------------------------------------------------------------------------
# Fixed file paths — edit these directly, they are NOT command-line args.
# Assumes this script lives in a `train/` folder, sitting next to a
# `Tokenizers/tokenized_dataset/` folder one level up:
#
#   project_root/
#     train/
#       train.py            <- this file
#       config.json
#       visualize_training.py
#     Tokenizers/
#       tokenized_dataset/
#         train_split.bin
#         validation.bin
# ---------------------------------------------------------------------------
CONFIG_PATH = "./config.json"

TRAIN_BIN_PATH = "../Tokenizers/tokenized_dataset/train_split.bin"
VAL_BIN_PATH = "../Tokenizers/tokenized_dataset/validation.bin"

OUTPUT_DIR = "./checkpoints/sinhala-run1"


class PackedBinaryDataset(Dataset):
    """Reads a flat, pre-tokenized .bin file via memmap and chops it into
    fixed-length, non-overlapping blocks of `block_size` tokens.

    Note: this discards any leftover tokens at the end that don't fill a
    full block. That's normal and expected for packed pretraining data.
    """

    def __init__(self, bin_path, block_size, vocab_size):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"Could not find {bin_path}. Run your tokenizing/packing "
                "step first, or check TRAIN_BIN_PATH / VAL_BIN_PATH at the "
                "top of this file."
            )

        # Must match the dtype used when the .bin file was written.
        dtype = np.uint16 if vocab_size < 65536 else np.uint32

        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.block_size = block_size
        self.n_blocks = len(self.data) // block_size

        print(f"\nDataset: {bin_path}")
        print(f"Tokens: {len(self.data):,}")
        print(f"Blocks: {self.n_blocks:,} (block_size={block_size})")

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size

        # .astype(np.int64) first so torch doesn't have to guess how to
        # widen an unsigned 16/32-bit numpy dtype into a signed int64.
        tokens = torch.from_numpy(self.data[start:end].astype(np.int64))

        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }


def parse_args():
    p = argparse.ArgumentParser()

    # wandb
    p.add_argument("--wandb_project", default="small-llm-scratch")
    p.add_argument("--run_name", default=None,
                    help="Name shown for this run in the wandb dashboard. "
                         "Give each experiment a distinct name.")

    # hyperparameters you'll want to experiment with
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32,
                    help="Per-device batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Simulates a bigger batch size without more memory. "
                         "Effective batch size = batch_size * this value.")
    p.add_argument("--num_train_epochs", type=float, default=3.0,
                    help="How many full passes over the training data")
    p.add_argument("--max_steps", type=int, default=-1,
                    help="If set to a positive number, stop after this many "
                         "training steps regardless of --num_train_epochs. "
                         "Useful for a quick smoke test, e.g. --max_steps 50, "
                         "before committing to a full multi-day run.")
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                    help="Fraction of training spent gradually ramping the "
                         "learning rate up from 0, which stabilizes early "
                         "training")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # 1. wandb.init starts a tracked "run". Everything the Trainer logs
    #    (loss, learning rate, eval metrics, GPU usage) will stream live
    #    to https://wandb.ai under this project/run name.
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # Detect whether a GPU is actually available. Note: this also works for
    # AMD GPUs -- a ROCm-built PyTorch still exposes itself through the same
    # torch.cuda.* API, so no code branch is needed for AMD vs NVIDIA here.
    # What differs is only the wheel you install (see README's AMD/ROCm
    # section) and which dtype is safe to train in (checked below).
    has_gpu = torch.cuda.is_available()
    print(f"GPU available: {has_gpu}")
    if has_gpu:
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # bf16 mixed precision is a GPU feature -- forcing it on a CPU-only
    # machine either errors out or provides no speed benefit. On GPU, not
    # every card supports bf16 (this varies more on AMD/ROCm than on recent
    # NVIDIA cards), so we ask PyTorch directly instead of assuming, and
    # fall back to fp16 rather than guessing wrong.
    use_bf16 = has_gpu and torch.cuda.is_bf16_supported()
    use_fp16 = has_gpu and not use_bf16
    if has_gpu:
        print(f"Mixed precision: {'bf16' if use_bf16 else 'fp16'}")
    else:
        n_threads = torch.get_num_threads()
        print(f"Running on CPU with {n_threads} threads. "
              "This will be slow for a full training run -- consider "
              "using --max_steps for a quick smoke test, then moving the "
              "full run to your AMD GPU machine.")

    # 2. Build the model from your config file, with FRESH random weights.
    #    This is different from AutoModel.from_pretrained(...), which would
    #    download pretrained weights. Here we're training from zero.
    #    from_json_file is used (not from_pretrained) because CONFIG_PATH
    #    points at the config.json file itself, not a model directory.
    config = LlamaConfig.from_json_file(CONFIG_PATH)
    model = LlamaForCausalLM(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {n_params:,} parameters (~{n_params/1e6:.1f}M)")
    wandb.summary["num_parameters"] = n_params

    # 3. Load the packed binary datasets directly via memmap. block_size is
    #    taken from the model config so training blocks always match the
    #    model's context window, and vocab_size determines whether the .bin
    #    files were written as uint16 or uint32.
    block_size = config.max_position_embeddings
    train_dataset = PackedBinaryDataset(TRAIN_BIN_PATH, block_size, config.vocab_size)
    eval_dataset = PackedBinaryDataset(VAL_BIN_PATH, block_size, config.vocab_size)

    # 4. Our dataset already returns same-length input_ids/labels tensors
    #    per example, so we only need to stack them into a batch --
    #    default_data_collator does exactly that. We do NOT use
    #    DataCollatorForLanguageModeling here: that collator is meant for
    #    on-the-fly masking/padding of variable-length examples, neither of
    #    which applies to our fixed-length packed blocks.
    data_collator = default_data_collator

    # 5. TrainingArguments controls the training loop's behavior.
    #    report_to="wandb" is the one line that connects training to your
    #    wandb dashboard — Trainer logs loss/lr/eval metrics automatically.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        run_name=args.run_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="wandb",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 6. Save the final model. Your SentencePiece tokenizer (.model file)
    #    is saved separately outside this script, so it's not written here.
    #    If you want it sitting alongside checkpoints for easy generation
    #    later, copy it into OUTPUT_DIR yourself, e.g.:
    #        cp /path/to/tokenizer.model {OUTPUT_DIR}/tokenizer.model
    trainer.save_model(OUTPUT_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()
