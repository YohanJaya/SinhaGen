"""
visualize_training.py

A small, local visualization tool. wandb's website already gives you full
interactive dashboards (this is the recommended place to compare runs), but
this script is handy for a quick offline look, or for including a plot in
a report.

It reads the `trainer_state.json` file that the Hugging Face Trainer
automatically writes into your checkpoint folder during training, and
plots training loss, validation loss, and learning rate over time.

HOW TO USE
----------
    python visualize_training.py --checkpoint_dir ./checkpoints/run1

This looks for ./checkpoints/run1/trainer_state.json and saves a PNG next
to it.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def load_log_history(checkpoint_dir: str):
    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        # Trainer sometimes nests the latest checkpoint in a subfolder like
        # checkpoint-1000/ — search for it if the top-level file is missing.
        candidates = [
            os.path.join(checkpoint_dir, d, "trainer_state.json")
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
        ]
        candidates = [c for c in candidates if os.path.exists(c)]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find trainer_state.json under {checkpoint_dir}"
            )
        state_path = sorted(candidates)[-1]  # most recent checkpoint

    with open(state_path, "r") as f:
        state = json.load(f)
    return state["log_history"]


def plot(log_history, output_path: str):
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    lr_steps, lr_values = [], []

    for entry in log_history:
        step = entry.get("step")
        if "loss" in entry:  # training loss, logged every `logging_steps`
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:  # validation loss, logged every `eval_steps`
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])
        if "learning_rate" in entry:
            lr_steps.append(step)
            lr_values.append(entry["learning_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(train_steps, train_loss, label="train loss", color="#4C72B0")
    if eval_loss:
        axes[0].plot(eval_steps, eval_loss, label="validation loss",
                     color="#DD8452", marker="o", markersize=3)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss over training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(lr_steps, lr_values, color="#55A868")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("Learning rate schedule")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True,
                         help="Folder passed as --output_dir to train.py")
    parser.add_argument("--output_path", default=None,
                         help="Where to save the PNG (default: inside checkpoint_dir)")
    args = parser.parse_args()

    log_history = load_log_history(args.checkpoint_dir)
    output_path = args.output_path or os.path.join(
        args.checkpoint_dir, "training_curves.png"
    )
    plot(log_history, output_path)


if __name__ == "__main__":
    main()