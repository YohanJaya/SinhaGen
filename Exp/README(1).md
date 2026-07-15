# Sinhala LLM Pretraining - Hyperparameter Search Results

This document summarizes five pretraining runs of a from-scratch Sinhala Llama-style decoder, logged to Weights & Biases (`sinhala-llm` project). The goal of the sweep was to find a stable learning rate and training duration before committing to a full training run.

---

## How the runs were launched

```bash
cd /workspace/train

# Run 1 - baseline
python train.py \
  --wandb_project sinhala-llm \
  --run_name run1-lr3e-4-bs32 \
  --learning_rate 3e-4 \
  --batch_size 32 \
  --num_train_epochs 3

# Run 2 - high LR, short run (KILLED - loss diverging upward)
python train.py \
  --wandb_project sinhala-llm \
  --run_name run2-lr3e-4-bs32 \
  --learning_rate 1e-2 \
  --batch_size 32 \
  --num_train_epochs 1

# Run 3 - high LR, longer run (KILLED - loss diverging upward)
python train.py \
  --wandb_project sinhala-llm \
  --run_name run3-lr3e-4-bs32 \
  --learning_rate 1e-2 \
  --batch_size 32 \
  --num_train_epochs 3

# Run 4 - low LR
python train.py \
  --wandb_project sinhala-llm \
  --run_name run4-lr3e-4-bs32 \
  --learning_rate 1e-4 \
  --batch_size 32 \
  --num_train_epochs 3

# Run 5 - baseline LR, extended training
python train.py \
  --wandb_project sinhala-llm \
  --run_name run5-lr3e-4-bs32 \
  --learning_rate 3e-4 \
  --batch_size 32 \
  --num_train_epochs 4
```

> **Note:** Runs 2 and 3 were manually killed mid-training because the loss was trending upward instead of decreasing, indicating the learning rate (`1e-2`) was too high for this model/optimizer setup.

---

## Run Summary

| Run | Learning Rate | Batch Size | Epochs | Status | Outcome |
|---|---|---|---|---|---|
| **Run 1** | `3e-4` | 32 | 3 |  Completed | Stable, smooth loss decay - solid baseline |
| **Run 2** | `1e-2` | 32 | 1 |  Killed | Loss diverged upward almost immediately |
| **Run 3** | `1e-2` | 32 | 3 |  Killed | Same divergence as Run 2, over a longer window |
| **Run 4** | `1e-4` | 32 | 3 |  Completed | Stable but slow - under-converged by step limit |
| **Run 5** | `3e-4` | 32 | 4 |  Completed | Best result - lowest train/eval loss, no overfitting |

---

## Run 1 - Baseline (`lr=3e-4`, 3 epochs)

Loss starts high and decreases smoothly and consistently across training, with no spikes or instability. Evaluation loss tracks the training loss closely, which is a good sign that the model is generalizing rather than memorizing. This run is treated as the reference point for the other configurations.

![Run 1 Summary](run1/image.png)
![Run 1 Train](run1/train.png)
![Run 1 Eval](run1/eval.png)

---

## Run 2 - High LR, 1 epoch (`lr=1e-2`) - KILLED

With a learning rate of `1e-2`, the loss does not decrease - it rises and becomes erratic almost from the start. This is the classic signature of the optimizer taking steps that are too large, pushing the weights past good minima and into regions where the loss increases. The run was terminated manually since it was clearly not going to recover.

![Run 2 Summary](run2/image.png)
![Run 2 Train](run2/train.png)
![Run 2 Eval](run2/eval.png)

---

## Run 3 - High LR, 3 epochs (`lr=1e-2`) - KILLED

A repeat of Run 2's learning rate but scheduled for more epochs, to double check the instability wasn't a one-off. The same divergence pattern appears - loss trending upward / oscillating rather than converging - confirming `1e-2` is unusable for this setup regardless of run length. Killed for the same reason as Run 2.

![Run 3 Summary](run3/image.png)
![Run 3 Train](run3/train.png)
![Run 3 Eval](run3/eval.png)

---

## Run 4 - Low LR (`lr=1e-4`, 3 epochs)

A much more conservative learning rate. Training is stable throughout, with no divergence, but the loss decreases more slowly than Run 1 and settles at a noticeably higher value by the same step count. This suggests `1e-4` is "safe" but leaves performance on the table within a fixed training budget.

![Run 4 Summary](run4/image.png)
![Run 4 Train](run4/train.png)
![Run 4 Eval](run4/eval.png)

---

## Run 5 - Extended Baseline (`lr=3e-4`, 4 epochs)

Same learning rate as Run 1, but trained for one additional epoch. Both training and evaluation loss continue to decrease beyond where Run 1 stopped, and evaluation loss does not turn upward - meaning the extra epoch did not cause overfitting. This is the best-performing run in the sweep.

![Run 5 Summary](run5/image.png)
![Run 5 Train](run5/train.png)
![Run 5 Eval](run5/eval.png)

---

## Takeaways

1. **`lr=1e-2` is unusable** for this model - it causes immediate loss divergence regardless of training length (Runs 2 & 3).
2. **`lr=1e-4` is safe but too slow** - it converges without instability but underperforms within the same step budget (Run 4).
3. **`lr=3e-4` is the sweet spot** - smooth, stable convergence (Run 1).
4. **More training helps, not hurts, at `lr=3e-4`** - extending from 3 to 4 epochs (Run 5) further reduced loss with no sign of overfitting, suggesting the model still has capacity/data headroom to train longer.
5. **Recommended config going forward:** `learning_rate=3e-4`, `batch_size=32`, and `epochs≥4`, based on Run 5 being the best result so far - worth trying an even longer run next.
