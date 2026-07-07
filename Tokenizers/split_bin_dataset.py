import numpy as np
import os


# ==========================
# Configuration
# ==========================

input_bin = "./tokenized_dataset/train.bin"

train_output = "./train_split.bin"
val_output = "./validation.bin"

vocab_size = 16000

val_ratio = 0.05   # 5% validation


# ==========================
# Split Script
# ==========================

def main():

    dtype = np.uint16 if vocab_size < 65536 else np.uint32

    print(f"Reading {input_bin} as {dtype}")

    # Memory map (does not load everything into RAM)
    tokens = np.memmap(
        input_bin,
        dtype=dtype,
        mode="r"
    )

    total_tokens = len(tokens)

    print(f"Total tokens: {total_tokens:,}")


    # Calculate split point
    val_tokens = int(total_tokens * val_ratio)

    train_tokens = total_tokens - val_tokens


    print(f"Train tokens: {train_tokens:,}")
    print(f"Validation tokens: {val_tokens:,}")


    # Create validation file
    print("Writing validation.bin...")

    with open(val_output, "wb") as f:
        tokens[train_tokens:].tofile(f)


    # Create training file
    print("Writing train.bin...")

    with open(train_output, "wb") as f:
        tokens[:train_tokens].tofile(f)


    print("Done!")
    print()
    print("Created:")
    print(train_output)
    print(val_output)


if __name__ == "__main__":
    main()