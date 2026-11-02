import json
import os
import time
import psutil
from cs336_basics.train_bpe import train_bpe


def train_and_save(dataset_path: str, vocab_size: int, output_prefix: str):
    print(f"Starting BPE training on: {dataset_path}")
    print(f"Target vocabulary size: {vocab_size}")

    start_time = time.time()
    process = psutil.Process(os.getpid())

    vocab, merges = train_bpe(
        input_path=dataset_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    elapsed_time = time.time() - start_time
    mem_usage_gb = process.memory_info().rss / (1024**3)

    print("-" * 50)
    print(f"Training completed in: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"Peak memory usage: {mem_usage_gb:.2f} GB")

    # Longest token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token in bytes: {longest_token}")
    print(f"Longest token (decoded): '{longest_token.decode('utf-8', errors='replace')}'")
    print(f"Longest token length: {len(longest_token)} bytes")
    print("-" * 50)

    # 1. Save merges to text file
    merges_file = f"{output_prefix}_merges.txt"
    with open(merges_file, "w", encoding="utf-8") as f:
        for pair in merges:
            f.write(f"{pair[0].decode('latin-1')} {pair[1].decode('latin-1')}\n")
    print(f"Saved merges to: {merges_file}")

    # 2. Save vocabulary to JSON
    vocab_file = f"{output_prefix}_vocab.json"
    serialized_vocab = {token_id: token_bytes.decode("latin-1") for token_id, token_bytes in vocab.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(serialized_vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to: {vocab_file}")


if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    dataset_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    output_prefix = "data/tinystories"

    train_and_save(dataset_path, vocab_size, output_prefix)