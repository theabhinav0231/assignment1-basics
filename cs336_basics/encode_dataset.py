import os
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def encode_and_save(
    input_text_path: str,
    output_bin_path: str,
    tokenizer: Tokenizer,
    chunk_size_chars: int = 100_000,
):
    print(f"\n[Encoding] {input_text_path} -> {output_bin_path}")
    t0 = time.time()
    
    # We will stream chunks of text and write directly to an array / binary file
    all_token_ids = []
    total_bytes = 0
    
    with open(input_text_path, "r", encoding="utf-8", errors="replace") as f:
        buffer = []
        buf_len = 0
        for line in f:
            buffer.append(line)
            buf_len += len(line)
            if buf_len >= chunk_size_chars:
                text_chunk = "".join(buffer)
                ids = tokenizer.encode(text_chunk)
                all_token_ids.extend(ids)
                total_bytes += len(text_chunk.encode("utf-8"))
                buffer = []
                buf_len = 0
                
        if buffer:
            text_chunk = "".join(buffer)
            ids = tokenizer.encode(text_chunk)
            all_token_ids.extend(ids)
            total_bytes += len(text_chunk.encode("utf-8"))

    # Convert to uint16 NumPy array
    tokens_arr = np.array(all_token_ids, dtype=np.uint16)
    np.save(output_bin_path, tokens_arr)
    
    elapsed = time.time() - t0
    compression = total_bytes / len(tokens_arr) if len(tokens_arr) > 0 else 0
    
    print(f"Encoded {len(tokens_arr):,} tokens in {elapsed:.2f}s ({total_bytes / (1024*1024*elapsed):.2f} MB/s)")
    print(f"Compression ratio: {compression:.4f} bytes/token")
    print(f"Saved binary array to: {output_bin_path}.npy (Size: {os.path.getsize(output_bin_path + '.npy') / (1024*1024):.2f} MB)")
    return tokens_arr

def main():
    os.makedirs("data/encoded", exist_ok=True)
    
    # 1. Load TinyStories Tokenizer
    ts_tok = Tokenizer.from_files(
        "data/tinystories_vocab.json",
        "data/tinystories_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    
    # Encode TinyStories validation set
    encode_and_save(
        "data/TinyStoriesV2-GPT4-valid.txt",
        "data/encoded/tinystories_valid",
        ts_tok
    )

if __name__ == "__main__":
    main()
