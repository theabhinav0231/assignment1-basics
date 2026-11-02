import os
import time
import modal

app = modal.App("cs336-owt-bpe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("regex", "psutil", "requests")
    .add_local_python_source("cs336_basics")
)


@app.function(
    image=image,
    cpu=16,               # 16 High-Performance vCPUs
    memory=64 * 1024,     # 64 GB RAM (Runs 100% in pure RAM without disk paging)
    timeout=3600,
)
def run_owt_bpe_cloud() -> tuple[str, str, float, float, bytes, str, int]:
    import gzip
    import json
    import os
    import psutil
    import requests
    from cs336_basics.train_bpe import train_bpe

    data_dir = "/tmp/data"
    os.makedirs(data_dir, exist_ok=True)
    gz_path = os.path.join(data_dir, "owt_train.txt.gz")
    txt_path = os.path.join(data_dir, "owt_train.txt")

    # 1. Download dataset at datacenter speeds (~10 Gbps)
    if not os.path.exists(txt_path):
        print("Downloading owt_train.txt.gz from Hugging Face at high speed...")
        url = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=32 * 1024 * 1024):
                    f.write(chunk)
        print(f"Downloaded {os.path.getsize(gz_path) / (1024**3):.2f} GB compressed archive.")

        print("Decompressing owt_train.txt...")
        with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
            while chunk := f_in.read(32 * 1024 * 1024):
                f_out.write(chunk)
        os.remove(gz_path)
        print(f"Decompressed {os.path.getsize(txt_path) / (1024**3):.2f} GB dataset.")

    # 2. Train BPE on 64 GB RAM with 16 workers
    print("Starting BPE training on OpenWebText (Vocab size: 32,000)...")
    start_time = time.time()
    process = psutil.Process(os.getpid())

    vocab, merges = train_bpe(
        input_path=txt_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        num_processes=16,
    )

    elapsed_time = time.time() - start_time
    mem_usage_gb = process.memory_info().rss / (1024**3)

    longest_token = max(vocab.values(), key=len)
    longest_decoded = longest_token.decode("utf-8", errors="replace")

    # Serialize merges
    merges_lines = [f"{p[0].decode('latin-1')} {p[1].decode('latin-1')}" for p in merges]
    merges_txt = "\n".join(merges_lines) + "\n"

    # Serialize vocab
    serialized_vocab = {token_id: token_bytes.decode("latin-1") for token_id, token_bytes in vocab.items()}
    vocab_json = json.dumps(serialized_vocab, ensure_ascii=False, indent=2)

    return (
        vocab_json,
        merges_txt,
        elapsed_time,
        mem_usage_gb,
        longest_token,
        longest_decoded,
        len(longest_token),
    )


@app.local_entrypoint()
def main():
    os.makedirs("data", exist_ok=True)
    print("🚀 Launching OpenWebText BPE training on Modal Cloud (16 vCPUs, 64 GB RAM)...")

    (
        vocab_json,
        merges_txt,
        elapsed_time,
        mem_usage_gb,
        longest_token,
        longest_decoded,
        longest_len,
    ) = run_owt_bpe_cloud.remote()

    # Save to local data directory
    vocab_path = "data/owt_vocab.json"
    merges_path = "data/owt_merges.txt"

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(vocab_json)

    with open(merges_path, "w", encoding="utf-8") as f:
        f.write(merges_txt)

    print("\n" + "=" * 60)
    print("✅ Modal BPE Training Completed Successfully!")
    print(f"Training Time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"Peak Memory Usage: {mem_usage_gb:.2f} GB")
    print(f"Longest Token in Bytes: {longest_token}")
    print(f"Longest Token (Decoded): '{longest_decoded}'")
    print(f"Longest Token Length: {longest_len} bytes")
    print("=" * 60)
    print(f"Saved vocabulary to: {vocab_path}")
    print(f"Saved merges to: {merges_path}")
