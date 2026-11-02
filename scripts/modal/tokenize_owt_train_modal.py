import os
import time
import modal
import numpy as np

app = modal.App("cs336-tokenize-owt-train")
vol = modal.Volume.from_name("cs336-dataset-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "regex", "requests")
    .add_local_python_source("cs336_basics")
)

def tokenize_chunk_worker(input_path: str, start: int, end: int, vocab_path: str, merges_path: str) -> list[int]:
    from cs336_basics.tokenizer import Tokenizer
    tok = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
    tokens = []
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            tokens.extend(tok.encode(line))
    return tokens

@app.function(
    image=image,
    volumes={"/vol": vol},
    cpu=16.0,
    memory=32768,  # 32 GB RAM
    timeout=3600,
)
def run_owt_train_tokenization(
    owt_vocab_json: str,
    owt_merges_txt: str,
):
    import requests
    import gzip
    import shutil
    import concurrent.futures

    print("=" * 70)
    print("🚀 Starting Modal Cloud Parallel Tokenization for OpenWebText Train")
    print("=" * 70)

    os.makedirs("/tmp/data", exist_ok=True)
    owt_train_gz = "/tmp/data/owt_train.txt.gz"
    owt_train_path = "/tmp/data/owt_train.txt"
    
    url = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    if not os.path.exists(owt_train_path):
        print(f"Downloading owt_train from {url}...")
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        with open(owt_train_gz, "wb") as f:
            for chunk in r.iter_content(chunk_size=10*1024*1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {owt_train_gz} ({os.path.getsize(owt_train_gz)/(1024*1024):.2f} MB)")
        
        print(f"Decompressing {owt_train_gz}...")
        with gzip.open(owt_train_gz, 'rb') as f_in:
            with open(owt_train_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out, length=10*1024*1024)
        os.remove(owt_train_gz)
        print(f"Decompressed to {owt_train_path} ({os.path.getsize(owt_train_path)/(1024*1024):.2f} MB)")

    # Save vocab & merges to temp files
    owt_vocab_path = "/tmp/owt_vocab.json"
    owt_merges_path = "/tmp/owt_merges.txt"
    with open(owt_vocab_path, "w", encoding="utf-8") as f:
        f.write(owt_vocab_json)
    with open(owt_merges_path, "w", encoding="utf-8") as f:
        f.write(owt_merges_txt)

    num_workers = 16
    file_size = os.path.getsize(owt_train_path)
    chunk_size = file_size // num_workers
    boundaries = [0]
    
    with open(owt_train_path, "rb") as f:
        for i in range(1, num_workers):
            f.seek(i * chunk_size)
            f.readline()
            boundaries.append(f.tell())
    boundaries.append(file_size)

    print(f"Tokenizing {owt_train_path} with {num_workers} parallel workers...")
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(tokenize_chunk_worker, owt_train_path, boundaries[i], boundaries[i+1], owt_vocab_path, owt_merges_path)
            for i in range(num_workers)
        ]
        all_tokens = []
        for i, fut in enumerate(futures):
            chunk_tokens = fut.result()
            print(f"Worker {i+1}/{num_workers} finished: {len(chunk_tokens):,} tokens")
            all_tokens.extend(chunk_tokens)
            
    elapsed = time.time() - t0
    arr = np.array(all_tokens, dtype=np.uint16)
    print(f"Finished tokenizing OWT Train: {len(arr):,} tokens in {elapsed:.2f}s ({file_size/(1024*1024*elapsed):.2f} MB/s)")

    # Save directly into Modal Volume
    out_vol_path = "/vol/owt_train.npy"
    np.save(out_vol_path, arr)
    vol.commit()
    print(f"Saved {out_vol_path} ({len(arr):,} tokens, {os.path.getsize(out_vol_path)/(1024*1024):.2f} MB)")
    print("Modal Volume committed successfully!")
    return len(arr)

@app.local_entrypoint()
def main():
    with open("data/owt_vocab.json", "r", encoding="utf-8") as f:
        owt_vocab_json = f.read()
    with open("data/owt_merges.txt", "r", encoding="utf-8") as f:
        owt_merges_txt = f.read()

    print("Launching parallel OWT Train tokenization on Modal Cloud...")
    num_tokens = run_owt_train_tokenization.remote(
        owt_vocab_json,
        owt_merges_txt,
    )
    print(f"\nOWT Train Tokenization Succeeded! Total tokens: {num_tokens:,}")
