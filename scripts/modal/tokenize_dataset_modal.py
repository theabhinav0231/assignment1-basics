import os
import time
import modal
import numpy as np

app = modal.App("cs336-tokenize-dataset")
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
def run_tokenization_pipeline(
    ts_vocab_json: str,
    ts_merges_txt: str,
    owt_vocab_json: str,
    owt_merges_txt: str,
):
    import requests
    import gzip
    import shutil
    import concurrent.futures

    print("=" * 70)
    print("🚀 Starting Modal Cloud Parallel Tokenization Pipeline (16 vCPUs)")
    print("=" * 70)

    os.makedirs("/tmp/data", exist_ok=True)
    
    files = {
        "ts_valid": (
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
            "/tmp/data/ts_valid.txt"
        ),
        "ts_train": (
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
            "/tmp/data/ts_train.txt"
        ),
        "owt_valid": (
            "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
            "/tmp/data/owt_valid.txt"
        ),
    }

    headers = {"User-Agent": "Mozilla/5.0"}
    for name, (url, path) in files.items():
        if not os.path.exists(path):
            print(f"Downloading {name} from {url}...")
            r = requests.get(url, headers=headers, stream=True)
            r.raise_for_status()
            
            if url.endswith(".gz"):
                gz_path = path + ".gz"
                with open(gz_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                print(f"Decompressing {gz_path}...")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
            else:
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
            print(f"Downloaded {path} ({os.path.getsize(path)/(1024*1024):.2f} MB)")

    # Save passed vocabularies and merges to temp files
    ts_vocab_path = "/tmp/ts_vocab.json"
    ts_merges_path = "/tmp/ts_merges.txt"
    owt_vocab_path = "/tmp/owt_vocab.json"
    owt_merges_path = "/tmp/owt_merges.txt"

    with open(ts_vocab_path, "w", encoding="utf-8") as f:
        f.write(ts_vocab_json)
    with open(ts_merges_path, "w", encoding="utf-8") as f:
        f.write(ts_merges_txt)
    with open(owt_vocab_path, "w", encoding="utf-8") as f:
        f.write(owt_vocab_json)
    with open(owt_merges_path, "w", encoding="utf-8") as f:
        f.write(owt_merges_txt)

    def tokenize_file_parallel(input_path: str, vocab_path: str, merges_path: str, num_workers: int = 16) -> np.ndarray:
        file_size = os.path.getsize(input_path)
        chunk_size = file_size // num_workers
        boundaries = [0]
        
        with open(input_path, "rb") as f:
            for i in range(1, num_workers):
                f.seek(i * chunk_size)
                f.readline()
                boundaries.append(f.tell())
        boundaries.append(file_size)

        print(f"Tokenizing {input_path} with {num_workers} parallel workers...")
        t0 = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(tokenize_chunk_worker, input_path, boundaries[i], boundaries[i+1], vocab_path, merges_path)
                for i in range(num_workers)
            ]
            all_tokens = []
            for fut in futures:
                all_tokens.extend(fut.result())
                
        elapsed = time.time() - t0
        arr = np.array(all_tokens, dtype=np.uint16)
        print(f"Finished {input_path}: {len(arr):,} tokens in {elapsed:.2f}s ({file_size/(1024*1024*elapsed):.2f} MB/s)")
        return arr

    # Tokenize and save directly to persistent Modal Volume
    ts_valid_arr = tokenize_file_parallel("/tmp/data/ts_valid.txt", ts_vocab_path, ts_merges_path, num_workers=8)
    np.save("/vol/tinystories_valid.npy", ts_valid_arr)
    print(f"Saved /vol/tinystories_valid.npy ({len(ts_valid_arr):,} tokens, {os.path.getsize('/vol/tinystories_valid.npy')/(1024*1024):.2f} MB)")

    ts_train_arr = tokenize_file_parallel("/tmp/data/ts_train.txt", ts_vocab_path, ts_merges_path, num_workers=16)
    np.save("/vol/tinystories_train.npy", ts_train_arr)
    print(f"Saved /vol/tinystories_train.npy ({len(ts_train_arr):,} tokens, {os.path.getsize('/vol/tinystories_train.npy')/(1024*1024):.2f} MB)")

    owt_valid_arr = tokenize_file_parallel("/tmp/data/owt_valid.txt", owt_vocab_path, owt_merges_path, num_workers=16)
    np.save("/vol/owt_valid.npy", owt_valid_arr)
    print(f"Saved /vol/owt_valid.npy ({len(owt_valid_arr):,} tokens, {os.path.getsize('/vol/owt_valid.npy')/(1024*1024):.2f} MB)")

    vol.commit()
    print("Modal Volume committed successfully!")
    return {
        "tinystories_valid_tokens": len(ts_valid_arr),
        "tinystories_train_tokens": len(ts_train_arr),
        "owt_valid_tokens": len(owt_valid_arr),
    }

@app.local_entrypoint()
def main():
    os.makedirs("data/encoded", exist_ok=True)
    
    with open("data/tinystories_vocab.json", "r", encoding="utf-8") as f:
        ts_vocab_json = f.read()
    with open("data/tinystories_merges.txt", "r", encoding="utf-8") as f:
        ts_merges_txt = f.read()
        
    with open("data/owt_vocab.json", "r", encoding="utf-8") as f:
        owt_vocab_json = f.read()
    with open("data/owt_merges.txt", "r", encoding="utf-8") as f:
        owt_merges_txt = f.read()

    print("Launching parallel tokenization on Modal Cloud with Modal Volume...")
    summary = run_tokenization_pipeline.remote(
        ts_vocab_json,
        ts_merges_txt,
        owt_vocab_json,
        owt_merges_txt,
    )
    print("\nTokenization Pipeline Succeeded!")
    for k, v in summary.items():
        print(f"  {k}: {v:,} tokens")
