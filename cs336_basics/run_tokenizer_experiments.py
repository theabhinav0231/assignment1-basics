import os
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def extract_documents(filepath: str, num_docs: int = 10) -> list[str]:
    """
    Extract the first `num_docs` documents from a text file delimited by <|endoftext|>.
    """
    docs = []
    current_doc = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "<|endoftext|>" in line:
                parts = line.split("<|endoftext|>")
                for i, part in enumerate(parts):
                    if current_doc or part:
                        current_doc.append(part)
                    if i < len(parts) - 1:
                        doc_text = "".join(current_doc).strip()
                        if doc_text:
                            docs.append(doc_text)
                            if len(docs) == num_docs:
                                return docs
                        current_doc = []
            else:
                current_doc.append(line)
                
    if current_doc and len(docs) < num_docs:
        doc_text = "".join(current_doc).strip()
        if doc_text:
            docs.append(doc_text)
    return docs

def compute_compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> tuple[int, int, float]:
    """
    Compute total UTF-8 bytes, total token IDs, and compression ratio (bytes / token).
    """
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        doc_bytes = len(doc.encode("utf-8"))
        token_ids = tokenizer.encode(doc)
        total_bytes += doc_bytes
        total_tokens += len(token_ids)
    ratio = total_bytes / total_tokens if total_tokens > 0 else 0.0
    return total_bytes, total_tokens, ratio

def run_experiments():
    print("=" * 70)
    print("CS336 Section 2.7: Tokenizer Experiments")
    print("=" * 70)
    
    # 1. Load trained tokenizers
    print("\n[1] Loading trained tokenizers...")
    ts_tok = Tokenizer.from_files(
        "data/tinystories_vocab.json",
        "data/tinystories_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    print(f"Loaded TinyStories Tokenizer (Vocab: {len(ts_tok.vocab)}, Merges: {len(ts_tok.merges)})")
    
    owt_tok = Tokenizer.from_files(
        "data/owt_vocab.json",
        "data/owt_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    print(f"Loaded OpenWebText Tokenizer (Vocab: {len(owt_tok.vocab)}, Merges: {len(owt_tok.merges)})")
    
    # 2. Extract 10 documents
    print("\n[2] Sampling 10 documents from TinyStories and OpenWebText...")
    ts_docs = extract_documents("data/TinyStoriesV2-GPT4-valid.txt", num_docs=10)
    owt_docs = extract_documents("data/owt_valid.txt", num_docs=10)
    print(f"Sampled {len(ts_docs)} TinyStories documents (Total chars: {sum(len(d) for d in ts_docs)})")
    print(f"Sampled {len(owt_docs)} OpenWebText documents (Total chars: {sum(len(d) for d in owt_docs)})")
    
    # 3. Problem 2.7 (a): In-domain Compression Ratios
    print("\n" + "-" * 70)
    print("Problem 2.7 (a): In-Domain Compression Ratios")
    print("-" * 70)
    ts_bytes, ts_tokens, ts_ratio = compute_compression_ratio(ts_tok, ts_docs)
    print(f"TinyStories Tokenizer on TinyStories Docs:")
    print(f"  Total UTF-8 Bytes: {ts_bytes:,} bytes")
    print(f"  Total Tokens:      {ts_tokens:,} tokens")
    print(f"  Compression Ratio: {ts_ratio:.4f} bytes/token")
    
    owt_bytes, owt_tokens, owt_ratio = compute_compression_ratio(owt_tok, owt_docs)
    print(f"\nOpenWebText Tokenizer on OpenWebText Docs:")
    print(f"  Total UTF-8 Bytes: {owt_bytes:,} bytes")
    print(f"  Total Tokens:      {owt_tokens:,} tokens")
    print(f"  Compression Ratio: {owt_ratio:.4f} bytes/token")
    
    # 4. Problem 2.7 (b): Cross-Domain Evaluation (OWT docs on TinyStories Tokenizer)
    print("\n" + "-" * 70)
    print("Problem 2.7 (b): Cross-Domain Evaluation (OpenWebText with TinyStories Tokenizer)")
    print("-" * 70)
    cross_bytes, cross_tokens, cross_ratio = compute_compression_ratio(ts_tok, owt_docs)
    print(f"TinyStories Tokenizer on OpenWebText Docs:")
    print(f"  Total UTF-8 Bytes: {cross_bytes:,} bytes")
    print(f"  Total Tokens:      {cross_tokens:,} tokens")
    print(f"  Compression Ratio: {cross_ratio:.4f} bytes/token")
    print(f"  Ratio Difference:  OWT Tokenizer achieves {owt_ratio:.4f} vs TinyStories {cross_ratio:.4f} bytes/token")
    print(f"  Token Overhead:    TinyStories tokenizer produces {cross_tokens - owt_tokens:,} (+{(cross_tokens - owt_tokens)/owt_tokens*100:.1f}%) more tokens on OWT.")

    # 5. Problem 2.7 (c): Throughput & The Pile Estimation
    print("\n" + "-" * 70)
    print("Problem 2.7 (c): Tokenizer Throughput & Pile Dataset Estimation")
    print("-" * 70)
    # Benchmark on 1 MB of text
    bench_text = "\n".join(owt_docs * 10)
    bench_bytes = len(bench_text.encode("utf-8"))
    
    # Warmup
    _ = owt_tok.encode(bench_text[:1000])
    
    t0 = time.perf_counter()
    _ = owt_tok.encode(bench_text)
    elapsed = time.perf_counter() - t0
    
    throughput_bytes_sec = bench_bytes / elapsed
    throughput_kb_sec = throughput_bytes_sec / 1024
    throughput_mb_sec = throughput_bytes_sec / (1024 * 1024)
    
    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * (1024 ** 3)
    pile_time_sec = pile_size_bytes / throughput_bytes_sec
    pile_time_hours = pile_time_sec / 3600
    pile_time_days = pile_time_hours / 24
    
    print(f"Benchmarked text size: {bench_bytes:,} bytes ({bench_bytes/1024:.1f} KB)")
    print(f"Encoding time:         {elapsed:.4f} seconds")
    print(f"Single-thread speed:   {throughput_mb_sec:.2f} MB/s ({throughput_bytes_sec:,.0f} bytes/s)")
    print(f"\nEstimated time to tokenize The Pile (825 GB):")
    print(f"  Single Process:      {pile_time_sec:,.1f} s (~{pile_time_hours:.2f} hours / {pile_time_days:.2f} days)")
    print(f"  16 Parallel Workers: {pile_time_hours / 16:.2f} hours")

    # 6. Problem 2.7 (d): Explain uint16
    print("\n" + "-" * 70)
    print("Problem 2.7 (d): uint16 Datatype Justification")
    print("-" * 70)
    print("Vocabulary sizes: TinyStories = 10,000, OpenWebText = 32,000.")
    print("Max uint16 capacity: 2^16 - 1 = 65,535.")
    print("Since both 10,000 <= 65,535 and 32,000 <= 65,535, uint16 fits all token IDs perfectly")
    print("while saving 50% memory compared to int32 (2 bytes/token vs 4 bytes/token).")
    print("=" * 70)

if __name__ == "__main__":
    run_experiments()
