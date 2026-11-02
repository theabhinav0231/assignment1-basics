import os
import json
import time
import tiktoken
from cs336_basics.tokenizer import Tokenizer
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

def run_unicode_check():
    t0 = time.time()
    print("Loading tokenizer from fixture files...")
    tok = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    print(f"Loaded tokenizer in {time.time() - t0:.2f}s (vocab: {len(tok.vocab)}, merges: {len(tok.merges)})")
    
    t0 = time.time()
    ref = tiktoken.get_encoding("gpt2")
    print(f"Loaded tiktoken in {time.time() - t0:.2f}s")
    
    test_str = "🙃"
    t0 = time.time()
    ref_ids = ref.encode(test_str)
    print(f"tiktoken encoded in {time.time() - t0:.4f}s -> {ref_ids}")
    
    t0 = time.time()
    ids = tok.encode(test_str)
    print(f"our tokenizer encoded in {time.time() - t0:.4f}s -> {ids}")
    
    print("IDs match:", ids == ref_ids)
    print("Decoded matches:", tok.decode(ids) == test_str)

if __name__ == "__main__":
    run_unicode_check()
