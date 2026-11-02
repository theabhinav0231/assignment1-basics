import os
import json
from cs336_basics.tokenizer import Tokenizer

def main():
    print("=" * 60)
    print("Testing Trained Tokenizers: TinyStories vs OpenWebText")
    print("=" * 60)
    
    # TinyStories
    ts_tok = Tokenizer.from_files("data/tinystories_vocab.json", "data/tinystories_merges.txt", ["<|endoftext|>"])
    ts_text = "Once upon a time, a little girl had an accomplishment. <|endoftext|>"
    ts_ids = ts_tok.encode(ts_text)
    ts_dec = ts_tok.decode(ts_ids)
    print(f"TinyStories Vocab Size: {len(ts_tok.vocab)}")
    print(f"TinyStories Merges: {len(ts_tok.merges)}")
    print(f"TinyStories Encode ({len(ts_ids)} tokens): {ts_ids}")
    print(f"TinyStories Decode Match: {ts_dec == ts_text}")
    print()

    # OpenWebText
    owt_tok = Tokenizer.from_files("data/owt_vocab.json", "data/owt_merges.txt", ["<|endoftext|>"])
    owt_text = "Hello from Stanford CS336! Deep learning and transformer architecture from scratch. <|endoftext|>"
    owt_ids = owt_tok.encode(owt_text)
    owt_dec = owt_tok.decode(owt_ids)
    print(f"OpenWebText Vocab Size: {len(owt_tok.vocab)}")
    print(f"OpenWebText Merges: {len(owt_tok.merges)}")
    print(f"OpenWebText Encode ({len(owt_ids)} tokens): {owt_ids}")
    print(f"OpenWebText Decode Match: {owt_dec == owt_text}")
    print("=" * 60)

if __name__ == "__main__":
    main()
