import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import modal

app = modal.App("cs336-text-generation")
vol = modal.Volume.from_name("cs336-dataset-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "psutil", "regex", "einops", "einx", "jaxtyping")
    .add_local_python_source("cs336_basics")
    .add_local_file("data/tinystories_vocab.json", "/root/data/tinystories_vocab.json")
    .add_local_file("data/tinystories_merges.txt", "/root/data/tinystories_merges.txt")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=600,
)
def generate_stories():
    import torch
    from cs336_basics.model import TransformerLM
    from cs336_basics.tokenizer import Tokenizer
    from cs336_basics.decode import decode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running generation on device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Initialize Tokenizer
    vocab_path = "/root/data/tinystories_vocab.json"
    merges_path = "/root/data/tinystories_merges.txt"
    print(f"Loading tokenizer from {vocab_path}...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"],
    )
    eos_id = tokenizer.token_to_id.get(b"<|endoftext|>", None)
    print(f"Tokenizer loaded (vocab size: {len(tokenizer.vocab)}, eos_token_id: {eos_id})")

    # 2. Initialize Model
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
        device=torch.device(device),
    ).to(device)

    # 3. Load Checkpoint
    checkpoint_path = "/vol/checkpoints/tinystories_baseline/checkpoint_best.pt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    step = checkpoint.get("iteration", "unknown")
    print(f"Successfully loaded checkpoint from step {step}!")
    model.eval()

    # 4. Prompts and Decoding Configurations
    test_cases = [
        {
            "name": "Story 1: Lily and the Kitten (Balanced Sampling: temp=0.8, top_p=0.9)",
            "prompt": "Once upon a time, there was a little girl named Lily.",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 256,
        },
        {
            "name": "Story 2: The Magic Box (Focused Sampling: temp=0.6, top_p=0.85)",
            "prompt": "One sunny day, a little dog named Toby found a magic box in the garden.",
            "temperature": 0.6,
            "top_p": 0.85,
            "max_tokens": 256,
        },
        {
            "name": "Story 3: Building a Castle (Creative Sampling: temp=1.0, top_p=0.95)",
            "prompt": "Tom and Mia wanted to build a big castle with wooden blocks.",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 256,
        },
        {
            "name": "Story 4: Unconditional Generation (from scratch: temp=0.75, top_p=0.9)",
            "prompt": "",
            "temperature": 0.75,
            "top_p": 0.9,
            "max_tokens": 256,
        },
    ]

    outputs = []
    print("\n" + "=" * 75)
    print("GENERATING SAMPLES FROM TRAINED TINYSTORIES MODEL")
    print("=" * 75)

    for tc in test_cases:
        prompt_text = tc["prompt"]
        prompt_ids = tokenizer.encode(prompt_text) if prompt_text else [tokenizer.encode("Once")[0]]
        
        gen_token_ids = decode(
            model=model,
            prompt_tokens=prompt_ids,
            max_new_tokens=tc["max_tokens"],
            temperature=tc["temperature"],
            top_p=tc["top_p"],
            eos_token_id=eos_id,
            context_length=256,
            device=device,
        )

        full_text = tokenizer.decode(gen_token_ids)
        num_generated = len(gen_token_ids) - len(prompt_ids)
        
        res = {
            "name": tc["name"],
            "prompt": prompt_text,
            "temperature": tc["temperature"],
            "top_p": tc["top_p"],
            "total_tokens": len(gen_token_ids),
            "new_tokens": num_generated,
            "generated_text": full_text,
        }
        outputs.append(res)

        print(f"\n--- {tc['name']} ---")
        print(f"Prompt: \"{prompt_text}\"")
        print(f"Params: temp={tc['temperature']}, top_p={tc['top_p']} | Tokens Generated: {num_generated} (Total: {len(gen_token_ids)})")
        print("-" * 50)
        print(full_text)
        print("-" * 50)

    return outputs


@app.local_entrypoint()
def main():
    results = generate_stories.remote()
    print("\n" + "=" * 75)
    print(f"Successfully generated {len(results)} sample stories!")
    print("=" * 75)
