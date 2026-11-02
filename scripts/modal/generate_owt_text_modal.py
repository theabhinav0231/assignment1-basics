import os
import sys
import json

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import modal

app = modal.App("cs336-owt-generate")
vol = modal.Volume.from_name("cs336-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "regex", "einops", "jaxtyping")
    .add_local_python_source("cs336_basics")
    .add_local_file("data/owt_vocab.json", "/root/owt_vocab.json")
    .add_local_file("data/owt_merges.txt", "/root/owt_merges.txt")
)

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=600,
)
def generate_samples(prompts: list[str], max_new_tokens: int = 150, temperature: float = 0.8, top_p: float = 0.9):
    import torch
    import json
    from cs336_basics.model import TransformerLM
    from cs336_basics.tokenizer import Tokenizer
    from cs336_basics.decode import decode

    # 1. Initialize Tokenizer
    tok = Tokenizer.from_files("/root/owt_vocab.json", "/root/owt_merges.txt", ["<|endoftext|>"])

    # 2. Initialize Model Architecture matching OWT run
    model = TransformerLM(
        vocab_size=32000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        ffn_type="swiglu",
        rope_theta=10000.0,
        use_rmsnorm=True,
        norm_position="pre",
        use_rope=True,
        device=torch.device("cuda"),
    ).cuda()

    # 3. Load Checkpoint
    ckpt_path = "/vol/checkpoints/owt_main_experiment/owt-h100-swiglu-lr-2.5e-3/checkpoint_best.pt"
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    results = []
    print("\n" + "=" * 70)
    print("Generating text with trained OpenWebText Language Model:")
    print("=" * 70)

    with torch.no_grad():
        for prompt in prompts:
            input_ids = tok.encode(prompt)
            output_tokens = decode(
                model=model,
                prompt_tokens=input_ids,
                max_new_tokens=max_new_tokens,
                context_length=256,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tok.encode("<|endoftext|>")[0] if "<|endoftext|>" in tok.vocab else None,
                device="cuda",
            )
            text = tok.decode(output_tokens)
            results.append({
                "prompt": prompt,
                "generated_text": text,
            })
            print(f"\n--- PROMPT: {prompt} ---")
            print(text)
            print("-" * 50)

    return results

@app.local_entrypoint()
def main():
    prompts = [
        "Once upon a time",
        "The president of the United States announced that",
        "In computer science, an algorithm is",
    ]
    results = generate_samples.remote(prompts=prompts, max_new_tokens=120, temperature=0.8, top_p=0.9)
    
    # Save local json
    out_path = r"assignment1-basics/plots/owt_generated_samples.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved generated samples to {out_path}")
