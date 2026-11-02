import os
import sys
import json

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import modal

app = modal.App("cs336-owt-45min-pretrain")
vol = modal.Volume.from_name("cs336-dataset-vol", create_if_missing=True)

GPU_TYPE = "H100"

WANDB_KEY = os.environ.get(
    "WANDB_API_KEY",
    "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"
)
wandb_secret = modal.Secret.from_dict({"WANDB_API_KEY": WANDB_KEY})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "psutil", "regex", "einops", "einx", "jaxtyping", "wandb")
    .add_local_python_source("cs336_basics")
    .add_local_file("train.py", "/root/train.py")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": vol},
    secrets=[wandb_secret],
    timeout=2700,  # 45 minutes limit
)
def run_45min_pretraining(config: dict):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import sys
    sys.path.insert(0, "/root")
    import argparse
    import torch
    from train import train

    lr = config.get("lr", 2.5e-3)
    max_iters = config.get("max_iters", 10000)
    warmup_iters = config.get("warmup_iters", 500)
    batch_size = config.get("batch_size", 256)
    grad_accum_steps = config.get("grad_accum_steps", 1)
    d_model = config.get("d_model", 512)
    num_layers = config.get("num_layers", 4)
    num_heads = config.get("num_heads", 16)
    d_ff = config.get("d_ff", 1344)
    ffn_type = config.get("ffn_type", "swiglu")
    vocab_size = config.get("vocab_size", 32000)
    tie_weights = config.get("tie_weights", True)
    use_bfloat16 = config.get("use_bfloat16", True)
    run_name = config.get("run_name", f"owt-h100-tied-bf16-b{batch_size}-10k-lr-{lr:.1e}")
    checkpoint_dir = f"/vol/checkpoints/owt_45min_pretrain/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    args = argparse.Namespace(
        train_data="/vol/owt_train.npy",
        val_data="/vol/owt_valid.npy",
        vocab_size=vocab_size,
        context_length=256,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        ffn_type=ffn_type,
        rope_theta=10000.0,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        min_lr=lr * 0.1,
        warmup_iters=warmup_iters,
        max_iters=max_iters,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip=1.0,
        device="cuda",
        seed=1337,
        log_interval=50,
        eval_interval=250,
        eval_iters=20,
        save_interval=1000,
        checkpoint_dir=checkpoint_dir,
        resume=None,
        wandb_project="cs336-assignment1-basics",
        wandb_run_name=run_name,
        use_rmsnorm=True,
        norm_position="pre",
        use_rope=True,
        tie_weights=tie_weights,
        use_bfloat16=use_bfloat16,
        max_time_seconds=2640.0,  # 44.0 minute cutoff (1 minute before Modal hard timeout)
    )

    print("=" * 75)
    print(f"--> [LAUNCH] OpenWebText 45-Minute Pretraining: {run_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Vocab: {vocab_size:,} | Architecture: Pre-Norm RMSNorm RoPE SwiGLU")
    print(f"Dimensions: d_model={d_model}, L={num_layers}, H={num_heads}, d_ff={d_ff}")
    print(f"Parameters: Tied Embeddings & LM Head (28,840,448 unique parameters)")
    print(f"Batch Size: {batch_size} (65,536 tok/step, single-pass) | Target Steps: {max_iters:,}")
    print(f"Total Token Horizon: {max_iters * batch_size * 256:,} tokens (~655.4M tokens)")
    print(f"Optimization: AdamW (lr={lr:.4f}, min_lr={args.min_lr:.5f}, warmup={warmup_iters} steps, cosine)")
    print(f"Precision: bfloat16 | Wall-Clock Cutoff: {args.max_time_seconds/60:.1f} minutes")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print("=" * 75)

    try:
        metrics = train(args)
        vol.commit()
        print(f"[SUCCESS] 45-Minute run {run_name} finished successfully and committed to volume.")
        return {
            "run_name": run_name,
            "lr": lr,
            "batch_size": batch_size,
            "tie_weights": tie_weights,
            "use_bfloat16": use_bfloat16,
            "status": "COMPLETED",
            "metrics": metrics,
        }
    except Exception as e:
        print(f"[ERROR] Run {run_name} encountered error: {e}")
        return {
            "run_name": run_name,
            "lr": lr,
            "status": f"FAILED: {e}",
            "metrics": None,
        }


@app.local_entrypoint()
def main():
    cfg = {
        "lr": 0.0025,
        "run_name": "owt-h100-tied-bf16-b256-10k-lr-2.5e-3",
        "batch_size": 256,
        "grad_accum_steps": 1,
        "max_iters": 10000,
        "warmup_iters": 500,
        "tie_weights": True,
        "use_bfloat16": True,
    }

    print("=" * 75)
    print(f"Launching Full 45-Minute OpenWebText Pretraining on Modal [{GPU_TYPE}]")
    print(f"Project: cs336-assignment1-basics | Run: {cfg['run_name']}")
    print(f"Dataset: OpenWebText (5.24B tokens) | Vocab: 32,000 | Batch: 256 (65,536 tok/step)")
    print(f"Steps: {cfg['max_iters']:,} (~655.4M tokens) | Warmup: {cfg['warmup_iters']} steps")
    print(f"Features: tie_weights=True, bfloat16=True, grad_accum_steps=1 (single pass)")
    print("=" * 75)

    result = run_45min_pretraining.remote(cfg)

    print("\n" + "=" * 75)
    print("Pretraining run finished. Summary:")
    print("=" * 75)
    metrics = result.get("metrics")
    if metrics:
        best_val = metrics.get("best_val_loss")
        final_val = metrics.get("final_val_loss")
        final_ppl = metrics.get("final_val_ppl")
        final_train = metrics.get("final_train_loss")
        tps = metrics.get("tokens_per_second", 0)
        elapsed = metrics.get("total_time_seconds", 0)
        print(f"Status: {result.get('status')}")
        print(f"Best Val Loss: {best_val:.4f} | Final Val Loss: {final_val:.4f} (PPL: {final_ppl:.2f})")
        print(f"Final Train Loss: {final_train:.4f} | Average Throughput: {tps:,.1f} tok/s")
        print(f"Total Wall-Clock Time: {elapsed/60:.2f} minutes")

    os.makedirs("plots", exist_ok=True)
    out_path = "plots/owt_45min_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved complete run metrics to {out_path}")
