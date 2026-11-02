import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import modal

app = modal.App("cs336-nope-ablation")
vol = modal.Volume.from_name("cs336-dataset-vol", create_if_missing=True)

GPU_TYPE = "A100-40GB"

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
    timeout=3600,
)
def run_nope_ablation(config: dict):
    import sys
    sys.path.insert(0, "/root")
    import argparse
    import torch
    from train import train

    lr = config["lr"]
    max_iters = config.get("max_iters", 2500)
    batch_size = config.get("batch_size", 128)
    use_rope = config.get("use_rope", False)
    run_name = config.get("run_name", f"{'rope' if use_rope else 'nope'}-lr-{lr:.1e}")
    checkpoint_dir = f"/vol/checkpoints/nope_ablation/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    args = argparse.Namespace(
        train_data="/vol/tinystories_train.npy",
        val_data="/vol/tinystories_valid.npy",
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
        batch_size=batch_size,
        lr=lr,
        min_lr=lr * 0.1,
        warmup_iters=250,
        max_iters=max_iters,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip=1.0,
        device="cuda",
        seed=1337,
        log_interval=20,
        eval_interval=100,
        eval_iters=20,
        save_interval=0,
        checkpoint_dir=checkpoint_dir,
        resume=None,
        wandb_project="cs336-assignment1-basics",
        wandb_run_name=run_name,
        use_rmsnorm=True,
        norm_position="pre",
        use_rope=use_rope,
    )

    print("=" * 70)
    print(f"--> [LAUNCH] NoPE Ablation: {run_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Positional Embeddings (RoPE): {use_rope} | LR: {lr:.2e} | Steps: {max_iters} | Batch: {batch_size}")
    print("=" * 70)

    try:
        train(args)
        print(f"[SUCCESS] Run {run_name} finished successfully.")
        return {"run_name": run_name, "use_rope": use_rope, "lr": lr, "status": "COMPLETED"}
    except Exception as e:
        print(f"[ERROR] Run {run_name} encountered error: {e}")
        return {"run_name": run_name, "use_rope": use_rope, "lr": lr, "status": f"FAILED: {e}"}


@app.local_entrypoint()
def main():
    configs = [
        # Baseline: RoPE enabled with optimal LR 3e-3
        {"lr": 3e-3, "use_rope": True, "max_iters": 2500, "run_name": "rope-baseline-lr-3e-3"},
        # Ablation: NoPE (No Positional Embeddings) with standard LR 3e-3
        {"lr": 3e-3, "use_rope": False, "max_iters": 2500, "run_name": "nope-ablation-lr-3e-3"},
    ]

    print("=" * 75)
    print(f"Launching {len(configs)} Parallel RoPE vs. NoPE Ablation Runs on Modal [{GPU_TYPE}]")
    print(f"Project: cs336-assignment1-basics")
    for cfg in configs:
        pos_str = "RoPE" if cfg["use_rope"] else "NoPE (None)"
        print(f"  * {cfg['run_name']}: PosEmb={pos_str}, LR={cfg['lr']:.1e}, Steps={cfg['max_iters']}")
    print("=" * 75)

    results = list(run_nope_ablation.map(configs))
    print("\n" + "=" * 75)
    print("ALL NO-PE ABLATION RUNS FINISHED:")
    for r in results:
        print(f"  * {r['run_name']}: {r['status']}")
    print("=" * 75)
