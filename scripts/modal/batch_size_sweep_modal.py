import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import modal

app = modal.App("cs336-batch-size-sweep")
vol = modal.Volume.from_name("cs336-dataset-vol", create_if_missing=True)

GPU_TYPE = "A100-40GB"

WANDB_KEY = os.environ.get(
    "WANDB_API_KEY",
    "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"
)
wandb_secret = modal.Secret.from_dict({"WANDB_API_KEY": WANDB_KEY})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "psutil", "einops", "einx", "jaxtyping", "tqdm", "wandb")
    .add_local_python_source("cs336_basics")
    .add_local_file("train.py", "/root/train.py")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": vol},
    secrets=[wandb_secret],
    timeout=3600 * 2,
)
def run_batch_size_experiment(config: dict):
    import sys
    sys.path.insert(0, "/root")
    import argparse
    import torch
    from train import train

    batch_size = config["batch_size"]
    lr = config["lr"]
    max_iters = config.get("max_iters", 2500)
    warmup_iters = config.get("warmup_iters", max(10, int(0.1 * max_iters)))
    eval_iters = config.get("eval_iters", max(10, min(100, 1280 // batch_size)))
    eval_interval = config.get("eval_interval", 100 if batch_size > 16 else 250)
    log_interval = config.get("log_interval", 10 if batch_size >= 64 else 50)
    run_name = config.get("run_name", f"bs-{batch_size}")
    checkpoint_dir = f"/vol/checkpoints/batch_size_sweep/{run_name}"
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
        warmup_iters=warmup_iters,
        max_iters=max_iters,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip=1.0,
        device="cuda",
        seed=1337,
        log_interval=log_interval,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        save_interval=0,
        checkpoint_dir=checkpoint_dir,
        resume=None,
        wandb_project="cs336-assignment1-basics",
        wandb_run_name=run_name,
    )

    print("=" * 70)
    print(f"--> [LAUNCH] Batch Size Experiment: {run_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch Size: {batch_size} | LR: {lr:.2e} | Steps: {max_iters} | Eval Iters: {eval_iters}")
    print(f"Tokens/step: {batch_size * 256:,} | Est. Total Tokens: {batch_size * 256 * max_iters:,}")
    print("=" * 70)

    try:
        train(args)
        print(f"[SUCCESS] Run {run_name} finished successfully.")
        return {"run_name": run_name, "batch_size": batch_size, "status": "COMPLETED"}
    except Exception as e:
        print(f"[ERROR] Run {run_name} encountered error: {e}")
        return {"run_name": run_name, "batch_size": batch_size, "status": f"FAILED: {e}"}


@app.local_entrypoint()
def main():
    configs = [
        {"batch_size": 1, "lr": 3e-4, "max_iters": 5000, "run_name": "bs-1-micro"},
        {"batch_size": 16, "lr": 1e-3, "max_iters": 2500, "run_name": "bs-16-small"},
        {"batch_size": 64, "lr": 2e-3, "max_iters": 2500, "run_name": "bs-64-medium"},
        {"batch_size": 128, "lr": 3e-3, "max_iters": 2500, "run_name": "bs-128-baseline"},
        {"batch_size": 512, "lr": 5e-3, "max_iters": 1500, "run_name": "bs-512-large"},
        {"batch_size": 1024, "lr": 6e-3, "max_iters": 1000, "run_name": "bs-1024-xlarge"},
        {"batch_size": 1536, "lr": 6.5e-3, "max_iters": 800, "run_name": "bs-1536-vram-limit"},
    ]

    print("=" * 75)
    print(f"Launching {len(configs)} Parallel Batch Size Experiments on Modal [{GPU_TYPE}]")
    print(f"Project: cs336-assignment1-basics")
    for cfg in configs:
        tokens = cfg['batch_size'] * 256 * cfg['max_iters']
        print(f"  * {cfg['run_name']}: B={cfg['batch_size']:4d}, LR={cfg['lr']:.1e}, Steps={cfg['max_iters']:5d}, Tokens={tokens:>11,}")
    print("=" * 75)

    results = list(run_batch_size_experiment.map(configs))
    print("\n" + "=" * 75)
    print("ALL BATCH SIZE RUNS COMPLETED:")
    for r in results:
        print(f"  * {r['run_name']}: {r['status']}")
    print("=" * 75)
