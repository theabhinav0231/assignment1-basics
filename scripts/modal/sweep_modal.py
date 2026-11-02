import os
import modal

app = modal.App("cs336-lr-sweep")
vol = modal.Volume.from_name("cs336-dataset-vol", create_if_missing=True)

# Selected GPU as requested: A100 40GB
GPU_TYPE = "A100-40GB"

# Weights & Biases API Key Secret
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
    timeout=3600 * 2,  # 2 hours max per run
)
def run_single_sweep(config: dict):
    import sys
    sys.path.insert(0, "/root")
    import argparse
    import torch
    from train import train

    lr = config["lr"]
    max_iters = config.get("max_iters", 2500)
    batch_size = config.get("batch_size", 128)
    warmup_iters = config.get("warmup_iters", 250)
    run_name = config.get("run_name", f"lr-{lr:.1e}")
    checkpoint_dir = f"/vol/checkpoints/lr_sweep/{run_name}"
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
        log_interval=10,
        eval_interval=100,
        eval_iters=20,
        save_interval=0,  # Skip periodic checkpoints to save disk space
        checkpoint_dir=checkpoint_dir,
        resume=None,
        wandb_project="cs336-assignment1-basics",
        wandb_run_name=run_name,
    )

    print("=" * 70)
    print(f"🚀 Starting Run: {run_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Learning Rate: {lr:.2e} | Warmup: {warmup_iters} | Max Iters: {max_iters}")
    print(f"Weights & Biases Run: {run_name}")
    print("=" * 70)

    try:
        train(args)
        print(f"✅ Run {run_name} finished successfully.")
    except Exception as e:
        print(f"⚠️ Run {run_name} terminated with error: {e}")

    return run_name


@app.local_entrypoint()
def main():
    sweep_configs = [
        {"lr": 5e-4, "max_iters": 2500, "run_name": "lr-5e-4-low"},
        {"lr": 1e-3, "max_iters": 2500, "run_name": "lr-1e-3-medium-low"},
        {"lr": 3e-3, "max_iters": 2500, "run_name": "lr-3e-3-baseline"},
        {"lr": 6e-3, "max_iters": 2500, "run_name": "lr-6e-3-edge-of-stability"},
        {"lr": 1.2e-2, "max_iters": 1000, "run_name": "lr-1.2e-2-divergent"},
        {"lr": 2.5e-2, "max_iters": 1000, "run_name": "lr-2.5e-2-divergent"},
    ]

    print("=" * 70)
    print(f"Launching {len(sweep_configs)} parallel learning rate runs on Modal [{GPU_TYPE}]")
    print(f"Project: cs336-assignment1-basics")
    for cfg in sweep_configs:
        print(f"  • {cfg['run_name']}: lr={cfg['lr']:.2e}, steps={cfg['max_iters']}")
    print("=" * 70)

    for completed in run_single_sweep.map(sweep_configs):
        print(f"--> [SWEEP PROGRESS] Finished run: {completed}")
