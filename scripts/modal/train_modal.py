import os
import modal

app = modal.App("cs336-tinystories-training")
vol = modal.Volume.from_name("cs336-dataset-vol", create_if_missing=True)

# Select GPU: "H100" (NVIDIA H100 80GB SXM5)
GPU_TYPE = os.environ.get("MODAL_GPU", "H100")

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
    timeout=3600 * 3,  # 3 hours max
)
def train_cloud(
    batch_size: int = 128,
    max_iters: int = 10000,
    lr: float = 3e-3,
    min_lr: float = 3e-4,
    warmup_iters: int = 500,
    log_interval: int = 10,
    eval_interval: int = 250,
    eval_iters: int = 20,
    save_interval: int = 1000,
    checkpoint_dir: str = "/vol/checkpoints/tinystories_baseline",
    resume: str | None = None,
    wandb_project: str | None = "cs336-assignment1-basics",
    wandb_run_name: str | None = "tinystories-baseline-b128-h100",
):
    import sys
    sys.path.insert(0, "/root")
    import argparse
    import torch
    from train import train

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
        min_lr=min_lr,
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
        save_interval=save_interval,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    print("=" * 70)
    print("🚀 Starting Modal Cloud GPU Training")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"Total VRAM: {total_mem:.0f} MB ({total_mem / 1024:.2f} GB)")
    print(f"Batch Size: {batch_size} | Max Iters: {max_iters} | Context Length: {args.context_length}")
    print(f"Weights & Biases Project: {wandb_project} | Run: {wandb_run_name}")
    print("=" * 70)

    train(args)

    vol.commit()
    print("Successfully committed checkpoint volume!")


@app.local_entrypoint()
def main(
    batch_size: int = 128,
    max_iters: int = 10000,
    log_interval: int = 10,
    eval_interval: int = 250,
    eval_iters: int = 20,
    save_interval: int = 1000,
    wandb_project: str = "cs336-assignment1-basics",
    wandb_run_name: str = "tinystories-baseline-b128-h100",
):
    print("=" * 70)
    print(f"Launching Modal Cloud Training [GPU: {GPU_TYPE}]")
    print(f"Batch size: {batch_size} | Total Steps: {max_iters} (Tokens: {batch_size * 256 * max_iters:,})")
    print(f"W&B: Project '{wandb_project}', Run '{wandb_run_name}'")
    print("=" * 70)

    train_cloud.remote(
        batch_size=batch_size,
        max_iters=max_iters,
        log_interval=log_interval,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        save_interval=save_interval,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
