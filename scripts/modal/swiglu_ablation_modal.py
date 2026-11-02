import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import modal

app = modal.App("cs336-swiglu-ablation")
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
def run_swiglu_ablation(config: dict):
    import sys
    sys.path.insert(0, "/root")
    import argparse
    import torch
    from train import train

    lr = config["lr"]
    max_iters = config.get("max_iters", 2500)
    batch_size = config.get("batch_size", 128)
    ffn_type = config.get("ffn_type", "swiglu")
    d_ff = config.get("d_ff", 1344 if ffn_type == "swiglu" else 2048)
    run_name = config.get("run_name", f"{ffn_type}-dff-{d_ff}-lr-{lr:.1e}")
    checkpoint_dir = f"/vol/checkpoints/swiglu_ablation/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    args = argparse.Namespace(
        train_data="/vol/tinystories_train.npy",
        val_data="/vol/tinystories_valid.npy",
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=d_ff,
        ffn_type=ffn_type,
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
        use_rope=True,
    )

    print("=" * 70)
    print(f"--> [LAUNCH] SwiGLU Ablation: {run_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FFN Type: {ffn_type} | d_ff: {d_ff} | LR: {lr:.2e} | Steps: {max_iters} | Batch: {batch_size}")
    print("=" * 70)

    try:
        train(args)
        print(f"[SUCCESS] Run {run_name} finished successfully.")
        return {"run_name": run_name, "ffn_type": ffn_type, "d_ff": d_ff, "lr": lr, "status": "COMPLETED"}
    except Exception as e:
        print(f"[ERROR] Run {run_name} encountered error: {e}")
        return {"run_name": run_name, "ffn_type": ffn_type, "d_ff": d_ff, "lr": lr, "status": f"FAILED: {e}"}


@app.local_entrypoint()
def main():
    configs = [
        # Baseline: SwiGLU with d_ff = 1344 (~8/3 * d_model)
        {
            "ffn_type": "swiglu",
            "d_ff": 1344,
            "lr": 3e-3,
            "max_iters": 2500,
            "run_name": "swiglu-baseline-lr-3e-3",
        },
        # Ablation: Ungated SiLU FFN with d_ff = 2048 (4 * d_model) to match parameter counts
        {
            "ffn_type": "silu",
            "d_ff": 2048,
            "lr": 3e-3,
            "max_iters": 2500,
            "run_name": "silu-ablation-lr-3e-3",
        },
    ]

    print("=" * 75)
    print(f"Launching {len(configs)} Parallel SwiGLU vs. SiLU Ablation Runs on Modal [{GPU_TYPE}]")
    print(f"Project: cs336-assignment1-basics")
    for cfg in configs:
        print(f"  • {cfg['run_name']}: ffn_type={cfg['ffn_type']}, d_ff={cfg['d_ff']}, lr={cfg['lr']}")
    print("=" * 75)

    results = list(run_swiglu_ablation.map(configs))
    print("\n" + "=" * 75)
    print("All runs finished. Summary:")
    for r in results:
        print(f"  {r['run_name']}: {r['status']}")
    print("=" * 75)
