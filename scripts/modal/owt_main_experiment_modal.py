import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import modal

app = modal.App("cs336-owt-main-experiment")
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
    timeout=3600,
)
def run_owt_training(config: dict):
    import sys
    sys.path.insert(0, "/root")
    import argparse
    import torch
    from train import train

    lr = config.get("lr", 2.5e-3)
    max_iters = config.get("max_iters", 2500)
    batch_size = config.get("batch_size", 128)
    d_model = config.get("d_model", 512)
    num_layers = config.get("num_layers", 4)
    num_heads = config.get("num_heads", 16)
    d_ff = config.get("d_ff", 1344)
    ffn_type = config.get("ffn_type", "swiglu")
    vocab_size = config.get("vocab_size", 32000)
    run_name = config.get("run_name", f"owt-h100-{ffn_type}-lr-{lr:.1e}")
    checkpoint_dir = f"/vol/checkpoints/owt_main_experiment/{run_name}"
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
        save_interval=1000,
        checkpoint_dir=checkpoint_dir,
        resume=None,
        wandb_project="cs336-assignment1-basics",
        wandb_run_name=run_name,
        use_rmsnorm=True,
        norm_position="pre",
        use_rope=True,
    )

    print("=" * 75)
    print(f"--> [LAUNCH] OpenWebText Pretraining: {run_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Vocab: {vocab_size:,} | Architecture: Pre-Norm + RMSNorm + RoPE + SwiGLU")
    print(f"Dimensions: d_model={d_model}, L={num_layers}, H={num_heads}, d_ff={d_ff}")
    print(f"Optimizer: AdamW (lr={lr:.2e}, min_lr={args.min_lr:.2e}, warmup=250 steps, cosine)")
    print(f"Throughput target: batch_size={batch_size} (32,768 tok/step), steps={max_iters} (~81.9M tokens)")
    print("=" * 75)

    try:
        train(args)
        vol.commit()
        print(f"[SUCCESS] Run {run_name} finished successfully and committed to volume.")
        return {"run_name": run_name, "status": "COMPLETED", "checkpoint_dir": checkpoint_dir}
    except Exception as e:
        print(f"[ERROR] Run {run_name} encountered error: {e}")
        return {"run_name": run_name, "status": f"FAILED: {e}"}


@app.local_entrypoint()
def main():
    cfg = {
        "lr": 2.5e-3,
        "max_iters": 2500,
        "batch_size": 128,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "ffn_type": "swiglu",
        "vocab_size": 32000,
        "run_name": "owt-h100-swiglu-lr-2.5e-3",
    }

    print("=" * 75)
    print(f"Launching OpenWebText Pretraining on Modal [{GPU_TYPE}]")
    print(f"Project: cs336-assignment1-basics | Run: {cfg['run_name']}")
    print("=" * 75)

    res = run_owt_training.remote(cfg)
    print("\n" + "=" * 75)
    print(f"Result: {res}")
    print("=" * 75)
