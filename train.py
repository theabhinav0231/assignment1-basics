import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
import psutil

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint


def load_dataset(path: str) -> np.ndarray:
    """
    Loads a tokenized dataset memory-efficiently using memory mapping.
    Supports both .npy files and raw binary files.
    """
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    else:
        # Default for raw binary tokenized datasets (uint16)
        return np.memmap(path, dtype=np.uint16, mode="r")


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
    use_bfloat16: bool = False,
) -> tuple[float, float]:
    """
    Compute average loss and perplexity over eval_iters random batches.
    """
    model.eval()
    losses = []
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bfloat16 and "cuda" in str(device)) else contextlib.nullcontext()
    eval_batch_size = min(batch_size, 128)
    for _ in range(eval_iters):
        x, y = get_batch(dataset, eval_batch_size, context_length, device)
        with ctx:
            logits = model(x)
            loss = cross_entropy(logits, y)
        losses.append(loss.item())
    model.train()
    mean_loss = float(np.mean(losses))
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, perplexity


def train(args):
    # 1. Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 2. Device setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    print(f"Using device: {device}")

    # 3. Memory-mapped data loading
    print(f"Loading training data from: {args.train_data}")
    train_data = load_dataset(args.train_data)
    print(f"Train dataset contains {len(train_data):,} tokens.")

    val_data = None
    if args.val_data and os.path.exists(args.val_data):
        print(f"Loading validation data from: {args.val_data}")
        val_data = load_dataset(args.val_data)
        print(f"Val dataset contains {len(val_data):,} tokens.")

    # 4. Initialize Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(device),
        use_rmsnorm=getattr(args, "use_rmsnorm", True),
        norm_position=getattr(args, "norm_position", "pre"),
        use_rope=getattr(args, "use_rope", True),
        ffn_type=getattr(args, "ffn_type", "swiglu"),
        tie_weights=getattr(args, "tie_weights", False),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized TransformerLM with {total_params:,} parameters.")

    # 5. Initialize Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 6. Resume from checkpoint if provided
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed at step {start_step}.")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    last_val_loss = float("inf")
    last_val_ppl = float("inf")
    t0 = time.time()
    tokens_per_step = args.batch_size * args.context_length
    use_bfloat16 = getattr(args, "use_bfloat16", False)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bfloat16 and "cuda" in str(device)) else contextlib.nullcontext()
    max_time_seconds = getattr(args, "max_time_seconds", 0)
    grad_accum_steps = getattr(args, "grad_accum_steps", 1)
    if grad_accum_steps <= 0:
        grad_accum_steps = 1
    assert args.batch_size % grad_accum_steps == 0, f"batch_size ({args.batch_size}) must be divisible by grad_accum_steps ({grad_accum_steps})"
    micro_batch_size = args.batch_size // grad_accum_steps

    history = {
        "train_steps": [],
        "train_loss": [],
        "train_ppl": [],
        "train_lr": [],
        "val_steps": [],
        "val_loss": [],
        "val_ppl": [],
    }

    # 7. Initialize Weights & Biases if requested
    use_wandb = bool(getattr(args, "wandb_project", None))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=getattr(args, "wandb_run_name", None),
                config=vars(args),
            )
            print(f"Weights & Biases logging initialized for project: {args.wandb_project}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb ({e}). Continuing without wandb.")
            use_wandb = False

    print("=" * 70)
    print(f"Starting training for {args.max_iters - start_step} steps...")
    print(f"Batch size: {args.batch_size} (micro: {micro_batch_size}, accum: {grad_accum_steps}) | Context length: {args.context_length} | Tokens/step: {tokens_per_step:,}")
    print("=" * 70)

    for step in range(start_step, args.max_iters):
        # 7. Update learning rate using cosine schedule
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 8. Sample training batch & accumulate gradients
        optimizer.zero_grad()
        step_loss = 0.0
        diverged = False

        for micro_idx in range(grad_accum_steps):
            x, y = get_batch(train_data, micro_batch_size, args.context_length, device)
            with autocast_ctx:
                logits = model(x)
                micro_loss = cross_entropy(logits, y)
                scaled_loss = micro_loss / grad_accum_steps

            if math.isnan(micro_loss.item()) or math.isinf(micro_loss.item()) or micro_loss.item() > 50.0:
                print(f"\n[ALERT] Divergence detected at Step {step} (micro {micro_idx}): Loss = {micro_loss.item()}")
                if use_wandb:
                    wandb.log({
                        "train/loss": float("nan"),
                        "train/diverged": 1,
                    }, step=step)
                diverged = True
                break

            scaled_loss.backward()
            step_loss += micro_loss.item() / grad_accum_steps

        if diverged:
            break

        # 10. Gradient clipping & optimizer step
        if args.grad_clip > 0.0:
            clip_gradient(model.parameters(), max_l2_norm=args.grad_clip)
        optimizer.step()

        # 12. Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            tps = (tokens_per_step * (step - start_step + 1)) / max(elapsed, 1e-5)
            ppl = math.exp(step_loss) if step_loss < 20 else float("inf")
            
            mem_info = ""
            ram_used = psutil.Process().memory_info().rss / (1024 ** 2)
            if torch.cuda.is_available() and (isinstance(device, str) and device.startswith("cuda") or getattr(device, "type", "") == "cuda"):
                cur_vram = torch.cuda.memory_allocated() / (1024 ** 2)
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                mem_info = f" | VRAM: {cur_vram:.0f}MB/{total_vram:.0f}MB (Peak: {peak_vram:.0f}MB) | RAM: {ram_used:.0f}MB"
            else:
                mem_info = f" | RAM: {ram_used:.0f}MB"

            print(
                f"Step {step:6d}/{args.max_iters} | "
                f"Loss: {step_loss:.4f} | "
                f"Ppl: {ppl:7.2f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tps:8.1f}{mem_info} | "
                f"Time: {elapsed:6.1f}s"
            )

            # Log training metrics
            history["train_steps"].append(step)
            history["train_loss"].append(float(step_loss))
            history["train_ppl"].append(float(ppl))
            history["train_lr"].append(float(lr))

            if use_wandb:
                wb_dict = {
                    "train/loss": step_loss,
                    "train/perplexity": ppl,
                    "train/lr": lr,
                    "train/tokens_seen": (step + 1) * tokens_per_step,
                    "train/throughput_tok_per_sec": tps,
                    "perf/ram_used_mb": ram_used,
                }
                if torch.cuda.is_available() and (isinstance(device, str) and device.startswith("cuda") or getattr(device, "type", "") == "cuda"):
                    wb_dict["perf/vram_allocated_mb"] = cur_vram
                    wb_dict["perf/vram_peak_mb"] = peak_vram
                wandb.log(wb_dict, step=step)

        # 13. Periodic Validation
        if val_data is not None and (step % args.eval_interval == 0 or step == args.max_iters - 1):
            val_loss, val_ppl = estimate_loss(
                model=model,
                dataset=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
                use_bfloat16=use_bfloat16,
            )
            last_val_loss = val_loss
            last_val_ppl = val_ppl
            history["val_steps"].append(step)
            history["val_loss"].append(float(val_loss))
            history["val_ppl"].append(float(val_ppl))
            print("-" * 70)
            print(f"--> VALIDATION [Step {step}]: Loss = {val_loss:.4f} | Perplexity = {val_ppl:.2f}")
            print("-" * 70)

            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                }, step=step)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.checkpoint_dir, "checkpoint_best.pt")
                save_checkpoint(model, optimizer, step, best_path)
                print(f"Saved new best checkpoint to {best_path}")

        # 14. Periodic Checkpoint Saving
        if args.save_interval > 0 and (step % args.save_interval == 0 or step == args.max_iters - 1):
            save_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{step}.pt")
            save_checkpoint(model, optimizer, step, save_path)
            print(f"Saved periodic checkpoint to {save_path}")

        # 15. Wall-clock timeout check
        if max_time_seconds > 0 and (time.time() - t0) >= max_time_seconds:
            print(f"\n[TIME LIMIT] Reached wall-clock cutoff of {max_time_seconds:.1f}s ({max_time_seconds/60:.1f}m) at step {step}.")
            break

    total_time = time.time() - t0
    print("=" * 70)
    print("Training complete!")
    final_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"Saved final checkpoint to {final_path}")

    metrics = {
        "best_val_loss": best_val_loss,
        "final_val_loss": last_val_loss,
        "final_val_ppl": last_val_ppl,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "total_time_seconds": total_time,
        "tokens_per_second": ((step - start_step + 1) * tokens_per_step) / max(total_time, 1e-5),
        "history": history,
    }
    metrics_path = os.path.join(args.checkpoint_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved run metrics to {metrics_path}")

    if use_wandb:
        wandb.finish()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CS336 Transformer LM Training Script")

    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to tokenized train dataset (.npy or binary)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to tokenized val dataset (.npy or binary)")

    # Model architecture arguments
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum sequence / context length")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden dimension d_model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer blocks")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="Inner FFN dimension (multiple of 64)")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE base frequency theta")
    parser.add_argument("--norm_position", type=str, default="pre", choices=["pre", "post"], help="Normalization placement: 'pre' or 'post'")
    parser.add_argument("--use_rmsnorm", action="store_true", default=True, help="Whether to use RMSNorm (default: True)")
    parser.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True, help="Whether to use RoPE or NoPE (default: True)")
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"], help="FFN activation type: 'swiglu' or 'silu'")
    parser.add_argument("--tie_weights", action="store_true", default=False, help="Whether to tie token embeddings and LM head weights")
    parser.add_argument("--use_bfloat16", action="store_true", default=False, help="Whether to enable torch bfloat16 mixed precision autocast")
    parser.add_argument("--max_time_seconds", type=float, default=0.0, help="Wall-clock time cutoff in seconds (0 = disabled)")
    # Optimizer & LR schedule arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (number of sequences)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-3, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-4, help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=500, help="Warmup iterations")
    parser.add_argument("--max_iters", type=int, default=5000, help="Total training steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW decoupled weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW numerical stability eps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Maximum gradient L2 norm for clipping (0 to disable)")
    # Execution & logging arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device string ('cpu', 'cuda', etc.)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10, help="Console log interval in steps")
    parser.add_argument("--eval_interval", type=int, default=250, help="Validation evaluation interval in steps")
    parser.add_argument("--eval_iters", type=int, default=20, help="Number of batches for estimating validation loss")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval in steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights and Biases project name (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights and Biases run name (optional)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
