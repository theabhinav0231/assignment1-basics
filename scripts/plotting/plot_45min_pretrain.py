import json
import matplotlib.pyplot as plt

data_path = "plots/owt_45min_data.json"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

metrics = data["metrics"]
history = metrics["history"]

total_time_min = metrics["total_time_seconds"] / 60.0
train_times = [s * total_time_min / 10000.0 for s in history["train_steps"]]
val_times = [s * total_time_min / 10000.0 for s in history["val_steps"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

c_train = "#0074D9"  # Deep blue
c_val = "#FF851B"    # Orange

# Panel 1: Training Loss vs Wall-Clock Time
ax1.plot(train_times, history["train_loss"], color=c_train, alpha=0.85, linewidth=1.5, label="Train Loss (Step 0 to 10,000)")
ax1.axvline(45.0, color="#E74C3C", linestyle=":", linewidth=1.5, label="Leaderboard Limit (45.0 min)")
ax1.set_title("OpenWebText 45-Min Pretraining: Training Loss vs. Wall-Clock Time", fontsize=12, fontweight="bold")
ax1.set_xlabel("Wall-Clock Time (Minutes)", fontsize=11)
ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11)
ax1.set_xlim(-0.5, 46.0)
ax1.set_ylim(1.9, 4.5)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=10, loc="upper right")

# Panel 2: Validation Loss vs Wall-Clock Time
ax2.plot(val_times, history["val_loss"], "o-", color=c_val, linewidth=2, markersize=3.5, label=f"Validation Loss (Final: {metrics['final_val_loss']:.4f}, PPL: {metrics['final_val_ppl']:.2f})")
ax2.axhline(2.3923, color="#7F8C8D", linestyle="--", linewidth=1.2, label="Untied Baseline (81.9M tok): 2.3923 (PPL 10.94)")
ax2.axhline(2.2841, color="#2ECC40", linestyle="--", linewidth=1.2, label="2.5k Tied Test (163.8M tok): 2.2841 (PPL 9.82)")
ax2.axvline(45.0, color="#E74C3C", linestyle=":", linewidth=1.5, label="Leaderboard Limit (45.0 min)")

ax2.set_title("Validation Loss Trajectory vs. Wall-Clock Time ($V=32,000$ OWT Valid)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Wall-Clock Time (Minutes)", fontsize=11)
ax2.set_ylabel("Validation Loss (nats)", fontsize=11)
ax2.set_xlim(-0.5, 46.0)
ax2.set_ylim(2.0, 3.5)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=9, loc="upper right")

# Add summary annotation box
summary_text = (
    f"Leaderboard Submission Performance:\n"
    f"• Hardware: NVIDIA H100-80GB SXM5 (Modal)\n"
    f"• Architecture: 4L / 16H / 512D / SwiGLU / RoPE / RMSNorm\n"
    f"• Weight Tying: Shared Embeddings & Head (28.8M params)\n"
    f"• Precision: bfloat16 | Batch: 256 (65,536 tok/step)\n"
    f"• Steps Completed: 10,000 / 10,000 (655.36M tokens)\n"
    f"• Total Wall-Clock Time: {total_time_min:.2f} min (Budget: 45.0 min)\n"
    f"• Average Throughput: {metrics['tokens_per_second']:,.0f} tok/s\n"
    f"• Final Validation Loss: {metrics['final_val_loss']:.4f} (Perplexity: {metrics['final_val_ppl']:.2f})\n"
    f"• Leaderboard Threshold: Beats 5.0 baseline by -2.8785 nats"
)
ax2.text(
    0.04, 0.05, summary_text,
    transform=ax2.transAxes,
    fontsize=8.5,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.95)
)

plt.tight_layout()
out_fig = "plots/owt_45min_pretrain.png"
plt.savefig(out_fig, bbox_inches="tight")
print(f"Saved pretraining plot to {out_fig}")
