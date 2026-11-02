import json
import matplotlib.pyplot as plt

data_path = "plots/owt_lr_comparison_data.json"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

run_20 = data["owt-h100-tied-bf16-b256-lr-2.0e-3"]["metrics"]
run_25 = data["owt-h100-tied-bf16-b256-lr-2.5e-3"]["metrics"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Colors
c_20 = "#0074D9"  # Blue for 2.0e-3
c_25 = "#FF851B"  # Orange for 2.5e-3

# Panel 1: Training Loss
ax1.plot(run_20["history"]["train_steps"], run_20["history"]["train_loss"], label="Model A: $\eta = 2.0 \\times 10^{-3}$ (Balanced)", color=c_20, alpha=0.85, linewidth=1.5)
ax1.plot(run_25["history"]["train_steps"], run_25["history"]["train_loss"], label="Model B: $\eta = 2.5 \\times 10^{-3}$ ($\sqrt{B/B_0}$)", color=c_25, alpha=0.85, linewidth=1.5)
ax1.set_title("Training Loss Trajectory ($B=256$, Tied Weights)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Optimization Steps", fontsize=11)
ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11)
ax1.set_ylim(2.1, 4.5)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=10, loc="upper right")

# Panel 2: Validation Loss
ax2.plot(run_20["history"]["val_steps"], run_20["history"]["val_loss"], "o-", label=f"Model A: $\eta = 2.0 \\times 10^{{-3}}$ (Final: {run_20['final_val_loss']:.4f})", color=c_20, linewidth=2, markersize=4)
ax2.plot(run_25["history"]["val_steps"], run_25["history"]["val_loss"], "s-", label=f"Model B: $\eta = 2.5 \\times 10^{{-3}}$ (Final: {run_25['final_val_loss']:.4f})", color=c_25, linewidth=2, markersize=4)
ax2.set_title("Validation Loss Comparison ($V=32,000$ OWT Valid)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Optimization Steps", fontsize=11)
ax2.set_ylabel("Validation Loss (nats)", fontsize=11)
ax2.set_ylim(2.2, 4.0)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=10, loc="upper right")

# Add summary annotation box
summary_text = (
    "Convergence Summary:\n"
    "• Model A (2.0e-3): Val Loss = 2.2841 (PPL 9.82)\n"
    "• Model B (2.5e-3): Val Loss = 2.2852 (PPL 9.83)\n"
    "• Gap: Δ = 0.0011 nats (<0.05% difference)\n"
    "• Both beat Untied Baseline (2.3923) by -0.108 nats"
)
ax2.text(
    0.05, 0.08, summary_text,
    transform=ax2.transAxes,
    fontsize=9.5,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.95)
)

plt.tight_layout()
out_fig = "plots/owt_lr_comparison.png"
plt.savefig(out_fig, bbox_inches="tight")
print(f"Saved comparison plot to {out_fig}")
