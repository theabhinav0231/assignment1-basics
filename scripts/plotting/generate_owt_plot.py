import re
import json
import os
import matplotlib.pyplot as plt

log_file = r'C:\Users\Abhinav\.gemini\antigravity\brain\c1fcb3ef-0e4c-4661-857a-d8c4f37f509b\.system_generated\tasks\task-6141.log'
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Pattern for validation steps
val_pattern = re.compile(r'--> VALIDATION \[Step (\d+)\]:\s*Loss\s*=\s*([0-9.]+)\s*\|\s*Perplexity\s*=\s*([0-9.]+)')

owt_steps = []
for m in val_pattern.finditer(text):
    step = int(m.group(1))
    loss = float(m.group(2))
    ppl = float(m.group(3))
    owt_steps.append({"step": step, "val_loss": loss, "val_ppl": ppl})

print(f"Parsed {len(owt_steps)} validation evaluation points for OpenWebText.")
print("Initial:", owt_steps[0])
print("Final:", owt_steps[-1])

# Load TinyStories baseline for comparison
ts_file = r'C:\Users\Abhinav\.gemini\antigravity\brain\c1fcb3ef-0e4c-4661-857a-d8c4f37f509b\swiglu_ablation_data.json'
with open(ts_file, 'r', encoding='utf-8') as f:
    ts_data = json.load(f)["swiglu_baseline"]

data_to_save = {
    "openwebtext_experiment": owt_steps,
    "tinystories_baseline": ts_data,
}

json_path = r'C:\Users\Abhinav\.gemini\antigravity\brain\c1fcb3ef-0e4c-4661-857a-d8c4f37f509b\owt_experiment_data.json'
with open(json_path, 'w') as f:
    json.dump(data_to_save, f, indent=2)
print("Saved data to:", json_path)

# Generate Publication Plot
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=200)

steps_owt = [d["step"] for d in owt_steps]
loss_owt = [d["val_loss"] for d in owt_steps]
ppl_owt = [d["val_ppl"] for d in owt_steps]

steps_ts = [d["step"] for d in ts_data]
loss_ts = [d["val_loss"] for d in ts_data]

# Panel 1: OpenWebText Validation Loss & Perplexity (Dual Axis)
color_loss = '#1f77b4'
color_ppl = '#ff7f0e'

ax1.plot(steps_owt, loss_owt, color=color_loss, linewidth=2.3, marker='o', markersize=3, label='OWT Validation Cross-Entropy Loss')
ax1.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax1.set_ylabel('Validation Loss (Cross-Entropy)', color=color_loss, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_ylim(2.0, 11.0)
ax1.set_title('OpenWebText Pretraining Trajectory (NVIDIA H100)', fontsize=13, fontweight='bold', pad=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Secondary axis for perplexity (log scale)
ax1_twin = ax1.twinx()
ax1_twin.plot(steps_owt, ppl_owt, color=color_ppl, linewidth=1.8, linestyle='--', label='OWT Perplexity')
ax1_twin.set_ylabel('Perplexity (Log Scale)', color=color_ppl, fontsize=12, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=color_ppl)
ax1_twin.set_yscale('log')
ax1_twin.grid(False)

# Annotate final loss and perplexity
ax1.annotate(f'Final Loss: {loss_owt[-1]:.4f}\n(Perplexity: {ppl_owt[-1]:.2f})',
             xy=(steps_owt[-1], loss_owt[-1]),
             xytext=(steps_owt[-1] - 800, loss_owt[-1] + 2.0),
             arrowprops=dict(facecolor=color_loss, shrink=0.08, width=1.2, headwidth=6),
             fontweight='bold', color=color_loss, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_loss, lw=1.2))

# Panel 2: Comparative Learning Curves (TinyStories vs. OpenWebText)
ax2.plot(steps_owt, loss_owt, color='#d62728', linewidth=2.3, label='OpenWebText (32k vocab, 34.0M params)')
ax2.plot(steps_ts, loss_ts, color='#2ca02c', linewidth=2.3, label='TinyStories (10k vocab, 22.7M params)')
ax2.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax2.set_ylabel('Validation Cross-Entropy Loss', fontsize=12, fontweight='bold')
ax2.set_title('Domain & Vocabulary Loss Gap: OWT vs. TinyStories', fontsize=13, fontweight='bold', pad=10)
ax2.legend(loc='center right', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylim(0.5, 11.0)

# Annotate the asymptotic loss gap
ax2.annotate('Entropy & Diversity Gap: ~1.57 nats',
             xy=(2000, 1.6), xytext=(800, 4.5),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.0, headwidth=5),
             fontweight='bold', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="gray", lw=1.0))

plt.tight_layout()

plot_out_brain = r'C:\Users\Abhinav\.gemini\antigravity\brain\c1fcb3ef-0e4c-4661-857a-d8c4f37f509b\owt_experiment.png'
plt.savefig(plot_out_brain, dpi=200, bbox_inches='tight')

plot_out_repo = r'assignment1-basics\plots\owt_experiment.png'
os.makedirs(os.path.dirname(plot_out_repo), exist_ok=True)
plt.savefig(plot_out_repo, dpi=200, bbox_inches='tight')
print("Plot successfully saved to:")
print("  -", plot_out_brain)
print("  -", plot_out_repo)
