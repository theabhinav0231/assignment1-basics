import os
import sys
import re
import json
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

log_file = r"C:/Users/Abhinav/.gemini/antigravity/brain/c1fcb3ef-0e4c-4661-857a-d8c4f37f509b/.system_generated/steps/5761/output.txt"

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

lines = text.splitlines()

eval_steps = list(range(0, 2500, 100)) + [2499]

# Parse validation logs
val_by_step = {s: [] for s in eval_steps}

for line in lines:
    m = re.search(r'--> VALIDATION \[Step (\d+)\]: Loss = ([\d\.]+) \| Perplexity = ([\d\.]+)', line)
    if m:
        s = int(m.group(1))
        loss = float(m.group(2))
        ppl = float(m.group(3))
        if s in val_by_step:
            val_by_step[s].append((loss, ppl))

trajectories = {
    "rope_baseline": {"steps": [], "loss": [], "ppl": []},
    "nope_ablation": {"steps": [], "loss": [], "ppl": []},
}

for s in eval_steps:
    vals = val_by_step[s]
    unique_vals = []
    for v in vals:
        if v not in unique_vals:
            unique_vals.append(v)
    
    if len(unique_vals) >= 2:
        # Lower loss is RoPE baseline, higher loss is NoPE ablation
        sorted_vals = sorted(unique_vals, key=lambda x: x[0])
        trajectories["rope_baseline"]["steps"].append(s)
        trajectories["rope_baseline"]["loss"].append(sorted_vals[0][0])
        trajectories["rope_baseline"]["ppl"].append(sorted_vals[0][1])

        trajectories["nope_ablation"]["steps"].append(s)
        trajectories["nope_ablation"]["loss"].append(sorted_vals[1][0])
        trajectories["nope_ablation"]["ppl"].append(sorted_vals[1][1])
    elif len(unique_vals) == 1 and s == 0:
        trajectories["rope_baseline"]["steps"].append(0)
        trajectories["rope_baseline"]["loss"].append(unique_vals[0][0])
        trajectories["rope_baseline"]["ppl"].append(unique_vals[0][1])

        trajectories["nope_ablation"]["steps"].append(0)
        trajectories["nope_ablation"]["loss"].append(unique_vals[0][0])
        trajectories["nope_ablation"]["ppl"].append(unique_vals[0][1])

print("Parsed RoPE vs NoPE:")
for k, v in trajectories.items():
    print(f"  {k}: {len(v['steps'])} steps, start={v['loss'][0]}, end={v['loss'][-1]}")

# Save JSON
out_dir = r"C:/Users/Abhinav/.gemini/antigravity/brain/c1fcb3ef-0e4c-4661-857a-d8c4f37f509b"
with open(os.path.join(out_dir, "nope_ablation_data.json"), "w") as f:
    json.dump(trajectories, f, indent=2)

# Create high-res publication plot
plt.figure(figsize=(8, 3.8), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

steps = trajectories["rope_baseline"]["steps"]
sub_idx = [i for i, s in enumerate(steps) if s >= 100]
s_sub = [steps[i] for i in sub_idx]

p_rope = [trajectories["rope_baseline"]["loss"][i] for i in sub_idx]
p_nope = [trajectories["nope_ablation"]["loss"][i] for i in sub_idx]

plt.plot(s_sub, p_rope, marker='o', markersize=3.5, linewidth=1.8, color='#0074D9', label=r'RoPE Baseline ($\theta=10000$)')
plt.plot(s_sub, p_nope, marker='s', markersize=3.5, linewidth=1.8, color='#FF4136', linestyle='--', label='NoPE Ablation (No Positional Embeddings)')

plt.title("RoPE vs. NoPE (No Positional Embeddings) Learning Curves (TinyStories)", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Training Step", fontsize=10, fontweight='bold')
plt.ylabel("Validation Cross-Entropy Loss", fontsize=10, fontweight='bold')
plt.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=9.5, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

# Annotation of final gap
plt.annotate(
    f"RoPE: 0.8231 (Ppl: 2.28)\nNoPE: 0.8867 (Ppl: 2.43)\nGap: +0.0636 loss",
    xy=(2499, 0.8231),
    xytext=(1600, 1.22),
    arrowprops=dict(facecolor='#333333', arrowstyle='->', lw=0.8),
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=0.8)
)

plt.tight_layout()
plot_path = os.path.join(out_dir, "nope_ablation.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to: {plot_path}")

# Also copy to assignment1-basics/plots/nope_ablation.png
import shutil
dst_plot = os.path.join(r"assignment1-basics/plots", "nope_ablation.png")
shutil.copyfile(plot_path, dst_plot)
print(f"Copied to: {dst_plot}")
