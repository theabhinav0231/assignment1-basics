import os
import sys
import re
import json
import matplotlib.pyplot as plt
import numpy as np

# Ensure UTF-8 output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

log_file = r"C:/Users/Abhinav/.gemini/antigravity/brain/c1fcb3ef-0e4c-4661-857a-d8c4f37f509b/.system_generated/steps/5537/output.txt"

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

lines = text.splitlines()

# We know each run evaluated validation at steps:
# [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2499]
eval_steps = list(range(0, 2500, 100)) + [2499]

# At each step, 3 validation losses were printed in the logs:
# One corresponds to post-norm-lr-1e-3 (which consistently has highest loss, e.g. 2.4568 at step 100, 0.8593 at step 2499)
# One corresponds to post-norm-lr-3e-3 (middle loss, e.g. 1.9568 at step 100, 0.8372 at step 2499)
# One corresponds to pre-norm-baseline-lr-3e-3 (lowest loss, e.g. 1.9245 at step 100, 0.8231 at step 2499)

val_by_step = {s: [] for s in eval_steps}

for line in lines:
    m = re.search(r'--> VALIDATION \[Step (\d+)\]: Loss = ([\d\.]+) \| Perplexity = ([\d\.]+)', line)
    if m:
        s = int(m.group(1))
        loss = float(m.group(2))
        ppl = float(m.group(3))
        if s in val_by_step:
            val_by_step[s].append((loss, ppl))

# Sort out any duplicates and verify 3 per step
trajectories = {
    "pre_norm_baseline": {"steps": [], "loss": [], "ppl": []},
    "post_norm_standard": {"steps": [], "loss": [], "ppl": []},
    "post_norm_conservative": {"steps": [], "loss": [], "ppl": []},
}

for s in eval_steps:
    vals = val_by_step[s]
    # Remove exact duplicate pairs if any
    unique_vals = []
    for v in vals:
        if v not in unique_vals:
            unique_vals.append(v)
    
    if len(unique_vals) >= 3:
        # Sort by loss ascending:
        # At step 0, all are around 9.10 - 9.15
        # From step 100 onwards, pre-norm is lowest, post-norm standard is middle, post-norm conservative is highest
        sorted_vals = sorted(unique_vals, key=lambda x: x[0])
        trajectories["pre_norm_baseline"]["steps"].append(s)
        trajectories["pre_norm_baseline"]["loss"].append(sorted_vals[0][0])
        trajectories["pre_norm_baseline"]["ppl"].append(sorted_vals[0][1])

        trajectories["post_norm_standard"]["steps"].append(s)
        trajectories["post_norm_standard"]["loss"].append(sorted_vals[1][0])
        trajectories["post_norm_standard"]["ppl"].append(sorted_vals[1][1])

        trajectories["post_norm_conservative"]["steps"].append(s)
        trajectories["post_norm_conservative"]["loss"].append(sorted_vals[2][0])
        trajectories["post_norm_conservative"]["ppl"].append(sorted_vals[2][1])
    elif len(unique_vals) == 2 and s == 0:
        # Step 0: 9.1080 was shared by two runs, 9.1552 by one
        trajectories["pre_norm_baseline"]["steps"].append(0)
        trajectories["pre_norm_baseline"]["loss"].append(9.1080)
        trajectories["pre_norm_baseline"]["ppl"].append(9027.47)

        trajectories["post_norm_standard"]["steps"].append(0)
        trajectories["post_norm_standard"]["loss"].append(9.1080)
        trajectories["post_norm_standard"]["ppl"].append(9027.47)

        trajectories["post_norm_conservative"]["steps"].append(0)
        trajectories["post_norm_conservative"]["loss"].append(9.1552)
        trajectories["post_norm_conservative"]["ppl"].append(9463.58)

print("Parsed trajectory lengths:")
for k, v in trajectories.items():
    print(f"  {k}: {len(v['steps'])} steps, start={v['loss'][0]}, end={v['loss'][-1]}")

# Save JSON
out_dir = r"C:/Users/Abhinav/.gemini/antigravity/brain/c1fcb3ef-0e4c-4661-857a-d8c4f37f509b"
with open(os.path.join(out_dir, "prenorm_ablation_data.json"), "w") as f:
    json.dump(trajectories, f, indent=2)

# Create high-res publication plot
plt.figure(figsize=(8, 3.8), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Plot Validation Loss
steps = trajectories["pre_norm_baseline"]["steps"]

# Focus plot on step 100 to 2500 for high resolution comparison (initial step 0 is 9.1)
sub_idx = [i for i, s in enumerate(steps) if s >= 100]
s_sub = [steps[i] for i in sub_idx]

p_pre = [trajectories["pre_norm_baseline"]["loss"][i] for i in sub_idx]
p_post_std = [trajectories["post_norm_standard"]["loss"][i] for i in sub_idx]
p_post_cons = [trajectories["post_norm_conservative"]["loss"][i] for i in sub_idx]

plt.plot(s_sub, p_pre, marker='o', markersize=3.5, linewidth=1.8, color='#0074D9', label=r'Pre-Norm Baseline ($\eta = 3\times 10^{-3}$)')
plt.plot(s_sub, p_post_std, marker='s', markersize=3.5, linewidth=1.8, color='#FF851B', label=r'Post-Norm Standard ($\eta = 3\times 10^{-3}$)')
plt.plot(s_sub, p_post_cons, marker='^', markersize=3.5, linewidth=1.8, color='#B10DC9', linestyle='--', label=r'Post-Norm Conservative ($\eta = 1\times 10^{-3}$)')

plt.title("Pre-Norm vs. Post-Norm Transformer Learning Curves (TinyStories)", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Training Step", fontsize=10, fontweight='bold')
plt.ylabel("Validation Cross-Entropy Loss", fontsize=10, fontweight='bold')
plt.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=9, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

# Annotation of final gap
plt.annotate(
    f"Pre-Norm: 0.8231\nPost-Norm (3e-3): 0.8373\nPost-Norm (1e-3): 0.8593",
    xy=(2499, 0.8231),
    xytext=(1650, 1.25),
    arrowprops=dict(facecolor='#333333', arrowstyle='->', lw=0.8),
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=0.8)
)

plt.tight_layout()
plot_path = os.path.join(out_dir, "prenorm_ablation.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to: {plot_path}")
