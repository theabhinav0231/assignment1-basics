import re
import json
import os
import matplotlib.pyplot as plt

log_file = r'C:\Users\Abhinav\.gemini\antigravity\brain\c1fcb3ef-0e4c-4661-857a-d8c4f37f509b\.system_generated\tasks\task-6001.log'
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# In the parallel output, lines from the two runs are interwoven.
# Notice: Step 0 has:
# 9.1552 (swiglu) and 9.2377 (silu).
# Let's inspect which line belongs to which by checking context or order.
# In task-6005 output:
# Line 1: --> VALIDATION [Step 0]: Loss = 9.1552 (swiglu)
# Line 2: --> VALIDATION [Step 0]: Loss = 9.2377 (silu)
# Line 3: --> VALIDATION [Step 100]: Loss = 2.0733 (silu)
# Line 4: --> VALIDATION [Step 100]: Loss = 1.9568 (swiglu)
# Notice at step 2499:
# 0.8398 is silu (from wandb summary right after: val/loss 0.83978)
# 0.8231 is swiglu (from wandb summary right after: val/loss 0.82308)

# Let's write an exact parser that matches step numbers
swiglu_steps = []
silu_steps = []

val_pattern = re.compile(r'--> VALIDATION \[Step (\d+)\]:\s*Loss\s*=\s*([0-9.]+)\s*\|\s*Perplexity\s*=\s*([0-9.]+)')

# Group by step
step_readings = {}
for line in lines:
    m = val_pattern.search(line)
    if m:
        step = int(m.group(1))
        loss = float(m.group(2))
        ppl = float(m.group(3))
        if step not in step_readings:
            step_readings[step] = []
        step_readings[step].append((loss, ppl))

swiglu_data = []
silu_data = []

for step in sorted(step_readings.keys()):
    readings = step_readings[step]
    if len(readings) == 2:
        # Lower loss is swiglu, higher loss is silu
        r1, r2 = readings[0], readings[1]
        swiglu_entry = min(r1, r2, key=lambda x: x[0])
        silu_entry = max(r1, r2, key=lambda x: x[0])
        swiglu_data.append({"step": step, "val_loss": swiglu_entry[0], "val_ppl": swiglu_entry[1]})
        silu_data.append({"step": step, "val_loss": silu_entry[0], "val_ppl": silu_entry[1]})
    elif len(readings) == 1:
        # Fallback if only 1 reading
        print(f"Warning: Only 1 reading for step {step}: {readings}")

print(f"Parsed {len(swiglu_data)} steps for SwiGLU and {len(silu_data)} steps for SiLU.")
print(f"SwiGLU final: {swiglu_data[-1]}")
print(f"SiLU final: {silu_data[-1]}")

data_to_save = {
    "swiglu_baseline": swiglu_data,
    "silu_ablation": silu_data,
}

json_path = r'C:\Users\Abhinav\.gemini\antigravity\brain\c1fcb3ef-0e4c-4661-857a-d8c4f37f509b\swiglu_ablation_data.json'
with open(json_path, 'w') as f:
    json.dump(data_to_save, f, indent=2)
print("Saved data to:", json_path)

# Generate Plot
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=200)

steps_swiglu = [d["step"] for d in swiglu_data]
loss_swiglu = [d["val_loss"] for d in swiglu_data]
ppl_swiglu = [d["val_ppl"] for d in swiglu_data]

steps_silu = [d["step"] for d in silu_data]
loss_silu = [d["val_loss"] for d in silu_data]
ppl_silu = [d["val_ppl"] for d in silu_data]

# Left: Full Loss Curve (zoom in from step 100 onwards to highlight difference)
ax1.plot(steps_swiglu, loss_swiglu, color='#1f77b4', linewidth=2.2, label=r'SwiGLU Baseline ($d_{\mathrm{ff}}=1344 \approx \frac{8}{3}d_{\mathrm{model}}$, 22.70M params)')
ax1.plot(steps_silu, loss_silu, color='#d62728', linewidth=2.2, linestyle='--', label=r'SiLU Ablation ($d_{\mathrm{ff}}=2048 = 4d_{\mathrm{model}}$, 22.83M params)')
ax1.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax1.set_ylabel('Validation Cross-Entropy Loss', fontsize=12, fontweight='bold')
ax1.set_title('Validation Loss Learning Curve (TinyStories)', fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='upper right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(0.75, 2.5)

# Add final loss annotation
ax1.annotate(f'SwiGLU: {loss_swiglu[-1]:.4f}', xy=(steps_swiglu[-1], loss_swiglu[-1]),
             xytext=(steps_swiglu[-1]-500, loss_swiglu[-1]-0.15),
             arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.2, headwidth=6),
             fontweight='bold', color='#1f77b4', fontsize=10)
ax1.annotate(f'SiLU: {loss_silu[-1]:.4f} (+0.0167)', xy=(steps_silu[-1], loss_silu[-1]),
             xytext=(steps_silu[-1]-600, loss_silu[-1]+0.2),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.2, headwidth=6),
             fontweight='bold', color='#d62728', fontsize=10)

# Right: Zoom-in on Steps 500-2500 (Loss Gap)
loss_gap = [si - sw for si, sw in zip(loss_silu, loss_swiglu)]
ax2.plot(steps_swiglu, loss_gap, color='#2ca02c', linewidth=2.0, marker='o', markersize=4, label=r'Loss Penalty of Removing GLU Gating ($\Delta$ Val Loss)')
ax2.axhline(0, color='gray', linestyle=':', linewidth=1.2)
ax2.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'Loss Gap ($\mathrm{Loss}_{\mathrm{SiLU}} - \mathrm{Loss}_{\mathrm{SwiGLU}}$)', fontsize=12, fontweight='bold')
ax2.set_title(r'Empirical Gating Benefit ($\Delta \mathrm{Loss}$ over Steps)', fontsize=13, fontweight='bold', pad=10)
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_xlim(100, 2550)
ax2.set_ylim(0.005, 0.065)

plt.tight_layout()

plot_out_brain = r'C:\Users\Abhinav\.gemini\antigravity\brain\c1fcb3ef-0e4c-4661-857a-d8c4f37f509b\swiglu_ablation.png'
plt.savefig(plot_out_brain, dpi=200, bbox_inches='tight')

plot_out_repo = r'assignment1-basics\plots\swiglu_ablation.png'
os.makedirs(os.path.dirname(plot_out_repo), exist_ok=True)
plt.savefig(plot_out_repo, dpi=200, bbox_inches='tight')
print("Plot successfully saved to:")
print("  -", plot_out_brain)
print("  -", plot_out_repo)
