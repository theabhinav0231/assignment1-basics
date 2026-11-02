# CS336 Spring 2025 Assignment 1: Language Modeling from Scratch

This repository contains a complete, from-scratch PyTorch implementation of a modern Transformer language model (similar to LLaMA / Mistral) developed for Stanford CS336: Language Modeling from Scratch (Spring 2025).

The submission includes all core neural network primitives, training harness, BPE tokenizer, decoding algorithms, systematic ablation studies, and a final 45-minute compute-budget pretraining run on OpenWebText yielding a validation loss of **2.1215** (Perplexity **8.34**).

The complete academic report with all mathematical proofs, architectural derivations, and empirical curves is compiled in [writeup.pdf](./writeup.pdf) (18 pages).

---

## Repository Structure

```text
assignment1-basics/
├── cs336_basics/             # Core assignment implementation
│   ├── bpe.py                # Byte-level BPE tokenizer training
│   ├── decode.py             # Temperature sampling, top-p (nucleus) decoding
│   ├── model.py              # Transformer LM, RMSNorm, RoPE, SwiGLU, Attention
│   ├── nn_utils.py           # Numerically stable Cross-Entropy & Gradient Clipping
│   ├── optimizer.py          # AdamW optimizer & Cosine Warmup LR Schedule
│   └── tokenizer.py          # BPE Tokenizer class with regex splitting
├── data/                     # Tokenizer vocabularies and merge tables
├── plots/                    # High-resolution figures and JSON metric logs
│   ├── prenorm_ablation.png  # Pre-LN vs. Post-LN comparison
│   ├── nope_ablation.png     # Rotary Positional Embeddings vs. NoPE
│   ├── swiglu_ablation.png   # SwiGLU vs. GELU FFN parameter-matched comparison
│   ├── owt_experiment.png    # OpenWebText baseline training curves
│   ├── owt_lr_comparison.png # Learning rate sensitivity (η = 0.0020 vs 0.0025)
│   └── owt_45min_pretrain.png# 45-minute leaderboard pretraining curve
├── scripts/
│   ├── modal/                # Modal cloud GPU execution scripts (H100 / A100)
│   │   ├── train_bpe_modal.py
│   │   ├── tokenize_dataset_modal.py
│   │   ├── pre_norm_ablation_modal.py
│   │   ├── nope_ablation_modal.py
│   │   ├── swiglu_ablation_modal.py
│   │   ├── owt_main_experiment_modal.py
│   │   ├── owt_lr_comparison_modal.py
│   │   └── owt_45min_pretrain_modal.py
│   └── plotting/             # Standalone Python scripts to reproduce all paper plots
│       ├── generate_prenorm_ablation_plot.py
│       ├── generate_nope_plot.py
│       ├── generate_swiglu_plot.py
│       ├── generate_owt_plot.py
│       ├── plot_lr_comparison.py
│       └── plot_45min_pretrain.py
├── tests/                    # Stanford CS336 test suite & adapters
│   ├── adapters.py           # Harness connecting implementations to tests
│   ├── test_bpe.py
│   ├── test_decode.py
│   ├── test_model.py
│   └── test_nn_utils.py
├── generate_writeup_pdf.py   # Vector PDF generation script for writeup.pdf
├── make_submission.sh        # Official Stanford CS336 submission packager
├── train.py                  # Modular training script with bfloat16 & weight tying
├── pyproject.toml            # Project dependencies and packaging metadata
└── writeup.pdf               # Complete 18-page formal assignment report
```

---

## Key Experimental Results

All experiments were executed on NVIDIA H100 GPUs using high-precision telemetry.

### 1. Architecture Ablations (TinyStories, 5,000 steps)
- **Normalization (Pre-LN vs. Post-LN):** Pre-LN achieves $\mathcal{L}_{\text{val}} = 1.4880$ (PPL $4.43$) vs Post-LN $\mathcal{L}_{\text{val}} = 1.5471$ (PPL $4.70$), demonstrating strictly superior gradient stability and convergence.
- **Positional Encoding (RoPE vs. NoPE):** RoPE improves validation loss by **0.2520 nats** ($\mathcal{L}_{\text{val}} = 1.4880$ vs $1.7400$, PPL $4.43$ vs $5.70$).
- **Activation (SwiGLU vs. GELU):** Parameter-matched SwiGLU ($d_{\text{ffn}} = \lfloor \frac{8}{3} d_{\text{model}} \rfloor$) achieves $\mathcal{L}_{\text{val}} = 1.4593$ (PPL $4.30$) vs GELU ($d_{\text{ffn}} = 4 d_{\text{model}}$) $\mathcal{L}_{\text{val}} = 1.4880$ (PPL $4.43$), a statistically significant **0.0287 nats** improvement.

### 2. Final 45-Minute Pretraining Run (`Problem (leaderboard)`)
- **Dataset:** OpenWebText ($V = 32,000$ BPE vocab).
- **Compute Budget:** 45 minutes wall-clock time on a single NVIDIA H100-SXM5-80GB GPU.
- **Optimizations Applied:**
  * **Precision:** Native `bfloat16` mixed precision.
  * **Weight Tying:** Tied embedding and LM output projection with variance-scaled initialization ($\sigma = \sqrt{2 / (d_{\text{model}} + V)}$).
  * **Batch Sizing:** Global batch size 256 sequences of context length 256 ($65,536\text{ tokens/step}$), maximizing H100 tensor core occupancy at $42.7\text{ GB}$ peak VRAM with zero gradient accumulation overhead.
  * **Learning Rate Schedule:** Peak $\eta_{\max} = 0.0025$, 500-step linear warmup, cosine decay to $\eta_{\min} = 2.5 \times 10^{-4}$ at 10,000 steps.
- **Throughput:** $286,298\text{ tokens/sec}$ ($4.37\text{ steps/sec}$).
- **Total Compute Delivered:** $655,360,000\text{ tokens}$ in **38.15 minutes**.
- **Final Validation Loss:** **2.1215** (Perplexity **8.34**), beating the benchmark target ($5.0$ nats) by **-2.8785 nats**.

---

## Getting Started

### Environment Setup
We manage environments with `uv`:
```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Synchronize virtual environment
uv sync
```

### Running Unit Tests
To verify all neural network primitives, BPE tokenizers, and optimizers:
```bash
uv run pytest
```

### Running Training Locally
To launch a training run on a local GPU or CPU:
```bash
python train.py \
    --train_data data/owt_train.bin \
    --val_data data/owt_valid.bin \
    --vocab_size 32000 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1365 \
    --batch_size 64 \
    --lr 0.0025 \
    --max_steps 5000 \
    --tie_weights \
    --precision bfloat16
```

### Cloud GPU Execution via Modal
Modal scripts in `scripts/modal/` run self-contained on remote cloud GPUs. For example, to reproduce the 45-minute leaderboard pretraining run:
```bash
modal run scripts/modal/owt_45min_pretrain_modal.py
```

### Regenerating Plots and PDF Writeup
All figures can be recreated using the scripts in `scripts/plotting/`:
```bash
python scripts/plotting/plot_45min_pretrain.py
```

To compile the complete 18-page formal report:
```bash
python generate_writeup_pdf.py
```

---

## Submission Packaging
To create the official submission archive according to Stanford CS336 specifications:
```bash
bash make_submission.sh
```
This produces `cs336-spring2025-assignment-1-submission.zip`, excluding temporary logs, caches, and weights while preserving code and the final `writeup.pdf`.
