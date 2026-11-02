def compute_accounting(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    token_emb = vocab_size * d_model
    attn_params = 4 * d_model * d_model
    ln1_params = d_model
    ln2_params = d_model
    ffn_params = 3 * d_model * d_ff
    layer_params = ln1_params + attn_params + ln2_params + ffn_params
    total_layers_params = num_layers * layer_params
    ln_final_params = d_model
    lm_head_params = d_model * vocab_size
    
    total_params = token_emb + total_layers_params + ln_final_params + lm_head_params
    memory_bytes_fp32 = total_params * 4
    memory_gb_fp32 = memory_bytes_fp32 / (1024**3)
    
    T = context_length
    attn_proj_flops = 4 * (2 * T * d_model * d_model)
    attn_core_flops = 2 * (2 * T * T * d_model)
    ffn_layer_flops = 3 * (2 * T * d_model * d_ff)
    
    layers_attn_proj_flops = num_layers * attn_proj_flops
    layers_attn_core_flops = num_layers * attn_core_flops
    layers_ffn_flops = num_layers * ffn_layer_flops
    total_layers_flops = layers_attn_proj_flops + layers_attn_core_flops + layers_ffn_flops
    lm_head_flops = 2 * T * d_model * vocab_size
    
    total_flops = total_layers_flops + lm_head_flops
    
    return {
        'total_params': total_params,
        'memory_gb': memory_gb_fp32,
        'total_flops': total_flops,
        'attn_proj_flops': layers_attn_proj_flops,
        'attn_core_flops': layers_attn_core_flops,
        'ffn_flops': layers_ffn_flops,
        'lm_head_flops': lm_head_flops,
        'pct_attn_proj': 100 * layers_attn_proj_flops / total_flops,
        'pct_attn_core': 100 * layers_attn_core_flops / total_flops,
        'pct_ffn': 100 * layers_ffn_flops / total_flops,
        'pct_lm_head': 100 * lm_head_flops / total_flops,
    }

models = {
    'GPT-2 Small': (50257, 1024, 12, 768, 12, 2048),
    'GPT-2 Medium': (50257, 1024, 24, 1024, 16, 2752),
    'GPT-2 Large': (50257, 1024, 36, 1280, 20, 3456),
    'GPT-2 XL': (50257, 1024, 48, 1600, 25, 4288),
    'GPT-2 XL (16k context)': (50257, 16384, 48, 1600, 25, 4288),
}

for name, (v, ctx, L, d, h, d_ff) in models.items():
    res = compute_accounting(v, ctx, L, d, h, d_ff)
    p = res['total_params']
    m = res['memory_gb']
    tf = res['total_flops']
    print(f'=== {name} ===')
    print(f'Params: {p:,}')
    print(f'Memory (FP32): {m:.3f} GB')
    print(f'Total FLOPs: {tf:.4e}')
    print(f'  Attn Projections: {res["pct_attn_proj"]:.2f}% ({res["attn_proj_flops"]:.3e})')
    print(f'  Attn Core: {res["pct_attn_core"]:.2f}% ({res["attn_core_flops"]:.3e})')
    print(f'  FFN (SwiGLU): {res["pct_ffn"]:.2f}% ({res["ffn_flops"]:.3e})')
    print(f'  LM Head: {res["pct_lm_head"]:.2f}% ({res["lm_head_flops"]:.3e})\n')
