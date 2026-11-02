import math
import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.nn_utils import softmax


class Linear(nn.Module):
    """
    Linear layer without bias: Y = X W^T

    Args:
        in_features (int): Dimensionality of input features (d_in).
        out_features (int): Dimensionality of output features (d_out).
        device (torch.device | None): Device to initialize parameters on.
        dtype (torch.dtype | None): Data type of parameters.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight matrix W of shape (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # Truncated normal distribution: N(0, 2 / (in_features + out_features)) truncated at [-3*sigma, 3*sigma]
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies linear transformation: Y = X W^T

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return x @ self.weight.T


class Embedding(nn.Module):
    """
    Embedding lookup table of shape (num_embeddings, embedding_dim).

    Args:
        num_embeddings (int): Total vocabulary size.
        embedding_dim (int): Dimensionality of embedding vectors (d_model).
        device (torch.device | None): Device to initialize parameters on.
        dtype (torch.dtype | None): Data type of parameters.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embedding table matrix of shape (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # Truncated normal initialization: N(0, 1) truncated at [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs embedding lookup for arbitrary batch-shaped token ID tensors.

        Args:
            token_ids (torch.Tensor): Integer token IDs of shape (...).
        Returns:
            torch.Tensor: Looked-up embeddings of shape (..., embedding_dim).
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        d_model (int): Hidden dimension of the model.
        eps (float): Numerical stability constant (default: 1e-5).
        device (torch.device | None): Device to initialize the parameters on.
        dtype (torch.dtype | None): Data type of parameters.
    """
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None= None,
            dtype: torch.dtype | None = None,
    ):

        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameter g of shape (d_model,), initialized to ones
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMSNorm to the input tensor across the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).
        Returns:
            torch.Tensor: Normalized tensor of shape (..., d_model) in the original dtype.
        """
        in_dtype = x.dtype
        # Upcast to float 32 to prevent overflow when sqauring
        x_fp32 = x.to(torch.float32)

        # RMS(x) = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale by gain parameter
        normed = (x_fp32 / rms) * self.weight.to(torch.float32)

        # Downcast back to original input dtype
        return normed.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SILU (Swish) activation function: x * sigmoid (x)
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network: W2(SiLU(W1(x)) * W3(x))

    Args:
        d_model (int): Dimensionality of input and output features.
        d_ff (int): Dimensionality of inner feed-forward layer.
        device (torch.device | None): Device to initialize parameters on.
        dtype (torch.dtype | None): Data type pf parameters.
    """
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Three linear projections (no bias)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).
        Returns:
        torch.Tensor: Output tensor of shape (..., d_model).
        """
        # Element-wise product of activated gate branch (w1) and up-projected branch (w3)
        gated = silu(self.w1(x)) * self.w3(x)
        # Down-project back to d_model
        return self.w2(gated)


class SiLUFFN(nn.Module):
    """
    Standard Feed-Forward Network with SiLU activation (ungated):
    FFN_SiLU(x) = W2(SiLU(W1(x)))

    Args:
        d_model (int): Dimensionality of input and output features.
        d_ff (int): Dimensionality of inner feed-forward layer (typically 4 * d_model).
        device (torch.device | None): Device to initialize parameters on.
        dtype (torch.dtype | None): Data type of parameters.
    """
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Two linear projections (no bias)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).
        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        return self.w2(silu(self.w1(x)))


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Args:
        thetha (float): Base frequency parameter (typically 10000.0).
        d_k (int): Embedding dimension of query/key vectors (must be even).
        max_seq_len (int): Maixum sequence length to precompute frequencies for.
        device (torch.device | None): Device to initialize precomputed buffers on.
    """
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 1. Frequency bands for each 2D pair: theta^(-2k / d_k)
        indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (indices / d_k)) # Shape: (d_k // 2,)

        # 2. Precompute grid of angles for position [0, 1, ..., max_seq_len - 1]
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(t, freqs) # Shape: (max_seq_len, d_k // 2)

        # 3. Store sin and cos values in non-persistent buffers (no gradients)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)

    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Applies 2D rotation to pairs of elements in query or key tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Integer positions of shape (..., seq_len).

        Returns:
            torch.Tensor: RoPE-rotated tensor of shape (..., seq_len, d_k).
        """
        seq_len = x.shape[-2]
        # If no positions are provided, default to [0, 1, ..., seq_len - 1]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        # Look up precomputed cos and sin for the requested token positions
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        # Reshape x into 2D pairs: (..., seq_len, d_k // 2, 2)
        x_paired = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x0 = x_paired[..., 0]
        x1 = x_paired[..., 1]
        # 2D rotation formula
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x0 * sin + x1 * cos
        # Reconstruct original tensor shape (..., seq_len, d_k)
        x_rot = torch.stack([x0_rot, x1_rot], dim=-1)
        return x_rot.flatten(-2)

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention.

    Args:
        Q (torch.Tensor): Query tensor of shape (..., queries, d_k).
        K (torch.Tensor): Key tensor of shape (..., keys, d_k).
        V (torch.Tensor): Value tensor of shape (..., keys, d_v)
        mask (torch.Tensor | None): Boolean mask of shape (..., queries, keys), where True indicates position to attend to.

    Returns:
        torch.Tensor: Output tensor of shape (..., queries, d_v).
    """
    d_k = Q.shape[-1]

    # 1. Compute raw attention scores (..., queries, d_k) @ (..., d_k, keys)-> (..., queries, keys)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. Apply boolean mask if provided: set False positions to -inf
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 3. Compute normalized attention weights along the keys dimension (-1)
    attn_weights = softmax(scores, dim=-1)

    # Handle edge cases where an entire row is masked out (results in 0/0 = NaN)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # 4. Compute weighted sum of values: (..., queries, keys) @ (..., keys, d_v) -> (..., queries, d_v)
    return attn_weights @ V


class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention with optional RoPE.

    Args:
        d_model (int): Total dimensionality of input and output.
        num_heads (int): Number of attention heads (d_model must be divisible by num_heads).
        max_seq_len (int | None): Max context length for RoPE caching.
        theta (float | None): RoPE base frequency parameter.
        rope (RotaryPositionalEmbedding | None): Pre-instantiated RoPE module.
        device (torch.device | None): Device to initialize parameters on.
        dtype (torch.dtype | None): Data type of parameters.
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int | None = None,
            theta: float | None = None,
            rope: RotaryPositionEmbedding | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        # 4 linear projections (no bias)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Setup RoPE if enabled
        if not use_rope:
            self.rope = None
        elif rope is not None:
            self.rope = rope
        elif theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).
            token_positions (torch.Tensor | None): Optional positions of shape (..., seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        seq_len = x.shape[-2]

        # 1. Linear projections: (..., seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split into multiple heads: (..., num_heads, seq_len, d_k)
        q = rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # 3. Apply RoPE to Query and Key if enabled
        if self.rope is not None:
            if token_positions is not None:
                pos = token_positions
                # Ensure pos has batch dimension matching q
                if pos.ndim < q.ndim - 2:
                    pos = pos.unsqueeze(0).expand(q.shape[0], -1)
                # Inseert head dimension: (..., 1, seq_len) to broadcast across all heads
                pos = pos.unsqueeze(-2)
            else:
                pos = None
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # 4. Construct lower-traingular causal mask: (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        # 5. Scaled dot-product attention
        attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # 6. Concatenate all heads: (..., seq_len, d_model)
        out = rearrange(attn_out, "... h seq d -> ... seq (h d)")

        # 7. Final output projection
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block with RMSNorm, Multi-Head Self-Attention (RoPE), and SwiGLU.

    Args:
        d_model (int): Hidden dimensionality of the model.
        num_heads(int): Number of attention heads.
        d_ff (int): Dimensionality of the inner feed-forward layer.
        max_seq_len (int): Maximum sequence length for RoPE.
        theta (float): Base frequency parameter for RoPE.
        rope (RotaryPositionEmbedding | None): Optional shared RoPE module.
        device (torch.device | None): Device to initialize parameters on.
        dtype (torch.dtype | None): Data type of parameters.
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            rope: RotaryPositionEmbedding | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            use_rmsnorm: bool = True,
            norm_position: str = "pre",
            use_rope: bool = True,
            ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rmsnorm = use_rmsnorm
        self.norm_position = norm_position
        self.use_rope = use_rope
        self.ffn_type = ffn_type

        # 1. Pre-attention RMSNorm & Causal Multi-Head Self-Attention
        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        else:
            self.ln1 = nn.Identity()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            rope=rope if use_rope else None,
            device=device,
            dtype=dtype,
            use_rope=use_rope,
        )

        # 2. Pre-FFN RMSNorm & Feed-Forward Network (SwiGLU or SiLUFFN)
        if use_rmsnorm:
            self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        else:
            self.ln2 = nn.Identity()
        if ffn_type == "silu":
            self.ffn = SiLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input activations of shape (batch, seq_len, d_model).
            token_positions (torch.Tensor | None): Token positions indices of shape (batch, seq_len).

        Returns:
            torch.Tensor: Output activations of shape (batch, seq_len, d_model).
        """
        if self.norm_position == "post":
            # Post-Norm (Eq. 27 & 28 in Stanford CS336 handout):
            # z = RMSNorm(x + MultiHeadSelfAttention(x))
            # y = RMSNorm(z + FFN(z))
            z = self.ln1(x + self.attn(x, token_positions=token_positions))
            y = self.ln2(z + self.ffn(z))
            return y
        else:
            # Pre-Norm:
            # 1. Step 1: Pre-Norm Attention with Residual Connection
            x = x + self.attn(self.ln1(x), token_positions=token_positions)

            # 2. Step 2: Pre-Norm SwiGLU with Residual Connection
            x = x + self.ffn(self.ln2(x))

            return x


class TransformerLM(nn.Module):
    """
    Full Autoregressive Transformer Language Model.

    Args:
        vocab_size (int): Size of the token vocabulary.
        context_length (int): Maximum sequence / context length.
        d_model (int): Hidden dimension of the model.
        num_layers (int): Number of Transformer blocks.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the inner feed-forward layer.
        rope_theta (float): Base frequency parameter for RoPE (default: 10000.0)
        device (torch.device | None): Device to initialize parameters on.
        dtype (torch.dtype | None): Data type of parameters.
    """
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float = 10000.0,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            use_rmsnorm: bool = True,
            norm_position: str = "pre",
            use_rope: bool = True,
            ffn_type: str = "swiglu",
            tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rmsnorm = use_rmsnorm
        self.norm_position = norm_position
        self.use_rope = use_rope
        self.ffn_type = ffn_type
        self.tie_weights = tie_weights

        # 1. Token embedding lookup table
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # 2. Shared Rotary Position Embedding module (reuses cached sin/cos across all layers)
        if use_rope:
            d_k = d_model // num_heads
            self.rope = RotaryPositionEmbedding(
                theta=rope_theta,
                d_k=d_k,
                max_seq_len=context_length,
                device=device,
            )
        else:
            self.rope = None

        # 3. Stack of N Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                rope=self.rope,
                device=device,
                dtype=dtype,
                use_rmsnorm=use_rmsnorm,
                norm_position=norm_position,
                use_rope=use_rope,
                ffn_type=ffn_type,
            )
            for _ in range(num_layers)
        ])

        # 4. Final RMSNorm layer
        if norm_position == "post":
            # In post-norm, the final block output is already normalized by ln2
            self.ln_final = nn.Identity()
        elif use_rmsnorm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()

        # 5. Output linear projection to vocabulary logits
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )
        if tie_weights:
            self.lm_head.weight = self.token_embeddings.weight
            # Properly scale the shared weight matrix so output logits do not explode
            std = math.sqrt(2.0 / (d_model + vocab_size))
            nn.init.trunc_normal_(self.token_embeddings.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(
            self,
            in_indices: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            in_indices (torch.Tesnor): Integer token IDs of shape (batch, seq_len).
            token_positions (torch.Tensor | None): Optional position IDs of shape (batch, seq_len).
        Returns:
            torch.Tensor: Unnormalized next-token logits of shape (batch, seq_len, vocab_size).
        """
        # 1. Look up token embeddings
        x = self.token_embeddings(in_indices)

        # 2. Sequentially pass through all Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # 3. Final normalization
        x = self.ln_final(x)

        # 4. Project to vocabulary logits
        logits = self.lm_head(x)

        return logits
