import torch
from typing import Sequence


def sample_top_p(
    probs: torch.Tensor,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Sample from a probability distribution using nucleus (top-p) sampling.

    Args:
        probs (torch.Tensor): Probability tensor of shape (batch_size, vocab_size).
        top_p (float): Nucleus sampling probability threshold (between 0.0 and 1.0).

    Returns:
        torch.Tensor: Sampled token IDs of shape (batch_size, 1).
    """
    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)

    # Sort probabilities descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens with cumulative probability above top_p (preserving the first token)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Zero out removed probabilities and renormalize
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample from the filtered distribution
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    # Map back to original vocabulary token IDs
    sampled_token = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)
    return sampled_token


def decode(
    model: torch.nn.Module,
    prompt_tokens: Sequence[int] | torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    eos_token_id: int | None = None,
    context_length: int = 256,
    device: str | torch.device = "cuda",
) -> list[int]:
    """
    Autoregressively generate tokens from a language model with temperature scaling
    and top-p (nucleus) sampling.

    Args:
        model (torch.nn.Module): Trained TransformerLM module.
        prompt_tokens: List or 1D/2D tensor of prefix token IDs.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Softmax temperature (lower = more deterministic, higher = more creative).
        top_p (float): Nucleus sampling threshold (0.0 < top_p <= 1.0).
        eos_token_id (int, optional): End-of-sequence token ID. Generation stops if encountered.
        context_length (int): Maximum context window length supported by the model.
        device: Device to run generation on.

    Returns:
        list[int]: Full sequence of token IDs including prompt and generated tokens.
    """
    model.eval()

    if isinstance(prompt_tokens, (list, tuple)):
        input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    elif isinstance(prompt_tokens, torch.Tensor):
        input_ids = prompt_tokens.to(device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported prompt_tokens type: {type(prompt_tokens)}")

    generated_tokens = list(input_ids[0].tolist())

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop input to context length if needed
            cond_input = input_ids[:, -context_length:]

            # Forward pass: compute logits for all positions
            logits = model(cond_input)

            # Extract logits for the last token position: shape (batch_size, vocab_size)
            next_token_logits = logits[:, -1, :]

            # Temperature scaling & sampling
            if temperature == 0.0:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / max(temperature, 1e-5)
                # Numerically stable softmax
                scaled_logits = scaled_logits - torch.max(scaled_logits, dim=-1, keepdim=True).values
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = sample_top_p(probs, top_p=top_p)

            token_id = int(next_token.item())
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if end-of-sequence token is generated
            if eos_token_id is not None and token_id == eos_token_id:
                break

    return generated_tokens
