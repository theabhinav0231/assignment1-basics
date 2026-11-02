from collections.abc import Iterable
import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor of arbitrary shape.
        dim (int): Dimension along which Softmax will be computed.
    Returns:
        torch.Tensor: Probability distribution tensor with the same shape as x.
    """
    # 1. Find the max value along the specified dimension
    # keepdim = True preserves the dimension for safe broadcasting
    x_max = torch.max(x, dim=dim, keepdim=True).values

    # 2. Subtract max for numerical stability, then compute exp
    exp_x = torch.exp(x - x_max)

    # 3. Normalize by the sum of exponentials
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Numerically stable Cross-Entropy loss.

    Args:
        inputs (torch.Tensor): Unnormalized predicted logits of shape (..., vocab_size)
        targets (torch.Tensor): Ground-truth integer token IDs of shape (...).
    Returns:
        torch.Tensor: Scalar loss averaged across all batch and sequence dimensions.
    """
    # 1. Max subtraction along vocab dimension (-1) for stability
    c = torch.max(inputs, dim=-1, keepdim=True).values

    # 2. Log-Sum-Exp computation: log(sum(exp(inputs - c)))
    shifted_inputs = inputs - c
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_inputs), dim=-1, keepdim=True))

    # 3. Gather the shifted logit for the correct target class: (inputs[..., target] - c)
    target_shifted_logit = torch.gather(shifted_inputs, dim=-1, index=targets.unsqueeze(-1))

    # 4. Loss per sample: log_sum_exp - (inputs[target] - c)
    loss = log_sum_exp - target_shifted_logit

    # 5. Return scalar mean across all batch/sequence dimensions
    return loss.mean()


def clip_gradient(
    parameters: Iterable[torch.nn.Parameter],
    max_12_norm: float | None = None,
    max_l2_norm: float | None = None,
    eps: float = 1e-6,
) -> None:
    """
    Clips the gradients of an iterable of parameters to a maximum L2-norm in-place.

    Args:
        parameters: Iterable of parameters whose gradients will be clipped.
        max_12_norm / max_l2_norm (float): Maximum allowed L2-norm M.
        eps (float): Small epsilon value for numerical stability (default: 1e-6).
    """
    threshold = max_l2_norm if max_l2_norm is not None else max_12_norm
    if threshold is None:
        raise ValueError("Either max_l2_norm or max_12_norm must be provided.")

    # 1. Filter parameters that have gradients
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return

    # 2. Compute the global L2-norm across all parameter gradients
    total_norm = torch.sqrt(sum(torch.sum(p.grad.data ** 2) for p in params_with_grad))

    # 3. If total_norm exceeds threshold, scale down in-place
    if total_norm > threshold:
        scale = threshold / (total_norm + eps)
        for p in params_with_grad:
            p.grad.data.mul_(scale)
