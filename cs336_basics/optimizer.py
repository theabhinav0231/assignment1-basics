import math
import torch
from torch.optim import Optimizer
from collections.abc import Callable
from typing import Optional

class AdamW(Optimizer):
    """
    AdamW Optimizer (Decoupled Weight Decay).

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate alpha (default: 1e-3).
        betas (tuple[float, float]): Coefficients (beta1, beta2) for computing
        running averages (default: (0.9, 0.999)).
        eps (float): Term added to denominator for numerical stability (default: 1e-8).
        weight_decay (float): Decoupled wegith decay coefficient lambda (default: 0.01).
    """
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01,):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state for this parameter on first steo
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # 1. Apply decoupled weight decay:
                # theta = theta - alpha * lambda * theta
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # 2. Update biased 1st moment: m = beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 3. Update biased 2nd moment: v = beta2 * v + (1 - beta2) * g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 4. Compute bias-corrected learning rate step:
                # alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                alpha_t = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

                # 5. Apply moment-adjusted parameter update:
                # theta = theta - alpha_t * (m / (sqrt(v) + eps))
                p.data.addcdiv_(m, torch.sqrt(v) + eps, value=-alpha_t)

        return loss


def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup and minimum lr floor.

    Args:
        it (int): Current iteration step t.
        max_learning_rate (float): Peak learning rate alpha_max.
        min_learning_rate (float): Floor/final learning rate alpha_min.
        warmup_iters (int): Number of warmup steps T_w.
        cosine_cycle_iters (int): Step T_c when cosine annealing reaches alpha_min.
    Returns:
        float: Computed learning rate for iteration it.
    """
    # 1. Linear warmup phase: 0 -> max_learning_rate
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    # 2. Cosine decay phase: max_learning_rate -> min_learning_rate
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1.0 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    
    # 3. Post-annealing phase: clamped at min_learning_rate
    else:
        return min_learning_rate