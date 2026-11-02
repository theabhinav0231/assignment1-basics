import numpy as np
import numpy.typing as npt
import torch

def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a batch of input sequences and next-token targets from a 
    1D tokenized dataset.

    Args:
        dataset (np.ndarray): 1D array of integer token IDs.
        batch_size (int): Number of sequences to sample in the batch.
        context_length (int): Length of each sequence (m).
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Pair of (inputs, targets) of shape (batch_size, context_length).
    """
    # 1. Compute valid upper bound for start indices
    max_start = len(dataset) -  context_length

    # 2. Sample random start positions uniformly across the dataset
    start_indices = np.random.randint(0, max_start, size=batch_size)

    # 3. Slice each sequence: inputs at [i: i + m] and targets at [i + 1: i + m + 1]
    inputs = np.stack([dataset[i : i + context_length] for i in start_indices])
    targets = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices])

    # 4. Convert to PyTorch integer tensors on the requested device
    x = torch.tensor(inputs, dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)

    return x, y