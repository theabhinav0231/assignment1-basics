import os
import typing
import torch


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    Saves model state, optimizer state and current training iteration to a
    file or file-like object.

    Args:
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer being used.
        iteration (int): Current iteration step number.
        out: File path or file-like object where the checkpoint should be written.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> int:
    """
    Loads model state and optimizer state from a checkpoint and returns the saved iteration.

    Args:
        src: File path or file-like object containing the checkpoint.
        model (torch.nn.Module): Model to restore weights into.
        optimizer (torch.optim.Optimizer): Optimizer to restore state into.

    Returns:
        int: The iteration number saved in the checkpoint.
    """
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
