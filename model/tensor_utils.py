"""Tensor conversion at NumPy interoperability boundaries."""

import torch


def tensor_to_numpy(tensor: torch.Tensor):
    """Detach to CPU, promoting only BF16 to a NumPy-supported dtype."""
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy()
