"""Useful factories to create matching data for an operator."""

import numpy as np

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False

TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = torch.cuda.is_available()


def image_from_op(operator):
    """Generate a random image."""
    if operator.smaps is None:
        img = np.random.randn(operator.n_coils, *operator.shape).astype(
            operator.cpx_dtype
        )
    elif operator.smaps is not None and operator.n_coils > 1:
        img = np.random.randn(*operator.shape).astype(operator.cpx_dtype)

    img += 1j * np.random.randn(*img.shape).astype(operator.cpx_dtype)
    return img


def kspace_from_op(operator):
    """Generate a random kspace data."""
    kspace = (1j * np.random.randn(operator.n_coils, operator.n_samples)).astype(
        operator.cpx_dtype
    )
    kspace += np.random.randn(operator.n_coils, operator.n_samples).astype(
        operator.cpx_dtype
    )
    return kspace


def to_interface(data, interface):
    """Make DATA an array from INTERFACE."""
    if interface == "cupy":
        return cp.array(data)
    elif interface == "torch":
        return torch.from_numpy(data).to("cuda")
    return data


def from_interface(data, interface):
    """Get DATA from INTERFACE as a numpy array."""
    if interface == "cupy":
        return data.get()
    elif interface == "torch":
        return data.to("cpu").numpy()
    return data
