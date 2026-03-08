"""GPU backend utility – uses CuPy (CUDA) when available, falls back to NumPy."""

import numpy as _numpy

try:
    import cupy as _cupy
    xp = _cupy
    HAS_CUDA = True
except ImportError:
    xp = _numpy
    HAS_CUDA = False


def to_numpy(arr):
    """Convert array to numpy (handles CuPy → NumPy conversion)."""
    if HAS_CUDA and isinstance(arr, _cupy.ndarray):
        return arr.get()
    return _numpy.asarray(arr)


def get_xp(use_cuda=True):
    """Return cupy if CUDA is available and requested, else numpy."""
    if use_cuda and HAS_CUDA:
        return _cupy
    return _numpy
