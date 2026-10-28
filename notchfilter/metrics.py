"""Image-quality metrics (PSNR, SSIM)."""

from __future__ import annotations

import numpy as np


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB (``inf`` for identical images)."""
    ref = np.asarray(reference, dtype=float)
    tst = np.asarray(test, dtype=float)
    if ref.shape != tst.shape:
        raise ValueError("images must have the same shape")
    mse = np.mean((ref - tst) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range**2 / mse))


def ssim(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Structural similarity (needs scikit-image; returns ``nan`` without it)."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:  # optional dependency
        return float("nan")
    channel_axis = -1 if np.ndim(reference) == 3 else None
    return float(
        structural_similarity(
            np.asarray(reference), np.asarray(test), data_range=data_range, channel_axis=channel_axis
        )
    )
