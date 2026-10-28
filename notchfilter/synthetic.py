"""Add synthetic periodic (sinusoidal) noise to an image, for testing and demos."""

from __future__ import annotations

import numpy as np


def add_periodic_noise(
    image: np.ndarray,
    frequencies: list[tuple[float, float]] | None = None,
    amplitude: float = 40.0,
) -> np.ndarray:
    """Add ``sum_k amplitude * sin(2*pi*(u_k*x/M + v_k*y/N))`` to the image.

    ``frequencies`` are ``(u, v)`` in cycles per image; the default gives a
    diagonal and a horizontal stripe pattern.  Returns ``uint8``.
    """
    img = np.asarray(image, dtype=float)
    m, n = img.shape[:2]
    frequencies = frequencies or [(m / 8, n / 8), (0, n / 4)]
    x, y = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    noise = sum(amplitude * np.sin(2 * np.pi * (u * x / m + v * y / n)) for u, v in frequencies)
    if img.ndim == 3:
        noise = noise[..., None]
    return np.clip(img + noise, 0, 255).astype(np.uint8)
