"""Spectrum display and automatic detection of periodic-noise spikes."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def to_gray(image: np.ndarray) -> np.ndarray:
    """Luma conversion for ``(H, W, C)`` arrays; 2-D arrays pass through."""
    img = np.asarray(image, dtype=float)
    if img.ndim == 2:
        return img
    if img.shape[2] >= 3:
        return img[..., :3] @ np.array([0.299, 0.587, 0.114])
    return img[..., 0]


def magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """Centred log-magnitude spectrum, scaled to ``uint8`` for display."""
    f = np.fft.fftshift(np.fft.fft2(to_gray(image)))
    mag = np.log1p(np.abs(f))
    mag = (mag - mag.min()) / max(np.ptp(mag), 1e-12)
    return (mag * 255).astype(np.uint8)


def detect_peaks(
    image: np.ndarray,
    max_peaks: int = 4,
    dc_radius: float = 10.0,
    min_distance: int = 5,
    min_prominence: float = 1.0,
) -> list[tuple[int, int]]:
    """Find the strongest spikes of periodic noise in the spectrum.

    The log-magnitude spectrum is compared with a median-filtered copy of
    itself (the smooth "background"); spikes are local maxima that stick out
    above that background.  Only one point per conjugate pair is returned
    (the upper half-plane), which is what :func:`notchfilter.notch_mask` expects.

    Parameters
    ----------
    max_peaks:      maximum number of pairs to return.
    dc_radius:      ignore everything this close to the DC term.
    min_distance:   minimum separation between two detected peaks (pixels).
    min_prominence: minimum height above background, in units of the
                    standard deviation of the residual spectrum.

    Returns
    -------
    List of ``(row, col)`` points in the centred spectrum, strongest first.
    """
    gray = to_gray(image)
    m, n = gray.shape
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    residual = mag - ndimage.median_filter(mag, size=15, mode="nearest")

    cu, cv = m // 2, n // 2
    rr, cc = np.ogrid[:m, :n]
    residual[np.hypot(rr - cu, cc - cv) <= dc_radius] = -np.inf
    # keep one of each conjugate pair: rows above centre (and left half of the centre row)
    residual[(rr > cu) | ((rr == cu) & (cc >= cv))] = -np.inf

    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return []
    threshold = finite.mean() + min_prominence * finite.std() * 3
    local_max = residual == ndimage.maximum_filter(residual, size=2 * min_distance + 1)
    candidates = np.argwhere(local_max & (residual > threshold))
    order = np.argsort(residual[tuple(candidates.T)])[::-1]
    return [(int(r), int(c)) for r, c in candidates[order][:max_peaks]]
