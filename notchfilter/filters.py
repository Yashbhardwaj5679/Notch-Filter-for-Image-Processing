"""Frequency-domain notch reject filters.

Conventions
-----------
* Spectra are *centred* (``np.fft.fftshift``), so the DC term sits at
  ``(M // 2, N // 2)``.
* A notch is specified by one point ``(row, col)`` in that centred spectrum.
  Its conjugate-symmetric partner is added automatically, so the filter
  always removes a *pair* of spikes ``(u_k, v_k)`` and ``(-u_k, -v_k)``.
* ``d0`` is the notch radius in spectrum pixels.

Equations (after Gonzalez & Woods, *Digital Image Processing*, 4th ed.)::

    Ideal        H = 0 if D_k <= D0 or D_-k <= D0, else 1
    Butterworth  H = prod_k  1/(1 + (D0/D_k)**(2n)) * 1/(1 + (D0/D_-k)**(2n))
    Gaussian     H = prod_k (1 - exp(-D_k**2 / (2 D0**2))) * (1 - exp(-D_-k**2 / (2 D0**2)))

where D_k / D_-k are the distances to the spike (u_k, v_k) and its mirror
(-u_k, -v_k).  Every filter therefore has the *same* meaning of ``d0``: a
radius in spectrum pixels around each spike.  (The alternative form
``D0**2 / (D_k * D_-k)`` makes the notch shrink as the spike moves away from
the centre, so the radius would not be comparable between spikes.)
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

FILTER_KINDS = ("ideal", "butterworth", "gaussian")
_EPS = 1e-12


def _as_points(points: Iterable[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(list(points), dtype=float).reshape(-1, 2)
    return pts


def notch_mask(
    shape: tuple[int, int],
    points: Iterable[Sequence[float]],
    kind: str = "butterworth",
    d0: float = 10.0,
    order: int = 2,
) -> np.ndarray:
    """Build a notch *reject* mask for a centred spectrum.

    Parameters
    ----------
    shape:  ``(M, N)`` of the image / spectrum.
    points: iterable of ``(row, col)`` noise-spike locations in the centred
            spectrum (one point per conjugate pair is enough).
    kind:   ``"ideal"``, ``"butterworth"`` or ``"gaussian"``.
    d0:     notch radius in pixels (must be > 0).
    order:  Butterworth order ``n`` (ignored by the other filters).

    Returns
    -------
    ``float64`` array of shape ``shape`` with values in ``[0, 1]``.
    """
    kind = kind.lower()
    if kind not in FILTER_KINDS:
        raise ValueError(f"kind must be one of {FILTER_KINDS}, got {kind!r}")
    if d0 <= 0:
        raise ValueError("d0 (notch radius) must be positive")
    if kind == "butterworth" and order < 1:
        raise ValueError("Butterworth order must be >= 1")

    m, n = shape
    cu, cv = m // 2, n // 2  # DC position after fftshift
    u = np.arange(m, dtype=float)[:, None] - cu
    v = np.arange(n, dtype=float)[None, :] - cv

    mask = np.ones((m, n), dtype=float)
    for r, c in _as_points(points):
        du, dv = r - cu, c - cv
        d_pos = np.hypot(u - du, v - dv)  # distance to  (u_k, v_k)
        d_neg = np.hypot(u + du, v + dv)  # distance to (-u_k, -v_k)
        if kind == "ideal":
            mask *= ~((d_pos <= d0) | (d_neg <= d0))
        elif kind == "butterworth":
            for d in (d_pos, d_neg):
                mask *= 1.0 / (1.0 + (d0 / np.maximum(d, _EPS)) ** (2 * order))
        else:  # gaussian
            for d in (d_pos, d_neg):
                mask *= 1.0 - np.exp(-(d**2) / (2.0 * d0**2))
    return mask


def _filter_channel(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(channel.astype(float)))
    return np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real


def apply_notch_filter(
    image: np.ndarray,
    points: Iterable[Sequence[float]],
    kind: str = "butterworth",
    d0: float = 10.0,
    order: int = 2,
) -> np.ndarray:
    """Remove periodic noise from a grayscale ``(H, W)`` or colour ``(H, W, C)``
    image and return a ``uint8`` result.

    The input is never modified.  Colour images are filtered channel by
    channel with the same mask.  The output is clipped to ``[0, 255]`` rather
    than min-max rescaled, so overall brightness is preserved.
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError("image must be 2-D (gray) or 3-D (H, W, C)")
    mask = notch_mask(img.shape[:2], points, kind, d0, order)
    if img.ndim == 2:
        out = _filter_channel(img, mask)
    else:
        out = np.stack([_filter_channel(img[..., c], mask) for c in range(img.shape[2])], axis=-1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
