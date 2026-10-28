"""Notch filters (Ideal, Butterworth, Gaussian) for periodic noise removal."""

from .filters import FILTER_KINDS, apply_notch_filter, notch_mask
from .metrics import psnr, ssim
from .spectrum import detect_peaks, magnitude_spectrum
from .synthetic import add_periodic_noise

__all__ = [
    "FILTER_KINDS",
    "apply_notch_filter",
    "notch_mask",
    "detect_peaks",
    "magnitude_spectrum",
    "add_periodic_noise",
    "psnr",
    "ssim",
]
__version__ = "2.0.0"
