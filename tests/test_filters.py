import numpy as np
import pytest

from notchfilter import (FILTER_KINDS, add_periodic_noise, apply_notch_filter, detect_peaks,
                         notch_mask, psnr)


@pytest.fixture(scope="module")
def clean():
    """Smooth synthetic test image with some structure (no external files needed)."""
    x, y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128), indexing="ij")
    img = 120 + 80 * np.exp(-((x - .5) ** 2 + (y - .5) ** 2) / .05) + 20 * x
    img[40:90, 50:60] = 230
    return img.astype(np.uint8)


@pytest.mark.parametrize("kind", FILTER_KINDS)
def test_mask_is_symmetric_and_bounded(kind):
    mask = notch_mask((64, 64), [(10, 20)], kind, d0=4)
    assert mask.shape == (64, 64)
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    # point symmetry about the DC pixel (32, 32)
    assert np.allclose(mask[1:, 1:], mask[1:, 1:][::-1, ::-1])
    assert mask[32, 32] == pytest.approx(1.0, abs=1e-2)       # DC is kept
    assert mask[10, 20] < 0.5 and mask[54, 44] < 0.5          # both spikes suppressed


def test_ideal_radius_is_in_pixels():
    mask = notch_mask((64, 64), [(10, 20)], "ideal", d0=3)
    assert mask[10, 23] == 0 and mask[10, 24] == 1


@pytest.mark.parametrize("kind", FILTER_KINDS)
def test_filter_reduces_periodic_noise(clean, kind):
    freqs = [(16, 16)]
    noisy = add_periodic_noise(clean, freqs, amplitude=40)
    # spike for (u, v) = (16, 16) cycles sits at (cu + 16, cv + 16) -> use its mirror in the upper half
    pt = (64 - 16, 64 - 16)
    out = apply_notch_filter(noisy, [pt], kind, d0=6, order=2)
    assert psnr(clean, out) > psnr(clean, noisy) + 6


def test_input_not_modified_and_dtype(clean):
    noisy = add_periodic_noise(clean)
    copy = noisy.copy()
    out = apply_notch_filter(noisy, [(48, 48)], "gaussian", d0=5)
    assert np.array_equal(noisy, copy) and out.dtype == np.uint8 and out.shape == noisy.shape


def test_colour_images(clean):
    rgb = np.stack([clean] * 3, axis=-1)
    out = apply_notch_filter(add_periodic_noise(rgb), [(48, 48)], "butterworth", d0=5)
    assert out.shape == rgb.shape


def test_auto_detection_finds_injected_noise(clean):
    noisy = add_periodic_noise(clean, [(16, 16), (0, 32)], amplitude=50)
    found = detect_peaks(noisy, max_peaks=2)
    expected = {(48, 48), (64, 32)}   # upper-half mirrors of (+16,+16) and (0,+32)
    assert expected <= {(r, c) for r, c in found} or all(
        min(abs(r - er) + abs(c - ec) for er, ec in expected) <= 1 for r, c in found)
    out = apply_notch_filter(noisy, found, "butterworth", d0=5)
    assert psnr(clean, out) > psnr(clean, noisy) + 6


def test_bad_arguments():
    with pytest.raises(ValueError):
        notch_mask((8, 8), [(1, 1)], "median")
    with pytest.raises(ValueError):
        notch_mask((8, 8), [(1, 1)], "ideal", d0=0)
