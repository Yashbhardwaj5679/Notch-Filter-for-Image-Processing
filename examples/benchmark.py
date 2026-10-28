"""Corrupt a clean image with known sinusoidal noise, remove it with each filter and
report PSNR / SSIM against the clean original (the 'quantitative metrics' use-case).

    python examples/benchmark.py [image] [--save docs/results/benchmark.png]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run without installing the package
from notchfilter import FILTER_KINDS, add_periodic_noise, apply_notch_filter, detect_peaks, psnr, ssim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image", nargs="?", default=ROOT / "sample_images" / "16.jpg")
ap.add_argument("--save", type=Path, default=ROOT / "docs" / "results" / "benchmark.png")
ap.add_argument("--radius", type=float, default=8)
args = ap.parse_args()

clean = np.asarray(Image.open(args.image).convert("L"))
noisy = add_periodic_noise(clean, amplitude=40)
points = detect_peaks(noisy, max_peaks=2)
print(f"detected spikes: {points}\n")
print(f"{'':<12}{'PSNR (dB)':>10}{'SSIM':>8}")
print(f"{'noisy':<12}{psnr(clean, noisy):>10.2f}{ssim(clean, noisy):>8.3f}")

panels = [("Clean", clean), ("Noisy", noisy)]
for kind in FILTER_KINDS:
    out = apply_notch_filter(noisy, points, kind, d0=args.radius, order=2)
    print(f"{kind:<12}{psnr(clean, out):>10.2f}{ssim(clean, out):>8.3f}")
    panels.append((kind.capitalize(), out))

fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
for ax, (title, img) in zip(axes, panels):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
args.save.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.save, dpi=120)
print(f"\nfigure saved to {args.save}")
