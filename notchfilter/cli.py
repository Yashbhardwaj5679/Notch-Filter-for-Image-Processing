"""Command-line interface:  python -m notchfilter --help"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .filters import FILTER_KINDS, apply_notch_filter
from .metrics import psnr, ssim
from .spectrum import detect_peaks, magnitude_spectrum
from .synthetic import add_periodic_noise


def _point(text: str) -> tuple[int, int]:
    try:
        r, c = text.split(",")
        return int(r), int(c)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("points must look like ROW,COL (e.g. 40,120)") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m notchfilter",
        description="Remove periodic noise from an image with notch filters.",
    )
    p.add_argument("input", type=Path, help="input image (png, jpg, jfif, ...)")
    p.add_argument("-o", "--output", type=Path, help="output file (default: <input>_<filter>.png)")
    p.add_argument("-f", "--filter", choices=(*FILTER_KINDS, "all"), default="butterworth")
    p.add_argument("-r", "--radius", type=float, default=10.0, help="notch radius D0 in pixels (default 10)")
    p.add_argument("-n", "--order", type=int, default=2, help="Butterworth order (default 2)")
    p.add_argument("-p", "--point", type=_point, action="append", default=[], metavar="ROW,COL",
                   help="noise spike in the centred spectrum; repeat for several")
    p.add_argument("-a", "--auto", type=int, metavar="N", default=0,
                   help="auto-detect up to N spike pairs instead of passing --point")
    p.add_argument("--gray", action="store_true", help="convert to grayscale first")
    p.add_argument("--add-noise", action="store_true",
                   help="corrupt the input with synthetic sinusoidal noise first (demo/testing)")
    p.add_argument("--amplitude", type=float, default=40.0, help="amplitude for --add-noise")
    p.add_argument("--compare", type=Path, metavar="FIG.png",
                   help="save a side-by-side comparison figure (images + spectra)")
    return p


def _comparison_figure(path: Path, noisy: np.ndarray, results: dict[str, np.ndarray], points) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("Input", noisy)] + [(k.capitalize(), v) for k, v in results.items()]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.4 * len(panels), 6.6), squeeze=False)
    for col, (title, img) in enumerate(panels):
        axes[0, col].imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
        axes[0, col].set_title(title)
        axes[1, col].imshow(magnitude_spectrum(img), cmap="gray")
        axes[1, col].set_title(f"{title} spectrum", fontsize=9)
        for r, c in points:
            h, w = img.shape[:2]
            for rr, cc in ((r, c), (2 * (h // 2) - r, 2 * (w // 2) - c)):
                axes[1, col].plot(cc, rr, "r+", ms=10)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 1

    img = Image.open(args.input)
    img = img.convert("L") if args.gray else img.convert("RGB" if img.mode not in ("L", "1") else "L")
    clean = np.asarray(img)
    noisy = add_periodic_noise(clean, amplitude=args.amplitude) if args.add_noise else clean

    points = list(args.point)
    if args.auto:
        points += detect_peaks(noisy, max_peaks=args.auto)
        print(f"auto-detected spikes (row, col): {points}")
    if not points:
        print("error: give at least one --point ROW,COL or use --auto N", file=sys.stderr)
        return 1

    kinds = FILTER_KINDS if args.filter == "all" else (args.filter,)
    results = {}
    if args.add_noise:
        print(f"{'noisy input':<12} PSNR {psnr(clean, noisy):6.2f} dB   SSIM {ssim(clean, noisy):.4f}")
    for kind in kinds:
        out = apply_notch_filter(noisy, points, kind, args.radius, args.order)
        results[kind] = out
        dest = args.output if (args.output and len(kinds) == 1) else (
            args.input.with_name(f"{args.input.stem}_{kind}.png"))
        Image.fromarray(out).save(dest)
        msg = f"{kind:<12} saved -> {dest}"
        if args.add_noise:
            msg += f"   PSNR {psnr(clean, out):6.2f} dB   SSIM {ssim(clean, out):.4f}"
        print(msg)
    if args.compare:
        _comparison_figure(args.compare, noisy, results, points)
        print(f"comparison figure -> {args.compare}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
