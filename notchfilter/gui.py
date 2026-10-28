"""Tkinter GUI: load an image, pick (or auto-detect) noise spikes, apply a notch filter."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

from .filters import apply_notch_filter
from .metrics import psnr
from .spectrum import detect_peaks, magnitude_spectrum

PREVIEW = 420  # max preview edge in pixels
FILETYPES = [("Images", "*.png *.jpg *.jpeg *.jfif *.bmp *.tif *.tiff"), ("All files", "*.*")]


def _thumb(arr: np.ndarray) -> ImageTk.PhotoImage:
    img = Image.fromarray(arr)
    img.thumbnail((PREVIEW, PREVIEW))
    return ImageTk.PhotoImage(img)


class MainApp:
    def __init__(self) -> None:
        try:  # crisp text on Windows; harmless elsewhere
            from ctypes import windll  # type: ignore[attr-defined]

            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.root = tk.Tk()
        self.root.title("Notch Filter - periodic noise removal")
        icon = Path(__file__).with_name("icon.png")
        if icon.exists():
            self.root.iconphoto(False, tk.PhotoImage(file=str(icon)))

        self.original: np.ndarray | None = None
        self.filtered: np.ndarray | None = None
        self.points: list[tuple[int, int]] = []

        left = tk.LabelFrame(self.root, text="Original image")
        right = tk.LabelFrame(self.root, text="Filtered image")
        left.pack(side="left", fill="both", padx=6, pady=6)
        right.pack(side="right", fill="both", padx=6, pady=6)

        self.lbl_orig = tk.Label(left, text="Load an image\nto preview it here", width=40, height=18)
        self.lbl_filt = tk.Label(right, text="Apply a filter\nto view the result here", width=40, height=18)
        self.lbl_orig.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.lbl_filt.pack(fill="both")

        tk.Button(left, text="Browse image", bg="lightblue", command=self.browse).grid(row=1, column=0, sticky="nsew")
        tk.Button(left, text="Apply filter", bg="lightblue", command=self.apply).grid(row=1, column=1, sticky="nsew")

        self.kind = tk.StringVar(value="Butterworth")
        tk.Label(left, text="Filter type").grid(row=2, column=0, sticky="w")
        tk.OptionMenu(left, self.kind, "Ideal", "Butterworth", "Gaussian").grid(row=2, column=1, sticky="nsew")

        self.auto = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="Auto-detect spikes", variable=self.auto).grid(row=3, column=0, columnspan=2, sticky="w")

        self.n_points = self._entry(left, 4, "Number of points", "4")
        self.radius = self._entry(left, 5, "Notch radius D0 (px)", "10")
        self.order = self._entry(left, 6, "Butterworth order", "2")

        tk.Button(right, text="Save image", bg="lightblue", command=self.save).pack(fill="x")
        tk.Button(right, text="Show summary (images + spectra)", bg="lightblue", command=self.summary).pack(fill="x")
        self.status = tk.Label(right, text="READY", anchor="w")
        self.status.pack(fill="x")

    @staticmethod
    def _entry(parent: tk.Widget, row: int, label: str, default: str) -> tk.Entry:
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        e = tk.Entry(parent)
        e.insert(0, default)
        e.grid(row=row, column=1, sticky="nsew")
        return e

    # ------------------------------------------------------------------ actions
    def browse(self) -> None:
        path = filedialog.askopenfilename(title="Load image", filetypes=FILETYPES)
        if not path:
            return
        try:
            img = Image.open(path)
            self.original = np.asarray(img.convert("L" if img.mode in ("L", "1") else "RGB"))
            self.filtered = None
            self.photo_o = _thumb(self.original)
            self.lbl_orig.configure(image=self.photo_o, text="", width=0, height=0)
        except Exception as exc:
            messagebox.showerror("Could not load image", str(exc))

    def _pick_points(self, n: int) -> list[tuple[int, int]]:
        plt.close("all")
        plt.imshow(magnitude_spectrum(self.original), cmap="gray")
        plt.title(f"Click {n} noise spikes (one of each symmetric pair is enough)")
        clicks = plt.ginput(n, timeout=-1)
        plt.close()
        return [(int(round(y)), int(round(x))) for x, y in clicks]  # (row, col)

    def apply(self) -> None:
        if self.original is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        try:
            self.status.configure(text="BUSY")
            self.root.update_idletasks()
            n = int(self.n_points.get())
            self.points = detect_peaks(self.original, max_peaks=n) if self.auto.get() else self._pick_points(n)
            if not self.points:
                raise ValueError("No noise spikes found/selected.")
            self.filtered = apply_notch_filter(
                self.original, self.points, self.kind.get().lower(),
                float(self.radius.get()), int(self.order.get()),
            )
            self.photo_f = _thumb(self.filtered)
            self.lbl_filt.configure(image=self.photo_f, text="", width=0, height=0)
            self.status.configure(text=f"READY - {len(self.points)} notch pair(s); "
                                       f"PSNR vs original {psnr(self.original, self.filtered):.1f} dB")
        except Exception as exc:
            self.status.configure(text="ERROR")
            messagebox.showerror("Error", str(exc))

    def save(self) -> None:
        if self.filtered is None:
            messagebox.showinfo("Nothing to save", "Apply a filter first.")
            return
        path = filedialog.asksaveasfilename(title="Save image", defaultextension=".png", filetypes=FILETYPES)
        if path:
            Image.fromarray(self.filtered).save(path)

    def summary(self) -> None:
        if self.filtered is None:
            messagebox.showinfo("Nothing to show", "Apply a filter first.")
            return
        _, ax = plt.subplots(2, 2, figsize=(10, 9))
        for a, arr, title in (
            (ax[0, 0], self.original, "Original"), (ax[0, 1], self.filtered, "Filtered"),
            (ax[1, 0], magnitude_spectrum(self.original), "Original spectrum"),
            (ax[1, 1], magnitude_spectrum(self.filtered), "Filtered spectrum"),
        ):
            a.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
            a.set_title(title)
            a.axis("off")
        plt.tight_layout()
        plt.show()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    MainApp().run()


if __name__ == "__main__":
    main()
