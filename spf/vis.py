"""Visualisation utilities for potential curves."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from .segmentation import LineSegmentor

# ── Style ────────────────────────────────────────────────────────────────────
_PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899",
            "#0ea5e9", "#84cc16"]
_SPF_COLOR = "#2563eb"
_REF_COLOR = "#94a3b8"
_BG = "#fafbfc"
_GRID_COLOR = "#e2e8f0"
_TEXT_COLOR = "#1e293b"
_FILL_ALPHA = 0.08
_DPI = 200


def _style_ax(ax, xlabel="Frame index", ylabel="Perceptual progress"):
    """Apply clean, modern styling to an axis."""
    ax.set_facecolor(_BG)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.6)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
    ax.set_xlabel(xlabel, fontsize=10, color=_TEXT_COLOR, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, color=_TEXT_COLOR, labelpad=6)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
        spine.set_linewidth(0.6)


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(str(save_path), dpi=_DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    else:
        plt.show()
    plt.close(fig)


# ── Public API ───────────────────────────────────────────────────────────────

def plot_potential(
    potential: np.ndarray,
    title: str = "Semantic Potential Function",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 4),
):
    """Plot a potential curve with a shaded deviation band from the linear reference."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style_ax(ax)
    n = len(potential)
    x = np.arange(n)
    linear = np.linspace(0, 1, n)

    # Shaded deviation band between SPF and linear reference
    ax.fill_between(x, potential, linear, color=_SPF_COLOR, alpha=_FILL_ALPHA)

    # Linear reference
    ax.plot(x, linear, "--", color=_REF_COLOR, linewidth=1.2, label="linear reference")

    # SPF curve — gradient-colored by slope
    ax.plot(x, potential, color=_SPF_COLOR, linewidth=1.8, label="SPF", zorder=3)
    ax.scatter(x, potential, c=_SPF_COLOR, s=4, zorder=4, edgecolors="none")

    ax.set_xlim(0, n - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=13, fontweight="600", color=_TEXT_COLOR, pad=12)
    ax.legend(framealpha=0.9, edgecolor=_GRID_COLOR, fontsize=9, loc="upper left")

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_potential_with_segments(
    potential: np.ndarray,
    segmentor: LineSegmentor,
    title: str = "SPF with Segments",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 4),
):
    """Plot potential curve with piecewise-linear segments and boundary markers."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style_ax(ax)
    n = len(potential)
    x = np.arange(n)

    # Potential as subtle dots
    ax.scatter(x, potential, c=_REF_COLOR, s=6, alpha=0.35, edgecolors="none", zorder=2)

    coeffs = segmentor.get_coefficients()

    # Build continuous polyline: each segment's fitted line, connected at boundaries
    prev_end_xy = None
    for idx, (start, end, slope, intercept) in enumerate(coeffs):
        color = _PALETTE[idx % len(_PALETTE)]
        xi = x[start : end + 1]
        yi = slope * xi + intercept

        # Connect to previous segment's endpoint
        if prev_end_xy is not None:
            cx = [prev_end_xy[0], xi[0]]
            cy = [prev_end_xy[1], yi[0]]
            ax.plot(cx, cy, linewidth=2.5, color=color, zorder=5)

        ax.plot(xi, yi, linewidth=2.5, color=color, zorder=5)
        if idx > 0:
            ax.axvline(start, color=_GRID_COLOR, linewidth=0.7, zorder=1)
        prev_end_xy = (xi[-1], yi[-1])

    ax.set_xlim(0, n - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=13, fontweight="600", color=_TEXT_COLOR, pad=12)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_spf_report(
    potential: np.ndarray,
    segmentor: LineSegmentor,
    frames: Optional[List[Image.Image]] = None,
    title: str = "SPF Analysis",
    save_path: Optional[Path] = None,
    num_thumbs: int = 9,
    figsize: tuple = (12, 7),
):
    """Combined report: SPF curve, segments, and optional frame strip.

    Args:
        potential: 1-D potential array.
        segmentor: A fitted :class:`LineSegmentor`.
        frames: Optional list of PIL images (all video frames). If provided,
            a frame strip is rendered along the bottom.
        title: Figure super-title.
        save_path: Save path (or show interactively).
        num_thumbs: Number of thumbnail frames to display.
        figsize: Figure size.
    """
    has_frames = frames is not None and len(frames) > 0
    nrows = 3 if has_frames else 2
    height_ratios = [3, 3, 1] if has_frames else [1, 1]

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(nrows, 1, height_ratios=height_ratios, hspace=0.35)

    n = len(potential)
    x = np.arange(n)
    linear = np.linspace(0, 1, n)

    # ── Panel 1: SPF curve ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1, xlabel="", ylabel="Perceptual progress")

    ax1.fill_between(x, potential, linear, color=_SPF_COLOR, alpha=_FILL_ALPHA)
    ax1.plot(x, linear, "--", color=_REF_COLOR, linewidth=1.0)
    ax1.plot(x, potential, color=_SPF_COLOR, linewidth=1.8, zorder=3)
    ax1.scatter(x, potential, c=_SPF_COLOR, s=3, zorder=4, edgecolors="none")

    ax1.set_xlim(0, n - 1)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("SPF Curve", fontsize=11, fontweight="600", color=_TEXT_COLOR, pad=8)

    # ── Panel 2: Segments ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _style_ax(ax2, xlabel="" if has_frames else "Frame index",
              ylabel="Perceptual progress")

    ax2.scatter(x, potential, c=_REF_COLOR, s=5, alpha=0.35, edgecolors="none",
                zorder=2)

    coeffs = segmentor.get_coefficients()
    prev_end_xy = None
    for idx, (start, end, slope, intercept) in enumerate(coeffs):
        color = _PALETTE[idx % len(_PALETTE)]
        xi = x[start : end + 1]
        yi = slope * xi + intercept
        if prev_end_xy is not None:
            ax2.plot([prev_end_xy[0], xi[0]], [prev_end_xy[1], yi[0]],
                     linewidth=2.5, color=color, zorder=5)
        ax2.plot(xi, yi, linewidth=2.5, color=color, zorder=5)
        if idx > 0:
            ax2.axvline(start, color=_GRID_COLOR, linewidth=0.7, zorder=1)
        prev_end_xy = (xi[-1], yi[-1])

    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title(f"Segments ({segmentor.num_segments})",
                  fontsize=11, fontweight="600", color=_TEXT_COLOR, pad=8)

    # ── Panel 3: Frame strip ────────────────────────────────────────────
    if has_frames:
        ax3 = fig.add_subplot(gs[2])
        ax3.set_axis_off()

        indices = np.linspace(0, len(frames) - 1, num_thumbs, dtype=int)
        thumb_h = 80
        sample = [frames[i].copy() for i in indices]
        aspect = sample[0].width / sample[0].height
        thumb_w = int(thumb_h * aspect)
        sample = [img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                  for img in sample]

        # Build strip as a single wide image
        gap = 4
        strip_w = num_thumbs * thumb_w + (num_thumbs - 1) * gap
        strip = Image.new("RGB", (strip_w, thumb_h), (255, 255, 255))
        for i, img in enumerate(sample):
            strip.paste(img, (i * (thumb_w + gap), 0))

        ax3.imshow(np.array(strip), aspect="auto")

        # Frame index labels below each thumbnail
        for i, idx in enumerate(indices):
            cx = (i * (thumb_w + gap) + thumb_w / 2) / strip_w
            ax3.text(cx, 1.08, f"#{idx}", transform=ax3.transAxes,
                     ha="center", va="top", fontsize=7, color=_TEXT_COLOR)

    fig.suptitle(title, fontsize=15, fontweight="700", color=_TEXT_COLOR, y=0.98)
    _save_or_show(fig, save_path)
