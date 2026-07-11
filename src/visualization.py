"""Matplotlib comparison figures for encoded images."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import absolute_difference


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    return f"{size_bytes / 1024:.1f} KiB"


def _imshow_image(axis: plt.Axes, image: np.ndarray) -> None:
    if image.ndim == 2:
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        axis.imshow(image, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])


def save_comparison_figure(
    reference: np.ndarray,
    jpeg: np.ndarray,
    output_path: Path,
    *,
    pattern_name: str,
    pattern_parameters: dict[str, object],
    jpeg_quality: int,
    png_size: int,
    jpeg_size: int,
    amplification: float = 8.0,
    show: bool = False,
    overwrite: bool = False,
) -> Path:
    """Save a side-by-side reference, JPEG, and amplified-difference figure."""

    if amplification <= 0:
        raise ValueError("amplification must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diff = np.clip(absolute_difference(reference, jpeg) * amplification, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    _imshow_image(axes[0], reference)
    axes[0].set_title(f"PNG reference\n{_format_size(png_size)}")
    _imshow_image(axes[1], jpeg)
    axes[1].set_title(f"JPEG q={jpeg_quality}\n{_format_size(jpeg_size)}")
    _imshow_image(axes[2], diff)
    axes[2].set_title(f"Abs diff x{amplification:g}")

    parameters = ", ".join(f"{key}={value}" for key, value in sorted(pattern_parameters.items()))
    subtitle = f"{pattern_name}"
    if parameters:
        subtitle = f"{subtitle}: {parameters}"
    fig.suptitle(subtitle)
    fig.savefig(output_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)
    return output_path
