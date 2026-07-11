"""Hide a payload image in codec-reactive high-frequency carrier noise."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .codecs import file_size, load_image_array, save_jpeg, save_png, validate_jpeg_quality, validate_subsampling
from .metrics import absolute_difference, compute_metrics, validate_same_shape
from .visualization import save_comparison_figure

ImageArray = NDArray[np.uint8]

CARRIER_WAVES = {"checkerboard", "horizontal-stripes", "vertical-stripes", "random"}


@dataclass(frozen=True)
class PayloadCarrierResult:
    """Metadata for one payload-in-carrier compression experiment."""

    carrier_path: str
    payload_path: str
    output_dir: str
    width: int
    height: int
    carrier_wave: str
    carrier_tile_size: int
    strength: float
    rgb_vector: tuple[float, float, float]
    seed: int
    modified_png_path: str
    payload_luma_path: str
    carrier_wave_path: str
    overview_path: str
    jpeg_results: list[dict[str, object]]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "carrier_path": self.carrier_path,
            "payload_path": self.payload_path,
            "output_dir": self.output_dir,
            "width": self.width,
            "height": self.height,
            "carrier_wave": self.carrier_wave,
            "carrier_tile_size": self.carrier_tile_size,
            "strength": self.strength,
            "rgb_vector": list(self.rgb_vector),
            "seed": self.seed,
            "modified_png_path": self.modified_png_path,
            "payload_luma_path": self.payload_luma_path,
            "carrier_wave_path": self.carrier_wave_path,
            "overview_path": self.overview_path,
            "jpeg_results": self.jpeg_results,
        }


def _write_json(path: Path, data: object, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _quality_tag(quality: int, subsampling: str | None) -> str:
    tag = f"quality_{quality}"
    if subsampling:
        tag = f"{tag}_{subsampling.replace(':', '')}"
    return tag


def load_rgb(path: Path) -> ImageArray:
    """Load an image as RGB uint8."""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_payload_luma(path: Path, *, width: int, height: int) -> NDArray[np.float64]:
    """Load and resize payload luminance to ``0..1``."""

    with Image.open(path) as image:
        resampling = getattr(Image.Resampling, "LANCZOS", Image.BICUBIC)
        payload = image.convert("L").resize((width, height), resampling)
    return np.asarray(payload, dtype=np.float64) / 255.0


def _validate_carrier_wave(carrier_wave: str) -> str:
    if carrier_wave not in CARRIER_WAVES:
        choices = ", ".join(sorted(CARRIER_WAVES))
        raise ValueError(f"carrier_wave must be one of: {choices}")
    return carrier_wave


def _validate_tile_size(tile_size: int) -> int:
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("carrier_tile_size must be a positive integer")
    return tile_size


def _validate_rgb_vector(rgb_vector: Iterable[float]) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in rgb_vector)
    if len(vector) != 3:
        raise ValueError("rgb_vector must contain exactly three numbers")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("rgb_vector must not be all zeros")
    return vector


def make_carrier_wave(
    width: int,
    height: int,
    *,
    carrier_wave: str = "checkerboard",
    tile_size: int = 1,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Return a ``-1/+1`` high-frequency carrier wave."""

    carrier_wave = _validate_carrier_wave(carrier_wave)
    tile_size = _validate_tile_size(tile_size)
    yy, xx = np.mgrid[0:height, 0:width]
    if carrier_wave == "checkerboard":
        values = ((xx // tile_size) + (yy // tile_size)) % 2
        return np.where(values == 0, -1.0, 1.0)
    if carrier_wave == "horizontal-stripes":
        values = (yy // tile_size) % 2
        return np.where(values == 0, -1.0, 1.0)
    if carrier_wave == "vertical-stripes":
        values = (xx // tile_size) % 2
        return np.where(values == 0, -1.0, 1.0)
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=(height, width))


def embed_payload_in_carrier(
    carrier_rgb: ImageArray,
    payload_luma: NDArray[np.float64],
    *,
    carrier_wave: str = "checkerboard",
    carrier_tile_size: int = 1,
    strength: float = 18.0,
    rgb_vector: Iterable[float] = (0.0, 1.0, 0.0),
    seed: int = 0,
) -> tuple[ImageArray, NDArray[np.float64]]:
    """Embed payload as high-frequency, payload-amplitude perturbation."""

    if carrier_rgb.dtype != np.uint8 or carrier_rgb.ndim != 3 or carrier_rgb.shape[2] != 3:
        raise ValueError("carrier_rgb must have shape (height, width, 3) and dtype uint8")
    if payload_luma.shape != carrier_rgb.shape[:2]:
        raise ValueError("payload_luma shape must match carrier height and width")
    if strength <= 0.0:
        raise ValueError("strength must be positive")
    rgb_vector = _validate_rgb_vector(rgb_vector)
    height, width = carrier_rgb.shape[:2]
    wave = make_carrier_wave(width, height, carrier_wave=carrier_wave, tile_size=carrier_tile_size, seed=seed)
    amplitude = payload_luma * float(strength)
    delta = wave[:, :, None] * amplitude[:, :, None] * np.array(rgb_vector, dtype=np.float64)[None, None, :]
    modified = np.clip(carrier_rgb.astype(np.float64) + delta, 0, 255)
    return np.rint(modified).astype(np.uint8), wave


def reveal_from_difference(reference: ImageArray, jpeg: ImageArray) -> ImageArray:
    """Return a contrast-stretched grayscale reveal from JPEG error magnitude."""

    diff = absolute_difference(reference, jpeg)
    if diff.ndim == 3:
        reveal = np.mean(diff, axis=2)
    else:
        reveal = diff
    high = float(np.percentile(reveal, 99.0))
    if high <= 0.0:
        return np.zeros(reveal.shape, dtype=np.uint8)
    return np.rint(np.clip(reveal / high, 0, 1) * 255).astype(np.uint8)


def normalized_correlation(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Return Pearson correlation with safe zero-variance handling."""

    first_flat = first.astype(np.float64).ravel()
    second_flat = second.astype(np.float64).ravel()
    first_centered = first_flat - float(np.mean(first_flat))
    second_centered = second_flat - float(np.mean(second_flat))
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def save_payload_embedding_overview(
    carrier_rgb: ImageArray,
    payload_luma_u8: ImageArray,
    carrier_wave_u8: ImageArray,
    modified_rgb: ImageArray,
    output_path: Path,
    *,
    carrier_name: str,
    payload_name: str,
    show: bool = False,
    overwrite: bool = False,
) -> Path:
    """Save carrier, payload, wave, and modified image."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {output_path}")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    axes[0].imshow(carrier_rgb, interpolation="nearest")
    axes[0].set_title("Carrier")
    axes[1].imshow(payload_luma_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title("Payload map")
    axes[2].imshow(carrier_wave_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[2].set_title("Carrier wave")
    axes[3].imshow(modified_rgb, interpolation="nearest")
    axes[3].set_title("Modified reference")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"{payload_name} embedded in {carrier_name}")
    fig.savefig(output_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def save_payload_reveal_figure(
    payload_luma_u8: ImageArray,
    reveal_u8: ImageArray,
    output_path: Path,
    *,
    jpeg_quality: int,
    correlation: float,
    show: bool = False,
    overwrite: bool = False,
) -> Path:
    """Save payload and compression-difference reveal side by side."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {output_path}")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(payload_luma_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("Payload")
    axes[1].imshow(reveal_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title(f"Reveal q={jpeg_quality}\nr={correlation:.3f}")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(output_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def run_payload_carrier_experiment(
    *,
    carrier_path: Path,
    payload_path: Path,
    output_dir: Path,
    carrier_wave: str = "checkerboard",
    carrier_tile_size: int = 1,
    strength: float = 18.0,
    rgb_vector: Iterable[float] = (0.0, 1.0, 0.0),
    seed: int = 0,
    jpeg_qualities: Iterable[int] = (95, 75, 50, 25),
    jpeg_subsampling: str | None = None,
    difference_amplification: float = 8.0,
    show: bool = False,
    overwrite: bool = False,
) -> PayloadCarrierResult:
    """Embed one image into carrier noise and compare JPEG reveal behavior."""

    qualities = [validate_jpeg_quality(quality) for quality in jpeg_qualities]
    if not qualities:
        raise ValueError("at least one JPEG quality is required")
    jpeg_subsampling = validate_subsampling(jpeg_subsampling)
    rgb_vector = _validate_rgb_vector(rgb_vector)

    carrier_rgb = load_rgb(carrier_path)
    height, width = carrier_rgb.shape[:2]
    payload_luma = load_payload_luma(payload_path, width=width, height=height)
    modified_rgb, wave = embed_payload_in_carrier(
        carrier_rgb,
        payload_luma,
        carrier_wave=carrier_wave,
        carrier_tile_size=carrier_tile_size,
        strength=strength,
        rgb_vector=rgb_vector,
        seed=seed,
    )
    payload_luma_u8 = np.rint(payload_luma * 255).astype(np.uint8)
    wave_u8 = np.where(wave < 0, 0, 255).astype(np.uint8)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_png(carrier_rgb, output_dir / "carrier_rgb.png", overwrite=overwrite)
    modified_png_path = save_png(modified_rgb, output_dir / "modified_reference.png", overwrite=overwrite)
    payload_luma_path = save_png(payload_luma_u8, output_dir / "payload_luma.png", overwrite=overwrite)
    carrier_wave_path = save_png(wave_u8, output_dir / "carrier_wave.png", overwrite=overwrite)
    overview_path = save_payload_embedding_overview(
        carrier_rgb,
        payload_luma_u8,
        wave_u8,
        modified_rgb,
        output_dir / "payload_embedding_overview.png",
        carrier_name=Path(carrier_path).name,
        payload_name=Path(payload_path).name,
        show=show,
        overwrite=overwrite,
    )

    reloaded_modified = load_image_array(modified_png_path)
    validate_same_shape(modified_rgb, reloaded_modified)
    png_size = file_size(modified_png_path)

    jpeg_results: list[dict[str, object]] = []
    for quality in qualities:
        tag = _quality_tag(quality, jpeg_subsampling)
        jpeg_path = save_jpeg(
            modified_rgb,
            output_dir / f"{tag}.jpg",
            quality=quality,
            subsampling=jpeg_subsampling,
            overwrite=overwrite,
        )
        jpeg_reloaded = load_image_array(jpeg_path)
        validate_same_shape(reloaded_modified, jpeg_reloaded)
        jpeg_size = file_size(jpeg_path)
        metrics = compute_metrics(reloaded_modified, jpeg_reloaded)
        comparison_path = save_comparison_figure(
            reloaded_modified,
            jpeg_reloaded,
            output_dir / f"jpeg_comparison_{tag}.png",
            pattern_name=f"{Path(payload_path).name} payload in {Path(carrier_path).name}",
            pattern_parameters={"wave": carrier_wave, "strength": strength, "tile": carrier_tile_size},
            jpeg_quality=quality,
            png_size=png_size,
            jpeg_size=jpeg_size,
            amplification=difference_amplification,
            show=show,
            overwrite=overwrite,
        )
        reveal = reveal_from_difference(reloaded_modified, jpeg_reloaded)
        reveal_path = save_png(reveal, output_dir / f"reveal_{tag}.png", overwrite=overwrite)
        correlation = normalized_correlation(payload_luma, reveal.astype(np.float64) / 255.0)
        reveal_figure_path = save_payload_reveal_figure(
            payload_luma_u8,
            reveal,
            output_dir / f"payload_reveal_{tag}.png",
            jpeg_quality=quality,
            correlation=correlation,
            show=show,
            overwrite=overwrite,
        )
        jpeg_results.append(
            {
                "jpeg_quality": quality,
                "jpeg_subsampling": jpeg_subsampling,
                "jpeg_path": str(jpeg_path),
                "comparison_path": str(comparison_path),
                "reveal_path": str(reveal_path),
                "payload_reveal_path": str(reveal_figure_path),
                "jpeg_file_size": jpeg_size,
                "compression_ratio_relative_to_png": png_size / jpeg_size if jpeg_size else float("inf"),
                "payload_reveal_correlation": correlation,
                **metrics.to_json_dict(),
            }
        )

    result = PayloadCarrierResult(
        carrier_path=str(carrier_path),
        payload_path=str(payload_path),
        output_dir=str(output_dir),
        width=width,
        height=height,
        carrier_wave=carrier_wave,
        carrier_tile_size=carrier_tile_size,
        strength=strength,
        rgb_vector=rgb_vector,
        seed=seed,
        modified_png_path=str(modified_png_path),
        payload_luma_path=str(payload_luma_path),
        carrier_wave_path=str(carrier_wave_path),
        overview_path=str(overview_path),
        jpeg_results=jpeg_results,
    )
    _write_json(output_dir / "payload_carrier_results.json", result.to_json_dict(), overwrite=overwrite)
    return result
