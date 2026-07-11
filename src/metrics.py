"""Difference and image-quality metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class ImageMetrics:
    """Basic pixel-error metrics for two images."""

    mean_absolute_error: float
    maximum_absolute_error: float
    mean_squared_error: float
    psnr_db: float

    def to_json_dict(self) -> dict[str, float | str]:
        """Return a JSON-safe dictionary."""

        psnr: float | str = "Infinity" if math.isinf(self.psnr_db) else self.psnr_db
        return {
            "mean_absolute_error": self.mean_absolute_error,
            "maximum_absolute_error": self.maximum_absolute_error,
            "mean_squared_error": self.mean_squared_error,
            "psnr_db": psnr,
        }


def validate_same_shape(reference: ImageArray, candidate: ImageArray) -> None:
    """Raise if two images cannot be compared directly."""

    if reference.shape != candidate.shape:
        raise ValueError(f"image shapes differ: {reference.shape} != {candidate.shape}")


def absolute_difference(reference: ImageArray, candidate: ImageArray) -> NDArray[np.float64]:
    """Return the absolute per-channel pixel difference."""

    validate_same_shape(reference, candidate)
    return np.abs(reference.astype(np.float64) - candidate.astype(np.float64))


def compute_metrics(reference: ImageArray, candidate: ImageArray) -> ImageMetrics:
    """Compute lightweight image-difference metrics."""

    diff = absolute_difference(reference, candidate)
    mae = float(np.mean(diff))
    max_abs = float(np.max(diff))
    mse = float(np.mean(diff**2))
    psnr = math.inf if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    return ImageMetrics(
        mean_absolute_error=mae,
        maximum_absolute_error=max_abs,
        mean_squared_error=mse,
        psnr_db=psnr,
    )
