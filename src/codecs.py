"""Image encoding and decoding helpers backed by Pillow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

ImageArray = NDArray[np.uint8]
Subsampling = Literal["4:4:4", "4:2:2", "4:2:0"]

_SUBSAMPLING_TO_PILLOW: dict[Subsampling, int] = {
    "4:4:4": 0,
    "4:2:2": 1,
    "4:2:0": 2,
}


def validate_image_array(array: ImageArray) -> ImageArray:
    """Validate and return a grayscale or RGB uint8 image array."""

    if not isinstance(array, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if array.dtype != np.uint8:
        raise TypeError("image array must use dtype uint8")
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    raise ValueError("image array must have shape (height, width) or (height, width, 3)")


def validate_jpeg_quality(quality: int) -> int:
    """Validate a Pillow JPEG quality for this project."""

    if not isinstance(quality, int):
        raise TypeError("JPEG quality must be an integer")
    if quality < 1 or quality > 95:
        raise ValueError("JPEG quality must be in the supported range 1..95")
    return quality


def validate_subsampling(subsampling: str | None) -> Subsampling | None:
    """Validate a JPEG chroma subsampling option."""

    if subsampling is None or subsampling == "default":
        return None
    if subsampling not in _SUBSAMPLING_TO_PILLOW:
        choices = ", ".join(_SUBSAMPLING_TO_PILLOW)
        raise ValueError(f"subsampling must be one of: default, {choices}")
    return subsampling  # type: ignore[return-value]


def array_to_image(array: ImageArray) -> Image.Image:
    """Convert a validated NumPy array into a Pillow image."""

    array = validate_image_array(array)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    return Image.fromarray(array, mode="RGB")


def _prepare_output_path(path: Path, overwrite: bool) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    return path


def save_png(array: ImageArray, path: Path, *, overwrite: bool = False) -> Path:
    """Save a lossless PNG image."""

    path = _prepare_output_path(path, overwrite)
    array_to_image(array).save(path, format="PNG")
    return path


def save_jpeg(
    array: ImageArray,
    path: Path,
    *,
    quality: int,
    subsampling: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Save a lossy JPEG image using Pillow."""

    quality = validate_jpeg_quality(quality)
    subsampling = validate_subsampling(subsampling)
    path = _prepare_output_path(path, overwrite)
    image = array_to_image(array)
    save_kwargs: dict[str, object] = {"format": "JPEG", "quality": quality}
    if subsampling is not None and image.mode == "RGB":
        save_kwargs["subsampling"] = _SUBSAMPLING_TO_PILLOW[subsampling]
    image.save(path, **save_kwargs)
    return path


def load_image_array(path: Path) -> ImageArray:
    """Reload an encoded image from disk as a grayscale or RGB uint8 array."""

    with Image.open(path) as image:
        if image.mode == "L":
            converted = image.copy()
        else:
            converted = image.convert("RGB")
        return np.asarray(converted, dtype=np.uint8)


def file_size(path: Path) -> int:
    """Return a file size in bytes."""

    return Path(path).stat().st_size
