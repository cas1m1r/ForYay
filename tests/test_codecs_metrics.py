from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from src.codecs import load_image_array, save_jpeg, save_png
from src.metrics import absolute_difference, compute_metrics


def _gradient(width: int = 32, height: int = 16) -> np.ndarray:
    row = np.linspace(0, 255, width, dtype=np.uint8)
    return np.repeat(row[None, :], height, axis=0)


def _rgb_test_image(width: int = 32, height: int = 16) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image[:, :, 1] = 128
    image[:, :, 2] = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
    return image


def test_png_round_trip_is_lossless(tmp_path) -> None:
    source = _gradient()
    png_path = save_png(source, tmp_path / "reference.png")

    reloaded = load_image_array(png_path)

    np.testing.assert_array_equal(reloaded, source)


def test_jpeg_encoding_creates_valid_file_with_dimensions(tmp_path) -> None:
    source = _rgb_test_image()
    jpeg_path = save_jpeg(source, tmp_path / "quality_75.jpg", quality=75, subsampling="4:4:4")

    with Image.open(jpeg_path) as image:
        assert image.size == (32, 16)
        assert image.mode == "RGB"


@pytest.mark.parametrize("quality", [0, 96])
def test_jpeg_quality_is_validated(tmp_path, quality: int) -> None:
    source = _gradient(width=16, height=16)

    with pytest.raises(ValueError, match="1..95"):
        save_jpeg(source, tmp_path / "bad.jpg", quality=quality)


def test_difference_and_metrics_for_identical_images() -> None:
    source = _gradient(width=16, height=16)

    diff = absolute_difference(source, source)
    metrics = compute_metrics(source, source)

    assert float(diff.max()) == 0.0
    assert metrics.mean_absolute_error == 0.0
    assert metrics.maximum_absolute_error == 0.0
    assert metrics.mean_squared_error == 0.0
    assert math.isinf(metrics.psnr_db)
    assert metrics.to_json_dict()["psnr_db"] == "Infinity"


def test_metrics_for_known_difference() -> None:
    reference = np.zeros((2, 2), dtype=np.uint8)
    candidate = np.array([[0, 10], [20, 30]], dtype=np.uint8)

    metrics = compute_metrics(reference, candidate)

    assert metrics.mean_absolute_error == 15.0
    assert metrics.maximum_absolute_error == 30.0
    assert metrics.mean_squared_error == 350.0
    assert metrics.psnr_db < math.inf


def test_difference_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        absolute_difference(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8))
