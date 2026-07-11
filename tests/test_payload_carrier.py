from __future__ import annotations

import json

import numpy as np

from src.codecs import save_png
from src.payload_carrier import (
    embed_payload_in_carrier,
    make_carrier_wave,
    normalized_correlation,
    reveal_from_difference,
    run_payload_carrier_experiment,
)


def test_checkerboard_carrier_wave_is_minus_one_plus_one() -> None:
    wave = make_carrier_wave(4, 4, carrier_wave="checkerboard", tile_size=1)

    assert set(np.unique(wave)) == {-1.0, 1.0}
    assert wave[0, 0] == -1.0
    assert wave[0, 1] == 1.0


def test_embed_payload_in_carrier_is_deterministic() -> None:
    carrier = np.full((8, 8, 3), 128, dtype=np.uint8)
    payload = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape(8, 8)

    first, first_wave = embed_payload_in_carrier(
        carrier,
        payload,
        carrier_wave="random",
        strength=12.0,
        seed=42,
    )
    second, second_wave = embed_payload_in_carrier(
        carrier,
        payload,
        carrier_wave="random",
        strength=12.0,
        seed=42,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_wave, second_wave)
    assert not np.array_equal(first, carrier)


def test_reveal_from_difference_and_correlation() -> None:
    reference = np.zeros((4, 4, 3), dtype=np.uint8)
    jpeg = reference.copy()
    jpeg[2:, 2:, :] = 10

    reveal = reveal_from_difference(reference, jpeg)

    assert reveal.shape == (4, 4)
    assert reveal.dtype == np.uint8
    assert int(reveal.max()) == 255
    assert normalized_correlation(reveal.astype(float), reveal.astype(float)) == 1.0


def test_payload_carrier_experiment_writes_outputs(tmp_path) -> None:
    carrier = np.full((32, 32, 3), 128, dtype=np.uint8)
    payload = np.zeros((32, 32, 3), dtype=np.uint8)
    payload[:, 16:, :] = 255
    carrier_path = tmp_path / "carrier.png"
    payload_path = tmp_path / "payload.png"
    output_dir = tmp_path / "out"
    save_png(carrier, carrier_path)
    save_png(payload, payload_path)

    result = run_payload_carrier_experiment(
        carrier_path=carrier_path,
        payload_path=payload_path,
        output_dir=output_dir,
        strength=12.0,
        jpeg_qualities=[75],
    )

    assert result.width == 32
    assert result.height == 32
    assert (output_dir / "modified_reference.png").exists()
    assert (output_dir / "payload_luma.png").exists()
    assert (output_dir / "carrier_wave.png").exists()
    assert (output_dir / "payload_embedding_overview.png").exists()
    assert (output_dir / "quality_75.jpg").exists()
    assert (output_dir / "reveal_quality_75.png").exists()
    metadata = json.loads((output_dir / "payload_carrier_results.json").read_text(encoding="utf-8"))
    assert metadata["carrier_wave"] == "checkerboard"
    assert metadata["jpeg_results"][0]["jpeg_quality"] == 75
