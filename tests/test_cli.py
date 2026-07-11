from __future__ import annotations

import json

import numpy as np

from src import cli
from src.codecs import save_png


def test_cli_argument_parsing_for_payload_carrier() -> None:
    args = cli.build_parser().parse_args(
        [
            "--carrier-image",
            "inputs/checker.png",
            "--payload-image",
            "inputs/FeelsHackerMan.png",
            "--payload-strength",
            "14",
            "--rgb-vector",
            "1",
            "0",
            "-1",
        ]
    )

    assert args.payload_strength == 14
    assert args.rgb_vector == [1.0, 0.0, -1.0]


def test_cli_runs_payload_carrier_experiment(tmp_path) -> None:
    carrier = np.full((32, 32, 3), 128, dtype=np.uint8)
    payload = np.zeros((32, 32, 3), dtype=np.uint8)
    payload[:, 16:, :] = 255
    carrier_path = tmp_path / "carrier.png"
    payload_path = tmp_path / "payload.png"
    output_dir = tmp_path / "out"
    save_png(carrier, carrier_path)
    save_png(payload, payload_path)

    exit_code = cli.main(
        [
            "--carrier-image",
            str(carrier_path),
            "--payload-image",
            str(payload_path),
            "--output-dir",
            str(output_dir),
            "--jpeg-quality",
            "75",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "modified_reference.png").exists()
    assert (output_dir / "quality_75.jpg").exists()
    assert (output_dir / "payload_reveal_quality_75.png").exists()
    metadata = json.loads((output_dir / "payload_carrier_results.json").read_text(encoding="utf-8"))
    assert metadata["jpeg_results"][0]["jpeg_quality"] == 75
