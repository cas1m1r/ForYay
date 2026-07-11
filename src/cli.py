"""Command-line interface for payload-in-carrier JPEG experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .payload_carrier import CARRIER_WAVES, run_payload_carrier_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Embed a payload image into high-frequency carrier noise, encode it as JPEG, "
            "and save reveal images from the JPEG error."
        )
    )
    parser.add_argument("--carrier-image", type=Path, default=Path("inputs") / "checker.png")
    parser.add_argument("--payload-image", type=Path, default=Path("inputs") / "FeelsHackerMan.png")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "payload_carrier")
    parser.add_argument("--carrier-wave", choices=sorted(CARRIER_WAVES), default="checkerboard")
    parser.add_argument("--carrier-tile-size", type=int, default=1)
    parser.add_argument("--payload-strength", type=float, default=18.0)
    parser.add_argument("--rgb-vector", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jpeg-quality", type=int, nargs="+", default=[95, 75, 50, 25])
    parser.add_argument("--subsampling", choices=["default", "4:4:4", "4:2:2", "4:2:0"], default="default")
    parser.add_argument("--difference-amplification", type=float, default=8.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing files created by a previous run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the payload carrier experiment."""

    parser = build_parser()
    args = parser.parse_args(argv)
    subsampling = None if args.subsampling == "default" else args.subsampling
    result = run_payload_carrier_experiment(
        carrier_path=args.carrier_image,
        payload_path=args.payload_image,
        output_dir=args.output_dir,
        carrier_wave=args.carrier_wave,
        carrier_tile_size=args.carrier_tile_size,
        strength=args.payload_strength,
        rgb_vector=args.rgb_vector,
        seed=args.seed,
        jpeg_qualities=args.jpeg_quality,
        jpeg_subsampling=subsampling,
        difference_amplification=args.difference_amplification,
        show=args.show,
        overwrite=args.overwrite,
    )
    print(f"Wrote payload carrier result under {args.output_dir}")
    print(f"Carrier: {result.carrier_path}")
    print(f"Payload: {result.payload_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
