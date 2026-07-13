# ForYay - JPEG Payload Carrier 
I dug into JPEGs and wondered: could quantization of a high-frequency spatial pattern act like a carrier for a second, embedded image? Turns out yes — compression itself can act as a mask, and diffing the compressed image against its lossless original reveals a ghost of the second image, with fidelity that tracks JPEG quality. *(Caveat: decoding currently needs that lossless reference — it's more a lens into how quantization discards information than a drop-a-message-in-a-JPEG scheme.)*

The core workflow is:

1. Load a visible carrier image.
2. Load a second image as a grayscale payload map.
3. Multiply the payload by a high-frequency `-1/+1` carrier wave.
4. Add that perturbation to the carrier in a configurable RGB direction.
5. Save a lossless PNG reference and lossy JPEG versions.
6. Save reveal images from `abs(PNG_reference - JPEG)`.

Pillow performs the PNG/JPEG encoding. The project does not implement a JPEG codec.

## Install

Use Python 3.10 or newer.

```bash
python -m pip install numpy Pillow matplotlib pytest
```

## Run

```bash
python -m src.cli ^
  --carrier-image inputs/checker.png ^
  --payload-image inputs/morty.png ^
  --output-dir outputs/payload_carrier ^
  --carrier-wave checkerboard ^
  --carrier-tile-size 1 ^
  --payload-strength 18 ^
  --rgb-vector 0 1 0 ^
  --seed 7 ^
  --jpeg-quality 95 75 50 25 ^
  --overwrite
```

On shells that do not use `^` for line continuation, place the command on one line or use that shell's continuation syntax.
```bash
python -m src.cli --carrier-image inputs/checker.png --payload-image inputs/morty.png --output-dir outputs/payload_carrier --carrier-wave checkerboard --carrier-tile-size 1 --payload-strength 18 --rgb-vector 0 1 0 --jpeg-quality 95 75 50 25 --overwrite
```

## Outputs

The output directory contains:

```text
carrier_rgb.png
payload_luma.png
carrier_wave.png
modified_reference.png
payload_embedding_overview.png
quality_95.jpg
jpeg_comparison_quality_95.png
reveal_quality_95.png
payload_reveal_quality_95.png
payload_carrier_results.json
```

`payload_reveal_quality_*.png` is the most important result: it shows the original payload next to the JPEG-error reveal and reports their correlation.

## Parameters

- `--carrier-wave`: `checkerboard`, `horizontal-stripes`, `vertical-stripes`, or `random`
- `--carrier-tile-size`: size of the carrier wave cells
- `--payload-strength`: perturbation amplitude before JPEG encoding
- `--rgb-vector`: RGB direction for the hidden signal, for example `0 1 0` for green or `1 0 -1` for red/blue opposition
- `--jpeg-quality`: one or more Pillow JPEG quality settings from `1..95`
- `--subsampling`: `default`, `4:4:4`, `4:2:2`, or `4:2:0`

Green-heavy vectors usually affect luma strongly. Red/blue-opposed vectors tend to produce more chroma-reactive artifacts.

## Python Entry Point

The supported public entry point is:

```bash
python -m src.cli
```

## Tests

```bash
python -m pytest
```

## License

MIT
