from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path

# QAT checkpoints in this project may be saved under legacy tf.keras/tf_keras.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("TensorFlow is required to convert the model. Install requirements.txt first.") from exc


def load_model_compat(model_path: Path):
    loaders = []
    try:
        from tf_keras import models as legacy_models  # type: ignore
        loaders.append(legacy_models.load_model)
    except Exception:
        pass
    loaders.append(tf.keras.models.load_model)

    last_error: Exception | None = None
    for loader in loaders:
        try:
            return loader(model_path)
        except Exception as exc:
            last_error = exc
    raise SystemExit(f"failed to load model for conversion: {last_error}")


def load_representative_dataset(calibration_dir: Path, image_size: int):
    dataset = tf.keras.utils.image_dataset_from_directory(
        calibration_dir,
        image_size=(image_size, image_size),
        batch_size=1,
        shuffle=False,
    )

    for images, _labels in dataset.take(100):
        yield [tf.cast(images, tf.float32)]


def load_representative_dataset_from_manifest(manifest_path: Path, image_size: int, samples: int, seed: int):
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))

    if not rows:
        raise SystemExit(f"manifest is empty: {manifest_path}")

    per_label: dict[str, list[dict]] = {}
    for row in rows:
        label = str(row.get("canonical_label") or row.get("label") or "")
        per_label.setdefault(label, []).append(row)

    rng = random.Random(seed)
    selected: list[dict] = []
    labels = sorted(per_label.keys())
    if labels:
        per_class_target = max(1, samples // len(labels))
        for label in labels:
            items = per_label[label][:]
            rng.shuffle(items)
            selected.extend(items[:per_class_target])

    if len(selected) < samples:
        remaining = [row for row in rows if row not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, samples - len(selected))])

    selected = selected[:samples]

    for row in selected:
        image_path = Path(str(row["image_path"]))
        image_bytes = tf.io.read_file(str(image_path))
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, 0)
        yield [image]


def write_c_array(tflite_bytes: bytes, output_path: Path, symbol_name: str) -> None:
    lines = ["#include <cstddef>", "#include <cstdint>", ""]
    lines.append(f"alignas(16) const unsigned char {symbol_name}[] = {{")
    for offset in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[offset : offset + 12]
        values = ", ".join(f"0x{byte:02x}" for byte in chunk)
        suffix = "," if offset + 12 < len(tflite_bytes) else ""
        lines.append(f"  {values}{suffix}")
    lines.append("};")
    lines.append(f"const unsigned int {symbol_name}_len = {len(tflite_bytes)};")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a trained Keras model to a fully quantized TFLite model.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a saved Keras model (.keras or SavedModel directory).")
    parser.add_argument("--representative-dir", type=Path, default=None, help="Directory containing calibration images organized by class.")
    parser.add_argument("--representative-manifest", type=Path, default=None, help="Optional JSONL manifest for representative sampling.")
    parser.add_argument("--representative-samples", type=int, default=240, help="Number of representative samples to use when --representative-manifest is provided.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for representative sampling from manifest.")
    parser.add_argument("--output-path", type=Path, required=True, help="Where to write the .tflite model.")
    parser.add_argument("--output-cpp", type=Path, default=None, help="Optional .cc file for embedding the model in firmware.")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--symbol-name", type=str, default="g_model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_path.exists():
        raise SystemExit(f"model path does not exist: {args.model_path}")
    if args.representative_manifest is None and args.representative_dir is None:
        raise SystemExit("provide either --representative-dir or --representative-manifest")
    if args.representative_manifest is not None and not args.representative_manifest.exists():
        raise SystemExit(f"representative manifest does not exist: {args.representative_manifest}")
    if args.representative_dir is not None and not args.representative_dir.exists():
        raise SystemExit(f"representative directory does not exist: {args.representative_dir}")

    if args.model_path.is_dir():
        saved_model_marker = args.model_path / "saved_model.pb"
        if saved_model_marker.exists():
            converter = tf.lite.TFLiteConverter.from_saved_model(str(args.model_path))
        else:
            model = load_model_compat(args.model_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        model = load_model_compat(args.model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.representative_manifest is not None:
        converter.representative_dataset = lambda: load_representative_dataset_from_manifest(
            args.representative_manifest,
            args.image_size,
            args.representative_samples,
            args.seed,
        )
    else:
        converter.representative_dataset = lambda: load_representative_dataset(args.representative_dir, args.image_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_bytes(tflite_model)

    if args.output_cpp is not None:
        args.output_cpp.parent.mkdir(parents=True, exist_ok=True)
        write_c_array(tflite_model, args.output_cpp, args.symbol_name)

    print(f"wrote {len(tflite_model)} bytes to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
