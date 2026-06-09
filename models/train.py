from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import os
import random
from collections import Counter
from pathlib import Path

# QAT with tensorflow-model-optimization currently relies on legacy tf.keras APIs.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("TensorFlow is required to train the model. Install requirements.txt first.") from exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    label: str


AUTOTUNE = tf.data.AUTOTUNE


def import_tfmot():
    try:
        import tensorflow_model_optimization as tfmot  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "QAT requires tensorflow-model-optimization. "
            "Install requirements.txt and retry."
        ) from exc
    return tfmot


def load_class_names(class_map_path: Path | None) -> list[str]:
    if class_map_path is None:
        return []

    with class_map_path.open("r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)

    if isinstance(raw_mapping, dict):
        return [str(key) for key in raw_mapping.keys()]
    if isinstance(raw_mapping, list):
        return [str(item) for item in raw_mapping]
    raise SystemExit(f"unsupported class map format: {class_map_path}")


def read_manifest(manifest_path: Path) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            samples.append(SampleRecord(image_path=str(payload["image_path"]), label=str(payload["canonical_label"])))
    return samples


def read_directory_samples(data_dir: Path) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(SampleRecord(image_path=str(image_path.resolve()), label=class_dir.name))
    return samples


def split_samples(samples: list[SampleRecord], validation_split: float, seed: int) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not 0.0 < validation_split < 1.0:
        raise SystemExit("validation_split must be between 0 and 1")

    per_label: dict[str, list[SampleRecord]] = {}
    for sample in samples:
        per_label.setdefault(sample.label, []).append(sample)

    rng = random.Random(seed)
    train_samples: list[SampleRecord] = []
    val_samples: list[SampleRecord] = []
    for label in sorted(per_label):
        label_samples = per_label[label][:]
        rng.shuffle(label_samples)
        validation_count = max(1, int(round(len(label_samples) * validation_split)))
        val_samples.extend(label_samples[:validation_count])
        train_samples.extend(label_samples[validation_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def encode_label(label: str, class_names: list[str]) -> int:
    try:
        return class_names.index(label)
    except ValueError as exc:
        raise SystemExit(f"label '{label}' not found in class_names") from exc


def build_dataset(
    samples: list[SampleRecord],
    class_names: list[str],
    image_size: int,
    batch_size: int,
    training: bool,
    normalize_to_unit: bool,
    augment: bool = False,
) -> tf.data.Dataset:
    image_paths = [sample.image_path for sample in samples]
    labels = [encode_label(sample.label, class_names) for sample in samples]

    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32)
        if normalize_to_unit:
            image = image / 255.0
        if augment and training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.08)
            image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        return image, label

    dataset = path_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        dataset = dataset.shuffle(buffer_size=max(128, len(samples)))
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def class_weight_map(train_samples: list[SampleRecord], class_names: list[str]) -> dict[int, float]:
    counts = Counter(sample.label for sample in train_samples)
    total = sum(counts.values())
    num_classes = len(class_names)
    weights: dict[int, float] = {}
    for index, class_name in enumerate(class_names):
        count = counts.get(class_name, 0)
        if count == 0:
            continue
        weights[index] = total / (num_classes * count)
    return weights


def separable_block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    x = tf.keras.layers.SeparableConv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def representative_dataset_from_samples(
    samples: list[SampleRecord],
    image_size: int,
    sample_count: int,
    seed: int,
):
    per_label: dict[str, list[SampleRecord]] = {}
    for sample in samples:
        per_label.setdefault(sample.label, []).append(sample)

    rng = random.Random(seed)
    selected: list[SampleRecord] = []
    labels = sorted(per_label.keys())
    if labels:
        per_class_target = max(1, sample_count // len(labels))
        for label in labels:
            items = per_label[label][:]
            rng.shuffle(items)
            selected.extend(items[:per_class_target])

    if len(selected) < sample_count:
        remaining = [sample for sample in samples if sample not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, sample_count - len(selected))])

    selected = selected[:sample_count]
    for sample in selected:
        image_bytes = tf.io.read_file(sample.image_path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]


def build_model(architecture: str, input_shape: tuple[int, int, int], num_classes: int, width_multiplier: float, pretrained: bool) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal", name="aug_flip")(x)
    x = tf.keras.layers.RandomRotation(0.12, name="aug_rot")(x)
    x = tf.keras.layers.RandomZoom(0.12, name="aug_zoom")(x)
    x = tf.keras.layers.RandomContrast(0.15, name="aug_contrast")(x)

    if architecture == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet" if pretrained else None,
            input_shape=input_shape,
            alpha=width_multiplier,
            pooling=None,
            name="mobilenetv2_backbone",
        )
        base_model.trainable = not pretrained
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        return tf.keras.Model(inputs, outputs, name="smart_pantry_mobilenetv2")

    if architecture == "cnn":
        x = inputs
        x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        return tf.keras.Model(inputs, outputs, name="smart_pantry_cnn")

    if architecture == "separable_cnn":
        x = inputs
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
        x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
        x = tf.keras.layers.Activation("relu", name="stem_relu")(x)

        x = separable_block(x, 48, 1, "block1")
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name="block1_pool")(x)
        x = separable_block(x, 64, 1, "block2")
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name="block2_pool")(x)
        x = separable_block(x, 96, 1, "block3")
        x = separable_block(x, 128, 1, "block4")
        x = tf.keras.layers.Dropout(0.25, name="sep_dropout")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="sep_gap")(x)
        x = tf.keras.layers.Dense(64, activation="relu", name="sep_head_dense")(x)
        x = tf.keras.layers.Dropout(0.25, name="sep_head_dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="sep_logits")(x)
        return tf.keras.Model(inputs, outputs, name="smart_pantry_separable_cnn")

    raise ValueError(f"unknown architecture: {architecture}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact TinyML image classifier.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with one subfolder per class.")
    parser.add_argument("--manifest", type=Path, default=None, help="JSONL manifest generated by data/assemble_dataset.py.")
    parser.add_argument("--class-map", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "label_map.json", help="JSON file defining the class order.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and exported artifacts.")
    parser.add_argument("--architecture", choices=("mobilenetv2", "cnn", "separable_cnn"), default="mobilenetv2")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width-multiplier", type=float, default=0.35)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use ImageNet pretrained weights for MobileNetV2.")
    parser.add_argument("--fine-tune-epochs", type=int, default=8, help="Additional epochs after unfreezing the backbone.")
    parser.add_argument("--fine-tune-layers", type=int, default=40, help="Number of MobileNetV2 layers to unfreeze during fine-tuning.")
    parser.add_argument("--qat", action=argparse.BooleanOptionalAction, default=False, help="Run quantization-aware training after float training.")
    parser.add_argument("--qat-from", type=Path, default=None, help="Optional path to a pretrained float Keras model for QAT-only runs.")
    parser.add_argument("--qat-epochs", type=int, default=6, help="QAT epochs (default: 6).")
    parser.add_argument("--qat-learning-rate", type=float, default=1e-5, help="Learning rate for QAT stage.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest is None and args.data_dir is None:
        raise SystemExit("provide either --manifest or --data-dir")
    if args.manifest is not None and not args.manifest.exists():
        raise SystemExit(f"manifest does not exist: {args.manifest}")
    if args.data_dir is not None and not args.data_dir.exists():
        raise SystemExit(f"data directory does not exist: {args.data_dir}")
    if args.qat_from is not None and not args.qat:
        raise SystemExit("--qat-from requires --qat")
    if args.qat_from is not None and not args.qat_from.exists():
        raise SystemExit(f"qat-from model does not exist: {args.qat_from}")

    class_names = load_class_names(args.class_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        samples = read_manifest(args.manifest)
    else:
        samples = read_directory_samples(args.data_dir)

    if not samples:
        raise SystemExit("no training samples found")

    if not class_names:
        class_names = sorted({sample.label for sample in samples})

    train_samples, val_samples = split_samples(samples, args.validation_split, args.seed)
    normalize_to_unit = args.architecture in {"cnn", "separable_cnn"}
    train_ds = build_dataset(
        train_samples,
        class_names,
        args.image_size,
        args.batch_size,
        training=True,
        normalize_to_unit=normalize_to_unit,
        augment=args.architecture == "separable_cnn",
    )
    val_ds = build_dataset(
        val_samples,
        class_names,
        args.image_size,
        args.batch_size,
        training=False,
        normalize_to_unit=normalize_to_unit,
        augment=False,
    )
    weights = class_weight_map(train_samples, class_names)

    checkpoint_path = args.output_dir / "best_model.keras"
    final_model_path = args.output_dir / "final_model.keras"
    qat_checkpoint_path = args.output_dir / "qat_best_model.keras"
    qat_final_model_path = args.output_dir / "qat_final_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    ]

    history_parts: list[dict[str, list[float]]] = []
    best_model_output = None
    final_model_output = None

    if args.qat_from is not None:
        try:
            model = tf.keras.models.load_model(args.qat_from)
        except Exception as exc:
            raise SystemExit(
                "failed to load --qat-from model under legacy tf.keras. "
                "Re-train float model with current train.py first, then run QAT."
            ) from exc
        best_model_output = str(args.qat_from.resolve())
        final_model_output = str(args.qat_from.resolve())
    else:
        model = build_model(args.architecture, (args.image_size, args.image_size, 3), len(class_names), args.width_multiplier, pretrained=args.pretrained)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            class_weight=weights,
        )
        history_parts.append(history.history)

        if args.architecture == "mobilenetv2" and args.pretrained and args.fine_tune_epochs > 0:
            base_model = model.get_layer("mobilenetv2_backbone")
            base_model.trainable = True
            freeze_until = max(0, len(base_model.layers) - args.fine_tune_layers)
            for layer in base_model.layers[:freeze_until]:
                layer.trainable = False
            for layer in base_model.layers[freeze_until:]:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
            fine_tune_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.fine_tune_epochs,
                callbacks=callbacks,
                class_weight=weights,
            )
            history_parts.append(fine_tune_history.history)

        model.save(str(final_model_path))
        best_model_output = str(checkpoint_path.resolve())
        final_model_output = str(final_model_path.resolve())

    qat_best_model = None
    qat_final_model = None
    qat_saved_model = None
    qat_tflite_model = None
    if args.qat:
        tfmot = import_tfmot()
        quantize_model = tfmot.quantization.keras.quantize_model
        nested_models = [layer.name for layer in model.layers if isinstance(layer, tf.keras.Model)]
        if nested_models:
            raise SystemExit(
                "QAT is not supported for models with nested submodels in this pipeline "
                f"(found: {nested_models}). Use --architecture cnn for QAT runs."
            )
        try:
            qat_model = quantize_model(model)
        except Exception as exc:
            raise SystemExit(f"failed to quantize model for QAT: {exc}") from exc

        qat_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(qat_checkpoint_path), monitor="val_accuracy", save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        ]
        qat_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.qat_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        qat_history = qat_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.qat_epochs,
            callbacks=qat_callbacks,
            class_weight=weights,
        )
        history_parts.append({f"qat_{k}": v for k, v in qat_history.history.items()})
        qat_model.save(str(qat_final_model_path))
        qat_tflite_path = args.output_dir / "qat_model.tflite"
        qat_converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        qat_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        qat_converter.representative_dataset = lambda: representative_dataset_from_samples(
            train_samples,
            args.image_size,
            min(240, len(train_samples)),
            args.seed,
        )
        qat_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        qat_converter.inference_input_type = tf.int8
        qat_converter.inference_output_type = tf.int8
        qat_tflite_bytes = qat_converter.convert()
        qat_tflite_path.write_bytes(qat_tflite_bytes)
        qat_best_model = str(qat_checkpoint_path.resolve())
        qat_final_model = str(qat_final_model_path.resolve())
        qat_tflite_model = str(qat_tflite_path.resolve())

    with (args.output_dir / "classes.json").open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)

    merged_history: dict[str, list[float]] = {}
    for part in history_parts:
        for key, values in part.items():
            merged_history.setdefault(key, []).extend(float(value) for value in values)

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(merged_history, handle, indent=2)

    print(json.dumps({
        "classes": class_names,
        "best_model": best_model_output,
        "final_model": final_model_output,
        "qat_best_model": qat_best_model,
        "qat_final_model": qat_final_model,
        "qat_saved_model": qat_saved_model,
        "qat_tflite_model": qat_tflite_model,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "class_weighted": bool(weights),
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
