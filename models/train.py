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
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS and not image_path.name.startswith("._"):
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
        # Decoding as 3 channels (RGB) to retain color features
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32)
        
        if normalize_to_unit:
            image = image / 255.0
            
        if augment and training:
            # --- Spatial / Geometric Augmentations ---
            image = tf.image.random_flip_left_right(image)
            
            scales = tf.random.uniform([], 0.85, 1.0)
            boxes = tf.stack([
                (1.0 - scales) / 2.0, (1.0 - scales) / 2.0,
                (1.0 + scales) / 2.0, (1.0 + scales) / 2.0
            ])
            image = tf.image.crop_and_resize(
                tf.expand_dims(image, 0), 
                tf.expand_dims(boxes, 0), 
                box_indices=[0], 
                crop_size=(image_size, image_size)
            )[0]

            # --- Color / Lighting Augmentations ---
            image = tf.image.random_brightness(image, max_delta=0.12)
            image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
            # Clip pixel values to [0.0, 1.0] to prevent out-of-bounds training data
            image = tf.clip_by_value(image, 0.0, 1.0)
            
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


def representative_dataset_from_samples(
    samples: list[SampleRecord],
    image_size: int,
    sample_count: int,
    seed: int,
    grayscale: bool = False,
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
        image = tf.cast(image, tf.float32) / 255.0
        if grayscale:
            image = tf.image.rgb_to_grayscale(image)
        image = tf.expand_dims(image, 0)
        yield [image]


def build_model(architecture: str, input_shape: tuple[int, int, int], num_classes: int, width_multiplier: float, pretrained: bool) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    
    if architecture == "micro_cnn":
        x = inputs
        # Upgraded to 4 layers (24-48-96-96 channels) for higher capacity.
        # Strides of 2 in the first two layers keep activation RAM under 60 KB.
        x = tf.keras.layers.Conv2D(24, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(48, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(96, 3, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(96, 3, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
        outputs = tf.keras.layers.Activation("softmax", name="softmax")(logits)
        return tf.keras.Model(inputs, outputs, name="smart_pantry_micro_cnn")
    
    raise ValueError(f"Unknown architecture: {architecture}")


def build_teacher_model(input_shape: tuple[int, int, int], num_classes: int, width_multiplier: float, pretrained: bool) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=input_shape,
        alpha=width_multiplier,
        pooling=None,
    )
    base_model._name = "mobilenetv2_backbone"
    base_model.trainable = False  # Freeze feature extractor initially
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    outputs = tf.keras.layers.Activation("softmax", name="softmax")(logits)
    return tf.keras.Model(inputs, outputs, name="smart_pantry_teacher_mobilenetv2")


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=3.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        
        # Create helper sub-models to extract logits and softmax outputs simultaneously
        self.student_train_model = tf.keras.Model(
            inputs=self.student.inputs,
            outputs=[self.student.get_layer("logits").output, self.student.output]
        )
        self.teacher_train_model = tf.keras.Model(
            inputs=self.teacher.inputs,
            outputs=[self.teacher.get_layer("logits").output, self.teacher.output]
        )

    def compile(self, optimizer, metrics, student_loss_fn):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")

    @property
    def metrics(self):
        return super().metrics + [self.student_loss_tracker, self.distillation_loss_tracker]

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        teacher_input = x
        student_input = tf.image.rgb_to_grayscale(x)

        # Forward pass of teacher
        teacher_logits, teacher_predictions = self.teacher_train_model(teacher_input, training=False)

        with tf.GradientTape() as tape:
            student_logits, student_predictions = self.student_train_model(student_input, training=True)

            student_loss = self.student_loss_fn(y, student_predictions, sample_weight=sample_weight)
            
            # KL divergence distillation loss (on softened logits!)
            distillation_loss = tf.keras.losses.kl_divergence(
                tf.nn.softmax(teacher_logits / self.temperature, axis=-1),
                tf.nn.softmax(student_logits / self.temperature, axis=-1)
            )

            if sample_weight is not None:
                distillation_loss = distillation_loss * tf.cast(sample_weight, distillation_loss.dtype)

            total_loss = (self.alpha * student_loss) + ((1.0 - self.alpha) * (self.temperature ** 2) * tf.reduce_mean(distillation_loss))

        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)

        results = {m.name: m.result() for m in self.metrics}
        return results

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        student_input = tf.image.rgb_to_grayscale(x)
        student_predictions = self.student(student_input, training=False)

        student_loss = self.student_loss_fn(y, student_predictions, sample_weight=sample_weight)

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        self.student_loss_tracker.update_state(student_loss)

        results = {m.name: m.result() for m in self.metrics}
        return results


class StudentCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, student_model):
        super().__init__()
        self.filepath = filepath
        self.student_model = student_model
        self.best_val_acc = -1.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.student_model.save(self.filepath)
            print(f"\nSaved best student model to {self.filepath} with val_accuracy: {val_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact TinyML image classifier.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with one subfolder per class.")
    parser.add_argument("--manifest", type=Path, default=None, help="JSONL manifest generated by data/assemble_dataset.py.")
    parser.add_argument("--class-map", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "label_map.json", help="JSON file defining the class order.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and exported artifacts.")
    parser.add_argument("--architecture", choices=("mobilenetv2", "cnn", "micro_cnn", "separable_cnn"), default="micro_cnn")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width-multiplier", type=float, default=0.35)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use ImageNet pretrained weights for MobileNetV2.")
    parser.add_argument("--fine-tune-epochs", type=int, default=8, help="Additional epochs after unfreezing the backbone.")
    parser.add_argument("--fine-tune-layers", type=int, default=40, help="Number of MobileNetV2 layers to unfreeze during fine-tuning.")
    
    # Grayscale & Distillation Arguments
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=False, help="Train a grayscale model.")
    parser.add_argument("--distill", action=argparse.BooleanOptionalAction, default=False, help="Train using knowledge distillation.")
    parser.add_argument("--teacher-epochs", type=int, default=12, help="Number of epochs to train the teacher model.")
    parser.add_argument("--temperature", type=float, default=3.0, help="Temperature for distillation.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha weight for distillation loss (weight for student CE loss).")

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
    
    # We load datasets as RGB. In non-distill modes, if grayscale is True, we will map them.
    # In distill mode, distiller handles grayscale mapping inside train_step/test_step.
    normalize_to_unit = args.architecture in {"cnn", "micro_cnn"}
    train_ds = build_dataset(
        train_samples,
        class_names,
        args.image_size,
        args.batch_size,
        training=True,
        normalize_to_unit=normalize_to_unit,
        augment=args.architecture in {"cnn", "micro_cnn"},
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

    checkpoint_path = args.output_dir / "best_model.h5"
    final_model_path = args.output_dir / "final_model.h5"
    qat_checkpoint_path = args.output_dir / "qat_best_model.h5"
    qat_final_model_path = args.output_dir / "qat_final_model.h5"

    grayscale = args.grayscale or args.distill
    num_channels = 1 if grayscale else 3

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
        if args.distill:
            # Distillation Workflow
            # 1. Train/fine-tune teacher model on RGB
            teacher_model = build_teacher_model((args.image_size, args.image_size, 3), len(class_names), args.width_multiplier, pretrained=args.pretrained)
            teacher_model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
            print("=== Training Teacher Model (MobileNetV2 RGB) ===")
            teacher_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.teacher_epochs,
                class_weight=weights,
            )

            # Optional: fine-tune teacher
            if args.pretrained and args.fine_tune_epochs > 0:
                print("=== Fine-tuning Teacher Model ===")
                base_model = teacher_model.get_layer("mobilenetv2_backbone")
                base_model.trainable = True
                freeze_until = max(0, len(base_model.layers) - args.fine_tune_layers)
                for layer in base_model.layers[:freeze_until]:
                    layer.trainable = False
                for layer in base_model.layers[freeze_until:]:
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = False
                
                teacher_model.compile(
                    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=["accuracy"],
                )
                teacher_model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=args.fine_tune_epochs,
                    class_weight=weights,
                )

            # 2. Build student model (Grayscale: 1 channel)
            model = build_model(args.architecture, (args.image_size, args.image_size, 1), len(class_names), args.width_multiplier, pretrained=args.pretrained)
            
            # 3. Instantiate distiller
            distiller = Distiller(student=model, teacher=teacher_model, temperature=args.temperature, alpha=args.alpha)
            distiller.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
                metrics=["accuracy"],
                student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
            )

            distill_callbacks = [
                StudentCheckpoint(str(checkpoint_path), model),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-5),
                tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
            ]

            print("=== Training Student Model via Knowledge Distillation (Grayscale) ===")
            history = distiller.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                callbacks=distill_callbacks,
                class_weight=weights,
            )
            history_parts.append(history.history)
            model.save(str(final_model_path))
            best_model_output = str(checkpoint_path.resolve())
            final_model_output = str(final_model_path.resolve())

        else:
            # Native Workflow
            if grayscale:
                # Map datasets to grayscale
                train_ds_native = train_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y), num_parallel_calls=AUTOTUNE)
                val_ds_native = val_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y), num_parallel_calls=AUTOTUNE)
            else:
                train_ds_native = train_ds
                val_ds_native = val_ds

            model = build_model(args.architecture, (args.image_size, args.image_size, num_channels), len(class_names), args.width_multiplier, pretrained=args.pretrained)
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", save_best_only=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-5),
                tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
            ]

            print("=== Training Native Model ===")
            history = model.fit(
                train_ds_native,
                validation_data=val_ds_native,
                epochs=args.epochs,
                callbacks=callbacks,
                class_weight=weights,
            )
            history_parts.append(history.history)
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
                f"(found: {nested_models}). Use --architecture cnn or micro_cnn for QAT runs."
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
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=args.qat_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        
        # Make sure QAT dataset has correct channels
        if grayscale:
            train_ds_qat = train_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y), num_parallel_calls=AUTOTUNE)
            val_ds_qat = val_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y), num_parallel_calls=AUTOTUNE)
        else:
            train_ds_qat = train_ds
            val_ds_qat = val_ds

        print("=== Fine-tuning with QAT ===")
        qat_history = qat_model.fit(
            train_ds_qat,
            validation_data=val_ds_qat,
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
            grayscale=grayscale,
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