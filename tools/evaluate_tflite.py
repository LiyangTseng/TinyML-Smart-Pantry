#!/usr/bin/env python3
"""Evaluate a TFLite model against the manifest (handles INT8 inputs/outputs).

Usage:
  python tools/evaluate_tflite.py --tflite artifacts/model/model.tflite --manifest artifacts/manifest.jsonl --image-size 96
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import sys
import random

try:
    import tensorflow as tf
except Exception as e:
    print('TensorFlow import failed:', e)
    sys.exit(2)


def load_manifest(manifest_path):
    with open(manifest_path, 'r') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows


def build_classes(rows, classes_file=None):
    if classes_file:
        return json.loads(Path(classes_file).read_text())
    seen = []
    for r in rows:
        lab = r.get('canonical_label') or r.get('label')
        if lab and lab not in seen:
            seen.append(lab)
    return seen


def split_like_training(rows, validation_split, seed):
    if not 0.0 < validation_split < 1.0:
        raise ValueError('validation_split must be between 0 and 1')

    per_label = {}
    for r in rows:
        label = r.get('canonical_label') or r.get('label')
        per_label.setdefault(label, []).append(r)

    rng = random.Random(seed)
    val = []
    train = []
    for label in sorted(per_label):
        label_rows = per_label[label][:]
        rng.shuffle(label_rows)
        validation_count = max(1, int(round(len(label_rows) * validation_split)))
        val.extend(label_rows[:validation_count])
        train.extend(label_rows[validation_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def quantize_input(x_float, scale, zero_point, dtype):
    # x_float is model-input-domain float32, then quantized using tensor params.
    q = np.round(x_float / scale + zero_point).astype(np.int32)
    if dtype == np.int8:
        q = np.clip(q, -128, 127).astype(np.int8)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255).astype(np.uint8)
    else:
        q = q.astype(dtype)
    return q


def dequantize_output(q, scale, zero_point):
    return (q.astype(np.float32) - zero_point) * scale


def preprocess_image_for_model(img_path, image_size, input_shape):
    # TFLite uses NHWC for image models. We adapt channels to model expectation.
    if len(input_shape) != 4:
        raise ValueError(f'Expected rank-4 input tensor, got shape: {input_shape}')

    expected_channels = int(input_shape[3])
    if expected_channels == 1:
        img = Image.open(img_path).convert('L').resize((image_size, image_size))
        x = np.array(img).astype(np.float32)
        x = np.expand_dims(x, axis=-1)
    elif expected_channels == 3:
        img = Image.open(img_path).convert('RGB').resize((image_size, image_size))
        x = np.array(img).astype(np.float32)
    else:
        raise ValueError(
            f'Unsupported input channel count in model: {expected_channels}. '
            'Expected 1 or 3.'
        )

    return np.expand_dims(x, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tflite', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--classes-file', default=None)
    p.add_argument('--val-size', type=int, default=300,
                   help='Number of last manifest rows to use as validation (default: 300)')
    p.add_argument(
        '--split-mode',
        choices=('train_compatible', 'tail'),
        default='train_compatible',
        help='How to choose validation samples: train_compatible reproduces models/train.py split.',
    )
    p.add_argument('--validation-split', type=float, default=0.2,
                   help='Validation split used by train_compatible mode (default: 0.2)')
    p.add_argument('--seed', type=int, default=42,
                   help='Seed used by train_compatible mode (default: 42)')
    p.add_argument(
        '--input-mode',
        choices=('auto', 'zero_one', 'zero_255'),
        default='auto',
        help='Input float domain before quantization. auto: infer from input quantization scale.',
    )
    args = p.parse_args()

    tflite_path = Path(args.tflite)
    manifest_path = Path(args.manifest)
    rows = load_manifest(manifest_path)
    if len(rows) == 0:
        print('Empty manifest:', manifest_path)
        return

    classes = build_classes(rows, args.classes_file)
    print('Using', len(classes), 'classes')

    if args.split_mode == 'train_compatible':
        _train, val = split_like_training(rows, args.validation_split, args.seed)
    else:
        if args.val_size and len(rows) >= args.val_size:
            val = rows[-args.val_size:]
        else:
            val = rows[int(len(rows) * 0.8):]

    interp = tf.lite.Interpreter(str(tflite_path))
    interp.allocate_tensors()
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]

    inp_dtype = np.dtype(input_details['dtype'])
    out_dtype = np.dtype(output_details['dtype'])
    inp_q = input_details.get('quantization', (0.0, 0))
    out_q = output_details.get('quantization', (0.0, 0))
    inp_scale, inp_zp = inp_q
    out_scale, out_zp = out_q

    print('TFLite input dtype', inp_dtype, 'quant', inp_q)
    print('TFLite output dtype', out_dtype, 'quant', out_q)
    print('TFLite input shape', tuple(input_details.get('shape', [])))

    if args.input_mode == 'auto':
        # Heuristic: int8 input with scale near 1 usually means 0..255 domain.
        if inp_dtype in (np.int8, np.uint8) and 0.75 <= float(inp_scale) <= 1.25:
            effective_input_mode = 'zero_255'
        else:
            effective_input_mode = 'zero_one'
    else:
        effective_input_mode = args.input_mode
    print('Input preprocessing mode:', effective_input_mode)

    correct = 0
    total = 0
    input_shape = tuple(input_details.get('shape', []))
    for r in val:
        img_path = Path(r['image_path'])
        if not img_path.exists():
            continue
        x = preprocess_image_for_model(img_path, args.image_size, input_shape)
        if effective_input_mode == 'zero_one':
            x = x / 255.0

        if inp_dtype in (np.int8, np.uint8):
            qx = quantize_input(x, inp_scale if inp_scale != 0 else 1.0, int(inp_zp), inp_dtype)
            interp.set_tensor(input_details['index'], qx)
        else:
            interp.set_tensor(input_details['index'], x.astype(inp_dtype))

        interp.invoke()
        out = interp.get_tensor(output_details['index'])
        if out_scale and out_scale != 0:
            out = dequantize_output(out, out_scale, int(out_zp))

        pred = int(np.argmax(out, axis=-1).squeeze())
        true_label = r.get('canonical_label') or r.get('label')
        try:
            true_idx = classes.index(true_label)
        except ValueError:
            # fallback: if manifest contains numeric index
            true_idx = r.get('canonical_label_index') or r.get('label_index') or 0

        if pred == int(true_idx):
            correct += 1
        total += 1

    if total == 0:
        print('No validation samples found')
        return
    print(f'Validation samples: {total}  Accuracy: {correct/total:.4f}')


if __name__ == '__main__':
    main()
