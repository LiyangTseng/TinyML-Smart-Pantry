# Data Assets

This directory holds the dataset assembly helpers and the project label mapping.

## Canonical labels

This project now targets Food-101 dish labels rather than raw ingredients. The starter six-class set is:

- apple_pie
- bread_pudding
- caesar_salad
- cheesecake
- deviled_eggs
- pizza

## Expected source layout

For the first implementation pass, each class should be stored in its own folder:

```text
source-root/
    apple_pie/
    bread_pudding/
    caesar_salad/
    cheesecake/
    deviled_eggs/
    pizza/
```

The `assemble_dataset.py` script scans that tree and writes a JSONL manifest for training and evaluation.

## Using the downloader and manifest

We provide a helper to fetch a subset of Food-101 into the local folder layout used by `assemble_dataset.py`.

1) Download a subset of Food-101 (defaults to the dish labels in `label_map.json`):
    ```bash
    python data/download_food101.py \
    --output-dir artifacts/food101_full \
    --label-map data/label_map.json \
    --max-per-class 1000
    ```

2) Alternatively specify exact canonical classes (comma-separated):
    ```bash
    python3 data/download_food101.py --output-dir data/source_root --classes apple_pie,pizza,cheesecake --max-per-class 1000
    ```

3) After you have a source root with one folder per class, create a JSONL manifest:
    ```bash
    python3 data/assemble_dataset.py --source-root artifacts/food101_full --output-manifest artifacts/manifest.jsonl
    ```

Important:
- Keep source root and manifest aligned. If you download into artifacts/food101_full, build the manifest from artifacts/food101_full.
- A stale manifest pointing to an older folder (for example data/source_root) will train/evaluate on the wrong dataset size.

The manifest is a newline-delimited JSON (JSONL) file where each line represents a single example:

```json
{ "image_path": "/abs/path/to/image.jpg", "canonical_label": "apple_pie", "source_label": "apple_pie", "source_root": "/abs/path/to/source_root" }
```

The training and evaluation workflow in this project uses the manifest directly:
- train: models/train.py --manifest artifacts/manifest.jsonl
- eval: tools/evaluate_tflite.py --manifest artifacts/manifest.jsonl

The manifest provides a simple, reproducible index of images and canonical labels used for experiments.

## Scope note

The repo now uses dish classification end-to-end. That means the model predicts labels such as `apple_pie`, `caesar_salad`, and `pizza`, and the host-side shelf-life lookup is keyed to those same dish labels.
