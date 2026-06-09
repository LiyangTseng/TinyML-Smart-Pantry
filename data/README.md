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
    python3 data/download_food101.py --output-dir data/source_root --max-per-class 250
    ```

2) Alternatively specify exact canonical classes (comma-separated):
    ```bash
    python3 data/download_food101.py --output-dir data/source_root --classes apple_pie,pizza,cheesecake --max-per-class 200
    ```

3) After you have a `source_root` with one folder per class, create a JSONL manifest:
    ```bash
    python3 data/assemble_dataset.py --source-root data/source_root --output-manifest artifacts/manifest.jsonl
    ```

The manifest is a newline-delimited JSON (JSONL) file where each line represents a single example:

```json
{ "image_path": "/abs/path/to/image.jpg", "canonical_label": "apple_pie", "source_label": "apple_pie", "source_root": "/abs/path/to/source_root" }
```

Training and conversion scripts can either consume the folder layout directly (see `models/train.py`) or read the manifest created above depending on your workflow. The manifest provides a simple, reproducible index of images and canonical labels used for experiments.

## Scope note

The repo now uses dish classification end-to-end. That means the model predicts labels such as `apple_pie`, `caesar_salad`, and `pizza`, and the host-side shelf-life lookup is keyed to those same dish labels.
