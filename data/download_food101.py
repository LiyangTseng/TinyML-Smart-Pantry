from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import tensorflow as tf
import tensorflow_datasets as tfds


def save_image(np_img: np.ndarray, out_path: Path) -> None:
    img = Image.fromarray(np_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=90)


def load_requested_labels(label_map_path: Path, requested: list[str] | None) -> list[tuple[str, str]]:
    with label_map_path.open("r", encoding="utf-8") as handle:
        label_map = json.load(handle)

    selected: list[tuple[str, str]] = []
    if requested is None:
        requested = list(label_map.keys())

    for canonical_label in requested:
        if canonical_label not in label_map:
            raise SystemExit(f"unknown canonical label: {canonical_label}")
        target = label_map[canonical_label]
        if isinstance(target, list):
            if not target:
                raise SystemExit(f"label map entry is empty for: {canonical_label}")
            target_label = str(target[0])
        else:
            target_label = str(target)
        selected.append((canonical_label, target_label))
    return selected


def download_subset(output_dir: Path, selected: list[tuple[str, str]], max_per_class: int) -> None:
    ds, info = tfds.load("food101", split="train+validation", with_info=True, as_supervised=True)
    label_names = info.features["label"].names
    label_index = {label.lower(): label for label in label_names}

    # map canonical -> exact Food-101 labels
    mapping: Dict[str, str] = {}
    for canonical_label, target_label in selected:
        resolved = label_index.get(target_label.lower())
        if resolved is None:
            print(f"WARNING: no Food-101 label matched '{target_label}' for canonical label '{canonical_label}'")
            continue
        mapping[canonical_label] = resolved
        print(f"Mapping project label '{canonical_label}' -> Food-101 label '{resolved}'")

    counters: Dict[str, int] = {s: 0 for s in mapping}

    for example in tfds.as_numpy(ds):
        image, label_id = example
        label_name = label_names[int(label_id)]
        for canonical, mapped in mapping.items():
            if label_name == mapped and counters[canonical] < max_per_class:
                out_path = output_dir / canonical / f"{counters[canonical]:05d}.jpg"
                save_image(image, out_path)
                counters[canonical] += 1
                if all(v >= max_per_class for v in counters.values()):
                    print("Downloaded requested number of images for all classes.")
                    return

    print("Download complete. Counts:")
    for k, v in counters.items():
        print(f"  {k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a subset of Food-101 into class folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write class subfolders.")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated list of canonical classes to download (defaults to data/label_map.json keys).")
    parser.add_argument("--label-map", type=Path, default=Path(__file__).with_name("label_map.json"), help="Path to the canonical-to-Food-101 label mapping JSON.")
    parser.add_argument("--max-per-class", type=int, default=1000, help="Maximum images to download per class (Food-101 max is 1000 across train+validation).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_per_class <= 0:
        raise SystemExit("--max-per-class must be a positive integer")
    if args.max_per_class > 1000:
        print("WARNING: Food-101 provides at most 1000 images per class in train+validation. Clamping to 1000.")
        args.max_per_class = 1000

    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception:
        pass

    if not args.label_map.exists():
        raise SystemExit(f"label map does not exist: {args.label_map}")

    if args.classes is None:
        with args.label_map.open("r", encoding="utf-8") as fh:
            label_map = json.load(fh)
        requested = list(label_map.keys())
    else:
        requested = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not requested:
        raise SystemExit("no classes specified")

    selected = load_requested_labels(args.label_map, requested)
    print(f"Using classes from {args.label_map}: {[canonical for canonical, _ in selected]}")
    download_subset(output_dir, selected, args.max_per_class)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
