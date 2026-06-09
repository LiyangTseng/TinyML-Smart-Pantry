from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    canonical_label: str
    source_label: str
    source_root: str

    def to_json(self) -> str:
        return json.dumps({
            "image_path": self.image_path,
            "canonical_label": self.canonical_label,
            "source_label": self.source_label,
            "source_root": self.source_root,
        }, ensure_ascii=True)


def load_label_map(label_map_path: Path) -> dict[str, str]:
    with label_map_path.open("r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)

    normalized: dict[str, str] = {}
    for canonical_label, aliases in raw_mapping.items():
        normalized[canonical_label.lower()] = canonical_label.lower()
        if isinstance(aliases, str):
            normalized[str(aliases).lower()] = canonical_label.lower()
            continue
        for alias in aliases:
            normalized[str(alias).lower()] = canonical_label.lower()
    return normalized


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def canonicalize_label(source_label: str, label_map: dict[str, str]) -> str | None:
    normalized = source_label.strip().lower().replace(" ", "_")
    if normalized in label_map:
        return label_map[normalized]
    return None


def build_manifest(source_root: Path, label_map_path: Path, output_manifest: Path) -> dict[str, int]:
    label_map = load_label_map(label_map_path)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_manifest.open("w", encoding="utf-8") as handle:
        for image_path in iter_image_files(source_root):
            source_label = image_path.parent.name
            canonical_label = canonicalize_label(source_label, label_map)
            if canonical_label is None:
                skipped += 1
                continue

            row = ManifestRow(
                image_path=str(image_path.resolve()),
                canonical_label=canonical_label,
                source_label=source_label,
                source_root=str(source_root.resolve()),
            )
            handle.write(row.to_json())
            handle.write("\n")
            written += 1

    return {"written": written, "skipped": skipped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a JSONL image manifest for the TinyML pantry project.")
    parser.add_argument("--source-root", type=Path, required=True, help="Root directory containing one subfolder per class.")
    parser.add_argument("--label-map", type=Path, default=Path(__file__).with_name("label_map.json"), help="Path to the label mapping JSON.")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Where to write the JSONL manifest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_root.exists():
        raise SystemExit(f"source root does not exist: {args.source_root}")
    if not args.label_map.exists():
        raise SystemExit(f"label map does not exist: {args.label_map}")

    summary = build_manifest(args.source_root, args.label_map, args.output_manifest)
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    print(json.dumps({"timestamp": timestamp, **summary, "output_manifest": str(args.output_manifest.resolve())}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
