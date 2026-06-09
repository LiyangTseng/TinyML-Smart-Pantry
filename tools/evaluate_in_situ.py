from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class PredictionRow:
    image_path: str
    true_label: str
    predicted_label: str
    confidence: float | None = None


def derive_true_label(image_path: str) -> str:
    return Path(image_path).parent.name


def read_predictions(path: Path) -> list[PredictionRow]:
    if path.suffix.lower() == ".csv":
        rows: list[PredictionRow] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw_row in reader:
                image_path = str(raw_row.get("image_path", "")).strip()
                true_label = str(raw_row.get("true_label", "")).strip() or derive_true_label(image_path)
                predicted_label = str(raw_row.get("predicted_label", "")).strip()
                confidence_value = raw_row.get("confidence")
                confidence = None if confidence_value in (None, "") else float(confidence_value)
                rows.append(PredictionRow(image_path, true_label, predicted_label, confidence))
        return rows

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                image_path = str(payload.get("image_path", ""))
                true_label = str(payload.get("true_label", "")).strip() or derive_true_label(image_path)
                predicted_label = str(payload.get("predicted_label", "")).strip()
                confidence = payload.get("confidence")
                rows.append(PredictionRow(image_path, true_label, predicted_label, None if confidence is None else float(confidence)))
        return rows

    raise SystemExit("predictions file must be .csv or .jsonl")


def compute_confusion_matrix(rows: list[PredictionRow]) -> tuple[list[str], dict[str, dict[str, int]]]:
    labels = sorted({row.true_label for row in rows} | {row.predicted_label for row in rows})
    matrix: dict[str, dict[str, int]] = {label: {other: 0 for other in labels} for label in labels}
    for row in rows:
        matrix[row.true_label][row.predicted_label] += 1
    return labels, matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate in-situ predictions for the TinyML pantry project.")
    parser.add_argument("--input", type=Path, required=True, help="CSV or JSONL file containing predictions.")
    parser.add_argument("--output-summary", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--output-confusion", type=Path, default=None, help="Optional CSV confusion matrix output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"input file does not exist: {args.input}")

    rows = read_predictions(args.input)
    if not rows:
        raise SystemExit("no prediction rows were found")

    correct = sum(1 for row in rows if row.true_label == row.predicted_label)
    accuracy = correct / len(rows)
    labels, matrix = compute_confusion_matrix(rows)
    per_class_counts = Counter(row.true_label for row in rows)

    summary = {
        "rows": len(rows),
        "correct": correct,
        "accuracy": accuracy,
        "labels": labels,
        "per_class_counts": dict(per_class_counts),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.output_summary is not None:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.output_confusion is not None:
        args.output_confusion.parent.mkdir(parents=True, exist_ok=True)
        with args.output_confusion.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_label", *labels])
            for true_label in labels:
                writer.writerow([true_label, *[matrix[true_label][predicted_label] for predicted_label in labels]])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
