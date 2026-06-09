from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import serial
except ImportError:  # pragma: no cover - optional dependency guard
    serial = None

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from usda_lookup import estimate_expiration, normalize_label


@dataclass(frozen=True)
class InventoryEntry:
    label: str
    confidence: float
    observed_at: str
    expires_at: str | None
    raw_message: str


def parse_message(message: str) -> tuple[str, float, float]:
    stripped = message.strip()
    if not stripped:
        raise ValueError("empty message")

    if stripped.startswith("{"):
        payload = json.loads(stripped)
        label = str(payload["label"])
        confidence = float(payload.get("confidence", 0.0))
        timestamp = float(payload.get("timestamp", datetime.now(tz=timezone.utc).timestamp()))
        return label, confidence, timestamp

    parts = [part.strip() for part in stripped.split(",")]
    if len(parts) < 2:
        raise ValueError(f"expected label,confidence[,timestamp] but received: {message!r}")

    label = parts[0]
    confidence = float(parts[1])
    timestamp = float(parts[2]) if len(parts) >= 3 and parts[2] else datetime.now(tz=timezone.utc).timestamp()
    return label, confidence, timestamp


def iter_messages_from_file(input_file: Path) -> Iterable[str]:
    with input_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if cleaned:
                yield cleaned


def iter_messages_from_serial(port: str, baud_rate: int) -> Iterable[str]:
    if serial is None:
        raise SystemExit("pyserial is required for serial input. Install requirements.txt first.")

    connection = serial.Serial(port, baud_rate, timeout=1)
    try:
        while True:
            line = connection.readline().decode("utf-8", errors="replace").strip()
            if line:
                yield line
    finally:
        connection.close()


def append_inventory_entry(inventory_path: Path, entry: InventoryEntry) -> None:
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry), ensure_ascii=True))
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive TinyML pantry events from serial or a test file.")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port for the MCU.")
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--input-file", type=Path, default=None, help="Optional file of sample messages to replay instead of serial.")
    parser.add_argument("--inventory-path", type=Path, required=True, help="JSONL file where inventory events will be appended.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.input_file is not None:
        message_source = iter_messages_from_file(args.input_file)
    else:
        message_source = iter_messages_from_serial(args.port, args.baud_rate)

    for message in message_source:
        label, confidence, timestamp = parse_message(message)
        observed_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        expires_at = estimate_expiration(normalize_label(label), observed_at)
        entry = InventoryEntry(
            label=normalize_label(label),
            confidence=confidence,
            observed_at=observed_at.isoformat(),
            expires_at=None if expires_at is None else expires_at.isoformat(),
            raw_message=message,
        )
        append_inventory_entry(args.inventory_path, entry)
        print(json.dumps(asdict(entry), ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
