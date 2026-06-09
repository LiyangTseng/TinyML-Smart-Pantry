from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class ShelfLifeRule:
    label: str
    shelf_days: int


_SHELF_LIFE_RULES: dict[str, ShelfLifeRule] = {
    "apple_pie": ShelfLifeRule("apple_pie", 3),
    "banana_bread": ShelfLifeRule("banana_bread", 5),
    "bread_pudding": ShelfLifeRule("bread_pudding", 3),
    "caesar_salad": ShelfLifeRule("caesar_salad", 2),
    "cheesecake": ShelfLifeRule("cheesecake", 5),
    "deviled_eggs": ShelfLifeRule("deviled_eggs", 2),
    "pizza": ShelfLifeRule("pizza", 3),
}

_ALIASES: dict[str, str] = {
    "apple pie": "apple_pie",
    "apple_pies": "apple_pie",
    "banana bread": "banana_bread",
    "bread pudding": "bread_pudding",
    "caesar salad": "caesar_salad",
    "cheese cake": "cheesecake",
    "deviled egg": "deviled_eggs",
    "deviled eggs": "deviled_eggs",
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace(" ", "_")
    return _ALIASES.get(normalized, normalized)


def get_shelf_life_days(label: str) -> int | None:
    canonical_label = normalize_label(label)
    rule = _SHELF_LIFE_RULES.get(canonical_label)
    return None if rule is None else rule.shelf_days


def estimate_expiration(label: str, observed_at: datetime | None = None) -> datetime | None:
    shelf_life_days = get_shelf_life_days(label)
    if shelf_life_days is None:
        return None

    timestamp = observed_at or datetime.now(tz=timezone.utc)
    return timestamp + timedelta(days=shelf_life_days)
