from __future__ import annotations

from typing import List


# Canonical list of allowed labels for manual annotation
ALLOWED_LABELS: List[str] = [
    "button",
    "photo",
    "design element",
    "text",
    "logo",
    "cta",
]


def normalize_label(value: str) -> str:
    """Normalize a label to canonical lowercase form without leading/trailing spaces."""
    return (value or "").strip().lower()


def is_allowed_label(value: str) -> bool:
    return normalize_label(value) in ALLOWED_LABELS


def compute_per_label_numbers(labels: List[str]) -> List[int]:
    """
    For a parallel list of labels, compute contiguous per-label numbering.
    Example: ["photo","text","photo"] -> [1,1,2]
    """
    counters: dict[str, int] = {}
    numbers: List[int] = []
    for lab in labels:
        key = normalize_label(lab)
        counters[key] = counters.get(key, 0) + 1
        numbers.append(counters[key])
    return numbers


