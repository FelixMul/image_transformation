"""Helpers for creating and writing workflow artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agentic.state import LayoutState, PlacementState


def ensure_iteration_dirs(state: LayoutState) -> Dict[str, Path]:
    """Ensure the directory structure for the current iteration exists."""

    root = state.get_iteration_dir()

    subdirs = {
        "vlm_input_text": root / "vlm_input_text",
        "vlm_output": root / "vlm_output",
        "layout_json": root / "layout_json",
        "final_product": root / "final_product",
    }

    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return {"root": root, **subdirs}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def serialize_placements(placements: Dict[int, PlacementState]) -> List[Dict[str, int | str]]:
    ordered = sorted(placements.values(), key=lambda item: item.object_id)
    return [
        {
            "object_id": placement.object_id,
            "name": placement.name,
            "x": placement.x,
            "y": placement.y,
            "width": placement.width,
            "height": placement.height,
        }
        for placement in ordered
    ]


__all__ = [
    "ensure_iteration_dirs",
    "write_text",
    "write_json",
    "serialize_placements",
]


