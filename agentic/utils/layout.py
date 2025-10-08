"""Deterministic placement helpers for the agentic workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from agentic.state import ObjectMeta, PlacementState


@dataclass
class _Size:
    width: int
    height: int


def _clamp_non_negative(value: int, label: str) -> int:
    if value < 0:
        raise ValueError(f"{label} cannot be negative")
    return value


def _measure_node(node: Dict, objects: Dict[int, ObjectMeta]) -> _Size:
    if "object_id" in node:
        oid = int(node["object_id"])
        meta = objects[oid]
        return _Size(meta.width, meta.height)

    direction = node.get("direction")
    if direction not in {"row", "column"}:
        raise ValueError("direction must be 'row' or 'column'")
    children = node.get("children", [])
    if not children:
        raise ValueError("container must have at least one child")
    gap_px = _clamp_non_negative(int(node.get("gap_px", 0)), "gap_px")
    padding_px = _clamp_non_negative(int(node.get("padding_px", 0)), "padding_px")

    measurements: List[_Size] = [_measure_node(child, objects) for child in children]

    if direction == "row":
        total_w = sum(sz.width for sz in measurements) + gap_px * (len(measurements) - 1)
        total_h = max(sz.height for sz in measurements)
    else:
        total_w = max(sz.width for sz in measurements)
        total_h = sum(sz.height for sz in measurements) + gap_px * (len(measurements) - 1)

    total_w += 2 * padding_px
    total_h += 2 * padding_px
    return _Size(total_w, total_h)


def _place_node(
    node: Dict,
    origin: Tuple[int, int],
    objects: Dict[int, ObjectMeta],
    placements: Dict[int, PlacementState],
) -> _Size:
    if "object_id" in node:
        oid = int(node["object_id"])
        meta = objects[oid]
        x, y = origin
        placements[oid] = PlacementState(
            object_id=oid,
            name=meta.name,
            x=x,
            y=y,
            width=meta.width,
            height=meta.height,
        )
        return _Size(meta.width, meta.height)

    direction = node.get("direction")
    gap_px = _clamp_non_negative(int(node.get("gap_px", 0)), "gap_px")
    padding_px = _clamp_non_negative(int(node.get("padding_px", 0)), "padding_px")
    children = node.get("children", [])
    if not children:
        raise ValueError("container must have at least one child")

    inner_origin = (origin[0] + padding_px, origin[1] + padding_px)
    cursor_x, cursor_y = inner_origin
    measurements: List[_Size] = []

    for child in children:
        child_size = _place_node(child, (cursor_x, cursor_y), objects, placements)
        measurements.append(child_size)
        if direction == "row":
            cursor_x += child_size.width + gap_px
        else:
            cursor_y += child_size.height + gap_px

    if direction == "row":
        total_w = sum(sz.width for sz in measurements) + gap_px * (len(measurements) - 1)
        total_h = max(sz.height for sz in measurements)
    else:
        total_w = max(sz.width for sz in measurements)
        total_h = sum(sz.height for sz in measurements) + gap_px * (len(measurements) - 1)

    total_w += 2 * padding_px
    total_h += 2 * padding_px
    return _Size(total_w, total_h)


def placements_from_flex(
    flex: Dict,
    canvas_size: Tuple[int, int],
    objects: Dict[int, ObjectMeta],
) -> Dict[int, PlacementState]:
    if "root" not in flex:
        raise ValueError("Flex JSON must include 'root'")
    root = flex["root"]

    placements: Dict[int, PlacementState] = {}
    total = _place_node(root, (0, 0), objects, placements)
    if total.width > canvas_size[0] or total.height > canvas_size[1]:
        raise ValueError(
            "Flex DSL produces placements larger than canvas; revise macro layout"
        )
    missing = set(objects.keys()) - set(placements.keys())
    if missing:
        raise ValueError(f"Placement missing required object ids: {sorted(missing)}")
    return placements


