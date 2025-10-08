"""Macro layouter tools that manipulate the Flex-DSL structure."""

from __future__ import annotations

from typing import Dict

from agentic.state import LayoutState
from agentic.utils import placements_from_flex


def _build_item(node: Dict, objects: Dict[int, object]) -> Dict:
    if "object_id" not in node:
        raise ValueError("Missing object_id in item")
    oid = int(node["object_id"])
    if oid not in objects:
        raise ValueError(f"Unknown object_id {oid}")
    name = node.get("name") or objects[oid].name
    return {
        "object_id": oid,
        "name": name,
    }


def _build_container(node: Dict, objects: Dict[int, object]) -> Dict:
    if node.get("type") != "flex":
        raise ValueError("Containers must have type=flex")
    direction = node.get("direction")
    if direction not in {"row", "column"}:
        raise ValueError("direction must be 'row' or 'column'")
    children = []
    for child in node.get("children", []):
        if "object_id" in child:
            children.append(_build_item(child, objects))
        else:
            children.append(_build_container(child, objects))
    if not children:
        raise ValueError("Containers must declare at least one child")
    return {
        "type": "flex",
        "direction": direction,
        "children": children,
    }


def set_flex_json(state: LayoutState, root: Dict, raw_text: str | None = None) -> str:
    """Replace the full Flex-DSL root container.

    Args:
        root: JSON object matching the DSL schema.
    """

    if root is None:
        raise ValueError("Flex JSON must include a root container")

    built = _build_container(root, state.objects)
    state.flex_json = {"root": built}
    state.flex_text = raw_text
    state.placements = placements_from_flex(state.flex_json, state.canvas_size, state.objects)
    return "Flex layout updated"



