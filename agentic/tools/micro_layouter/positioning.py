"""Tools that move objects in absolute pixel space."""

from __future__ import annotations

from agentic.state import LayoutState, PlacementState


def _resolve_object(state: LayoutState, object_identifier: str) -> PlacementState:
    """Resolve an object either by object_id or name (case-insensitive)."""

    state.ensure_placements()

    # Try numeric id first
    if object_identifier.isdigit():
        oid = int(object_identifier)
        placement = state.placements.get(oid)
        if placement is None:
            raise ValueError(f"Object id {oid} has no placement yet")
        return placement

    # Fallback to name lookup
    identifier_lower = object_identifier.lower()
    for placement in state.placements.values():
        if placement.name.lower() == identifier_lower:
            return placement
    raise ValueError(f"No placement found for '{object_identifier}'")


def _format_response(placement: PlacementState) -> str:
    return (
        f"Placement for {placement.name} (id={placement.object_id}) now at "
        f"({placement.x}, {placement.y})"
    )


def adjust_y(state: LayoutState, object: str, pixels: int) -> str:
    """Move an object vertically.

    Positive pixels move the object *down*, negative pixels move it *up*.
    """

    target = _resolve_object(state, str(object))
    target.move_dy(int(pixels))
    return _format_response(target)


def adjust_x(state: LayoutState, object: str, pixels: int) -> str:
    """Move an object horizontally.

    Positive pixels move the object *right*, negative pixels move it *left*.
    """

    target = _resolve_object(state, str(object))
    target.move_dx(int(pixels))
    return _format_response(target)


