"""Validation node for ensuring placements cover all objects."""

from __future__ import annotations

from typing import Callable

from agentic.state import LayoutState


def build_validator_node(required_ids: list[int]) -> Callable[[LayoutState], LayoutState]:
    def node(state: LayoutState) -> LayoutState:
        missing = [oid for oid in required_ids if oid not in state.placements]
        if missing:
            state.validation_errors.append(
                f"Missing placements for object ids: {missing}"
            )
            raise ValueError("Coverage validation failed")
        state.phase = "validated"
        return state

    return node


