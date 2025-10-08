"""Compositor node that renders placements into an image."""

from __future__ import annotations

from typing import Callable

from PIL import Image

from agentic.state import LayoutState
from agentic.utils.artifacts import ensure_iteration_dirs, serialize_placements, write_json
from background_resizing import fill_solid


def build_compositor_node() -> Callable[[LayoutState], LayoutState]:
    def node(state: LayoutState) -> LayoutState:
        state.ensure_placements()
        background = fill_solid(str(state.background_path), state.canvas_size)
        object_images = {}
        for oid, meta in state.objects.items():
            path = state.objects_dir / meta.filename
            object_images[oid] = Image.open(path).convert("RGBA")
        placements = [
            {
                "object_id": placement.object_id,
                "box": [
                    placement.x,
                    placement.y,
                    placement.x + placement.width,
                    placement.y + placement.height,
                ],
            }
            for placement in state.placements.values()
        ]
        artifact_dirs = ensure_iteration_dirs(state)

        canvas = background.copy()
        for placement in state.placements.values():
            obj_image = object_images[placement.object_id]
            if obj_image.size != (placement.width, placement.height):
                raise ValueError(
                    "Placement size mismatch; scaling objects is not permitted"
                )
            canvas.alpha_composite(obj_image, dest=(placement.x, placement.y))
        final_dir = artifact_dirs["final_product"]
        out_path = final_dir / f"draft_macro_iter_{state.iteration:02d}.png"
        canvas.save(out_path)
        state.current_composite_path = out_path

        placements_path = artifact_dirs["layout_json"] / f"layout_macro_iter_{state.iteration:02d}.json"
        write_json(placements_path, {"placements": serialize_placements(state.placements)})

        return state

    return node

