"""Macro layouter LangGraph node."""

from __future__ import annotations

from typing import Callable, Dict

from langchain_core.runnables import Runnable

from agentic.state import LayoutState
from agentic.tools.macro_layouter import TOOL_REGISTRY
from agentic.utils import extract_json_object, load_prompt
from agentic.utils.artifacts import ensure_iteration_dirs, write_text, write_json


def _format_object_summary(state: LayoutState) -> str:
    lines = []
    for meta in state.objects.values():
        lines.append(
            f"- {meta.object_id}: {meta.name} ({meta.width}x{meta.height})"
        )
    return "\n".join(lines)


def build_macro_node(model: Runnable) -> Callable[[LayoutState], LayoutState]:
    """Return a node function that runs the macro layouter agent."""

    prompt_template = load_prompt("macro_layouter")

    def node(state: LayoutState) -> LayoutState:
        state.iteration = 0
        state.should_stop = False
        prompt = (
            prompt_template
            .replace("{{OBJECT_SUMMARY}}", _format_object_summary(state))
            .replace("{{CANVAS_WIDTH}}", str(state.canvas_size[0]))
            .replace("{{CANVAS_HEIGHT}}", str(state.canvas_size[1]))
            .replace("{{RATIO}}", state.ratio)
        )
        messages = state.messages + [
            {"role": "system", "content": prompt},
        ]
        response = model.invoke({"messages": messages})
        text = response.content if hasattr(response, "content") else str(response)
        state.messages.append({"role": "assistant", "content": text})
        state.flex_text = text
        state.last_macro_text = text

        # Persist artifacts to ease debugging
        artifact_dirs = ensure_iteration_dirs(state)
        write_text(artifact_dirs["vlm_input_text"] / f"planner_prompt_iter_{state.iteration:02d}.txt", prompt)
        write_text(artifact_dirs["vlm_output"] / f"vlm_raw_iter_{state.iteration:02d}.txt", text)
        try:
            json_obj: Dict = extract_json_object(text)
        except ValueError as exc:
            # Save failure details for post-mortem
            write_text(
                artifact_dirs["vlm_output"] / f"failed_output_iter_{state.iteration:02d}.txt",
                f"Parse error: {exc}\n\nRAW OUTPUT:\n{text}",
            )
            raise ValueError("Macro layouter must return JSON") from exc
        root = json_obj.get("root") if isinstance(json_obj, dict) else None
        if root is None and isinstance(json_obj, dict) and json_obj.get("type") == "flex" and "children" in json_obj:
            root = json_obj
        if root is None:
            # Save extracted JSON for inspection
            from json import dumps as _dumps
            try:
                extracted = _dumps(json_obj, indent=2)
            except Exception:
                extracted = str(json_obj)
            write_text(
                artifact_dirs["vlm_output"] / f"failed_output_iter_{state.iteration:02d}.txt",
                "Missing root container; expected {\"root\": {...}} or single flex container.\n\nExtracted JSON:\n" + extracted,
            )
            raise ValueError("Flex JSON must include a root container or be a single flex container")
        write_json(artifact_dirs["vlm_output"] / f"layout_flex_iter_{state.iteration:02d}.json", {"root": root})
        TOOL_REGISTRY["set_flex_json"](state, root, text)
        state.phase = "macro"
        return state

    return node


