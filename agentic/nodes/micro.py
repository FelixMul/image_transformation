"""Micro layouter LangGraph node."""

from __future__ import annotations

import json
from typing import Callable, Dict, List

from langchain_core.runnables import Runnable

from agentic.state import LayoutState
from agentic.tools.micro_layouter import TOOL_REGISTRY, TOOL_DEFINITIONS
from agentic.utils import load_prompt


def _format_current_placements(state: LayoutState) -> str:
    if not state.placements:
        return "No placements available yet."

    lines: List[str] = []
    for placement in sorted(state.placements.values(), key=lambda p: p.object_id):
        lines.append(
            (
                f"- {placement.name} (id={placement.object_id}) "
                f"@ ({placement.x}, {placement.y}) size={placement.width}x{placement.height}"
            )
        )
    return "\n".join(lines)


def _format_feedback(state: LayoutState) -> str:
    if state.critic_notes:
        return state.critic_notes[-1]
    return state.last_critic_text or "No critic feedback provided."


def _parse_tool_args(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON arguments for tool call: {raw}") from exc


def build_micro_node(model: Runnable) -> Callable[[LayoutState], LayoutState]:
    """Return a node function that runs the micro layouter agent."""

    prompt_template = load_prompt("micro_layouter")

    def node(state: LayoutState) -> LayoutState:
        state.ensure_placements()

        prompt = (
            prompt_template
            .replace("{{CURRENT_PLACEMENTS}}", _format_current_placements(state))
            .replace("{{CRITIC_FEEDBACK}}", _format_feedback(state))
        )

        messages = state.messages + [
            {"role": "system", "content": prompt},
        ]

        response = model.invoke({
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
        })

        text = getattr(response, "content", "")
        tool_calls = list(getattr(response, "tool_calls", []) or [])

        executed: List[Dict] = []
        for call in tool_calls:
            tool_name = call.get("function", {}).get("name")
            if not tool_name:
                continue
            tool_fn = TOOL_REGISTRY.get(tool_name)
            if tool_fn is None:
                raise ValueError(f"Unknown tool '{tool_name}' requested by micro layouter")

            args_raw = call.get("function", {}).get("arguments", "")
            args = _parse_tool_args(args_raw)
            print(f"[micro] tool call {tool_name}({args})")
            result = tool_fn(state, **args)
            executed.append(
                {
                    "id": call.get("id"),
                    "tool": tool_name,
                    "arguments": args,
                    "result": result,
                }
            )

        state.last_tool_calls = executed

        if text:
            state.messages.append({"role": "assistant", "content": text})
        else:
            state.messages.append({"role": "assistant", "content": ""})
        state.last_micro_text = text
        state.phase = "micro"
        state.iteration += 1
        return state

    return node



