"""Critic LangGraph node."""

from __future__ import annotations

from typing import Callable

from langchain_core.runnables import Runnable

from agentic.state import LayoutState
from agentic.utils import load_prompt


def _format_critic_context(state: LayoutState) -> str:
    lines = []
    lines.append(f"Canvas: {state.canvas_size[0]}x{state.canvas_size[1]} (ratio {state.ratio})")
    if state.placements:
        lines.append("Placements:")
        for placement in state.placements.values():
            lines.append(
                f"- {placement.name} (id={placement.object_id}) box=[{placement.x}, {placement.y}, {placement.x + placement.width}, {placement.y + placement.height}]"
            )
    else:
        lines.append("Placements: none")
    return "\n".join(lines)


def build_critic_node(model: Runnable) -> Callable[[LayoutState], LayoutState]:
    prompt_template = load_prompt("critic")

    def node(state: LayoutState) -> LayoutState:
        context_prompt = prompt_template.replace("{{CONTEXT}}", _format_critic_context(state))
        messages = state.messages + [
            {"role": "system", "content": context_prompt},
        ]
        response = model.invoke({"messages": messages})
        text = response.content if hasattr(response, "content") else str(response)
        state.last_critic_text = text
        state.critic_notes.append(text)
        state.messages.append({"role": "assistant", "content": text})
        state.phase = "critique"
        return state

    return node


