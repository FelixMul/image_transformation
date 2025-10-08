"""LangGraph workflow definition for the agentic macro placement pipeline."""

from __future__ import annotations

from typing import Dict

from langgraph.graph import END, StateGraph

from agentic.nodes import (
    build_compositor_node,
    build_critic_node,
    build_macro_node,
    build_micro_node,
    build_validator_node,
)
from agentic.state import LayoutState, ObjectMeta


def build_workflow(
    model_macro,
    model_micro,
    model_critic,
    objects: Dict[int, ObjectMeta],
) -> StateGraph:
    graph = StateGraph(LayoutState)

    graph.add_node("macro", build_macro_node(model_macro))
    graph.add_node("micro", build_micro_node(model_micro))
    graph.add_node("critic", build_critic_node(model_critic))
    graph.add_node("validator", build_validator_node(list(objects)))
    graph.add_node("compositor", build_compositor_node())

    graph.set_entry_point("macro")
    graph.add_edge("macro", "validator")
    graph.add_edge("validator", "compositor")
    graph.add_edge("compositor", "critic")
    graph.add_conditional_edges(
        "critic",
        lambda state: "STOP" if state.should_stop or state.iteration >= state.max_iterations else "CONTINUE",
        {
            "STOP": END,
            "CONTINUE": "micro",
        },
    )
    graph.add_edge("micro", "validator")

    return graph


