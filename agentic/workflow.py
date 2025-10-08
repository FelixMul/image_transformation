"""Workflow orchestration helpers for the agentic pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from langgraph.graph import StateGraph

from agentic.graph import build_workflow
from agentic.state import LayoutState, RunContext
from agentic.utils import ensure_bundle, load_objects


def initialize_state(
    image_name: str,
    ratio: str,
    bundle_dir: Path,
    base_run_dir: Path,
    api_type: str,
    api_key: str | None,
    temperature: float,
    max_iterations: int,
    canvas_size: Tuple[int, int],
    original_image_path: Path,
) -> LayoutState:
    """Create an initial LayoutState from bundle artifacts."""

    background_path, results_json_path, objects_dir = ensure_bundle(bundle_dir)
    objects = load_objects(results_json_path, objects_dir)
    run_root = base_run_dir / image_name
    run_root.mkdir(parents=True, exist_ok=True)

    run_context = RunContext(
        image_name=image_name,
        ratio=ratio,
        canvas_size=canvas_size,
        bundle_dir=bundle_dir,
        background_path=background_path,
        objects_dir=objects_dir,
        results_json_path=results_json_path,
        original_image_path=original_image_path,
        run_root=run_root,
        max_iterations=max_iterations,
    )

    return LayoutState(
        run=run_context,
        objects=objects,
        api_type=api_type,
        api_key=api_key,
        temperature=temperature,
        messages=[],
        max_iterations=max_iterations,
    )


def compile_workflow(
    state: LayoutState,
    macro_model,
    micro_model,
    critic_model,
) -> StateGraph:
    """Build the LangGraph workflow with the provided models."""

    return build_workflow(macro_model, micro_model, critic_model, state.objects)


