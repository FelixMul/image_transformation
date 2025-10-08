"""Workflow state definitions for the agentic macro placement pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

from langgraph.graph import add_messages


@dataclass
class ObjectMeta:
    """Metadata for an individual segmented object."""

    object_id: int
    name: str
    filename: str
    width: int
    height: int


@dataclass
class PlacementState:
    """Absolute placement for an object on the canvas."""

    object_id: int
    name: str
    x: int
    y: int
    width: int
    height: int

    def move_dx(self, delta: int) -> None:
        """Shift horizontally by ``delta`` pixels (positive→right, negative→left)."""

        self.x += delta

    def move_dy(self, delta: int) -> None:
        """Shift vertically by ``delta`` pixels (positive→down, negative→up)."""

        self.y += delta


@dataclass
class RunContext:
    image_name: str
    ratio: str
    canvas_size: Tuple[int, int]
    bundle_dir: Path
    background_path: Path
    objects_dir: Path
    results_json_path: Path
    original_image_path: Path
    run_root: Path
    max_iterations: int


@dataclass
class LayoutState:
    """Workflow state propagated through the LangGraph pipeline."""

    # Immutable input context -------------------------------------------------
    run: RunContext
    objects: Dict[int, ObjectMeta]
    api_type: str
    api_key: Optional[str]
    temperature: float

    # Conversation memory -----------------------------------------------------
    messages: Annotated[List, add_messages]

    # Planner output ---------------------------------------------------------
    flex_json: Optional[Dict] = None
    flex_text: Optional[str] = None

    # Deterministic placements (overrides DSL when present) -------------------
    placements: Dict[int, PlacementState] = field(default_factory=dict)

    # Iteration tracking ------------------------------------------------------
    iteration: int = 0
    phase: str = "macro"  # macro → micro → critique
    max_iterations: int = 0

    # Diagnostics -------------------------------------------------------------
    critic_notes: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    should_stop: bool = False
    current_composite_path: Optional[Path] = None
    last_macro_text: Optional[str] = None
    last_micro_text: Optional[str] = None
    last_critic_text: Optional[str] = None
    last_tool_calls: List[Dict] = field(default_factory=list)

    def ensure_placements(self) -> None:
        """Initialise the placements dictionary lazily."""

        if self.placements is None:
            self.placements = {}

    @property
    def canvas_size(self) -> Tuple[int, int]:
        return self.run.canvas_size

    @property
    def ratio(self) -> str:
        return self.run.ratio

    @property
    def background_path(self) -> Path:
        return self.run.background_path

    @property
    def objects_dir(self) -> Path:
        return self.run.objects_dir

    @property
    def results_json_path(self) -> Path:
        return self.run.results_json_path

    @property
    def original_image_path(self) -> Path:
        return self.run.original_image_path

    @property
    def base_artifacts_dir(self) -> Path:
        return self.run.run_root

    def get_iteration_dir(self) -> Path:
        return self.run.run_root / f"iteration_{self.iteration:02d}"

    @property
    def iteration_dir(self) -> Path:
        return self.get_iteration_dir()

    def register_placement(self, placement: PlacementState) -> None:
        """Add or replace a placement entry."""

        self.ensure_placements()
        self.placements[placement.object_id] = placement

    def get_unplaced_object_ids(self) -> List[int]:
        """Return object ids that are missing placements."""

        placed = set(self.placements or {})
        missing = [oid for oid in self.objects if oid not in placed]
        return missing


