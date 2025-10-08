"""Prompt loading helpers."""

from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load prompt text from ``prompts/<name>.txt``."""

    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


