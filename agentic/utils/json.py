"""Helpers for parsing JSON embedded in LLM outputs."""

from __future__ import annotations

import json
from typing import Any, Optional


def _find_json_object(raw: str) -> Optional[str]:
    depth = 0
    start = None
    for idx, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return raw[start : idx + 1]
    return None


def extract_json_object(raw: str) -> Any:
    """Extract the first JSON object embedded in ``raw`` text.

    Raises ``ValueError`` if no valid JSON object can be parsed.
    """
    # 1) Try fenced code block first: ```json ... ``` or ``` ... ```
    try:
        start_idx = raw.index("```")
        end_idx = raw.index("```", start_idx + 3)
        fenced = raw[start_idx + 3 : end_idx].strip()
        # drop optional language tag (e.g., "json\n")
        if "\n" in fenced:
            first_line, rest = fenced.split("\n", 1)
            if first_line.strip().lower() in {"json", "json5", "javascript", "js"}:
                fenced = rest.strip()
        if fenced.startswith("{") and fenced.endswith("}"):
            return json.loads(fenced)
    except ValueError:
        pass

    # 2) Fallback: find first balanced brace object
    snippet = _find_json_object(raw)
    if snippet is None:
        raise ValueError("No JSON object found in output")
    snippet = snippet.strip()
    if snippet == "{}":
        return {}
    return json.loads(snippet)


