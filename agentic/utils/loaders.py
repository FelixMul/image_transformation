"""Bundle loading helpers for the agentic workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image

from agentic.state import ObjectMeta


def load_objects(results_json_path: Path, objects_dir: Path) -> Dict[int, ObjectMeta]:
    """Load object metadata (including intrinsic size) from the bundle."""

    items = json.loads(results_json_path.read_text(encoding="utf-8"))
    objects: Dict[int, ObjectMeta] = {}
    for item in items:
        oid = int(item["object_id"])
        filename = Path(item["filename"]).name
        image_path = objects_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Object PNG missing: {image_path}")
        with Image.open(image_path) as im:
            width, height = im.size
        objects[oid] = ObjectMeta(
            object_id=oid,
            name=item.get("label", f"object_{oid}"),
            filename=filename,
            width=width,
            height=height,
        )
    return objects


def ensure_bundle(bundle_dir: Path) -> Tuple[Path, Path, Path]:
    """Return (background_path, results_json_path, objects_dir) after validation."""

    background_path = bundle_dir / "background.png"
    results_json_path = bundle_dir / "results.json"
    objects_dir = bundle_dir / "objects"
    missing = [
        str(path)
        for path in (background_path, results_json_path, objects_dir)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing expected bundle artifacts: " + ", ".join(missing)
        )
    return background_path, results_json_path, objects_dir


