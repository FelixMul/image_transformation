from typing import Dict, List, Tuple
from PIL import Image
import os


def composite(background_img: Image.Image, object_images: Dict[int, Image.Image], placements: List[Dict]) -> Image.Image:
    """Composite objects onto the background according to placements.

    placements: list of {object_id, box: [x1,y1,x2,y2]}
    """
    canvas = background_img.copy()
    for p in placements:
        oid = int(p["object_id"]) if not isinstance(p["object_id"], int) else p["object_id"]
        if oid not in object_images:
            continue
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        obj = object_images[oid]
        resized = obj.resize((w, h), Image.LANCZOS)
        canvas.alpha_composite(resized, dest=(x1, y1))
    return canvas


def load_object_images(results_json_path: str) -> Dict[int, Image.Image]:
    import json
    with open(results_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    images: Dict[int, Image.Image] = {}
    base_dir = os.path.dirname(results_json_path)
    for it in items:
        oid = int(it["object_id"])
        path = os.path.join(base_dir, it["filename"])
        images[oid] = Image.open(path).convert("RGBA")
    return images


