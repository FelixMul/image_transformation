from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import shutil

from utils.timing import StepTimer
from layout_constraints import compute_canvas_size
from background_resizing import fill_solid
from compositor import composite, load_object_images
from api_client import get_api_client


SCRIPT_DIR = Path(__file__).parent.resolve()

def _vlm_request_critic(
    contact_sheet: Image.Image,
    original_input_path: str,
    composite_path: str,
    prev_flex_json: Dict,
    previous_critic_text: str,
    previous_refiner_json: Dict | None,
    row_bad: List[Tuple[str, str]],
    col_bad: List[Tuple[str, str]],
    ratio: str,
    api_type: str,
    best_practices: str,
    summary_text: str,
    role_lines: List[str],
    context_note: str,
    api_key: str | None = None,
    critic_custom_design_rules: str | None = None,
) -> Tuple[str, str]:
    """Return (critic_prompt, critic_raw_text). Temperature is fixed at 0.3.

    The critic sees: composed draft, original input, contact sheet; outputs free-form text
    with a very strict score /10, violations of HARD constraints, and actionable issues.
    """
    api_client = get_api_client(api_type, api_key=api_key)
    contact_b64 = _encode_pil_to_b64_png(contact_sheet)
    with open(original_input_path, "rb") as f_orig:
        original_b64 = base64.b64encode(f_orig.read()).decode("utf-8")
    with open(composite_path, "rb") as f_comp:
        composite_b64 = base64.b64encode(f_comp.read()).decode("utf-8")

    prev_json_str = json.dumps(prev_flex_json, indent=2)
    row_bad_str = ", ".join([f"({a}, {b})" for a, b in row_bad]) or "none"
    col_bad_str = ", ".join([f"({a}, {b})" for a, b in col_bad]) or "none"

    # Use custom design rules if provided, otherwise use the passed best_practices
    design_rules_to_use = critic_custom_design_rules.strip() if (critic_custom_design_rules and critic_custom_design_rules.strip()) else best_practices

    shared_context_block = _build_shared_prompt_context(
        design_rules_to_use, summary_text, role_lines, row_bad_str, col_bad_str
    )

    # Always construct the full prompt with all dynamic data
    critic_prompt = f"""### PERSONA

You are a professional Creative Director and a strict Design Critic.
TASK

Your goal is to evaluate the provided layout draft. Your primary focus is to determine how well the draft preserves the visual intent, balance, and core message of the original advertisement while adapting it to a new format. You must be specific, honest, and actionable. Do not generate a solution or JSON. If you are shown earlier conversation turns, use them to judge whether the latest refinement addressed your critique.

{shared_context_block}
EVALUATION & OUTPUT INSTRUCTIONS

Analyze the draft and provide your critique structured into the following sections. Be concise but specific.

1. Overall Score (out of 10):

    A single number from 0 to 10. (10=Perfect, 7=Acceptable, <5=Major flaws).

2. Preservation of Original Intent:

    How well does the draft maintain the original's visual hierarchy?

    Is the focus on the correct elements (e.g., the product, the main message)?

    Does the new layout feel like a professional adaptation or a random assortment of parts?

3. Hard Constraint Violations:

    Did the layout violate any of the non-negotiable rules from the CORE CONTEXT?

    Name the specific objects and rules that were broken (e.g., "Violates Row Nesting Conflict: ('Logo', 'Main Image')").

4. Composition & Design Issues:

    Comment on balance, alignment, negative space, and visual flow.

    Is the logo placement appropriate? Is the Call-to-Action (CTA) prominent and logically placed?

    Are there any awkward gaps, crowded areas, or margin violations?

5. Actionable Improvement Plan:

    Provide a clear, imperative list of MINOR TWEAKS for the next agent. The refiner can ONLY adjust spacing, padding, and fine-tune positions - it CANNOT change container structure, direction, or reorder objects.
    - Address objects one by one, referencing both their label and object_id (e.g., "CTA (object_id 3)").
    - Focus on micro adjustments that preserve the macro structure.

    GOOD examples:
    - "Logo (object_id 5): Increase padding_px by 20 to separate it from the headline."
    - "Hero text (object_id 2): Add gap_px of 15 to its parent column to open vertical breathing room."
    - "CTA (object_id 3): Pin horizontally to center so it aligns with the logo."
    - "Body copy (object_id 4): Add offset_px y:-10 to pull it closer to the hero image."
    
    BAD examples (DO NOT suggest these):
    - "Change the root container's direction to 'column'"
    - "Create a nested row container"
    - "Swap the positions of objects"
    - "Reorder the children array" """

    images_list = [contact_b64]
    # Do not include the original background-with-holes image; only add original input if desired
    if original_b64:
        images_list.append(original_b64)
    if composite_b64:
        images_list.append(composite_b64)

    messages: List[Dict[str, Union[str, List[str]]]] = [
        {"role": "system", "content": "You are a strict design critic. Output only plain text. Be concise and specific."}
    ]
    if previous_critic_text and previous_refiner_json:
        prev_json_str = json.dumps(previous_refiner_json, indent=2)
        messages.append(
            {
                "role": "assistant",
                "content": prev_json_str,
            }
        )
        messages.append({"role": "user", "content": previous_critic_text})
    messages.append({"role": "user", "content": critic_prompt, "images": images_list})

    raw_text = ""
    try:
        response = api_client.chat_completion(messages=messages, temperature=0.3)
        raw_text = response.get("message", {}).get("content", "")
    except Exception as e:
        print(f"[critic] VLM API error: {e}")
        raw_text = f"[critic_api_error] {str(e)}"

    return critic_prompt, raw_text

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_original_size(output_dir: Path) -> Tuple[int, int]:
    bg_path = output_dir / "background.png"
    with Image.open(bg_path).convert("RGBA") as im:
        return im.size


# --------------------------- Labeled contact sheet ---------------------------

def _build_labeled_contact_sheet(
    objects_dir: str,
    results_json_path: str,
    thumb_size: Tuple[int, int] = (256, 256),
    cols: int = 4,
    label_height: int = 72,
    font_size: int = 24,
) -> Image.Image:
    with open(results_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    # Sort by object_id for consistency
    items_sorted = sorted(items, key=lambda it: int(it["object_id"]))

    # Try to load a larger TrueType font; fallback to default bitmap font
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=font_size)
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

    thumbs: List[Image.Image] = []
    labels: List[str] = []
    for it in items_sorted:
        file_abs = str(Path(results_json_path).parent / it["filename"])
        im = Image.open(file_abs).convert("RGBA")
        th = im.copy()
        th.thumbnail(thumb_size, Image.LANCZOS)
        thumbs.append(th)
        labels.append(str(it.get("label", f"id_{it['object_id']}")))

    if not thumbs:
        return Image.new("RGBA", (thumb_size[0], thumb_size[1] + label_height), (255, 255, 255, 255))

    rows = (len(thumbs) + cols - 1) // cols
    cell_w = thumb_size[0]
    cell_h = thumb_size[1] + label_height
    w = cols * cell_w
    h = rows * cell_h

    sheet = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for idx, th in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x_cell = c * cell_w
        y_cell = r * cell_h
        # Center thumbnail in top area
        x = x_cell + (cell_w - th.width) // 2
        y = y_cell + (thumb_size[1] - th.height) // 2
        sheet.alpha_composite(th, dest=(x, y))
        # Draw label centered below
        label = labels[idx]
        # Measure text size robustly
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th_text = bbox[3] - bbox[1]
        except Exception:
            try:
                # Fallback to font size estimation
                if font is not None and hasattr(font, "getsize"):
                    tw, th_text = font.getsize(label)
                else:
                    # crude fallback
                    tw = int(len(label) * 7)
                    th_text = 12
            except Exception:
                tw = int(len(label) * 7)
                th_text = 12
        tx = x_cell + (cell_w - tw) // 2
        ty = y_cell + thumb_size[1] + max(0, (label_height - th_text) // 2)
        draw.text((tx, ty), label, fill=(0, 0, 0, 255), font=font)

    return sheet


# --------------------------- Flex DSL handling ---------------------------

FlexNode = Dict[str, Union[str, int, float, bool, List[Dict]]]

ALLOWED_JUSTIFY = {"start", "center", "end", "space_between", "space_around"}
ALLOWED_ALIGN = {"start", "center", "end"}
ALLOWED_DIRECTION = {"row", "column"}
STICK_TO_EDGES = {"left", "right", "top", "bottom"}


def _sanitize_padding(value: Union[int, Dict], oid: int) -> Dict[str, int]:
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"padding_px for object_id {oid} must be non-negative")
        return {"left": value, "right": value, "top": value, "bottom": value}
    if isinstance(value, dict):
        allowed_keys = {"left", "right", "top", "bottom"}
        extra = set(value.keys()) - allowed_keys
        if extra:
            raise ValueError(
                f"padding_px for object_id {oid} has unsupported keys: {sorted(extra)}"
            )
        result: Dict[str, int] = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        for key in result:
            raw_val = value.get(key, 0)
            if not isinstance(raw_val, int):
                raise ValueError(
                    f"padding_px[{key}] for object_id {oid} must be an integer"
                )
            if raw_val < 0:
                raise ValueError(
                    f"padding_px[{key}] for object_id {oid} must be non-negative"
                )
            result[key] = raw_val
        return result
    raise ValueError(
        f"padding_px for object_id {oid} must be int or dict with left/right/top/bottom"
    )


def _sanitize_pin(value: Dict, oid: int) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"pin for object_id {oid} must be an object with axis keys")
    allowed_keys = {"horizontal", "vertical"}
    extra = set(value.keys()) - allowed_keys
    if extra:
        raise ValueError(f"pin for object_id {oid} has unsupported keys: {sorted(extra)}")
    result: Dict[str, str] = {}
    for axis in allowed_keys:
        axis_val = value.get(axis)
        if axis_val is None:
            continue
        if axis_val not in ALLOWED_ALIGN:
            raise ValueError(
                f"pin.{axis} for object_id {oid} must be one of {sorted(ALLOWED_ALIGN)}"
            )
        result[axis] = axis_val
    return result


def _sanitize_offset(value: Dict, oid: int) -> Dict[str, int]:
    if value is None:
        return {"x": 0, "y": 0}
    if not isinstance(value, dict):
        raise ValueError(f"offset_px for object_id {oid} must be an object with x/y")
    allowed_keys = {"x", "y"}
    extra = set(value.keys()) - allowed_keys
    if extra:
        raise ValueError(
            f"offset_px for object_id {oid} has unsupported keys: {sorted(extra)}"
        )
    result: Dict[str, int] = {}
    for axis in allowed_keys:
        raw_val = value.get(axis, 0)
        if not isinstance(raw_val, int):
            raise ValueError(f"offset_px.{axis} for object_id {oid} must be an integer")
        result[axis] = raw_val
    return result


def _sanitize_stick_to(value: Dict, oid: int) -> Dict[str, Union[List[str], int]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"stick_to for object_id {oid} must be an object with edges and margin_px"
        )
    allowed_keys = {"edges", "margin_px"}
    extra = set(value.keys()) - allowed_keys
    if extra:
        raise ValueError(
            f"stick_to for object_id {oid} has unsupported keys: {sorted(extra)}"
        )
    edges = value.get("edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError(f"stick_to.edges for object_id {oid} must be a non-empty list")
    normalized_edges: List[str] = []
    for edge in edges:
        if not isinstance(edge, str):
            raise ValueError(
                f"stick_to.edges entries for object_id {oid} must be strings"
            )
        edge_lower = edge.lower()
        if edge_lower not in STICK_TO_EDGES:
            raise ValueError(
                f"stick_to.edge '{edge}' for object_id {oid} is not supported"
            )
        if edge_lower in normalized_edges:
            raise ValueError(
                f"stick_to.edges for object_id {oid} contains duplicate '{edge_lower}'"
            )
        normalized_edges.append(edge_lower)
    if "left" in normalized_edges and "right" in normalized_edges:
        raise ValueError(
            f"stick_to.edges for object_id {oid} cannot include both 'left' and 'right'"
        )
    if "top" in normalized_edges and "bottom" in normalized_edges:
        raise ValueError(
            f"stick_to.edges for object_id {oid} cannot include both 'top' and 'bottom'"
        )
    margin = value.get("margin_px", 0)
    if not isinstance(margin, int):
        raise ValueError(f"stick_to.margin_px for object_id {oid} must be an integer")
    if margin < 0:
        raise ValueError(f"stick_to.margin_px for object_id {oid} must be non-negative")
    return {"edges": normalized_edges, "margin_px": margin}


def _extract_json_maybe(content: str) -> str:
    s = content.strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        if len(parts) >= 3:
            s = parts[1]
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    return s


def _validate_flex_dsl(
    data: Dict,
    required_obj_ids: List[int],
    id_to_label: Dict[int, str],
    row_bad_pairs: set[frozenset[str]] | None = None,
    col_bad_pairs: set[frozenset[str]] | None = None,
    *,
    allow_object_tuning: bool = False,
    allow_container_spacing: bool = False,
    allow_container_align: bool = False,
    forbid_root_spacing: bool = False,
    drop_container_styling: bool = False,
    sanitize: bool = False,
) -> Tuple[Dict, List[int]]:
    # Basic structure
    if not isinstance(data, dict):
        raise ValueError("DSL root must be an object")
    if "root" not in data:
        raise ValueError("Missing root container")
    root = data["root"]
    if not isinstance(root, dict):
        raise ValueError("root must be an object")

    # Depth <= 2: root container -> children (either objects or one more level containers)
    seen_ids: List[int] = []
    violations: List[str] = []

    def gather_leaf_item_names(node: Dict) -> List[str]:
        """Return all item names under this node (itself if object, else all descendant objects)."""
        result: List[str] = []
        if isinstance(node, dict):
            if "object_id" in node:
                nm = node.get("name")
                if isinstance(nm, str) and nm:
                    result.append(nm)
            else:
                for sub in (node.get("children", []) or []):
                    if isinstance(sub, dict):
                        result.extend(gather_leaf_item_names(sub))
        return result

    def check_conflicts(node: Dict) -> None:
        direction = node.get("direction")
        children = node.get("children", []) or []
        if not children:
            return
        # Collect leaf name sets per direct child (object or container)
        child_name_sets: List[List[str]] = []
        for ch in children:
            if isinstance(ch, dict):
                child_name_sets.append(gather_leaf_item_names(ch))
            else:
                child_name_sets.append([])
        # Cross-child cartesian conflict check (prevents bypass via grouping)
        m = len(child_name_sets)
        for i in range(m):
            for j in range(i + 1, m):
                for ai in child_name_sets[i]:
                    for bj in child_name_sets[j]:
                        pair = frozenset({ai, bj})
                        if direction == "row" and row_bad_pairs and pair in row_bad_pairs:
                            violations.append(f"Row container indirectly nests non-nestable pair via grouping: {ai} + {bj}")
                        if direction == "column" and col_bad_pairs and pair in col_bad_pairs:
                            violations.append(f"Column container indirectly nests non-nestable pair via grouping: {ai} + {bj}")

    def validate_container(node: Dict, depth: int) -> None:
        if depth > 2:
            raise ValueError("Nesting depth > 2 not allowed")
        if node.get("type") != "flex":
            raise ValueError("Only type=flex containers supported")
        if node.get("direction") not in ALLOWED_DIRECTION:
            raise ValueError("direction must be 'row' or 'column'")
        justify_present = "justify" in node
        align_present = "align" in node
        justify = node.get("justify", "center")
        align = node.get("align", "center")
        if justify not in ALLOWED_JUSTIFY:
            raise ValueError("invalid justify")
        if align not in ALLOWED_ALIGN:
            raise ValueError("invalid align")
        gap_present = "gap_px" in node
        pad_present = "padding_px" in node
        gap = node.get("gap_px", 0)
        pad = node.get("padding_px", 0)
        if drop_container_styling and sanitize:
            node.pop("gap_px", None)
            node.pop("padding_px", None)
            node.pop("justify", None)
            node.pop("align", None)
            gap_present = False
            pad_present = False
            gap = 0
            pad = 0
            justify_present = False
            align_present = False
        if not allow_container_align:
            if (justify_present and justify not in (None, "center")) or (align_present and align not in (None, "center")):
                if sanitize:
                    node.pop("justify", None)
                    node.pop("align", None)
                    justify_present = False
                    align_present = False
                else:
                    raise ValueError("container align/justify not allowed in this mode")
            elif sanitize:
                if justify_present and justify in (None, "center"):
                    node.pop("justify", None)
                    justify_present = False
                if align_present and align in (None, "center"):
                    node.pop("align", None)
                    align_present = False
        if forbid_root_spacing and depth == 1:
            if (gap_present and gap != 0) or (pad_present and pad != 0):
                raise ValueError("root container cannot set gap_px or padding_px")
            if sanitize:
                if gap_present:
                    node.pop("gap_px", None)
                if pad_present:
                    node.pop("padding_px", None)
            if justify_present and justify != "center":
                raise ValueError("root container cannot change justify")
            if align_present and align != "center":
                raise ValueError("root container cannot change align")
            if sanitize:
                if justify_present:
                    node.pop("justify", None)
                    justify_present = False
                if align_present:
                    node.pop("align", None)
                    align_present = False
        if not allow_container_spacing and not drop_container_styling:
            if gap_present and gap not in (0, None):
                raise ValueError("gap_px not allowed in this mode")
            if pad_present and pad not in (0, None):
                raise ValueError("padding_px not allowed in this mode")
            if sanitize:
                if gap_present:
                    node.pop("gap_px", None)
                if pad_present:
                    node.pop("padding_px", None)
        if not isinstance(gap, int) or gap < 0:
            raise ValueError("gap_px must be non-negative int")
        if not isinstance(pad, int) or pad < 0:
            raise ValueError("padding_px must be non-negative int")
        children = node.get("children", [])
        if not isinstance(children, list) or not children:
            raise ValueError("flex container must have children")
        for ch in children:
            if not isinstance(ch, dict):
                raise ValueError("child must be object")
            if "object_id" in ch:
                oid = int(ch["object_id"])
                seen_ids.append(oid)
                # Require name to match known label to guarantee clarity
                name = ch.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"missing or invalid 'name' for object_id {oid}")
                known = id_to_label.get(oid, "").strip()
                if known and name.strip() != known:
                    raise ValueError(f"name mismatch for object_id {oid}: got '{name}', expected '{known}'")
                if not allow_object_tuning:
                    extras = set(ch.keys()) - {"object_id", "name"}
                    if extras:
                        if sanitize:
                            for key in extras:
                                ch.pop(key, None)
                        else:
                            raise ValueError(
                                f"object_id {oid} includes unsupported fields {sorted(extras)} in baseline mode"
                            )
                else:
                    padding_raw = ch.get("padding_px")
                    _ = _sanitize_padding(padding_raw, oid) if padding_raw is not None else None
                    pin_raw = ch.get("pin")
                    _sanitize_pin(pin_raw, oid) if pin_raw is not None else None
                    offset_raw = ch.get("offset_px")
                    _sanitize_offset(offset_raw, oid) if offset_raw is not None else None
                    stick_raw = ch.get("stick_to")
                    _sanitize_stick_to(stick_raw, oid) if stick_raw is not None else None
            else:
                # nested container
                validate_container(ch, depth + 1)
        # After validating children, check pairwise non-nesting constraints at this container level
        check_conflicts(node)

        validate_container(root, 1)

    # Coverage and duplicates
    seen_sorted = sorted(seen_ids)
    req_sorted = sorted(required_obj_ids)
    if seen_sorted != req_sorted:
        raise ValueError(f"object_id coverage mismatch. seen={seen_sorted}, required={req_sorted}")

    if violations:
        raise ValueError("; ".join(violations))

    return data, seen_ids


def _compute_nesting_conflicts(results_json_path: str, canvas_size: Tuple[int, int], margin_pct: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Return (row_conflicts, col_conflicts) as pairs of label names that cannot fit together
    in the same row (sum widths > inner width) or same column (sum heights > inner height).
    """
    with open(results_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    base_dir = Path(results_json_path).parent
    w, h = canvas_size
    inner_w = max(1, int(w - 2 * (margin_pct * w)))
    inner_h = max(1, int(h - 2 * (margin_pct * h)))
    # Require at least a minimal gap between siblings when checking feasibility
    min_gap = max(8, int(min(w, h) * 0.01))

    labels: List[str] = []
    sizes: List[Tuple[int, int]] = []
    for it in items:
        label = str(it.get("label", "")).strip() or f"id_{it.get('object_id')}"
        path = base_dir / it["filename"]
        iw, ih = 0, 0
        # Prefer actual cutout image size; fallback to bounding_box from results.json
        try:
            with Image.open(path).convert("RGBA") as im:
                iw, ih = im.size
        except Exception:
            try:
                x1, y1, x2, y2 = it.get("bounding_box", [0, 0, 0, 0])
                iw = max(0, int(x2 - x1))
                ih = max(0, int(y2 - y1))
            except Exception:
                iw, ih = 0, 0
        labels.append(label)
        sizes.append((iw, ih))

    row_bad_set: set[Tuple[str, str]] = set()
    col_bad_set: set[Tuple[str, str]] = set()
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            wi, hi = sizes[i]
            wj, hj = sizes[j]
            # If any single item already exceeds inner dimension, it cannot share the container in that axis
            if wi >= inner_w or wj >= inner_w or (wi + wj + min_gap) > inner_w:
                row_bad_set.add(tuple(sorted((labels[i], labels[j]))))
            if hi >= inner_h or hj >= inner_h or (hi + hj + min_gap) > inner_h:
                col_bad_set.add(tuple(sorted((labels[i], labels[j]))))
    row_bad = sorted(list(row_bad_set))
    col_bad = sorted(list(col_bad_set))
    return row_bad, col_bad


def _measure_flex_node(node: Dict, images: Dict[int, Image.Image]) -> Tuple[int, int]:
    """
    Compute the intrinsic size of a node (object or container) without scaling.
    - For objects: return image size or (0,0) if missing
    - For containers: recursively measure children according to direction and include gap/padding
    """
    if isinstance(node, dict) and "object_id" in node:
        try:
            oid = int(node["object_id"])
        except Exception:
            return 0, 0
        img = images.get(oid)
        padding = node.get("padding_px")
        pad = (
            _sanitize_padding(padding, oid)
            if padding is not None
            else {"left": 0, "right": 0, "top": 0, "bottom": 0}
        )
        width, height = img.size if img is not None else (0, 0)
        width = max(0, width + pad["left"] + pad["right"])
        height = max(0, height + pad["top"] + pad["bottom"])
        return (width, height)

    # Container
    direction = node.get("direction", "row")
    gap_px = int(node.get("gap_px", 0))
    padding_px = int(node.get("padding_px", 0))
    children: List[Dict] = node.get("children", []) or []
    if not children:
        # padding-only box
        return max(0, padding_px * 2), max(0, padding_px * 2)

    # Measure children recursively
    measured: List[Tuple[int, int]] = []
    for ch in children:
        if isinstance(ch, dict):
            measured.append(_measure_flex_node(ch, images))
        else:
            measured.append((0, 0))

    if direction == "row":
        total_w = sum(w for w, _ in measured) + gap_px * (len(measured) - 1 if len(measured) > 1 else 0)
        total_h = max((h for _, h in measured), default=0)
    else:
        total_w = max((w for w, _ in measured), default=0)
        total_h = sum(h for _, h in measured) + gap_px * (len(measured) - 1 if len(measured) > 1 else 0)

    total_w = max(0, total_w + 2 * max(0, padding_px))
    total_h = max(0, total_h + 2 * max(0, padding_px))
    return int(total_w), int(total_h)


def _place_flex_container(node: Dict, origin: Tuple[int, int], size: Tuple[int, int], images: Dict[int, Image.Image], placements: List[Dict], parent_cell: str) -> None:
    x0, y0 = origin
    cw, ch = size

    direction = node.get("direction", "row")
    justify = node.get("justify", "center")
    align = node.get("align", "center")
    gap_px = int(node.get("gap_px", 0))
    padding_px = int(node.get("padding_px", 0))

    inner_x = x0 + padding_px
    inner_y = y0 + padding_px
    inner_w = max(0, cw - 2 * padding_px)
    inner_h = max(0, ch - 2 * padding_px)

    children: List[Dict] = node.get("children", [])

    def place_object_node(obj_node: Dict, slot_origin: Tuple[int, int], slot_size: Tuple[int, int]) -> None:
        oid = int(obj_node.get("object_id", -1))
        img = images.get(oid)
        img_w, img_h = img.size if img is not None else (0, 0)

        padding_raw = obj_node.get("padding_px")
        padding = (
            _sanitize_padding(padding_raw, oid)
            if padding_raw is not None
            else {"left": 0, "right": 0, "top": 0, "bottom": 0}
        )
        pin_raw = obj_node.get("pin")
        pin = _sanitize_pin(pin_raw, oid) if pin_raw is not None else {}
        offset_raw = obj_node.get("offset_px")
        offset = (
            _sanitize_offset(offset_raw, oid)
            if offset_raw is not None
            else {"x": 0, "y": 0}
        )
        stick_raw = obj_node.get("stick_to")
        stick = _sanitize_stick_to(stick_raw, oid) if stick_raw is not None else {}

        slot_x1, slot_y1 = slot_origin
        slot_w, slot_h = slot_size
        slot_x2 = slot_x1 + slot_w
        slot_y2 = slot_y1 + slot_h

        inner_x1 = slot_x1 + padding["left"]
        inner_y1 = slot_y1 + padding["top"]
        inner_x2 = max(inner_x1, slot_x2 - padding["right"])
        inner_y2 = max(inner_y1, slot_y2 - padding["bottom"])
        inner_w_local = max(0, inner_x2 - inner_x1)
        inner_h_local = max(0, inner_y2 - inner_y1)

        scale = 1.0
        if img is not None and img_w > 0 and img_h > 0:
            scale_candidates: List[float] = [1.0]
            if inner_w_local > 0:
                scale_candidates.append(inner_w_local / img_w)
            if inner_h_local > 0:
                scale_candidates.append(inner_h_local / img_h)
            scale = max(0.0, min(scale_candidates)) if scale_candidates else 1.0
            target_w = int(round(img_w * scale))
            target_h = int(round(img_h * scale))
        else:
            target_w = inner_w_local
            target_h = inner_h_local

        target_w = max(0, min(target_w, inner_w_local))
        target_h = max(0, min(target_h, inner_h_local))

        horizontal_mode = pin.get("horizontal")
        if horizontal_mode is None:
            horizontal_mode = align if direction == "column" else "start"
        vertical_mode = pin.get("vertical")
        if vertical_mode is None:
            vertical_mode = align if direction == "row" else "start"

        remaining_w = max(0, inner_w_local - target_w)
        remaining_h = max(0, inner_h_local - target_h)

        if horizontal_mode == "center":
            base_x = inner_x1 + remaining_w / 2
        elif horizontal_mode == "end":
            base_x = inner_x2 - target_w
        else:
            base_x = inner_x1

        if vertical_mode == "center":
            base_y = inner_y1 + remaining_h / 2
        elif vertical_mode == "end":
            base_y = inner_y2 - target_h
        else:
            base_y = inner_y1

        if stick:
            margin = stick.get("margin_px", 0)
            edges = stick.get("edges", [])
            if "left" in edges:
                base_x = inner_x1 + margin
            elif "right" in edges:
                base_x = inner_x2 - margin - target_w
            if "top" in edges:
                base_y = inner_y1 + margin
            elif "bottom" in edges:
                base_y = inner_y2 - margin - target_h

        base_x += offset.get("x", 0)
        base_y += offset.get("y", 0)

        min_x = inner_x1
        max_x = inner_x2 - target_w
        if max_x < min_x:
            max_x = min_x
        base_x = min(max(base_x, min_x), max_x)

        min_y = inner_y1
        max_y = inner_y2 - target_h
        if max_y < min_y:
            max_y = min_y
        base_y = min(max(base_y, min_y), max_y)

        final_x1 = int(round(base_x))
        final_y1 = int(round(base_y))
        final_x2 = final_x1 + int(target_w)
        final_y2 = final_y1 + int(target_h)

        if final_x2 > inner_x2:
            delta = final_x2 - inner_x2
            final_x1 -= delta
            final_x2 -= delta
        if final_x1 < inner_x1:
            delta = inner_x1 - final_x1
            final_x1 += delta
            final_x2 += delta
        if final_y2 > inner_y2:
            delta = final_y2 - inner_y2
            final_y1 -= delta
            final_y2 -= delta
        if final_y1 < inner_y1:
            delta = inner_y1 - final_y1
            final_y1 += delta
            final_y2 += delta

        placement_entry: Dict[str, Union[int, float, Dict, List]] = {
            "object_id": oid,
            "cell": parent_cell,
            "box": [int(final_x1), int(final_y1), int(final_x2), int(final_y2)],
            "scale": float(scale),
        }
        if padding_raw is not None:
            placement_entry["padding_px"] = padding
        if pin_raw is not None and pin:
            placement_entry["pin"] = pin
        if offset_raw is not None:
            placement_entry["offset_px"] = offset
        elif offset.get("x", 0) or offset.get("y", 0):
            placement_entry["offset_px"] = offset
        if stick_raw is not None and stick:
            placement_entry["stick_to"] = stick

        placements.append(placement_entry)

    # Measure intrinsic sizes for children (objects and nested containers)
    child_sizes: List[Tuple[int, int]] = []
    for ch in children:
        if "object_id" in ch:
            try:
                oid = int(ch["object_id"])
            except Exception:
                child_sizes.append((0, 0))
                continue
            img = images.get(oid)
            child_sizes.append(_measure_flex_node(ch, images))
        else:
            child_sizes.append(_measure_flex_node(ch, images))

    if direction == "row":
        total_w = sum(w for w, _ in child_sizes) + gap_px * (len(children) - 1 if len(children) > 0 else 0)
        if justify == "start":
            cur_x = inner_x
            gap_between = gap_px
        elif justify == "center":
            cur_x = inner_x + max(0, (inner_w - total_w) // 2)
            gap_between = gap_px
        elif justify == "end":
            cur_x = inner_x + max(0, (inner_w - total_w))
            gap_between = gap_px
        elif justify == "space_between" and len(children) > 1:
            cur_x = inner_x
            gap_between = (inner_w - sum(w for w, _ in child_sizes)) // (len(children) - 1)
            if gap_between < 0:
                gap_between = 0
        elif justify == "space_around" and len(children) > 0:
            gap_between = (inner_w - sum(w for w, _ in child_sizes)) // (len(children))
            if gap_between < 0:
                gap_between = 0
            cur_x = inner_x + gap_between // 2
        else:
            cur_x = inner_x
            gap_between = gap_px

        for idx, ch in enumerate(children):
            w, h = child_sizes[idx]
            if align == "start":
                py = inner_y
            elif align == "center":
                py = inner_y + (inner_h - h) // 2
            elif align == "end":
                py = inner_y + (inner_h - h)
            else:
                py = inner_y + (inner_h - h) // 2

            px = cur_x
            if "object_id" in ch:
                place_object_node(ch, (px, py), (w, h))
            else:
                _place_flex_container(
                    ch, (px, py), (w, h), images, placements, parent_cell
                )
            cur_x = cur_x + w + gap_between

    else:
        total_h = sum(h for _, h in child_sizes) + gap_px * (len(children) - 1 if len(children) > 0 else 0)
        if justify == "start":
            cur_y = inner_y
            gap_between = gap_px
        elif justify == "center":
            cur_y = inner_y + max(0, (inner_h - total_h) // 2)
            gap_between = gap_px
        elif justify == "end":
            cur_y = inner_y + max(0, (inner_h - total_h))
            gap_between = gap_px
        elif justify == "space_between" and len(children) > 1:
            cur_y = inner_y
            gap_between = (inner_h - sum(h for _, h in child_sizes)) // (len(children) - 1)
            if gap_between < 0:
                gap_between = 0
        elif justify == "space_around" and len(children) > 0:
            gap_between = (inner_h - sum(h for _, h in child_sizes)) // (len(children))
            if gap_between < 0:
                gap_between = 0
            cur_y = inner_y + gap_between // 2
        else:
            cur_y = inner_y
            gap_between = gap_px

        for idx, ch in enumerate(children):
            w, h = child_sizes[idx]
            if align == "start":
                px = inner_x
            elif align == "center":
                px = inner_x + (inner_w - w) // 2
            elif align == "end":
                px = inner_x + (inner_w - w)
            else:
                px = inner_x + (inner_w - w) // 2

            py = cur_y
            if "object_id" in ch:
                place_object_node(ch, (px, py), (w, h))
            else:
                _place_flex_container(
                    ch, (px, py), (w, h), images, placements, parent_cell
                )
            cur_y = cur_y + h + gap_between


def _clamp_boxes_to_canvas(placements: List[Dict], canvas_size: Tuple[int, int]) -> None:
    tw, th = canvas_size
    for p in placements:
        x1, y1, x2, y2 = p["box"]
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, min(x1, tw - w))
        y1 = max(0, min(y1, th - h))
        x2 = x1 + w
        y2 = y1 + h
        p["box"] = [int(x1), int(y1), int(x2), int(y2)]


def _save_overlay_debug(placements: List[Dict], canvas_size: Tuple[int, int], path: Path) -> None:
    w, h = canvas_size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (255, 99, 71, 180),
        (135, 206, 235, 180),
        (60, 179, 113, 180),
        (238, 130, 238, 180),
        (255, 215, 0, 180),
        (30, 144, 255, 180),
    ]
    for idx, p in enumerate(placements):
        x1, y1, x2, y2 = p["box"]
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    overlay.save(path)


def _best_practices_text(aspect_family: str) -> str:
    return (
        "Best practices (macro placement):\n"
        "- photo: foreground image. Ratio type: Vertical: if only 1 image, avoid top. Square: if only 1 image, avoid top-left. Horizontal/U-wide: center vertically.\n"
        "- design element: drawn/geometric. Typically center vertically.\n"
        "- text: copy with font attributes. Ratio type: Vertical: center horizontally; Horizontal/U-wide: typically center vertically and often stacked below other objects.\n"
        "- composite image: composed of image/design/text.\n"
        "- logo: special composite (brand). Ratio type: Vertical: top or bottom or center if prominent. Square: near a corner or centered top/bottom or mid with prominence. Horizontal/U-wide: far left or right, sometimes central; typically centered vertically.\n"
        "- CTA: special composite (button-like) with design+verb text. Vertical/Square: lower half but not close to bottom; Square can be centered or slightly right. Horizontal: right half, not close to right margin; U-wide: centered vertically, toward right.\n"
        f"- Aspect family: {aspect_family}. Apply the corresponding guidance above."
    )


def _ratio_family(ratio: str) -> str:
    try:
        w, h = ratio.split(":")
        w = float(w); h = float(h)
        r = w / max(1e-6, h)
        if (h / max(1e-6, w)) > 2.2:
            return "vertical"
        if r > 3.2:
            return "u-wide"
        if r > 2.2:
            return "horizontal"
        return "square"
    except Exception:
        return "unknown"


def _encode_pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_shared_prompt_context(
    best_practices_text: str,
    summary_text: str,
    role_lines: List[str],
    row_bad_str: str,
    col_bad_str: str,
) -> str:
    shared_context = f"""### CORE CONTEXT

This section contains the data, rules, and schema you must adhere to.
1. Hard Constraints (Non-Negotiable)

    The layout's nesting depth MUST NOT exceed 2.

    Every object_id provided in the Data Reference MUST be used exactly once.

    The following pairs of objects are too large to fit together in the same container along the specified axis. This rule CANNOT be bypassed by grouping.

        Row Nesting Conflicts: {row_bad_str}

        Column Nesting Conflicts: {col_bad_str}

2. Guiding Principles (Aesthetic & Structural Advice)

{best_practices_text}

    Vertical Ratios: Prefer a single column. Preserve top-to-bottom reading order.

    Ultra-Wide/Horizontal Ratios: Prefer a single row. Preserve left-to-right flow.

    Square Ratios: Aim for a balanced, grid-like composition.

    Nesting: Avoid nesting unless it is clearly present in the original design.

3. Data Reference

    Objects Summary (id, name, role, original bbox_norm):
    {summary_text}

    Roles Map:
    {", ".join(role_lines)}

4. DSL Schema

Your JSON output must conform strictly to this structure.
{{
"root": {{
"type": "flex",
"direction": "row|column",
"justify": "start|center|end|space_between|space_around",
"align": "start|center|end",
"gap_px": int (optional),
"padding_px": int (optional),
"children": [ <item_or_container>, ... ]
}}
}}

    An <item> is: {{ "object_id": <int>, "name": "<string EXACT label>" }}

    A <container> is another flex object, subject to the depth limit."""
    return shared_context


def _vlm_request_flex(
    contact_sheet: Image.Image,
    background_path: str,
    original_input_path: str,
    results_json_path: str,
    ratio: str,
    canvas_size: Tuple[int, int],
    api_type: str,
    temperature: float,
    margin_pct: float,
    planner_addendum: str = "",
    api_key: str | None = None,
    planner_custom_design_rules: str | None = None,
) -> Tuple[Dict, str, str, List[str], str, str]:
    with open(results_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    obj_ids = [int(it["object_id"]) for it in items]
    id_to_label = {int(it["object_id"]): str(it.get("label", "")).strip() for it in items}

    iw, ih = Image.open(background_path).size
    summary_lines = []
    role_lines = []
    for it in items:
        oid = int(it["object_id"])
        label = it.get("label", "")
        low = label.lower()
        if "logo" in low:
            role = "logo"
        elif "cta" in low:
            role = "cta"
        elif "text" in low or "copy" in low:
            role = "text"
        elif "design" in low or "shape" in low or "element" in low:
            role = "design"
        else:
            role = "image"
        x1, y1, x2, y2 = it.get("bounding_box", [0, 0, 0, 0])
        nx1 = round(x1 / max(1, iw), 4)
        ny1 = round(y1 / max(1, ih), 4)
        nx2 = round(x2 / max(1, iw), 4)
        ny2 = round(y2 / max(1, ih), 4)
        summary_lines.append(f"id={oid}, name='{label}', role={role}, bbox_norm=[{nx1},{ny1},{nx2},{ny2}]")
        role_lines.append(f"{oid}:{role}")
    summary_text = "\n".join(summary_lines)

    # Compute non-nestable pairs
    row_bad, col_bad = _compute_nesting_conflicts(results_json_path, canvas_size, margin_pct)
    row_bad_str = ", ".join([f"({a}, {b})" for a, b in row_bad]) or "none"
    col_bad_str = ", ".join([f"({a}, {b})" for a, b in col_bad]) or "none"

    w, h = canvas_size
    aspect_family = _ratio_family(ratio)
    
    # Use custom design rules if provided, otherwise use defaults
    if planner_custom_design_rules and planner_custom_design_rules.strip():
        best_practices = planner_custom_design_rules.strip()
    else:
        best_practices = _best_practices_text(aspect_family)

    shared_context_block = _build_shared_prompt_context(
        best_practices, summary_text, role_lines, row_bad_str, col_bad_str
    )

    # Always construct the full prompt with all dynamic data
    base_prompt = f"""### PERSONA

You are a pragmatic Layout Planner.
TASK

Your goal is to generate a valid first-draft layout in the Flex DSL JSON format. Analyze the original image to understand its visual intent and use the object data as your guide. Your layout must fit within the provided target canvas.

IMPORTANT: You must work strictly at the MACRO level:
- You may only decide container directions (row/column) and which objects belong in each container.
- Do NOT set or mention any spacing or alignment properties (no gap_px, padding_px, justify, align, pin, offset, stick_to).
- Do NOT invent new containers beyond depth 2, and do not duplicate or drop objects.

{shared_context_block}
OUTPUT INSTRUCTIONS

    Your output must be ONLY the valid JSON object.

    Do not include any explanations, comments, or markdown code fences.

ADDITIONAL GUIDANCE (optional):
{planner_addendum}
"""

    api_client = get_api_client(api_type, api_key=api_key)
    contact_b64 = _encode_pil_to_b64_png(contact_sheet)
    with open(background_path, "rb") as f:
        background_b64 = base64.b64encode(f.read()).decode("utf-8")
    # Include the very original input image for more context
    try:
        with open(original_input_path, "rb") as f_orig:
            original_b64 = base64.b64encode(f_orig.read()).decode("utf-8")
    except Exception:
        original_b64 = ""
    messages = [
        {"role": "system", "content": "You are a JSON generator. Follow HARD CONSTRAINTS strictly. Output ONLY valid JSON matching the schema. No markdown, no explanations."},
        {"role": "user", "content": base_prompt, "images": [contact_b64, background_b64] + ([original_b64] if original_b64 else [])},
    ]
    raw_text = ""
    try:
        response = api_client.chat_completion(messages=messages, temperature=temperature)
        raw_text = response.get("message", {}).get("content", "")
        try:
            data = json.loads(_extract_json_maybe(raw_text))
        except Exception:
            print("[flex] Invalid JSON from VLM; see raw text for details.")
            data = {"error": "invalid_json", "raw": raw_text[:1000]}
    except Exception as e:
        print(f"[flex] VLM API error: {e}")
        data = {"error": "api_error", "detail": str(e)}

    return data, base_prompt, summary_text, role_lines, raw_text, best_practices


def _vlm_request_refine(
    contact_sheet: Image.Image,
    background_path: str,
    original_input_path: str,
    composite_prev_path: str,
    prev_flex_json: Dict,
    ratio: str,
    canvas_size: Tuple[int, int],
    api_type: str,
    temperature: float,
    allowed_ids: List[int],
    id_to_label: Dict[int, str],
    row_bad: List[Tuple[str, str]],
    col_bad: List[Tuple[str, str]],
    critic_text: str,
    summary_text: str,
    role_lines: List[str],
    extra_instructions: str = "",
    refiner_addendum: str = "",
    api_key: str | None = None,
    refiner_custom_design_rules: str | None = None,
) -> Tuple[Dict, str, str]:
    """
    Request a refined Flex DSL JSON from the VLM, given the previous composite render and previous Flex JSON.
    Returns (new_flex_json, prompt_text, raw_text).
    """
    w, h = canvas_size
    iw, ih = Image.open(background_path).size
    aspect_family = _ratio_family(ratio)
    
    # Use custom design rules if provided, otherwise use defaults
    if refiner_custom_design_rules and refiner_custom_design_rules.strip():
        best_practices = refiner_custom_design_rules.strip()
    else:
        best_practices = _best_practices_text(aspect_family)

    row_bad_str = ", ".join([f"({a}, {b})" for a, b in row_bad]) or "none"
    col_bad_str = ", ".join([f"({a}, {b})" for a, b in col_bad]) or "none"
    shared_context_block = _build_shared_prompt_context(
        best_practices, summary_text, role_lines, row_bad_str, col_bad_str
    )

    prev_json_str = json.dumps(prev_flex_json, indent=2)

    # Prepare images
    api_client = get_api_client(api_type, api_key=api_key)
    contact_b64 = _encode_pil_to_b64_png(contact_sheet)
    with open(background_path, "rb") as f_bg:
        background_b64 = base64.b64encode(f_bg.read()).decode("utf-8")
    # Include the very original input image for more context
    try:
        with open(original_input_path, "rb") as f_orig:
            original_b64 = base64.b64encode(f_orig.read()).decode("utf-8")
    except Exception:
        original_b64 = ""
    with open(composite_prev_path, "rb") as f_prev:
        composite_b64 = base64.b64encode(f_prev.read()).decode("utf-8")

    # Build chat-based conversation format
    # Turn 1: Initial task description with context
    initial_task_sections = [
        "### ROLE\n\nYou are the Layout Improver.",
        "### OBJECTIVE\nYour job is to make MINOR adjustments to Flex JSON layouts. You fine-tune object placement, NOT redesign layouts.",
        shared_context_block,
        "### STRICT RULES\n\n1. DO NOT CHANGE CONTAINER STRUCTURE:\n   - You MUST NOT change 'direction' (row/column) in ANY container, including root\n   - You MUST NOT add or remove containers\n   - You MUST NOT add or remove objects\n   - Do not reorder the children arrays\n\n2. ROOT CONTAINER IS FROZEN:\n   - No gap_px, padding_px, justify, or align changes at the root\n\n3. ALLOWED TWEAKS FOR NON-ROOT ITEMS:\n   - gap_px (only on non-root containers)\n   - padding_px (only on non-root containers)\n   - justify and align (only on non-root containers; preserve macro intent)\n   - Object-level padding_px, pin, offset_px, stick_to\n\n4. If feedback suggests structural changes (like \"change direction to column\"), IGNORE those. You only make micro adjustments.",
    ]
    if refiner_addendum:
        initial_task_sections.append("### ADDITIONAL GUIDANCE\n" + refiner_addendum)
    
    initial_task = "\n\n".join(initial_task_sections)
    initial_task += "\n\n### OUTPUT FORMAT\nReturn ONLY the JSON object—no prose, no markdown."

    # Turn 2: Show what the refiner previously generated (its own output)
    assistant_previous_output = prev_json_str

    # Turn 3: Critic's feedback on that output
    critic_feedback_sections = [
        "Here is the feedback from the Creative Director on your previous layout:",
        "",
        "### CRITIC'S EVALUATION",
        critic_text,
        "",
        "### YOUR TASK NOW",
        "Based on this critique, improve the layout JSON. Remember:",
        "- Only make MINOR tweaks (padding, gaps, pins, offsets)",
        "- Do NOT change container structure, direction, or object order",
        "- Address every specific point the critic raised",
        "- Return ONLY the improved JSON—no explanations",
    ]
    
    if extra_instructions:
        critic_feedback_sections.append("\n### VALIDATION ERRORS (fix these strictly)")
        critic_feedback_sections.append(extra_instructions)
    
    critic_feedback = "\n".join(critic_feedback_sections)

    # Construct multi-turn messages
    messages = [
        {"role": "system", "content": "You are a Layout Improver. You iteratively refine Flex JSON layouts based on feedback. Follow constraints strictly. Output ONLY valid JSON."},
        {"role": "user", "content": initial_task, "images": [contact_b64, background_b64] + ([original_b64] if original_b64 else []) + [composite_b64]},
        {"role": "assistant", "content": assistant_previous_output},
        {"role": "user", "content": critic_feedback},
    ]

    # For logging purposes, reconstruct the full prompt text
    prompt_text = f"""=== CONVERSATION-BASED REFINEMENT ===

[USER - Initial Task]
{initial_task}

[ASSISTANT - Your Previous Output]
{assistant_previous_output}

[USER - Critic Feedback & Improvement Request]
{critic_feedback}
"""
    raw_text = ""
    try:
        response = api_client.chat_completion(messages=messages, temperature=temperature)
        raw_text = response.get("message", {}).get("content", "")
        try:
            data = json.loads(_extract_json_maybe(raw_text))
        except Exception:
            print("[refine] Invalid JSON from VLM; see raw text for details.")
            data = {"error": "invalid_json", "raw": raw_text[:1000]}
    except Exception as e:
        print(f"[refine] VLM API error: {e}")
        data = {"error": "api_error", "detail": str(e)}

    return data, prompt_text, raw_text

def _compose_candidates_grid(image_paths: List[Path], out_path: Path) -> None:
    imgs = [Image.open(p).convert("RGBA") for p in image_paths if p.exists()]
    if not imgs:
        return
    # normalize to same size (use first as reference)
    ref_w, ref_h = imgs[0].size
    norm = [im.resize((ref_w, ref_h), Image.LANCZOS) for im in imgs]
    grid_w = ref_w * 2
    grid_h = ref_h * 2
    grid = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 255))
    positions = [(0, 0), (ref_w, 0), (0, ref_h), (ref_w, ref_h)]
    for im, (x, y) in zip(norm, positions):
        grid.alpha_composite(im, dest=(x, y))
    grid.save(out_path)


# --------------------------- Main flow ---------------------------

def run_macro_only(
    output_dir: Path,
    ratio: str,
    align: str,
    margin: float,
    api_type: str = "auto",
    samples: int = 1,
    temperature: float = 1.0,
    refine_iters: int = 10,
    original_input_path: str | None = None,
    api_key: str | None = None,
    planner_addendum: str = "",
    refiner_addendum: str = "",
    planner_custom_design_rules: str | None = None,
    critic_custom_design_rules: str | None = None,
    refiner_custom_design_rules: str | None = None,
) -> None:
    print("\n=== Running macro placement with Flex DSL and iterative refinement ===")

    def iter_dirs(base: Path, idx: int) -> Tuple[Path, Path, Path, Path, Path]:
        iter_name = f"iteration_{idx:02d}"
        out_iter = base / iter_name
        out_final = out_iter / "final_product"
        out_in_text = out_iter / "vlm_input_text"
        out_in_img = out_iter / "vlm_input_image"
        out_vlm = out_iter / "vlm_output"
        out_layout = out_iter / "layout_json"
        for d in [out_iter, out_final, out_in_text, out_in_img, out_vlm, out_layout]:
            ensure_dir(d)
        return out_final, out_in_text, out_in_img, out_vlm, out_layout

    base_out = SCRIPT_DIR / "output_macro_placement" / output_dir.name
    # Clean previous outputs for this image to avoid mixing runs
    try:
        if base_out.exists():
            shutil.rmtree(base_out)
    except Exception:
        pass
    ensure_dir(base_out)

    timer = StepTimer()

    bg_path = output_dir / "background.png"
    results_json_path = output_dir / "results.json"
    objects_dir = output_dir / "objects"

    with timer.time_step("prepare"):
        ow, oh = read_original_size(output_dir)
        canvas_size = compute_canvas_size((ow, oh), ratio)
        margin = margin # Use the margin argument directly
        meta = {
            "ratio": ratio,
            "align": align,
            "margin": margin,
            "api": api_type,
            "canvas_size": {"width": canvas_size[0], "height": canvas_size[1]},
            "original_image": {"width": ow, "height": oh},
            "samples": samples,
            "temperature": temperature,
            "refine_iters": refine_iters,
        }

    # iteration_00: baseline
    with timer.time_step("contact_sheet"):
        sheet = _build_labeled_contact_sheet(str(objects_dir), str(results_json_path))

    out_final_0, out_text_0, out_img_0, out_vlm_0, out_layout_0 = iter_dirs(base_out, 0)
    # Save metadata and inputs
    with open(out_text_0 / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    sheet.save(out_img_0 / "contact_sheet.png")
    # Copy background for reference
    try:
        shutil.copyfile(bg_path, out_img_0 / "background.png")
    except Exception:
        pass
    # Build and save the target-ratio background canvas for VLM inputs
    canvas_img_boot = fill_solid(str(bg_path), canvas_size)
    canvas_path_0 = out_img_0 / "canvas.png"
    try:
        canvas_img_boot.save(canvas_path_0)
    except Exception:
        pass
    # Copy the original input image if available
    if original_input_path:
        try:
            src_name = Path(original_input_path).name
            shutil.copyfile(original_input_path, out_img_0 / src_name)
        except Exception:
            pass

    with timer.time_step("vlm_flex_baseline"):
        flex_raw, prompt_text, _summary_text, role_lines, raw_text, best_practices = _vlm_request_flex(
            sheet,
            str(canvas_path_0),
            original_input_path or "",
            str(results_json_path),
            ratio,
            canvas_size,
            api_type,
            temperature,
            margin,
            planner_addendum=planner_addendum,
            api_key=api_key,
            planner_custom_design_rules=planner_custom_design_rules,
        )
        with open(out_vlm_0 / "layout_flex_iter_00.json", "w", encoding="utf-8") as f:
            json.dump(flex_raw, f, indent=2)
        (out_vlm_0 / "vlm_raw_iter_00.txt").write_text(raw_text, encoding="utf-8")
        (out_text_0 / "prompt_flex.txt").write_text(prompt_text, encoding="utf-8")
        (out_text_0 / "best_practices.txt").write_text(best_practices, encoding="utf-8")

        with open(results_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        required_ids = [int(it["object_id"]) for it in items]
        id_to_label = {int(it["object_id"]): str(it.get("label", "")).strip() for it in items}
        # Compute conflicts once
        row_bad, col_bad = _compute_nesting_conflicts(str(results_json_path), canvas_size, margin)
        try:
            _validate_flex_dsl(
                flex_raw,
                required_ids,
                id_to_label,
                row_bad_pairs={frozenset({a, b}) for a, b in row_bad},
                col_bad_pairs={frozenset({a, b}) for a, b in col_bad},
                allow_container_spacing=False,
                allow_container_align=False,
                forbid_root_spacing=True,
                drop_container_styling=True,
                sanitize=True,
            )
        except Exception as e:
            print(f"[validate] Baseline layout failed validation: {e}")
            (out_text_0 / "flex_validation_error_iter_00.txt").write_text(str(e), encoding="utf-8")
            failed_path = out_vlm_0 / "failed_output.txt"
            with open(failed_path, "a", encoding="utf-8") as f_failed:
                f_failed.write(f"validation_error: {e}\n")
                f_failed.write(json.dumps(flex_raw, indent=2))
                f_failed.write("\n---\n")
            # Continue — inputs already saved; subsequent refine step may fix
            pass

    with timer.time_step("compose_baseline"):
        objects = load_object_images(str(results_json_path))
        placements_px: List[Dict] = []
        root = flex_raw["root"]
        parent_cell = "flex_root"
        _place_flex_container(root, (0, 0), canvas_size, objects, placements_px, parent_cell)
        _clamp_boxes_to_canvas(placements_px, canvas_size)

        final_json = {
            "canvas": {"width": canvas_size[0], "height": canvas_size[1], "margin": margin, "align": align},
            "placements": [
                {**p, "name": id_to_label.get(int(p["object_id"]), str(int(p["object_id"])))} for p in placements_px
            ],
        }
        with open(out_layout_0 / "layout_macro_iter_00.json", "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)

        # Use the precomputed canvas.png for composition to avoid recomputing
        bg_img = Image.open(canvas_path_0).convert("RGBA")
        draft = composite(bg_img, objects, final_json["placements"])
        draft_path_prev = out_final_0 / "draft_macro_iter_00.png"
        draft.save(draft_path_prev)
        _save_overlay_debug(final_json["placements"], canvas_size, out_final_0 / "overlay_debug_iter_00.png")
        provenance = {"method": "flex", "fallback": False, "iteration": 0}
        with open(out_layout_0 / "provenance_iter_00.json", "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)

    # Refinement iterations
    contact_sheet_img = sheet  # reuse in memory
    previous_critic_text = ""
    previous_refiner_json: Dict | None = None
    for i in range(1, max(0, refine_iters) + 1):
        out_final_i, out_text_i, out_img_i, out_vlm_i, out_layout_i = iter_dirs(base_out, i)
        # Save composite_prev reference
        try:
            shutil.copyfile(draft_path_prev, out_img_i / "composite_prev.png")
        except Exception:
            pass
        # Save background and canvas again for completeness
        try:
            shutil.copyfile(bg_path, out_img_i / "background.png")
        except Exception:
            pass
        try:
            # Reuse the baseline canvas for subsequent iterations (same canvas_size)
            shutil.copyfile(canvas_path_0, out_img_i / "canvas.png")
        except Exception:
            pass
        # Critic step: evaluate current draft
        with timer.time_step(f"vlm_critic_iter_{i:02d}"):
            context_note = "Do not produce JSON. Provide an honest critique with score and violations."
            critic_prompt, critic_raw = _vlm_request_critic(
                contact_sheet_img,
                original_input_path or str(bg_path),
                str(draft_path_prev),
                flex_raw,
                previous_critic_text,
                previous_refiner_json,
                row_bad,
                col_bad,
                ratio,
                api_type,
                best_practices,
                _summary_text,
                role_lines,
                context_note,
                api_key=api_key,
                critic_custom_design_rules=critic_custom_design_rules,
            )
            (out_text_i / f"critic_prompt_iter_{i:02d}.txt").write_text(critic_prompt, encoding="utf-8")
            (out_vlm_i / f"critic_raw_iter_{i:02d}.txt").write_text(critic_raw, encoding="utf-8")

        # Build refine request (improver), include critic text
        with timer.time_step(f"vlm_refine_iter_{i:02d}"):
            extra_instr = ""
            refine_raw, refine_prompt, refine_raw_text = _vlm_request_refine(
                contact_sheet_img,
                str(canvas_path_0),
                original_input_path or "",
                str(draft_path_prev),
                flex_raw,
                ratio,
                canvas_size,
                api_type,
                temperature,
                required_ids,
                id_to_label,
                row_bad,
                col_bad,
                critic_raw,
                _summary_text,
                role_lines,
                extra_instr,
                refiner_addendum=refiner_addendum,
                api_key=api_key,
                refiner_custom_design_rules=refiner_custom_design_rules,
            )
            with open(out_vlm_i / f"layout_flex_iter_{i:02d}.json", "w", encoding="utf-8") as f:
                json.dump(refine_raw, f, indent=2)
            (out_vlm_i / f"vlm_raw_iter_{i:02d}.txt").write_text(refine_raw_text, encoding="utf-8")
            (out_text_i / f"prompt_refine_iter_{i:02d}.txt").write_text(refine_prompt, encoding="utf-8")

            try:
                _validate_flex_dsl(
                    refine_raw,
                    required_ids,
                    id_to_label,
                    row_bad_pairs={frozenset({a, b}) for a, b in row_bad},
                    col_bad_pairs={frozenset({a, b}) for a, b in col_bad},
                    allow_object_tuning=True,
                    allow_container_spacing=True,
                    allow_container_align=True,
                    forbid_root_spacing=True,
                )
            except Exception as e:
                print(f"[validate] Iter {i:02d} refine failed validation: {e}")
                (out_text_i / f"flex_validation_error_iter_{i:02d}.txt").write_text(str(e), encoding="utf-8")
                failed_path = out_vlm_i / f"failed_output_iter_{i:02d}.txt"
                with open(failed_path, "a", encoding="utf-8") as f_failed:
                    f_failed.write(f"validation_error: {e}\n")
                    f_failed.write(json.dumps(refine_raw, indent=2))
                    f_failed.write("\n---\n")
                # Save inputs are already persisted; Retry once with explicit validator feedback
                extra_instr = str(e)
                refine_raw, refine_prompt, refine_raw_text = _vlm_request_refine(
                    contact_sheet_img,
                    str(canvas_path_0),
                    original_input_path or "",
                    str(draft_path_prev),
                    flex_raw,
                    ratio,
                    canvas_size,
                    api_type,
                    temperature,
                    required_ids,
                    id_to_label,
                    row_bad,
                    col_bad,
                    critic_raw,
                    _summary_text,
                    role_lines,
                    extra_instr,
                    refiner_addendum=refiner_addendum,
                    api_key=api_key,
                    refiner_custom_design_rules=refiner_custom_design_rules,
                )
                with open(out_vlm_i / f"layout_flex_iter_{i:02d}_retry.json", "w", encoding="utf-8") as f:
                    json.dump(refine_raw, f, indent=2)
                (out_vlm_i / f"vlm_raw_iter_{i:02d}_retry.txt").write_text(refine_raw_text, encoding="utf-8")
                (out_text_i / f"prompt_refine_iter_{i:02d}_retry.txt").write_text(refine_prompt, encoding="utf-8")
                # Validate again or record error then continue to next iteration
                try:
                    _validate_flex_dsl(
                        refine_raw,
                        required_ids,
                        id_to_label,
                        row_bad_pairs={frozenset({a, b}) for a, b in row_bad},
                        col_bad_pairs={frozenset({a, b}) for a, b in col_bad},
                        allow_object_tuning=True,
                        allow_container_spacing=True,
                        allow_container_align=True,
                        forbid_root_spacing=True,
                    )
                except Exception as e2:
                    print(f"[validate] Iter {i:02d} refine retry failed validation: {e2}")
                    (out_text_i / f"flex_validation_error_iter_{i:02d}_retry.txt").write_text(str(e2), encoding="utf-8")
                    failed_retry_path = out_vlm_i / f"failed_output_iter_{i:02d}_retry.txt"
                    with open(failed_retry_path, "a", encoding="utf-8") as f_failed:
                        f_failed.write(f"validation_error: {e2}\n")
                        f_failed.write(json.dumps(refine_raw, indent=2))
                        f_failed.write("\n---\n")
                    # do not raise — we want artifacts preserved for inspection
                    pass

            # Stop if no structural change (identical JSON)
            try:
                prev_norm = json.dumps(flex_raw, sort_keys=True)
                cur_norm = json.dumps(refine_raw, sort_keys=True)
                if prev_norm == cur_norm:
                    print(f"[refine] Iter {i:02d} produced identical JSON; skipping compose.")
                    (out_text_i / f"no_change_iter_{i:02d}.txt").write_text("Refined JSON identical to previous; aborting.", encoding="utf-8")
                    # Skip raising; continue to compose to keep artifacts consistent
                    continue
            except Exception:
                pass

        with timer.time_step(f"compose_iter_{i:02d}"):
            objects = load_object_images(str(results_json_path))
            placements_px_i: List[Dict] = []
            root_i = refine_raw["root"]
            parent_cell = "flex_root"
            _place_flex_container(root_i, (0, 0), canvas_size, objects, placements_px_i, parent_cell)
            _clamp_boxes_to_canvas(placements_px_i, canvas_size)

            final_json_i = {
                "canvas": {"width": canvas_size[0], "height": canvas_size[1], "margin": margin, "align": align},
                "placements": [
                    {**p, "name": id_to_label.get(int(p["object_id"]), str(int(p["object_id"])))} for p in placements_px_i
                ],
            }
            with open(out_layout_i / f"layout_macro_iter_{i:02d}.json", "w", encoding="utf-8") as f:
                json.dump(final_json_i, f, indent=2)

            # Use the same precomputed canvas for all iterations
            bg_img = Image.open(canvas_path_0).convert("RGBA")
            draft_i = composite(bg_img, objects, final_json_i["placements"])
            draft_path_prev = out_final_i / f"draft_macro_iter_{i:02d}.png"
            draft_i.save(draft_path_prev)
            _save_overlay_debug(final_json_i["placements"], canvas_size, out_final_i / f"overlay_debug_iter_{i:02d}.png")
            provenance_i = {"method": "flex_refine", "fallback": False, "iteration": i}
            with open(out_layout_i / f"provenance_iter_{i:02d}.json", "w", encoding="utf-8") as f:
                json.dump(provenance_i, f, indent=2)

        # Update current flex_raw for next iteration input
        flex_raw = refine_raw
        previous_refiner_json = refine_raw
        previous_critic_text = critic_raw

    # Write timer at the base_out level
    timer.write_to_file(str(base_out / "time_log.txt"))
    print(f"Macro outputs (with refinements) saved to: {base_out}")


def main():
    parser = argparse.ArgumentParser(description="Macro placement using VLM Flex-DSL with iterative refinement and deterministic compositor.")
    parser.add_argument("--image", required=True, help="Path to input image used with auto-segmenter (e.g., input/ms_laptop.png)")
    parser.add_argument("--ratio", required=True, help="Target aspect ratio W:H (e.g., 9:16)")
    parser.add_argument("--align", default="center", choices=["center", "edge"], help="Alignment mode")
    parser.add_argument("--margin", type=float, default=0.05, help="Safe margin percentage (0-0.3)")
    parser.add_argument("--api", choices=["auto", "ollama", "nebius"], default="auto", help="API to use for VLM stage")
    parser.add_argument("--samples", type=int, default=1, help="Number of macro candidates to generate (default: 1)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for VLM (default: 1.0)")
    parser.add_argument("--refine-iters", type=int, default=10, help="Number of refinement iterations (default: 10; can be 0, 5, 15, ...)")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output_dir = image_path.parent.parent / "output" / image_path.stem
    if not output_dir.exists():
        raise FileNotFoundError(f"Expected segmentation outputs at {output_dir}")

    for d in [SCRIPT_DIR / "output_macro_placement" / output_dir.name]:
        ensure_dir(d)

    run_macro_only(
        output_dir,
        args.ratio,
        args.align,
        args.margin,
        api_type=args.api,
        samples=args.samples,
        temperature=args.temperature,
        refine_iters=args.refine_iters,
        original_input_path=str(image_path),
    )


if __name__ == "__main__":
    main()