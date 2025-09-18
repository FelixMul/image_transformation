from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
from PIL import Image
import json
import math
import os


CellName = str


GRID_CELLS: List[CellName] = [
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]


@dataclass
class ObjectMeta:
    object_id: int
    label: str
    file: str
    width: int
    height: int


@dataclass
class Placement:
    object_id: int
    cell: CellName
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    scale: float


def parse_ratio(ratio: str) -> float:
    parts = ratio.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid ratio '{ratio}', expected W:H")
    w = float(parts[0])
    h = float(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError("Ratio components must be positive")
    return w / h


def compute_canvas_size(original_size: Tuple[int, int], ratio: str) -> Tuple[int, int]:
    """
    Compute canvas size that preserves total pixel count while achieving target aspect ratio.
    
    Rule: W * H ≈ original_pixels, where W/H = target_ratio
    This ensures comparable image size regardless of aspect ratio transformation.
    """
    ow, oh = original_size
    original_pixels = ow * oh
    target_ratio = parse_ratio(ratio)  # W/H
    
    # From W/H = target_ratio and W*H = original_pixels:
    # H = W / target_ratio
    # W * (W / target_ratio) = original_pixels
    # W² / target_ratio = original_pixels
    # W² = original_pixels * target_ratio
    # W = sqrt(original_pixels * target_ratio)
    
    target_width = math.sqrt(original_pixels * target_ratio)
    target_height = math.sqrt(original_pixels / target_ratio)
    
    # Round to integers
    tw = int(round(target_width))
    th = int(round(target_height))
    
    # Ensure minimum size
    tw = max(1, tw)
    th = max(1, th)
    
    print(f"Canvas sizing: {ow}x{oh} ({original_pixels:,} px) → {tw}x{th} ({tw*th:,} px, ratio {tw/th:.3f})")
    
    return tw, th


def grid_cells(canvas_size: Tuple[int, int], margin_pct: float) -> Dict[CellName, Tuple[int, int, int, int]]:
    tw, th = canvas_size
    mx = int(round(tw * margin_pct))
    my = int(round(th * margin_pct))
    x1 = mx
    y1 = my
    x2 = tw - mx
    y2 = th - my
    cw = x2 - x1
    ch = y2 - y1
    col_w = cw // 3
    row_h = ch // 3

    rects: Dict[CellName, Tuple[int, int, int, int]] = {}
    names = [
        ("top_left", 0, 0), ("top_center", 1, 0), ("top_right", 2, 0),
        ("middle_left", 0, 1), ("middle_center", 1, 1), ("middle_right", 2, 1),
        ("bottom_left", 0, 2), ("bottom_center", 1, 2), ("bottom_right", 2, 2),
    ]
    for name, cx, cy in names:
        sx = x1 + cx * col_w
        sy = y1 + cy * row_h
        ex = sx + col_w
        ey = sy + row_h
        rects[name] = (sx, sy, ex, ey)
    return rects


def _cell_row_col(cell: CellName) -> Tuple[int, int]:
    idx = GRID_CELLS.index(cell)
    row = idx // 3
    col = idx % 3
    return row, col


def _load_object_meta(objects_dir: str, results_json_path: str) -> Dict[int, ObjectMeta]:
    with open(results_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    meta: Dict[int, ObjectMeta] = {}
    for it in items:
        oid = int(it["object_id"])
        file_rel = it["filename"]
        file_abs = os.path.join(os.path.dirname(results_json_path), file_rel)
        with Image.open(file_abs).convert("RGBA") as im:
            w, h = im.size
        meta[oid] = ObjectMeta(
            object_id=oid,
            label=it.get("label", ""),
            file=file_abs,
            width=w,
            height=h,
        )
    return meta


def baseline_cell_assignments(results_json_path: str) -> List[Tuple[int, CellName]]:
    with open(results_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    # Largest area first
    sized = []
    for it in items:
        x1, y1, x2, y2 = it.get("bounding_box", [0, 0, 0, 0])
        area = max(1, int((x2 - x1) * (y2 - y1)))
        sized.append((int(it["object_id"]), area, it.get("label", "")))
    sized.sort(key=lambda t: t[1], reverse=True)

    order = [
        "middle_center",
        "top_center",
        "bottom_center",
        "middle_left",
        "middle_right",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ]
    # naive logo/text heuristic
    # place any label containing 'logo' or 'text' to top_right if free, else bottom_right
    placements: List[Tuple[int, CellName]] = []
    used: set = set()
    # First hero (largest)
    if sized:
        hero_id = sized[0][0]
        placements.append((hero_id, "middle_center"))
        used.add("middle_center")
    # Prioritize logo/text
    for oid, _, label in sized[1:]:
        low = label.lower()
        if "logo" in low or "text" in low:
            target = "top_right" if "top_right" not in used else (
                "bottom_right" if "bottom_right" not in used else None
            )
            if target:
                placements.append((oid, target))
                used.add(target)
    # Fill the rest
    for oid, _, _ in sized[1:]:
        if oid in [p[0] for p in placements]:
            continue
        for cell in order:
            if cell not in used:
                placements.append((oid, cell))
                used.add(cell)
                break
    return placements


def finalize_layout(assignments: List[Tuple[int, CellName]], results_json_path: str, canvas_size: Tuple[int, int], cells: Dict[CellName, Tuple[int, int, int, int]], align: str = "center", spacing_px: int = 8) -> List[Placement]:
    meta = _load_object_meta(os.path.join(os.path.dirname(results_json_path), "objects"), results_json_path)
    # Group by cell
    by_cell: Dict[CellName, List[ObjectMeta]] = {name: [] for name in GRID_CELLS}
    for oid, cell in assignments:
        if cell not in by_cell:
            by_cell[cell] = []
        if oid not in meta:
            continue
        by_cell[cell].append(meta[oid])

    placements: List[Placement] = []
    for cell, objs in by_cell.items():
        if not objs:
            continue
        cell_rect = cells[cell]
        x1, y1, x2, y2 = cell_rect
        cw = max(1, x2 - x1)
        ch = max(1, y2 - y1)
        n = len(objs)
        if n == 1:
            om = objs[0]
            scale = 1.0  # No scaling - keep original size
            w = om.width
            h = om.height
            if align == "center":
                px = x1 + (cw - w) // 2
                py = y1 + (ch - h) // 2
            else:
                row, col = _cell_row_col(cell)
                px = x1 if col == 0 else (x1 + (cw - w) // 2 if col == 1 else x2 - w)
                py = y1 if row == 0 else (y1 + (ch - h) // 2 if row == 1 else y2 - h)
            placements.append(Placement(om.object_id, cell, (px, py, px + w, py + h), scale))
        else:
            # tile horizontally if wider than tall, else vertically
            horizontal = cw >= ch
            if horizontal:
                s = 1.0  # No scaling - keep original size
                cur_x = x1
                total_scaled_w = sum(o.width for o in objs) + spacing_px * (n - 1)
                if align == "center":
                    cur_x = x1 + (cw - total_scaled_w) // 2
                elif _cell_row_col(cell)[1] == 2:
                    cur_x = x2 - total_scaled_w
                for o in objs:
                    w = o.width
                    h = o.height
                    if align == "center":
                        py = y1 + (ch - h) // 2
                    else:
                        row = _cell_row_col(cell)[0]
                        py = y1 if row == 0 else (y1 + (ch - h) // 2 if row == 1 else y2 - h)
                    placements.append(Placement(o.object_id, cell, (cur_x, py, cur_x + w, py + h), s))
                    cur_x += w + spacing_px
            else:
                s = 1.0  # No scaling - keep original size
                cur_y = y1
                total_scaled_h = sum(o.height for o in objs) + spacing_px * (n - 1)
                if align == "center":
                    cur_y = y1 + (ch - total_scaled_h) // 2
                elif _cell_row_col(cell)[0] == 2:
                    cur_y = y2 - total_scaled_h
                for o in objs:
                    w = o.width
                    h = o.height
                    if align == "center":
                        px = x1 + (cw - w) // 2
                    else:
                        col = _cell_row_col(cell)[1]
                        px = x1 if col == 0 else (x1 + (cw - w) // 2 if col == 1 else x2 - w)
                    placements.append(Placement(o.object_id, cell, (px, cur_y, px + w, cur_y + h), s))
                    cur_y += h + spacing_px

    return placements


def pack_flow(
    # La fonction reçoit maintenant directement les objets avec leur taille déjà mise à jour
    scaled_objs: List[ObjectMeta], 
    canvas_size: Tuple[int, int],
    layout_params: Dict,
    meta: Dict[int, ObjectMeta]
) -> Tuple[List[Placement], Tuple[int, int]]:
    """
    Génère un layout en 'flow' à partir d'objets déjà mis à l'échelle.
    Cette version ne charge plus de données, elle ne fait que positionner.
    """
    # 1. Extraire les paramètres de positionnement (on ignore les échelles ici)
    align = layout_params.get("align", "center")
    orientation = layout_params.get("orientation", "auto")
    global_margin = layout_params.get("global_margin_px", 20)
    global_spacing = layout_params.get("global_spacing_px", 20)
    # Plus besoin de charger les meta ou d'appliquer les object_scales, c'est déjà fait en amont !

    tw, th = canvas_size
    if orientation == "auto":
        orientation = "vertical" if th >= tw else "horizontal"

    # 2. Calculer le placement en utilisant directement les objets reçus
    placements: List[Placement] = []
    if orientation == "vertical":
        content_height = sum(o.height for o in scaled_objs)
        total_content_size = content_height + (len(scaled_objs) - 1) * global_spacing
        cursor_y = (th - total_content_size) // 2  # Centrage vertical

        for o in scaled_objs:
            px = (tw - o.width) // 2 if align == "center" else global_margin
            py = cursor_y
            
            # On récupère l'échelle actuelle pour la stocker dans le JSON final
            current_scale = o.width / meta[o.object_id].width if meta[o.object_id].width > 0 else 1.0

            placements.append(Placement(o.object_id, "flow_vertical", (px, py, px + o.width, py + o.height), current_scale))
            cursor_y += o.height + global_spacing

    else: # Horizontal
        content_width = sum(o.width for o in scaled_objs)
        total_content_size = content_width + (len(scaled_objs) - 1) * global_spacing
        cursor_x = (tw - total_content_size) // 2 # Centrage horizontal

        for o in scaled_objs:
            py = (th - o.height) // 2 if align == "center" else global_margin
            px = cursor_x

            current_scale = o.width / meta[o.object_id].width if meta[o.object_id].width > 0 else 1.0

            placements.append(Placement(o.object_id, "flow_horizontal", (px, py, px + o.width, py + o.height), current_scale))
            cursor_x += o.width + global_spacing
            
    # La fonction retourne la taille de toile originale, non modifiée
    return placements, canvas_size


def layout_final_json(placements: List[Placement], canvas_size: Tuple[int, int], margin_pct: float, align: str) -> Dict:
    data = {
        "canvas": {"width": canvas_size[0], "height": canvas_size[1], "margin": margin_pct, "align": align},
        "placements": [],
    }
    for p in placements:
        data["placements"].append({
            "object_id": p.object_id,
            "cell": p.cell,
            "box": [int(p.box[0]), int(p.box[1]), int(p.box[2]), int(p.box[3])],
            "scale": float(p.scale),
        })
    return data


