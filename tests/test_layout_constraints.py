import math
from layout_constraints import compute_canvas_size, parse_ratio, grid_cells


def test_compute_canvas_size_preserves_ratio_and_pixels():
    ow, oh = 1920, 1080
    ratio = "9:16"
    tw, th = compute_canvas_size((ow, oh), ratio)
    target = parse_ratio(ratio)
    assert abs((tw / th) - target) < 0.02
    orig_px = ow * oh
    new_px = tw * th
    assert abs(new_px - orig_px) / orig_px < 0.02


def test_grid_cells_bounds_and_count():
    tw, th = 1000, 2000
    cells = grid_cells((tw, th), margin_pct=0.05)
    assert len(cells) == 9
    for (x1, y1, x2, y2) in cells.values():
        assert 0 <= x1 < x2 <= tw
        assert 0 <= y1 < y2 <= th
