from PIL import Image, ImageStat
import numpy as np
from typing import Tuple


def _load_background_rgba(background_path: str) -> Image.Image:
    img = Image.open(background_path).convert("RGBA")
    return img


def _median_color_nontransparent(img_rgba: Image.Image) -> Tuple[int, int, int]:
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3]
    mask = alpha > 0
    if not np.any(mask):
        # fallback to overall median if fully transparent (unlikely)
        rgb = arr[:, :, :3].reshape(-1, 3)
        med = np.median(rgb, axis=0)
        return tuple(int(x) for x in med.tolist())
    rgb = arr[:, :, :3][mask]
    med = np.median(rgb, axis=0)
    return tuple(int(x) for x in med.tolist())


def fill_solid(background_path: str, canvas_size: Tuple[int, int]) -> Image.Image:
    """Create a solid background using the median non-transparent color of background.png.

    Returns an RGBA image of size canvas_size.
    """
    bg = _load_background_rgba(background_path)
    color = _median_color_nontransparent(bg)
    solid = Image.new("RGBA", canvas_size, color + (255,))
    return solid


def _edge_strip_median_colors(img: Image.Image, strip_px: int = 8) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[0], arr.shape[1]
    a = arr[:, :, 3]

    def med_rgb(region):
        alpha = region[:, :, 3]
        mask = alpha > 0
        if np.any(mask):
            rgb = region[:, :, :3][mask]
        else:
            rgb = region[:, :, :3].reshape(-1, 3)
        med = np.median(rgb, axis=0)
        return tuple(int(x) for x in med.tolist())

    left = med_rgb(arr[:, :min(strip_px, w), :])
    right = med_rgb(arr[:, max(0, w - strip_px):, :])
    top = med_rgb(arr[:min(strip_px, h), :, :])
    bottom = med_rgb(arr[max(0, h - strip_px):, :, :])
    return left, right, top, bottom


def _axis_variance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    # simple squared distance as variance proxy
    return float((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)


def fill_gradient(background_path: str, canvas_size: Tuple[int, int]) -> Image.Image:
    """Create a linear gradient background from edge medians.

    Chooses horizontal or vertical direction based on lower color variance.
    """
    bg = _load_background_rgba(background_path)
    left, right, top, bottom = _edge_strip_median_colors(bg)

    horiz_var = _axis_variance(left, right)
    vert_var = _axis_variance(top, bottom)

    width, height = canvas_size
    gradient = Image.new("RGBA", (width, height))
    arr = np.zeros((height, width, 4), dtype=np.uint8)

    if horiz_var <= vert_var:
        # horizontal gradient left -> right
        c1 = np.array(left, dtype=np.float32)
        c2 = np.array(right, dtype=np.float32)
        for x in range(width):
            t = x / max(1, width - 1)
            rgb = (1 - t) * c1 + t * c2
            arr[:, x, :3] = rgb.astype(np.uint8)
        arr[:, :, 3] = 255
    else:
        # vertical gradient top -> bottom
        c1 = np.array(top, dtype=np.float32)
        c2 = np.array(bottom, dtype=np.float32)
        for y in range(height):
            t = y / max(1, height - 1)
            rgb = (1 - t) * c1 + t * c2
            arr[y, :, :3] = rgb.astype(np.uint8)
        arr[:, :, 3] = 255

    gradient = Image.fromarray(arr, mode="RGBA")
    return gradient


