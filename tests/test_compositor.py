from PIL import Image
from compositor import composite


def test_composite_places_object_pixel():
    bg = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
    obj = Image.new("RGBA", (2, 2), (0, 255, 0, 255))
    objects = {1: obj}
    placements = [{"object_id": 1, "box": [4, 4, 6, 6]}]
    out = composite(bg, objects, placements)
    assert out.getpixel((4, 4))[:3] == (0, 255, 0)
