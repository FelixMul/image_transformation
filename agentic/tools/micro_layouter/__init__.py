"""Micro layouter tools for pixel-level adjustments."""

from .positioning import adjust_x, adjust_y

TOOL_REGISTRY = {
    "adjust_y": adjust_y,
    "adjust_x": adjust_x,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "adjust_y",
            "description": "Move an object vertically by a number of pixels (positive=down, negative=up).",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "The object identifier; accepts label text or numeric id.",
                    },
                    "pixels": {
                        "type": "integer",
                        "description": "The number of pixels to move. Positive moves down, negative moves up.",
                    },
                },
                "required": ["object", "pixels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_x",
            "description": "Move an object horizontally by a number of pixels (positive=right, negative=left).",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "The object identifier; accepts label text or numeric id.",
                    },
                    "pixels": {
                        "type": "integer",
                        "description": "The number of pixels to move. Positive moves right, negative moves left.",
                    },
                },
                "required": ["object", "pixels"],
            },
        },
    },
]

__all__ = ["TOOL_REGISTRY", "TOOL_DEFINITIONS"]


