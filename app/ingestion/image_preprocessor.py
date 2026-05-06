"""Image normalization helpers."""
from __future__ import annotations

from PIL import Image


def resize_max_edge(image: Image.Image, max_edge: int = 1280) -> Image.Image:
    """Resize so the longest edge equals max_edge, preserving aspect ratio."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image.convert("RGB") if image.mode != "RGB" else image
    scale = max_edge / longest
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS).convert("RGB")
