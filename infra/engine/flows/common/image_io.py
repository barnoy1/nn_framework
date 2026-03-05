from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str) -> List[Path]:
    root = Path(folder)
    return [path for path in sorted(root.iterdir()) if path.suffix.lower() in IMG_EXTS]


def load_pil_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        if image.mode == "P" and isinstance(image.info.get("transparency"), bytes):
            prepared = image.convert("RGBA").convert("RGB")
        elif image.mode == "RGB":
            prepared = image.copy()
        elif image.mode == "L":
            prepared = image.convert("RGB")
        elif "A" in image.mode:
            prepared = image.convert("RGBA").convert("RGB")
        else:
            prepared = image.convert("RGB")
        return prepared


def load_rgb_image(image_path: Path) -> np.ndarray:
    return np.array(load_pil_image(image_path))
