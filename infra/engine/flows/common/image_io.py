from __future__ import annotations

from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str) -> List[Path]:
    root = Path(folder)
    return [path for path in sorted(root.iterdir()) if path.suffix.lower() in IMG_EXTS]
