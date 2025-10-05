from pathlib import Path
from typing import List, Tuple

SRC_DIR = Path(__file__).resolve().parent
IMAGES_DIR = SRC_DIR / "images"
BALL_IMAGES_DIR = IMAGES_DIR / "balls"
BULLSEYE_TARGET_IMAGE = BALL_IMAGES_DIR / "BullseyeTarget.png"
TABLE_IMAGES_DIR = IMAGES_DIR / "table"
LAYOUTS_DIR = SRC_DIR / "layouts"
TOURNAMENTS_DIR = LAYOUTS_DIR / "Tournaments"

IMAGE_FILETYPES: List[Tuple[str, str]] = [
    (
        "Images",
        "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff *.PNG *.JPG *.JPEG *.BMP *.GIF *.WEBP *.TIF *.TIFF",
    ),
    ("All Files", "*.*"),
]

JSON_FILETYPES: List[Tuple[str, str]] = [("JSON files", "*.json"), ("All Files", "*.*")]


def ensure_directories() -> None:
    for path in (IMAGES_DIR, BALL_IMAGES_DIR, TABLE_IMAGES_DIR, LAYOUTS_DIR, TOURNAMENTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


ensure_directories()
