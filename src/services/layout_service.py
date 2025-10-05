from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from config import IMAGES_DIR, LAYOUTS_DIR, TABLE_IMAGES_DIR
from services.asset_service import normalize_asset_path


class LayoutService:
    """Handles layout persistence and asset path normalization."""

    def __init__(
        self,
        layouts_dir: Path | str = LAYOUTS_DIR,
        images_dir: Path | str = IMAGES_DIR,
        tables_dir: Path | str = TABLE_IMAGES_DIR,
    ) -> None:
        self.layouts_dir = Path(layouts_dir)
        self.images_dir = Path(images_dir)
        self.tables_dir = Path(tables_dir)

    # ----------------------------- Persistence -----------------------------
    def read(self, path: str | Path) -> Dict[str, Any]:
        target = Path(path)
        with target.open("r", encoding="utf-8") as handle:
            data: Dict[str, Any] = json.load(handle)
        return data

    def write(self, path: str | Path, data: Dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    # ----------------------------- Normalisation -----------------------------
    def sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a deepcopy with asset paths rebased onto the local images directories."""
        sanitized = copy.deepcopy(data)

        table_path = sanitized.get("table")
        normalized_table = normalize_asset_path(table_path, str(self.tables_dir))
        if normalized_table and Path(normalized_table).exists():
            sanitized["table"] = normalized_table

        balls = sanitized.get("balls")
        if isinstance(balls, list):
            for ball in balls:
                if not isinstance(ball, dict):
                    continue
                ball_path = ball.get("path")
                normalized_ball = normalize_asset_path(ball_path, str(self.images_dir))
                if normalized_ball and Path(normalized_ball).exists():
                    ball["path"] = normalized_ball
        return sanitized

    # ----------------------------- Application -----------------------------
    def apply_to_canvas(self, canvas: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = self.sanitize(data)
        canvas.restore(sanitized)
        return sanitized


__all__ = ["LayoutService"]
