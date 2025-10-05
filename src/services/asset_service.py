"""Asset cataloging and path normalization helpers."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

try:
    from config import BALL_IMAGES_DIR, IMAGES_DIR
except ImportError:  # pragma: no cover
    from ..config import BALL_IMAGES_DIR, IMAGES_DIR


def list_ball_assets(extra_directories: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    assets: List[Tuple[str, str]] = []
    search_dirs = [BALL_IMAGES_DIR]
    if extra_directories:
        search_dirs.extend(os.path.abspath(path) for path in extra_directories)

    seen = set()
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            if not entry.lower().endswith(".png"):
                continue
            name = os.path.splitext(entry)[0]
            path = os.path.join(directory, entry)
            normalized = os.path.normpath(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            assets.append((name, normalized))
    return assets


def _rebase_path_to_root(path: str, root: str) -> Optional[str]:
    if not path or not root:
        return None
    normalized = path.replace("\\", "/")
    root_name = os.path.basename(root).lower()
    marker = f"/{root_name}/"
    lowered = normalized.lower()
    idx = lowered.rfind(marker)
    if idx != -1:
        relative = normalized[idx + len(marker):].strip("/")
        if relative:
            candidate = os.path.join(root, *relative.split("/"))
            if os.path.exists(candidate):
                return candidate
    tail = os.path.basename(path)
    if tail:
        candidate = os.path.join(root, tail)
        if os.path.exists(candidate):
            return candidate
    return None


def normalize_asset_path(path: Optional[str], root: str = str(IMAGES_DIR)) -> Optional[str]:
    if not path:
        return path
    candidate = os.path.normpath(path)
    if os.path.exists(candidate):
        return candidate
    rebased = _rebase_path_to_root(candidate, root)
    return rebased if rebased else candidate

