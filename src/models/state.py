from dataclasses import dataclass
from typing import Optional


@dataclass
class BallState:
    name: str
    x: Optional[float] = None
    y: Optional[float] = None
    visible: bool = True
    path: Optional[str] = None
    u: Optional[float] = None
    v: Optional[float] = None


@dataclass
class DrawingState:
    kind: str
    color: str
    width: int
    u1: float
    v1: float
    u2: float
    v2: float


@dataclass(frozen=True)
class ShotReference:
    tournament_path: str
    match_index: int
    rack_index: int
    shot_index: int
