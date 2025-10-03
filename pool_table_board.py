#!/usr/bin/env python3
"""
Pool Table Board (Presenter Mode) — Table-relative & Drawing Tools

New:
- Draw lines and arrows with variable width and color.
- Toolbar: Tool (Line/Arrow), Color picker, Width slider, Undo, Clear.
- Drawings mirror to projector and are saved/loaded with the layout.
- All ball positions and drawings use table-relative (u,v) so they match across displays.

Dependencies
    pip install pillow screeninfo
Run
    python pool_table_board.py
"""

import copy
import json
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser, simpledialog
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Callable, Any

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("This app requires Pillow. Install with: pip install pillow")

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None

# Shared list of image patterns for file dialogs. Tk accepts space-separated glob patterns.
IMAGE_FILETYPES = [
    (
        "Images",
        "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff *.PNG *.JPG *.JPEG *.BMP *.GIF *.WEBP *.TIF *.TIFF",
    ),
    ("All Files", "*.*"),
]

JSON_FILETYPES = [("JSON files", "*.json"), ("All Files", "*.*")]

LAYOUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layouts")
os.makedirs(LAYOUTS_DIR, exist_ok=True)

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
BALL_IMAGES_DIR = os.path.join(IMAGES_DIR, "balls")
TABLE_IMAGES_DIR = os.path.join(IMAGES_DIR, "table")
os.makedirs(BALL_IMAGES_DIR, exist_ok=True)
os.makedirs(TABLE_IMAGES_DIR, exist_ok=True)


def list_ball_assets() -> List[Tuple[str, str]]:
    assets: List[Tuple[str, str]] = []
    if os.path.isdir(BALL_IMAGES_DIR):
        for entry in sorted(os.listdir(BALL_IMAGES_DIR)):
            if not entry.lower().endswith(".png"):
                continue
            name = os.path.splitext(entry)[0]
            path = os.path.join(BALL_IMAGES_DIR, entry)
            assets.append((name, path))
    return assets
TOURNAMENTS_DIR = os.path.join(LAYOUTS_DIR, "Tournaments")
os.makedirs(TOURNAMENTS_DIR, exist_ok=True)


def convert_legacy_tournaments():
    """Convert folder-based tournaments into single JSON files per tournament."""
    try:
        entries = sorted(os.listdir(TOURNAMENTS_DIR))
    except FileNotFoundError:
        return

    for entry in entries:
        entry_path = os.path.join(TOURNAMENTS_DIR, entry)
        if entry.lower().endswith(".json"):
            continue
        if not os.path.isdir(entry_path):
            continue
        if entry.lower().endswith("_legacy") or "_legacy" in entry.lower():
            continue

        matches_dir = os.path.join(entry_path, "MATCHES")
        if not os.path.isdir(matches_dir):
            continue

        target_json = os.path.join(TOURNAMENTS_DIR, f"{entry}.json")
        if os.path.exists(target_json):
            # Assume already converted
            continue

        tournament_name = entry.replace("_", " ")
        matches_data = []

        for match_entry in sorted(os.listdir(matches_dir)):
            match_path = os.path.join(matches_dir, match_entry)
            if not os.path.isdir(match_path):
                continue

            player_one, player_two = infer_players_from_folder_name(match_entry)
            match_shots = []

            for shot_entry in sorted(os.listdir(match_path)):
                if not shot_entry.lower().endswith(".json"):
                    continue
                shot_path = os.path.join(match_path, shot_entry)
                if not os.path.isfile(shot_path):
                    continue
                try:
                    with open(shot_path, "r", encoding="utf-8") as handle:
                        shot_data = json.load(handle)
                except Exception:
                    continue

                shot_name = os.path.splitext(shot_entry)[0].replace("_", " ")
                match_shots.append(
                    {
                        "name": shot_name,
                        "player": "playerOne",
                        "data": shot_data,
                    }
                )

            matches_data.append(
                {
                    "name": match_entry,
                    "playerOne": player_one,
                    "playerTwo": player_two,
                    "racks": [
                        {
                            "name": "Rack 1",
                            "break": player_one,
                            "shots": match_shots,
                        }
                    ],
                }
            )

        document = {"name": tournament_name, "matches": matches_data}
        try:
            with open(target_json, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
        except Exception:
            continue

        backup_path = entry_path + "_legacy"
        suffix = 1
        while os.path.exists(backup_path):
            suffix += 1
            backup_path = f"{entry_path}_legacy{suffix}"
        try:
            shutil.move(entry_path, backup_path)
        except Exception:
            pass


def infer_players_from_folder_name(folder_name: str) -> Tuple[str, str]:
    marker = "_VS_"
    upper = folder_name.upper()
    if marker in upper:
        idx = upper.find(marker)
        left = folder_name[:idx]
        right = folder_name[idx + len(marker) :]
    else:
        parts = folder_name.split("_")
        split = len(parts) // 2
        left = "_".join(parts[:split]) if split else folder_name
        right = "_".join(parts[split:]) if split else folder_name

    def normalize(segment: str, fallback: str) -> str:
        cleaned = " ".join(segment.replace("_", " ").split())
        cleaned = cleaned.strip()
        return cleaned or fallback

    player_one = normalize(left, "Player 1")
    player_two = normalize(right, "Player 2")
    return player_one, player_two

# ----------------------------- Data Models -----------------------------

@dataclass
class BallState:
    name: str
    x: Optional[float] = None  # legacy canvas pixel position (editor)
    y: Optional[float] = None
    visible: bool = True
    path: Optional[str] = None
    u: Optional[float] = None  # normalized table-relative coordinates
    v: Optional[float] = None

@dataclass
class DrawingState:
    kind: str          # "line" or "arrow"
    color: str         # e.g., "#ff0000"
    width: int         # stroke width in pixels (applied 1:1 on both displays)
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

# ----------------------------- Helpers -----------------------------

def normalize_match_structure(match: Dict[str, Any]) -> None:
    if not isinstance(match, dict):
        return

    racks = match.get("racks")
    if not isinstance(racks, list):
        legacy_shots = match.get("shots")
        if isinstance(legacy_shots, list) and legacy_shots:
            default_breaker = match.get("playerOne") or "Player 1"
            racks = [
                {
                    "name": "Rack 1",
                    "break": default_breaker,
                    "shots": [shot for shot in legacy_shots if isinstance(shot, dict)],
                }
            ]
        else:
            racks = []
        match["racks"] = racks
        match.pop("shots", None)
    else:
        racks = [rack for rack in racks if isinstance(rack, dict)]
        match["racks"] = racks

    for idx, rack in enumerate(match["racks"]):
        if not isinstance(rack, dict):
            match["racks"][idx] = {}
            rack = match["racks"][idx]

        rack_name = rack.get("name")
        if rack_name is not None and not isinstance(rack_name, str):
            rack.pop("name", None)

        breaker = rack.get("break")
        if not isinstance(breaker, str) or not breaker.strip():
            rack["break"] = match.get("playerOne") or "Player 1"

        shots = rack.get("shots")
        if not isinstance(shots, list):
            rack["shots"] = []
        else:
            rack["shots"] = [shot for shot in shots if isinstance(shot, dict)]


def normalize_tournament_document(document: Dict[str, Any]) -> None:
    if not isinstance(document, dict):
        return

    matches = document.get("matches")
    if not isinstance(matches, list):
        document["matches"] = []
        return

    cleaned_matches: List[Dict[str, Any]] = []
    for match in matches:
        if isinstance(match, dict):
            normalize_match_structure(match)
            cleaned_matches.append(match)
    document["matches"] = cleaned_matches


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


PLAYER_TWO_TOKENS = {
    "playertwo",
    "player_two",
    "player 2",
    "player two",
    "two",
    "p2",
    "player2",
}


def resolve_player_key(raw_value: Optional[str]) -> str:
    token = (raw_value or "").strip().lower()
    if token in PLAYER_TWO_TOKENS:
        return "playerTwo"
    return "playerOne"


def resolve_player_key_with_names(
    raw_value: Optional[str],
    player_one_name: Optional[str],
    player_two_name: Optional[str],
) -> str:
    token = (raw_value or "").strip().lower()
    if player_two_name and token == player_two_name.strip().lower():
        return "playerTwo"
    if player_one_name and token == player_one_name.strip().lower():
        return "playerOne"
    return resolve_player_key(raw_value)

# ----------------------------- View Models -----------------------------

class BallSprite:
    """
    A draggable, hideable ball image on a Tkinter Canvas.
    Holds the original PIL image, can re-render at any scale,
    and tracks normalized (u,v) relative to the current table rect.
    """
    def __init__(self, canvas: tk.Canvas, name: str, img_path: str, x: int, y: int, scale: float = 1.0):
        self.canvas = canvas
        self.name = name
        self.img_path = img_path
        self.visible = True

        # Load original image
        self.base_pil: Image.Image = Image.open(img_path).convert("RGBA")

        # Render state
        self.current_scale: float = 1.0
        self.tk_img: Optional[ImageTk.PhotoImage] = None

        # Logical coords (relative to table rect). Defaults to center.
        self.u: float = 0.5
        self.v: float = 0.5

        # Canvas item
        self.item_id = canvas.create_image(x, y, anchor="center", tags=("ball", self.name))
        self.apply_scale(scale)

    # ----- hit/pos -----
    def contains(self, x: int, y: int) -> bool:
        ix, iy = self.position()
        w, h = self.base_pil.size
        w = int(w * self.current_scale)
        h = int(h * self.current_scale)
        return (ix - w/2 <= x <= ix + w/2) and (iy - h/2 <= y <= iy + h/2)

    def position(self) -> Tuple[int, int]:
        coords = self.canvas.coords(self.item_id)
        if coords:
            return int(coords[0]), int(coords[1])
        return 0, 0

    def move_to(self, x: int, y: int):
        self.canvas.coords(self.item_id, x, y)

    # ----- visibility -----
    def show(self):
        if not self.visible:
            self.visible = True
            self.canvas.itemconfigure(self.item_id, state="normal")

    def hide(self):
        if self.visible:
            self.visible = False
            self.canvas.itemconfigure(self.item_id, state="hidden")

    # ----- scaling -----
    def apply_scale(self, scale: float):
        scale = clamp(scale, 0.05, 20.0)
        self.current_scale = scale
        bw, bh = self.base_pil.size
        new_size = (max(1, int(bw * scale)), max(1, int(bh * scale)))
        resized = self.base_pil.resize(new_size, Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.itemconfigure(self.item_id, image=self.tk_img)

    # ----- UV mapping helpers -----
    def set_uv_from_canvas_xy(self, x: int, y: int, table_rect: Tuple[int, int, int, int]):
        """Update (u,v) based on a canvas-pixel position and the given table rect (l,t,w,h)."""
        l, t, w, h = table_rect
        if w <= 0 or h <= 0:
            return
        self.u = (x - l) / w
        self.v = (y - t) / h

    def place_by_uv(self, table_rect: Tuple[int, int, int, int]):
        """Place the ball on canvas using current (u,v) and the given table rect (l,t,w,h)."""
        l, t, w, h = table_rect
        x = l + self.u * w
        y = t + self.v * h
        self.move_to(int(x), int(y))

    # ----- persistence -----
    def to_state(self, table_rect: Tuple[int, int, int, int]) -> BallState:
        x, y = self.position()
        l, t, w, h = table_rect
        if w > 0 and h > 0:
            try:
                self.u = (x - l) / w
                self.v = (y - t) / h
            except ZeroDivisionError:
                pass
        return BallState(name=self.name, x=x, y=y, visible=self.visible, path=self.img_path, u=self.u, v=self.v)

    def from_state(
        self,
        state: BallState,
        table_rect: Tuple[int, int, int, int],
        saved_table_rect: Optional[Tuple[float, float, float, float]] = None,
    ):
        # Prefer u,v if present; else fallback to legacy canvas coordinates
        if state.u is not None and state.v is not None:
            self.u = float(state.u)
            self.v = float(state.v)
            self.place_by_uv(table_rect)
        else:
            applied = False
            if (
                saved_table_rect is not None
                and len(saved_table_rect) == 4
                and state.x is not None
                and state.y is not None
            ):
                l, t, w, h = saved_table_rect
                if w and h:
                    try:
                        self.u = (float(state.x) - l) / w
                        self.v = (float(state.y) - t) / h
                        applied = True
                    except ZeroDivisionError:
                        applied = False
            if not applied and state.x is not None and state.y is not None:
                self.set_uv_from_canvas_xy(int(state.x), int(state.y), table_rect)
                applied = True
            if not applied:
                self.u = 0.5
                self.v = 0.5
            self.place_by_uv(table_rect)
        if state.visible:
            self.show()
        else:
            self.hide()

# ----------------------------- Drawings Model -----------------------------

class DrawingLayer:
    """
    Manages drawn objects (lines/arrows) on a canvas in table-relative coordinates.
    """
    def __init__(self, canvas: tk.Canvas, get_table_rect: Callable[[], Tuple[int,int,int,int]]):
        self.canvas = canvas
        self.get_table_rect = get_table_rect
        self.items: List[Tuple[int, DrawingState]] = []  # (canvas_item_id, state)
        self.preview_id: Optional[int] = None  # temporary during drag

    def clear(self):
        for item_id, _ in self.items:
            self.canvas.delete(item_id)
        self.items.clear()

    def undo(self):
        if not self.items:
            return
        item_id, _ = self.items.pop()
        self.canvas.delete(item_id)

    def add_from_canvas_points(self, kind: str, color: str, width: int, x1: int, y1: int, x2: int, y2: int):
        l, t, w, h = self.get_table_rect()
        if w <= 0 or h <= 0:
            return
        u1 = (x1 - l) / w
        v1 = (y1 - t) / h
        u2 = (x2 - l) / w
        v2 = (y2 - t) / h
        state = DrawingState(kind=kind, color=color, width=width, u1=u1, v1=v1, u2=u2, v2=v2)
        item_id = self._render_state(state)
        self.items.append((item_id, state))

    def _render_state(self, st: DrawingState) -> int:
        l, t, w, h = self.get_table_rect()
        x1 = l + st.u1 * w
        y1 = t + st.v1 * h
        x2 = l + st.u2 * w
        y2 = t + st.v2 * h
        arrowopt = tk.LAST if st.kind == "arrow" else tk.NONE
        return self.canvas.create_line(
            int(x1), int(y1), int(x2), int(y2),
            fill=st.color, width=st.width, arrow=arrowopt, capstyle=tk.ROUND, smooth=False
        )

    def rerender_all(self):
        # Redraw everything (e.g., on resize or table change)
        for item_id, _ in self.items:
            self.canvas.delete(item_id)
        new_items: List[Tuple[int, DrawingState]] = []
        for _, st in self.items:
            new_id = self._render_state(st)
            new_items.append((new_id, st))
        self.items = new_items

    # Preview helpers during drag
    def preview(self, kind: str, color: str, width: int, x1: int, y1: int, x2: int, y2: int):
        if self.preview_id is not None:
            self.canvas.delete(self.preview_id)
            self.preview_id = None
        arrowopt = tk.LAST if kind == "arrow" else tk.NONE
        self.preview_id = self.canvas.create_line(
            x1, y1, x2, y2, fill=color, width=width, arrow=arrowopt, capstyle=tk.ROUND
        )

    def clear_preview(self):
        if self.preview_id is not None:
            self.canvas.delete(self.preview_id)
            self.preview_id = None

    # Serialize/restore
    def serialize(self) -> List[Dict]:
        return [asdict(st) for _, st in self.items]

    def restore(self, data: List[Dict]):
        self.clear()
        for d in data:
            st = DrawingState(**d)
            item_id = self._render_state(st)
            self.items.append((item_id, st))

# ----------------------------- Editor Canvas -----------------------------

class PoolTableCanvas(tk.Frame):
    """
    Editor canvas: draws table and balls, handles drag, drawing tools,
    and maintains table rect. Notifies listeners (e.g., projector).
    """
    def __init__(self, master, width=1000, height=560, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, width=width, height=height, bg="#0a5d2a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.table_img_path: Optional[str] = None
        self._table_pil: Optional[Image.Image] = None
        self._table_tk: Optional[ImageTk.PhotoImage] = None
        self._table_item: Optional[int] = None

        # Current fitted table rectangle on this canvas (left, top, width, height)
        self._table_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)

        self.balls: List[BallSprite] = []
        self._drag_target: Optional[BallSprite] = None
        self._drag_dx = 0
        self._drag_dy = 0

        self.ball_scale: float = 1.0
        self.table_scale: float = 1.0
        self.table_offset_x: float = 0.0
        self.table_offset_y: float = 0.0
        self.webcam_enabled: bool = False
        self.webcam_source: int = 0
        self.webcam_opacity: float = 0.5
        self.webcam_margin: int = 20
        self._webcam_capture: Optional[object] = None
        self._webcam_job: Optional[str] = None
        self._webcam_item: Optional[int] = None
        self._webcam_photo: Optional[ImageTk.PhotoImage] = None
        self._listeners: List[Callable[[Dict], None]] = []

        # Drawing state
        self.draw_layer = DrawingLayer(self.canvas, self.get_table_rect)
        self.tool_mode: str = "select"  # "select", "line", "arrow"
        self.stroke_color: str = "#ff0000"
        self.stroke_width: int = 4
        self._draw_start: Optional[Tuple[int,int]] = None

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.on_resize)

    def get_table_rect(self) -> Tuple[int,int,int,int]:
        return self._table_rect

    # ---------- Listener mgmt ----------
    def add_listener(self, callback: Callable[[Dict], None]):
        if callback not in self._listeners:
            self._listeners.append(callback)

    def _notify(self):
        state = self.serialize()
        for cb in list(self._listeners):
            try:
                cb(state)
            except Exception:
                pass

    # ---------- Table handling ----------
    def load_table_image(self, path: str):
        try:
            pil = Image.open(path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open table image:\n{e}")
            return

        self.table_img_path = path
        self._table_pil = pil
        self._render_table_to_canvas()
        # Re-place all balls by current uv
        for b in self.balls:
            b.place_by_uv(self._table_rect)
        self.draw_layer.rerender_all()
        self._notify()

    def _render_table_to_canvas(self):
        # Compute fitted table rectangle and render
        cw = self.canvas.winfo_width() or int(self.canvas["width"])
        ch = self.canvas.winfo_height() or int(self.canvas["height"])
        self.canvas.delete("table")
        self._table_item = None
        self._table_rect = (0, 0, 0, 0)

        if not self._table_pil:
            return

        iw, ih = self._table_pil.size
        base_scale = min(cw / iw, ch / ih)
        scale = max(0.05, base_scale * self.table_scale)
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        pil_scaled = self._table_pil.resize((new_w, new_h), Image.LANCZOS)
        self._table_tk = ImageTk.PhotoImage(pil_scaled)

        # Centered with offsets
        cx = (cw / 2.0) + self.table_offset_x
        cy = (ch / 2.0) + self.table_offset_y
        left = int(round(cx - new_w / 2))
        top = int(round(cy - new_h / 2))
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        self._table_rect = (left, top, new_w, new_h)

        self._table_item = self.canvas.create_image(cx_int, cy_int, image=self._table_tk, anchor="center", tags=("table",))

        # Ensure layers/z-order: table at bottom, then drawings, then balls
        for item_id, _ in self.draw_layer.items:
            self.canvas.tag_raise(item_id)
        for b in self.balls:
            self.canvas.tag_raise(b.item_id)

    def on_resize(self, event):
        self._render_table_to_canvas()
        # Re-place balls and drawings
        for b in self.balls:
            b.place_by_uv(self._table_rect)
        self.draw_layer.rerender_all()
        self._position_webcam_item()
        self._notify()

    # ---------- Ball handling ----------
    def add_ball(self, name: str, img_path: str):
        cw = self.canvas.winfo_width() or int(self.canvas["width"])
        ch = self.canvas.winfo_height() or int(self.canvas["height"])
        cx, cy = cw // 2, ch // 2
        ball = BallSprite(self.canvas, name, img_path, cx, cy, self.ball_scale)
        if self._table_rect[2] > 0 and self._table_rect[3] > 0:
            ball.u, ball.v = 0.5, 0.5
            ball.place_by_uv(self._table_rect)
        self.balls.append(ball)
        self.canvas.tag_raise(ball.item_id)
        self._notify()



    def set_ball_scale(self, scale: float):
        self.ball_scale = scale
        for b in self.balls:
            b.apply_scale(scale)
            b.place_by_uv(self._table_rect)
        self._notify()

    def set_table_scale(self, scale: float):
        self.table_scale = clamp(scale, 0.2, 2.0)
        self._render_table_to_canvas()
        for b in self.balls:
            b.place_by_uv(self._table_rect)
        self.draw_layer.rerender_all()
        self._position_webcam_item()
        self._notify()

    def set_table_offset(self, offset_x: Optional[float] = None, offset_y: Optional[float] = None):
        """Adjust the table position relative to the canvas centre."""
        cw = self.canvas.winfo_width() or int(self.canvas["width"])
        ch = self.canvas.winfo_height() or int(self.canvas["height"])
        changed = False
        if offset_x is not None:
            try:
                target_x = float(offset_x)
            except (TypeError, ValueError):
                target_x = self.table_offset_x
            limit_x = max(cw, 1)
            new_x = clamp(target_x, -limit_x, limit_x)
            if abs(new_x - self.table_offset_x) > 1e-3:
                self.table_offset_x = new_x
                changed = True
        if offset_y is not None:
            try:
                target_y = float(offset_y)
            except (TypeError, ValueError):
                target_y = self.table_offset_y
            limit_y = max(ch, 1)
            new_y = clamp(target_y, -limit_y, limit_y)
            if abs(new_y - self.table_offset_y) > 1e-3:
                self.table_offset_y = new_y
                changed = True
        if not changed:
            return
        self._render_table_to_canvas()
        for b in self.balls:
            b.place_by_uv(self._table_rect)
        self.draw_layer.rerender_all()
        self._position_webcam_item()
        self._notify()

    def set_webcam_opacity(self, opacity: float):
        opacity = clamp(opacity, 0.0, 1.0)
        if abs(opacity - self.webcam_opacity) < 1e-4:
            return
        self.webcam_opacity = opacity
        self._position_webcam_item()
        self._notify()

    def start_webcam(self, source: int = 0) -> bool:
        if cv2 is None:
            messagebox.showerror(
                "Missing dependency",
                "Webcam support requires opencv-python. Install with: pip install opencv-python",
            )
            return False
        if self.webcam_enabled:
            return True
        cap = cv2.VideoCapture(source)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            messagebox.showerror("Webcam", f"Unable to open webcam source {source}.")
            return False
        self._webcam_capture = cap
        self.webcam_source = source
        self.webcam_enabled = True
        if self._webcam_item is None:
            self._webcam_item = self.canvas.create_image(0, 0, anchor="center", tags=("webcam",))
        self._schedule_webcam_frame()
        self._notify()
        return True

    def stop_webcam(self):
        if not self.webcam_enabled:
            return
        self.webcam_enabled = False
        if self._webcam_job:
            try:
                self.canvas.after_cancel(self._webcam_job)
            except Exception:
                pass
            self._webcam_job = None
        if self._webcam_capture is not None:
            try:
                self._webcam_capture.release()
            except Exception:
                pass
            self._webcam_capture = None
        if self._webcam_item is not None:
            self.canvas.delete(self._webcam_item)
            self._webcam_item = None
        self._webcam_photo = None
        self._notify()

    def _schedule_webcam_frame(self):
        if not self.webcam_enabled or self._webcam_capture is None:
            return
        if self._webcam_job:
            try:
                self.canvas.after_cancel(self._webcam_job)
            except Exception:
                pass
        self._webcam_job = self.canvas.after(33, self._update_webcam_frame)

    def _update_webcam_frame(self):
        if not self.webcam_enabled or self._webcam_capture is None:
            return
        ret, frame = self._webcam_capture.read()
        if not ret:
            messagebox.showerror("Webcam", "Lost connection to webcam. Stopping feed.")
            self.stop_webcam()
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        cw = max(1, self.canvas.winfo_width() or int(self.canvas["width"]))
        ch = max(1, self.canvas.winfo_height() or int(self.canvas["height"]))
        if pil_img.width <= 0 or pil_img.height <= 0:
            return
        scale = max(cw / pil_img.width, ch / pil_img.height)
        resized = pil_img.resize((max(1, int(pil_img.width * scale)), max(1, int(pil_img.height * scale))), Image.LANCZOS)
        left = max(0, (resized.width - cw) // 2)
        top = max(0, (resized.height - ch) // 2)
        cropped = resized.crop((left, top, left + cw, top + ch))
        rgba = cropped.convert("RGBA")
        alpha = int(255 * clamp(self.webcam_opacity, 0.0, 1.0))
        rgba.putalpha(alpha)
        self._webcam_photo = ImageTk.PhotoImage(rgba)
        if self._webcam_item is None:
            self._webcam_item = self.canvas.create_image(cw // 2, ch // 2, image=self._webcam_photo, anchor="center", tags=("webcam",))
        else:
            self.canvas.itemconfigure(self._webcam_item, image=self._webcam_photo)
            self.canvas.coords(self._webcam_item, cw // 2, ch // 2)
        self.canvas.tag_raise(self._webcam_item)
        self._schedule_webcam_frame()

    def _position_webcam_item(self):
        if not self.webcam_enabled or not self._webcam_item:
            return
        cw = max(1, self.canvas.winfo_width() or int(self.canvas["width"]))
        ch = max(1, self.canvas.winfo_height() or int(self.canvas["height"]))
        self.canvas.coords(self._webcam_item, (cw // 2), (ch // 2))
        self.canvas.tag_raise(self._webcam_item)

    def shutdown(self):
        self.stop_webcam()
    def _find_ball_at(self, x: int, y: int) -> Optional[BallSprite]:
        hits = self.canvas.find_overlapping(x, y, x, y)
        for item in reversed(hits):
            for b in self.balls:
                if b.item_id == item and b.visible and b.contains(x, y):
                    return b
        return None

    # ---------- Mouse handlers ----------
    def on_mouse_down(self, event):
        if self.tool_mode in ("line", "arrow"):
            self._draw_start = (event.x, event.y)
            return

        # Select/drag balls
        target = self._find_ball_at(event.x, event.y)
        if target:
            bx, by = target.position()
            self._drag_target = target
            self._drag_dx = event.x - bx
            self._drag_dy = event.y - by
            self.canvas.tag_raise(target.item_id)

    def on_mouse_drag(self, event):
        if self.tool_mode in ("line", "arrow"):
            if self._draw_start:
                x1, y1 = self._draw_start
                self.draw_layer.preview(self.tool_mode, self.stroke_color, self.stroke_width, x1, y1, event.x, event.y)
            return

        if self._drag_target:
            nx = event.x - self._drag_dx
            ny = event.y - self._drag_dy
            self._drag_target.move_to(nx, ny)
            self._drag_target.set_uv_from_canvas_xy(nx, ny, self._table_rect)
            self._notify()

    def on_mouse_up(self, event):
        if self.tool_mode in ("line", "arrow"):
            if self._draw_start:
                x1, y1 = self._draw_start
                x2, y2 = event.x, event.y
                self.draw_layer.clear_preview()
                self.draw_layer.add_from_canvas_points(self.tool_mode, self.stroke_color, self.stroke_width, x1, y1, x2, y2)
                self._draw_start = None
                self._notify()
            return

        self._drag_target = None

    # ---------- State (save/load) ----------
    def serialize(self) -> Dict:
        return {
            "table": self.table_img_path,
            "table_scale": self.table_scale,
            "ball_scale": self.ball_scale,
            "table_offset": {"x": self.table_offset_x, "y": self.table_offset_y},
            "table_rect": self._table_rect,
            "webcam": {"enabled": self.webcam_enabled, "opacity": self.webcam_opacity, "source": self.webcam_source},
            "balls": [asdict(b.to_state(self._table_rect)) for b in self.balls],
            "drawings": self.draw_layer.serialize(),
        }

    def restore(self, data: Dict):
        offset_state = data.get("table_offset", {})
        cw = self.canvas.winfo_width() or int(self.canvas["width"])
        ch = self.canvas.winfo_height() or int(self.canvas["height"])
        if isinstance(offset_state, dict):
            try:
                raw_x = float(offset_state.get("x", 0.0))
            except (TypeError, ValueError):
                raw_x = 0.0
            try:
                raw_y = float(offset_state.get("y", 0.0))
            except (TypeError, ValueError):
                raw_y = 0.0
            self.table_offset_x = clamp(raw_x, -max(cw, 1), max(cw, 1))
            self.table_offset_y = clamp(raw_y, -max(ch, 1), max(ch, 1))
        else:
            self.table_offset_x = 0.0
            self.table_offset_y = 0.0

        table = data.get("table")
        if table and os.path.exists(table):
            self.load_table_image(table)

        saved_table_scale = clamp(float(data.get("table_scale", 1.0)), 0.2, 2.0)
        self.table_scale = saved_table_scale

        webcam_state = data.get("webcam", {})
        if isinstance(webcam_state, dict):
            opacity_val = webcam_state.get("opacity")
            if opacity_val is None:
                opacity_val = webcam_state.get("scale")
            if opacity_val is not None:
                try:
                    self.webcam_opacity = clamp(float(opacity_val), 0.0, 1.0)
                except (TypeError, ValueError):
                    pass

        saved_scale = float(data.get("ball_scale", 1.0))
        self.ball_scale = saved_scale

        saved_table_rect: Optional[Tuple[float, float, float, float]] = None
        raw_saved_rect = data.get("table_rect")
        if isinstance(raw_saved_rect, (list, tuple)) and len(raw_saved_rect) == 4:
            try:
                saved_table_rect = tuple(float(v) for v in raw_saved_rect)
            except (TypeError, ValueError):
                saved_table_rect = None

        # Clear balls
        for b in self.balls:
            self.canvas.delete(b.item_id)
        self.balls.clear()

        self._render_table_to_canvas()

        for bstate in data.get("balls", []):
            path = bstate.get("path")
            name = bstate.get("name")
            if path and os.path.exists(path):
                cw = self.canvas.winfo_width() or int(self.canvas["width"])
                ch = self.canvas.winfo_height() or int(self.canvas["height"])
                cx, cy = cw // 2, ch // 2
                ball = BallSprite(self.canvas, name, path, cx, cy, self.ball_scale)
                ball.from_state(BallState(**bstate), self._table_rect, saved_table_rect)
                self.balls.append(ball)
                self.canvas.tag_raise(ball.item_id)

        # Restore drawings
        self.draw_layer.restore(data.get("drawings", []))
        self._notify()

# ----------------------------- Projector Window -----------------------------

class ProjectorWindow:
    """
    A borderless window on a chosen display that mirrors the table+balls+drawings.
    Uses (u,v) mapping to place consistently with the editor.
    """
    def __init__(self, parent: tk.Tk, monitor_index: int, on_close: Optional[Callable[[], None]] = None):
        if get_monitors is None:
            raise RuntimeError("screeninfo is not installed. Run: pip install screeninfo")

        self.parent = parent
        self.monitor_index = monitor_index
        self.on_close = on_close
        self._closed = False
        self.top = tk.Toplevel(parent)
        self.top.attributes("-topmost", True)
        self.top.overrideredirect(True)  # borderless
        self.top.protocol("WM_DELETE_WINDOW", self.close)

        monitors = get_monitors()
        if monitor_index < 0 or monitor_index >= len(monitors):
            raise ValueError("Invalid monitor index")
        mon = monitors[monitor_index]
        x, y, w, h = mon.x, mon.y, mon.width, mon.height
        self.top.geometry(f"{w}x{h}+{x}+{y}")
        self.top.lift()
        try:
            self.top.focus_force()
        except tk.TclError:
            pass

        self.canvas = tk.Canvas(self.top, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Allow users to exit quickly if the wrong display is used.
        self.top.bind("<Escape>", self._handle_escape)
        self.top.bind("<KeyPress-Escape>", self._handle_escape)
        self.canvas.bind("<Escape>", self._handle_escape)
        self.canvas.bind("<KeyPress-Escape>", self._handle_escape)
        self.canvas.focus_set()
        self.top.after(50, self.canvas.focus_set)

        self._table_pil: Optional[Image.Image] = None
        self._table_tk: Optional[ImageTk.PhotoImage] = None
        self._table_item: Optional[int] = None
        self._table_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._editor_table_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)

        # Balls: name -> dict
        self.balls: Dict[str, Dict[str, object]] = {}
        # Drawings: list of (canvas_item_id, DrawingState)
        self.drawings: List[Tuple[int, DrawingState]] = []

        self.global_scale: float = 1.0
        self.table_scale: float = 1.0

        self.canvas.bind("<Configure>", self._on_resize)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.top and self.top.winfo_exists():
            self.top.destroy()
        if self.on_close:
            self.on_close()

    def _handle_escape(self, _event=None):
        self.close()

    def _on_resize(self, event):
        self._render_table()
        self._render_balls()
        self._render_drawings()

    def apply_state(self, state: Dict):
        # Table
        table_path = state.get("table")
        if table_path and os.path.exists(table_path):
            self._table_pil = Image.open(table_path).convert("RGBA")
        else:
            self._table_pil = None

        editor_rect = state.get("table_rect", (0, 0, 0, 0))
        if isinstance(editor_rect, (list, tuple)) and len(editor_rect) == 4:
            self._editor_table_rect = tuple(float(v) for v in editor_rect)
        else:
            self._editor_table_rect = (0, 0, 0, 0)

        # Balls
        desired = {b["name"] for b in state.get("balls", [])}
        for name in list(self.balls.keys()):
            if name not in desired:
                item = self.balls[name].get("item")
                if item:
                    self.canvas.delete(item)
                del self.balls[name]

        for b in state.get("balls", []):
            name = b["name"]
            path = b.get("path")
            if name not in self.balls:
                if path and os.path.exists(path):
                    self.balls[name] = {
                        "pil": Image.open(path).convert("RGBA"),
                        "tk": None,
                        "item": None,
                        "u": float(b.get("u", 0.5)),
                        "v": float(b.get("v", 0.5)),
                        "visible": bool(b.get("visible", True)),
                    }
            else:
                self.balls[name]["u"] = float(b.get("u", 0.5))
                self.balls[name]["v"] = float(b.get("v", 0.5))
                self.balls[name]["visible"] = bool(b.get("visible", True))

        # Drawings
        # Clear and recreate from state
        for item_id, _ in self.drawings:
            self.canvas.delete(item_id)
        self.drawings.clear()
        for d in state.get("drawings", []):
            st = DrawingState(**d)
            item_id = self._render_drawing(st)
            self.drawings.append((item_id, st))

        self.global_scale = float(state.get("ball_scale", 1.0))
        self.table_scale = clamp(float(state.get("table_scale", 1.0)), 0.2, 2.0)

        # Render
        self._render_table()
        self._render_balls()
        self._render_drawings()

    def _render_table(self):
        self.canvas.delete("table")
        self._table_rect = (0, 0, 0, 0)
        if not self._table_pil:
            return

        cw = self.canvas.winfo_width() or 1
        ch = self.canvas.winfo_height() or 1
        iw, ih = self._table_pil.size
        base_scale = min(cw / iw, ch / ih)
        scale = max(0.05, base_scale * self.table_scale)
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        pil_scaled = self._table_pil.resize((new_w, new_h), Image.LANCZOS)
        self._table_tk = ImageTk.PhotoImage(pil_scaled)

        cx, cy = cw // 2, ch // 2
        left = cx - new_w // 2
        top = cy - new_h // 2
        self._table_rect = (left, top, new_w, new_h)

        self._table_item = self.canvas.create_image(cx, cy, image=self._table_tk, anchor="center", tags=("table",))

    def _render_balls(self):
        tr_left, tr_top, tr_w, tr_h = self._table_rect
        editor_w = self._editor_table_rect[2] or 1
        editor_h = self._editor_table_rect[3] or 1
        if editor_w <= 0 or editor_h <= 0 or tr_w <= 0 or tr_h <= 0:
            scale_ratio = 1.0
        else:
            ratio_w = tr_w / editor_w
            ratio_h = tr_h / editor_h
            scale_ratio = min(ratio_w, ratio_h)
        for name, rec in self.balls.items():
            pil_img: Image.Image = rec["pil"]
            scale = self.global_scale * scale_ratio
            bw, bh = pil_img.size
            new_size = (max(1, int(bw * scale)), max(1, int(bh * scale)))
            resized = pil_img.resize(new_size, Image.LANCZOS)
            rec["tk"] = ImageTk.PhotoImage(resized)

            u, v = rec["u"], rec["v"]
            x = tr_left + u * tr_w
            y = tr_top + v * tr_h

            if rec["item"] is None:
                rec["item"] = self.canvas.create_image(int(x), int(y), image=rec["tk"], anchor="center", tags=("ball", name))
            else:
                self.canvas.itemconfigure(rec["item"], image=rec["tk"])
                self.canvas.coords(rec["item"], int(x), int(y))

            self.canvas.itemconfigure(rec["item"], state=("normal" if rec["visible"] else "hidden"))

        for name, rec in self.balls.items():
            if rec["item"]:
                self.canvas.tag_raise(rec["item"])

    def _render_drawing(self, st: DrawingState) -> int:
        l, t, w, h = self._table_rect
        x1 = l + st.u1 * w
        y1 = t + st.v1 * h
        x2 = l + st.u2 * w
        y2 = t + st.v2 * h
        arrowopt = tk.LAST if st.kind == "arrow" else tk.NONE
        return self.canvas.create_line(
            int(x1), int(y1), int(x2), int(y2),
            fill=st.color, width=st.width, arrow=arrowopt, capstyle=tk.ROUND, smooth=False
        )

    def _render_drawings(self):
        # Clear and redraw from stored states
        for item_id, _ in self.drawings:
            self.canvas.delete(item_id)
        new_items: List[Tuple[int, DrawingState]] = []
        for _, st in self.drawings:
            new_id = self._render_drawing(st)
            new_items.append((new_id, st))
        self.drawings = new_items

# ----------------------------- Sidebar -----------------------------

class Sidebar(tk.Frame):
    """
    Right pane: image loading, visibility toggles, ball size slider,
    projector display selection, and drawing tools.
    """
    def __init__(self, master, table_canvas: PoolTableCanvas, **kwargs):
        super().__init__(master, **kwargs)
        self.table_canvas = table_canvas
        self.ball_vars: Dict[str, tk.BooleanVar] = {}
        self.projector: Optional[ProjectorWindow] = None

        # Scrollable content container
        self._scroll_canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        self._vscroll = ttk.Scrollbar(self, orient="vertical", command=self._scroll_canvas.yview)
        self._vscroll.pack(side="right", fill="y")
        self._scroll_canvas.configure(yscrollcommand=self._vscroll.set)

        self.content = tk.Frame(self._scroll_canvas)
        self._content_window = self._scroll_canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_scroll_content_configure)
        self._scroll_canvas.bind("<Configure>", self._on_scroll_canvas_configure)

        body = self.content

        # File controls
        self.btn_load_table = tk.Button(body, text="Load Table Image...", command=self.load_table_dialog)
        self.btn_save_layout = tk.Button(body, text="Save Layout...", command=self.save_layout_dialog)
        self.btn_load_layout = tk.Button(body, text="Load Layout...", command=self.load_layout_dialog)

        self.btn_load_table.pack(fill="x", pady=(6, 3), padx=6)

        tk.Label(body, text="Add Ball", anchor="w", font=("Segoe UI", 10, "bold")).pack(fill="x", padx=8, pady=(2, 0))
        self.ball_choice_var = tk.StringVar()
        self.ball_catalog_combo = ttk.Combobox(
            body,
            state="readonly",
            textvariable=self.ball_choice_var,
            values=[],
        )
        self.ball_catalog_combo.pack(fill="x", padx=10, pady=(0, 4))

        self.btn_add_ball = tk.Button(body, text="Add Selected Ball", command=self.add_selected_ball)
        self.btn_add_ball.pack(fill="x", padx=10, pady=(0, 6))

        # Table size slider
        self.table_size_label = tk.Label(body, text="Table Size (%)", anchor="w", font=("Segoe UI", 10, "bold"))
        self.table_size_label.pack(fill="x", padx=8, pady=(10, 0))

        self.table_size_var = tk.DoubleVar(value=self.table_canvas.table_scale * 100.0)
        self.table_size_slider = tk.Scale(body, from_=20, to=200, orient="horizontal",
                                          variable=self.table_size_var, command=self.on_table_size_change)
        self.table_size_slider.pack(fill="x", padx=10, pady=(4, 6))

        # Table offset sliders
        canvas_widget = self.table_canvas.canvas
        cw = canvas_widget.winfo_width() or int(canvas_widget["width"])
        ch = canvas_widget.winfo_height() or int(canvas_widget["height"])
        offset_limit_x = max(400, cw)
        offset_limit_y = max(400, ch)

        tk.Label(body, text="Table Offset (px)", anchor="w", font=("Segoe UI", 10, "bold")).pack(fill="x", padx=8, pady=(0, 0))
        self.table_offset_x_var = tk.DoubleVar(value=self.table_canvas.table_offset_x)
        self.table_offset_x_slider = tk.Scale(
            body, from_=-offset_limit_x, to=offset_limit_x, orient="horizontal", resolution=1,
            variable=self.table_offset_x_var, command=self.on_table_offset_x_change, label="Horizontal (X)"
        )
        self.table_offset_x_slider.pack(fill="x", padx=10, pady=(2, 2))

        self.table_offset_y_var = tk.DoubleVar(value=self.table_canvas.table_offset_y)
        self.table_offset_y_slider = tk.Scale(
            body, from_=-offset_limit_y, to=offset_limit_y, orient="horizontal", resolution=1,
            variable=self.table_offset_y_var, command=self.on_table_offset_y_change, label="Vertical (Y)"
        )
        self.table_offset_y_slider.pack(fill="x", padx=10, pady=(0, 6))

        # Ball size slider
        self.size_label = tk.Label(body, text="Ball Size (%)", anchor="w", font=("Segoe UI", 10, "bold"))
        self.size_label.pack(fill="x", padx=8, pady=(4, 0))

        self.ball_size_var = tk.DoubleVar(value=self.table_canvas.ball_scale * 100.0)
        self.size_slider = tk.Scale(body, from_=20, to=200, orient="horizontal",
                                    variable=self.ball_size_var, command=self.on_size_change)
        self.size_slider.pack(fill="x", padx=10, pady=(4, 6))

        ttk.Separator(body, orient="horizontal").pack(fill="x", padx=8, pady=6)
        tk.Label(body, text="Webcam", anchor="w", font=("Segoe UI", 10, "bold")).pack(fill="x", padx=8)

        self.webcam_toggle_btn = tk.Button(body, text="Start Webcam", command=self.on_webcam_toggle)
        self.webcam_toggle_btn.pack(fill="x", padx=10, pady=(2, 2))

        self.webcam_opacity_var = tk.DoubleVar(value=self.table_canvas.webcam_opacity * 100.0)
        self.webcam_opacity_slider = tk.Scale(body, from_=0, to=100, orient="horizontal", resolution=1,
                                              variable=self.webcam_opacity_var, command=self.on_webcam_opacity_change)
        self.webcam_opacity_slider.pack(fill="x", padx=10, pady=(0, 10))

        # Drawing tools
        ttk.Separator(body, orient="horizontal").pack(fill="x", padx=8, pady=6)
        tk.Label(body, text="Drawing Tools", anchor="w", font=("Segoe UI", 10, "bold")).pack(fill="x", padx=8)

        tool_frame = tk.Frame(body)
        tool_frame.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(tool_frame, text="Tool:").pack(side="left")
        self.tool_choice = tk.StringVar(value="select")
        ttk.Combobox(tool_frame, state="readonly", textvariable=self.tool_choice,
                     values=["select", "line", "arrow"], width=10).pack(side="left", padx=6)

        color_frame = tk.Frame(body)
        color_frame.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(color_frame, text="Color:").pack(side="left")
        self.color_preview = tk.Label(color_frame, text="   ", bg="#ff0000", relief="sunken", width=3)
        self.color_preview.pack(side="left", padx=6)
        tk.Button(color_frame, text="Pick...", command=self.pick_color).pack(side="left")

        width_frame = tk.Frame(body)
        width_frame.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(width_frame, text="Width:").pack(side="left")
        self.width_var = tk.IntVar(value=4)
        tk.Scale(width_frame, from_=1, to=20, orient="horizontal", variable=self.width_var, length=140).pack(side="left", padx=6)

        draw_buttons = tk.Frame(body)
        draw_buttons.pack(fill="x", padx=8, pady=(4, 8))
        tk.Button(draw_buttons, text="Undo", command=self.undo_drawing).pack(side="left")
        tk.Button(draw_buttons, text="Clear", command=self.clear_drawings).pack(side="left", padx=6)

        # Projector display selection
        ttk.Separator(body, orient="horizontal").pack(fill="x", padx=8, pady=6)
        self.display_label = tk.Label(body, text="Projector Display", anchor="w", font=("Segoe UI", 10, "bold"))
        self.display_label.pack(fill="x", padx=8, pady=(2, 0))

        self.monitor_list = self._get_monitors_summary()
        default_choice = self.monitor_list[0] if self.monitor_list else "No displays found"
        self.display_choice = tk.StringVar(value=default_choice)
        self.display_dropdown = ttk.Combobox(body, state="readonly", textvariable=self.display_choice, values=self.monitor_list)
        self.display_dropdown.pack(fill="x", padx=10, pady=(4, 6))

        self.open_proj_btn = tk.Button(body, text="Open Projector Window", command=self.open_projector)
        self.close_proj_btn = tk.Button(body, text="Close Projector Window", command=self.close_projector)
        self.open_proj_btn.pack(fill="x", padx=10, pady=(2, 2))
        self.close_proj_btn.pack(fill="x", padx=10, pady=(2, 10))

        self.btn_save_layout.pack(fill="x", pady=(6, 3), padx=6)
        self.btn_load_layout.pack(fill="x", pady=3, padx=6)

        # Balls list
        self.sep = tk.Label(body, text="Balls", anchor="w", font=("Segoe UI", 10, "bold"))
        self.sep.pack(fill="x", padx=8, pady=(12, 0))

        self.ball_frame = tk.Frame(body, bd=1, relief="sunken")
        self.ball_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # Hook editor for mirroring
        self.table_canvas.add_listener(self._on_canvas_update)

        # Global mouse-wheel bindings restricted in handler
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Button-5>", self._on_mousewheel)

        self._update_webcam_controls()

        self._update_ball_catalog()

        # Initialize tool bindings
        self.tool_choice.trace_add("write", self.on_tool_change)
        self.on_tool_change()

        self.refresh_ball_list()
        self._schedule_scroll_update()

    def _widget_within_sidebar(self, widget) -> bool:
        while widget is not None:
            if widget is self:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_mousewheel(self, event):
        widget = None
        if hasattr(event, "x_root") and hasattr(event, "y_root"):
            try:
                widget = self.winfo_containing(int(event.x_root), int(event.y_root))
            except tk.TclError:
                widget = None
        if widget is None or not self._widget_within_sidebar(widget):
            return
        if getattr(event, "delta", 0):
            self._scroll_canvas.yview_scroll(int(-event.delta / 120), "units")
        else:
            num = getattr(event, "num", None)
            if num == 4:
                self._scroll_canvas.yview_scroll(-3, "units")
            elif num == 5:
                self._scroll_canvas.yview_scroll(3, "units")

    def _on_scroll_content_configure(self, _event=None):
        bbox = self._scroll_canvas.bbox("all")
        if bbox is not None:
            self._scroll_canvas.configure(scrollregion=bbox)

    def _on_scroll_canvas_configure(self, event):
        self._scroll_canvas.itemconfigure(self._content_window, width=event.width)

    def _schedule_scroll_update(self):
        if not hasattr(self, "_scroll_canvas"):
            return
        self.after_idle(lambda: self._on_scroll_content_configure())

    def _update_webcam_controls(self):
        btn_text = "Stop Webcam" if self.table_canvas.webcam_enabled else "Start Webcam"
        self.webcam_toggle_btn.configure(text=btn_text)

    def on_webcam_toggle(self):
        if self.table_canvas.webcam_enabled:
            self.table_canvas.stop_webcam()
        else:
            started = self.table_canvas.start_webcam()
            if not started:
                return
        self._update_webcam_controls()

    def on_webcam_opacity_change(self, _value):
        try:
            opacity = float(self.webcam_opacity_var.get()) / 100.0
        except (TypeError, tk.TclError):
            return
        self.table_canvas.set_webcam_opacity(opacity)

    def on_table_size_change(self, _value):
        scale = self.table_size_var.get() / 100.0
        self.table_canvas.set_table_scale(scale)

    def on_table_offset_x_change(self, value):
        try:
            offset = float(value)
        except (TypeError, ValueError):
            return
        self.table_canvas.set_table_offset(offset_x=offset)

    def on_table_offset_y_change(self, value):
        try:
            offset = float(value)
        except (TypeError, ValueError):
            return
        self.table_canvas.set_table_offset(offset_y=offset)

    def on_tool_change(self, *args):
        mode = self.tool_choice.get()
        self.table_canvas.tool_mode = mode
        # Update editor drawing params too
        self.table_canvas.stroke_color = self.color_preview["bg"]
        self.table_canvas.stroke_width = int(self.width_var.get())

    def pick_color(self):
        initial = self.color_preview["bg"]
        color = colorchooser.askcolor(initialcolor=initial)[1]
        if color:
            self.color_preview.config(bg=color)
            self.table_canvas.stroke_color = color

    def undo_drawing(self):
        self.table_canvas.draw_layer.undo()
        self.table_canvas._notify()

    def clear_drawings(self):
        if messagebox.askyesno("Clear drawings", "Remove all drawings?"):
            self.table_canvas.draw_layer.clear()
            self.table_canvas._notify()

    def _get_monitors_summary(self) -> List[str]:
        res = []
        try:
            if get_monitors is None:
                return ["Install 'screeninfo' to enumerate displays (pip install screeninfo)"]
            mons = get_monitors()
            for idx, m in enumerate(mons):
                res.append(f"{idx}: {m.width}x{m.height} @ ({m.x},{m.y})")
        except Exception as e:
            res = [f"Error: {e}"]
        return res

    def on_size_change(self, _value):
        scale = self.ball_size_var.get() / 100.0
        self.table_canvas.set_ball_scale(scale)

    def refresh_ball_list(self):
        self._update_ball_catalog()
        for w in self.ball_frame.winfo_children():
            w.destroy()
        self.ball_vars.clear()

        for ball in self.table_canvas.balls:
            var = tk.BooleanVar(value=ball.visible)
            self.ball_vars[ball.name] = var
            cb = tk.Checkbutton(self.ball_frame, text=ball.name, variable=var,
                                command=lambda b=ball, v=var: self.toggle_ball(b, v))
            cb.pack(anchor="w", padx=6, pady=2)

        self._schedule_scroll_update()

    def toggle_ball(self, ball: BallSprite, var: tk.BooleanVar):
        if var.get():
            ball.show()
        else:
            ball.hide()
        self.table_canvas._notify()

    def load_table_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Pool Table Image",
            filetypes=IMAGE_FILETYPES,
            initialdir=IMAGES_DIR if os.path.isdir(IMAGES_DIR) else None,
        )
        if path:
            self.table_canvas.load_table_image(path)

    def load_balls_dialog(self):
        self.prompt_add_ball_from_catalog()

    def _update_ball_catalog(self):
        catalog = list_ball_assets()
        self.ball_catalog = catalog
        names = [name for name, _ in catalog]
        self.ball_catalog_combo.configure(values=names)
        if names:
            if self.ball_choice_var.get() not in names:
                self.ball_choice_var.set(names[0])
            self.btn_add_ball.configure(state=tk.NORMAL)
        else:
            self.ball_choice_var.set("")
            self.btn_add_ball.configure(state=tk.DISABLED)

    def add_selected_ball(self):
        catalog = getattr(self, "ball_catalog", [])
        if not catalog:
            messagebox.showerror("Add Ball", "No ball images available in the catalog.")
            return

        selected = self.ball_choice_var.get() or catalog[0][0]
        path = None
        for name, p in catalog:
            if name == selected:
                path = p
                display_name = name
                break
        else:
            path = catalog[0][1]
            display_name = catalog[0][0]

        existing_names = {b.name for b in self.table_canvas.balls}
        candidate = display_name
        index = 2
        while candidate in existing_names:
            candidate = f"{display_name}_{index}"
            index += 1

        try:
            self.table_canvas.add_ball(candidate, path)
        except Exception as exc:
            messagebox.showerror("Add Ball", f"Failed to add ball:\n{exc}")
            return

        self.refresh_ball_list()
        self.table_canvas._notify()

    def prompt_add_ball_from_catalog(self):
        self._update_ball_catalog()
        catalog = getattr(self, "ball_catalog", [])
        if not catalog:
            messagebox.showerror("Add Ball", "No ball images available in the catalog.")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Add Ball")
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(dialog, text="Select ball:").pack(padx=12, pady=(12, 6))

        local_var = tk.StringVar(value=self.ball_choice_var.get() or catalog[0][0])
        combo = ttk.Combobox(dialog, state="readonly", textvariable=local_var, values=[name for name, _ in catalog], width=18)
        combo.pack(padx=12, pady=(0, 10))
        combo.focus_set()

        def confirm():
            self.ball_choice_var.set(local_var.get())
            dialog.destroy()
            self.add_selected_ball()

        tk.Button(dialog, text="Add", command=confirm).pack(padx=12, pady=(0, 10))
        dialog.bind("<Return>", lambda _e: confirm())
        dialog.bind("<Escape>", lambda _e: dialog.destroy())

    def save_layout_dialog(self):
        data = self.table_canvas.serialize()
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=JSON_FILETYPES,
            title="Save Layout",
            initialdir=LAYOUTS_DIR,
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save layout:\n{e}")
        else:
            messagebox.showinfo("Saved", f"Layout saved to:\n{path}")

    def load_layout_dialog(self):
        path = filedialog.askopenfilename(
            title="Load Layout JSON",
            filetypes=JSON_FILETYPES,
            initialdir=LAYOUTS_DIR,
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.table_canvas.restore(data)
            self.table_size_var.set(self.table_canvas.table_scale * 100.0)
            self.table_offset_x_var.set(self.table_canvas.table_offset_x)
            self.table_offset_y_var.set(self.table_canvas.table_offset_y)
            self.webcam_opacity_var.set(self.table_canvas.webcam_opacity * 100.0)
            self.ball_size_var.set(self.table_canvas.ball_scale * 100.0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load layout:\n{e}")
        else:
            self.refresh_ball_list()

    def open_projector(self):
        selected = self.display_choice.get()
        if ":" not in selected:
            messagebox.showerror("No display", "No display selected or 'screeninfo' not installed.")
            return
        if get_monitors is None:
            messagebox.showerror("Missing dependency", "Install 'screeninfo' first:\n\npip install screeninfo")
            return
        idx_str = selected.split(":")[0].strip()
        try:
            idx = int(idx_str)
        except ValueError:
            messagebox.showerror("Invalid selection", "Could not parse selected display index.")
            return

        self.close_projector()

        try:
            self.projector = ProjectorWindow(self.master, idx, on_close=self._on_projector_closed)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open projector window:\n{e}")
            self.projector = None
            return

        self.projector.apply_state(self.table_canvas.serialize())

    def close_projector(self):
        if self.projector:
            self.projector.close()
            self.projector = None

    def _on_projector_closed(self):
        self.projector = None

    def _on_canvas_update(self, state: Dict):
        if self.projector:
            self.projector.apply_state(state)

        if isinstance(state, dict):
            table_percent = float(state.get("table_scale", self.table_canvas.table_scale)) * 100.0
            if abs(self.table_size_var.get() - table_percent) > 0.5:
                self.table_size_var.set(table_percent)

            ball_percent = float(state.get("ball_scale", self.table_canvas.ball_scale)) * 100.0
            if abs(self.ball_size_var.get() - ball_percent) > 0.5:
                self.ball_size_var.set(ball_percent)

            canvas_widget = self.table_canvas.canvas
            cw = canvas_widget.winfo_width() or int(canvas_widget["width"])
            ch = canvas_widget.winfo_height() or int(canvas_widget["height"])
            offset_limit_x = max(400, cw)
            offset_limit_y = max(400, ch)
            self.table_offset_x_slider.configure(from_=-offset_limit_x, to=offset_limit_x)
            self.table_offset_y_slider.configure(from_=-offset_limit_y, to=offset_limit_y)

            offset_state = state.get("table_offset", {})
            if isinstance(offset_state, dict):
                try:
                    desired_x = float(offset_state.get("x", self.table_canvas.table_offset_x))
                except (TypeError, ValueError):
                    desired_x = self.table_canvas.table_offset_x
                try:
                    desired_y = float(offset_state.get("y", self.table_canvas.table_offset_y))
                except (TypeError, ValueError):
                    desired_y = self.table_canvas.table_offset_y
                if abs(self.table_offset_x_var.get() - desired_x) > 0.5:
                    self.table_offset_x_var.set(desired_x)
                if abs(self.table_offset_y_var.get() - desired_y) > 0.5:
                    self.table_offset_y_var.set(desired_y)

            webcam_state = state.get("webcam", {})
            if isinstance(webcam_state, dict):
                opacity_val = webcam_state.get("opacity")
                if opacity_val is None:
                    opacity_val = webcam_state.get("scale")
                if opacity_val is not None:
                    desired = float(opacity_val) * 100.0
                    if abs(self.webcam_opacity_var.get() - desired) > 0.5:
                        self.webcam_opacity_var.set(desired)
                if "enabled" in webcam_state:
                    self._update_webcam_controls()

        self._schedule_scroll_update()


class TournamentBrowser(tk.Frame):
    """Left pane: browse tournaments, matches, and load shots."""

    def __init__(
        self,
        master,
        on_select_shot: Callable[[ShotReference], None],
        on_request_refresh: Optional[Callable[[], None]] = None,
        on_add_rack: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.on_select_shot = on_select_shot
        self.on_request_refresh = on_request_refresh
        self.on_add_rack = on_add_rack
        self._item_refs: Dict[str, ShotReference] = {}
        self._item_meta: Dict[str, Dict[str, Any]] = {}
        self._current_data: List[Dict] = []

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(container, show="tree", selectmode="browse")
        self.tree.heading("#0", text="Tournaments", anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        self.add_rack_btn = tk.Button(self, text="Add Rack", command=self._handle_add_rack)
        if not callable(self.on_add_rack):
            self.add_rack_btn.configure(state=tk.DISABLED)
        self.add_rack_btn.pack(fill="x", padx=6, pady=(6, 0))

        self.refresh_btn = tk.Button(self, text="Refresh List", command=self._handle_refresh_request)
        self.refresh_btn.pack(fill="x", padx=6, pady=6)

        self._set_add_rack_enabled(False)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def refresh(self, tournaments: List[Dict], selected: Optional[ShotReference] = None):
        self._current_data = tournaments
        self.tree.delete(*self.tree.get_children())
        self._item_refs.clear()
        self._item_meta.clear()
        self._set_add_rack_enabled(False)

        if not tournaments:
            placeholder = self.tree.insert("", "end", text="No tournaments found", open=False)
            self.tree.item(placeholder, tags=("placeholder",))
            self._item_meta[placeholder] = {"type": "placeholder"}
            return

        selected_item_id: Optional[str] = None

        for tournament in tournaments:
            t_name = tournament.get("name", "Tournament")
            matches = tournament.get("matches", [])
            tournament_path = tournament.get("path", "")
            tournament_id = self.tree.insert("", "end", text=t_name, open=True)
            self._item_meta[tournament_id] = {
                "type": "tournament",
                "tournament_path": tournament_path,
            }
            if not matches:
                empty_id = self.tree.insert(tournament_id, "end", text="(No matches)", open=False)
                self.tree.item(empty_id, tags=("placeholder",))
                self._item_meta[empty_id] = {"type": "placeholder"}
                continue

            for match_index, match in enumerate(matches):
                match_label = match.get("name", "Match")
                player_one = match.get("player_one", "Player 1")
                player_two = match.get("player_two", "Player 2")
                match_id = self.tree.insert(
                    tournament_id,
                    "end",
                    text=f"{match_label} — {player_one} vs {player_two}",
                    open=True,
                )
                self._item_meta[match_id] = {
                    "type": "match",
                    "tournament_path": tournament_path,
                    "match_index": match_index,
                    "match_name": match_label,
                    "player_one": player_one,
                    "player_two": player_two,
                }

                racks = match.get("racks", [])
                if not racks:
                    empty_match = self.tree.insert(match_id, "end", text="(No racks)", open=False)
                    self.tree.item(empty_match, tags=("placeholder",))
                    self._item_meta[empty_match] = {"type": "placeholder"}
                    continue
                for rack_index, rack in enumerate(racks):
                    rack_label = rack.get("label") or f"Rack {rack_index + 1}"
                    rack_break = rack.get("break")
                    if rack_break:
                        rack_text = f"{rack_label} — Break: {rack_break}"
                    else:
                        rack_text = rack_label
                    rack_id = self.tree.insert(match_id, "end", text=rack_text, open=True)
                    self._item_meta[rack_id] = {
                        "type": "rack",
                        "tournament_path": tournament_path,
                        "match_index": match_index,
                        "rack_index": rack_index,
                    }

                    shots = rack.get("shots", [])
                    if not shots:
                        empty_rack = self.tree.insert(rack_id, "end", text="(No shots)", open=False)
                        self.tree.item(empty_rack, tags=("placeholder",))
                        self._item_meta[empty_rack] = {"type": "placeholder"}
                        continue
                    for shot in shots:
                        label = shot.get("label", "Shot")
                        shot_id = self.tree.insert(rack_id, "end", text=label, open=False)
                        ref = shot.get("reference")
                        if isinstance(ref, ShotReference):
                            self._item_refs[shot_id] = ref
                            self._item_meta[shot_id] = {"type": "shot", "reference": ref}
                            if selected is not None and ref == selected:
                                selected_item_id = shot_id
                        else:
                            self._item_meta[shot_id] = {"type": "placeholder"}

        if selected_item_id is not None:
            try:
                self.tree.selection_set(selected_item_id)
                self.tree.focus(selected_item_id)
                self.tree.see(selected_item_id)
                self._set_add_rack_enabled(True)
            except tk.TclError:
                pass

    def _handle_refresh_request(self):
        if callable(self.on_request_refresh):
            self.on_request_refresh()

    def _on_tree_select(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            self._set_add_rack_enabled(False)
            return
        item_id = selection[0]
        match_context = self._resolve_item_to_match(item_id)
        self._set_add_rack_enabled(match_context is not None)

        meta = self._item_meta.get(item_id)
        if not meta or meta.get("type") != "shot":
            return
        reference = meta.get("reference")
        if isinstance(reference, ShotReference):
            self.on_select_shot(reference)

    def _resolve_item_to_match(self, item_id: str) -> Optional[Tuple[str, int]]:
        meta = self._item_meta.get(item_id)
        if not meta:
            return None
        item_type = meta.get("type")
        if item_type == "match":
            path = meta.get("tournament_path")
            match_index = meta.get("match_index")
            if isinstance(path, str) and isinstance(match_index, int):
                return path, match_index
        if item_type == "rack":
            path = meta.get("tournament_path")
            match_index = meta.get("match_index")
            if isinstance(path, str) and isinstance(match_index, int):
                return path, match_index
        if item_type == "shot":
            reference = meta.get("reference")
            if isinstance(reference, ShotReference):
                return reference.tournament_path, reference.match_index
        return None

    def _handle_add_rack(self):
        if not callable(self.on_add_rack):
            return
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Add Rack", "Select a match, rack, or shot to add a rack.")
            return
        context = self._resolve_item_to_match(selection[0])
        if context is None:
            messagebox.showinfo("Add Rack", "Select a match, rack, or shot to add a rack.")
            return
        tournament_path, match_index = context
        self.on_add_rack({"tournament_path": tournament_path, "match_index": match_index})

    def _set_add_rack_enabled(self, enabled: bool):
        if not hasattr(self, "add_rack_btn") or self.add_rack_btn is None:
            return
        if not callable(self.on_add_rack):
            self.add_rack_btn.configure(state=tk.DISABLED)
            return
        new_state = tk.NORMAL if enabled else tk.DISABLED
        if str(self.add_rack_btn["state"]) != str(new_state):
            self.add_rack_btn.configure(state=new_state)

    def focus_match(self, tournament_path: str, match_index: int):
        for item_id, meta in self._item_meta.items():
            if (
                meta.get("type") == "match"
                and meta.get("tournament_path") == tournament_path
                and int(meta.get("match_index", -1)) == match_index
            ):
                try:
                    self.tree.item(item_id, open=True)
                    self.tree.selection_set(item_id)
                    self.tree.focus(item_id)
                    self.tree.see(item_id)
                    self._set_add_rack_enabled(True)
                except tk.TclError:
                    pass
                break

# ----------------------------- Main App -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pool Table Board")
        self.geometry("1300x750")
        self.minsize(980, 560)

        convert_legacy_tournaments()

        self.sidebar_toggle_var = tk.BooleanVar(value=False)
        self.tournament_toggle_var = tk.BooleanVar(value=True)

        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Table Image...", command=self._proxy_load_table)
        file_menu.add_command(label="Add Ball From Library...", command=self._proxy_load_balls)
        file_menu.add_separator()
        file_menu.add_command(label="Save Layout...", command=self._proxy_save_layout)
        file_menu.add_command(label="Load Layout...", command=self._proxy_load_layout)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_checkbutton(
            label="Tournament Browser",
            variable=self.tournament_toggle_var,
            command=self.toggle_tournament_browser,
        )
        view_menu.add_checkbutton(
            label="Settings Panel",
            variable=self.sidebar_toggle_var,
            command=self.toggle_sidebar,
        )
        menubar.add_cascade(label="View", menu=view_menu)

        self.tournaments_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tournaments", menu=self.tournaments_menu)
        self.config(menu=menubar)

        self._tournament_editor_window: Optional[tk.Toplevel] = None
        self._tournament_editor_tree: Optional[ttk.Treeview] = None
        self._tournament_editor_tree_meta: Dict[str, Dict[str, object]] = {}
        self._tournament_editor_message: Optional[tk.StringVar] = None
        self._tournament_editor_default_message: str = "Select a tournament or match, or create a new tournament."
        self._tournament_editor_canvas: Optional[PoolTableCanvas] = None
        self._tournament_editor_player_frame: Optional[tk.Widget] = None
        self._tournament_editor_player_vars: Optional[Tuple[tk.StringVar, tk.StringVar]] = None
        self._tournament_editor_shot_frame: Optional[tk.Widget] = None
        self._tournament_editor_shot_listbox: Optional[tk.Listbox] = None
        self._tournament_editor_rack_listbox: Optional[tk.Listbox] = None
        self._tournament_editor_rack_items: List[Tuple[str, int]] = []
        self._tournament_editor_shot_items: List[Tuple[str, ShotReference]] = []
        self._tournament_editor_active_match: Optional[Tuple[str, int]] = None
        self._tournament_editor_active_rack: Optional[Tuple[str, int, int]] = None
        self._match_player_names: Dict[Tuple[str, int], Tuple[str, str]] = {}
        self._tournament_editor_save_btn: Optional[tk.Button] = None
        self._tournament_editor_add_rack_btn: Optional[tk.Button] = None
        self._tournament_editor_current_ref: Optional[ShotReference] = None

        self._tournament_documents: Dict[str, Dict] = {}
        self._tournaments_summary: List[Dict] = []
        self._current_shot_reference: Optional[ShotReference] = None

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(0, weight=1)

        self.table_canvas = PoolTableCanvas(self, width=1000, height=600, bd=0)
        self.table_canvas.grid(row=0, column=1, sticky="nsew")

        self.tournament_browser = TournamentBrowser(
            self,
            on_select_shot=self.load_layout_from_path,
            on_request_refresh=self.refresh_tournaments,
            on_add_rack=self._handle_browser_add_rack,
            width=260,
            bd=1,
            relief="groove",
        )
        self.tournament_browser.grid(row=0, column=0, sticky="ns")
        self.tournament_browser.grid_propagate(False)

        self.sidebar = Sidebar(self, self.table_canvas, width=340, bd=1, relief="groove")
        self.sidebar.grid(row=0, column=2, sticky="ns")
        self.sidebar.grid_propagate(False)

        self._sidebar_visible = True
        self.hide_sidebar(initial=True)

        self._tournament_visible = True
        if not self.tournament_toggle_var.get():
            self.hide_tournament_browser(initial=True)

        self.refresh_tournaments()

    def destroy(self):
        if hasattr(self, "table_canvas"):
            self.table_canvas.shutdown()
        if self._tournament_editor_canvas is not None:
            self._tournament_editor_canvas.shutdown()
        super().destroy()

    def toggle_tournament_browser(self):
        if self.tournament_toggle_var.get():
            self.show_tournament_browser()
        else:
            self.hide_tournament_browser()

    def toggle_sidebar(self):
        if self.sidebar_toggle_var.get():
            self.show_sidebar()
        else:
            self.hide_sidebar()

    def show_tournament_browser(self):
        if getattr(self, "_tournament_visible", False):
            return
        self.tournament_browser.grid()
        self.tournament_browser.grid_propagate(False)
        self._tournament_visible = True
        if not self.tournament_toggle_var.get():
            self.tournament_toggle_var.set(True)

    def hide_tournament_browser(self, initial: bool = False):
        if not getattr(self, "_tournament_visible", False) and not initial:
            return
        self.tournament_browser.grid_remove()
        self._tournament_visible = False
        if self.tournament_toggle_var.get():
            self.tournament_toggle_var.set(False)

    def show_sidebar(self):
        if getattr(self, "_sidebar_visible", False):
            return
        self.sidebar.grid()
        self.sidebar.grid_propagate(False)
        self._sidebar_visible = True
        if not self.sidebar_toggle_var.get():
            self.sidebar_toggle_var.set(True)

    def hide_sidebar(self, initial: bool = False):
        if not getattr(self, "_sidebar_visible", False) and not initial:
            return
        self.sidebar.grid_remove()
        self._sidebar_visible = False
        if self.sidebar_toggle_var.get():
            self.sidebar_toggle_var.set(False)

    def _proxy_load_table(self):
        self.sidebar.load_table_dialog()

    def _proxy_load_balls(self):
        self.sidebar.prompt_add_ball_from_catalog()

    def _proxy_save_layout(self):
        self.sidebar.save_layout_dialog()

    def _proxy_load_layout(self):
        self.sidebar.load_layout_dialog()

    def _prompt_add_ball_to_shot(self):
        if self._tournament_editor_canvas is None:
            messagebox.showinfo("Add Ball", "Open or create a shot before adding balls.")
            return

        catalog = self._ball_catalog()
        if not catalog:
            messagebox.showerror("Add Ball", "No ball images available in the catalog.")
            return

        selected_name = None
        if isinstance(getattr(self, "_tournament_editor_ball_var", None), tk.StringVar):
            selected_name = self._tournament_editor_ball_var.get()
        if not selected_name or selected_name not in [name for name, _ in catalog]:
            selected_name = catalog[0][0]

        path = None
        for name, asset_path in catalog:
            if name == selected_name:
                path = asset_path
                break
        if path is None:
            messagebox.showerror("Add Ball", "Selected ball is unavailable.")
            return

        existing_names = {ball.name for ball in self._tournament_editor_canvas.balls}
        candidate = selected_name
        index = 2
        while candidate in existing_names:
            candidate = f"{selected_name}_{index}"
            index += 1

        try:
            self._tournament_editor_canvas.add_ball(candidate, path)
        except Exception as exc:
            messagebox.showerror("Add Ball", f"Failed to add ball:\n{exc}")
            return

        self._tournament_editor_canvas.draw_layer.rerender_all()

    def _prompt_add_drawing_to_shot(self):
        if self._tournament_editor_canvas is None:
            messagebox.showinfo("Add Drawing", "Open or create a shot before adding drawings.")
            return

        self._tournament_editor_canvas.tool_mode = "line"
        if hasattr(self, "sidebar"):
            self.sidebar.tool_choice.set("line")
            self.sidebar.on_tool_change()
        messagebox.showinfo(
            "Add Drawing",
            "Drawing tool switched to Line. Click and drag on the editor canvas to sketch.",
        )

    def refresh_tournaments(self):
        self._tournament_documents.clear()
        self._tournaments_summary = self._load_tournaments_from_disk()
        self._match_player_names.clear()

        if hasattr(self, "tournament_browser"):
            self.tournament_browser.refresh(self._tournaments_summary, selected=self._current_shot_reference)

        self._build_tournaments_menu()
        self._refresh_tournament_editor()
        self._update_tournament_ball_catalog()

    def _handle_browser_add_rack(self, context: Dict[str, Any]):
        if not isinstance(context, dict):
            return
        tournament_path = context.get("tournament_path")
        match_index = context.get("match_index")
        if not isinstance(tournament_path, str) or not isinstance(match_index, int):
            messagebox.showerror("Add Rack", "Unable to determine match for new rack.")
            return

        new_rack_index = self._add_rack_to_match(tournament_path, match_index, parent=self)
        if new_rack_index is None:
            return

        self.after(0, lambda: self.tournament_browser.focus_match(tournament_path, match_index))
        if (
            self._tournament_editor_window is not None
            and self._tournament_editor_window.winfo_exists()
        ):
            def focus_editor():
                self._focus_editor_match(tournament_path, match_index)
                self.after(50, lambda: self._focus_editor_rack(tournament_path, match_index, new_rack_index))

            self.after(0, focus_editor)

    def _ball_catalog(self) -> List[Tuple[str, str]]:
        return list_ball_assets()

    def _update_tournament_ball_catalog(self):
        combo = getattr(self, "_tournament_editor_ball_combo", None)
        if combo is None:
            return
        try:
            combo.winfo_exists()
        except tk.TclError:
            self._tournament_editor_ball_combo = None
            return

        catalog = self._ball_catalog()
        names = [name for name, _ in catalog]
        combo.configure(values=names)
        self._tournament_editor_ball_catalog = catalog

        if not catalog:
            if hasattr(self, "_tournament_editor_ball_var") and isinstance(self._tournament_editor_ball_var, tk.StringVar):
                self._tournament_editor_ball_var.set("")
            combo.configure(state="disabled")
        else:
            combo.configure(state="readonly")
            if hasattr(self, "_tournament_editor_ball_var") and isinstance(self._tournament_editor_ball_var, tk.StringVar):
                current = self._tournament_editor_ball_var.get()
                if current not in names:
                    self._tournament_editor_ball_var.set(names[0])

    def _load_tournaments_from_disk(self) -> List[Dict]:
        tournaments: List[Dict] = []
        if not os.path.isdir(TOURNAMENTS_DIR):
            return tournaments

        for entry in sorted(os.listdir(TOURNAMENTS_DIR)):
            if not entry.lower().endswith(".json"):
                continue
            tournament_path = os.path.join(TOURNAMENTS_DIR, entry)
            if not os.path.isfile(tournament_path):
                continue

            try:
                with open(tournament_path, "r", encoding="utf-8") as handle:
                    document = json.load(handle)
            except Exception:
                continue

            if not isinstance(document, dict):
                continue

            normalize_tournament_document(document)
            matches_raw = document.get("matches", [])
            if not isinstance(matches_raw, list):
                matches_raw = []

            summary_matches: List[Dict] = []
            for match_index, match_raw in enumerate(matches_raw):
                if not isinstance(match_raw, dict):
                    continue

                match_name = match_raw.get("name") or f"Match {match_index + 1}"
                player_one = match_raw.get("playerOne") or "Player 1"
                player_two = match_raw.get("playerTwo") or "Player 2"

                racks_raw = match_raw.get("racks", [])
                if not isinstance(racks_raw, list):
                    racks_raw = []

                summary_racks: List[Dict] = []
                for rack_index, rack_raw in enumerate(racks_raw):
                    if not isinstance(rack_raw, dict):
                        continue

                    shots_raw = rack_raw.get("shots", [])
                    if not isinstance(shots_raw, list):
                        shots_raw = []

                    summary_shots: List[Dict] = []
                    for shot_index, shot_raw in enumerate(shots_raw):
                        if not isinstance(shot_raw, dict):
                            continue
                        data = shot_raw.get("data")
                        if not isinstance(data, dict):
                            continue

                        player_key = shot_raw.get("player")
                        if player_key not in ("playerOne", "playerTwo"):
                            player_key = "playerOne"

                        label = shot_raw.get("name") or f"Shot {shot_index + 1}"
                        reference = ShotReference(tournament_path, match_index, rack_index, shot_index)

                        summary_shots.append(
                            {
                                "label": label,
                                "player": player_key,
                                "reference": reference,
                            }
                        )

                    rack_label = rack_raw.get("name") or f"Rack {rack_index + 1}"
                    rack_break = rack_raw.get("break") or player_one

                    summary_racks.append(
                        {
                            "label": rack_label,
                            "break": rack_break,
                            "shots": summary_shots,
                        }
                    )

                summary_matches.append(
                    {
                        "name": match_name.replace("_", " "),
                        "player_one": player_one,
                        "player_two": player_two,
                        "racks": summary_racks,
                    }
                )

            display_name = document.get("name")
            if not display_name:
                display_name = os.path.splitext(entry)[0].replace("_", " ")

            tournaments.append(
                {
                    "name": display_name,
                    "path": tournament_path,
                    "matches": summary_matches,
                }
            )

            self._tournament_documents[tournament_path] = document

        return tournaments

    def _get_tournament_document(self, path: str) -> Optional[Dict]:
        document = self._tournament_documents.get(path)
        if document is not None:
            return document

        if not os.path.isfile(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
        except Exception:
            return None

        if not isinstance(document, dict):
            return None

        normalize_tournament_document(document)

        self._tournament_documents[path] = document
        return document

    def _get_shot_data(self, reference: ShotReference) -> Optional[Dict]:
        document = self._get_tournament_document(reference.tournament_path)
        if document is None:
            messagebox.showerror("Error", "Unable to load tournament data for the selected shot.")
            return None

        matches = document.get("matches", [])
        if not isinstance(matches, list):
            messagebox.showerror("Error", "Tournament data is malformed.")
            return None

        try:
            match = matches[reference.match_index]
            racks = match.get("racks", [])
            rack = racks[reference.rack_index]
            shots = rack.get("shots", [])
            shot = shots[reference.shot_index]
        except (IndexError, AttributeError, TypeError):
            messagebox.showerror("Error", "Shot reference is out of date.")
            return None

        if not isinstance(shot, dict):
            messagebox.showerror("Error", "Shot data is malformed.")
            return None

        data = shot.get("data")
        if not isinstance(data, dict):
            messagebox.showerror("Error", "Shot layout data is missing or invalid.")
            return None

        return copy.deepcopy(data)

    def _update_shot_data(
        self,
        reference: ShotReference,
        data: Dict,
        player_names: Optional[Tuple[str, str]] = None,
    ) -> bool:
        document = self._get_tournament_document(reference.tournament_path)
        if document is None:
            messagebox.showerror("Save Shot", "Unable to locate tournament data for saving.")
            return False

        matches = document.get("matches", [])
        if not isinstance(matches, list):
            messagebox.showerror("Save Shot", "Tournament data is malformed.")
            return False

        try:
            match = matches[reference.match_index]
            racks = match.get("racks", [])
            rack = racks[reference.rack_index]
            shots = rack.get("shots", [])
            shot = shots[reference.shot_index]
        except (IndexError, AttributeError, TypeError):
            messagebox.showerror("Save Shot", "Shot reference is out of date.")
            return False

        if not isinstance(shot, dict):
            messagebox.showerror("Save Shot", "Shot data is malformed.")
            return False

        shot["data"] = data

        if player_names is not None:
            player_one, player_two = player_names
            match["playerOne"] = player_one
            match["playerTwo"] = player_two

        try:
            with open(reference.tournament_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
        except Exception as exc:
            messagebox.showerror("Save Shot", f"Failed to save tournament file:\n{exc}")
            return False

        self._tournament_documents[reference.tournament_path] = document
        matches_summary: List[Dict] = []
        for tournament in self._tournaments_summary:
            if tournament.get("path") != reference.tournament_path:
                continue
            matches_summary = tournament.get("matches", [])
            if 0 <= reference.match_index < len(matches_summary):
                match_summary = matches_summary[reference.match_index]
                if player_names is not None:
                    match_summary["player_one"], match_summary["player_two"] = player_names
            break

        if player_names is not None and self._tournament_editor_tree is not None:
            for item_id, meta in self._tournament_editor_tree_meta.items():
                if meta.get("type") != "match":
                    continue
                if (
                    meta.get("tournament_path") == reference.tournament_path
                    and int(meta.get("match_index", -1)) == reference.match_index
                ):
                    meta["player_one"], meta["player_two"] = player_names
                    if 0 <= reference.match_index < len(matches_summary):
                        meta["racks"] = matches_summary[reference.match_index].get("racks", [])
                    try:
                        new_text = f"{meta.get('match_name', 'Match')} — {player_names[0]} vs {player_names[1]}"
                        self._tournament_editor_tree.item(item_id, text=new_text)
                    except tk.TclError:
                        pass
                    break
        return True

    def _build_tournaments_menu(self):
        self.tournaments_menu.delete(0, "end")
        tournaments = self._tournaments_summary

        if not tournaments:
            self.tournaments_menu.add_command(label="No tournaments available", state="disabled")
        else:
            for tournament in tournaments:
                tournament_name = tournament.get("name", "Tournament")
                matches = tournament.get("matches", [])
                match_menu = tk.Menu(self.tournaments_menu, tearoff=0)
                if not matches:
                    match_menu.add_command(label="No matches", state="disabled")
                else:
                    for match_index, match in enumerate(matches):
                        match_name = match.get("name", f"Match {match_index + 1}")
                        racks = match.get("racks", [])
                        rack_menu = tk.Menu(match_menu, tearoff=0)
                        if not racks:
                            rack_menu.add_command(label="No racks", state="disabled")
                        else:
                            for rack_index, rack in enumerate(racks):
                                rack_label = rack.get("label") or f"Rack {rack_index + 1}"
                                rack_break = rack.get("break")
                                rack_text = f"{rack_label} — Break: {rack_break}" if rack_break else rack_label
                                shots = rack.get("shots", [])
                                shot_menu = tk.Menu(rack_menu, tearoff=0)
                                if not shots:
                                    shot_menu.add_command(label="No shots", state="disabled")
                                else:
                                    for shot in shots:
                                        reference = shot.get("reference")
                                        if not isinstance(reference, ShotReference):
                                            continue
                                        shot_label = shot.get("label", "Shot")
                                        shot_menu.add_command(
                                            label=shot_label,
                                            command=lambda r=reference: self.load_layout_from_path(r),
                                        )
                                rack_menu.add_cascade(label=rack_text, menu=shot_menu)
                        match_menu.add_cascade(label=match_name, menu=rack_menu)
                self.tournaments_menu.add_cascade(label=tournament_name, menu=match_menu)

        if self.tournaments_menu.index("end") is not None:
            self.tournaments_menu.add_separator()
        self.tournaments_menu.add_command(
            label="Edit Tournament",
            command=self.open_tournament_editor,
        )
        self.tournaments_menu.add_command(label="Refresh", command=self.refresh_tournaments)

    def open_tournament_editor(self):
        if self._tournament_editor_window is not None:
            if self._tournament_editor_window.winfo_exists():
                self._tournament_editor_window.deiconify()
                self._tournament_editor_window.lift()
                self._tournament_editor_window.focus_force()
                return
            self._tournament_editor_window = None

        window = tk.Toplevel(self)
        window.title("Edit Tournament")
        window.geometry("1024x700")
        window.minsize(900, 600)

        main_frame = tk.Frame(window)
        main_frame.pack(fill="both", expand=True)

        list_frame = tk.Frame(main_frame, bd=1, relief="sunken")
        list_frame.pack(side="left", fill="y", padx=(12, 8), pady=12)

        tk.Label(list_frame, text="Tournaments & Matches").pack(anchor="w", padx=8, pady=(8, 4))

        tree = ttk.Treeview(list_frame, show="tree", selectmode="browse", height=12)
        tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        new_tournament_btn = tk.Button(
            list_frame,
            text="New Tournament",
            command=self._prompt_new_tournament,
        )
        new_tournament_btn.pack(fill="x", padx=8, pady=(0, 4))

        new_match_btn = tk.Button(
            list_frame,
            text="New Match",
            command=self._prompt_new_match,
        )
        new_match_btn.pack(fill="x", padx=8, pady=(0, 8))

        canvas_frame = tk.Frame(main_frame, bd=1, relief="sunken")
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 0), pady=12)

        editor_canvas = PoolTableCanvas(canvas_frame, width=720, height=520, bd=0)
        editor_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        detail_frame = tk.Frame(main_frame, bd=1, relief="sunken")
        detail_frame.pack(side="right", fill="y", padx=(8, 12), pady=12)

        default_message = self._tournament_editor_default_message
        message_var = tk.StringVar(value=default_message)
        message_label = tk.Label(
            detail_frame,
            textvariable=message_var,
            wraplength=260,
            justify="left",
            anchor="nw",
        )
        message_label.pack(fill="x", expand=False, padx=12, pady=(12, 0))

        players_frame = tk.LabelFrame(detail_frame, text="Players")
        player1_var = tk.StringVar()
        player2_var = tk.StringVar()

        tk.Label(players_frame, text="Player 1").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        player1_entry = tk.Entry(players_frame, textvariable=player1_var)
        player1_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=(8, 4))

        tk.Label(players_frame, text="Player 2").grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))
        player2_entry = tk.Entry(players_frame, textvariable=player2_var)
        player2_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))

        players_frame.columnconfigure(1, weight=1)
        players_frame.pack(fill="x", expand=False, padx=12, pady=(12, 6))
        players_frame.pack_forget()

        save_btn = tk.Button(
            detail_frame,
            text="Save Shot",
            command=self._save_current_match_shot,
            state=tk.DISABLED,
        )
        save_btn.pack(fill="x", expand=False, padx=12, pady=(0, 6))
        save_btn.pack_forget()

        shots_frame = tk.LabelFrame(detail_frame, text="Racks & Shots")
        shots_inner = tk.Frame(shots_frame)
        shots_inner.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        rack_container = tk.Frame(shots_inner)
        rack_container.pack(side="left", fill="y", padx=(0, 6))
        tk.Label(rack_container, text="Racks").pack(anchor="w")

        rack_listbox = tk.Listbox(rack_container, exportselection=False, height=8)
        rack_listbox.pack(side="left", fill="y", expand=False)

        rack_scroll = ttk.Scrollbar(rack_container, orient="vertical", command=rack_listbox.yview)
        rack_scroll.pack(side="right", fill="y")
        rack_listbox.configure(yscrollcommand=rack_scroll.set)

        shots_container = tk.Frame(shots_inner)
        shots_container.pack(side="left", fill="both", expand=True)
        tk.Label(shots_container, text="Shots").pack(anchor="w")

        shot_listbox = tk.Listbox(shots_container, exportselection=False, height=8)
        shot_listbox.pack(side="left", fill="both", expand=True)

        shot_scroll = ttk.Scrollbar(shots_container, orient="vertical", command=shot_listbox.yview)
        shot_scroll.pack(side="right", fill="y")
        shot_listbox.configure(yscrollcommand=shot_scroll.set)

        shot_btn_frame = tk.Frame(shots_frame)
        shot_btn_frame.pack(fill="x", padx=8, pady=(0, 8))

        rack_buttons_frame = tk.Frame(shot_btn_frame)
        rack_buttons_frame.pack(side="left")

        add_rack_btn = tk.Button(
            rack_buttons_frame,
            text="Add Rack",
            command=self._handle_editor_add_rack,
        )
        add_rack_btn.pack(side="left", padx=(0, 4))

        new_shot_btn = tk.Button(
            rack_buttons_frame,
            text="Add Shot",
            command=self._prompt_new_shot,
        )
        new_shot_btn.pack(side="left")

        add_assets_frame = tk.Frame(shot_btn_frame)
        add_assets_frame.pack(side="left", padx=(8, 0))

        self._tournament_editor_ball_var = tk.StringVar()
        self._tournament_editor_ball_combo = ttk.Combobox(
            add_assets_frame,
            state="readonly",
            textvariable=self._tournament_editor_ball_var,
            values=[],
            width=16,
        )
        self._tournament_editor_ball_combo.pack(side="left", padx=(0, 4))
        self._tournament_editor_ball_catalog: List[Tuple[str, str]] = []

        add_ball_btn = tk.Button(
            add_assets_frame,
            text="Add Ball",
            command=self._prompt_add_ball_to_shot,
        )
        add_ball_btn.pack(side="left", padx=(0, 4))

        add_drawing_btn = tk.Button(
            add_assets_frame,
            text="Add Drawing",
            command=self._prompt_add_drawing_to_shot,
        )
        add_drawing_btn.pack(side="left")

        open_shot_btn = tk.Button(
            shot_btn_frame,
            text="Open Selected Shot",
            command=self._open_selected_match_shot,
        )
        open_shot_btn.pack(side="right")

        shots_frame.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        shots_frame.pack_forget()

        def on_players_changed(*_args):
            if self._tournament_editor_active_match is None:
                return
            self._match_player_names[self._tournament_editor_active_match] = (
                player1_var.get(),
                player2_var.get(),
            )

        player1_var.trace_add("write", on_players_changed)
        player2_var.trace_add("write", on_players_changed)

        def on_shot_activate(_event=None):
            self._open_selected_match_shot()

        def on_shot_select(_event=None):
            self._open_selected_match_shot()

        def on_rack_select(_event=None):
            self._handle_rack_selection()

        shot_listbox.bind("<Double-Button-1>", on_shot_activate)
        shot_listbox.bind("<<ListboxSelect>>", on_shot_select)
        rack_listbox.bind("<<ListboxSelect>>", on_rack_select)
        rack_listbox.bind("<Double-Button-1>", on_rack_select)

        def on_select(_event=None):
            selection = tree.selection()
            if not selection:
                message_var.set(default_message)
                self._hide_match_detail()
                return
            item_id = selection[0]
            meta = self._tournament_editor_tree_meta.get(item_id)
            if not meta:
                message_var.set(default_message)
                self._hide_match_detail()
                return
            if meta.get("type") == "match":
                self._show_match_detail(meta)
                racks = meta.get("racks") or []
                tournament_name = meta.get("tournament_name", "")
                match_name = meta.get("match_name", "")
                rack_count = len(racks)
                shot_count = 0
                for rack in racks:
                    shots = rack.get("shots") if isinstance(rack, dict) else []
                    if isinstance(shots, list):
                        shot_count += len(shots)
                message_var.set(
                    f"Selected match: {match_name} (Tournament: {tournament_name}) — {rack_count} rack(s), {shot_count} shot(s)."
                )
            else:
                tournament_name = meta.get("tournament_name", "")
                message_var.set(f"Selected tournament: {tournament_name}")
                self._hide_match_detail()

        tree.bind("<<TreeviewSelect>>", on_select)

        def on_close():
            self._tournament_editor_window = None
            self._tournament_editor_tree = None
            self._tournament_editor_tree_meta = {}
            self._tournament_editor_message = None
            if self._tournament_editor_canvas is not None:
                self._tournament_editor_canvas.shutdown()
            self._tournament_editor_canvas = None
            self._tournament_editor_player_frame = None
            self._tournament_editor_player_vars = None
            self._tournament_editor_shot_frame = None
            self._tournament_editor_shot_listbox = None
            self._tournament_editor_rack_listbox = None
            self._tournament_editor_rack_items = []
            self._tournament_editor_shot_items = []
            self._tournament_editor_active_match = None
            self._tournament_editor_active_rack = None
            self._tournament_editor_save_btn = None
            self._tournament_editor_add_rack_btn = None
            self._tournament_editor_current_ref = None
            self._tournament_editor_ball_combo = None
            self._tournament_editor_ball_var = None
            self._tournament_editor_ball_catalog = []
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", on_close)
        self._tournament_editor_window = window
        self._tournament_editor_tree = tree
        self._tournament_editor_message = message_var
        self._tournament_editor_canvas = editor_canvas
        self._tournament_editor_player_frame = players_frame
        self._tournament_editor_player_vars = (player1_var, player2_var)
        self._tournament_editor_shot_frame = shots_frame
        self._tournament_editor_shot_listbox = shot_listbox
        self._tournament_editor_rack_listbox = rack_listbox
        self._tournament_editor_add_rack_btn = add_rack_btn
        self._tournament_editor_save_btn = save_btn

        self._hide_match_detail()
        self._refresh_tournament_editor()
        self._update_tournament_ball_catalog()

    def _refresh_tournament_editor(self, select_name: Optional[str] = None):
        if (
            self._tournament_editor_window is None
            or not self._tournament_editor_window.winfo_exists()
            or self._tournament_editor_tree is None
        ):
            return

        tree = self._tournament_editor_tree
        tree.delete(*tree.get_children())
        self._tournament_editor_tree_meta.clear()

        tournament_items: Dict[str, str] = {}
        for tournament in self._tournaments_summary:
            tournament_name = tournament.get("name", "Tournament")
            tournament_path = tournament.get("path", "")
            matches = tournament.get("matches", [])
            tournament_id = tree.insert("", "end", text=tournament_name, open=True)
            tournament_items[tournament_name] = tournament_id
            self._tournament_editor_tree_meta[tournament_id] = {
                "type": "tournament",
                "tournament_name": tournament_name,
                "tournament_path": tournament_path,
            }
            for match_index, match in enumerate(matches):
                match_label = match.get("name", f"Match {match_index + 1}")
                player_one = match.get("player_one", "Player 1")
                player_two = match.get("player_two", "Player 2")
                match_text = f"{match_label} — {player_one} vs {player_two}"
                match_id = tree.insert(tournament_id, "end", text=match_text)
                self._tournament_editor_tree_meta[match_id] = {
                    "type": "match",
                    "tournament_name": tournament_name,
                    "tournament_path": tournament_path,
                    "match_index": match_index,
                    "match_name": match_label,
                    "player_one": player_one,
                    "player_two": player_two,
                    "racks": match.get("racks", []),
                }

        if not tournament_items:
            if self._tournament_editor_message is not None:
                self._tournament_editor_message.set("No tournaments found. Create a new one to get started.")
            return

        target_name = select_name if select_name in tournament_items else next(iter(tournament_items))
        target_id = tournament_items.get(target_name)
        if target_id is None:
            return
        tree.selection_set(target_id)
        tree.focus(target_id)
        tree.see(target_id)
        tree.event_generate("<<TreeviewSelect>>")

    def _show_match_detail(self, meta: Dict):
        tournament_path = meta.get("tournament_path", "")
        tournament_name = meta.get("tournament_name", "")
        match_index = int(meta.get("match_index", 0))
        match_name = meta.get("match_name", f"Match {match_index + 1}")
        player_one = meta.get("player_one", "Player 1")
        player_two = meta.get("player_two", "Player 2")
        racks = meta.get("racks") or []

        match_key = (tournament_path, match_index)
        self._tournament_editor_active_match = match_key
        self._tournament_editor_active_rack = None
        self._tournament_editor_current_ref = None
        self._set_editor_add_rack_enabled(True)

        if self._tournament_editor_player_frame is not None:
            if not self._tournament_editor_player_frame.winfo_manager():
                self._tournament_editor_player_frame.pack(fill="x", expand=False, padx=12, pady=(12, 6))

        if self._tournament_editor_save_btn is not None:
            if not self._tournament_editor_save_btn.winfo_manager():
                self._tournament_editor_save_btn.pack(fill="x", expand=False, padx=12, pady=(0, 6))
            self._update_save_button_state(False)

        if self._tournament_editor_shot_frame is not None:
            if not self._tournament_editor_shot_frame.winfo_manager():
                self._tournament_editor_shot_frame.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        if self._tournament_editor_player_vars is not None:
            player1_var, player2_var = self._tournament_editor_player_vars
            stored = self._match_player_names.get(match_key)
            if stored is None:
                inferred = self._infer_players_from_match_folder(match_name)
                primary = player_one or inferred[0]
                secondary = player_two or inferred[1]
                stored = (primary, secondary)
                self._match_player_names[match_key] = stored
            # Avoid unnecessary updates if values already match
            if player1_var.get() != stored[0]:
                player1_var.set(stored[0])
            if player2_var.get() != stored[1]:
                player2_var.set(stored[1])

        self._populate_match_racks(racks, tournament_path, match_index)

    def _hide_match_detail(self):
        self._tournament_editor_active_match = None
        self._tournament_editor_active_rack = None
        self._tournament_editor_current_ref = None
        if self._tournament_editor_player_frame is not None and self._tournament_editor_player_frame.winfo_manager():
            self._tournament_editor_player_frame.pack_forget()
        if self._tournament_editor_shot_frame is not None and self._tournament_editor_shot_frame.winfo_manager():
            self._tournament_editor_shot_frame.pack_forget()
        if self._tournament_editor_rack_listbox is not None:
            self._tournament_editor_rack_listbox.delete(0, "end")
        self._tournament_editor_rack_items = []
        if self._tournament_editor_shot_listbox is not None:
            self._tournament_editor_shot_listbox.delete(0, "end")
        self._tournament_editor_shot_items = []
        self._update_save_button_state(False)
        if self._tournament_editor_save_btn is not None and self._tournament_editor_save_btn.winfo_manager():
            self._tournament_editor_save_btn.pack_forget()
        self._set_editor_add_rack_enabled(False)

    def _populate_match_racks(self, racks: List[Dict], tournament_path: str, match_index: int):
        self._tournament_editor_rack_items = []
        if self._tournament_editor_rack_listbox is None:
            return

        listbox = self._tournament_editor_rack_listbox
        listbox.delete(0, "end")
        self._tournament_editor_active_rack = None

        if not racks:
            listbox.insert("end", "(No racks in this match)")
            index = listbox.size() - 1
            if index >= 0:
                listbox.itemconfig(index, foreground="#666666")
            self._populate_rack_shots([], tournament_path, match_index, None)
            return

        for rack_index, rack in enumerate(racks):
            label = rack.get("name") or f"Rack {rack_index + 1}"
            breaker = rack.get("break")
            text = f"{label} — Break: {breaker}" if breaker else label
            self._tournament_editor_rack_items.append((label, rack_index))
            listbox.insert("end", text)

        try:
            listbox.selection_set(0)
            listbox.see(0)
        except tk.TclError:
            pass

        self._set_active_rack(tournament_path, match_index, 0, racks)

    def _populate_rack_shots(
        self,
        racks: List[Dict],
        tournament_path: str,
        match_index: int,
        rack_index: Optional[int],
    ):
        self._tournament_editor_shot_items = []
        if self._tournament_editor_shot_listbox is None:
            return

        listbox = self._tournament_editor_shot_listbox
        listbox.delete(0, "end")

        if rack_index is None or rack_index < 0 or rack_index >= len(racks):
            listbox.insert("end", "(Select a rack to view shots)")
            index = listbox.size() - 1
            if index >= 0:
                listbox.itemconfig(index, foreground="#666666")
            return

        rack = racks[rack_index]
        shots = rack.get("shots") if isinstance(rack, dict) else []
        if not isinstance(shots, list) or not shots:
            listbox.insert("end", "(No shots in this rack)")
            index = listbox.size() - 1
            if index >= 0:
                listbox.itemconfig(index, foreground="#666666")
            return

        self._update_save_button_state(False)
        for shot in shots:
            label = shot.get("label", "Shot")
            reference = shot.get("reference")
            if not isinstance(reference, ShotReference):
                continue
            self._tournament_editor_shot_items.append((label, reference))
            listbox.insert("end", label)

    def _handle_rack_selection(self):
        match_key = self._tournament_editor_active_match
        if match_key is None:
            return
        if self._tournament_editor_rack_listbox is None:
            return

        selection = self._tournament_editor_rack_listbox.curselection()
        if not selection:
            self._set_active_rack(match_key[0], match_key[1], -1)
            return

        rack_index = selection[0]
        if rack_index >= len(self._tournament_editor_rack_items):
            self._set_active_rack(match_key[0], match_key[1], -1)
            return

        match_summary = self._get_match_summary(match_key[0], match_key[1])
        racks = match_summary.get("racks", []) if match_summary else []
        self._set_active_rack(match_key[0], match_key[1], rack_index, racks)

    def _set_active_rack(
        self,
        tournament_path: str,
        match_index: int,
        rack_index: int,
        racks: Optional[List[Dict]] = None,
    ):
        if racks is None:
            match_summary = self._get_match_summary(tournament_path, match_index)
            racks = match_summary.get("racks", []) if match_summary else []

        if not isinstance(racks, list) or rack_index < 0 or rack_index >= len(racks):
            self._tournament_editor_active_rack = None
            self._populate_rack_shots(racks or [], tournament_path, match_index, None)
            return

        self._tournament_editor_active_rack = (tournament_path, match_index, rack_index)
        self._populate_rack_shots(racks, tournament_path, match_index, rack_index)

    def _get_match_summary(self, tournament_path: str, match_index: int) -> Optional[Dict[str, Any]]:
        for tournament in self._tournaments_summary:
            if tournament.get("path") != tournament_path:
                continue
            matches = tournament.get("matches", [])
            if not isinstance(matches, list):
                return None
            if 0 <= match_index < len(matches):
                return matches[match_index]
        return None

    def _set_editor_add_rack_enabled(self, enabled: bool):
        btn = self._tournament_editor_add_rack_btn
        if btn is None:
            return
        new_state = tk.NORMAL if enabled else tk.DISABLED
        if str(btn["state"]) != str(new_state):
            btn.configure(state=new_state)

    def _update_save_button_state(self, enabled: bool):
        if self._tournament_editor_save_btn is None:
            return
        new_state = tk.NORMAL if enabled else tk.DISABLED
        if str(self._tournament_editor_save_btn["state"]) != str(new_state):
            self._tournament_editor_save_btn.configure(state=new_state)

    def _infer_players_from_match_folder(self, folder: Optional[str]) -> Tuple[str, str]:
        default_names = ("Player 1", "Player 2")
        if not folder:
            return default_names

        normalized = folder.replace(" ", "_")
        return infer_players_from_folder_name(normalized)

    def _open_selected_match_shot(self):
        if self._tournament_editor_shot_listbox is None:
            return
        selection = self._tournament_editor_shot_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if index >= len(self._tournament_editor_shot_items):
            return
        label, reference = self._tournament_editor_shot_items[index]
        if not isinstance(reference, ShotReference):
            return
        data = self._get_shot_data(reference)
        if data is None:
            return
        if not self._apply_layout_to_table_canvas(data):
            return
        self._apply_layout_to_editor_canvas(data)
        self._tournament_editor_current_ref = reference
        self._current_shot_reference = reference
        self._update_save_button_state(True)

    def _save_current_match_shot(self):
        if self._tournament_editor_current_ref is None:
            messagebox.showinfo("Save Shot", "Select a match shot before saving.")
            return
        if self._tournament_editor_canvas is None:
            messagebox.showerror("Save Shot", "Editor canvas is not available.")
            return

        reference = self._tournament_editor_current_ref
        data = self._tournament_editor_canvas.serialize()

        player_names: Optional[Tuple[str, str]] = None
        if self._tournament_editor_player_vars is not None:
            player1_var, player2_var = self._tournament_editor_player_vars
            player_one = player1_var.get().strip() or "Player 1"
            player_two = player2_var.get().strip() or "Player 2"
            player_names = (player_one, player_two)

        if not self._update_shot_data(reference, data, player_names):
            return

        if not self._apply_layout_to_table_canvas(data):
            return

        if player_names is not None and self._tournament_editor_active_match is not None:
            self._match_player_names[self._tournament_editor_active_match] = player_names

        messagebox.showinfo("Save Shot", "Shot saved successfully.")

        if hasattr(self, "tournament_browser"):
            self.tournament_browser.refresh(self._tournaments_summary, selected=self._current_shot_reference)

    def _read_layout_data(self, path: str) -> Optional[Dict]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load shot:\n{exc}")
            return None

    def _apply_layout_to_table_canvas(self, data: Dict) -> bool:
        try:
            self.table_canvas.restore(data)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to apply layout:\n{exc}")
            return False
        if hasattr(self, "sidebar"):
            self.sidebar.refresh_ball_list()
        return True

    def _apply_layout_to_editor_canvas(self, data: Dict):
        if self._tournament_editor_canvas is None:
            return
        try:
            self._tournament_editor_canvas.restore(data)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to display shot in editor:\n{exc}")

    def _prompt_new_tournament(self):
        if self._tournament_editor_window is None:
            return

        name = simpledialog.askstring("New Tournament", "Enter tournament name:", parent=self._tournament_editor_window)
        if name is None:
            return

        cleaned = " ".join(name.strip().split())
        if not cleaned:
            messagebox.showinfo("New Tournament", "Please enter a valid tournament name.")
            return

        file_stem = cleaned.replace(" ", "_")
        tournament_path = os.path.join(TOURNAMENTS_DIR, f"{file_stem}.json")

        if os.path.exists(tournament_path):
            messagebox.showerror(
                "New Tournament",
                f"A tournament named '{cleaned}' already exists.",
            )
            return

        document = {"name": cleaned, "matches": []}

        try:
            with open(tournament_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
        except OSError as exc:
            messagebox.showerror("New Tournament", f"Failed to create tournament:\n{exc}")
            return

        self.refresh_tournaments()
        self._refresh_tournament_editor(select_name=cleaned)

    def _get_selected_editor_meta(self) -> Optional[Dict]:
        if self._tournament_editor_tree is None:
            return None
        selection = self._tournament_editor_tree.selection()
        if not selection:
            return None
        return self._tournament_editor_tree_meta.get(selection[0])

    def _focus_editor_match(self, tournament_path: str, match_index: int):
        if self._tournament_editor_tree is None:
            return
        target_id: Optional[str] = None
        for item_id, meta in self._tournament_editor_tree_meta.items():
            if meta.get("type") != "match":
                continue
            if (
                meta.get("tournament_path") == tournament_path
                and int(meta.get("match_index", -1)) == match_index
            ):
                target_id = item_id
                break
        if target_id is None:
            return
        try:
            self._tournament_editor_tree.selection_set(target_id)
            self._tournament_editor_tree.focus(target_id)
            self._tournament_editor_tree.see(target_id)
            self._tournament_editor_tree.event_generate("<<TreeviewSelect>>")
        except tk.TclError:
            pass

    def _focus_editor_rack(self, tournament_path: str, match_index: int, rack_index: int):
        if self._tournament_editor_rack_listbox is None:
            return
        match_key = (tournament_path, match_index)
        if self._tournament_editor_active_match != match_key:
            self._focus_editor_match(tournament_path, match_index)
        if self._tournament_editor_rack_listbox is None:
            return
        if rack_index < 0 or rack_index >= len(self._tournament_editor_rack_items):
            return
        try:
            self._tournament_editor_rack_listbox.selection_clear(0, "end")
            self._tournament_editor_rack_listbox.selection_set(rack_index)
            self._tournament_editor_rack_listbox.see(rack_index)
        except tk.TclError:
            pass
        self._handle_rack_selection()

    def _focus_editor_shot(self, reference: ShotReference):
        if self._tournament_editor_shot_listbox is None:
            return
        match_key = (reference.tournament_path, reference.match_index)
        if self._tournament_editor_active_match != match_key:
            self._focus_editor_match(reference.tournament_path, reference.match_index)
        if (
            self._tournament_editor_active_rack is None
            or self._tournament_editor_active_rack[2] != reference.rack_index
        ):
            self._focus_editor_rack(reference.tournament_path, reference.match_index, reference.rack_index)
        for index, (_, ref) in enumerate(self._tournament_editor_shot_items):
            if ref == reference:
                try:
                    self._tournament_editor_shot_listbox.selection_clear(0, "end")
                    self._tournament_editor_shot_listbox.selection_set(index)
                    self._tournament_editor_shot_listbox.see(index)
                except tk.TclError:
                    pass
                self._open_selected_match_shot()
                break

    def _prompt_new_match(self):
        if self._tournament_editor_window is None:
            return

        meta = self._get_selected_editor_meta()
        if not meta:
            messagebox.showinfo("New Match", "Select a tournament before adding a match.")
            return

        if meta.get("type") == "match":
            tournament_path = meta.get("tournament_path")
        elif meta.get("type") == "tournament":
            tournament_path = meta.get("tournament_path")
        else:
            messagebox.showinfo("New Match", "Select a tournament before adding a match.")
            return

        if not tournament_path:
            messagebox.showerror("New Match", "Unable to determine tournament path.")
            return

        document = self._get_tournament_document(tournament_path)
        if document is None:
            messagebox.showerror("New Match", "Failed to load tournament data.")
            return

        matches = document.setdefault("matches", [])
        default_name = f"Match {len(matches) + 1}"

        match_name_input = simpledialog.askstring(
            "New Match",
            "Match name:",
            initialvalue=default_name,
            parent=self._tournament_editor_window,
        )
        if match_name_input is None:
            return
        match_name = " ".join(match_name_input.strip().split()) or default_name

        inferred_one, inferred_two = infer_players_from_folder_name(match_name.replace(" ", "_"))
        player_one = simpledialog.askstring(
            "New Match",
            "Player one name:",
            initialvalue=inferred_one,
            parent=self._tournament_editor_window,
        )
        if player_one is None:
            return
        player_two = simpledialog.askstring(
            "New Match",
            "Player two name:",
            initialvalue=inferred_two,
            parent=self._tournament_editor_window,
        )
        if player_two is None:
            return

        player_one = " ".join(player_one.strip().split()) or inferred_one
        player_two = " ".join(player_two.strip().split()) or inferred_two

        new_match = {
            "name": match_name,
            "playerOne": player_one,
            "playerTwo": player_two,
            "racks": [],
        }
        matches.append(new_match)

        try:
            with open(tournament_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
        except OSError as exc:
            messagebox.showerror("New Match", f"Failed to save tournament:\n{exc}")
            matches.pop()
            return

        self._tournament_documents[tournament_path] = document
        new_match_index = len(matches) - 1
        self._current_shot_reference = None

        self.refresh_tournaments()
        self._match_player_names[(tournament_path, new_match_index)] = (player_one, player_two)
        self.after(0, lambda: self._focus_editor_match(tournament_path, new_match_index))

    def _add_rack_to_match(
        self,
        tournament_path: str,
        match_index: int,
        parent: Optional[tk.Misc] = None,
    ) -> Optional[int]:
        document = self._get_tournament_document(tournament_path)
        if document is None:
            messagebox.showerror("Add Rack", "Failed to load tournament data.")
            return None

        matches = document.get("matches", [])
        if not isinstance(matches, list) or match_index < 0 or match_index >= len(matches):
            messagebox.showerror("Add Rack", "Match reference no longer exists.")
            return None

        match = matches[match_index]
        normalize_match_structure(match)
        racks = match.setdefault("racks", [])

        default_name = f"Rack {len(racks) + 1}"
        parent_window = parent if parent is not None else self

        rack_name_input = simpledialog.askstring(
            "Add Rack",
            "Rack name:",
            initialvalue=default_name,
            parent=parent_window,
        )
        if rack_name_input is None:
            return None
        rack_name = " ".join(rack_name_input.strip().split()) or default_name

        player_one_name = match.get("playerOne") or "Player 1"
        player_two_name = match.get("playerTwo") or "Player 2"

        breaker_input = simpledialog.askstring(
            "Add Rack",
            "Breaker (playerOne/playerTwo):",
            initialvalue="playerOne",
            parent=parent_window,
        )
        if breaker_input is None:
            return None

        breaker_key = resolve_player_key_with_names(breaker_input, player_one_name, player_two_name)
        breaker_name = player_two_name if breaker_key == "playerTwo" else player_one_name

        new_rack = {
            "name": rack_name,
            "break": breaker_name,
            "shots": [],
        }
        racks.append(new_rack)

        try:
            with open(tournament_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
        except OSError as exc:
            messagebox.showerror("Add Rack", f"Failed to save tournament:\n{exc}")
            racks.pop()
            return None

        normalize_tournament_document(document)
        self._tournament_documents[tournament_path] = document
        self._current_shot_reference = None

        self.refresh_tournaments()
        return len(racks) - 1

    def _handle_editor_add_rack(self):
        if self._tournament_editor_window is None:
            return
        if self._tournament_editor_active_match is None:
            messagebox.showinfo("Add Rack", "Select a match before adding a rack.")
            return

        tournament_path, match_index = self._tournament_editor_active_match
        new_rack_index = self._add_rack_to_match(
            tournament_path,
            match_index,
            parent=self._tournament_editor_window,
        )
        if new_rack_index is None:
            return

        def focus_new_rack():
            self._focus_editor_match(tournament_path, match_index)
            self.after(50, lambda: self._focus_editor_rack(tournament_path, match_index, new_rack_index))

        self.after(0, focus_new_rack)

    def _prompt_new_shot(self):
        if self._tournament_editor_window is None:
            return
        if self._tournament_editor_active_match is None:
            messagebox.showinfo("Add Shot", "Select a match before adding a shot.")
            return

        if self._tournament_editor_active_rack is None:
            messagebox.showinfo("Add Shot", "Select a rack before adding a shot.")
            return

        tournament_path, match_index, rack_index = self._tournament_editor_active_rack
        document = self._get_tournament_document(tournament_path)
        if document is None:
            messagebox.showerror("Add Shot", "Failed to load tournament data.")
            return

        matches = document.get("matches", [])
        if not isinstance(matches, list) or match_index >= len(matches):
            messagebox.showerror("Add Shot", "Match reference no longer exists.")
            return

        match = matches[match_index]
        normalize_match_structure(match)
        racks = match.setdefault("racks", [])
        if rack_index < 0 or rack_index >= len(racks):
            messagebox.showerror("Add Shot", "Rack reference no longer exists.")
            return

        rack = racks[rack_index]
        shots = rack.setdefault("shots", [])
        default_name = f"Shot {len(shots) + 1}"

        shot_name_input = simpledialog.askstring(
            "Add Shot",
            "Shot name:",
            initialvalue=default_name,
            parent=self._tournament_editor_window,
        )
        if shot_name_input is None:
            return
        shot_name = " ".join(shot_name_input.strip().split()) or default_name

        current_players = (
            match.get("playerOne", "Player 1"),
            match.get("playerTwo", "Player 2"),
        )

        shooter_input = simpledialog.askstring(
            "Add Shot",
            "Shooter (playerOne/playerTwo):",
            initialvalue="playerOne",
            parent=self._tournament_editor_window,
        )
        if shooter_input is None:
            return
        player_key = resolve_player_key_with_names(shooter_input, *current_players)

        base_state = {
            "table": os.path.join(TABLE_IMAGES_DIR, "Table.png"),
            "table_scale": 1.0,
            "ball_scale": 0.33,
            "table_offset": {"x": 0.0, "y": 0.0},
            "table_rect": [0, 158, 720, 444],
            "webcam": {"enabled": False, "opacity": 0.5, "source": 0},
            "balls": [
                {
                    "name": "TwoBall",
                    "x": 424,
                    "y": 349,
                    "visible": True,
                    "path": os.path.join(BALL_IMAGES_DIR, "TwoBall.png"),
                    "u": 0.5888888888888889,
                    "v": 0.43018018018018017,
                }
            ],
            "drawings": [],
        }

        new_shot = {
            "name": shot_name,
            "player": player_key,
            "data": copy.deepcopy(base_state),
        }
        shots.append(new_shot)
        new_reference = ShotReference(tournament_path, match_index, rack_index, len(shots) - 1)

        try:
            with open(tournament_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
        except OSError as exc:
            messagebox.showerror("Add Shot", f"Failed to save tournament:\n{exc}")
            shots.pop()
            return

        self._tournament_documents[tournament_path] = document
        self._current_shot_reference = new_reference

        self.refresh_tournaments()
        self._match_player_names[(tournament_path, match_index)] = (
            match.get("playerOne", current_players[0]),
            match.get("playerTwo", current_players[1]),
        )

        def focus_new_shot():
            self._focus_editor_match(tournament_path, match_index)
            def focus_rack_then_shot():
                self._focus_editor_rack(tournament_path, match_index, rack_index)
                self.after(50, lambda: self._focus_editor_shot(new_reference))

            self.after(50, focus_rack_then_shot)

        self.after(0, focus_new_shot)

    def load_layout_from_path(self, source):
        if isinstance(source, ShotReference):
            reference = source
            data = self._get_shot_data(reference)
            if data is None:
                return
            if not self._apply_layout_to_table_canvas(data):
                return
            self._apply_layout_to_editor_canvas(data)
            self._current_shot_reference = reference
            if self._tournament_editor_save_btn is not None and self._tournament_editor_save_btn.winfo_manager():
                self._tournament_editor_current_ref = reference
                self._update_save_button_state(True)
            return

        if isinstance(source, str):
            data = self._read_layout_data(source)
            if data is None:
                return
            if not self._apply_layout_to_table_canvas(data):
                return
            self._apply_layout_to_editor_canvas(data)
            self._current_shot_reference = None
            if self._tournament_editor_save_btn is not None and self._tournament_editor_save_btn.winfo_manager():
                self._tournament_editor_current_ref = None
                self._update_save_button_state(False)
            return

        messagebox.showerror("Error", "Unsupported shot selection type.")

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
