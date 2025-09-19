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

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Callable

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("This app requires Pillow. Install with: pip install pillow")

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

# ----------------------------- Data Models -----------------------------

@dataclass
class BallState:
    name: str
    x: float          # canvas pixel position (editor) — retained for backward compat
    y: float
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

# ----------------------------- Helpers -----------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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
    def to_state(self) -> BallState:
        x, y = self.position()
        return BallState(name=self.name, x=x, y=y, visible=self.visible, path=self.img_path, u=self.u, v=self.v)

    def from_state(self, state: BallState, table_rect: Tuple[int, int, int, int]):
        # Prefer u,v if present; else fallback to x,y -> uv
        if state.u is not None and state.v is not None:
            self.u = float(state.u)
            self.v = float(state.v)
            self.place_by_uv(table_rect)
        else:
            # Backward compat: derive u,v from saved x,y
            self.set_uv_from_canvas_xy(int(state.x), int(state.y), table_rect)
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
        scale = min(cw / iw, ch / ih)
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        pil_scaled = self._table_pil.resize((new_w, new_h), Image.LANCZOS)
        self._table_tk = ImageTk.PhotoImage(pil_scaled)

        # Centered
        cx, cy = cw // 2, ch // 2
        left = cx - new_w // 2
        top = cy - new_h // 2
        self._table_rect = (left, top, new_w, new_h)

        self._table_item = self.canvas.create_image(cx, cy, image=self._table_tk, anchor="center", tags=("table",))

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
            "ball_scale": self.ball_scale,
            "table_rect": self._table_rect,
            "balls": [asdict(b.to_state()) for b in self.balls],
            "drawings": self.draw_layer.serialize(),
        }

    def restore(self, data: Dict):
        table = data.get("table")
        if table and os.path.exists(table):
            self.load_table_image(table)

        saved_scale = float(data.get("ball_scale", 1.0))
        self.ball_scale = saved_scale

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
                ball.from_state(BallState(**bstate), self._table_rect)
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
        scale = min(cw / iw, ch / ih)
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

        # File controls
        self.btn_load_table = tk.Button(self, text="Load Table Image...", command=self.load_table_dialog)
        self.btn_load_balls = tk.Button(self, text="Load Ball Images...", command=self.load_balls_dialog)
        self.btn_save_layout = tk.Button(self, text="Save Layout...", command=self.save_layout_dialog)
        self.btn_load_layout = tk.Button(self, text="Load Layout...", command=self.load_layout_dialog)

        self.btn_load_table.pack(fill="x", pady=(6, 3), padx=6)
        self.btn_load_balls.pack(fill="x", pady=3, padx=6)

        # Ball size slider
        self.size_label = tk.Label(self, text="Ball Size (%)", anchor="w", font=("Segoe UI", 10, "bold"))
        self.size_label.pack(fill="x", padx=8, pady=(10, 0))

        self.ball_size_var = tk.DoubleVar(value=100.0)
        self.size_slider = tk.Scale(self, from_=20, to=200, orient="horizontal",
                                    variable=self.ball_size_var, command=self.on_size_change)
        self.size_slider.pack(fill="x", padx=10, pady=(4, 10))

        # Drawing tools
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=8, pady=6)
        tk.Label(self, text="Drawing Tools", anchor="w", font=("Segoe UI", 10, "bold")).pack(fill="x", padx=8)

        tool_frame = tk.Frame(self)
        tool_frame.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(tool_frame, text="Tool:").pack(side="left")
        self.tool_choice = tk.StringVar(value="select")
        ttk.Combobox(tool_frame, state="readonly", textvariable=self.tool_choice,
                     values=["select", "line", "arrow"], width=10).pack(side="left", padx=6)

        color_frame = tk.Frame(self)
        color_frame.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(color_frame, text="Color:").pack(side="left")
        self.color_preview = tk.Label(color_frame, text="   ", bg="#ff0000", relief="sunken", width=3)
        self.color_preview.pack(side="left", padx=6)
        tk.Button(color_frame, text="Pick...", command=self.pick_color).pack(side="left")

        width_frame = tk.Frame(self)
        width_frame.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(width_frame, text="Width:").pack(side="left")
        self.width_var = tk.IntVar(value=4)
        tk.Scale(width_frame, from_=1, to=20, orient="horizontal", variable=self.width_var, length=140).pack(side="left", padx=6)

        draw_buttons = tk.Frame(self)
        draw_buttons.pack(fill="x", padx=8, pady=(4, 8))
        tk.Button(draw_buttons, text="Undo", command=self.undo_drawing).pack(side="left")
        tk.Button(draw_buttons, text="Clear", command=self.clear_drawings).pack(side="left", padx=6)

        # Projector display selection
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=8, pady=6)
        self.display_label = tk.Label(self, text="Projector Display", anchor="w", font=("Segoe UI", 10, "bold"))
        self.display_label.pack(fill="x", padx=8, pady=(2, 0))

        self.monitor_list = self._get_monitors_summary()
        default_choice = self.monitor_list[0] if self.monitor_list else "No displays found"
        self.display_choice = tk.StringVar(value=default_choice)
        self.display_dropdown = ttk.Combobox(self, state="readonly", textvariable=self.display_choice, values=self.monitor_list)
        self.display_dropdown.pack(fill="x", padx=10, pady=(4, 6))

        self.open_proj_btn = tk.Button(self, text="Open Projector Window", command=self.open_projector)
        self.close_proj_btn = tk.Button(self, text="Close Projector Window", command=self.close_projector)
        self.open_proj_btn.pack(fill="x", padx=10, pady=(2, 2))
        self.close_proj_btn.pack(fill="x", padx=10, pady=(2, 10))

        self.btn_save_layout.pack(fill="x", pady=(6, 3), padx=6)
        self.btn_load_layout.pack(fill="x", pady=3, padx=6)

        # Balls list
        self.sep = tk.Label(self, text="Balls", anchor="w", font=("Segoe UI", 10, "bold"))
        self.sep.pack(fill="x", padx=8, pady=(12, 0))

        self.ball_frame = tk.Frame(self, bd=1, relief="sunken")
        self.ball_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # Hook editor for mirroring
        self.table_canvas.add_listener(self._on_canvas_update)

        # Initialize tool bindings
        self.tool_choice.trace_add("write", self.on_tool_change)
        self.on_tool_change()

        self.refresh_ball_list()

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
        for w in self.ball_frame.winfo_children():
            w.destroy()
        self.ball_vars.clear()

        for ball in self.table_canvas.balls:
            var = tk.BooleanVar(value=ball.visible)
            self.ball_vars[ball.name] = var
            cb = tk.Checkbutton(self.ball_frame, text=ball.name, variable=var,
                                command=lambda b=ball, v=var: self.toggle_ball(b, v))
            cb.pack(anchor="w", padx=6, pady=2)

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
        paths = filedialog.askopenfilenames(
            title="Select Ball Images (PNG with transparency recommended)",
            filetypes=IMAGE_FILETYPES,
            initialdir=IMAGES_DIR if os.path.isdir(IMAGES_DIR) else None,
        )
        existing_names = {b.name for b in self.table_canvas.balls}
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            name = base
            i = 2
            while name in existing_names:
                name = f"{base}_{i}"
                i += 1
            self.table_canvas.add_ball(name, p)
            existing_names.add(name)

        self.refresh_ball_list()

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

# ----------------------------- Main App -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pool Table Board")
        self.geometry("1300x750")
        self.minsize(980, 560)

        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Table Image...", command=self._proxy_load_table)
        file_menu.add_command(label="Load Ball Images...", command=self._proxy_load_balls)
        file_menu.add_separator()
        file_menu.add_command(label="Save Layout...", command=self._proxy_save_layout)
        file_menu.add_command(label="Load Layout...", command=self._proxy_load_layout)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.table_canvas = PoolTableCanvas(self, width=1000, height=600, bd=0)
        self.table_canvas.grid(row=0, column=0, sticky="nsew")

        self.sidebar = Sidebar(self, self.table_canvas, width=340, bd=1, relief="groove")
        self.sidebar.grid(row=0, column=1, sticky="ns")
        self.sidebar.grid_propagate(False)

    def _proxy_load_table(self):
        self.sidebar.load_table_dialog()

    def _proxy_load_balls(self):
        self.sidebar.load_balls_dialog()

    def _proxy_save_layout(self):
        self.sidebar.save_layout_dialog()

    def _proxy_load_layout(self):
        self.sidebar.load_layout_dialog()

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
