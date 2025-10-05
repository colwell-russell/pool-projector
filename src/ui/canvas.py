from __future__ import annotations

import os
import tkinter as tk
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageTk

from config import IMAGES_DIR, TABLE_IMAGES_DIR
from models import BallState, DrawingState
from services import normalize_asset_path
from utils import clamp

try:
    import cv2
except ImportError:
    cv2 = None

from tkinter import messagebox

from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageTk

from config import IMAGES_DIR, TABLE_IMAGES_DIR
from models import BallState, DrawingState
from services import normalize_asset_path
from utils import clamp

try:
    import cv2
except ImportError:
    cv2 = None

from tkinter import messagebox

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



    def remove_ball(self, name: str) -> bool:
        """Remove a ball by name; returns True if removed."""
        for index, ball in enumerate(list(self.balls)):
            if ball.name != name:
                continue
            try:
                self.canvas.delete(ball.item_id)
            except tk.TclError:
                pass
            del self.balls[index]
            self._notify()
            return True
        return False


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
        table_path = normalize_asset_path(table, TABLE_IMAGES_DIR)
        if table_path and os.path.exists(table_path):
            data["table"] = table_path
            self.load_table_image(table_path)

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
            path = normalize_asset_path(bstate.get("path"), IMAGES_DIR)
            name = bstate.get("name")
            if path and os.path.exists(path):
                bstate["path"] = path
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



