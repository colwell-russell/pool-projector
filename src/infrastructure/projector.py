from __future__ import annotations

import os
import tkinter as tk
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageTk

from config import IMAGES_DIR, TABLE_IMAGES_DIR
from models import DrawingState
from services import normalize_asset_path
from utils import clamp

try:
    from screeninfo import get_monitors
except ImportError:  # pragma: no cover
    get_monitors = None


class ProjectorWindow:
    """Borderless window that mirrors the editor state on a secondary display."""

    def __init__(
        self,
        parent: tk.Tk,
        monitor_index: int,
        on_close: Optional[callable] = None,
    ) -> None:
        if get_monitors is None:
            raise RuntimeError("screeninfo is not installed. Run: pip install screeninfo")

        self.parent = parent
        self.monitor_index = monitor_index
        self.on_close = on_close
        self._closed = False
        self.top = tk.Toplevel(parent)
        self.top.attributes("-topmost", True)
        self.top.overrideredirect(True)
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

        self.balls: Dict[str, Dict[str, object]] = {}
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

    def _on_resize(self, _event=None):
        self._render_table()
        self._render_balls()
        self._render_drawings()

    def apply_state(self, state: Dict) -> None:
        table_path = normalize_asset_path(state.get("table"), str(TABLE_IMAGES_DIR))
        if table_path and os.path.exists(table_path):
            state["table"] = table_path
            self._table_pil = Image.open(table_path).convert("RGBA")
        else:
            self._table_pil = None

        editor_rect = state.get("table_rect", (0, 0, 0, 0))
        if isinstance(editor_rect, (list, tuple)) and len(editor_rect) == 4:
            self._editor_table_rect = tuple(float(v) for v in editor_rect)
        else:
            self._editor_table_rect = (0, 0, 0, 0)

        desired = {b["name"] for b in state.get("balls", [])}
        for name in list(self.balls.keys()):
            if name not in desired:
                item = self.balls[name].get("item")
                if item:
                    self.canvas.delete(item)
                del self.balls[name]

        for b in state.get("balls", []):
            name = b["name"]
            path = normalize_asset_path(b.get("path"), str(IMAGES_DIR))
            if not path or not os.path.exists(path):
                continue
            b["path"] = path
            if name not in self.balls:
                self.balls[name] = {
                    "pil": Image.open(path).convert("RGBA"),
                    "tk": None,
                    "item": None,
                    "u": float(b.get("u", 0.5)),
                    "v": float(b.get("v", 0.5)),
                    "visible": bool(b.get("visible", True)),
                }
            else:
                self.balls[name]["pil"] = Image.open(path).convert("RGBA")
                self.balls[name]["u"] = float(b.get("u", 0.5))
                self.balls[name]["v"] = float(b.get("v", 0.5))
                self.balls[name]["visible"] = bool(b.get("visible", True))

        for item_id, _ in self.drawings:
            self.canvas.delete(item_id)
        self.drawings.clear()
        for d in state.get("drawings", []):
            st = DrawingState(**d)
            item_id = self._render_drawing(st)
            self.drawings.append((item_id, st))

        self.global_scale = float(state.get("ball_scale", 1.0))
        self.table_scale = clamp(float(state.get("table_scale", 1.0)), 0.2, 2.0)

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

        self._table_item = self.canvas.create_image(
            cx,
            cy,
            image=self._table_tk,
            anchor="center",
            tags=("table",),
        )

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
                rec["item"] = self.canvas.create_image(
                    int(x),
                    int(y),
                    image=rec["tk"],
                    anchor="center",
                    tags=("ball", name),
                )
            else:
                self.canvas.itemconfigure(rec["item"], image=rec["tk"])
                self.canvas.coords(rec["item"], int(x), int(y))

            self.canvas.itemconfigure(
                rec["item"], state=("normal" if rec["visible"] else "hidden")
            )

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
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            fill=st.color,
            width=st.width,
            arrow=arrowopt,
            capstyle=tk.ROUND,
            smooth=False,
        )


__all__ = ["ProjectorWindow"]
