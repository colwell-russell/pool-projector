#!/usr/bin/env python3
"""
Pool Table Board (Presenter Mode)

Features
- Load a pool table image (background).
- Load multiple ball images (PNG with transparency recommended).
- Drag balls around on the main canvas.
- Sidebar checkboxes to toggle each ball’s visibility.
- Global ball size slider (20%–200%).
- Save/Load layout to JSON (includes positions, visibility, and ball scale).
- Display picker: open a borderless "projector window" on a chosen monitor that mirrors the table+balls.
- Live mirroring (moves, visibility, scale) to the projector window.

Dependencies
    pip install pillow screeninfo
Run
    python pool_table_board.py
"""

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Callable

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("This app requires Pillow. Install with: pip install pillow")

try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None  # We'll guard usage and show a helpful message later.

# ----------------------------- Data Models -----------------------------

@dataclass
class BallState:
    name: str
    x: float
    y: float
    visible: bool = True
    path: Optional[str] = None

# ----------------------------- View Models -----------------------------

class BallSprite:
    """
    A draggable, hideable ball image on a Tkinter Canvas.
    Holds onto the original PIL image and can re-render at any scale.
    """
    def __init__(self, canvas: tk.Canvas, name: str, img_path: str, x: int, y: int, scale: float = 1.0):
        self.canvas = canvas
        self.name = name
        self.img_path = img_path
        self.visible = True

        # Load original image with PIL for transparency support
        self.base_pil: Image.Image = Image.open(img_path).convert("RGBA")

        # Current render state
        self.current_scale: float = 1.0
        self.tk_img: Optional[ImageTk.PhotoImage] = None

        # Create canvas item
        self.item_id = canvas.create_image(x, y, anchor="center", tags=("ball", self.name))
        self.apply_scale(scale)

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

    def show(self):
        if not self.visible:
            self.visible = True
            self.canvas.itemconfigure(self.item_id, state="normal")

    def hide(self):
        if self.visible:
            self.visible = False
            self.canvas.itemconfigure(self.item_id, state="hidden")

    def apply_scale(self, scale: float):
        """Render the ball image at a given scale (1.0 = 100%)."""
        scale = max(0.05, min(scale, 20.0))  # clamp
        self.current_scale = scale

        bw, bh = self.base_pil.size
        new_size = (max(1, int(bw * scale)), max(1, int(bh * scale)))
        resized = self.base_pil.resize(new_size, Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.itemconfigure(self.item_id, image=self.tk_img)

    def to_state(self) -> BallState:
        x, y = self.position()
        return BallState(name=self.name, x=x, y=y, visible=self.visible, path=self.img_path)

    def from_state(self, state: BallState):
        self.move_to(state.x, state.y)
        if state.visible:
            self.show()
        else:
            self.hide()

class PoolTableCanvas(tk.Frame):
    """
    Left pane: Canvas that draws the table image and ball sprites.
    Handles drag interactions and notifies listeners (e.g., projector window).
    """
    def __init__(self, master, width=1000, height=560, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, width=width, height=height, bg="#0a5d2a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.table_img_path: Optional[str] = None
        self._table_pil: Optional[Image.Image] = None
        self._table_tk: Optional[ImageTk.PhotoImage] = None
        self._table_item: Optional[int] = None

        self.balls: List[BallSprite] = []
        self._drag_target: Optional[BallSprite] = None
        self._drag_dx = 0
        self._drag_dy = 0

        # Global ball scale (1.0 = 100%)
        self.ball_scale: float = 1.0

        # Listeners for state updates (projector window, etc.)
        self._listeners: List[Callable[[Dict], None]] = []

        # Bind mouse handlers for dragging
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.on_resize)

    # ---------- Listener management ----------
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
        self._notify()

    def _render_table_to_canvas(self):
        if not self._table_pil:
            self.canvas.delete("table")
            self._table_item = None
            return

        cw = self.canvas.winfo_width() or int(self.canvas["width"])
        ch = self.canvas.winfo_height() or int(self.canvas["height"])

        iw, ih = self._table_pil.size
        scale = min(cw / iw, ch / ih)
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        pil_scaled = self._table_pil.resize(new_size, Image.LANCZOS)
        self._table_tk = ImageTk.PhotoImage(pil_scaled)

        self.canvas.delete("table")
        self._table_item = self.canvas.create_image(cw // 2, ch // 2, image=self._table_tk, anchor="center", tags=("table",))

        for b in self.balls:
            self.canvas.tag_raise(b.item_id)

    def on_resize(self, event):
        self._render_table_to_canvas()
        self._notify()

    # ---------- Ball handling ----------
    def add_ball(self, name: str, img_path: str):
        cw = self.canvas.winfo_width() or int(self.canvas["width"])
        ch = self.canvas.winfo_height() or int(self.canvas["height"])
        x, y = cw // 2, ch // 2
        ball = BallSprite(self.canvas, name, img_path, x, y, self.ball_scale)
        self.balls.append(ball)
        self.canvas.tag_raise(ball.item_id)
        self._notify()

    def set_ball_scale(self, scale: float):
        """Apply a new global scale to all balls."""
        self.ball_scale = scale
        for b in self.balls:
            b.apply_scale(scale)
        self._notify()

    def find_ball_at(self, x: int, y: int) -> Optional[BallSprite]:
        hits = self.canvas.find_overlapping(x, y, x, y)
        for item in reversed(hits):
            for b in self.balls:
                if b.item_id == item and b.visible and b.contains(x, y):
                    return b
        return None

    def on_mouse_down(self, event):
        target = self.find_ball_at(event.x, event.y)
        if target:
            bx, by = target.position()
            self._drag_target = target
            self._drag_dx = event.x - bx
            self._drag_dy = event.y - by
            self.canvas.tag_raise(target.item_id)

    def on_mouse_drag(self, event):
        if self._drag_target:
            nx = event.x - self._drag_dx
            ny = event.y - self._drag_dy
            self._drag_target.move_to(nx, ny)
            self._notify()

    def on_mouse_up(self, event):
        self._drag_target = None

    # ---------- State (save/load) ----------
    def serialize(self) -> Dict:
        return {
            "table": self.table_img_path,
            "ball_scale": self.ball_scale,
            "canvas_size": (self.canvas.winfo_width(), self.canvas.winfo_height()),
            "balls": [asdict(b.to_state()) for b in self.balls]
        }

    def restore(self, data: Dict):
        table = data.get("table")
        if table and os.path.exists(table):
            self.load_table_image(table)

        saved_scale = data.get("ball_scale", 1.0)
        self.ball_scale = float(saved_scale)

        for b in self.balls:
            self.canvas.delete(b.item_id)
        self.balls.clear()

        for bstate in data.get("balls", []):
            path = bstate.get("path")
            name = bstate.get("name")
            if path and os.path.exists(path):
                self.add_ball(name, path)
                self.balls[-1].from_state(BallState(**bstate))
                self.balls[-1].apply_scale(self.ball_scale)

        self._notify()

# ----------------------------- Projector Window -----------------------------

class ProjectorWindow:
    """
    A borderless window on a chosen display that mirrors the table+balls.
    """
    def __init__(self, parent: tk.Tk, monitor_index: int):
        if get_monitors is None:
            raise RuntimeError("screeninfo is not installed. Run: pip install screeninfo")

        self.parent = parent
        self.monitor_index = monitor_index
        self.top = tk.Toplevel(parent)
        self.top.attributes("-topmost", True)
        self.top.overrideredirect(True)  # borderless

        # Position on the selected monitor
        monitors = get_monitors()
        if monitor_index < 0 or monitor_index >= len(monitors):
            raise ValueError("Invalid monitor index")
        mon = monitors[monitor_index]
        x, y, w, h = mon.x, mon.y, mon.width, mon.height
        self.top.geometry(f"{w}x{h}+{x}+{y}")

        self.canvas = tk.Canvas(self.top, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Content images
        self._table_pil: Optional[Image.Image] = None
        self._table_tk: Optional[ImageTk.PhotoImage] = None
        self._table_item: Optional[int] = None

        self.ball_images: Dict[str, Dict[str, object]] = {}  # name -> {'pil': PIL, 'tk': PhotoImage, 'item': id, 'visible': bool}

        # Global scale
        self.global_scale: float = 1.0

        # Resize handling
        self.canvas.bind("<Configure>", self._on_resize)

    def close(self):
        if self.top and self.top.winfo_exists():
            self.top.destroy()

    def _on_resize(self, event):
        # Re-render table (fit) and reapply ball images at current positions
        self._render_table()
        self._render_balls()

    def apply_state(self, state: Dict):
        # Update table
        table_path = state.get("table")
        if table_path and os.path.exists(table_path):
            self._table_pil = Image.open(table_path).convert("RGBA")
        else:
            self._table_pil = None

        # Sync ball objects dictionary (create missing entries, remove old)
        desired_names = {b["name"] for b in state.get("balls", [])}
        for name in list(self.ball_images.keys()):
            if name not in desired_names:
                item_id = self.ball_images[name].get("item")
                if item_id:
                    self.canvas.delete(item_id)
                del self.ball_images[name]

        # Ensure we have entries for all balls
        for b in state.get("balls", []):
            name = b["name"]
            path = b.get("path")
            if name not in self.ball_images:
                if path and os.path.exists(path):
                    self.ball_images[name] = {
                        "pil": Image.open(path).convert("RGBA"),
                        "tk": None,
                        "item": None,
                        "x": b["x"],
                        "y": b["y"],
                    }
            else:
                # Update position (visibility handled later)
                self.ball_images[name]["x"] = b["x"]
                self.ball_images[name]["y"] = b["y"]
                # (If path changed, we could reload pil; omitted for simplicity)

        # Save scale
        self.global_scale = float(state.get("ball_scale", 1.0))

        # Render
        self._render_table()
        self._render_balls()
        # Apply visibility last
        for b in state.get("balls", []):
            self._set_ball_visibility(b["name"], b.get("visible", True))

    def _render_table(self):
        self.canvas.delete("table")
        if not self._table_pil:
            return

        cw = self.canvas.winfo_width() or 1
        ch = self.canvas.winfo_height() or 1
        iw, ih = self._table_pil.size
        scale = min(cw / iw, ch / ih)
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        pil_scaled = self._table_pil.resize(new_size, Image.LANCZOS)
        self._table_tk = ImageTk.PhotoImage(pil_scaled)
        self._table_item = self.canvas.create_image(cw // 2, ch // 2, image=self._table_tk, anchor="center", tags=("table",))

    def _render_balls(self):
        for name, rec in self.ball_images.items():
            pil_img: Image.Image = rec["pil"]
            scale = self.global_scale
            bw, bh = pil_img.size
            new_size = (max(1, int(bw * scale)), max(1, int(bh * scale)))
            resized = pil_img.resize(new_size, Image.LANCZOS)
            rec["tk"] = ImageTk.PhotoImage(resized)
            x, y = int(rec["x"]), int(rec["y"])
            if rec["item"] is None:
                rec["item"] = self.canvas.create_image(x, y, image=rec["tk"], anchor="center", tags=("ball", name))
            else:
                self.canvas.itemconfigure(rec["item"], image=rec["tk"])
                self.canvas.coords(rec["item"], x, y)

        for name, rec in self.ball_images.items():
            if rec["item"]:
                self.canvas.tag_raise(rec["item"])

    def _set_ball_visibility(self, name: str, visible: bool):
        rec = self.ball_images.get(name)
        if not rec:
            return
        item = rec.get("item")
        if not item:
            return
        self.canvas.itemconfigure(item, state="normal" if visible else "hidden")

# ----------------------------- Sidebar -----------------------------

class Sidebar(tk.Frame):
    """
    Right pane: controls for loading images, toggling ball visibility, ball size slider,
    and selecting the display for the projector window.
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

        # Projector display selection
        self.display_label = tk.Label(self, text="Projector Display", anchor="w", font=("Segoe UI", 10, "bold"))
        self.display_label.pack(fill="x", padx=8, pady=(10, 0))

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

        self.sep = tk.Label(self, text="Balls", anchor="w", font=("Segoe UI", 10, "bold"))
        self.sep.pack(fill="x", padx=8, pady=(12, 0))

        self.ball_frame = tk.Frame(self, bd=1, relief="sunken")
        self.ball_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # Subscribe to canvas updates for projector mirroring
        self.table_canvas.add_listener(self._on_canvas_update)

        self.refresh_ball_list()

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
        # Notify so projector mirrors visibility immediately
        self.table_canvas._notify()

    def load_table_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Pool Table Image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;*.tif;*.tiff")]
        )
        if path:
            self.table_canvas.load_table_image(path)

    def load_balls_dialog(self):
        paths = filedialog.askopenfilenames(
            title="Select Ball Images (PNG with transparency recommended)",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;*.tif;*.tiff")]
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
            filetypes=[("JSON files", "*.json")],
            title="Save Layout"
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
            filetypes=[("JSON files", "*.json")]
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
            self.projector = ProjectorWindow(self.master, idx)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open projector window:\n{e}")
            self.projector = None
            return

        # Push current state immediately
        self.projector.apply_state(self.table_canvas.serialize())

    def close_projector(self):
        if self.projector:
            self.projector.close()
            self.projector = None

    def _on_canvas_update(self, state: Dict):
        if self.projector:
            self.projector.apply_state(state)

# ----------------------------- Main App -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pool Table Board")
        self.geometry("1200x700")
        self.minsize(900, 500)

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

        self.table_canvas = PoolTableCanvas(self, width=1000, height=560, bd=0)
        self.table_canvas.grid(row=0, column=0, sticky="nsew")

        self.sidebar = Sidebar(self, self.table_canvas, width=320, bd=1, relief="groove")
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
