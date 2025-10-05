from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable, Dict, List, Optional

from services import LayoutService, list_ball_assets
from infrastructure import ProjectorWindow
from ui.canvas import PoolTableCanvas


try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None

class Sidebar(tk.Frame):
    """
    Right pane: image loading, visibility toggles, ball size slider,
    projector display selection, and drawing tools.
    """
    def __init__(self, master, table_canvas: PoolTableCanvas, layout_service: LayoutService, **kwargs):
        super().__init__(master, **kwargs)
        self.layout_service = layout_service
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







