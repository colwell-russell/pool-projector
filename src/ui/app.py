from __future__ import annotations

import copy
import os
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    BALL_IMAGES_DIR,
    BULLSEYE_TARGET_IMAGE,
    IMAGE_FILETYPES,
    JSON_FILETYPES,
    LAYOUTS_DIR,
    TABLE_IMAGES_DIR,
)
from models import ShotReference
from services import (
    LayoutService,
    TournamentService,
    TournamentServiceError,
    infer_players_from_folder_name,
    list_ball_assets,
    normalize_match_structure,
    resolve_player_key,
    resolve_player_key_with_names,
)
from ui.canvas import PoolTableCanvas
from ui.sidebar import Sidebar
from ui.tournament_browser import TournamentBrowser
from infrastructure import ProjectorWindow

try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pool Table Board")
        self.geometry("1300x750")
        self.minsize(980, 560)

        self.layout_service = LayoutService()
        self.tournament_service = TournamentService()

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
        self._tournament_editor_ball_remove_var: Optional[tk.StringVar] = None
        self._tournament_editor_ball_remove_combo: Optional[ttk.Combobox] = None
        self._tournament_editor_remove_ball_btn: Optional[tk.Button] = None
        self._tournament_editor_remove_rack_btn: Optional[tk.Button] = None
        self._tournament_editor_remove_shot_btn: Optional[tk.Button] = None

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

        self.sidebar = Sidebar(self, self.table_canvas, self.layout_service, width=340, bd=1, relief="groove")
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
        self._refresh_tournament_editor_ball_names()

    def _add_bullseye_target_to_shot(self):
        if self._tournament_editor_canvas is None:
            messagebox.showinfo("Add Target", "Open or create a shot before adding targets.")
            return

        target_path = BULLSEYE_TARGET_IMAGE
        if not target_path.exists():
            messagebox.showerror("Add Target", "Bullseye asset is missing. Restore 'src/images/balls/BullseyeTarget.png'.")
            return

        base_name = "Bullseye"
        existing_names = {ball.name for ball in self._tournament_editor_canvas.balls}
        candidate = base_name
        index = 2
        while candidate in existing_names:
            candidate = f"{base_name}_{index}"
            index += 1

        try:
            self._tournament_editor_canvas.add_ball(candidate, str(target_path))
        except Exception as exc:
            messagebox.showerror("Add Target", f"Failed to add target:\n{exc}")
            return

        self._tournament_editor_canvas.draw_layer.rerender_all()
        self._refresh_tournament_editor_ball_names()

    def _remove_ball_from_shot(self):
        if self._tournament_editor_canvas is None:
            messagebox.showinfo("Remove Ball", "Open or create a shot before removing balls.")
            return

        if not self._tournament_editor_canvas.balls:
            messagebox.showinfo("Remove Ball", "No balls to remove.")
            self._refresh_tournament_editor_ball_names()
            return

        var = getattr(self, "_tournament_editor_ball_remove_var", None)
        selected_name = ""
        if isinstance(var, tk.StringVar):
            selected_name = var.get().strip()
        if not selected_name:
            messagebox.showinfo("Remove Ball", "Select a ball to remove.")
            return

        if not self._tournament_editor_canvas.remove_ball(selected_name):
            messagebox.showerror("Remove Ball", f"Ball '{selected_name}' could not be found.")
            self._refresh_tournament_editor_ball_names()
            return

        self._tournament_editor_canvas.draw_layer.rerender_all()
        self._refresh_tournament_editor_ball_names()

    def _refresh_tournament_editor_ball_names(self):
        combo = getattr(self, "_tournament_editor_ball_remove_combo", None)
        var = getattr(self, "_tournament_editor_ball_remove_var", None)
        btn = getattr(self, "_tournament_editor_remove_ball_btn", None)

        if combo is None or var is None:
            return
        try:
            exists = bool(combo.winfo_exists())
        except tk.TclError:
            return
        if not exists:
            return

        active = (
            self._tournament_editor_canvas is not None
            and self._tournament_editor_current_ref is not None
        )

        if not active:
            names = []
        else:
            names = [ball.name for ball in self._tournament_editor_canvas.balls]

        combo.configure(values=names)
        if names:
            if var.get() not in names:
                var.set(names[0])
            combo.configure(state="readonly")
            if btn is not None:
                btn.configure(state=tk.NORMAL)
        else:
            var.set("")
            combo.configure(state="disabled")
            if btn is not None:
                btn.configure(state=tk.DISABLED)

    def _prompt_player_selection(
        self,
        title: str,
        message: str,
        player_one: Optional[str],
        player_two: Optional[str],
        parent: Optional[tk.Misc] = None,
        default_key: str = "playerOne",
    ) -> Optional[str]:
        if parent is not None:
            parent_window: tk.Misc = parent
        elif (
            self._tournament_editor_window is not None
            and self._tournament_editor_window.winfo_exists()
        ):
            parent_window = self._tournament_editor_window
        else:
            parent_window = self

        display_one = (player_one or "Player 1").strip() or "Player 1"
        display_two = (player_two or "Player 2").strip() or "Player 2"
        options: List[Tuple[str, str]] = [
            (f"{display_one} (Player 1)", "playerOne"),
            (f"{display_two} (Player 2)", "playerTwo"),
        ]
        valid_keys = {value for _, value in options}
        default = default_key if default_key in valid_keys else options[0][1]

        class _PlayerSelectionDialog(simpledialog.Dialog):
            def body(self, master):
                tk.Label(master, text=message, anchor="w").pack(fill="x", padx=8, pady=(8, 4))
                self._labels = [label for label, _ in options]
                self._values = {label: value for label, value in options}
                start_label = next((label for label, value in options if value == default), self._labels[0])
                self._var = tk.StringVar(value=start_label)
                combo = ttk.Combobox(master, state="readonly", values=self._labels, textvariable=self._var)
                combo.pack(fill="x", padx=8, pady=(0, 8))
                combo.focus_set()
                return combo

            def apply(self):
                selection_var = getattr(self, "_var", None)
                labels_map = getattr(self, "_values", {})
                chosen = selection_var.get() if selection_var is not None else None
                self.result = labels_map.get(chosen)

        dialog = _PlayerSelectionDialog(parent_window, title)
        return getattr(dialog, "result", None)

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
        self._tournaments_summary = self._load_tournaments_from_disk()
        self._match_player_names.clear()

        if hasattr(self, "tournament_browser"):
            self.tournament_browser.refresh(self._tournaments_summary, selected=self._current_shot_reference)

        self._build_tournaments_menu()
        self._refresh_tournament_editor()
        self._update_tournament_ball_catalog()
        self._refresh_tournament_editor_ball_names()

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
        try:
            return self.tournament_service.list_tournaments()
        except TournamentServiceError as exc:
            messagebox.showerror("Tournaments", str(exc))
            return []

    def _get_tournament_document(self, path: str) -> Optional[Dict]:
        try:
            return self.tournament_service.get_document(path)
        except TournamentServiceError:
            return None

    def _get_shot_data(self, reference: ShotReference) -> Optional[Dict]:
        try:
            return self.tournament_service.get_shot_layout(reference)
        except TournamentServiceError as exc:
            messagebox.showerror("Error", str(exc))
            return None

    def _update_shot_data(
        self,
        reference: ShotReference,
        data: Dict,
        player_names: Optional[Tuple[str, str]] = None,
    ) -> bool:
        try:
            self.tournament_service.update_shot_layout(reference, data, player_names)
        except TournamentServiceError as exc:
            messagebox.showerror("Save Shot", str(exc))
            return False

        canonical_names: Optional[Tuple[str, str]] = None
        if player_names is not None:
            document = self._get_tournament_document(reference.tournament_path)
            if document is not None:
                matches = document.get("matches") or []
                try:
                    match = matches[reference.match_index]
                except (IndexError, TypeError):
                    match = None
                if isinstance(match, dict):
                    player_one = (match.get("playerOne") or "Player 1").strip() or "Player 1"
                    player_two = (match.get("playerTwo") or "Player 2").strip() or "Player 2"
                    canonical_names = (player_one, player_two)
            if canonical_names is None:
                canonical_names = (player_names[0], player_names[1])

        matches_summary: List[Dict] = []
        for tournament in self._tournaments_summary:
            if tournament.get("path") != reference.tournament_path:
                continue
            matches_summary = tournament.get("matches", [])
            if canonical_names is not None and 0 <= reference.match_index < len(matches_summary):
                match_summary = matches_summary[reference.match_index]
                match_summary["player_one"], match_summary["player_two"] = canonical_names
            break

        if canonical_names is not None:
            match_key = (reference.tournament_path, reference.match_index)
            self._match_player_names[match_key] = canonical_names
            if self._tournament_editor_tree is not None:
                for item_id, meta in self._tournament_editor_tree_meta.items():
                    if meta.get("type") != "match":
                        continue
                    if (
                        meta.get("tournament_path") == reference.tournament_path
                        and int(meta.get("match_index", -1)) == reference.match_index
                    ):
                        meta["player_one"], meta["player_two"] = canonical_names
                        if 0 <= reference.match_index < len(matches_summary):
                            meta["racks"] = matches_summary[reference.match_index].get("racks", [])
                        try:
                            new_text = f"{meta.get('match_name', 'Match')} - {canonical_names[0]} vs {canonical_names[1]}"
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
        window.geometry("1524x700")
        window.minsize(1400, 600)

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

        controls_frame = tk.Frame(shots_inner)
        controls_frame.pack(side="left", fill="y", padx=(8, 0))

        tk.Label(controls_frame, text="Rack Actions", anchor="w").pack(fill="x")
        add_rack_btn = tk.Button(
            controls_frame,
            text="Add Rack",
            command=self._handle_editor_add_rack,
        )
        add_rack_btn.pack(fill="x", pady=(2, 4))

        remove_rack_btn = tk.Button(
            controls_frame,
            text="Remove Rack",
            command=self._remove_selected_rack,
            state=tk.DISABLED,
        )
        remove_rack_btn.pack(fill="x", pady=(0, 12))

        tk.Label(controls_frame, text="Shot Actions", anchor="w").pack(fill="x")
        new_shot_btn = tk.Button(
            controls_frame,
            text="Add Shot",
            command=self._prompt_new_shot,
        )
        new_shot_btn.pack(fill="x", pady=(2, 4))

        remove_shot_btn = tk.Button(
            controls_frame,
            text="Remove Shot",
            command=self._remove_selected_shot,
            state=tk.DISABLED,
        )
        remove_shot_btn.pack(fill="x", pady=(0, 4))

        open_shot_btn = tk.Button(
            controls_frame,
            text="Open Selected Shot",
            command=self._open_selected_match_shot,
        )
        open_shot_btn.pack(fill="x", pady=(0, 12))

        tk.Label(controls_frame, text="Ball Tools", anchor="w").pack(fill="x")
        self._tournament_editor_ball_var = tk.StringVar()
        self._tournament_editor_ball_combo = ttk.Combobox(
            controls_frame,
            state="readonly",
            textvariable=self._tournament_editor_ball_var,
            values=[],
            width=16,
        )
        self._tournament_editor_ball_combo.pack(fill="x", pady=(2, 4))
        self._tournament_editor_ball_catalog: List[Tuple[str, str]] = []

        add_ball_btn = tk.Button(
            controls_frame,
            text="Add Ball",
            command=self._prompt_add_ball_to_shot,
        )
        add_ball_btn.pack(fill="x", pady=(0, 4))

        add_target_btn = tk.Button(
            controls_frame,
            text="Add Bullseye Target",
            command=self._add_bullseye_target_to_shot,
        )
        add_target_btn.pack(fill="x", pady=(0, 4))

        add_drawing_btn = tk.Button(
            controls_frame,
            text="Add Drawing",
            command=self._prompt_add_drawing_to_shot,
        )
        add_drawing_btn.pack(fill="x", pady=(0, 8))

        tk.Label(controls_frame, text="Active Balls", anchor="w").pack(fill="x")
        self._tournament_editor_ball_remove_var = tk.StringVar()
        self._tournament_editor_ball_remove_combo = ttk.Combobox(
            controls_frame,
            state="disabled",
            textvariable=self._tournament_editor_ball_remove_var,
            values=[],
            width=16,
        )
        self._tournament_editor_ball_remove_combo.pack(fill="x", pady=(2, 4))

        remove_ball_btn = tk.Button(
            controls_frame,
            text="Remove Ball",
            command=self._remove_ball_from_shot,
            state=tk.DISABLED,
        )
        remove_ball_btn.pack(fill="x")

        self._tournament_editor_remove_ball_btn = remove_ball_btn

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
            self._tournament_editor_ball_remove_combo = None
            self._tournament_editor_ball_remove_var = None
            self._tournament_editor_remove_ball_btn = None
            self._tournament_editor_remove_rack_btn = None
            self._tournament_editor_remove_shot_btn = None
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
        self._tournament_editor_remove_rack_btn = remove_rack_btn
        self._tournament_editor_remove_shot_btn = remove_shot_btn
        self._tournament_editor_save_btn = save_btn

        self._hide_match_detail()
        self._refresh_tournament_editor()
        self._update_tournament_ball_catalog()
        self._refresh_tournament_editor_ball_names()

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
        self._set_editor_remove_rack_enabled(False)
        self._set_editor_remove_shot_enabled(False)

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
        self._set_editor_remove_rack_enabled(False)
        self._set_editor_remove_shot_enabled(False)
        self._refresh_tournament_editor_ball_names()

    def _populate_match_racks(self, racks: List[Dict], tournament_path: str, match_index: int):
        self._tournament_editor_rack_items = []
        if self._tournament_editor_rack_listbox is None:
            return

        listbox = self._tournament_editor_rack_listbox
        listbox.delete(0, "end")
        self._tournament_editor_active_rack = None
        self._set_editor_remove_rack_enabled(False)

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
        self._set_editor_remove_shot_enabled(False)

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

        self._update_remove_shot_button_state()

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
            self._update_remove_rack_button_state()
            return

        self._tournament_editor_active_rack = (tournament_path, match_index, rack_index)
        self._populate_rack_shots(racks, tournament_path, match_index, rack_index)
        self._update_remove_rack_button_state()

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

    def _set_editor_remove_rack_enabled(self, enabled: bool):
        btn = self._tournament_editor_remove_rack_btn
        if btn is None:
            return
        new_state = tk.NORMAL if enabled else tk.DISABLED
        if str(btn["state"]) != str(new_state):
            btn.configure(state=new_state)

    def _set_editor_remove_shot_enabled(self, enabled: bool):
        btn = self._tournament_editor_remove_shot_btn
        if btn is None:
            return
        new_state = tk.NORMAL if enabled else tk.DISABLED
        if str(btn["state"]) != str(new_state):
            btn.configure(state=new_state)

    def _update_remove_rack_button_state(self):
        btn = self._tournament_editor_remove_rack_btn
        if btn is None:
            return
        listbox = self._tournament_editor_rack_listbox
        enabled = False
        if (
            self._tournament_editor_active_match is not None
            and listbox is not None
            and self._tournament_editor_rack_items
        ):
            try:
                selection = listbox.curselection()
            except tk.TclError:
                selection = ()
            if selection:
                index = selection[0]
                enabled = 0 <= index < len(self._tournament_editor_rack_items)
        self._set_editor_remove_rack_enabled(enabled)

    def _update_remove_shot_button_state(self):
        btn = self._tournament_editor_remove_shot_btn
        if btn is None:
            return
        listbox = self._tournament_editor_shot_listbox
        enabled = False
        if (
            self._tournament_editor_active_rack is not None
            and listbox is not None
            and self._tournament_editor_shot_items
        ):
            try:
                selection = listbox.curselection()
            except tk.TclError:
                selection = ()
            if selection:
                index = selection[0]
                enabled = 0 <= index < len(self._tournament_editor_shot_items)
        self._set_editor_remove_shot_enabled(enabled)

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
            self._set_editor_remove_shot_enabled(False)
            return
        try:
            selection = self._tournament_editor_shot_listbox.curselection()
        except tk.TclError:
            selection = ()
        if not selection:
            self._set_editor_remove_shot_enabled(False)
            return
        index = selection[0]
        if index >= len(self._tournament_editor_shot_items):
            self._set_editor_remove_shot_enabled(False)
            return
        self._set_editor_remove_shot_enabled(True)
        label, reference = self._tournament_editor_shot_items[index]
        if not isinstance(reference, ShotReference):
            self._set_editor_remove_shot_enabled(False)
            return
        data = self._get_shot_data(reference)
        if data is None:
            self._set_editor_remove_shot_enabled(False)
            return
        if not self._apply_layout_to_table_canvas(data):
            self._set_editor_remove_shot_enabled(False)
            return
        self._apply_layout_to_editor_canvas(data)
        self._tournament_editor_current_ref = reference
        self._current_shot_reference = reference
        self._refresh_tournament_editor_ball_names()
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
            data = self.layout_service.read(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load shot:\n{exc}")
            return None
        return self.layout_service.sanitize(data)

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
        else:
            self._refresh_tournament_editor_ball_names()

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

        try:
            tournament_path = self.tournament_service.create_tournament(cleaned)
        except TournamentServiceError as exc:
            messagebox.showerror("New Tournament", str(exc))
            return

        canonical_name = cleaned
        document = self._get_tournament_document(str(tournament_path))
        if isinstance(document, dict):
            stored_name = document.get("name")
            if isinstance(stored_name, str) and stored_name.strip():
                canonical_name = stored_name

        self.refresh_tournaments()
        self._refresh_tournament_editor(select_name=canonical_name)

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

        try:
            new_match_index = self.tournament_service.add_match(
                tournament_path,
                match_name,
                player_one,
                player_two,
            )
        except TournamentServiceError as exc:
            messagebox.showerror("New Match", str(exc))
            return

        self._current_shot_reference = None

        document = self._get_tournament_document(tournament_path)
        if isinstance(document, dict):
            matches = document.get("matches") or []
            if 0 <= new_match_index < len(matches):
                stored_match = matches[new_match_index]
                if isinstance(stored_match, dict):
                    self._match_player_names[(tournament_path, new_match_index)] = (
                        (stored_match.get("playerOne") or player_one).strip() or "Player 1",
                        (stored_match.get("playerTwo") or player_two).strip() or "Player 2",
                    )

        self.refresh_tournaments()
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
        if parent is not None:
            parent_window = parent
        elif (
            self._tournament_editor_window is not None
            and self._tournament_editor_window.winfo_exists()
        ):
            parent_window = self._tournament_editor_window
        else:
            parent_window = self

        rack_name_input = simpledialog.askstring(
            "Add Rack",
            "Rack name:",
            initialvalue=default_name,
            parent=parent_window,
        )
        if rack_name_input is None:
            return None
        rack_name = " ".join(rack_name_input.strip().split()) or default_name

        player_one_name = (match.get("playerOne") or "Player 1").strip() or "Player 1"
        player_two_name = (match.get("playerTwo") or "Player 2").strip() or "Player 2"

        breaker_key = self._prompt_player_selection(
            "Add Rack",
            "Select the breaker:",
            player_one_name,
            player_two_name,
            parent_window,
            default_key="playerOne",
        )
        if breaker_key is None:
            return None

        breaker_name = player_two_name if breaker_key == "playerTwo" else player_one_name

        try:
            new_index = self.tournament_service.add_rack(tournament_path, match_index, rack_name, breaker_name)
        except TournamentServiceError as exc:
            messagebox.showerror("Add Rack", str(exc))
            return None

        self._current_shot_reference = None
        self.refresh_tournaments()
        return new_index

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

    def _remove_selected_rack(self):
        if self._tournament_editor_window is None:
            return
        match_key = self._tournament_editor_active_match
        if match_key is None:
            messagebox.showinfo("Remove Rack", "Select a match before removing a rack.")
            return
        listbox = self._tournament_editor_rack_listbox
        if listbox is None:
            return
        try:
            selection = listbox.curselection()
        except tk.TclError:
            selection = ()
        if not selection:
            messagebox.showinfo("Remove Rack", "Select a rack to remove.")
            return
        index = selection[0]
        if index >= len(self._tournament_editor_rack_items):
            messagebox.showinfo("Remove Rack", "Select a rack to remove.")
            return
        rack_label, rack_index = self._tournament_editor_rack_items[index]
        if not messagebox.askyesno("Remove Rack", f"Remove rack '{rack_label}' and all shots?"):
            return
        tournament_path, match_index = match_key

        try:
            self.tournament_service.remove_rack(tournament_path, match_index, rack_index)
        except TournamentServiceError as exc:
            messagebox.showerror("Remove Rack", str(exc))
            return

        def adjust_reference(ref):
            if ref is None:
                return None
            if ref.tournament_path != tournament_path or ref.match_index != match_index:
                return ref
            if ref.rack_index == rack_index:
                return None
            if ref.rack_index > rack_index:
                return ShotReference(ref.tournament_path, ref.match_index, ref.rack_index - 1, ref.shot_index)
            return ref

        self._current_shot_reference = adjust_reference(self._current_shot_reference)
        editor_ref = adjust_reference(self._tournament_editor_current_ref)
        if editor_ref is None:
            self._tournament_editor_current_ref = None
            self._update_save_button_state(False)
            self._refresh_tournament_editor_ball_names()
        else:
            self._tournament_editor_current_ref = editor_ref

        document = self._get_tournament_document(tournament_path)
        remaining_racks: List[Dict[str, Any]] = []
        if isinstance(document, dict):
            matches = document.get("matches") or []
            if 0 <= match_index < len(matches):
                match = matches[match_index]
                if isinstance(match, dict):
                    normalize_match_structure(match)
                    racks = match.get("racks")
                    if isinstance(racks, list):
                        remaining_racks = racks

        new_focus_index = min(rack_index, max(len(remaining_racks) - 1, 0)) if remaining_racks else None
        target_shots = []
        if remaining_racks and new_focus_index is not None and new_focus_index >= 0:
            target_rack = remaining_racks[new_focus_index]
            if isinstance(target_rack, dict):
                potential = target_rack.get("shots")
                if isinstance(potential, list):
                    target_shots = potential
        new_shot_ref = None
        if target_shots:
            new_shot_ref = ShotReference(tournament_path, match_index, new_focus_index, 0)

        self._set_editor_remove_rack_enabled(False)
        self._set_editor_remove_shot_enabled(False)
        self.refresh_tournaments()

        def refocus():
            self._focus_editor_match(tournament_path, match_index)
            if remaining_racks and new_focus_index is not None and new_focus_index >= 0:
                self.after(50, lambda: self._focus_editor_rack(tournament_path, match_index, new_focus_index))
                if new_shot_ref is not None:
                    self.after(100, lambda: self._focus_editor_shot(new_shot_ref))
        self.after(0, refocus)

    def _remove_selected_shot(self):
        if self._tournament_editor_window is None:
            return
        active = self._tournament_editor_active_rack
        if active is None:
            messagebox.showinfo("Remove Shot", "Select a rack before removing a shot.")
            return
        listbox = self._tournament_editor_shot_listbox
        if listbox is None:
            return
        try:
            selection = listbox.curselection()
        except tk.TclError:
            selection = ()
        if not selection:
            messagebox.showinfo("Remove Shot", "Select a shot to remove.")
            return
        index = selection[0]
        if index >= len(self._tournament_editor_shot_items):
            messagebox.showinfo("Remove Shot", "Select a shot to remove.")
            return
        label, reference = self._tournament_editor_shot_items[index]
        if not isinstance(reference, ShotReference):
            messagebox.showinfo("Remove Shot", "Select a shot to remove.")
            return
        if not messagebox.askyesno("Remove Shot", f"Remove shot '{label}'?"):
            return

        try:
            self.tournament_service.remove_shot(reference)
        except TournamentServiceError as exc:
            messagebox.showerror("Remove Shot", str(exc))
            return

        def adjust_reference(ref):
            if ref is None:
                return None
            if ref.tournament_path != reference.tournament_path or ref.match_index != reference.match_index or ref.rack_index != reference.rack_index:
                return ref
            if ref.shot_index == reference.shot_index:
                return None
            if ref.shot_index > reference.shot_index:
                return ShotReference(ref.tournament_path, ref.match_index, ref.rack_index, ref.shot_index - 1)
            return ref

        self._current_shot_reference = adjust_reference(self._current_shot_reference)
        editor_ref = adjust_reference(self._tournament_editor_current_ref)
        if editor_ref is None:
            self._tournament_editor_current_ref = None
            self._update_save_button_state(False)
            self._refresh_tournament_editor_ball_names()
        else:
            self._tournament_editor_current_ref = editor_ref

        document = self._get_tournament_document(reference.tournament_path)
        remaining_shots: List[Dict[str, Any]] = []
        if isinstance(document, dict):
            matches = document.get("matches") or []
            if 0 <= reference.match_index < len(matches):
                match = matches[reference.match_index]
                if isinstance(match, dict):
                    normalize_match_structure(match)
                    racks = match.get("racks")
                    if isinstance(racks, list) and 0 <= reference.rack_index < len(racks):
                        rack = racks[reference.rack_index]
                        if isinstance(rack, dict):
                            shots = rack.get("shots")
                            if isinstance(shots, list):
                                remaining_shots = shots

        new_shot_index = min(reference.shot_index, max(len(remaining_shots) - 1, 0)) if remaining_shots else None

        self._set_editor_remove_shot_enabled(False)
        self.refresh_tournaments()

        def refocus():
            self._focus_editor_match(reference.tournament_path, reference.match_index)
            self._focus_editor_rack(reference.tournament_path, reference.match_index, reference.rack_index)
            if remaining_shots and new_shot_index is not None and new_shot_index >= 0:
                new_ref = ShotReference(reference.tournament_path, reference.match_index, reference.rack_index, new_shot_index)
                self.after(50, lambda: self._focus_editor_shot(new_ref))
        self.after(0, refocus)

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
            (match.get("playerOne") or "Player 1").strip() or "Player 1",
            (match.get("playerTwo") or "Player 2").strip() or "Player 2",
        )

        shooter_key = self._prompt_player_selection(
            "Add Shot",
            "Select the shooter:",
            current_players[0],
            current_players[1],
            self._tournament_editor_window,
            default_key="playerOne",
        )
        if shooter_key is None:
            return
        player_key = shooter_key

        
        base_state = {
            "table": os.path.join(TABLE_IMAGES_DIR, "Table.png"),
            "table_scale": 1.0,
            "ball_scale": 0.33,
            "table_offset": {"x": 0.0, "y": 0.0},
            "table_rect": [0, 158, 720, 444],
            "webcam": {"enabled": False, "opacity": 0.5, "source": 0},
            "balls": [],
            "drawings": [],
        }
        if(len(shots) > 0):
            last_shot = shots[len(shots) - 1]
            base_state['balls'] = last_shot['data']['balls']

        try:
            new_reference = self.tournament_service.add_shot(
                tournament_path,
                match_index,
                rack_index,
                shot_name,
                player_key,
                base_state,
            )
        except TournamentServiceError as exc:
            messagebox.showerror("Add Shot", str(exc))
            return

        self._current_shot_reference = new_reference

        document = self._get_tournament_document(tournament_path)
        if isinstance(document, dict):
            matches = document.get("matches") or []
            if 0 <= match_index < len(matches):
                latest_match = matches[match_index]
                if isinstance(latest_match, dict):
                    self._match_player_names[(tournament_path, match_index)] = (
                        (latest_match.get("playerOne") or current_players[0]).strip() or "Player 1",
                        (latest_match.get("playerTwo") or current_players[1]).strip() or "Player 2",
                    )

        self.refresh_tournaments()

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












