from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from models import ShotReference

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


