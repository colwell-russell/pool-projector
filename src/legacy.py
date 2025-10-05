from __future__ import annotations

import json
import os
import shutil
from typing import List

from config import TOURNAMENTS_DIR
from services import infer_players_from_folder_name


def convert_legacy_tournaments() -> None:
    """Convert folder-based tournaments into consolidated JSON files."""
    try:
        entries: List[str] = sorted(os.listdir(TOURNAMENTS_DIR))
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
                match_shots.append({"name": shot_name, "player": "playerOne", "data": shot_data})

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

        if not matches_data:
            continue

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


__all__ = ["convert_legacy_tournaments"]
