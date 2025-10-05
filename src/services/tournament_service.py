"""Tournament domain utilities and service layer."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import TOURNAMENTS_DIR
from models import ShotReference

PLAYER_TWO_TOKENS = {
    "playertwo",
    "player_two",
    "player 2",
    "player two",
    "two",
    "p2",
    "player2",
}


class TournamentServiceError(RuntimeError):
    """Base error for tournament service failures."""

    def __init__(self, message: str, *, path: Optional[Path] = None, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.path = Path(path) if path is not None else None
        if cause is not None:
            self.__cause__ = cause


class TournamentNotFoundError(TournamentServiceError):
    """Raised when a tournament file cannot be located."""


class TournamentValidationError(TournamentServiceError):
    """Raised when the requested operation fails validation."""


class TournamentPersistenceError(TournamentServiceError):
    """Raised when underlying storage cannot be written."""


class TournamentService:
    """Encapsulates tournament JSON persistence and domain mutations."""

    def __init__(self, tournaments_dir: Path | str = TOURNAMENTS_DIR) -> None:
        self.tournaments_dir = Path(tournaments_dir)
        self._document_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API - Queries
    # ------------------------------------------------------------------
    def list_tournaments(self) -> List[Dict[str, Any]]:
        """Return tournament summaries suitable for UI binding."""
        if not self.tournaments_dir.exists():
            return []

        summaries: List[Dict[str, Any]] = []
        for path in sorted(self.tournaments_dir.glob("*.json")):
            try:
                document = self._load_document(path, refresh=True)
            except TournamentServiceError:
                continue
            summaries.append(self._build_summary(path, document))
        return summaries

    def get_document(self, tournament_path: Path | str) -> Dict[str, Any]:
        """Return the normalized in-memory document for a tournament."""
        path = self._ensure_path(tournament_path)
        return self._load_document(path)

    def get_shot_layout(self, reference: ShotReference) -> Dict[str, Any]:
        """Return a deepcopy of the layout data for the referenced shot."""
        document, match, rack, shot = self._resolve_shot(reference)
        data = shot.get("data")
        if not isinstance(data, dict):
            raise TournamentValidationError("Shot layout data is missing or invalid.", path=Path(reference.tournament_path))
        return copy.deepcopy(data)

    # ------------------------------------------------------------------
    # Public API - Commands
    # ------------------------------------------------------------------
    def create_tournament(self, name: str) -> Path:
        cleaned = self._clean_name(name)
        if not cleaned:
            raise TournamentValidationError("Tournament name cannot be blank.")

        slug = cleaned.replace(" ", "_")
        path = self.tournaments_dir / f"{slug}.json"
        if path.exists():
            raise TournamentValidationError(f"A tournament named '{cleaned}' already exists.", path=path)

        document: Dict[str, Any] = {"name": cleaned, "matches": []}
        self._write_document(path, document)
        return path

    def add_match(self, tournament_path: Path | str, name: str, player_one: str, player_two: str) -> int:
        document = self.get_document(tournament_path)
        matches = document.setdefault("matches", [])
        if not isinstance(matches, list):
            raise TournamentValidationError("Tournament data is malformed.", path=self._ensure_path(tournament_path))

        cleaned_name = self._clean_name(name) or f"Match {len(matches) + 1}"
        player_one = self._clean_name(player_one) or "Player 1"
        player_two = self._clean_name(player_two) or "Player 2"

        matches.append(
            {
                "name": cleaned_name,
                "playerOne": player_one,
                "playerTwo": player_two,
                "racks": [],
            }
        )
        self._write_document(self._ensure_path(tournament_path), document)
        return len(matches) - 1

    def add_rack(
        self,
        tournament_path: Path | str,
        match_index: int,
        rack_name: str,
        breaker_name: str,
    ) -> int:
        document, match = self._resolve_match(tournament_path, match_index)
        racks = match.setdefault("racks", [])
        if not isinstance(racks, list):
            raise TournamentValidationError("Match data is malformed.", path=self._ensure_path(tournament_path))

        rack_label = self._clean_name(rack_name) or f"Rack {len(racks) + 1}"
        breaker = self._clean_name(breaker_name) or (match.get("playerOne") or "Player 1")

        racks.append({"name": rack_label, "break": breaker, "shots": []})
        self._write_document(self._ensure_path(tournament_path), document)
        return len(racks) - 1

    def remove_rack(self, tournament_path: Path | str, match_index: int, rack_index: int) -> None:
        document, match = self._resolve_match(tournament_path, match_index)
        racks = match.get("racks")
        if not isinstance(racks, list) or not (0 <= rack_index < len(racks)):
            raise TournamentValidationError("Rack reference no longer exists.", path=self._ensure_path(tournament_path))

        racks.pop(rack_index)
        self._write_document(self._ensure_path(tournament_path), document)

    def add_shot(
        self,
        tournament_path: Path | str,
        match_index: int,
        rack_index: int,
        shot_name: str,
        player_key: str,
        layout_data: Dict[str, Any],
    ) -> ShotReference:
        document, match, rack = self._resolve_rack(tournament_path, match_index, rack_index)
        shots = rack.setdefault("shots", [])
        if not isinstance(shots, list):
            raise TournamentValidationError("Rack data is malformed.", path=self._ensure_path(tournament_path))

        shot_label = self._clean_name(shot_name) or f"Shot {len(shots) + 1}"
        shooter_key = player_key if player_key in {"playerOne", "playerTwo"} else "playerOne"

        shots.append(
            {
                "name": shot_label,
                "player": shooter_key,
                "data": copy.deepcopy(layout_data),
            }
        )
        self._write_document(self._ensure_path(tournament_path), document)

        return ShotReference(str(self._ensure_path(tournament_path)), match_index, rack_index, len(shots) - 1)

    def remove_shot(self, reference: ShotReference) -> None:
        document, match, rack, shot = self._resolve_shot(reference)
        shots = rack.get("shots")
        if not isinstance(shots, list):
            raise TournamentValidationError("Rack data is malformed.", path=Path(reference.tournament_path))

        if not (0 <= reference.shot_index < len(shots)):
            raise TournamentValidationError("Shot reference no longer exists.", path=Path(reference.tournament_path))

        shots.pop(reference.shot_index)
        self._write_document(Path(reference.tournament_path), document)

    def update_shot_layout(
        self,
        reference: ShotReference,
        layout_data: Dict[str, Any],
        player_names: Optional[Tuple[str, str]] = None,
    ) -> None:
        document, match, rack, shot = self._resolve_shot(reference)
        shot["data"] = copy.deepcopy(layout_data)

        if player_names is not None:
            player_one, player_two = player_names
            match["playerOne"] = self._clean_name(player_one) or "Player 1"
            match["playerTwo"] = self._clean_name(player_two) or "Player 2"

        self._write_document(Path(reference.tournament_path), document)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_path(self, tournament_path: Path | str) -> Path:
        path = tournament_path if isinstance(tournament_path, Path) else Path(tournament_path)
        if not path.is_absolute():
            path = self.tournaments_dir / path.name
        return path

    def _load_document(self, path: Path, *, refresh: bool = False) -> Dict[str, Any]:
        key = str(path)
        if not refresh and key in self._document_cache:
            return self._document_cache[key]

        if not path.exists():
            raise TournamentNotFoundError("Tournament file not found.", path=path)

        try:
            with path.open("r", encoding="utf-8") as handle:
                document: Dict[str, Any] = json.load(handle)
        except json.JSONDecodeError as exc:
            raise TournamentValidationError("Tournament file contains invalid JSON.", path=path, cause=exc) from exc
        except OSError as exc:
            raise TournamentServiceError("Unable to read tournament file.", path=path, cause=exc) from exc

        if not isinstance(document, dict):
            raise TournamentValidationError("Tournament file must contain a JSON object.", path=path)

        normalize_tournament_document(document)
        self._document_cache[key] = document
        return document

    def _write_document(self, path: Path, document: Dict[str, Any]) -> None:
        normalize_tournament_document(document)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
        except OSError as exc:
            raise TournamentPersistenceError("Failed to save tournament file.", path=path, cause=exc) from exc

        self._document_cache[str(path)] = document

    def _build_summary(self, path: Path, document: Dict[str, Any]) -> Dict[str, Any]:
        matches_raw = document.get("matches", [])
        if not isinstance(matches_raw, list):
            matches_raw = []

        summary_matches: List[Dict[str, Any]] = []
        for match_index, match_raw in enumerate(matches_raw):
            if not isinstance(match_raw, dict):
                continue

            match_name = match_raw.get("name") or f"Match {match_index + 1}"
            player_one = match_raw.get("playerOne") or "Player 1"
            player_two = match_raw.get("playerTwo") or "Player 2"

            racks_raw = match_raw.get("racks", [])
            racks_raw = racks_raw if isinstance(racks_raw, list) else []
            summary_racks: List[Dict[str, Any]] = []

            for rack_index, rack_raw in enumerate(racks_raw):
                if not isinstance(rack_raw, dict):
                    continue

                rack_label = rack_raw.get("name") or f"Rack {rack_index + 1}"
                rack_break = rack_raw.get("break") or player_one

                shots_raw = rack_raw.get("shots", [])
                shots_raw = shots_raw if isinstance(shots_raw, list) else []
                summary_shots: List[Dict[str, Any]] = []

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
                    reference = ShotReference(str(path), match_index, rack_index, shot_index)
                    summary_shots.append(
                        {
                            "label": label,
                            "player": player_key,
                            "reference": reference,
                        }
                    )

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
            display_name = path.stem.replace("_", " ")

        return {
            "name": display_name,
            "path": str(path),
            "matches": summary_matches,
        }

    def _resolve_match(self, tournament_path: Path | str, match_index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        document = self.get_document(tournament_path)
        matches = document.get("matches")
        if not isinstance(matches, list) or not (0 <= match_index < len(matches)):
            raise TournamentValidationError("Match reference no longer exists.", path=self._ensure_path(tournament_path))

        match = matches[match_index]
        if not isinstance(match, dict):
            raise TournamentValidationError("Match data is malformed.", path=self._ensure_path(tournament_path))

        normalize_match_structure(match)
        return document, match

    def _resolve_rack(
        self,
        tournament_path: Path | str,
        match_index: int,
        rack_index: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        document, match = self._resolve_match(tournament_path, match_index)
        racks = match.get("racks")
        if not isinstance(racks, list) or not (0 <= rack_index < len(racks)):
            raise TournamentValidationError("Rack reference no longer exists.", path=self._ensure_path(tournament_path))

        rack = racks[rack_index]
        if not isinstance(rack, dict):
            raise TournamentValidationError("Rack data is malformed.", path=self._ensure_path(tournament_path))

        if "shots" not in rack or not isinstance(rack["shots"], list):
            rack["shots"] = []

        return document, match, rack

    def _resolve_shot(
        self,
        reference: ShotReference,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        document, match, rack = self._resolve_rack(
            reference.tournament_path,
            reference.match_index,
            reference.rack_index,
        )
        shots = rack.get("shots")
        if not isinstance(shots, list) or not (0 <= reference.shot_index < len(shots)):
            raise TournamentValidationError("Shot reference no longer exists.", path=Path(reference.tournament_path))

        shot = shots[reference.shot_index]
        if not isinstance(shot, dict):
            raise TournamentValidationError("Shot data is malformed.", path=Path(reference.tournament_path))

        return document, match, rack, shot

    @staticmethod
    def _clean_name(value: str) -> str:
        return " ".join((value or "").strip().split())


# ----------------------------------------------------------------------
# Supporting helpers retained for compatibility.
# ----------------------------------------------------------------------

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


__all__ = [
    "TournamentService",
    "TournamentServiceError",
    "TournamentNotFoundError",
    "TournamentValidationError",
    "TournamentPersistenceError",
    "infer_players_from_folder_name",
    "normalize_match_structure",
    "normalize_tournament_document",
    "resolve_player_key",
    "resolve_player_key_with_names",
]
