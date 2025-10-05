# Repository Guidelines

## Project Structure & Module Organization
- The application now lives under `src/` as a package:
  - `src/pool_table_board.py` bootstraps the Tkinter app and wires UI/services.
  - `src/ui/` hosts presentation classes (`canvas.py`, `sidebar.py`, `tournament_browser.py`).
  - `src/services/` contains reusable domain helpers (asset cataloguing, tournament normalization).
  - `src/infrastructure/` holds side-effect heavy components such as the projector window.
  - `src/models/` defines dataclasses for persisted layout/tournament state.
  - `src/utils/` and `src/config.py` centralise shared helpers and path constants.
- Assets and presets now live at `src/images/` and `src/layouts/`; keep new textures and layouts there so relative paths resolve.
- Leave historical snapshots (`pool_table_board_0.0.X.py`) untouched unless cutting a new tag.

## Build, Test, and Development Commands
- Create a virtual env (`python -m venv .venv`; activate via `.\\.venv\\Scripts\\Activate` on PowerShell or `source .venv/bin/activate` on Unix).
- `pip install -r requirements.txt` installs Pillow, ScreenInfo, and OpenCV; rerun after updating dependencies.
- Launch the editor from the repo root with `python -m src.pool_table_board` (preferred) or `python src/pool_table_board.py`. Running inside `src/` also works (`python pool_table_board.py`).
- Quick syntax check: `python -m compileall src`.

## Coding Style & Naming Conventions
- Adopt PEP 8 with 4-space indent and ~100 char soft wrap; keep widget wiring readable by grouping related callbacks.
- Classes such as `Sidebar`/`PoolTableCanvas` use PascalCase; functions and event handlers stay snake_case (e.g., `on_webcam_toggle`).
- Constants belong in UPPER_SNAKE_CASE near the top of their module; include compact comments around geometry or video blending to aid future editors.

## Testing Guidelines
- Manual pass: load demo assets, drag table/ball sliders, toggle webcam, and adjust opacity to confirm scaling and overlay behaviour.
- Add pytest modules under `tests/` for pure logic when you introduce them; document coverage expectations in the PR body until automation exists.
- Attach screenshots or short clips for UI changes, especially the editor/projector comparison.

## Commit & Pull Request Guidelines
- Use imperative commit subjects (`git commit -m "Add webcam overlay"`), keeping each commit scoped to one feature or fix.
- Confirm assets resolve and layouts open before pushing; avoid committing personal practice layouts.
- PRs should summarize intent, list manual test steps, reference issues, and include before/after visuals when altering the UI.

## Asset & Configuration Tips
- Store distributable textures in `src/images/` and reference them via relative paths so projector sessions remain portable.
- Use descriptive layout filenames such as `src/layouts/break_practice.json` and mention new presets in PR notes.
- Webcam failures usually stem from OS permissions or device conflicts; reinstall `opencv-python` only after confirming access rights.
