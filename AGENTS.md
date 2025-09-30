# Repository Guidelines

## Project Structure & Module Organization
- `pool_table_board.py` is the main Tkinter app; keep helpers close to their call sites or split into modules when they exceed ~200 lines.
- Versioned snapshots `pool_table_board_0.0.X.py` are release references—leave untouched unless cutting a new tag.
- `images/` holds table/ball textures, `layouts/` stores JSON presets from Save Layout, and `.venv/` stays local; installable deps live in `requirements.txt`.

## Build, Test, and Development Commands
- Create a virtual env (`python -m venv .venv`; activate via `.\\.venv\\Scripts\\Activate` on PowerShell or `source .venv/bin/activate` on Unix).
- `pip install -r requirements.txt` installs Pillow, ScreenInfo, and OpenCV; rerun after updating dependencies.
- Launch the editor with `python pool_table_board.py` (or `py` on Windows); verify table, ball, and webcam controls before publishing.
- Quick syntax check: `py -m compileall pool_table_board.py`.

## Coding Style & Naming Conventions
- Adopt PEP 8 with 4-space indent and ~100 char soft wrap; keep widget wiring readable by grouping related callbacks.
- Classes such as `Sidebar`/`PoolTableCanvas` use PascalCase; functions and event handlers stay snake_case (e.g., `on_webcam_toggle`).
- Constants belong in UPPER_SNAKE_CASE near the top; include compact comments around geometry or video blending to aid future editors.

## Testing Guidelines
- Manual pass: load demo assets, drag table/ball sliders, toggle webcam, and adjust opacity to confirm scaling and overlay behaviour.
- Add pytest modules under `tests/` for pure logic when you introduce them; document coverage expectations in the PR body until automation exists.
- Attach screenshots or short clips for UI changes, with especially the editor/projector comparison.

## Commit & Pull Request Guidelines
- Use imperative commit subjects (`git commit -m "Add webcam overlay"`), keeping each commit scoped to one feature or fix.
- Confirm assets resolve and layouts open before pushing; avoid committing personal practice layouts.
- PRs should summarize intent, list manual test steps, reference issues, and include before/after visuals when altering the UI.

## Asset & Configuration Tips
- Store distributable textures in `images/` and reference them via relative paths so projector sessions remain portable.
- Use descriptive layout filenames such as `layouts/break_practice.json` and mention new presets in PR notes.
- Webcam failures usually stem from OS permissions or device conflicts; reinstall `opencv-python` only after confirming access rights.