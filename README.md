# Pool Table Board

An interactive Python application for visualizing pool tables, balls, and practice scenarios.  
Supports Windows, Ubuntu Desktop, and macOS.

## Features

- **Table & Balls**
  - Load a pool table image as background.
  - Load multiple pool ball images (PNG recommended) and drag them into position.
  - Toggle ball visibility and adjust the global ball size (20%–200%).

- **Drawing Tools**
  - Draw lines or arrows directly on the table.
  - Choose stroke color and width, undo the last stroke, or clear everything.

- **Layouts**
  - Save and load layouts to JSON (balls + drawings + sizes) for repeatable drills.

- **Projector Mode**
  - Select an external display/monitor and open a borderless, fullscreen mirror.
  - Table, balls, and drawings stay in sync using table-relative coordinates.

## Project Layout

The application now ships as a proper package under `src/`:

```
src/
+-- pool_table_board.py      # Entry point / App composition
+-- config.py                # Paths, file dialogs, directory bootstrap
+-- legacy.py                # One-time migration helpers
+-- models/                  # Dataclasses (BallState, DrawingState, ShotReference)
+-- services/                # Reusable domain helpers (assets, tournaments)
+-- ui/                      # Tkinter presentation (canvas, sidebar, tournament browser)
+-- infrastructure/          # Projector window and other side-effect components
+-- utils/                   # Generic helpers (clamp, etc.)
+-- images/                  # Ball/table textures and sample layouts
+-- layouts/                 # Saved layout presets (plus Tournaments JSON docs)
```

Assets and layouts should be stored under `src/images/` and `src/layouts/` respectively so relative paths resolve when the editor or projector loads them.

## Requirements

- Python 3.8+
- Tkinter (bundled with Python; on Ubuntu install with `sudo apt install python3-tk`)
- Pip packages:
  ```bash
  pip install -r requirements.txt
  ```

## Setup & Usage

1. Create/activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate
   # or Unix shells
   source .venv/bin/activate
   ```
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the editor from the repo root:
   ```bash
   python -m src.pool_table_board
   ```
   (Alternatively: `python src/pool_table_board.py`, or from inside `src/`: `python pool_table_board.py`.)

### Typical Workflow
1. Load a table image (`File ? Load Table Image...`). Dialogs default to `src/images/`.
2. Add balls from the library and drag them into position.
3. Adjust visibility, table offsets, and ball size from the sidebar.
4. Use drawing tools for lines/arrows.
5. Save layouts (`File ? Save Layout...`) to `src/layouts/` for reuse.
6. Open a projector window, select a display, and close it with `Esc` if needed.

## Development Tips

- Quick syntax check: `python -m compileall src`
- Most logic is now testable without Tk; add new unit tests under `tests/` for services/models.
- Avoid referencing absolute paths; rely on `config.py` helpers to keep projector sessions portable.

## License

MIT License
