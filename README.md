# Pool Table Board

An interactive Python application for visualizing pool tables, balls, and practice scenarios.  
Supports Windows, Ubuntu Desktop, and macOS.

## Features

- **Table & Balls**
  - Load a pool table image as background.
  - Load multiple pool ball images (PNG recommended).
  - Drag balls around, toggle visibility via checkboxes.
  - Adjustable ball size slider (20%–200%).

- **Drawing Tools**
  - Draw lines or arrows directly on the table.
  - Choose stroke color and width.
  - Undo last drawing or clear all drawings.

- **Layouts**
  - Save and load layouts to JSON (balls + drawings + sizes).

- **Projector Mode**
  - Select a display/monitor from dropdown.
  - Open a borderless, fullscreen window on that display.
  - Table, balls, and drawings are mirrored live using table-relative coordinates.

## Requirements

- Python 3.8+
- Tkinter (comes with Python, install with `sudo apt install python3-tk` if missing on Ubuntu)
- Pip packages:
  ```bash
  pip install pillow screeninfo
  ```

## Usage

Run the application:

```bash
python3 pool_table_board.py
```

### Steps
1. Load a table image (`File → Load Table Image…`).
2. Load ball images (`File → Load Ball Images…`). Drag to place.
3. Use checkboxes to toggle ball visibility.
4. Adjust **Ball Size (%)** with the slider.
5. Switch to **Drawing Tools** to add lines/arrows.
6. Save or load layouts to preserve work.
7. Select a **Projector Display** and open a projector window.

## Multi-Monitor Setup on Ubuntu

The app uses the `screeninfo` library to detect displays.  
Displays are listed in the sidebar like:

```
0: 1920x1080 @ (0,0)
1: 1280x1024 @ (1920,0)
```

Choose the correct display and click **Open Projector Window**.

## Development

This project is structured as a single script (`pool_table_board.py`) with:
- Object-oriented design (`BallSprite`, `PoolTableCanvas`, `DrawingLayer`, etc.)
- Table-relative coordinate system for cross-display consistency
- Sidebar UI for controls

## License

MIT License
