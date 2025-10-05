#!/usr/bin/env python3
"""
Pool Table Board (Presenter Mode) - Table-relative & Drawing Tools

Launch the Tk application after migrating legacy tournament data to JSON.
"""

from legacy import convert_legacy_tournaments
from ui.app import App


def main() -> None:
    convert_legacy_tournaments()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
