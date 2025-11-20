from __future__ import annotations

from src.interfaces.gui import GUIApplication


def main() -> None:
    app = GUIApplication()
    raise SystemExit(app.run())


if __name__ == "__main__":
    main()
