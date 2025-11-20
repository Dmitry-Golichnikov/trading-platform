#!/usr/bin/env python
"""
Launcher script для desktop GUI.

Использование:
    python src/interfaces/gui/scripts/run_gui.py

Или из корня проекта:
    python -m src.interfaces.gui.scripts.run_gui
"""

import sys
from pathlib import Path


def _ensure_project_root() -> None:
    """Добавить корень проекта в PYTHONPATH."""
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _run() -> int:
    _ensure_project_root()
    from src.interfaces.gui.desktop.main import main

    return main()


if __name__ == "__main__":
    sys.exit(_run())
