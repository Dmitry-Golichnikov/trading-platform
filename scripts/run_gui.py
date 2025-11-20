#!/usr/bin/env python
import sys
from pathlib import Path

from src.interfaces.gui.desktop.main import main

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

if __name__ == "__main__":
    main()
