"""Detection CLI entry point; execute in the Linux container."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "smoke")
    main()
