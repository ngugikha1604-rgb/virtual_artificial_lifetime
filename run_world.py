"""Open the Virtual Lifetime world viewer.

Usage:
    python run_world.py
    python run_world.py --load

Viewer controls include Space (pause), Enter (single tick while paused),
arrows (speed/follow), S (save), L (load latest save), and R (new world).
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VIEWER_DIR = ROOT_DIR / "src" / "rl"
if str(VIEWER_DIR) not in sys.path:
    sys.path.insert(0, str(VIEWER_DIR))

from live_viewer import WorldViewer2D, WORLD_SAVE_PATH


def main():
    parser = argparse.ArgumentParser(description="Watch the Virtual Lifetime world")
    parser.add_argument(
        "--load", action="store_true",
        help="open results/world_save.json instead of creating a new world",
    )
    args = parser.parse_args()

    viewer = WorldViewer2D()
    if args.load and WORLD_SAVE_PATH.exists():
        viewer.load_world()
    elif args.load:
        print(f"No saved world found at {WORLD_SAVE_PATH}; starting a new world.")
    viewer.run()


if __name__ == "__main__":
    main()
