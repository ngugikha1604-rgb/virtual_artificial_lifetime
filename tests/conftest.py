import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for d in (SRC, SRC / "world", SRC / "rl", SRC / "brain"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
