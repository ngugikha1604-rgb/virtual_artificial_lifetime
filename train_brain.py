"""Train the shared resident brain.

Usage:
    python train_brain.py --lifetimes 1000

The learned weights are written to results/best_model.pt. Training can be
repeated; the existing brain and training state are resumed automatically.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from pretrain_single_agent import pretrain
from config import NUM_LIFETIMES


def main():
    parser = argparse.ArgumentParser(description="Train the Virtual Lifetime brain")
    parser.add_argument(
        "--lifetimes", type=int, default=NUM_LIFETIMES,
        help=f"lifetimes to train this session (default: {NUM_LIFETIMES})",
    )
    args = parser.parse_args()
    if args.lifetimes <= 0:
        parser.error("--lifetimes must be positive")
    pretrain(num_lifetimes=args.lifetimes)


if __name__ == "__main__":
    main()
