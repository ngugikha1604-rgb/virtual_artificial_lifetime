"""A small, deterministic controller for a world that works from tick one.

This is deliberately not an RL policy.  It reads the same local observation as
the neural controller and applies a few understandable priorities:

1. move toward visible fresh food;
2. turn away from visible hazards or blocked cells;
3. otherwise keep exploring forward.

It gives a new world a useful baseline even when no model checkpoint exists.
"""
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config


class _FixedPolicy:
    """Compatibility surface used by the viewer and resident telemetry."""

    def __init__(self):
        self.epsilon = 0.0

    def decay(self, rate=None):
        return self.epsilon


class RuleBasedController:
    """Deterministic observation-only controller.

    The controller intentionally exposes ``act`` and ``forward`` like
    ``TorchQAgent`` so the existing world tick can use either controller.
    ``net`` is kept as ``None`` to make it explicit that this controller has no
    trainable model or checkpoint state.
    """

    net = None

    def __init__(self):
        self.policy = _FixedPolicy()

    @staticmethod
    def _grid(x):
        values = np.asarray(x, dtype=np.float32)
        return values[:config.OBS_GRID_FLAT].reshape(
            config.NUM_CELL_CLASSES, config.VIEW_H, config.VIEW_W
        )

    def _scores(self, x):
        grid = self._grid(x)
        food = grid[5] > 0.5
        food |= grid[6] > 0.5
        hazard = (grid[7] > 0.5) | (grid[8] > 0.5)
        walls = grid[1] > 0.5
        center_left = (config.VIEW_W - 1) // 2
        center_right = config.VIEW_W // 2

        scores = np.full(config.NUM_ACTIONS, -1.0, dtype=np.float32)
        scores[0] = 0.0

        # The view is ordered from farthest to nearest.  Prefer the nearest
        # visible food, then use its column to choose forward/turn-left/right.
        food_cells = np.argwhere(food)
        if len(food_cells):
            row, col = food_cells[np.argmax(food_cells[:, 0])]
            if col < center_left:
                scores[3] = 4.0       # turn left
            elif col > center_right:
                scores[4] = 4.0       # turn right
            else:
                scores[1] = 5.0       # move toward food

        # Never deliberately walk into a hazard.  A hazard in the central
        # forward corridor makes turning preferable to moving forward.
        forward_hazard = bool(np.any(hazard[:, center_left:center_right + 1]))
        if forward_hazard:
            scores[1] = -3.0
            if scores[3] <= 0 and scores[4] <= 0:
                scores[3] = 3.0

        # If the nearest central cell is blocked, turning is safer than
        # repeatedly paying energy to bump into a wall.
        near_row = config.VIEW_H - 1
        if np.any(walls[near_row, center_left:center_right + 1]):
            scores[1] = min(scores[1], -2.0)
            if scores[3] <= 0 and scores[4] <= 0:
                scores[4] = 2.0

        # Avoid selecting a negative stay score when the only useful choice is
        # a turn; argmax remains deterministic and easy to inspect in viewer.
        return scores

    def forward(self, x, h, c):
        return self._scores(x), np.asarray(h).copy(), np.asarray(c).copy()

    def act(self, x, h, c):
        q_values, h_new, c_new = self.forward(x, h, c)
        return int(np.argmax(q_values)), q_values, h_new, c_new

    def learn_windows_burned_in(self, windows):
        return 0.0

    def drain_learn_stats(self):
        return None

    def clone(self):
        return RuleBasedController()
