"""
Epsilon-greedy policy.

Decides which action to take given a set of Q-values predicted by the brain.

- With probability `epsilon`: pick a random action (explore).
- Otherwise: pick the action with the highest predicted Q-value (exploit).

`epsilon` decays over time so the agent explores a lot early on (when its
Q-value estimates are still meaningless) and relies more on what it has
learned as training progresses.
"""

import random
import numpy as np
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from config import EPSILON_START, EPSILON_MIN, EPSILON_DECAY


class EpsilonGreedyPolicy:
    def __init__(self, epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, q_values, num_actions):
        """
        q_values: array of Q-values, one per action (output of the network)
        num_actions: total number of possible actions

        Returns an integer action index.
        """
        if random.random() < self.epsilon:
            return random.randint(0, num_actions - 1)
        return int(np.argmax(q_values))

    def decay(self):
        """Shrink epsilon after each episode, but never below epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
