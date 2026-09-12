"""
lstm_replay_buffer.py — episode buffer for truncated-BPTT Q-learning.

The earlier Option-B learner stored isolated (s,a,r,s') transitions together
with their (h_next,c_next) so a *single-step* Bellman update could use the
right LSTM context without unrolling. That never teaches the LSTM to _carry_
a memory across ticks, because each transition backprops only one step.

This buffer stores **whole episodes** (contiguous tick sequences of ONE
lifetime) plus, for every tick, the LSTM hidden state (h0, c0) the network had
*before* that tick. A "window" of N contiguous ticks from an episode is then
enough to run truncated BPTT: initialise the LSTM with the stored (h0,c0) of
the first tick and unroll forward through the N observations in a single
autograd graph, so gradient flows back through all N LSTM steps and the cell
learns to maintain state across that span.

Per-tick record (episode order):
      obs_x     : observation vector (OBS_SIZE,) at this tick
      h0, c0    : LSTM hidden/cell the network had BEFORE seeing obs_x
                   (== the (h,c) world_tick passed into brain.act/forward,
                    which for a contiguous episode equals the h/c that the
                    previous tick's forward produced — we keep it explicitly
                    so any window can start mid-episode without replaying
                    ticks before it)
      action    : int
      reward    : float (received that tick)
      done      : bool -- True on the LAST tick of an episode only
"""
import random
import numpy as np
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from config import REPLAY_CAPACITY


class LSTMReplayBuffer:
    """
    Ring buffer of finished + in-progress EPISODES (keeps at most `capacity`
    episodes; older ones are dropped when full).

    start_episode(): begin a fresh lifetime's episode (drops any un-closed one).
    push(x, h0, c0, action, reward, done): append one tick to the current
        episode; done=True closes it (the next push starts a new episode).
    sample_windows(k, window_n): return up to `k` windows, each a contiguous
        slice of <= window_n ticks from one random closed episode (>= 2 ticks).
    __len__(): number of stored (closed) episodes.
    """

    def __init__(self, capacity=REPLAY_CAPACITY):
        self.capacity = capacity
        self._episodes = []
        self._current = []      # in-progress episode being appended to

    def push(self, x, h0, c0, action, reward, done):
        self._current.append((x.copy(), h0.copy(), c0.copy(),
                              int(action), float(reward), bool(done)))
        if done:
            # finish the episode and keep only episodes with >1 tick (a lone
            # death tick carries no learnable transition).
            if len(self._current) > 1:
                self._episodes.append(self._current)
                if len(self._episodes) > self.capacity:
                    self._episodes.pop(0)   # drop oldest
            self._current = []

    def _purge_on_start(self):
        # called before building a fresh episode if one is half-open.
        if self._current:
            self._current = []

    def sample_episodes(self, k):
        """`k` random CLOSED episodes. Returns [] while fewer than 1 exist."""
        if not self._episodes:
            return []
        chosen = self._episodes if len(self._episodes) <= k else \
            random.sample(self._episodes, k)
        return chosen

    def sample_windows(self, k, window_n):
        """Sample `k` random episodes and carve ONE truncated window per
        episode (a random contiguous slice of <= window_n ticks, or the whole
        episode if it is shorter). Every episode returned has >= 2 ticks, so
        every window has at least one learnable (s,a,r) pair. Returns [] when
        no buffered episodes exist yet."""
        windows = []
        for ep in self.sample_episodes(k):
            L = len(ep)
            if L < 2:
                continue
            if L <= window_n:
                windows.append(ep)
            else:
                start = random.randint(0, L - window_n)
                windows.append(ep[start:start + window_n])
        return windows

    def start_episode(self):
        """Signal the buffer that a new lifetime has begun; discards any
        un-closed (aborted) partial episode from an interrupted run."""
        self._purge_on_start()

    def __len__(self):
        return len(self._episodes)
