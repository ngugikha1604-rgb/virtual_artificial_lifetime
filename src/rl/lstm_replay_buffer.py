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
from config import REPLAY_CAPACITY, BURN_IN_N


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

    def current_tail(self, window_n):
        """The most recent up-to-window_n ticks of the CURRENTLY IN-PROGRESS
        episode (not yet closed by done=True). Returns [] if fewer than 2
        ticks have been pushed yet (no learnable transition).

        Added for src/rl/population.py's ecosystem individuals: each one only
        ever lives ONE episode, so len(self) (closed-episode count) can never
        reach MIN_EPISODES during its own life, and sample_windows()'s "K
        random CLOSED episodes" model doesn't fit at all — there is nothing
        to sample from until after death, by which point the individual is
        gone. Learning from the live tail of your own unfolding life instead
        of a pool of past lives is the only thing that makes sense here.
        """
        if len(self._current) < 2:
            return []
        return self._current[-window_n:]

    def sample_current_windows(self, k, window_n, burn_in_n=BURN_IN_N):
        """Sample up to `k` windows from the CURRENTLY IN-PROGRESS episode
        (self._current), together with their preceding burn-in context slices.

        Returns a list of pairs: [(burn_in_slice, learn_slice), ...]
        - If len(self._current) < 2: returns [] (no learnable transition).
        - If len(self._current) < window_n + 1: fallback to a single window
          containing the entire in-progress episode with empty burn-in (starts
          at zero_state).
        - If len(self._current) >= window_n + 1: samples `k` start indices `s`
          in [0, len(self._current) - window_n] (without replacement if enough
          distinct start points exist, with replacement otherwise).
          For each `s`:
            burn_in_slice = self._current[max(0, s - burn_in_n) : s]
            learn_slice   = self._current[s : s + window_n]
        """
        L = len(self._current)
        if L < 2:
            return []
        if L < window_n + 1:
            return [([], self._current[:])]

        num_starts = L - window_n + 1
        starts_range = range(num_starts)
        if num_starts >= k:
            starts = random.sample(starts_range, k)
        else:
            starts = random.choices(starts_range, k=k)

        pairs = []
        for s in starts:
            burn_start = max(0, s - burn_in_n)
            burn_in_slice = self._current[burn_start:s]
            learn_slice = self._current[s:s + window_n]
            pairs.append((burn_in_slice, learn_slice))
        return pairs

    def start_episode(self):
        """Signal the buffer that a new lifetime has begun; discards any
        un-closed (aborted) partial episode from an interrupted run."""
        self._purge_on_start()

    def __len__(self):
        return len(self._episodes)
