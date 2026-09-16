"""
lstm_q_network.py — observation builders + network factory for the Conv-LSTM Q-net.

Two responsibilities live here, deliberately split:

1. PURE OBSERVATION BUILDERS (NumPy only — framework-agnostic, used by every
   driver: run_episode, world_tick, visualize, live_viewer):
     - encode_observation(...)   -> obs vector (OBS_SIZE,) = [multi-hot flat + body]
     - build_observation         alias of encode_observation
     - zero_state()              -> (np HIDDEN, np HIDDEN) initial hidden/cell
   These never touch torch; world_tick and friends keep passing NumPy and
   never see a tensor (the NumPy boundary lives at the agent).

2. NETWORK FACTORY (PyTorch):
     - build_lstm_network()      -> ConvLSTMDQN() (one nn.Module). The returned
       object owns conv + LSTM cell + output head so torch manages gradients
       + Adam. Feed it obs *vectors* from build_observation; see TorchQAgent.

The encoding is MULTI-HOT, not one-hot: terrain (wall/water/soil/grass, plus
"unknown" for behind-agent cells) and entity (food_low/food_high/hazard) are
two INDEPENDENT channel groups that can both be "on" for the same cell (e.g.
food sitting on water sets both bits) — see World.get_local_view_layers()
for where the two raw grids come from. This used to collapse to a single
mutually-exclusive code per cell (entity hides the terrain under it); see
progress.md for why that changed.
The resulting sizes (INPUT_SIZE = CONV_OUT + 3, OBS_SIZE =
NUM_CELL_CLASSES*VIEW_H*VIEW_W + 3) are documented in src/config.py; the view
may be non-square.
"""
import sys
from pathlib import Path
import numpy as np

BRAIN_DIR = Path(__file__).resolve().parent.parent / "brain"
SRC_DIR   = Path(__file__).resolve().parent.parent
for _dir in (BRAIN_DIR, SRC_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from config import NUM_CELL_CLASSES, HIDDEN_SIZE


# 1. Pure NumPy observation builders
def _multi_hot_grid(terrain_grid, entity_grid, other_agent_grid=None):
    """
    terrain_grid, entity_grid (H, W) int arrays from
    World.get_local_view_layers() -> multi-hot (NUM_CELL_CLASSES, H, W).

    Each cell sets its terrain-code bit (always exactly one) AND, independently,
    its entity-code bit if an entity is present (-1 means none). If
    `other_agent_grid` is provided (bool/int H×W array), any cell with a 1
    also sets channel 9 ("other agent present here"). Behind-agent rows have
    other_agent_grid=0 by convention (callers fill them as zeros).
    """
    C = NUM_CELL_CLASSES
    h, w = terrain_grid.shape
    multi = np.zeros((C, h * w), dtype=np.float64)
    terrain_flat = np.asarray(terrain_grid, dtype=int).reshape(-1)
    entity_flat  = np.asarray(entity_grid, dtype=int).reshape(-1)
    for n in range(h * w):
        multi[terrain_flat[n], n] = 1.0
        e = entity_flat[n]
        if e >= 0:
            multi[e, n] = 1.0
    if other_agent_grid is not None:
        agent_flat = np.asarray(other_agent_grid, dtype=bool).reshape(-1)
        multi[9, agent_flat] = 1.0
    return multi.reshape(C, h, w)


def encode_observation(terrain_grid, entity_grid, internal_state,
                       other_agent_grid=None):
    """(terrain_grid, entity_grid) + body [(3,)] -> obs vector (OBS_SIZE,) float32.

    `other_agent_grid`: optional bool/int (H, W) array — True/1 wherever
    another living agent occupies that cell in the local view. Defaults to
    None (all-zeros channel 9), so single-agent callers (run_episode.py,
    live_viewer.py) need no changes.
    """
    multi = _multi_hot_grid(terrain_grid, entity_grid, other_agent_grid)
    return np.concatenate([multi.reshape(-1),
                           np.asarray(internal_state, dtype=np.float32)
                           ]).astype(np.float32)


build_observation = encode_observation   # backward-compatible alias


def zero_state():
    """Fresh hidden/cell state (np float32 zeros) — reset at the start of a
    lifetime. TorchQAgent converts these to tensors at its network boundary."""
    return (np.zeros(HIDDEN_SIZE, dtype=np.float32),
            np.zeros(HIDDEN_SIZE, dtype=np.float32))


# 2. PyTorch network factory
def build_lstm_network():
    """Return a fresh ConvLSTMDQN PyTorch module."""
    from torch_q_net import ConvLSTMDQN
    return ConvLSTMDQN()
