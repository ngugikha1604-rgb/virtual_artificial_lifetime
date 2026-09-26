"""
model_io.py — save / load for the PyTorch Conv-LSTM brain.

No longer hand-packs NumPy arrays into .npz. The brain is a single nn.Module
(see rl/torch_q_net.ConvLSTMDQN) whose whole parameter set lives in
net.state_dict(), so persistence is one torch.save / torch.load.

Signature change from the old NumPy version:
    save_lstm_weights(net, file_path)   # net is the nn.Module, not (conv,lstm,out)
    load_lstm_weights(net, file_path)   # in-place load_ into net

load tolerates an incompatible-architecture torch checkpoint (shape / key
mismatch) and reports it so the caller can drop to a fresh net. It does NOT
claim to read old NumPy-format .npz files — those belong to the retired
NumPy brain and are intentionally not loadable here.
"""
import torch
from pathlib import Path


def load_weights_or_fresh(build_net_fn, file_path, verbose=True):
    """Try to load `file_path` into a freshly-built net; the "resume if
    possible, else start fresh" pattern every entry point needs
    (run_experiment.py, pretrain_single_agent.py, live_viewer.py all used to
    duplicate this same ~10-line try/except by hand — centralised here
    2026-09 after a senior-style codebase review flagged it).

    `build_net_fn`: zero-arg callable returning a fresh net (e.g.
    lstm_q_network.build_lstm_network) — taken as a parameter rather than
    imported directly so this brain-layer module doesn't need to depend on
    the RL-layer's network factory.

    Returns (net, weights_loaded: bool). weights_loaded is False both when
    `file_path` doesn't exist yet AND when it exists but is incompatible
    (load_lstm_weights raised ValueError — architecture changed) — either
    way the returned net is a fresh one from build_net_fn(), never a
    partially-loaded one.
    """
    net = build_net_fn()
    if not Path(file_path).exists():
        if verbose:
            print(f"[weights] No saved weights found at {file_path} — starting fresh.")
        return net, False
    try:
        load_lstm_weights(net, file_path)
        return net, True
    except ValueError as e:
        if verbose:
            print(f"[weights] {e}")
            print("[weights] Re-initialising a fresh random-weight brain.")
        return build_net_fn(), False


def save_lstm_weights(net, file_path):
    """Save the full parameter dict of the net to file_path (.pt)."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict()}, file_path)
    print(f"Saved Conv-LSTM (torch) weights to {file_path}")


def load_lstm_weights(net, file_path):
    """
    Load parameters into net in place. Raises ValueError when the checkpoint
    is not a compatible torch brain (not a state dict, unknown key, or shape
    mismatch). The caller catches it and falls back to a fresh net.
    """
    try:
        ckpt = torch.load(file_path, map_location="cpu", weights_only=True)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint geometry unknown: {type(ckpt)}")
        # Shape-compatibility guard: reject old-architecture / re-sized nets.
        current = net.state_dict()
        for key, val in state.items():
            if key not in current:
                raise ValueError(f"Unknown key '{key}' in {file_path} (arch changed?)")
            if tuple(val.shape) != tuple(current[key].shape):
                raise ValueError(f"Key '{key}' shape {tuple(val.shape)} != current "
                                 f"{tuple(current[key].shape)} (input/channels changed).")
        net.load_state_dict(state)
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        # Includes corrupt/foreign files: an old .npz is a zip archive and
        # torch.load cannot read it -> surfaces here as an OSError/EOFError.
        raise ValueError(f"Cannot load brain weights from {file_path}: {e}")
    print(f"Successfully loaded Conv-LSTM (torch) weights from {file_path}")

