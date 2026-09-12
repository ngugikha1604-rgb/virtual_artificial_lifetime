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

