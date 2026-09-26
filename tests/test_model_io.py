import torch
from model_io import save_lstm_weights, load_lstm_weights, load_weights_or_fresh
from lstm_q_network import build_lstm_network


def _corrupt_shape(path):
    """Overwrite a saved checkpoint's conv.weight with an incompatible
    shape, simulating an architecture change (e.g. NUM_CELL_CLASSES or
    VIEW_H changing) without actually needing a different config."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt["state_dict"]["conv.weight"] = torch.zeros(1, 1, 1, 1)
    torch.save(ckpt, path)


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "brain.pt"
    net = build_lstm_network()
    original = net.conv.weight.detach().clone()
    save_lstm_weights(net, path)

    net2 = build_lstm_network()
    load_lstm_weights(net2, path)
    assert torch.allclose(net2.conv.weight, original)


def test_load_rejects_shape_mismatch(tmp_path):
    path = tmp_path / "brain.pt"
    save_lstm_weights(build_lstm_network(), path)
    _corrupt_shape(path)

    raised = False
    try:
        load_lstm_weights(build_lstm_network(), path)
    except ValueError:
        raised = True
    assert raised


def test_load_weights_or_fresh_missing_file(tmp_path):
    net, loaded = load_weights_or_fresh(build_lstm_network, tmp_path / "nope.pt",
                                        verbose=False)
    assert loaded is False
    assert net is not None


def test_load_weights_or_fresh_falls_back_on_incompatible_checkpoint(tmp_path):
    """Regression test: this is exactly the path a BEHIND_ROWS/NUM_CELL_CLASSES
    change hits — must fall back to a fresh net, never raise, never partially
    load."""
    path = tmp_path / "brain.pt"
    save_lstm_weights(build_lstm_network(), path)
    _corrupt_shape(path)

    net, loaded = load_weights_or_fresh(build_lstm_network, path, verbose=False)
    assert loaded is False
    assert net is not None


def test_load_weights_or_fresh_succeeds_on_valid_checkpoint(tmp_path):
    path = tmp_path / "brain.pt"
    original = build_lstm_network()
    save_lstm_weights(original, path)

    net, loaded = load_weights_or_fresh(build_lstm_network, path, verbose=False)
    assert loaded is True
    assert torch.allclose(net.conv.weight, original.conv.weight)
