from training_state import save_training_state, load_training_state


def test_round_trip(tmp_path):
    path = tmp_path / "state.json"
    save_training_state(path, epsilon=0.42, lifetimes_trained=17, best_metric=12.5)
    loaded = load_training_state(path)
    assert loaded["epsilon"] == 0.42
    assert loaded["lifetimes_trained"] == 17
    assert loaded["best_metric"] == 12.5


def test_negative_infinity_round_trips_through_null():
    """Regression test: an earlier version wrote best_metric=-inf as the
    non-standard JSON token `-Infinity` — must be stored as `null` and
    translated back to -inf on load (see training_state.py's docstring)."""
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        save_training_state(path, epsilon=1.0, lifetimes_trained=0,
                            best_metric=float("-inf"))
        raw = open(path).read()
        assert "-Infinity" not in raw
        assert "null" in raw
        loaded = load_training_state(path)
        assert loaded["best_metric"] == float("-inf")
    finally:
        os.remove(path)


def test_missing_file_returns_none(tmp_path):
    assert load_training_state(tmp_path / "does_not_exist.json") is None


def test_corrupt_file_returns_none_not_raise(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    assert load_training_state(path) is None
