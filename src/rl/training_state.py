"""
training_state.py — persistence for TRAINING-RUN state (epsilon, cumulative
lifetime count, best-so-far metric) that lives ALONGSIDE the brain weights but
is NOT part of the brain itself.

Kept deliberately separate from src/brain/model_io.py: model_io only ever
knows about net.state_dict() (see its docstring) — mixing trainer bookkeeping
into it would blur "the brain" (portable, architecture-checked weights) with
"one particular training run's progress" (epsilon, how far it got, its best
score so far), which is a different concern with a different lifecycle (e.g.
you may want to reset epsilon without touching the brain at all — see
run_experiment.py's --reset-epsilon flag).

Without this file existing, every invocation of `run_experiment.py --lifetimes
N` to continue training reset epsilon to a hard-coded guess (0.3) regardless
of how many lifetimes the loaded brain had already actually seen, and there
was no persisted notion of "best" performance across separate invocations.
"""
import json
from pathlib import Path


def save_training_state(path, epsilon, lifetimes_trained, best_metric):
    """Write (or overwrite) the training-state JSON at `path`.

    best_metric starts life as float("-inf") ("no best yet"). Python's json
    module happily writes that as the bare token `-Infinity`, which IS valid
    to Python's own reader but is non-standard JSON that other tools/parsers
    may choke on — so it's stored as `null` instead and translated back to
    -inf in load_training_state(), keeping the file readable by anything.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    best_metric = float(best_metric)
    with open(p, "w") as f:
        json.dump({
            "epsilon": float(epsilon),
            "lifetimes_trained": int(lifetimes_trained),
            "best_metric": None if best_metric == float("-inf") else best_metric,
        }, f, indent=2)


def load_training_state(path):
    """Returns a dict with keys epsilon/lifetimes_trained/best_metric, or
    None if the file doesn't exist or can't be parsed (treated the same as
    "no saved state" by callers — never raises). best_metric of None ("no
    best recorded yet") is translated back to float("-inf")."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        best_metric = data["best_metric"]
        return {
            "epsilon":           float(data["epsilon"]),
            "lifetimes_trained": int(data["lifetimes_trained"]),
            "best_metric":       float("-inf") if best_metric is None else float(best_metric),
        }
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None
