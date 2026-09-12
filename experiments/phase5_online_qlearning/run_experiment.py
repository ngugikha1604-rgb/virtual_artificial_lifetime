"""
run_experiment.py — single experiment + visualization entry point (Conv-LSTM torch).

Runs the virtual-lifetime training benchmark end to end and, in one command,
produces every artifact you need to inspect + judge the run:

    1. builds a fresh Conv-LSTM Q-network (torch),
    2. trains across NUM_LIFETIMES lifetimes (continues from results/best_model.pt
       if it exists, else from a random net; also resumes epsilon exploration
       state from results/training_state.json if one exists — see
       src/rl/training_state.py — instead of guessing a fixed restart value),
    3. saves back to that SAME results/best_model.pt — the one file every
       driver in this project (this script, live_viewer.py) loads from and
       saves to, so there is exactly one "the model" at any time, never a
       "wait, which .pt is the real one" question. Saved every config.SAVE_EVERY
       lifetimes (crash safety) AND immediately whenever the rolling-mean
       reward over the last config.BEST_METRIC_WINDOW lifetimes beats the
       best ever seen (best_metric, tracked in training_state.json) — see
       config.py's Checkpointing section. NOTE: because there is only one
       file, a periodic save can still overwrite it with weights that later
       turn out worse than an earlier point — unlike having a separate
       "best-only" file, this does NOT protect against late-training
       degradation, only against losing an interrupted run's progress. If
       that protection turns out to matter, ask to bring back a second file.
    4. records the ACTUAL last training lifetime (record=True) and renders it to
       results/lifetime_demo_lstm.gif — so the GIF matches a real logged row,
       not a separate untracked episode,
    5. writes per-lifetime metrics to results/phase6_multi_entity.csv,
    6. prints a first-10% vs last-10% summary.

All tunable constants live in src/config.py — edit that one file.

The heavy lifting (per-tick physics, replay push, windowed BPTT learning
cadence) is in src/rl/run_episode.py / world_tick.py; this file is the thin
experiment driver.

Usage:
    python run_experiment.py                   # full default run (4000 lt)
    python run_experiment.py --lifetimes 30    # short smoke / regressions run
    python run_experiment.py --no-train        # if best_model.pt exists: render
                                               # a GIF + CSV from a fresh demo
                                               # lifetime only; leave existing
                                               # CSV intact
    python run_experiment.py --reset-epsilon   # keep the loaded model weights
                                               # but restart exploration at
                                               # EPSILON_START instead of
                                               # resuming the saved epsilon —
                                               # use when training seems stuck
                                               # in a narrow policy. To throw
                                               # away the MODEL too and start
                                               # fully from scratch, just
                                               # delete the results/ files
                                               # yourself instead of a flag.
"""

import argparse
import csv
import random
import sys
from collections import deque
from pathlib import Path
import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT   = EXPERIMENT_DIR.parent.parent
RL_DIR         = PROJECT_ROOT / "src" / "rl"
WORLD_DIR      = PROJECT_ROOT / "src" / "world"
BRAIN_DIR      = PROJECT_ROOT / "src" / "brain"
SRC_DIR        = PROJECT_ROOT / "src"
RESULTS_DIR    = PROJECT_ROOT / "results"
for _dir in (RL_DIR, WORLD_DIR, BRAIN_DIR, SRC_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from run_episode import run_episode
from world import World
from agent import Agent
from lstm_q_network import build_lstm_network
from torch_agent import TorchQAgent
from policy import EpsilonGreedyPolicy
from lstm_replay_buffer import LSTMReplayBuffer
from visualize import save_lifetime_gif
from model_io import save_lstm_weights, load_lstm_weights
from training_state import save_training_state, load_training_state
from config import (WORLD_SIZE, NUM_FOOD_LOW, NUM_FOOD_HIGH, NUM_HAZARDS,
                    MAX_AGE, NUM_LIFETIMES, PRINT_EVERY, SEED,
                    DEMO_EPSILON, DEMO_FPS, EPSILON_START, EPSILON_DECAY, EPSILON_MIN,
                    SAVE_EVERY, BEST_METRIC_WINDOW)

# ONE model file, shared by every driver (this script, live_viewer.py). See
# module docstring point 3 for why there's no separate "latest" vs "best"
# file anymore (there used to be — simplified on request).
WEIGHTS_PATH = RESULTS_DIR / "best_model.pt"
STATE_PATH   = RESULTS_DIR / "training_state.json"
CSV_PATH     = RESULTS_DIR / "phase6_multi_entity.csv"
GIF_PATH     = RESULTS_DIR / "lifetime_demo_lstm.gif"


def make_policy(epsilon_start):
    """Training policy: starts at epsilon_start, decays to EPSILON_MIN at rate
    EPSILON_DECAY (see config docstring)."""
    return EpsilonGreedyPolicy(epsilon=epsilon_start, epsilon_min=EPSILON_MIN,
                               epsilon_decay=EPSILON_DECAY)


def _save_checkpoint(net, epsilon, lifetimes_trained, best_metric, tag):
    """Save best_model.pt + training_state.json together — they must always
    agree with each other (an epsilon saved next to weights it doesn't match
    would resume training with the wrong exploration rate)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_lstm_weights(net, WEIGHTS_PATH)
    save_training_state(STATE_PATH, epsilon, lifetimes_trained, best_metric)
    print(f"[checkpoint] ({tag}) saved {WEIGHTS_PATH.name} + training_state.json "
          f"— epsilon={epsilon:.4f}, lifetimes_trained={lifetimes_trained}, "
          f"best_metric={best_metric:+.3f}")


def run_experiment(num_lifetimes, no_train, reset_epsilon=False):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    net = build_lstm_network()

    # Load existing weights to continue training if available.
    weights_exist = WEIGHTS_PATH.exists()
    if weights_exist:
        try:
            print(f"[weights] Found existing weights at {WEIGHTS_PATH} — loading to continue.")
            load_lstm_weights(net, WEIGHTS_PATH)
        except ValueError as e:
            print(f"[weights] {e}")
            print("[weights] Re-initialising a fresh Conv-LSTM brain from scratch.")
            net = build_lstm_network()
            weights_exist = False
    else:
        print("[weights] No saved weights found — starting from a fresh random brain.")

    # Resume (or reset) exploration + bookkeeping state. Kept separate from
    # the weights file on purpose — see training_state.py's docstring — so
    # --reset-epsilon can restart exploration without touching the model,
    # and so `lifetimes_trained`/`best_metric` stay meaningful across however
    # many separate `--lifetimes N` invocations actually trained this brain.
    state = load_training_state(STATE_PATH)
    if reset_epsilon:
        epsilon_start = EPSILON_START
        lifetimes_trained = state["lifetimes_trained"] if state else 0
        best_metric_so_far = state["best_metric"] if state else float("-inf")
        print(f"[epsilon] --reset-epsilon: exploration restarted at {epsilon_start} "
              f"(model weights untouched; lifetimes_trained/best_metric carried over).")
    elif state is not None:
        epsilon_start = state["epsilon"]
        lifetimes_trained = state["lifetimes_trained"]
        best_metric_so_far = state["best_metric"]
        print(f"[epsilon] Resuming saved exploration state: epsilon={epsilon_start:.4f} "
              f"({lifetimes_trained} lifetimes trained so far, "
              f"best_metric={best_metric_so_far:+.3f}).")
    elif weights_exist:
        # Weights from before training_state.json existed — no epsilon on
        # record, fall back to the old fixed guess rather than assuming 1.0
        # (which would re-explore wildly against an already-trained brain).
        epsilon_start = 0.3
        lifetimes_trained = 0
        best_metric_so_far = float("-inf")
        print("[epsilon] No training_state.json found alongside existing weights "
              "(pre-dates this feature) — falling back to epsilon=0.3.")
    else:
        epsilon_start = EPSILON_START
        lifetimes_trained = 0
        best_metric_so_far = float("-inf")

    log = []
    frames = None
    demo_stats = None
    terrain_for_gif = None

    if not no_train:
        policy = make_policy(epsilon_start=epsilon_start)
        brain  = TorchQAgent(net, policy)
        replay = LSTMReplayBuffer()
        reward_window = deque(maxlen=BEST_METRIC_WINDOW)

        print(f"[train] Running {num_lifetimes} lifetimes "
              f"(world {WORLD_SIZE}x{WORLD_SIZE}, {NUM_FOOD_LOW}L+{NUM_FOOD_HIGH}H food, "
              f"{NUM_HAZARDS} hazards, max_age={MAX_AGE})...")
        for lt in range(num_lifetimes):
            world = World(width=WORLD_SIZE, height=WORLD_SIZE,
                          num_food_low=NUM_FOOD_LOW, num_food_high=NUM_FOOD_HIGH,
                          num_hazards=NUM_HAZARDS)
            agent = Agent(world, max_age=MAX_AGE)
            is_last = (lt == num_lifetimes - 1)
            total_reward, steps, food_eaten, hazard_hits, lt_frames = run_episode(
                world, agent, brain, replay, record=is_last)
            policy.decay()
            lifetimes_trained += 1
            reward_window.append(total_reward)
            log.append({
                "lifetime":     lt,
                "steps":        steps,
                "food_eaten":   food_eaten,
                "hazard_hits":  hazard_hits,
                "total_reward": round(total_reward, 3),
                "epsilon":      round(policy.epsilon, 4),
            })
            if is_last:
                frames = lt_frames
                terrain_for_gif = world.terrain
                demo_stats = (steps, food_eaten, hazard_hits, total_reward)

            # Rolling-mean "new best" check — once the window is full, so an
            # early lucky/unlucky single lifetime can't crown a false best.
            # Saves IMMEDIATELY on improvement, in addition to the periodic
            # save below, so a genuine improvement isn't left waiting for the
            # next SAVE_EVERY boundary.
            new_best = False
            if len(reward_window) == BEST_METRIC_WINDOW:
                rolling_mean = sum(reward_window) / BEST_METRIC_WINDOW
                if rolling_mean > best_metric_so_far:
                    best_metric_so_far = rolling_mean
                    new_best = True
                    print(f"[best] New best rolling mean reward {rolling_mean:+.3f} "
                          f"(avg over last {BEST_METRIC_WINDOW} lifetimes) at lt={lt}")

            # Checkpoint — periodic (crash safety) OR on a new best. Both
            # write to the SAME best_model.pt (see module docstring point 3).
            if new_best or (lt + 1) % SAVE_EVERY == 0:
                _save_checkpoint(net, policy.epsilon, lifetimes_trained,
                                 best_metric_so_far, tag=f"lt={lt}")

            if lt % PRINT_EVERY == 0 or is_last:
                print(f"lt={lt:4d} steps={steps:3d} food={food_eaten:2d} "
                      f"hazard={hazard_hits:2d} reward={total_reward:+.2f} "
                      f"eps={policy.epsilon:.3f} buf={len(replay)}")

        # Final save always happens, exactly at the true end state, even if
        # it doesn't land on a SAVE_EVERY boundary.
        _save_checkpoint(net, policy.epsilon, lifetimes_trained, best_metric_so_far,
                         tag="end of run")
        print(f"\n[demo] GIF will be the literal last training lifetime above "
              f"(lt={num_lifetimes - 1}, {len(frames) - 1} ticks, "
              f"food={demo_stats[1]}, hazard={demo_stats[2]}) — same numbers "
              f"as the last row of the CSV.")
    else:
        if not weights_exist:
            raise SystemExit("[no-train] asked to skip training but no weights exist to load. "
                             "Re-run without --no-train first.")
        print("[no-train] Skipping training; no training lifetime exists to record from, "
              "so recording one fresh (untracked) demo episode with the loaded weights "
              f"instead (epsilon={DEMO_EPSILON}).")
        demo_policy = make_policy(epsilon_start=DEMO_EPSILON)
        demo_brain  = TorchQAgent(net, demo_policy)
        demo_replay = LSTMReplayBuffer()
        demo_world = World(width=WORLD_SIZE, height=WORLD_SIZE,
                           num_food_low=NUM_FOOD_LOW, num_food_high=NUM_FOOD_HIGH,
                           num_hazards=NUM_HAZARDS)
        demo_agent = Agent(demo_world, max_age=MAX_AGE)
        total_reward, steps, food_eaten, hazard_hits, frames = run_episode(
            demo_world, demo_agent, demo_brain, demo_replay, train=False, record=True)
        terrain_for_gif = demo_world.terrain
        demo_stats = (steps, food_eaten, hazard_hits, total_reward)
        print(f"[demo] Recorded a fresh (untracked, not in the CSV) episode: "
              f"{steps} ticks, food={food_eaten}, hazard={hazard_hits}, "
              f"reward={total_reward:+.2f}. Not one of the training lifetimes; "
              f"there is nothing to compare in the CSV because --no-train ran none.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[gif] Rendering {len(frames)} frames -> {GIF_PATH} ...")
    save_lifetime_gif(frames, WORLD_SIZE, WORLD_SIZE, GIF_PATH, fps=DEMO_FPS,
                      terrain=terrain_for_gif)
    print(f"[gif] Saved to {GIF_PATH}")

    if log:
        summarize(log)
        save_csv(log)
        print(f"\n[weights] {WEIGHTS_PATH}")
        print(f"[state]   {STATE_PATH}")
    else:
        # --no-train logged no rows; DON'T overwrite the existing CSV with an
        # empty header-only file (that used to silently destroy history).
        print(f"\n[csv] --no-train produced no log rows; leaving the existing "
              f"{CSV_PATH} untouched.")
    return log


def summarize(log):
    if not log:
        print("\n[summary] No training log (used --no-train).")
        return
    tenth = max(1, len(log) // 10)
    first, last = log[:tenth], log[-tenth:]
    avg = lambda rows, k: sum(r[k] for r in rows) / len(rows)
    print("\n--- Summary (averaged over first/last 10% of lifetimes) ---")
    print(f"steps        first={avg(first, 'steps'):.1f}  last={avg(last, 'steps'):.1f}")
    print(f"food_eaten   first={avg(first, 'food_eaten'):.2f}  last={avg(last, 'food_eaten'):.2f}")
    print(f"hazard_hits  first={avg(first, 'hazard_hits'):.2f}  last={avg(last, 'hazard_hits'):.2f}")
    print(f"total_reward first={avg(first, 'total_reward'):+.3f}  last={avg(last, 'total_reward'):+.3f}")


def save_csv(log):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lifetime", "steps", "food_eaten",
                                          "hazard_hits", "total_reward", "epsilon"])
        w.writeheader()
        w.writerows(log)
    print(f"[csv]     {CSV_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train virtual-lifetime agent, save weights, "
                                            "record + render a demo GIF, and log metrics.")
    p.add_argument("--lifetimes", type=int, default=NUM_LIFETIMES,
                   help=f"number of training lifetimes (default: {NUM_LIFETIMES} from config)")
    p.add_argument("--no-train", action="store_true",
                   help="skip training if weights exist; only demo + gif + csv")
    p.add_argument("--reset-epsilon", action="store_true",
                   help="keep the loaded model weights but restart exploration at "
                        "EPSILON_START instead of resuming the saved epsilon from "
                        "training_state.json")
    args = p.parse_args()
    run_experiment(num_lifetimes=args.lifetimes, no_train=args.no_train,
                   reset_epsilon=args.reset_epsilon)
