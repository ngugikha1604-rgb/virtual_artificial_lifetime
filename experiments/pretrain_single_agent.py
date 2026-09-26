"""
pretrain_single_agent.py — single-agent pretraining entry point, meant to run
BEFORE run_experiment.py's ecosystem, not instead of it.

Why this exists (2026-09): Khanh noticed the ecosystem (run_experiment.py /
live_viewer.py) doesn't seem to learn obviously smart behavior even after a
fair number of ticks — it sees food and turns away instead of eating it —
while the OLD single-agent training (run_episode.py's run_episode(), driven
lifetime-after-lifetime by a single persistent brain) used to learn visibly
fast. The reason isn't a bug in either — it's that they learn from
genuinely different amounts/kinds of data per gradient step:

    ecosystem (population.py's ecosystem_step): each learn_windows() call
    gets exactly ONE window — the tail of the ONE life that individual is
    currently, still-in-progress living. No diversity, no past-episode pool.

    single-agent (this script, via run_episode()): each learn_windows() call
    gets BATCH_SIZE=8 windows, sampled from a buffer of up to
    REPLAY_CAPACITY=200 *already-completed* past lifetimes — far richer,
    far less noisy gradient signal.

Rather than redesigning the ecosystem's learning mechanism (a bigger,
riskier change — see progress.md's Roadmap), Khanh's chosen approach is
simpler and lower-risk: use the single-agent loop (proven to learn well) to
build a solid BASELINE brain first, saved to the same shared
results/best_model.pt / training_state.json every other tool reads — then
run_experiment.py's ecosystem takes over from that already-competent
founder for population dynamics/evolution, instead of trying to teach
basic "see food, go eat it" navigation from a barely-trained one.

This script does NOT replace run_experiment.py, and doesn't need to be run
before every ecosystem session — it's a one-time (or occasional, if you
want to raise the baseline further) foundation-building step.

Loads existing weights/epsilon/best_metric from results/ exactly like
run_experiment.py does (same files, same fallback logic), so running this
after already having ecosystem-trained progress CONTINUES from it rather
than starting over, and vice versa — the two entry points hand off cleanly
in either direction because they always read/write the identical files.

Checkpointing mirrors the ORIGINAL single-agent design documented in
progress.md ("Đã sửa thêm... best-model checkpointing", before the
ecosystem rewrite repurposed SAVE_EVERY's unit from lifetimes to ticks for
ITS OWN loop): every SAVE_EVERY lifetimes (crash safety) OR whenever the
rolling mean of total_reward over the last BEST_METRIC_WINDOW lifetimes
beats the best ever recorded, save. The rolling-mean check only starts
once at least BEST_METRIC_WINDOW lifetimes have completed THIS session
(avoids a lucky/unlucky single early lifetime looking like "the new best").
Verified end-to-end in a real run: fires correctly once BEST_METRIC_WINDOW
lifetimes accumulate (best_metric goes from null -> a real rolling mean).

Usage:
    python pretrain_single_agent.py                  # default NUM_LIFETIMES run
    python pretrain_single_agent.py --lifetimes 2000  # shorter/longer run
"""

import argparse
import sys
from collections import deque
from pathlib import Path
import numpy as np
import random
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT   = EXPERIMENT_DIR.parent
RL_DIR         = PROJECT_ROOT / "src" / "rl"
WORLD_DIR      = PROJECT_ROOT / "src" / "world"
BRAIN_DIR      = PROJECT_ROOT / "src" / "brain"
SRC_DIR        = PROJECT_ROOT / "src"
RESULTS_DIR    = PROJECT_ROOT / "results"
for _dir in (RL_DIR, WORLD_DIR, BRAIN_DIR, SRC_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from world import World
from agent import Agent
from lstm_q_network import build_lstm_network
from model_io import load_weights_or_fresh, save_lstm_weights
from torch_agent import TorchQAgent
from lstm_replay_buffer import LSTMReplayBuffer
from policy import EpsilonGreedyPolicy
from run_episode import run_episode
from training_state import load_training_state, save_training_state
from config import (WORLD_SIZE, NUM_FOOD_LOW, NUM_FOOD_HIGH, NUM_HAZARDS, MAX_AGE,
                    SEED, EPSILON_START, NUM_LIFETIMES, PRINT_EVERY,
                    SAVE_EVERY, BEST_METRIC_WINDOW)

# Same shared files every other entry point (run_experiment.py, live_viewer.py)
# reads/writes — see this module's docstring for why that matters.
WEIGHTS_PATH = RESULTS_DIR / "best_model.pt"
STATE_PATH   = RESULTS_DIR / "training_state.json"


def pretrain(num_lifetimes):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    net, weights_exist = load_weights_or_fresh(build_lstm_network, WEIGHTS_PATH)
    if weights_exist:
        print(f"[weights] Found existing weights at {WEIGHTS_PATH} — continuing "
              f"from them.")

    state = load_training_state(STATE_PATH)
    if state is not None:
        epsilon_start = state["epsilon"]
        lifetimes_already_trained = state["lifetimes_trained"]
        best_metric_so_far = state["best_metric"]
        print(f"[epsilon] Resuming saved exploration state: epsilon={epsilon_start:.4f}, "
              f"best_metric={best_metric_so_far:+.3f}.")
    elif weights_exist:
        epsilon_start = 0.3
        lifetimes_already_trained = 0
        best_metric_so_far = float("-inf")
        print("[epsilon] No training_state.json found alongside existing weights "
              "(pre-dates this feature, or is ecosystem-only) — falling back to "
              "epsilon=0.3.")
    else:
        epsilon_start = EPSILON_START
        lifetimes_already_trained = 0
        best_metric_so_far = float("-inf")

    policy = EpsilonGreedyPolicy(epsilon=epsilon_start)
    brain  = TorchQAgent(net, policy)
    replay = LSTMReplayBuffer()

    print(f"[train] Pretraining {num_lifetimes} lifetimes (world "
          f"{WORLD_SIZE}x{WORLD_SIZE}, single persistent brain)...")

    recent_rewards = deque(maxlen=BEST_METRIC_WINDOW)

    for lt in range(1, num_lifetimes + 1):
        world = World(width=WORLD_SIZE, height=WORLD_SIZE,
                      num_food_low=NUM_FOOD_LOW, num_food_high=NUM_FOOD_HIGH,
                      num_hazards=NUM_HAZARDS)
        agent = Agent(world, max_age=MAX_AGE)
        total_reward, steps, food_eaten, hazard_hits, _ = run_episode(
            world, agent, brain, replay)
        policy.decay()
        recent_rewards.append(total_reward)
        lifetimes_already_trained += 1

        if len(recent_rewards) >= BEST_METRIC_WINDOW:
            rolling_mean = sum(recent_rewards) / len(recent_rewards)
            if rolling_mean > best_metric_so_far:
                best_metric_so_far = rolling_mean
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                save_lstm_weights(net, WEIGHTS_PATH)
                save_training_state(STATE_PATH, epsilon=policy.epsilon,
                                    lifetimes_trained=lifetimes_already_trained,
                                    best_metric=best_metric_so_far)
                print(f"[best] New best rolling mean ({rolling_mean:+.2f} over last "
                      f"{BEST_METRIC_WINDOW} lifetimes) -> saved {WEIGHTS_PATH.name}")

        if lt % SAVE_EVERY == 0:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            save_lstm_weights(net, WEIGHTS_PATH)
            save_training_state(STATE_PATH, epsilon=policy.epsilon,
                                lifetimes_trained=lifetimes_already_trained,
                                best_metric=best_metric_so_far)

        if lt % PRINT_EVERY == 0 or lt == num_lifetimes:
            print(f"lt={lt:5d} ticks={steps:3d} food={food_eaten:2d} "
                  f"hazard={hazard_hits:2d} eps={policy.epsilon:.3f} "
                  f"reward={total_reward:+.2f} buf={len(replay)}")

    # Always save once more at the end, regardless of whether the final
    # lifetime happened to be a new rolling-mean best — otherwise a run that
    # ends mid-plateau could exit without persisting its last SAVE_EVERY-
    # aligned progress.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_lstm_weights(net, WEIGHTS_PATH)
    save_training_state(STATE_PATH, epsilon=policy.epsilon,
                        lifetimes_trained=lifetimes_already_trained,
                        best_metric=best_metric_so_far)

    print(f"\n--- Summary ---")
    print(f"lifetimes this session: {num_lifetimes} "
          f"(cumulative: {lifetimes_already_trained})")
    print(f"final epsilon: {policy.epsilon:.4f}")
    print(f"best rolling-mean total_reward seen: {best_metric_so_far:+.2f}")
    print(f"[weights] {WEIGHTS_PATH}")
    print(f"[state]   {STATE_PATH}")
    print(f"\nDone. run_experiment.py / live_viewer.py will now found their "
          f"population from this brain.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pretrain a single persistent brain "
                                            "across many lifetimes (rich, diverse "
                                            "replay-window learning) as a baseline "
                                            "before running the ecosystem.")
    p.add_argument("--lifetimes", type=int, default=NUM_LIFETIMES,
                   help=f"number of lifetimes to train this session (default: "
                        f"{NUM_LIFETIMES})")
    args = p.parse_args()
    pretrain(num_lifetimes=args.lifetimes)
