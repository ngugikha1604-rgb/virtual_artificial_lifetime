"""
run_experiment.py — ecosystem training entry point (Conv-LSTM torch, multi-agent).

Runs the persistent, dynamic-population simulation (src/rl/population.py) for
--ticks ticks and, in one command, produces every artifact you need to judge
the run:

    1. loads results/best_model.pt if it exists (continuing from wherever
       training left off) to found the initial population, else starts
       INITIAL_POPULATION fresh random-weight founders,
    2. every tick, checks whether any individual — living, or one that just
       died this very tick — has beaten the best cumulative total_reward
       ("fitness") ever seen, and if so saves THAT individual's brain to
       results/best_model.pt immediately (population.check_and_save_best) —
       there is only ONE model file, same as the old single-agent version,
       just chosen by "best individual in the population" now instead of
       "best rolling-mean lifetime",
    3. renders results/ecosystem_lifetime.gif (every living agent + food/
       hazard/seeds + a population-size-over-time chart, sampled every
       config.ECOSYSTEM_SNAPSHOT_EVERY ticks — not every tick, see that
       constant's comment),
    4. writes results/ecosystem_log.csv (population size / births / deaths /
       max generation, logged every config.ECOSYSTEM_LOG_EVERY ticks),
    5. prints a final summary.

This REPLACES the old single-agent, sequential-lifetimes version of this
file (train one agent, one lifetime at a time) — that mode's machinery
(run_episode.py's single-lifetime loop) still exists and still works, it's
just no longer what this entry point drives. See progress.md for why: the
single-agent driver is superseded by the ecosystem one for actual training
now that reproduction exists, but run_episode.py itself is untouched and
still usable directly (e.g. by src/rl/population.py's Individual, which
reuses its compute_reward, and by anything wanting a single-agent A/B
comparison against the ecosystem's evolved population).

Usage:
    python run_experiment.py                  # default NUM_ECOSYSTEM_TICKS run
    python run_experiment.py --ticks 10000    # longer run
    python run_experiment.py --reset-epsilon  # keep results/best_model.pt's
                                              # weights but found the initial
                                              # population with fresh
                                              # EPSILON_START exploration
                                              # instead of resuming saved
                                              # epsilon from training_state.json
"""

import argparse
import copy
import random
import sys
from pathlib import Path
import numpy as np
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
from lstm_q_network import build_lstm_network
from model_io import load_lstm_weights
from training_state import load_training_state
from population import spawn_founder, ecosystem_step, check_and_save_best
from visualize import ecosystem_snapshot, save_ecosystem_gif
from config import (WORLD_SIZE, NUM_FOOD_LOW, NUM_FOOD_HIGH, NUM_HAZARDS, SEED,
                    INITIAL_POPULATION, EPSILON_START, NUM_ECOSYSTEM_TICKS,
                    ECOSYSTEM_SNAPSHOT_EVERY, ECOSYSTEM_LOG_EVERY, ECOSYSTEM_GIF_FPS,
                    SAVE_EVERY)

# ONE model file, shared by every driver (this script, live_viewer.py).
WEIGHTS_PATH = RESULTS_DIR / "best_model.pt"
STATE_PATH   = RESULTS_DIR / "training_state.json"
GIF_PATH     = RESULTS_DIR / "ecosystem_lifetime.gif"
CSV_PATH     = RESULTS_DIR / "ecosystem_log.csv"


def run_experiment(num_ticks, reset_epsilon=False):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    world = World(width=WORLD_SIZE, height=WORLD_SIZE,
                  num_food_low=NUM_FOOD_LOW, num_food_high=NUM_FOOD_HIGH,
                  num_hazards=NUM_HAZARDS)

    # Load existing weights to found the population from, if available.
    base_net = build_lstm_network()
    weights_exist = WEIGHTS_PATH.exists()
    if weights_exist:
        try:
            print(f"[weights] Found existing weights at {WEIGHTS_PATH} — founding "
                  f"the population from them.")
            load_lstm_weights(base_net, WEIGHTS_PATH)
        except ValueError as e:
            print(f"[weights] {e}")
            print("[weights] Re-initialising fresh random-weight founder(s).")
            base_net = build_lstm_network()
            weights_exist = False
    else:
        print("[weights] No saved weights found — starting from fresh random founder(s).")

    state = load_training_state(STATE_PATH)
    if reset_epsilon:
        epsilon_start = EPSILON_START
        best_metric_so_far = state["best_metric"] if state else float("-inf")
        print(f"[epsilon] --reset-epsilon: founder(s) start exploration at "
              f"{epsilon_start} (weights untouched; best_metric carried over).")
    elif state is not None:
        epsilon_start = state["epsilon"]
        best_metric_so_far = state["best_metric"]
        print(f"[epsilon] Resuming saved exploration state: epsilon={epsilon_start:.4f}, "
              f"best_metric={best_metric_so_far:+.3f}.")
    elif weights_exist:
        epsilon_start = 0.3
        best_metric_so_far = float("-inf")
        print("[epsilon] No training_state.json found alongside existing weights "
              "(pre-dates this feature, or was single-agent) — falling back to epsilon=0.3.")
    else:
        epsilon_start = EPSILON_START
        best_metric_so_far = float("-inf")

    # Founders get the loaded weights AS-IS (unmutated) — each one an
    # independent copy if there's more than one, since Individual.__init__
    # stores a *reference* to whatever net object it's given and each
    # individual's own optimizer will start diverging it immediately.
    population = [spawn_founder(world, epsilon=epsilon_start,
                                net=copy.deepcopy(base_net) if i > 0 else base_net)
                 for i in range(INITIAL_POPULATION)]

    print(f"[train] Starting with {len(population)} founder(s), running {num_ticks} "
          f"ticks (world {WORLD_SIZE}x{WORLD_SIZE})...")

    frames = [ecosystem_snapshot(world, population, 0)]
    log = []
    total_births = 0
    total_deaths = 0
    max_generation_ever = 0

    for tick in range(1, num_ticks + 1):
        births, deaths = ecosystem_step(world, population, auto_reseed=True)
        total_births += len(births)
        total_deaths += len(deaths)
        if population:
            max_generation_ever = max(max_generation_ever,
                                      max(ind.generation for ind in population))

        best_metric_so_far = check_and_save_best(deaths, best_metric_so_far,
                                                  WEIGHTS_PATH, STATE_PATH)
        if tick % SAVE_EVERY == 0:
            best_metric_so_far = check_and_save_best(population, best_metric_so_far,
                                                      WEIGHTS_PATH, STATE_PATH)

        if tick % ECOSYSTEM_SNAPSHOT_EVERY == 0:
            frames.append(ecosystem_snapshot(world, population, tick))

        if tick % ECOSYSTEM_LOG_EVERY == 0:
            log.append({
                "tick": tick, "population": len(population),
                "births_total": total_births, "deaths_total": total_deaths,
                "max_generation": max_generation_ever,
                "best_metric": round(best_metric_so_far, 3),
            })
            print(f"tick={tick:6d} pop={len(population):2d} "
                  f"births_total={total_births:4d} deaths_total={total_deaths:4d} "
                  f"max_gen={max_generation_ever} best_metric={best_metric_so_far:+.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[gif] Rendering {len(frames)} frames -> {GIF_PATH} ...")
    save_ecosystem_gif(frames, WORLD_SIZE, WORLD_SIZE, GIF_PATH,
                       fps=ECOSYSTEM_GIF_FPS, terrain=world.terrain)
    print(f"[gif] Saved to {GIF_PATH}")

    import csv
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tick", "population", "births_total",
                                          "deaths_total", "max_generation", "best_metric"])
        w.writeheader()
        w.writerows(log)
    print(f"[csv] {CSV_PATH}")

    print(f"\n--- Summary ---")
    print(f"final population: {len(population)}")
    print(f"total births: {total_births}, total deaths: {total_deaths}")
    print(f"max generation reached: {max_generation_ever}")
    print(f"best individual fitness (total_reward) ever seen: {best_metric_so_far:+.2f}")
    print(f"[weights] {WEIGHTS_PATH}")
    print(f"[state]   {STATE_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a multi-agent reproduction/"
                                            "evolution ecosystem and checkpoint "
                                            "the best individual to best_model.pt.")
    p.add_argument("--ticks", type=int, default=NUM_ECOSYSTEM_TICKS,
                   help=f"number of ticks to run (default: {NUM_ECOSYSTEM_TICKS})")
    p.add_argument("--reset-epsilon", action="store_true",
                   help="keep the loaded weights but found the population with "
                        "fresh EPSILON_START exploration instead of resuming "
                        "the saved epsilon from training_state.json")
    args = p.parse_args()
    run_experiment(num_ticks=args.ticks, reset_epsilon=args.reset_epsilon)
