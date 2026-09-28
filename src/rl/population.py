"""
population.py — multi-agent "ecosystem" layer, built ON TOP OF the existing
single-agent machinery (World, Agent, world_tick, TorchQAgent, compute_reward)
WITHOUT MODIFYING ANY OF THEM. That turned out to be possible because none of
them ever stored "the one agent" as global/implicit state — every call takes
agent/brain/observation/hidden-state as explicit parameters. World itself
doesn't even hold a reference to an agent (positions are passed in per call).
So the exact same functions that already run a single lifetime can be called
once per living individual per tick; the only genuinely NEW code is the
population bookkeeping and the reproduction mechanic itself.

An Individual bundles everything ONE living agent needs to keep acting +
learning — its body (Agent), its own brain (TorchQAgent: own net, own
optimizer, own target net), its own replay buffer, its own epsilon policy,
its own (x, h, c) — i.e. exactly the set of objects run_episode.py juggles
for a single lifetime, just multiplied per individual instead of assumed to
be singular. It also tracks its own cumulative `total_reward` — the
ecosystem's per-individual "fitness score", used by check_and_save_best()
below to decide which individual's brain is worth checkpointing.

Reproduction (see config.py's "Reproduction / ecosystem" section for the
full design rationale) is automatic: whenever a living individual's energy
reaches ENERGY_TO_REPRODUCE, it spawns a mutated copy of itself next to
itself, if there's room under MAX_POPULATION and a free adjacent cell
exists. The child's brain starts as a copy of the parent's CURRENT weights
(whatever it has learned so far in its own life) plus Gaussian noise
(MUTATION_STD) on every parameter — "biến dị gen" (genetic mutation) layered
on top of the parent's already-learned behavior, not a restart from scratch.

Explicitly OUT OF SCOPE for this first version (see progress.md):
  - agents do not perceive each other (no observation channel for "other
    agent"), and do not block/collide with each other's cells — the only
    interaction between individuals is indirect, through shared food/hazard/
    space contention.
"""
import copy
import sys
from pathlib import Path
import numpy as np
import torch

RL_DIR    = Path(__file__).resolve().parent
WORLD_DIR = RL_DIR.parent / "world"
BRAIN_DIR = RL_DIR.parent / "brain"
SRC_DIR   = RL_DIR.parent
for _dir in (RL_DIR, WORLD_DIR, BRAIN_DIR, SRC_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from agent import Agent
from lstm_q_network import build_lstm_network, build_observation, zero_state
from torch_agent import TorchQAgent
from policy import EpsilonGreedyPolicy
from lstm_replay_buffer import LSTMReplayBuffer
from world_tick import world_tick
from run_episode import compute_reward
from model_io import save_lstm_weights
from training_state import save_training_state
from behavior import RuleBasedController
import config


class Individual:
    """One living agent's full body + brain + learning state — the
    multi-agent analogue of the (agent, brain, replay, x, h, c) tuple
    run_episode.py juggles for a single lifetime. Each Individual represents
    exactly one continuous life, from birth to death; it is never reset or
    reused (a "next lifetime" is a new Individual, e.g. via reproduce() or
    spawn_founder(), just like run_episode.py's driver loop creates a new
    World+Agent per lifetime)."""
    _next_id = 0

    def __init__(self, world, net, epsilon, generation=0, parent_id=None,
                 position=None, facing=None, controller="neural", brain=None):
        self.id = Individual._next_id
        Individual._next_id += 1
        self.generation   = generation
        self.parent_id    = parent_id
        self.total_reward = 0.0   # this individual's own fitness score, see
                                  # check_and_save_best() below
        self.event_history = []
        self.last_action = 0
        self.last_position = None
        self.last_event = None

        self.agent = Agent(world, max_age=config.MAX_AGE)
        if position is not None:
            self.agent.position = position
        if facing is not None:
            self.agent.facing = facing

        if brain is not None:
            self.brain = brain
        elif controller == "rules":
            self.brain = RuleBasedController()
        else:
            policy = EpsilonGreedyPolicy(epsilon=epsilon, epsilon_min=config.EPSILON_MIN,
                                         epsilon_decay=config.EPSILON_DECAY)
            self.brain = TorchQAgent(net, policy)
        self.replay = LSTMReplayBuffer()
        self.replay.start_episode()
        self.steps  = 0   # local per-individual tick counter, gates LEARN_EVERY
                          # exactly like run_episode.py's `steps` (NOT agent.age,
                          # which is 1-indexed post-tick and would shift the cadence)

        self.h, self.c = zero_state()
        terrain_grid, entity_grid = world.get_local_view_layers(self.agent.position,
                                                                 self.agent.facing)
        # No other agents visible at birth (population not yet known here);
        # the first real observation with other_agent_grid is built by
        # ecosystem_step on the very first tick.
        self.x = build_observation(terrain_grid, entity_grid,
                                   self.agent.internal_state, None)


def spawn_founder(world, epsilon=config.EPSILON_START, generation=0, net=None,
                  position=None, facing=None, controller="neural", parent_id=None):
    """A new individual seeding the population. `net` defaults to a fresh
    random-weight brain (used to re-seed the ecosystem after a total
    extinction); pass an existing net (e.g. loaded from results/best_model.pt)
    to found a population that CONTINUES from previously trained weights
    instead of starting over — the founder gets that net AS-IS, unmutated
    (only children born via reproduce() get mutation; a founder is a direct
    continuation of a specific saved brain, not a variant of it)."""
    if net is None and controller != "rules":
        net = build_lstm_network()
    return Individual(world, net, epsilon, generation=generation,
                      position=position, facing=facing, controller=controller,
                      parent_id=parent_id)


def mutate_weights(net):
    """A deep copy of `net` with Gaussian noise (config.MUTATION_STD) added
    to every parameter tensor — see config.py's MUTATION_STD comment for why
    this magnitude. Never mutates `net` itself; the parent keeps its own
    (unmutated, still-training) weights."""
    child_net = copy.deepcopy(net)
    with torch.no_grad():
        for p in child_net.parameters():
            p.add_(torch.randn_like(p) * config.MUTATION_STD)
    return child_net


def reproduce(parent, world, current_population_size, child_epsilon=None,
              occupied_positions=None):
    """Try to spawn a mutated child of `parent`. Returns the new Individual,
    or None if there's no room under MAX_POPULATION or no free cell exists
    next to the parent (in which case the parent's energy is left untouched
    — REPRODUCTION_ENERGY_COST is only paid on an actual birth, not a failed
    attempt, so a crowded world doesn't quietly starve a would-be parent).

    `child_epsilon`: starting epsilon for the child. Defaults to
    config.EPSILON_START (full re-explore per generation — Khanh's original
    ecosystem design choice, see progress.md). live_viewer.py passes
    config.LIVE_VIEWER_START_EPSILON here instead (viewer-only override —
    see that constant's comment in config.py); run_experiment.py doesn't
    pass this, so real training is completely unaffected."""
    if current_population_size >= config.MAX_POPULATION:
        return None
    cell = world.random_adjacent_cell(parent.agent.position,
                                      occupied=occupied_positions)
    if cell is None:
        return None

    parent.agent.energy -= config.REPRODUCTION_ENERGY_COST
    if parent.brain.net is None:
        child_brain = parent.brain.clone()
        child_net = None
    else:
        child_brain = None
        child_net = mutate_weights(parent.brain.net)
    epsilon = child_epsilon if child_epsilon is not None else config.EPSILON_START
    return Individual(world, child_net, epsilon,
                      generation=parent.generation + 1, parent_id=parent.id,
                      position=cell, facing=parent.agent.facing,
                      controller="neural", brain=child_brain)


def ecosystem_step(world, population, auto_reseed=True, child_epsilon=None,
                   tick_count=None, enable_learning=True, on_event=None,
                   decay_exploration=True):
    """
    Advance the whole ecosystem by ONE tick: every living individual acts
    once (world_tick), reproduces automatically if it qualifies, and (if
    enable_learning is True) learns from the tail of its own life.

    `enable_learning` (default True): controls gradient updates and replay.
    `decay_exploration` is separate so a viewer can freeze weights while still
    letting a resident become less random as its life progresses.

    `on_event` (default None): optional callable(individual, event_dict)
    called whenever an individual encounters a world event (food, hazard, starvation).
    """
    order = list(population)
    np.random.shuffle(order)

    births, deaths = [], []

    # Snapshot all positions BEFORE anyone moves this tick — this is the
    # "simultaneous tick" model: every agent observes the same consistent
    # start-of-tick state when it decides its action, regardless of the
    # sequential processing order imposed by the for-loop below.
    positions_snapshot = frozenset(ind.agent.position for ind in population)

    for ind in order:
        # Exclude this agent's own position from the "other agents" set.
        other_pos = positions_snapshot - {ind.agent.position}
        result = world_tick(world, ind.agent, ind.brain, ind.x, ind.h, ind.c,
                            other_positions=other_pos)

        ind.last_action = result.action
        ind.last_position = ind.agent.position
        ind.last_event = result.event

        if on_event is not None:
            if result.event is not None:
                on_event(ind, result.event)
            elif result.starved:
                on_event(ind, {"category": "starvation", "type": "starved"})

        reward = compute_reward(result.event, result.prev_pos, result.prev_facing,
                                ind.agent.position, world, starved=result.starved,
                                done=result.done, age=ind.agent.age,
                                survived_full_life=(result.done and ind.agent.health > 0),
                                energy=ind.agent.energy, health=ind.agent.health)

        if not result.done and ind.agent.energy >= config.ENERGY_TO_REPRODUCE:
            occupied_positions = {
                other.agent.position for other in population
                if other is not ind and other not in deaths
            }
            occupied_positions.update(child.agent.position for child in births)
            child = reproduce(ind, world, len(population) + len(births),
                              child_epsilon=child_epsilon,
                              occupied_positions=occupied_positions)
            if child is not None:
                births.append(child)
                reward += config.REPRODUCE_BONUS

        # PRE-tick hidden state: the (h,c) that world_tick was given for this
        # tick. Must be captured BEFORE ind.h/ind.c are overwritten below.
        h_prev, c_prev = ind.h, ind.c

        ind.total_reward += reward
        ind.x, ind.h, ind.c = result.x_next, result.h_new, result.c_new
        ind.steps += 1

        if enable_learning:
            # Phase B: store the tick in ind's own episode with its PRE-tick
            # hidden (h,c) as the anchor — mirrors run_episode.py exactly.
            ind.replay.push(result.x_used, h_prev, c_prev, result.action, reward, result.done)

            if ind.steps % config.LEARN_EVERY == 0:
                windows = ind.replay.sample_current_windows(
                    config.BATCH_SIZE_ECOSYSTEM, config.WINDOW_N, burn_in_n=config.BURN_IN_N
                )
                if windows:
                    ind.brain.learn_windows_burned_in(windows)

        # Exploration schedule is independent from gradient learning.
        if decay_exploration:
            ind.brain.policy.decay(rate=config.INLIFE_EPSILON_DECAY)

        if result.done:
            deaths.append(ind)

    for d in deaths:
        population.remove(d)
    population.extend(births)

    if not population and auto_reseed:
        population.append(spawn_founder(world))

    # ── Periodic learn-stats log ──────────────────────────────────────────
    # Drain stats from every living individual's brain once per
    # LOG_LEARN_EVERY ticks.  Each brain's drain_learn_stats() returns the
    # aggregated stats since the LAST drain (or since construction), so the
    # window is always exactly the last LOG_LEARN_EVERY ticks — never stale.
    # We aggregate across individuals by a simple mean so the log represents
    # "the typical brain in the population", not any one individual.
    if (tick_count is not None
            and config.LOG_LEARN_EVERY > 0
            and tick_count > 0
            and tick_count % config.LOG_LEARN_EVERY == 0):
        all_stats = [ind.brain.drain_learn_stats() for ind in population]
        all_stats = [s for s in all_stats if s is not None]  # skip brains with no learn yet
        if all_stats:
            n_learners   = len(all_stats)
            total_calls  = sum(s["n"] for s in all_stats)
            mean_loss    = sum(s["loss_mean"] for s in all_stats) / n_learners
            last_loss    = sum(s["loss_last"] for s in all_stats) / n_learners
            q_min        = min(s["q_min"]     for s in all_stats)
            q_max        = max(s["q_max"]     for s in all_stats)
            grad_norm    = sum(s["grad_norm"] for s in all_stats) / n_learners
            tgt_syncs    = sum(s["target_syncs"] for s in all_stats)
            print(
                f"[learn] tick={tick_count:>6}  pop={len(population):>3}  "
                f"learners={n_learners}  calls={total_calls}  "
                f"loss_mean={mean_loss:.4f}  loss_last={last_loss:.4f}  "
                f"Q=[{q_min:+.2f}, {q_max:+.2f}]  "
                f"grad={grad_norm:.3f}  tgt_syncs={tgt_syncs}"
            )

    return births, deaths



def check_and_save_best(candidates, best_metric_so_far, net_path, state_path):
    """
    The ecosystem analogue of run_experiment.py's old single-agent "rolling-
    mean beats best_metric_so_far -> save" check: given a list of individuals
    whose fitness (total_reward) should be considered right now, saves the
    best one's brain to `net_path` + persists the new best_metric_so_far to
    `state_path` (training_state.py's file format, reused here for
    continuity with the single-agent checkpoint even though its other fields
    mean something slightly different in ecosystem mode — see the
    save_training_state call below) IF it beats best_metric_so_far.

    Deliberately does NOT decide *which* individuals to check — that's the
    caller's job, and matters a lot for write frequency: passing the full
    living population EVERY tick would re-trigger a disk save on almost
    every tick for as long as the current leader stays alive (its
    total_reward strictly increases each tick it lives). The intended
    pattern (see run_experiment.py / live_viewer.py) is: call with just
    `deaths` every tick (a death's final score is a natural, infrequent,
    one-time event worth checking immediately), and separately call with the
    full living `population` only every SAVE_EVERY ticks or so, as a coarser
    crash-safety net that also catches a still-alive leader.

    Returns the (possibly updated) best_metric_so_far — the caller is
    responsible for holding onto it and passing it back in on the next call.
    """
    if not candidates:
        return best_metric_so_far

    best_ind = max(candidates, key=lambda ind: ind.total_reward)
    if best_ind.total_reward > best_metric_so_far:
        best_metric_so_far = best_ind.total_reward
        save_lstm_weights(best_ind.brain.net, net_path)
        # "lifetimes_trained" is repurposed as "generation reached by the best
        # individual" here — there's no single "how many lifetimes" number in
        # an ecosystem with overlapping, branching individual lifespans, and
        # generation is the closest ecosystem analogue of "how much
        # cumulative training/evolution has happened so far".
        save_training_state(state_path, epsilon=best_ind.brain.policy.epsilon,
                            lifetimes_trained=best_ind.generation,
                            best_metric=best_metric_so_far)
        print(f"[best] New best individual (id={best_ind.id}, gen={best_ind.generation}, "
              f"total_reward={best_ind.total_reward:+.2f}) -> saved {Path(net_path).name}")
    return best_metric_so_far
