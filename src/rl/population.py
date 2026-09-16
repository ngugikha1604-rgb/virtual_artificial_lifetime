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
                 position=None, facing=None):
        self.id = Individual._next_id
        Individual._next_id += 1
        self.generation   = generation
        self.parent_id    = parent_id
        self.total_reward = 0.0   # this individual's own fitness score, see
                                  # check_and_save_best() below

        self.agent = Agent(world, max_age=config.MAX_AGE)
        if position is not None:
            self.agent.position = position
        if facing is not None:
            self.agent.facing = facing

        policy = EpsilonGreedyPolicy(epsilon=epsilon, epsilon_min=config.EPSILON_MIN,
                                     epsilon_decay=config.EPSILON_DECAY)
        self.brain  = TorchQAgent(net, policy)
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


def spawn_founder(world, epsilon=config.EPSILON_START, generation=0, net=None):
    """A new individual seeding the population. `net` defaults to a fresh
    random-weight brain (used to re-seed the ecosystem after a total
    extinction); pass an existing net (e.g. loaded from results/best_model.pt)
    to found a population that CONTINUES from previously trained weights
    instead of starting over — the founder gets that net AS-IS, unmutated
    (only children born via reproduce() get mutation; a founder is a direct
    continuation of a specific saved brain, not a variant of it)."""
    if net is None:
        net = build_lstm_network()
    return Individual(world, net, epsilon, generation=generation)


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


def reproduce(parent, world, current_population_size):
    """Try to spawn a mutated child of `parent`. Returns the new Individual,
    or None if there's no room under MAX_POPULATION or no free cell exists
    next to the parent (in which case the parent's energy is left untouched
    — REPRODUCTION_ENERGY_COST is only paid on an actual birth, not a failed
    attempt, so a crowded world doesn't quietly starve a would-be parent)."""
    if current_population_size >= config.MAX_POPULATION:
        return None
    cell = world.random_adjacent_cell(parent.agent.position)
    if cell is None:
        return None

    parent.agent.energy -= config.REPRODUCTION_ENERGY_COST
    child_net = mutate_weights(parent.brain.net)
    return Individual(world, child_net, config.EPSILON_START,
                      generation=parent.generation + 1, parent_id=parent.id,
                      position=cell, facing=parent.agent.facing)


def ecosystem_step(world, population, auto_reseed=True):
    """
    Advance the whole ecosystem by ONE tick: every living individual acts
    once (world_tick — the exact same function run_episode.py uses for a
    single agent), reproduces automatically if it qualifies, and learns from
    the TAIL of its own still-unfolding life (LSTMReplayBuffer.current_tail)
    every LEARN_EVERY ticks — NOT the same mechanism as run_episode.py's
    sample_windows()-over-K-closed-episodes, because an ecosystem individual
    only ever lives ONE episode total; there is no pool of past completed
    lives of ITS OWN to sample from during its life (see current_tail's
    docstring in lstm_replay_buffer.py for why).

    Processing order is reshuffled each tick — otherwise whichever individual
    happened to be first in `population` would systematically win any
    contested single-serving food adjacent to more than one individual (the
    only "interaction" that exists at all between individuals in this phase,
    since they don't block or perceive each other — see population.py's
    module docstring).

    Dead individuals are removed at the end of the tick. If the population
    drops to zero and `auto_reseed` is True (the default — used by the
    unattended training drivers), a fresh founder is spawned so the
    simulation keeps going (an "extinction reseed" — shows up as a
    population dip to 1 in a log). Pass auto_reseed=False (e.g. for an
    interactive viewer where a real "population extinct, please reset"
    moment is wanted) to instead leave `population` empty.

    Returns (births: list[Individual], deaths: list[Individual]) for logging.
    """
    order = list(population)
    np.random.shuffle(order)

    births, deaths = [], []

    # Snapshot all positions BEFORE anyone moves this tick — this is the
    # "simultaneous tick" model: every agent observes the same consistent
    # start-of-tick state when it decides its action, regardless of the
    # sequential processing order imposed by the for-loop below. Without
    # this, agents processed later would see a mix of pre- and post-move
    # positions from agents processed earlier, which is neither simultaneous
    # nor sequential — just inconsistent.
    positions_snapshot = frozenset(ind.agent.position for ind in population)

    for ind in order:
        # Exclude this agent's own position from the "other agents" set.
        other_pos = positions_snapshot - {ind.agent.position}
        result = world_tick(world, ind.agent, ind.brain, ind.x, ind.h, ind.c,
                            other_positions=other_pos)

        reward = compute_reward(result.event, result.prev_pos, result.prev_facing,
                                ind.agent.position, world, starved=result.starved,
                                done=result.done, age=ind.agent.age,
                                survived_full_life=(result.done and ind.agent.health > 0))

        if not result.done and ind.agent.energy >= config.ENERGY_TO_REPRODUCE:
            child = reproduce(ind, world, len(population) + len(births))
            if child is not None:
                births.append(child)
                reward += config.REPRODUCE_BONUS

        # Phase B: store the tick in ind's own episode with its PRE-tick
        # hidden (h,c) as the anchor — mirrors run_episode.py exactly.
        # Use result.x_used (the observation world_tick actually fed to the
        # brain) rather than ind.x (the stored observation without channel 9
        # injected), so the (obs, action) pair in the buffer is consistent.
        ind.replay.push(result.x_used, ind.h, ind.c, result.action, reward, result.done)
        ind.total_reward += reward
        ind.x, ind.h, ind.c = result.x_next, result.h_new, result.c_new
        ind.steps += 1

        if ind.steps % config.LEARN_EVERY == 0:
            window = ind.replay.current_tail(config.WINDOW_N)
            if window:
                ind.brain.learn_windows([window])

        if result.done:
            ind.brain.policy.decay()
            deaths.append(ind)

    for d in deaths:
        population.remove(d)
    population.extend(births)

    if not population and auto_reseed:
        population.append(spawn_founder(world))

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
