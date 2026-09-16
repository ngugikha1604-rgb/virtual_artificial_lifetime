"""
world_tick.py — the ONE "advance the world by one tick" step, shared by every
driver of a lifetime: run_episode.py's training loop, visualize.py's demo
recorder, and live_viewer.py's interactive stepper.

Why this file exists:
  Before this, the same ~20 lines (ask the brain for an action or hold it
  during cooldown, call world.step(), apply energy cost + starvation, apply
  the food/hazard event to the body, age the agent, build the next
  observation) were copy-pasted in THREE places. Any change to tick
  mechanics — a new event type, a new body stat, a change to how starvation
  works — had to be applied identically in three files by hand. This is
  exactly how live_viewer.py silently drifted from run_episode.py before
  (see progress.md's live_viewer rewrite note). Centralising it here means
  a mechanics change is a one-file edit, and all three drivers pick it up
  automatically.

What stays OUT of this file (kept as each caller's own concern):
  - reward shaping            (only run_episode.py / live_viewer.py need it)
  - replay push + learning    (only run_episode.py / live_viewer.py do this)
  - counters (food_eaten...)  (caller decides what to count from the event)
  - rendering / drawing       (visualize.py / live_viewer.py only)
This keeps world_tick() a pure "physics" step — no policy about what the
caller does with the result.
"""
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np
from lstm_q_network import build_observation

STAY_ACTION = 0  # forced action while the agent is busy on a cooldown tick


@dataclass
class TickResult:
    action: int
    q_values: object            # np.ndarray — Q-values from this tick's forward pass
    h_new: object
    c_new: object
    prev_pos: Tuple[int, int]
    prev_facing: int             # facing BEFORE this tick's action (world.step
                                  # already turned agent.facing to the new value
                                  # by the time callers see it) — needed so a
                                  # caller's reward shaping can ask "what could
                                  # the agent actually see when it chose this
                                  # action" via world.visible_entity_positions()
    event: Optional[dict]       # None | {"type": str, "category": "food"|"hazard"}
    starved: bool                # True if THIS tick's energy cost hit 0 and
                                  # applied starvation damage (mirrors `event`
                                  # being the hazard/food signal, but starving
                                  # isn't an `event` from world.step() — it's a
                                  # body-mechanics side effect of the energy
                                  # cost — so it needs its own field for a
                                  # caller's reward shaping to see it)
    done: bool                  # True once this tick killed/ended the agent
    x_used: object              # observation actually fed to brain.act/forward
                                  # this tick (may differ from the x passed in
                                  # when other_positions injects channel 9) —
                                  # callers must push THIS to replay, not the
                                  # original x, so stored (obs, action) pairs
                                  # are consistent
    x_next: object              # next observation vector, ready for the next tick


def _make_other_agent_grid(world, position, facing, other_positions):
    """Build a (VIEW_H, VIEW_W) bool grid marking cells in the agent's vision
    cone that are occupied by another agent. Uses the same row/column geometry
    as World.get_local_view_layers() so the spatial layout is consistent.

    `other_positions`: a set/frozenset of (x, y) tuples — the positions of
    every OTHER living agent (the current agent's own position excluded by
    the caller). Behind-agent rows stay False (unknown, same convention as
    every other channel). Out-of-bounds cells are never in other_positions so
    no bounds check needed here.
    """
    fx, fy = world._fwd(facing)
    rx, ry = world._right(facing)
    px, py = position
    half        = world.vision_width // 2
    col_offsets = range(-half, world.vision_width - half)

    grid = np.zeros((world.vision_range + world.behind_rows, world.vision_width),
                    dtype=bool)
    for row in range(world.vision_range):
        dist = world.vision_range - row
        for ci, off in enumerate(col_offsets):
            cell = (px + fx*dist + rx*off, py + fy*dist + ry*off)
            if cell in other_positions:
                grid[row, ci] = True
    return grid


def world_tick(world, agent, brain, x, h, c,
               other_positions=None) -> "TickResult":
    """
    Advance world + agent body by exactly one tick.

    `other_positions`: optional frozenset/set of (x, y) tuples — positions of
    every OTHER living agent at the START of this tick (a snapshot taken before
    any agent has moved this tick). When provided, channel 9 is injected into
    the observation the brain uses to decide this tick's action, so the agent
    truly sees its neighbours when it acts — not one tick later.

    Timing model (simultaneous-tick / snapshot):
      1. Caller snapshots all positions BEFORE anyone moves.
      2. world_tick rebuilds the current observation x WITH channel 9 from
         that snapshot, then calls brain.act(x_with_agents, ...).
      3. world.step() moves this agent.
      4. x_next is built from the new position + the SAME snapshot (still the
         start-of-tick positions — the next tick's ecosystem_step will take a
         fresh snapshot, so x_next will be updated then).
    This avoids the "agent processed later sees a half-updated state" problem
    that arises when each agent's x_next is built after it moves but before
    others have moved.

    Defaults to None (channel 9 all-zero) so single-agent callers
    (run_episode.py, live_viewer.py) require no changes.

    Mutates `agent` in place. Returns everything a caller needs to compute
    reward, push to replay, update counters, or draw a frame.
    """
    # ── Step 1: inject current other-agent positions into x before acting ──
    if other_positions:
        cur_terrain, cur_entity = world.get_local_view_layers(agent.position,
                                                               agent.facing)
        cur_other_grid = _make_other_agent_grid(world, agent.position,
                                                agent.facing, other_positions)
        x_for_act = build_observation(cur_terrain, cur_entity,
                                      agent.internal_state, cur_other_grid)
    else:
        x_for_act = x   # single-agent path: x already correct (no channel 9)

    # ── Step 2: act or hold ────────────────────────────────────────────────
    if agent.is_free:
        action, q_values, h_new, c_new = brain.act(x_for_act, h, c)
        free_tick = True
    else:
        q_values, h_new, c_new = brain.forward(x_for_act, h, c)
        action = STAY_ACTION
        free_tick = False

    prev_pos    = agent.position
    prev_facing = agent.facing
    new_pos, new_facing, event, entered_water = world.step(action, agent.position,
                                                            agent.facing)
    agent.position = new_pos
    agent.facing   = new_facing

    if free_tick:
        agent.commit_action(action, water=entered_water)
    else:
        agent.cooldown_tick()

    starved = agent.apply_energy_cost(action)
    agent.apply_event(event)
    done = not agent.advance_age()

    # ── Step 3: build x_next using same snapshot (fresh snapshot next tick) ─
    next_terrain, next_entity = world.get_local_view_layers(agent.position,
                                                             agent.facing)
    if other_positions:
        next_other_grid = _make_other_agent_grid(world, agent.position,
                                                 agent.facing, other_positions)
        x_next = build_observation(next_terrain, next_entity,
                                   agent.internal_state, next_other_grid)
    else:
        x_next = build_observation(next_terrain, next_entity,
                                   agent.internal_state, None)

    # x_next stored in ind.x will have channel 9 from this tick's snapshot.
    # On the next tick, ecosystem_step takes a fresh snapshot and world_tick
    # rebuilds x_for_act again — so channel 9 is always current at act time.
    return TickResult(action=action, q_values=q_values, h_new=h_new, c_new=c_new,
                      prev_pos=prev_pos, prev_facing=prev_facing, event=event,
                      starved=starved, done=done,
                      x_used=x_for_act, x_next=x_next)
