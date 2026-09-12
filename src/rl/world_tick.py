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
from typing import Optional, Tuple

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
    x_next: object              # next observation vector, ready for the next tick


def world_tick(world, agent, brain, x, h, c) -> TickResult:
    """
    Advance world + agent body by exactly one tick.

    Mutates `agent` in place (position, facing, energy, health, age, alive,
    busy_ticks_remaining) — same side effects the three duplicated copies
    used to have, just in one place now. Returns everything a caller needs
    to compute reward, push to replay, update counters, or draw a frame.
    """
    if agent.is_free:
        action, q_values, h_new, c_new = brain.act(x, h, c)
        free_tick = True
    else:
        q_values, h_new, c_new = brain.forward(x, h, c)
        action = STAY_ACTION
        free_tick = False

    prev_pos    = agent.position
    prev_facing = agent.facing
    new_pos, new_facing, event, entered_water = world.step(action, agent.position, agent.facing)
    agent.position = new_pos
    agent.facing   = new_facing

    # commit_action/cooldown_tick are called AFTER world.step() (not before,
    # as originally) specifically so commit_action can see entered_water —
    # neither call reads agent.position, so this reordering changes nothing
    # else about tick semantics.
    if free_tick:
        agent.commit_action(action, water=entered_water)
    else:
        agent.cooldown_tick()

    starved = agent.apply_energy_cost(action)
    agent.apply_event(event)
    done = not agent.advance_age()

    next_terrain, next_entity = world.get_local_view_layers(agent.position, agent.facing)
    x_next = build_observation(next_terrain, next_entity, agent.internal_state)

    return TickResult(action=action, q_values=q_values, h_new=h_new, c_new=c_new,
                       prev_pos=prev_pos, prev_facing=prev_facing, event=event,
                       starved=starved, done=done, x_next=x_next)
