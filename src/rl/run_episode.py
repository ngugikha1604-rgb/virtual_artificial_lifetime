import sys
from pathlib import Path

RL_DIR    = Path(__file__).resolve().parent
WORLD_DIR = RL_DIR.parent / "world"
SRC_DIR   = RL_DIR.parent
for _dir in (RL_DIR, WORLD_DIR, SRC_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from world import World
from agent import Agent
from lstm_q_network import build_lstm_network, build_observation, zero_state
from torch_agent import TorchQAgent
from lstm_replay_buffer import LSTMReplayBuffer
from policy import EpsilonGreedyPolicy
from world_tick import world_tick
from config import (WORLD_SIZE, NUM_FOOD_LOW, NUM_FOOD_HIGH, NUM_HAZARDS, MAX_AGE,
                    SURVIVAL_BONUS, FOOD_BONUS, HAZARD_PENALTY, STARVATION_PENALTY,
                    SHAPE_WEIGHT_FOOD, SHAPE_WEIGHT_HAZARD,
                    AGE_BONUS_PER_TICK, MAX_AGE_SURVIVAL_BONUS,
                    BATCH_SIZE, WINDOW_N, MIN_EPISODES, LEARN_EVERY)


def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def _nearest_dist_delta(prev_pos, new_pos, positions):
    if not positions:
        return None
    prev_d = min(_manhattan(prev_pos, p) for p in positions)
    new_d  = min(_manhattan(new_pos,  p) for p in positions)
    return prev_d, new_d


def compute_reward(event, prev_pos, prev_facing, new_pos, world, starved=False,
                    done=False, age=0, survived_full_life=False):
    """
    reward = survival bonus
           + food bonus (on eat) OR food-attraction shaping
           + hazard-avoidance shaping (always active)
           - flat hazard penalty (every tick standing on a hazard cell)
           - flat starvation penalty (every tick starvation damage is applied)
           + [only on the tick the episode ends] age bonus + completion bonus

    Both shapings are potential-based (Ng et al. 1999) — additive-safe under
    full observability. This world is a POMDP (the agent only ever sees its
    local view), so shaping is restricted to entities that were actually
    inside the agent's vision cone from (prev_pos, prev_facing) — i.e. exactly
    what the agent could see when it chose this tick's action — via
    world.visible_entity_positions(). Shaping toward/away from an off-screen
    entity would reward the agent for something it had no way to perceive,
    which is a noisy/unlearnable signal rather than useful guidance (two
    ticks with an identical local view could get different shaping purely
    because of where an unseen food/hazard happens to sit).

    Hazard and starvation are both lethal-if-sustained sources of health loss,
    but only hazard used to get a flat per-tick penalty here — starvation's
    only signal was indirect (dying sooner costs future SURVIVAL_BONUS/food).
    `starved` (world_tick.TickResult.starved) makes starvation legible the
    same explicit way hazard already was.

    `done`/`age`/`survived_full_life` add a once-per-lifetime terminal reward
    (see config.py's AGE_BONUS_PER_TICK/MAX_AGE_SURVIVAL_BONUS comment) so
    "how long did you survive" is a much more explicit, comparable-scale
    signal than SURVIVAL_BONUS accumulating alone ever was — age bonus is
    paid regardless of cause of death, the completion bonus only if the
    episode ended by reaching MAX_AGE with health still > 0 (old age, not a
    hazard/starvation death).
    """
    reward   = SURVIVAL_BONUS
    max_dist = world.width + world.height - 2

    if event is not None and event["category"] == "food":
        reward += FOOD_BONUS[event["type"]]
    else:
        # Only shape toward non-rotten food — rotten food has a negative bonus
        # when eaten, so pulling the agent toward it would be counterproductive.
        visible_food = world.visible_entity_positions(prev_pos, prev_facing, category="food")
        visible_food = [p for p in visible_food
                        if world.entity_at(p) and world.entity_at(p)["type"] != "rotten_food"]
        deltas = _nearest_dist_delta(prev_pos, new_pos, visible_food)
        if deltas is not None:
            prev_d, new_d = deltas
            reward += SHAPE_WEIGHT_FOOD * (prev_d - new_d) / max_dist

    visible_hazard = world.visible_entity_positions(prev_pos, prev_facing, category="hazard")
    hazard_deltas = _nearest_dist_delta(prev_pos, new_pos, visible_hazard)
    if hazard_deltas is not None:
        prev_d, new_d = hazard_deltas
        reward += SHAPE_WEIGHT_HAZARD * (new_d - prev_d) / max_dist

    if event is not None and event["category"] == "hazard":
        reward -= HAZARD_PENALTY

    if starved:
        reward -= STARVATION_PENALTY

    if done:
        reward += AGE_BONUS_PER_TICK * age
        if survived_full_life:
            reward += MAX_AGE_SURVIVAL_BONUS

    return reward


def run_episode(world, agent, brain, replay, train=True, render=False, record=False):
    """
    Tick-based loop for ONE lifetime.

    Per-tick PHYSICS (choose/hold action, move, apply energy cost + event,
    age/alive check, build next observation) lives in world_tick.world_tick()
    — shared with visualize.py's demo recorder and live_viewer.py's
    interactive stepper, so a mechanics change only needs editing once. This
    function adds what's specific to TRAINING on top of that tick: reward
    shaping, replay push (one tick appended to the current episode), periodic
    learn_windows() calls over sampled episode windows, and event counters.

    Ticks (not decisions) are counted — a forward-move decision spans 3 ticks
    (ACTION_DURATION[1] = 3) because it stays busy on a cooldown for 2 of them,
    but every tick, free or busy, still produces one full (obs, action, reward,
    done) record that the episode buffer stores with its pre-tick hidden (h,c)
    as an anchor for truncated BPTT.

    If `record=True`, also collects a visualize.snapshot() per tick and
    returns it as `frames` — this lets a caller (run_experiment.py) build
    the demo GIF from the literal, actual training lifetime whose stats get
    logged, instead of only being able to visualize a separate, untracked
    episode. Recording is pure bookkeeping (consumes no extra randomness),
    so it never changes training dynamics: same seed -> identical
    total_reward/steps/food_eaten/hazard_hits whether record is True or False.

    Returns (total_reward, ticks_alive, food_eaten, hazard_hits, frames).
    `frames` is None unless record=True.
    """
    h, c  = zero_state()
    replay.start_episode()   # Phase B: mark a fresh lifetime for the episode buffer
    terrain_grid, entity_grid = world.get_local_view_layers(agent.position, agent.facing)
    x     = build_observation(terrain_grid, entity_grid, agent.internal_state)
    total_reward = 0.0
    steps        = 0
    food_eaten   = 0
    hazard_hits  = 0

    frames = None
    if record:
        from visualize import snapshot  # local import: run_episode.py stays
        frames = [snapshot(world, agent, 0, None)]  # usable w/o matplotlib

    while agent.alive:
        result = world_tick(world, agent, brain, x, h, c)

        if result.event is not None:
            if result.event["category"] == "food":
                food_eaten += 1
            elif result.event["category"] == "hazard":
                hazard_hits += 1

        reward = compute_reward(result.event, result.prev_pos, result.prev_facing,
                                agent.position, world, starved=result.starved,
                                done=result.done, age=agent.age,
                                survived_full_life=(result.done and agent.health > 0))
        # Phase B: store the tick in the current EPISODE with its pre-tick
        # hidden (h,c) as the anchor so a later window can unroll from it.
        replay.push(x, h, c, result.action, reward, result.done)
        if train and len(replay) >= MIN_EPISODES and steps % LEARN_EVERY == 0:
            brain.learn_windows(replay.sample_windows(BATCH_SIZE, WINDOW_N))

        if record:
            frames.append(snapshot(world, agent, agent.age, result.event))

        if render:
            tag = ""
            if result.event is not None:
                tag = (f"ate={result.event['type']}"
                       if result.event["category"] == "food" else "HAZARD")
            busy_tag = "free" if agent.is_free else f"busy({agent.busy_ticks_remaining})"
            print(f"tick={steps:3d} pos={agent.position} e={agent.energy:4.0f} "
                  f"hp={agent.health:4.0f} action={result.action} {busy_tag} {tag} "
                  f"r={reward:+.3f}")

        x, h, c = result.x_next, result.h_new, result.c_new
        total_reward += reward
        steps += 1

    return total_reward, steps, food_eaten, hazard_hits, frames


def main(num_lifetimes=4000,
         world_size=WORLD_SIZE, num_food_low=NUM_FOOD_LOW,
         num_food_high=NUM_FOOD_HIGH, num_hazards=NUM_HAZARDS, max_age=MAX_AGE):
    net = build_lstm_network()
    policy = EpsilonGreedyPolicy()
    brain  = TorchQAgent(net, policy)
    replay = LSTMReplayBuffer()
    for lt in range(num_lifetimes):
        world = World(width=world_size, height=world_size,
                      num_food_low=num_food_low, num_food_high=num_food_high,
                      num_hazards=num_hazards)
        agent = Agent(world, max_age=max_age)
        total_reward, steps, food_eaten, hazard_hits, _ = run_episode(world, agent, brain, replay)
        policy.decay()
        if lt % 50 == 0 or lt == num_lifetimes - 1:
            print(f"lt={lt:4d} ticks={steps:3d} food={food_eaten:2d} "
                  f"hazard={hazard_hits:2d} eps={policy.epsilon:.3f} buf={len(replay)}")

if __name__ == "__main__":
    main()
