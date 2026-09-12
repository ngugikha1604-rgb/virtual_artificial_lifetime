import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (MAX_ENERGY, START_ENERGY, MAX_HEALTH, START_HEALTH,
                    DEFAULT_MAX_AGE, STARVATION_DAMAGE, HAZARD_DAMAGE,
                    ENERGY_COST, FOOD_EFFECTS, ACTION_DURATION,
                    WATER_DURATION_MULTIPLIER)

class Agent:
    MAX_ENERGY        = MAX_ENERGY
    START_ENERGY      = START_ENERGY
    MAX_HEALTH        = MAX_HEALTH
    START_HEALTH      = START_HEALTH
    DEFAULT_MAX_AGE   = DEFAULT_MAX_AGE
    STARVATION_DAMAGE = STARVATION_DAMAGE
    HAZARD_DAMAGE     = HAZARD_DAMAGE
    ENERGY_COST       = ENERGY_COST
    FOOD_EFFECTS      = FOOD_EFFECTS
    ACTION_DURATION   = ACTION_DURATION
    WATER_DURATION_MULTIPLIER = WATER_DURATION_MULTIPLIER

    def __init__(self, world, max_age=None):
        self.world    = world
        self.max_age  = max_age if max_age is not None else self.DEFAULT_MAX_AGE
        self.position, self.facing = world.spawn_point()
        self.energy   = self.START_ENERGY
        self.health   = self.START_HEALTH
        self.age      = 0
        self.alive    = True
        self.busy_ticks_remaining = 0   # 0 = free to choose a new action

    @property
    def is_free(self):
        return self.busy_ticks_remaining <= 0

    def commit_action(self, action, water=False):
        """Call once, on the tick a NEW action is submitted to world.step().
        `water` (True iff this move's destination is a water cell — see
        World.step's entered_water) multiplies a forward/backward move's
        duration by WATER_DURATION_MULTIPLIER: since every busy tick still
        costs ENERGY_COST regardless of action, a longer duration is
        automatically a proportionally larger energy cost too — one
        multiplier models both "wading takes longer" and "wading tires you
        out more" (see config.py's Terrain section)."""
        duration = self.ACTION_DURATION[action]
        if water and action in (1, 2):
            duration *= self.WATER_DURATION_MULTIPLIER
        self.busy_ticks_remaining = duration - 1

    def cooldown_tick(self):
        """Call once per forced-STAY tick while busy_ticks_remaining > 0."""
        self.busy_ticks_remaining -= 1

    @property
    def internal_state(self):
        return np.array([
            self.energy / self.MAX_ENERGY,
            self.health / self.MAX_HEALTH,
            self.age    / self.max_age,
        ], dtype=float)

    def eat(self, food_type):
        effect = self.FOOD_EFFECTS[food_type]
        self.energy = min(self.energy + effect["energy"], self.MAX_ENERGY)
        self.health = min(self.health + effect["health"], self.MAX_HEALTH)

    def take_hazard_damage(self):
        self.health -= self.HAZARD_DAMAGE

    def apply_energy_cost(self, action):
        """One tick's worth of energy cost for `action`, plus starvation
        damage if that drains the tank to zero. Centralised here (instead of
        duplicated in every driver loop) so a change to the cost/starvation
        rule only needs to happen in one place. Returns True on a tick
        starvation damage was actually applied (energy was already at/hit 0)
        so callers can mirror it in reward shaping the same way a hazard tick
        already is (see run_episode.compute_reward's STARVATION_PENALTY) —
        starving and standing on a hazard are both "taking lethal damage this
        tick", so both should be legible to the reward, not just one of them."""
        self.energy -= self.ENERGY_COST[action]
        if self.energy <= 0:
            self.energy = 0
            self.health -= self.STARVATION_DAMAGE
            return True
        return False

    def apply_event(self, event):
        """Apply a world.step() event (food eaten / hazard standing) to the
        body. Returns the event unchanged for the caller's own bookkeeping
        (counters, reward)."""
        if event is not None:
            if event["category"] == "food":
                self.eat(event["type"])
            elif event["category"] == "hazard":
                self.take_hazard_damage()
        return event

    def advance_age(self):
        """One tick older; updates `alive` from health/age. Call once per
        tick after applying event + energy cost."""
        self.age += 1
        self.alive = self.health > 0 and self.age < self.max_age
        return self.alive
