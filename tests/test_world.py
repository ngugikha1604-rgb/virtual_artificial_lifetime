import numpy as np
from world import World


def test_spawn_point_never_on_wall_or_hazard():
    """Regression test for the 2026-09 random-spawn change — verified
    manually against 500 seeds at the time, now checked automatically."""
    for seed in range(200):
        np.random.seed(seed)
        w = World()
        (sx, sy), facing = w.spawn_point()
        assert 0 <= sx < w.width and 0 <= sy < w.height
        assert w.terrain_at((sx, sy)) != World.CELL_WALL
        ent = w.entity_at((sx, sy))
        if ent is not None:
            assert World.ENTITY_SPECS[ent["type"]]["category"] != "hazard"


def test_spawn_point_is_cached_not_rerolled():
    """spawn_point() must return the SAME value on every call within one
    World — _init_terrain() and _init_entities() both call it and must agree
    on the same cell, or terrain protection and food_starter placement
    desync (see World._roll_spawn_point's docstring)."""
    np.random.seed(0)
    w = World()
    assert w.spawn_point() == w.spawn_point()


def test_every_zone_has_at_least_one_food():
    """Regression test for the food-distribution invariant (config.py's
    NUM_FOOD_LOW + NUM_FOOD_HIGH == zone count comment) — verified manually
    against 200 seeds after the 2026-09 spawn-reservation change."""
    for seed in range(100):
        np.random.seed(seed)
        w = World()
        counts = w._zone_food_counts()
        assert counts, "expected at least one zone"
        assert all(c >= 1 for c in counts.values())
