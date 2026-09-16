import numpy as np
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from config import (VISION_RANGE, VISION_WIDTH, BEHIND_ROWS,
                    WORLD_SIZE, NUM_FOOD_LOW, NUM_FOOD_HIGH, NUM_HAZARDS,
                    NUM_FOOD_STARTER,
                    NUM_WATER_PATCHES, WATER_PATCH_SIZE,
                    NUM_WALL_SEGMENTS, WALL_SEGMENT_LEN, SOIL_FRACTION,
                    FOOD_SEED_BASE_RATE, FOOD_SEED_SPREAD_BONUS,
                    FOOD_SEED_MATURATION_MIN, FOOD_SEED_MATURATION_MAX,
                    FOOD_GROWTH_TYPE_SPLIT, MAX_FOOD_ON_WORLD,
                    FOOD_AGE_STAGE_TICKS)


class World:
    """
    Facing directions (clockwise): 0=up, 1=right, 2=down, 3=left

    Local-view cell codes (keep consistent everywhere the view is decoded):
      0 = behind agent (unknown)
      1 = wall (terrain — interior OR world boundary, see get_local_view)
      2 = water (terrain, walkable but costs extra — see config.py)
      3 = soil (terrain, freely walkable)
      4 = grass (terrain, freely walkable)
      5 = food_low (food_starter also renders as this code — see ENTITY_SPECS)
      6 = food_high
      7 = hazard
    food_starter is a SEPARATE entity type (own placement/respawn rules, see
    ENTITY_SPECS and _init_entities) but deliberately shares food_low's code
    5 rather than getting its own — the point of food_starter is a guaranteed
    early win, and it only ever exists for (at most) one bite per lifetime,
    so giving it a unique one-hot channel would mean the network gets almost
    no exposure to learn what that channel means; sharing food_low's code
    means the network recognizes it immediately as "the food it already knows
    about", which is the whole point. (The GIF/live_viewer still draw it with
    a visually distinct marker for a human watching — that's a separate,
    human-facing rendering layer, not what the agent's observation encodes.)
    Terrain codes (1-4) only ever appear where no entity currently occupies
    the cell — an entity's code always takes priority when both are present
    (get_local_view shows "what's actually there", and only one thing can be
    "there" per cell in this simple x-ray-view model; terrain doesn't occlude
    vision of what's on top of it). The mapping and one-hot channel count live
    in src/config.py (NUM_CELL_CLASSES) — keep this table and that constant
    in sync.

    Entities are stored as a flat list of {"type": str, "pos": (x, y)}.
    At most one entity occupies a given cell at a time. This structure is
    generic on purpose — adding a new entity type later means adding one
    entry to ENTITY_SPECS, not touching the grid / step logic.

    Terrain is a SEPARATE, persistent grid (self.terrain, generated once in
    __init__ — see _init_terrain) that entities sit on top of. Unlike
    entities it never changes during a lifetime. wall terrain blocks movement
    (World.step reverts a move that would land on one); water terrain is
    walkable but the RL layer (Agent.commit_action) charges extra busy-ticks
    for a move that lands on it — World.step signals this via the 4th return
    value, entered_water.

    The local-view geometry defaults (VISION_RANGE / VISION_WIDTH /
    BEHIND_ROWS) are read from src/config.py (the single source of truth) so a
    plain `World()` always produces a view whose shape the Conv-LSTM was built
    for. Override them only if you re-build the network to match.
    """
    FACING_VECTORS = {0:(0,1), 1:(1,0), 2:(0,-1), 3:(-1,0)}

    CELL_UNKNOWN = 0
    CELL_WALL    = 1
    CELL_WATER   = 2
    CELL_SOIL    = 3
    CELL_GRASS   = 4

    # entity type -> local-view code + reward/behavior category (+ optional
    # "respawns": False for a one-shot entity that's removed, not repositioned,
    # once eaten/triggered — defaults to True via ENTITY_SPECS[...].get(...)).
    # food_low/food_high are respawns=False now too (like food_starter): new
    # food no longer teleports in instantly on eat, it only comes from the
    # per-tick growth simulation (_advance_food_growth) — see config.py's
    # "Food growth" section for why. Hazard is the only entity that still
    # instantly relocates (_respawn_entity), unaffected by this change.
    ENTITY_SPECS = {
        "food_low":     {"code": 5, "category": "food", "respawns": False},
        "food_high":    {"code": 6, "category": "food", "respawns": False},
        "hazard":       {"code": 7, "category": "hazard"},
        # code intentionally == food_low's (see class docstring) — NOT a new
        # channel, so NUM_CELL_CLASSES is NOT bumped for food_starter.
        "food_starter": {"code": 5, "category": "food", "respawns": False},
        # Rotten food: gets its OWN code/channel (8) so the network can
        # distinguish it from fresh food. Eating it applies a negative energy
        # effect (see config.FOOD_EFFECTS). "respawns": False — it simply
        # disappears once eaten (or after its aging stage expires).
        "rotten_food":  {"code": 8, "category": "food", "respawns": False},
    }

    def __init__(self, width=WORLD_SIZE, height=WORLD_SIZE,
                 vision_range=VISION_RANGE,
                 behind_rows=BEHIND_ROWS, vision_width=VISION_WIDTH,
                 num_food_low=NUM_FOOD_LOW, num_food_high=NUM_FOOD_HIGH,
                 num_hazards=NUM_HAZARDS, num_food_starter=NUM_FOOD_STARTER,
                 num_water_patches=NUM_WATER_PATCHES, water_patch_size=WATER_PATCH_SIZE,
                 num_wall_segments=NUM_WALL_SEGMENTS, wall_segment_len=WALL_SEGMENT_LEN,
                 soil_fraction=SOIL_FRACTION):
        self.width        = width
        self.height       = height
        self.time         = 0
        self.vision_range = vision_range
        self.behind_rows  = behind_rows
        self.vision_width = vision_width
        self.num_water_patches = num_water_patches
        self.water_patch_size  = water_patch_size
        self.num_wall_segments = num_wall_segments
        self.wall_segment_len  = wall_segment_len
        self.soil_fraction     = soil_fraction
        self.entity_counts = {
            "food_low":     num_food_low,
            "food_high":    num_food_high,
            "hazard":       num_hazards,
            "food_starter": num_food_starter,
        }
        self.entities = []  # list of {"type": str, "pos": (x, y)}
        self.seeds = {}     # {(x, y): ticks_remaining} — pending food growth, see _advance_food_growth
        self.food_ages = {} # {id(entity): ticks_at_current_stage} — food aging, see _advance_food_aging
        self._init_terrain()   # BEFORE entities: entity spawn avoids wall cells
        self._init_entities()

    def spawn_point(self):
        """Where a newly-created Agent starts: world center, facing "up" (0).
        Single source of truth used by both World (terrain generation keeps
        this cell + the cell directly in front of it clear of wall/water, and
        food_starter is placed exactly in front of it) and Agent.__init__ —
        so the two can never silently drift apart."""
        return (self.width // 2, self.height // 2), 0

    @property
    def frame_size(self):
        # total cells in the local view = VIEW_W * VIEW_H (rows x cols)
        return self.vision_width * (self.vision_range + self.behind_rows)

    @property
    def food_positions(self):
        """All food-category entity positions (any type). Kept for callers
        that only care 'is there food nearby', not which kind."""
        return self.positions_by_category("food")

    @property
    def hazard_positions(self):
        return self.positions_by_category("hazard")

    def positions_by_category(self, category):
        return [e["pos"] for e in self.entities
                if self.ENTITY_SPECS[e["type"]]["category"] == category]

    def positions_by_type(self, entity_type):
        return [e["pos"] for e in self.entities if e["type"] == entity_type]

    def visible_entity_positions(self, position, facing, category=None):
        """Absolute positions of entities inside the agent's FORWARD vision
        cone from `position`/`facing` — i.e. exactly the cells get_local_view()
        would mark with something other than CELL_UNKNOWN in the forward rows
        (rows behind the agent are always unknown, so never counted visible).
        Optionally filter to one ENTITY_SPECS category ("food"/"hazard").

        Used by reward shaping (src/rl/run_episode.py:compute_reward) so the
        "move toward food / away from hazard" shaping signal only fires for
        entities the agent could actually have perceived when it chose the
        action — using ALL entities regardless of visibility (the previous
        behavior) leaks privileged global-state information into a POMDP:
        potential-based shaping (Ng et al. 1999) is only proven policy-
        invariant under full observability, and two states that look
        identical through the local view could otherwise receive different
        shaping just because of where an off-screen food/hazard happens to be
        — a noisy, unlearnable signal from the network's point of view.
        """
        fx, fy = self._fwd(facing)
        rx, ry = self._right(facing)
        px, py = position
        half        = self.vision_width // 2
        col_offsets = range(-half, self.vision_width - half)

        visible_cells = set()
        for row in range(self.vision_range):
            dist = self.vision_range - row
            for off in col_offsets:
                visible_cells.add((px + fx*dist + rx*off, py + fy*dist + ry*off))

        out = []
        for e in self.entities:
            if e["pos"] not in visible_cells:
                continue
            if category is not None and self.ENTITY_SPECS[e["type"]]["category"] != category:
                continue
            out.append(e["pos"])
        return out

    # ── Zone helpers ─────────────────────────────────────────────────────────
    ZONE_SIZE = 3   # each zone is ZONE_SIZE x ZONE_SIZE cells

    def _zones(self):
        """Return list of (zone_col, zone_row) indices that fit in the grid."""
        n_cols = max(1, self.width  // self.ZONE_SIZE)
        n_rows = max(1, self.height // self.ZONE_SIZE)
        return [(zc, zr) for zr in range(n_rows) for zc in range(n_cols)]

    def _cells_in_zone(self, zone_col, zone_row):
        """All (x, y) cells that belong to zone (zone_col, zone_row)."""
        x0 = zone_col * self.ZONE_SIZE
        y0 = zone_row * self.ZONE_SIZE
        return [(x, y)
                for x in range(x0, min(x0 + self.ZONE_SIZE, self.width))
                for y in range(y0, min(y0 + self.ZONE_SIZE, self.height))]

    def _random_free_cell_in_zone(self, zone_col, zone_row, avoid=None):
        """Pick a random free cell inside the given zone. Falls back to any
        free cell in the zone; if the entire zone is occupied, falls back to
        the global _random_free_cell."""
        occupied = {e["pos"] for e in self.entities}
        if avoid is not None:
            occupied.add(avoid)
        candidates = [c for c in self._cells_in_zone(zone_col, zone_row)
                      if c not in occupied]
        if candidates:
            idx = np.random.randint(len(candidates))
            return candidates[idx]
        # zone fully occupied — fall back to global
        return self._random_free_cell(avoid=avoid)

    def _random_free_cell(self, avoid=None):
        occupied = {e["pos"] for e in self.entities}
        if avoid is not None:
            occupied.add(avoid)
        while True:
            cell = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            if cell not in occupied:
                return cell

    def random_adjacent_cell(self, pos):
        """A free (non-wall), in-bounds cell orthogonally adjacent to `pos`,
        or None if all 4 neighbors are wall/out-of-bounds. Used by
        src/rl/population.py to place a newborn agent next to its parent.
        Does NOT consider other agents' positions — agents don't block or
        occupy cells exclusively from each other's perspective in this first
        ecosystem phase (see progress.md), only terrain/entities matter here,
        exactly like every other placement helper in this file."""
        x, y = pos
        candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        candidates = [c for c in candidates
                     if self._in_bounds(c) and self.terrain[c[0], c[1]] != self.CELL_WALL]
        if not candidates:
            return None
        return candidates[np.random.randint(len(candidates))]

    def terrain_at(self, pos):
        """Terrain code at `pos`; out-of-bounds counts as wall (the world
        boundary has always behaved like a wall — see get_local_view/step)."""
        if not self._in_bounds(pos):
            return self.CELL_WALL
        x, y = pos
        return int(self.terrain[x, y])

    def _init_terrain(self):
        """Generate this world's static terrain grid: a random grass/soil base
        fill, then a few small water patches + wall line-segments carved out
        of it (see config.py's Terrain section for the reasoning + tunables).
        Always run BEFORE _init_entities() so entity spawning can avoid wall
        cells, and before Agent.__init__ picks its center spawn cell.
        """
        soil_mask = np.random.random((self.width, self.height)) < self.soil_fraction
        self.terrain = np.where(soil_mask, self.CELL_SOIL, self.CELL_GRASS).astype(int)

        # Never let the agent spawn already boxed in — the world-center cell
        # AND the cell directly in front of it (spawn_point(), where
        # food_starter will be placed) are always kept clear of wall/water.
        center, facing = self.spawn_point()
        fx, fy = self._fwd(facing)
        front = (center[0] + fx, center[1] + fy)
        protected = {center, front} if self._in_bounds(front) else {center}

        for _ in range(self.num_water_patches):
            self._carve_rect_patch(self.CELL_WATER, self.water_patch_size, protected)
        for _ in range(self.num_wall_segments):
            self._carve_wall_segment(self.wall_segment_len, protected)

    def _carve_rect_patch(self, terrain_code, size, protected, max_attempts=20):
        """Try to stamp a size x size square of `terrain_code`, retrying a
        random top-left corner up to max_attempts times if it would cover a
        protected cell. Silently gives up (skips this patch) rather than
        crashing if it can't find a spot — a missing decorative patch is
        harmless, unlike an infinite loop on a small/crowded world."""
        for _ in range(max_attempts):
            x0 = np.random.randint(0, max(1, self.width - size + 1))
            y0 = np.random.randint(0, max(1, self.height - size + 1))
            cells = [(x, y) for x in range(x0, x0 + size)
                     for y in range(y0, y0 + size) if self._in_bounds((x, y))]
            if any(c in protected for c in cells):
                continue
            for x, y in cells:
                self.terrain[x, y] = terrain_code
            return

    def _carve_wall_segment(self, length, protected, max_attempts=20):
        """Try to stamp a straight wall segment (random orientation + start),
        retrying like _carve_rect_patch."""
        for _ in range(max_attempts):
            horizontal = np.random.random() < 0.5
            if horizontal:
                x0 = np.random.randint(0, max(1, self.width - length + 1))
                y0 = np.random.randint(0, self.height)
                cells = [(x0 + i, y0) for i in range(length)]
            else:
                x0 = np.random.randint(0, self.width)
                y0 = np.random.randint(0, max(1, self.height - length + 1))
                cells = [(x0, y0 + i) for i in range(length)]
            cells = [c for c in cells if self._in_bounds(c)]
            if any(c in protected for c in cells):
                continue
            for x, y in cells:
                self.terrain[x, y] = self.CELL_WALL
            return

    def _init_entities(self):
        """Place entities so food/hazard are spread across 3x3 zones.

        Strategy:
          0. food_starter (if configured) is placed FIRST, at a fixed cell
             (one step in front of spawn_point()) — not zone-random like the
             rest, since its whole point is a guaranteed, predictable spot a
             newborn agent immediately sees. Placed before everything else so
             the zone-based placement below (which checks occupied cells via
             _random_free_cell*) never overlaps it.
          1. Shuffle the zone list so assignment order is random each episode.
          2. food_low + food_high are pooled into ONE shuffled list and
             round-robined across the SAME shuffled zone order together (not
             two separate per-type passes) — this is what actually guarantees
             >=1 food per zone once the combined count reaches the zone count
             (see config.py's comment on NUM_FOOD_LOW/HIGH); round-robining
             each type separately could leave a zone with neither type just
             by chance even if the totals would otherwise cover every zone.
          3. Hazards get the same zone-balanced treatment independently.
          4. Within each assigned zone, the specific cell is chosen randomly.
        """
        self.entities = []
        zones = self._zones()

        num_starter = self.entity_counts.get("food_starter", 0)
        if num_starter > 0:
            center, facing = self.spawn_point()
            fx, fy = self._fwd(facing)
            front = (center[0] + fx, center[1] + fy)
            if not self._in_bounds(front):
                front = center   # degenerate fallback for a tiny/edge-case world
            self.entities.append({"type": "food_starter", "pos": front})

        food_types = [t for t, spec in self.ENTITY_SPECS.items()
                      if spec["category"] == "food" and t != "food_starter"]
        food_plan = []
        for t in food_types:
            food_plan += [t] * self.entity_counts.get(t, 0)
        np.random.shuffle(food_plan)   # which TYPE lands in which zone is random

        zone_order = zones.copy()
        np.random.shuffle(zone_order)
        for i, entity_type in enumerate(food_plan):
            zc, zr = zone_order[i % len(zone_order)]
            pos = self._random_free_cell_in_zone(zc, zr)
            self.entities.append({"type": entity_type, "pos": pos})

        non_food_types = [t for t in self.entity_counts
                          if t not in food_types and t != "food_starter"]
        for entity_type in non_food_types:
            count = self.entity_counts[entity_type]
            zone_order = zones.copy()
            np.random.shuffle(zone_order)
            for i in range(count):
                zc, zr = zone_order[i % len(zone_order)]
                pos = self._random_free_cell_in_zone(zc, zr)
                self.entities.append({"type": entity_type, "pos": pos})

    def _zone_food_counts(self):
        """Return dict {(zc, zr): food_count} for all zones."""
        zones = self._zones()
        counts = {z: 0 for z in zones}
        for e in self.entities:
            if self.ENTITY_SPECS[e["type"]]["category"] == "food":
                x, y = e["pos"]
                zc = x // self.ZONE_SIZE
                zr = y // self.ZONE_SIZE
                key = (zc, zr)
                if key in counts:
                    counts[key] += 1
        return counts

    def _respawn_entity(self, entity, avoid_position=None):
        """Respawn into the zone that currently has the fewest entities of
        the same category. In practice only ever called for hazard now (all
        food types are respawns=False, see ENTITY_SPECS) — kept generic
        rather than hazard-only in case a future entity wants this exact
        "instant relocation, zone-balanced" behavior again."""
        if self.ENTITY_SPECS[entity["type"]]["category"] == "food":
            zone_counts = self._zone_food_counts()
            # pick zone(s) with minimum food
            min_count = min(zone_counts.values())
            sparse_zones = [z for z, c in zone_counts.items() if c == min_count]
            zc, zr = sparse_zones[np.random.randint(len(sparse_zones))]
            entity["pos"] = self._random_free_cell_in_zone(zc, zr,
                                                           avoid=avoid_position)
        else:
            entity["pos"] = self._random_free_cell(avoid=avoid_position)

    def _advance_food_growth(self):
        """One tick of the food growth simulation — see config.py's "Food
        growth" section for the full design rationale. Called once per tick
        from step(), regardless of what action happened that tick (growth is
        a background world process, independent of the agent).

        Two stages:
          1. Mature any seeds whose countdown reached 0 into a real food
             entity (type chosen via FOOD_GROWTH_TYPE_SPLIT).
          2. If there's still room under MAX_FOOD_ON_WORLD, roll for NEW
             seeds on eligible empty grass/soil cells — base rate always
             applies; a spreading bonus additionally applies wherever an
             orthogonally-adjacent cell currently holds a MATURE food item
             (not another seed — only grown food "reproduces").
        """
        # ---- 1. mature existing seeds ----
        matured = []
        for pos in list(self.seeds.keys()):
            self.seeds[pos] -= 1
            if self.seeds[pos] <= 0:
                matured.append(pos)
        for pos in matured:
            del self.seeds[pos]
            food_type = ("food_low" if np.random.random() < FOOD_GROWTH_TYPE_SPLIT["food_low"]
                        else "food_high")
            entity = {"type": food_type, "pos": pos}
            self.entities.append(entity)
            self.food_ages[id(entity)] = (0, 0)   # (ticks_in_stage, stage_idx)

        # ---- 2. roll for new seeds (vectorized over the whole grid) ----
        num_food = len(self.positions_by_category("food"))
        if num_food + len(self.seeds) >= MAX_FOOD_ON_WORLD:
            return   # at carrying capacity — no new seeds this tick

        has_food = np.zeros((self.width, self.height), dtype=bool)
        for pos in self.positions_by_category("food"):
            has_food[pos[0], pos[1]] = True
        # orthogonal-neighbor OR, with edges padded False (no wraparound)
        neighbor_has_food = np.zeros_like(has_food)
        neighbor_has_food[1:,  :] |= has_food[:-1, :]   # neighbor to the west
        neighbor_has_food[:-1, :] |= has_food[1:,  :]   # neighbor to the east
        neighbor_has_food[:, 1:]  |= has_food[:, :-1]   # neighbor to the south
        neighbor_has_food[:, :-1] |= has_food[:, 1:]    # neighbor to the north

        prob = np.zeros((self.width, self.height), dtype=float)
        for terrain_code, terrain_name in ((self.CELL_SOIL, "soil"), (self.CELL_GRASS, "grass")):
            mask = self.terrain == terrain_code
            prob[mask] = FOOD_SEED_BASE_RATE[terrain_name]
            spread_mask = mask & neighbor_has_food
            prob[spread_mask] += FOOD_SEED_SPREAD_BONUS[terrain_name]

        # ineligible: already an entity there, or already seeded
        for e in self.entities:
            prob[e["pos"][0], e["pos"][1]] = 0.0
        for pos in self.seeds:
            prob[pos[0], pos[1]] = 0.0

        roll = np.random.random((self.width, self.height))
        new_seed_cells = np.argwhere(roll < prob)
        for x, y in new_seed_cells:
            ticks = np.random.randint(FOOD_SEED_MATURATION_MIN, FOOD_SEED_MATURATION_MAX + 1)
            self.seeds[(int(x), int(y))] = int(ticks)

    def _advance_food_aging(self):
        """Advance the aging lifecycle of all food entities that have a tracked
        age (food_starter is exempt — it exists until eaten, see config.py).

        Lifecycle per stage (each stage = FOOD_AGE_STAGE_TICKS ticks):
          food_low  -> food_high
          food_high -> food_low
          food_low  -> rotten_food
          rotten_food -> removed from world

        The stage counter is keyed by id(entity) in self.food_ages. When an
        entity transitions, its age counter resets to 0 for the next stage.
        When rotten_food expires, it's removed entirely (no respawn, no growth
        credit — it just rots away). food_starter entities are never added to
        food_ages so they are silently skipped here.
        """
        # Aging lifecycle transitions in order
        NEXT_STAGE = {
            "food_low":    "food_high",
            "food_high":   "food_low",
            # second food_low stage -> rotten: tracked by counting transitions
            "rotten_food": None,   # None = remove from world
        }
        # food_low can be either "stage 1" (-> food_high) or "stage 3" (-> rotten).
        # We track this by storing the number of completed transitions per entity
        # in food_ages as (ticks_in_stage, completed_transitions).
        # For simplicity: food_ages[id] = [ticks_elapsed, stage_index]
        # stage_index: 0=food_low->high, 1=food_high->low, 2=food_low->rotten, 3=rotten->gone

        to_remove = []
        for entity in list(self.entities):
            eid = id(entity)
            if eid not in self.food_ages:
                continue   # food_starter or entity not tracked
            ticks, stage_idx = self.food_ages[eid]
            ticks += 1
            if ticks < FOOD_AGE_STAGE_TICKS:
                self.food_ages[eid] = (ticks, stage_idx)
                continue
            # Stage complete — advance
            next_stage_idx = stage_idx + 1
            if next_stage_idx == 1:
                entity["type"] = "food_high"
            elif next_stage_idx == 2:
                entity["type"] = "food_low"
            elif next_stage_idx == 3:
                entity["type"] = "rotten_food"
            else:
                # rotten stage done -> remove
                to_remove.append(entity)
                del self.food_ages[eid]
                continue
            self.food_ages[eid] = (0, next_stage_idx)

        for entity in to_remove:
            self.entities.remove(entity)

    def entity_at(self, pos):
        for e in self.entities:
            if e["pos"] == pos:
                return e
        return None

    def _fwd(self, facing):
        return self.FACING_VECTORS[facing]

    def _right(self, facing):
        fx, fy = self.FACING_VECTORS[facing]
        return (fy, -fx)

    def _in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def get_local_view(self, position, facing):
        """
        4x4 grid rotated to agent facing. See class docstring for cell codes.

        Row 0 = farthest visible (distance=vision_range)
        Row (vision_range-1) = immediately ahead
        Rows >= vision_range = behind agent (always CELL_UNKNOWN)

        Collapses terrain + entity into ONE code per cell (entity wins if
        both present) — kept for callers that just want a single
        representative code to DISPLAY per cell (e.g. live_viewer's mini
        local-vision panel). The network's actual observation encoding does
        NOT use this — see get_local_view_layers() below for the version that
        keeps terrain and entity as independent bits (multi-hot).
        """
        fx, fy = self._fwd(facing)
        rx, ry = self._right(facing)
        px, py = position
        half        = self.vision_width // 2
        col_offsets = list(range(-half, self.vision_width - half))

        grid = np.full((self.vision_range + self.behind_rows, self.vision_width),
                        self.CELL_UNKNOWN, dtype=int)
        for row in range(self.vision_range):
            dist = self.vision_range - row
            for ci, off in enumerate(col_offsets):
                cell = (px + fx*dist + rx*off, py + fy*dist + ry*off)
                if not self._in_bounds(cell):
                    grid[row, ci] = self.CELL_WALL
                    continue
                entity = self.entity_at(cell)
                if entity is not None:
                    grid[row, ci] = self.ENTITY_SPECS[entity["type"]]["code"]
                else:
                    grid[row, ci] = self.terrain[cell[0], cell[1]]
        # rows [vision_range:] stay CELL_UNKNOWN (behind — unknown)
        return grid

    def get_local_view_layers(self, position, facing):
        """
        Like get_local_view() but keeps terrain and entity as two INDEPENDENT
        grids instead of collapsing them into one code per cell — this is
        what lstm_q_network.encode_observation() actually uses to build a
        TRUE multi-hot (not mutually-exclusive) observation: a cell with,
        say, food_low sitting on water sets BOTH the water bit and the
        food_low bit in the final vector, instead of the entity's code
        silently hiding what terrain is underneath it.

        Returns (terrain_grid, entity_grid), both shape
        (vision_range+behind_rows, vision_width):
          terrain_grid: CELL_WALL/WATER/SOIL/GRASS for a visible cell (wall
            also covers out-of-bounds, same as get_local_view), or
            CELL_UNKNOWN(0) for a behind-agent row. Exactly one value per
            cell — terrain (including "unknown") is mutually exclusive with
            itself, just not with entity_grid.
          entity_grid: the cell's ENTITY_SPECS code if an entity is there,
            else -1 (sentinel for "no entity") — completely independent of
            terrain_grid, so both can be "true" for the same cell at once.
        """
        fx, fy = self._fwd(facing)
        rx, ry = self._right(facing)
        px, py = position
        half        = self.vision_width // 2
        col_offsets = list(range(-half, self.vision_width - half))

        shape = (self.vision_range + self.behind_rows, self.vision_width)
        terrain_grid = np.full(shape, self.CELL_UNKNOWN, dtype=int)
        entity_grid  = np.full(shape, -1, dtype=int)

        for row in range(self.vision_range):
            dist = self.vision_range - row
            for ci, off in enumerate(col_offsets):
                cell = (px + fx*dist + rx*off, py + fy*dist + ry*off)
                if not self._in_bounds(cell):
                    terrain_grid[row, ci] = self.CELL_WALL
                    continue
                terrain_grid[row, ci] = self.terrain[cell[0], cell[1]]
                entity = self.entity_at(cell)
                if entity is not None:
                    entity_grid[row, ci] = self.ENTITY_SPECS[entity["type"]]["code"]
        # rows [vision_range:] stay CELL_UNKNOWN / -1 (behind agent, nothing known)
        return terrain_grid, entity_grid

    def step(self, action, position, facing):
        """
        Actions: 0=stay, 1=forward, 2=backward, 3=turn_left, 4=turn_right

        Returns (new_position, new_facing, event, entered_water).
        event is None, or {"type": entity_type, "category": "food"|"hazard"}.
        entered_water is True iff this was a forward/backward move whose
        destination cell is water (used by Agent.commit_action to charge
        extra busy-ticks for wading — see config.WATER_DURATION_MULTIPLIER).

        - A move (forward/backward) into a WALL cell is blocked: the agent
          "bumps" and new_position == position (still costs the tick's normal
          energy, just like any other action — World doesn't apply energy
          costs, see Agent.apply_energy_cost).
        - A "food" event fires once, the step the agent lands on that food
          cell; the food is immediately respawned elsewhere UNLESS its
          ENTITY_SPECS entry sets "respawns": False (e.g. food_starter), in
          which case it's removed instead — gone for the rest of the lifetime.
        - A "hazard" event fires every step the agent's resulting position
          coincides with a hazard cell (including standing still on it) —
          hazards are not consumed/respawned.
        Applying damage/healing and shaping reward from these events is the
        RL layer's responsibility, not World's.
        """
        fx, fy     = self._fwd(facing)
        new_pos    = position
        new_facing = facing

        if action == 1:
            candidate = (min(max(position[0]+fx, 0), self.width-1),
                        min(max(position[1]+fy, 0), self.height-1))
            if self.terrain_at(candidate) != self.CELL_WALL:
                new_pos = candidate
        elif action == 2:
            candidate = (min(max(position[0]-fx, 0), self.width-1),
                        min(max(position[1]-fy, 0), self.height-1))
            if self.terrain_at(candidate) != self.CELL_WALL:
                new_pos = candidate
        elif action == 3:
            new_facing = (facing - 1) % 4
        elif action == 4:
            new_facing = (facing + 1) % 4
        elif action != 0:
            raise ValueError(f"Unknown action: {action}")

        entered_water = (action in (1, 2) and new_pos != position
                        and self.terrain_at(new_pos) == self.CELL_WATER)

        event  = None
        entity = self.entity_at(new_pos)
        if entity is not None:
            spec  = self.ENTITY_SPECS[entity["type"]]
            event = {"type": entity["type"], "category": spec["category"]}
            if spec["category"] == "food":
                if spec.get("respawns", True):
                    self._respawn_entity(entity, avoid_position=new_pos)
                else:
                    self.food_ages.pop(id(entity), None)   # clean up age tracker
                    self.entities.remove(entity)

        self.time += 1
        self._advance_food_growth()
        self._advance_food_aging()
        return new_pos, new_facing, event, entered_water
