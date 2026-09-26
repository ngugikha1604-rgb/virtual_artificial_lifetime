import sys
from pathlib import Path
import numpy as np
import pygame

# Set up paths
RL_DIR      = Path(__file__).resolve().parent
ROOT_DIR    = RL_DIR.parent.parent
BRAIN_DIR   = ROOT_DIR / "src" / "brain"
WORLD_DIR   = ROOT_DIR / "src" / "world"
RESULTS_DIR = ROOT_DIR / "results"
SRC_DIR     = ROOT_DIR / "src"

for d in (RL_DIR, BRAIN_DIR, WORLD_DIR, SRC_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from world import World
from lstm_q_network import build_lstm_network
from model_io import load_weights_or_fresh
from population import spawn_founder, ecosystem_step, check_and_save_best
from training_state import load_training_state
from config import ECOSYSTEM_SAVE_EVERY_TICKS, LIVE_VIEWER_START_EPSILON

# ── Viewer defaults (2026-09) ─────────────────────────────────────────────
# One single interactive viewer now (2026-09: merged the short-lived
# god_view.py experiment back in here — the two had ended up almost
# identical, so Khanh asked to keep only one). This IS the "watch a whole
# miniature world from above" experience: a much bigger world than the
# 10x10 it trains on, run continuously and slowly, styled to be readable
# and a little fun even for someone who has no idea what a Q-value is.
#
# VIEWER_WORLD_SIZE is 5x config.WORLD_SIZE (10 -> 50) — a pure viewer-side
# default, does NOT touch config.py (training at 10x10 is unaffected).
#
# Food/hazard counts do NOT just reuse config.NUM_FOOD_LOW/HIGH/NUM_HAZARDS
# (5/4/3) unscaled: those were sized for a 10x10 world's zone grid ((10//3)**2
# = 9 zones; World._init_entities's "at least one food per zone" guarantee
# only holds because NUM_FOOD_LOW+NUM_FOOD_HIGH=9 exactly covers 9 zones — see
# config.py's comment). A 50x50 world has (50//3)**2 = 256 zones, so reusing
# 5/4/3 unscaled would leave most of a 2500-cell world completely empty.
# Instead, scaled by area (50x50 / 10x10 = 25x) AND rebalanced to again
# exactly cover every zone (256 * 5/9 ~= 142 food_low, 256 * 4/9 ~= 114
# food_high, 142+114=256 covers all 256 zones). Hazards don't have a "cover
# every zone" guarantee to preserve, so a plain area-scale (3 * 25 = 75) is
# enough.
VIEWER_WORLD_SIZE     = 50
VIEWER_NUM_FOOD_LOW   = 142
VIEWER_NUM_FOOD_HIGH  = 114
VIEWER_NUM_HAZARDS    = 75
VIEWER_TARGET_FPS     = 2      # slow/contemplative by default, still adjustable live

# Colors
COLOR_BG        = (24, 26, 33)
COLOR_CARD      = (38, 42, 51)
COLOR_GRID_BG   = (240, 243, 246)
COLOR_GRID_LINE = (210, 215, 222)
COLOR_FOOD_LOW  = (230, 126, 34)   # orange
COLOR_FOOD_HIGH = (241, 196, 15)   # gold
COLOR_FOOD_STARTER = (26, 188, 156)  # teal — visually distinct one-shot food
COLOR_FOOD_ROTTEN = (101, 67, 33)  # dark brown — visually distinct rotted food
COLOR_SEED      = (154, 205, 50)   # yellowgreen — small dot for a growing seed
COLOR_HAZARD    = (192, 57, 43)    # dark red
COLOR_WALL      = (90, 95, 105)    # gray
COLOR_WATER     = (93, 173, 226)   # blue
COLOR_SOIL      = (150, 111, 51)   # brown
COLOR_GRASS     = (163, 209, 130)  # light green
COLOR_TEXT      = (225, 229, 234)
COLOR_TEXT_DIM  = (150, 155, 165)
COLOR_ENERGY_GOOD = (46, 204, 113)
COLOR_ENERGY_WARN = (241, 196, 15)
COLOR_ENERGY_CRIT = (231, 76, 60)
COLOR_HEALTH_GOOD = (52, 152, 219)
COLOR_HEALTH_WARN = (230, 126, 34)
COLOR_HEALTH_CRIT = (192, 57, 43)
COLOR_BAR_BG    = (52, 57, 68)
COLOR_HIGHLIGHT = (155, 89, 182)
COLOR_EXTINCT   = (231, 76, 60)
COLOR_SPOTLIGHT_GLOW = (241, 196, 15)

# Creature body color, keyed by an overall wellbeing score (min(energy_pct,
# health_pct)) rather than a fixed color — this is the main "readable at a
# glance, even with zero ML background" upgrade (2026-09): a healthy world
# looks like a field of green triangles, a struggling one visibly reddens,
# no legend required to understand "this one looks like it's doing fine."
def _creature_color(wellbeing_pct):
    if wellbeing_pct > 0.6:
        return (46, 204, 113)    # thriving — green
    if wellbeing_pct > 0.3:
        return (241, 196, 15)    # getting by — amber
    return (231, 76, 60)         # struggling — red

# Local-view cell code -> color (see World docstring for the code table).
# NOTE: food_starter is NOT a separate code (it shares food_low's code 5, so
# the mini "local vision" panel correctly shows it identically to food_low —
# that's what the agent actually perceives). Its distinct COLOR_FOOD_STARTER
# marker below is only used in the main world-map drawing, which draws by
# entity TYPE, not by this code table — a human-only visual aid.
CELL_COLORS = {
    World.CELL_UNKNOWN: (28, 30, 36),     # behind / unknown (dark)
    World.CELL_WALL:    COLOR_WALL,
    World.CELL_WATER:   COLOR_WATER,
    World.CELL_SOIL:    COLOR_SOIL,
    World.CELL_GRASS:   COLOR_GRASS,
    5: COLOR_FOOD_LOW,                    # food_low (+ food_starter, same code)
    6: COLOR_FOOD_HIGH,                   # food_high
    7: COLOR_HAZARD,                      # hazard
    8: COLOR_FOOD_ROTTEN,                 # rotten_food
}

ACTION_NAMES  = ["\u0110\u1ee9ng y\u00ean", "Ti\u1ebfn", "L\u00f9i", "Quay tr\u00e1i", "Quay ph\u1ea3i"]
FACING_NAMES  = ["L\u00ean \u2191", "Ph\u1ea3i \u2192", "Xu\u1ed1ng \u2193", "Tr\u00e1i \u2190"]
FACING_ANGLES = {0: -np.pi/2, 1: 0, 2: np.pi/2, 3: np.pi}


def _mood_label(energy_pct, health_pct):
    """A one-line, jargon-free read on how the spotlighted creature is
    doing, derived from the same energy/health fractions that already drive
    the reward formula (config.py's SURVIVAL_BONUS) and the two stat bars —
    not a new signal, just a plain-language translation of an existing one
    for someone who has no reason to know what "energy" or "health" mean in
    this simulation otherwise."""
    worst = min(energy_pct, health_pct)
    if worst > 0.7:
        return "Kh\u1ecfe m\u1ea1nh, \u0111ang s\u1ed1ng t\u1ed1t", COLOR_ENERGY_GOOD
    if worst > 0.4:
        return "\u1ed4n, nh\u01b0ng n\u00ean t\u00ecm th\u00eam \u0103n", COLOR_ENERGY_WARN
    if worst > 0.15:
        return "\u0110ang \u0111\u00f3i / y\u1ebfu, kh\u00e1 nguy hi\u1ec3m", COLOR_ENERGY_CRIT
    return "S\u1eafp ki\u1ec7t s\u1ee9c!", COLOR_ENERGY_CRIT


class WorldViewer2D:
    """
    Interactive Pygame viewer for the ECOSYSTEM (src/rl/population.py) — NOT
    a single agent's single lifetime. Runs indefinitely, slowly, in a much
    bigger world than training uses, meant to be watched (2026-09 redesign,
    "god view": Khanh wanted something that reads like watching a small
    world from above, understandable and a little fun even for someone with
    zero ML background — plain-language panels, a click-to-follow creature,
    and body color that reads as "healthy" or "struggling" at a glance
    instead of a fixed color + a wall of Q-value numbers).

    It only stops on a real "population extinct" (all individuals dead,
    nobody left to reseed from — reset manually with R) or when you press R
    yourself to start a fresh population.

    Always founds a (re)started population from results/best_model.pt if it
    exists (same file run_experiment.py trains/checkpoints — see that file's
    module docstring), and keeps contributing back to that SAME shared
    best_model.pt/training_state.json as it runs — this viewer is one more
    training source feeding the shared best brain, not a sandboxed-off demo.
    """
    def __init__(self, world_size=VIEWER_WORLD_SIZE, num_food_low=VIEWER_NUM_FOOD_LOW,
                 num_food_high=VIEWER_NUM_FOOD_HIGH, num_hazards=VIEWER_NUM_HAZARDS,
                 trained_weights_path=None):
        pygame.init()
        pygame.font.init()

        self.width = 1260
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Th\u1ebf Gi\u1edbi \u1ea2o \u2014 Quan s\u00e1t nh\u01b0 m\u1ed9t v\u1ecb th\u1ea7n")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_main  = pygame.font.SysFont("segoeui", 14)
        self.font_bold  = pygame.font.SysFont("segoeui", 14, bold=True)
        self.font_small = pygame.font.SysFont("segoeui", 12)

        # World / entity config (defaults are the VIEWER_* constants above,
        # NOT config.py's training-sized WORLD_SIZE/NUM_FOOD_LOW/etc.)
        self.world_size    = world_size
        self.num_food_low  = num_food_low
        self.num_food_high = num_food_high
        self.num_hazards   = num_hazards

        if trained_weights_path is None:
            # Same file run_experiment.py loads/saves.
            trained_weights_path = RESULTS_DIR / "best_model.pt"
        self.weights_path = Path(trained_weights_path)

        # Simulation controls
        self.paused = False
        self.target_fps = VIEWER_TARGET_FPS
        self.tick_count = 0
        self.session_births = 0
        self.session_deaths = 0
        # self.best_metric_so_far is set inside _load_founder_net() (called
        # by _reset_world() right below) from training_state.json, NOT
        # hardcoded to -inf here — see that method's docstring for why that
        # distinction matters.

        # Grid layout cache, filled in by _draw_world_grid() each frame and
        # read back by the click-to-select handler in run() — kept in one
        # place so the two always agree on where cell (0,0) is on screen.
        self._grid_offset_x = 30
        self._grid_offset_y = 30
        self._grid_pixel_size = 750
        self._cell_size = max(1, self._grid_pixel_size // self.world_size)

        self._reset_world()

    def _load_founder_net(self):
        """Build a fresh net and load results/best_model.pt into it if that
        file exists; returns (net, epsilon). Also reads training_state.json
        for self.best_metric_so_far (see below) — CRITICAL: without this, a
        fresh session would start best_metric_so_far at -inf and could
        overwrite an already-good best_model.pt with a worse individual the
        moment ANY individual dies, silently destroying prior training
        progress. Shared by __init__ and every manual reset (R) so a reset
        always continues from the latest trained brain, not a disconnected
        fresh-random one.

        NOTE: unlike best_metric_so_far, epsilon is NOT read from
        training_state.json here — see the viewer-only override right below
        for why (config.LIVE_VIEWER_START_EPSILON)."""
        state = load_training_state(RESULTS_DIR / "training_state.json")
        if state is not None:
            self.best_metric_so_far = state["best_metric"]
        else:
            self.best_metric_so_far = float("-inf")
        # Viewer-only override (2026-09): ignore training_state.json's saved
        # epsilon here — that value can be anywhere up to 1.0 depending how
        # early real training still is (not itself a bug, see
        # INLIFE_EPSILON_DECAY's comment in config.py), which made a fresh
        # viewer session mostly show near-fully-random behavior. Only the
        # STARTING value is overridden; best_metric_so_far above is still
        # loaded correctly from the real training state.
        epsilon = LIVE_VIEWER_START_EPSILON
        net, weights_loaded = load_weights_or_fresh(build_lstm_network, self.weights_path,
                                                     verbose=False)
        if weights_loaded:
            print(f"Loaded trained Conv-LSTM brain from {self.weights_path} "
                  f"(best_metric on record: {self.best_metric_so_far:+.2f})")
        else:
            print("No usable saved weights found. Starting with a fresh random brain.")
        return net, epsilon

    def _reset_world(self):
        """(Re)found the ecosystem: fresh (big) World, one founder individual
        loaded from results/best_model.pt (or fresh-random if none exists).
        Called at startup and whenever R is pressed."""
        self.world = World(width=self.world_size, height=self.world_size,
                           num_food_low=self.num_food_low,
                           num_food_high=self.num_food_high,
                           num_hazards=self.num_hazards)
        net, epsilon = self._load_founder_net()
        self.population = [spawn_founder(self.world, epsilon=epsilon, net=net)]
        self.extinct = False
        self.spotlight_id = self.population[0].id

    def _spotlight(self):
        """The individual the right-hand panels currently describe. Sticky
        preference for keeping the SAME individual spotlighted tick to tick
        (by id) so the panels don't visually flicker; only auto-re-picks
        (oldest living one) when the previously-spotlighted one has died.
        A person can override this any time by clicking a creature (see
        run()'s mouse handling) or with the [ / ] cycle keys."""
        if not self.population:
            return None
        current = next((ind for ind in self.population if ind.id == self.spotlight_id), None)
        if current is not None:
            return current
        best = max(self.population, key=lambda ind: ind.agent.age)
        self.spotlight_id = best.id
        return best

    def _cycle_spotlight(self, step):
        """Move the spotlight to the next/previous living individual (by id
        order), wrapping around. step is +1 or -1."""
        if not self.population:
            return
        ids = sorted(ind.id for ind in self.population)
        if self.spotlight_id not in ids:
            self.spotlight_id = ids[0]
            return
        idx = ids.index(self.spotlight_id)
        self.spotlight_id = ids[(idx + step) % len(ids)]

    def _select_at_screen_pos(self, mx, my):
        """Click-to-follow: if the click landed inside the world grid, find
        the closest living creature to that cell and spotlight it. A simple,
        no-explanation-needed way for someone with zero context to pick who
        to watch, instead of the sticky-oldest default being their only
        option."""
        gx = mx - self._grid_offset_x
        gy = my - self._grid_offset_y
        if not (0 <= gx < self._grid_pixel_size and 0 <= gy < self._grid_pixel_size):
            return
        if not self.population:
            return
        cell_x = gx // self._cell_size
        cell_y = self.world_size - 1 - (gy // self._cell_size)
        best_ind, best_dist = None, None
        for ind in self.population:
            ax, ay = ind.agent.position
            dist = abs(ax - cell_x) + abs(ay - cell_y)
            if best_dist is None or dist < best_dist:
                best_ind, best_dist = ind, dist
        if best_ind is not None:
            self.spotlight_id = best_ind.id

    def step_simulation(self):
        """One ecosystem tick (src/rl/population.py's ecosystem_step) for the
        WHOLE population — not a single agent's tick. auto_reseed=False: a
        real extinction here means the simulation actually stops (self.extinct
        = True) instead of silently reseeding, since the viewer's whole point
        is to let you watch a specific ongoing population live or die out."""
        if self.extinct:
            return

        births, deaths = ecosystem_step(self.world, self.population, auto_reseed=False,
                                        child_epsilon=LIVE_VIEWER_START_EPSILON,
                                        tick_count=self.tick_count + 1)

        self.session_births += len(births)
        self.session_deaths += len(deaths)
        self.tick_count += 1

        # Checkpoint: a death's final score, checked every tick (cheap, rare
        # event) + the living population, checked only every
        # ECOSYSTEM_SAVE_EVERY_TICKS ticks (crash-safety net without a disk
        # write almost every tick — see check_and_save_best's docstring for
        # why that matters).
        self.best_metric_so_far = check_and_save_best(
            deaths, self.best_metric_so_far, self.weights_path,
            RESULTS_DIR / "training_state.json")
        if self.tick_count % ECOSYSTEM_SAVE_EVERY_TICKS == 0:
            self.best_metric_so_far = check_and_save_best(
                self.population, self.best_metric_so_far, self.weights_path,
                RESULTS_DIR / "training_state.json")

        if not self.population:
            self.extinct = True
            print("Population extinct. Press R to found a new one from best_model.pt.")

    def draw(self):
        self.screen.fill(COLOR_BG)
        self._draw_world_grid(offset_x=30, offset_y=30, grid_pixel_size=750)
        self._draw_right_panel(start_x=810, start_y=30)
        self._draw_control_bar(start_x=30, start_y=860)
        pygame.display.flip()

    def _draw_world_grid(self, offset_x, offset_y, grid_pixel_size):
        cell_size = max(1, grid_pixel_size // self.world_size)
        # Cache the layout so click handling (in run()) matches exactly.
        self._grid_offset_x, self._grid_offset_y = offset_x, offset_y
        self._grid_pixel_size, self._cell_size = grid_pixel_size, cell_size

        grid_rect = pygame.Rect(offset_x, offset_y, grid_pixel_size, grid_pixel_size)
        pygame.draw.rect(self.screen, COLOR_GRID_BG, grid_rect, border_radius=8)

        # Terrain background — drawn before grid lines/entities/agents.
        for tx in range(self.world_size):
            for ty in range(self.world_size):
                terrain_code = int(self.world.terrain[tx, ty])
                color = CELL_COLORS.get(terrain_code, COLOR_GRID_BG)
                rect = pygame.Rect(offset_x + tx * cell_size,
                                   offset_y + (self.world_size - 1 - ty) * cell_size,
                                   cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect)

        # Grid lines are skipped past a density where they'd just be visual
        # noise (a 50-wide grid at ~15px/cell would be 51 near-solid lines) —
        # only draw them when each cell is large enough to make lines
        # meaningful rather than clutter.
        if cell_size >= 8:
            for i in range(self.world_size + 1):
                pygame.draw.line(self.screen, COLOR_GRID_LINE,
                                 (offset_x + i * cell_size, offset_y),
                                 (offset_x + i * cell_size, offset_y + grid_pixel_size), 1)
                pygame.draw.line(self.screen, COLOR_GRID_LINE,
                                 (offset_x, offset_y + i * cell_size),
                                 (offset_x + grid_pixel_size, offset_y + i * cell_size), 1)

        # Seeds (growing food, not yet visible to any agent — human-only marker)
        for (sx, sy) in self.world.seeds:
            scr_x = offset_x + sx * cell_size + cell_size // 2
            scr_y = offset_y + (self.world_size - 1 - sy) * cell_size + cell_size // 2
            pygame.draw.circle(self.screen, COLOR_SEED, (scr_x, scr_y), max(2, cell_size // 6))

        # Entities
        for entity in self.world.entities:
            ex, ey = entity["pos"]
            scr_x = offset_x + ex * cell_size + cell_size // 2
            scr_y = offset_y + (self.world_size - 1 - ey) * cell_size + cell_size // 2
            if entity["type"] == "food_low":
                pygame.draw.circle(self.screen, COLOR_FOOD_LOW, (scr_x, scr_y), max(3, cell_size // 4))
            elif entity["type"] == "food_high":
                r = max(4, cell_size // 3)
                pygame.draw.circle(self.screen, COLOR_FOOD_HIGH, (scr_x, scr_y), r)
                pygame.draw.circle(self.screen, (255, 255, 255), (scr_x - r//3, scr_y - r//3), max(1, r // 3))
            elif entity["type"] == "hazard":
                r = max(4, cell_size // 3)
                pygame.draw.line(self.screen, COLOR_HAZARD, (scr_x - r, scr_y - r), (scr_x + r, scr_y + r), 3)
                pygame.draw.line(self.screen, COLOR_HAZARD, (scr_x - r, scr_y + r), (scr_x + r, scr_y - r), 3)
            elif entity["type"] == "food_starter":
                r = max(4, cell_size // 3)
                pts = [(scr_x, scr_y - r), (scr_x + r, scr_y), (scr_x, scr_y + r), (scr_x - r, scr_y)]
                pygame.draw.polygon(self.screen, COLOR_FOOD_STARTER, pts)
            elif entity["type"] == "rotten_food":
                r = max(3, cell_size // 4)
                pygame.draw.circle(self.screen, COLOR_FOOD_ROTTEN, (scr_x, scr_y), r)
                pygame.draw.line(self.screen, (60, 30, 10), (scr_x-r, scr_y-r), (scr_x+r, scr_y+r), 1)
                pygame.draw.line(self.screen, (60, 30, 10), (scr_x-r, scr_y+r), (scr_x+r, scr_y-r), 1)

        # Every living agent (triangle pointing in its own facing direction).
        # Body color reflects wellbeing (green/amber/red — see
        # _creature_color) so the whole world reads at a glance without any
        # legend. The spotlighted individual gets a soft glow ring behind it
        # so it's easy to keep track of, especially in a big/busy world.
        for ind in self.population:
            ax_grid, ay_grid = ind.agent.position
            cx = offset_x + ax_grid * cell_size + cell_size // 2
            cy = offset_y + (self.world_size - 1 - ay_grid) * cell_size + cell_size // 2

            if ind.id == self.spotlight_id:
                glow_r = max(6, int(cell_size * 1.4))
                pygame.draw.circle(self.screen, COLOR_SPOTLIGHT_GLOW, (cx, cy), glow_r, 2)

            wellbeing = min(ind.agent.energy / ind.agent.MAX_ENERGY,
                            ind.agent.health / ind.agent.MAX_HEALTH)
            body_color = _creature_color(wellbeing)
            angle = FACING_ANGLES[ind.agent.facing]
            r = max(3, cell_size // 2.4)
            p1 = (cx + r * np.cos(angle), cy + r * np.sin(angle))
            p2 = (cx + r * np.cos(angle + 2.5), cy + r * np.sin(angle + 2.5))
            p3 = (cx + r * np.cos(angle - 2.5), cy + r * np.sin(angle - 2.5))
            pygame.draw.polygon(self.screen, body_color, [p1, p2, p3])
            outline_w = 1 if cell_size < 12 else 2
            pygame.draw.polygon(self.screen, (255, 255, 255), [p1, p2, p3], outline_w)

        pygame.draw.rect(self.screen, COLOR_TEXT_DIM, grid_rect, 2, border_radius=8)

        if self.extinct:
            msg = self.font_title.render("TH\u1ebe GI\u1edaI \u0110\u00c3 TUY\u1ec6T DI\u1ec6T \u2014 nh\u1ea5n R \u0111\u1ec3 b\u1eaft \u0111\u1ea7u l\u1ea1i", True, COLOR_EXTINCT)
            self.screen.blit(msg, (offset_x + 30, offset_y + grid_pixel_size // 2 - 10))

        # Legend
        legend_y = offset_y + grid_pixel_size + 8
        items = [("Th\u1ee9c \u0103n nh\u1ecf", COLOR_FOOD_LOW), ("Th\u1ee9c \u0103n to", COLOR_FOOD_HIGH),
                ("Th\u1ee9c \u0103n kh\u1edfi \u0111\u1ea7u", COLOR_FOOD_STARTER), ("\u0110\u00e3 th\u1ed1i", COLOR_FOOD_ROTTEN),
                ("H\u1ea1t gi\u1ed1ng", COLOR_SEED), ("Nguy hi\u1ec3m", COLOR_HAZARD),
                ("T\u01b0\u1eddng", COLOR_WALL), ("N\u01b0\u1edbc", COLOR_WATER)]
        lx = offset_x
        for label, color in items:
            pygame.draw.circle(self.screen, color, (lx + 6, legend_y + 6), 6)
            t = self.font_small.render(label, True, COLOR_TEXT_DIM)
            self.screen.blit(t, (lx + 16, legend_y))
            lx += 16 + t.get_width() + 20

        hint = self.font_small.render(
            "M\u00e0u sinh v\u1eadt: xanh = kh\u1ecfe, v\u00e0ng = t\u1ea1m \u1ed5n, \u0111\u1ecf = y\u1ebfu / nguy hi\u1ec3m",
            True, COLOR_TEXT_DIM)
        self.screen.blit(hint, (offset_x, legend_y + 22))

    def _draw_right_panel(self, start_x, start_y):
        spot = self._spotlight()

        # 1. Local view (spotlight individual)
        card_rect1 = pygame.Rect(start_x, start_y, 410, 165)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect1, border_radius=8)
        title = "T\u1ea6M NH\u00cc\u041e C\u1ee6A SINH V\u1eacT"
        if spot is not None:
            title = f"T\u1ea6M NH\u00ccN C\u1ee6A #{spot.id} (th\u1ebf h\u1ec7 {spot.generation})"
        txt_title1 = self.font_title.render(title, True, COLOR_TEXT)
        self.screen.blit(txt_title1, (start_x + 15, start_y + 12))

        if spot is not None:
            local_grid = self.world.get_local_view(spot.agent.position, spot.agent.facing)
            vh, vw = local_grid.shape
            grid_start_x = start_x + 20
            grid_start_y = start_y + 45
            mini_cell = 25
            for r in range(vh):
                for c in range(vw):
                    val = local_grid[r, c]
                    rect = pygame.Rect(grid_start_x + c * mini_cell, grid_start_y + r * mini_cell,
                                        mini_cell - 2, mini_cell - 2)
                    pygame.draw.rect(self.screen, CELL_COLORS.get(val, (0, 0, 0)), rect, border_radius=3)

            legend_x = start_x + 140
            legend_y = start_y + 45
            lbls = [
                "\u0110\u00e2y l\u00e0 nh\u1eefng g\u00ec sinh v\u1eadt nh\u00ecn th\u1ea5y",
                "H\u00e0ng tr\u00ean: xa nh\u1ea5t ph\u00eda tr\u01b0\u1edbc",
                "H\u00e0ng d\u01b0\u1edbi: ph\u00eda sau l\u01b0ng",
                "M\u00e0u gi\u1ed1ng b\u1ea3n \u0111\u1ed3 ch\u00ednh",
            ]
            for idx, txt in enumerate(lbls):
                t = self.font_small.render(txt, True, COLOR_TEXT_DIM)
                self.screen.blit(t, (legend_x, legend_y + idx * 22))
        else:
            t = self.font_main.render("Kh\u00f4ng c\u00f2n sinh v\u1eadt n\u00e0o \u0111\u1ec3 theo d\u00f5i.", True, COLOR_TEXT_DIM)
            self.screen.blit(t, (start_x + 20, start_y + 60))

        # 2. "What is it thinking" — plain-language decision meter, built
        # from the exact same Q-values a technical viewer would show as
        # numbers, but framed as "what does it want to do" for someone who
        # has no reason to know what a Q-value is.
        card_rect2 = pygame.Rect(start_x, start_y + 180, 410, 205)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect2, border_radius=8)
        txt_title2 = self.font_title.render("SINH V\u1eacT \u0110ANG NGH\u0128 G\u00cc?", True, COLOR_TEXT)
        self.screen.blit(txt_title2, (start_x + 15, start_y + 192))

        if spot is not None:
            q_values, _, _ = spot.brain.forward(spot.x, spot.h, spot.c)
            chosen = int(np.argmax(q_values))
            chart_x = start_x + 25
            chart_y = start_y + 225
            max_q = max(1.0, np.max(np.abs(q_values)))
            for i, (name, val) in enumerate(zip(ACTION_NAMES, q_values)):
                bx = chart_x + i * 75
                by = chart_y + 100
                is_chosen = (i == chosen)
                bar_color = COLOR_HIGHLIGHT if is_chosen else COLOR_HEALTH_GOOD
                h = int((val / (max_q * 1.2)) * 70)
                bar_rect = (pygame.Rect(bx + 15, by - h, 35, max(4, h)) if h >= 0
                           else pygame.Rect(bx + 15, by, 35, min(-4, -h)))
                pygame.draw.rect(self.screen, COLOR_BAR_BG, (bx + 15, by - 70, 35, 70), border_radius=4)
                pygame.draw.rect(self.screen, bar_color, bar_rect, border_radius=4)
                t_name = self.font_small.render(name, True, COLOR_HIGHLIGHT if is_chosen else COLOR_TEXT_DIM)
                self.screen.blit(t_name, (bx + 5, by + 8))
            chosen_txt = self.font_bold.render(
                f"\u2192 \u0110ang ch\u1ecdn: {ACTION_NAMES[chosen]}", True, COLOR_HIGHLIGHT)
            self.screen.blit(chosen_txt, (start_x + 20, start_y + 355))
            explore_pct = spot.brain.policy.epsilon * 100
            explore_txt = self.font_small.render(
                f"(\u0111\u1ed9 ng\u1eabu h\u1ee9ng c\u00f2n l\u1ea1i: {explore_pct:.0f}%)", True, COLOR_TEXT_DIM)
            self.screen.blit(explore_txt, (start_x + 220, start_y + 358))

        # 3. World status + spotlight body telemetry
        card_rect3 = pygame.Rect(start_x, start_y + 395, 410, 175)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect3, border_radius=8)
        txt_title3 = self.font_title.render("T\u00ccNH TR\u1ea0NG TH\u1ebe GI\u1edaI", True, COLOR_TEXT)
        self.screen.blit(txt_title3, (start_x + 15, start_y + 402))

        if spot is not None:
            bar_w = 175
            bar_h = 14
            e_bar_x = start_x + 20
            e_bar_y = start_y + 448
            pygame.draw.rect(self.screen, COLOR_BAR_BG, (e_bar_x, e_bar_y, bar_w, bar_h), border_radius=8)
            energy_pct = max(0.0, min(1.0, spot.agent.energy / spot.agent.MAX_ENERGY))
            e_color = COLOR_ENERGY_GOOD if energy_pct > 0.5 else (COLOR_ENERGY_WARN if energy_pct > 0.2 else COLOR_ENERGY_CRIT)
            if energy_pct > 0:
                pygame.draw.rect(self.screen, e_color, (e_bar_x, e_bar_y, int(bar_w * energy_pct), bar_h), border_radius=8)
            txt_e = self.font_small.render(f"N\u0103ng l\u01b0\u1ee3ng: {spot.agent.energy:.0f}/{spot.agent.MAX_ENERGY:.0f}", True, COLOR_TEXT)
            self.screen.blit(txt_e, (e_bar_x, start_y + 430))

            h_bar_x = start_x + 215
            h_bar_y = start_y + 448
            pygame.draw.rect(self.screen, COLOR_BAR_BG, (h_bar_x, h_bar_y, bar_w, bar_h), border_radius=8)
            health_pct = max(0.0, min(1.0, spot.agent.health / spot.agent.MAX_HEALTH))
            h_color = COLOR_HEALTH_GOOD if health_pct > 0.5 else (COLOR_HEALTH_WARN if health_pct > 0.2 else COLOR_HEALTH_CRIT)
            if health_pct > 0:
                pygame.draw.rect(self.screen, h_color, (h_bar_x, h_bar_y, int(bar_w * health_pct), bar_h), border_radius=8)
            txt_h = self.font_small.render(f"M\u00e1u: {spot.agent.health:.0f}/{spot.agent.MAX_HEALTH:.0f}", True, COLOR_TEXT)
            self.screen.blit(txt_h, (h_bar_x, start_y + 430))

            mood_txt, mood_color = _mood_label(energy_pct, health_pct)
            t_mood = self.font_bold.render(mood_txt, True, mood_color)
            self.screen.blit(t_mood, (start_x + 20, start_y + 470))

        m_x = start_x + 20
        m_y = start_y + 495
        info_items = [
            (f"S\u1ed1 sinh v\u1eadt \u0111ang s\u1ed1ng: {len(self.population)}", COLOR_TEXT),
            (f"Th\u1eddi gian: {self.tick_count} tick", COLOR_TEXT),
            (f"Sinh ra (phi\u00ean n\u00e0y): {self.session_births}", COLOR_ENERGY_GOOD),
            (f"Qua \u0111\u1eddi (phi\u00ean n\u00e0y): {self.session_deaths}", COLOR_HAZARD),
            (f"Th\u1ebf h\u1ec7 cao nh\u1ea5t: {max((i.generation for i in self.population), default='-')}", COLOR_TEXT),
            (f"K\u1ef7 l\u1ee5c t\u1ed1t nh\u1ea5t: {self.best_metric_so_far:+.1f}"
             if self.best_metric_so_far > float("-inf") else "K\u1ef7 l\u1ee5c t\u1ed1t nh\u1ea5t: ch\u01b0a c\u00f3", COLOR_TEXT_DIM),
        ]
        for idx, (label_txt, col) in enumerate(info_items):
            col_idx = idx % 2
            row_idx = idx // 2
            t = self.font_main.render(label_txt, True, col)
            self.screen.blit(t, (m_x + col_idx * 195, m_y + row_idx * 18))

    def _draw_control_bar(self, start_x, start_y):
        bar_rect = pygame.Rect(start_x, start_y, self.width - 2 * start_x, 30)
        pygame.draw.rect(self.screen, COLOR_CARD, bar_rect, border_radius=6)

        status = "TUY\u1ec6T DI\u1ec6T" if self.extinct else ("T\u1ea0M D\u1eeaNG" if self.paused else "\u0110ANG CH\u1ea0Y")
        status_color = COLOR_EXTINCT if self.extinct else (COLOR_HIGHLIGHT if self.paused else COLOR_ENERGY_GOOD)
        status_str = f"{status}  |  T\u1ed1c \u0111\u1ed9: {self.target_fps}  |  Th\u1ebf gi\u1edbi: {self.world_size}x{self.world_size}"
        t_status = self.font_bold.render(status_str, True, status_color)
        self.screen.blit(t_status, (start_x + 15, start_y + 5))

        help_str = ("[Click] Ch\u1ecdn sinh v\u1eadt | [Space] T\u1ea1m d\u1eebng | [\u2190/\u2192] \u0110\u1ed5i sinh v\u1eadt "
                   "| [\u2191/\u2193] T\u1ed1c \u0111\u1ed9 | [R] B\u1eaft \u0111\u1ea7u l\u1ea1i | [S] L\u01b0u")
        t_help = self.font_small.render(help_str, True, COLOR_TEXT_DIM)
        self.screen.blit(t_help, (start_x + 480, start_y + 7))

    def save_best(self):
        """Manually force a check-and-save of the CURRENT living population's
        best individual (not gated on beating the previous record's value at
        the exact instant you press S — it still won't overwrite with
        something worse, check_and_save_best already only saves on genuine
        improvement)."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.best_metric_so_far = check_and_save_best(
            self.population, self.best_metric_so_far, self.weights_path,
            RESULTS_DIR / "training_state.json")
        print(f"[save] Checked population for a new best (current record: "
              f"{self.best_metric_so_far:+.2f}).")

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._select_at_screen_pos(*event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_RETURN and self.paused:
                        self.step_simulation()
                    elif event.key == pygame.K_UP:
                        self.target_fps = min(60, self.target_fps + 1)
                    elif event.key == pygame.K_DOWN:
                        self.target_fps = max(1, self.target_fps - 1)
                    elif event.key == pygame.K_LEFT:
                        self._cycle_spotlight(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._cycle_spotlight(1)
                    elif event.key == pygame.K_r:
                        self._reset_world()
                    elif event.key == pygame.K_s:
                        self.save_best()

            if not self.paused:
                self.step_simulation()

            self.draw()
            self.clock.tick(self.target_fps)

        print("Saving best individual before exit...")
        self.save_best()
        pygame.quit()


if __name__ == "__main__":
    viewer = WorldViewer2D()
    viewer.run()
