import sys
import os
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
from model_io import load_lstm_weights
from population import spawn_founder, ecosystem_step, check_and_save_best
from training_state import load_training_state
from config import (WORLD_SIZE, NUM_FOOD_LOW, NUM_FOOD_HIGH, NUM_HAZARDS,
                    SAVE_EVERY)

# Colors
COLOR_BG        = (30, 34, 42)
COLOR_CARD      = (40, 44, 52)
COLOR_GRID_BG   = (240, 243, 246)
COLOR_GRID_LINE = (210, 215, 222)
COLOR_AGENT     = (41, 128, 185)
COLOR_AGENT_SPOTLIGHT = (241, 196, 15)  # gold outline for the spotlighted individual
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
COLOR_TEXT      = (220, 225, 230)
COLOR_TEXT_DIM  = (140, 145, 155)
COLOR_ENERGY_GOOD = (46, 204, 113)
COLOR_ENERGY_WARN = (241, 196, 15)
COLOR_ENERGY_CRIT = (231, 76, 60)
COLOR_HEALTH_GOOD = (52, 152, 219)
COLOR_HEALTH_WARN = (230, 126, 34)
COLOR_HEALTH_CRIT = (192, 57, 43)
COLOR_BAR_BG    = (55, 60, 72)
COLOR_HIGHLIGHT = (155, 89, 182)
COLOR_EXTINCT   = (231, 76, 60)

# Local-view cell code -> color (see World docstring for the code table).
# NOTE: food_starter is NOT a separate code (it shares food_low's code 5, so
# the mini "AGENT LOCAL VISION" panel correctly shows it identically to
# food_low — that's what the agent actually perceives). Its distinct
# COLOR_FOOD_STARTER marker below is only used in the main world-map drawing,
# which draws by entity TYPE, not by this code table — a human-only visual
# aid, not something the agent's observation encodes.
CELL_COLORS = {
    World.CELL_UNKNOWN: (30, 32, 38),     # behind / unknown (dark)
    World.CELL_WALL:    COLOR_WALL,
    World.CELL_WATER:   COLOR_WATER,
    World.CELL_SOIL:    COLOR_SOIL,
    World.CELL_GRASS:   COLOR_GRASS,
    5: COLOR_FOOD_LOW,                    # food_low (+ food_starter, same code)
    6: COLOR_FOOD_HIGH,                   # food_high
    7: COLOR_HAZARD,                      # hazard
    8: COLOR_FOOD_ROTTEN,                 # rotten_food
}

ACTION_NAMES = ["Stay", "Fwd", "Bwd", "TurnL", "TurnR"]
FACING_NAMES = ["UP \u2191", "RIGHT \u2192", "DOWN \u2193", "LEFT \u2190"]
FACING_ANGLES = {0: -np.pi/2, 1: 0, 2: np.pi/2, 3: np.pi}


class WorldViewer2D:
    """
    An interactive Pygame viewer for the ECOSYSTEM (src/rl/population.py) —
    NOT a single agent's single lifetime. It runs indefinitely: individuals
    act, learn, and reproduce every tick, for as long as ANY of them are
    alive. It only stops on a real "population extinct" (all individuals
    dead, nobody left to reseed from — reset manually with R) or when you
    press R yourself to start a fresh population. This is a deliberate
    change from the old single-agent viewer (see progress.md): the world
    itself, not one agent's lifetime, is the thing being watched now.

    Always founds a (re)started population from results/best_model.pt if it
    exists (same file run_experiment.py trains/checkpoints — see that file's
    module docstring), so this viewer is always watching a continuation of
    whatever training has produced so far, not a separate/disconnected demo.
    """
    def __init__(self, world_size=WORLD_SIZE, num_food_low=NUM_FOOD_LOW,
                 num_food_high=NUM_FOOD_HIGH, num_hazards=NUM_HAZARDS,
                 trained_weights_path=None):
        pygame.init()
        pygame.font.init()

        self.width = 1040
        self.height = 620
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Virtual Lifetime AI \u2014 Ecosystem Visualizer (LSTM)")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_main  = pygame.font.SysFont("segoeui", 14)
        self.font_bold  = pygame.font.SysFont("segoeui", 14, bold=True)
        self.font_small = pygame.font.SysFont("segoeui", 12)

        # World / entity config (defaults come from src/config.py)
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
        self.target_fps = 6
        self.tick_count = 0
        self.session_births = 0
        self.session_deaths = 0
        # self.best_metric_so_far is set inside _load_founder_net() (called
        # by _reset_world() right below) from training_state.json, NOT
        # hardcoded to -inf here — see that method's docstring for why that
        # distinction matters.

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
        fresh-random one."""
        net = build_lstm_network()
        epsilon = 0.3   # matches run_experiment.py's fallback for resuming
                        # without a training_state.json
        state = load_training_state(RESULTS_DIR / "training_state.json")
        if state is not None:
            self.best_metric_so_far = state["best_metric"]
            epsilon = state["epsilon"]
        else:
            self.best_metric_so_far = float("-inf")
        if os.path.exists(self.weights_path):
            try:
                load_lstm_weights(net, self.weights_path)
                print(f"Loaded trained Conv-LSTM brain from {self.weights_path} "
                      f"(best_metric on record: {self.best_metric_so_far:+.2f})")
            except Exception as e:
                print(f"Could not load weights: {e}. Starting from a fresh random brain.")
                net = build_lstm_network()
        else:
            print("No saved weights found. Starting with a fresh random brain.")
        return net, epsilon

    def _reset_world(self):
        """(Re)found the ecosystem: fresh World, one founder individual
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
        """The individual the right-hand panels currently describe — the
        oldest living one (interesting to watch: likely the most evolved/
        successful lineage currently active), with a sticky preference for
        keeping the SAME individual spotlighted tick to tick (by id) so the
        panels don't visually flicker between individuals of similar age;
        only re-picks when the previously-spotlighted one is no longer
        alive."""
        if not self.population:
            return None
        current = next((ind for ind in self.population if ind.id == self.spotlight_id), None)
        if current is not None:
            return current
        best = max(self.population, key=lambda ind: ind.agent.age)
        self.spotlight_id = best.id
        return best

    def step_simulation(self):
        """One ecosystem tick (src/rl/population.py's ecosystem_step) for the
        WHOLE population — not a single agent's tick. auto_reseed=False: a
        real extinction here means the simulation actually stops (self.extinct
        = True) instead of silently reseeding, since the viewer's whole point
        is to let you watch a specific ongoing population live or die out."""
        if self.extinct:
            return

        births, deaths = ecosystem_step(self.world, self.population, auto_reseed=False)
        self.session_births += len(births)
        self.session_deaths += len(deaths)
        self.tick_count += 1

        # Checkpoint: a death's final score, checked every tick (cheap, rare
        # event) + the living population, checked only every SAVE_EVERY
        # ticks (crash-safety net without a disk write almost every tick —
        # see check_and_save_best's docstring for why that matters).
        self.best_metric_so_far = check_and_save_best(
            deaths, self.best_metric_so_far, self.weights_path,
            RESULTS_DIR / "training_state.json")
        if self.tick_count % SAVE_EVERY == 0:
            self.best_metric_so_far = check_and_save_best(
                self.population, self.best_metric_so_far, self.weights_path,
                RESULTS_DIR / "training_state.json")

        if not self.population:
            self.extinct = True
            print("Population extinct. Press R to found a new one from best_model.pt.")

    def draw(self):
        self.screen.fill(COLOR_BG)
        self._draw_world_grid(offset_x=30, offset_y=30, grid_pixel_size=500)
        self._draw_right_panel(start_x=600, start_y=30)
        self._draw_control_bar(start_x=30, start_y=580)
        pygame.display.flip()

    def _draw_world_grid(self, offset_x, offset_y, grid_pixel_size):
        cell_size = grid_pixel_size // self.world_size

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

        for i in range(self.world_size + 1):
            pygame.draw.line(self.screen, COLOR_GRID_LINE,
                             (offset_x + i * cell_size, offset_y),
                             (offset_x + i * cell_size, offset_y + grid_pixel_size), 2)
            pygame.draw.line(self.screen, COLOR_GRID_LINE,
                             (offset_x, offset_y + i * cell_size),
                             (offset_x + grid_pixel_size, offset_y + i * cell_size), 2)

        # Seeds (growing food, not yet visible to any agent — human-only marker)
        for (sx, sy) in self.world.seeds:
            scr_x = offset_x + sx * cell_size + cell_size // 2
            scr_y = offset_y + (self.world_size - 1 - sy) * cell_size + cell_size // 2
            pygame.draw.circle(self.screen, COLOR_SEED, (scr_x, scr_y), max(3, cell_size // 6))

        # Entities
        for entity in self.world.entities:
            ex, ey = entity["pos"]
            scr_x = offset_x + ex * cell_size + cell_size // 2
            scr_y = offset_y + (self.world_size - 1 - ey) * cell_size + cell_size // 2
            if entity["type"] == "food_low":
                pygame.draw.circle(self.screen, COLOR_FOOD_LOW, (scr_x, scr_y), max(4, cell_size // 4))
            elif entity["type"] == "food_high":
                r = max(5, cell_size // 3)
                pygame.draw.circle(self.screen, COLOR_FOOD_HIGH, (scr_x, scr_y), r)
                pygame.draw.circle(self.screen, (255, 255, 255), (scr_x - r//3, scr_y - r//3), max(2, r // 3))
            elif entity["type"] == "hazard":
                r = max(6, cell_size // 3)
                pygame.draw.line(self.screen, COLOR_HAZARD, (scr_x - r, scr_y - r), (scr_x + r, scr_y + r), 4)
                pygame.draw.line(self.screen, COLOR_HAZARD, (scr_x - r, scr_y + r), (scr_x + r, scr_y - r), 4)
            elif entity["type"] == "food_starter":
                r = max(5, cell_size // 3)
                pts = [(scr_x, scr_y - r), (scr_x + r, scr_y), (scr_x, scr_y + r), (scr_x - r, scr_y)]
                pygame.draw.polygon(self.screen, COLOR_FOOD_STARTER, pts)
            elif entity["type"] == "rotten_food":
                r = max(4, cell_size // 4)
                pygame.draw.circle(self.screen, COLOR_FOOD_ROTTEN, (scr_x, scr_y), r)
                # small X overlay to signal "bad food"
                pygame.draw.line(self.screen, (60, 30, 10), (scr_x-r, scr_y-r), (scr_x+r, scr_y+r), 2)
                pygame.draw.line(self.screen, (60, 30, 10), (scr_x-r, scr_y+r), (scr_x+r, scr_y-r), 2)

        # Every living agent (triangle pointing in its own facing direction).
        # The spotlighted individual gets a gold outline so it's easy to spot
        # which triangle the right-hand panels are describing.
        for ind in self.population:
            ax_grid, ay_grid = ind.agent.position
            cx = offset_x + ax_grid * cell_size + cell_size // 2
            cy = offset_y + (self.world_size - 1 - ay_grid) * cell_size + cell_size // 2
            angle = FACING_ANGLES[ind.agent.facing]
            r = cell_size // 2.8
            p1 = (cx + r * np.cos(angle), cy + r * np.sin(angle))
            p2 = (cx + r * np.cos(angle + 2.5), cy + r * np.sin(angle + 2.5))
            p3 = (cx + r * np.cos(angle - 2.5), cy + r * np.sin(angle - 2.5))
            pygame.draw.polygon(self.screen, COLOR_AGENT, [p1, p2, p3])
            outline = COLOR_AGENT_SPOTLIGHT if ind.id == self.spotlight_id else (255, 255, 255)
            pygame.draw.polygon(self.screen, outline, [p1, p2, p3], 2)

        pygame.draw.rect(self.screen, COLOR_TEXT_DIM, grid_rect, 2, border_radius=8)

        if self.extinct:
            msg = self.font_title.render("POPULATION EXTINCT \u2014 press R to reset", True, COLOR_EXTINCT)
            self.screen.blit(msg, (offset_x + 40, offset_y + grid_pixel_size // 2 - 10))

        # Legend
        legend_y = offset_y + grid_pixel_size + 8
        items = [("Food (low)", COLOR_FOOD_LOW), ("Food (high)", COLOR_FOOD_HIGH),
                ("Food (starter)", COLOR_FOOD_STARTER), ("Rotten", COLOR_FOOD_ROTTEN),
                ("Seed", COLOR_SEED), ("Hazard", COLOR_HAZARD),
                ("Wall", COLOR_WALL), ("Water", COLOR_WATER)]
        lx = offset_x
        for label, color in items:
            pygame.draw.circle(self.screen, color, (lx + 6, legend_y + 6), 6)
            t = self.font_small.render(label, True, COLOR_TEXT_DIM)
            self.screen.blit(t, (lx + 16, legend_y))
            lx += 16 + t.get_width() + 20

    def _draw_right_panel(self, start_x, start_y):
        spot = self._spotlight()

        # 1. Local View (spotlight individual)
        card_rect1 = pygame.Rect(start_x, start_y, 410, 165)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect1, border_radius=8)
        title = "AGENT LOCAL VISION"
        if spot is not None:
            title += f"  (spotlight: #{spot.id}, gen {spot.generation})"
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
                (f"Grid {vh}x{vw} (ahead rows + behind)", COLOR_TEXT_DIM),
                ("Row 0: Farthest ahead", COLOR_TEXT_DIM),
                (f"Rows {self.world.vision_range}+ : behind agent", COLOR_TEXT_DIM),
                ("Colors match world map legend", COLOR_TEXT_DIM),
            ]
            for idx, (txt, color) in enumerate(lbls):
                t = self.font_small.render(txt, True, color)
                self.screen.blit(t, (legend_x, legend_y + idx * 22))
        else:
            t = self.font_main.render("No living individual to spotlight.", True, COLOR_TEXT_DIM)
            self.screen.blit(t, (start_x + 20, start_y + 60))

        # 2. Q-value telemetry (spotlight individual, forward pass w/o acting)
        card_rect2 = pygame.Rect(start_x, start_y + 180, 410, 205)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect2, border_radius=8)
        txt_title2 = self.font_title.render("Q-VALUE TELEMETRY Q(s, a)", True, COLOR_TEXT)
        self.screen.blit(txt_title2, (start_x + 15, start_y + 192))

        if spot is not None:
            q_values, _, _ = spot.brain.forward(spot.x, spot.h, spot.c)
            chosen = int(np.argmax(q_values)) if spot.brain.policy.epsilon < 1e-9 else None
            chart_x = start_x + 25
            chart_y = start_y + 230
            max_q = max(1.0, np.max(np.abs(q_values)))
            for i, (name, val) in enumerate(zip(ACTION_NAMES, q_values)):
                bx = chart_x + i * 75
                by = chart_y + 110
                is_chosen = (i == chosen)
                bar_color = COLOR_HIGHLIGHT if is_chosen else COLOR_AGENT
                h = int((val / (max_q * 1.2)) * 80)
                bar_rect = (pygame.Rect(bx + 15, by - h, 35, max(4, h)) if h >= 0
                           else pygame.Rect(bx + 15, by, 35, min(-4, -h)))
                pygame.draw.rect(self.screen, COLOR_BAR_BG, (bx + 15, by - 80, 35, 80), border_radius=4)
                pygame.draw.rect(self.screen, bar_color, bar_rect, border_radius=4)
                t_name = self.font_bold.render(name, True, COLOR_HIGHLIGHT if is_chosen else COLOR_TEXT)
                self.screen.blit(t_name, (bx + 15, by + 8))
                t_val = self.font_small.render(f"{val:+.2f}", True, COLOR_TEXT_DIM)
                self.screen.blit(t_val, (bx + 12, by + 26))

        # 3. Population + spotlight body telemetry
        card_rect3 = pygame.Rect(start_x, start_y + 395, 410, 175)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect3, border_radius=8)

        if spot is not None:
            bar_w = 175
            bar_h = 16
            e_bar_x = start_x + 20
            e_bar_y = start_y + 425
            pygame.draw.rect(self.screen, COLOR_BAR_BG, (e_bar_x, e_bar_y, bar_w, bar_h), border_radius=8)
            energy_pct = max(0.0, min(1.0, spot.agent.energy / spot.agent.MAX_ENERGY))
            e_color = COLOR_ENERGY_GOOD if energy_pct > 0.5 else (COLOR_ENERGY_WARN if energy_pct > 0.2 else COLOR_ENERGY_CRIT)
            if energy_pct > 0:
                pygame.draw.rect(self.screen, e_color, (e_bar_x, e_bar_y, int(bar_w * energy_pct), bar_h), border_radius=8)
            txt_e = self.font_bold.render(f"ENERGY: {spot.agent.energy:.0f}/{spot.agent.MAX_ENERGY:.0f}", True, COLOR_TEXT)
            self.screen.blit(txt_e, (e_bar_x, start_y + 404))

            h_bar_x = start_x + 215
            h_bar_y = start_y + 425
            pygame.draw.rect(self.screen, COLOR_BAR_BG, (h_bar_x, h_bar_y, bar_w, bar_h), border_radius=8)
            health_pct = max(0.0, min(1.0, spot.agent.health / spot.agent.MAX_HEALTH))
            h_color = COLOR_HEALTH_GOOD if health_pct > 0.5 else (COLOR_HEALTH_WARN if health_pct > 0.2 else COLOR_HEALTH_CRIT)
            if health_pct > 0:
                pygame.draw.rect(self.screen, h_color, (h_bar_x, h_bar_y, int(bar_w * health_pct), bar_h), border_radius=8)
            txt_h = self.font_bold.render(f"HEALTH: {spot.agent.health:.0f}/{spot.agent.MAX_HEALTH:.0f}", True, COLOR_TEXT)
            self.screen.blit(txt_h, (h_bar_x, start_y + 404))

        m_x = start_x + 20
        m_y = start_y + 446
        info_items = [
            (f"Population: {len(self.population)}", COLOR_TEXT),
            (f"Ticks this run: {self.tick_count}", COLOR_TEXT),
            (f"Births (session): {self.session_births}", COLOR_ENERGY_GOOD),
            (f"Deaths (session): {self.session_deaths}", COLOR_HAZARD),
            (f"Max generation: {max((i.generation for i in self.population), default='-')}", COLOR_TEXT),
            (f"Best fitness saved: {self.best_metric_so_far:+.2f}"
             if self.best_metric_so_far > float("-inf") else "Best fitness saved: -", COLOR_TEXT_DIM),
        ]
        if spot is not None:
            info_items += [
                (f"Spotlight age: {spot.agent.age}/{spot.agent.max_age}", COLOR_TEXT),
                (f"Spotlight facing: {FACING_NAMES[spot.agent.facing]}", COLOR_TEXT),
                (f"Spotlight epsilon: {spot.brain.policy.epsilon:.3f}", COLOR_TEXT_DIM),
                (f"Spotlight fitness so far: {spot.total_reward:+.2f}", COLOR_TEXT_DIM),
            ]
        for idx, (label_txt, col) in enumerate(info_items):
            col_idx = idx % 2
            row_idx = idx // 2
            t = self.font_main.render(label_txt, True, col)
            self.screen.blit(t, (m_x + col_idx * 195, m_y + row_idx * 18))

    def _draw_control_bar(self, start_x, start_y):
        bar_rect = pygame.Rect(start_x, start_y, 980, 30)
        pygame.draw.rect(self.screen, COLOR_CARD, bar_rect, border_radius=6)

        status = "EXTINCT" if self.extinct else ("PAUSED" if self.paused else "RUNNING")
        status_color = COLOR_EXTINCT if self.extinct else (COLOR_HIGHLIGHT if self.paused else COLOR_ENERGY_GOOD)
        status_str = f"STATUS: {status} | SPEED: {self.target_fps} FPS"
        t_status = self.font_bold.render(status_str, True, status_color)
        self.screen.blit(t_status, (start_x + 15, start_y + 5))

        help_str = "[SPACE] Pause | [->] Step | [UP/DN] Speed | [R] Reset population | [S] Save best"
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
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_RIGHT and self.paused:
                        self.step_simulation()
                    elif event.key == pygame.K_UP:
                        self.target_fps = min(60, self.target_fps + 2)
                    elif event.key == pygame.K_DOWN:
                        self.target_fps = max(1, self.target_fps - 2)
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
