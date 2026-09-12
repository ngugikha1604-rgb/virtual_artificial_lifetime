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
from agent import Agent
from lstm_q_network import build_lstm_network, build_observation, zero_state
from torch_agent import TorchQAgent
from lstm_replay_buffer import LSTMReplayBuffer
from policy import EpsilonGreedyPolicy
from model_io import load_lstm_weights, save_lstm_weights
from run_episode import compute_reward
from world_tick import world_tick
from config import (WORLD_SIZE, NUM_FOOD_LOW, NUM_FOOD_HIGH, NUM_HAZARDS,
                    MAX_AGE, NUM_ACTIONS, REPLAY_CAPACITY, EPSILON_MIN,
                    EPSILON_DECAY, BATCH_SIZE, WINDOW_N, MIN_EPISODES, LEARN_EVERY)

# Colors
COLOR_BG        = (30, 34, 42)
COLOR_CARD      = (40, 44, 52)
COLOR_GRID_BG   = (240, 243, 246)
COLOR_GRID_LINE = (210, 215, 222)
COLOR_AGENT     = (41, 128, 185)
COLOR_FOOD_LOW  = (230, 126, 34)   # orange
COLOR_FOOD_HIGH = (241, 196, 15)   # gold
COLOR_FOOD_STARTER = (26, 188, 156)  # teal — visually distinct one-shot food
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
}

ACTION_NAMES = ["Stay", "Fwd", "Bwd", "TurnL", "TurnR"]
FACING_NAMES = ["UP \u2191", "RIGHT \u2192", "DOWN \u2193", "LEFT \u2190"]


class WorldViewer2D:
    def __init__(self, world_size=WORLD_SIZE, num_food_low=NUM_FOOD_LOW,
                 num_food_high=NUM_FOOD_HIGH, num_hazards=NUM_HAZARDS,
                 max_age=MAX_AGE, trained_weights_path=None):
        pygame.init()
        pygame.font.init()

        self.width = 1040
        self.height = 620
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Virtual Lifetime AI \u2014 2D World Visualizer (LSTM)")
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
        self.max_age       = max_age

        # Conv-LSTM brain (single torch module)
        self.net = build_lstm_network()

        if trained_weights_path is None:
            # Same file run_experiment.py loads/saves — see that file's
            # docstring point 3 for why there's only ONE model file now.
            trained_weights_path = RESULTS_DIR / "best_model.pt"

        if os.path.exists(trained_weights_path):
            try:
                load_lstm_weights(self.net, trained_weights_path)
                print(f"Loaded trained Conv-LSTM brain from {trained_weights_path}")
                print("Note: if these weights were trained before food_low/food_high/"
                      "hazard existed, the agent may look confused at first \u2014 turn "
                      "on Training (T) to let it adapt.")
                # Keep a low exploration floor so a loaded (possibly greedy-
                # oscillating) brain can still break out of loops.
                self.policy = EpsilonGreedyPolicy(epsilon=0.15, epsilon_min=EPSILON_MIN,
                                                 epsilon_decay=EPSILON_DECAY)
            except Exception as e:
                print(f"Could not load weights: {e}. Re-initialising a fresh brain.")
                self.net = build_lstm_network()
                self.policy = EpsilonGreedyPolicy()
        else:
            print("No saved Conv-LSTM weights found. Starting with initial brain.")
            self.policy = EpsilonGreedyPolicy()

        self.brain  = TorchQAgent(self.net, self.policy)
        self.replay = LSTMReplayBuffer(capacity=REPLAY_CAPACITY)

        # Simulation controls
        self.paused = False
        self.target_fps = 6
        self.training_enabled = False
        self.lifetime_count = 1
        self.total_food_eaten = 0
        self.total_hazard_hits = 0
        self.weights_path = Path(trained_weights_path)

        self._reset_episode()

    def _reset_episode(self):
        self.world = World(width=self.world_size, height=self.world_size,
                            num_food_low=self.num_food_low,
                            num_food_high=self.num_food_high,
                            num_hazards=self.num_hazards)
        self.agent = Agent(self.world, max_age=self.max_age)
        self.replay.start_episode()   # Phase B: begin a fresh episode buffer
        self.h, self.c = zero_state()
        grid = self.world.get_local_view(self.agent.position, self.agent.facing)
        self.state = build_observation(grid, self.agent.internal_state)

        self.last_action    = None
        self.last_q_values  = np.zeros(NUM_ACTIONS)
        self.last_reward    = 0.0
        self.last_event     = None

    def step_simulation(self):
        """Tick-based step. Physics per tick comes from world_tick.py (shared
        with run_episode.py's training loop and visualize.py's demo
        recorder) — this method only adds what's specific to the interactive
        viewer: reward display, online-training replay push, and the
        food/hazard counters shown in the UI."""
        if not self.agent.alive:
            if self.training_enabled:
                self.policy.decay()   # once per lifetime, NOT once per tick
            self.lifetime_count += 1
            self._reset_episode()
            return

        prev_pos = self.agent.position
        result = world_tick(self.world, self.agent, self.brain, self.state, self.h, self.c)

        self.last_action   = result.action
        self.last_q_values = result.q_values

        if result.event is not None:
            if result.event["category"] == "food":
                self.total_food_eaten += 1
            elif result.event["category"] == "hazard":
                self.total_hazard_hits += 1

        reward = compute_reward(result.event, prev_pos, result.prev_facing,
                                self.agent.position, self.world, starved=result.starved)
        self.last_reward = reward
        self.last_event  = result.event

        if self.training_enabled:
            # Phase B: store the tick in the CURRENT episode with its pre-tick
            # hidden anchor; learn on truncated windows once episodes exist.
            self.replay.push(self.state, self.h, self.c, result.action, reward,
                             result.done)
            # Learn on the same cadence as run_episode (LEARN_EVERY ticks), not
            # every single tick — otherwise the interactive viewer both chokes
            # the Pygame FPS and updates weights 4x more often than training.
            if len(self.replay) >= MIN_EPISODES and self.agent.age % LEARN_EVERY == 0:
                self.brain.learn_windows(
                    self.replay.sample_windows(BATCH_SIZE, WINDOW_N))
            # NOTE: policy.decay() intentionally NOT called here — moved to
            # once-per-lifetime above (was a bug before: decaying every tick
            # collapsed epsilon almost immediately).

        self.state = result.x_next
        self.h, self.c = result.h_new, result.c_new

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

        # Terrain background (static per lifetime) — drawn before grid lines/
        # entities/agent so those still render on top of it.
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

        # Vision cone highlight
        fx, fy = World.FACING_VECTORS[self.agent.facing]
        rx, ry = self.world._right(self.agent.facing)
        px, py = self.agent.position
        col_offsets = list(range(-2, 2))

        s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
        s.fill((52, 152, 219, 45))

        for row in range(self.world.vision_range):
            dist = self.world.vision_range - row
            for ci, off in enumerate(col_offsets):
                cx = px + fx * dist + rx * off
                cy = py + fy * dist + ry * off
                if 0 <= cx < self.world_size and 0 <= cy < self.world_size:
                    scr_x = offset_x + cx * cell_size
                    scr_y = offset_y + (self.world_size - 1 - cy) * cell_size
                    self.screen.blit(s, (scr_x, scr_y))
                    pygame.draw.rect(self.screen, (52, 152, 219, 100), (scr_x, scr_y, cell_size, cell_size), 1)

        # Entities: food_low (small orange dot), food_high (bigger gold dot), hazard (red X)
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

        # Agent (triangle pointing in facing direction)
        ax_grid, ay_grid = self.agent.position
        agent_center_x = offset_x + ax_grid * cell_size + cell_size // 2
        agent_center_y = offset_y + (self.world_size - 1 - ay_grid) * cell_size + cell_size // 2

        facing_angles = {0: -np.pi/2, 1: 0, 2: np.pi/2, 3: np.pi}
        angle = facing_angles[self.agent.facing]

        r = cell_size // 2.8
        p1 = (agent_center_x + r * np.cos(angle), agent_center_y + r * np.sin(angle))
        p2 = (agent_center_x + r * np.cos(angle + 2.5), agent_center_y + r * np.sin(angle + 2.5))
        p3 = (agent_center_x + r * np.cos(angle - 2.5), agent_center_y + r * np.sin(angle - 2.5))

        pygame.draw.polygon(self.screen, COLOR_AGENT, [p1, p2, p3])
        pygame.draw.polygon(self.screen, (255, 255, 255), [p1, p2, p3], 2)

        pygame.draw.rect(self.screen, COLOR_TEXT_DIM, grid_rect, 2, border_radius=8)

        # Legend
        legend_y = offset_y + grid_pixel_size + 8
        items = [("Food (low)", COLOR_FOOD_LOW), ("Food (high)", COLOR_FOOD_HIGH),
                ("Food (starter)", COLOR_FOOD_STARTER),
                ("Hazard", COLOR_HAZARD), ("Wall", COLOR_WALL), ("Water", COLOR_WATER)]
        lx = offset_x
        for label, color in items:
            pygame.draw.circle(self.screen, color, (lx + 6, legend_y + 6), 6)
            t = self.font_small.render(label, True, COLOR_TEXT_DIM)
            self.screen.blit(t, (lx + 16, legend_y))
            lx += 16 + t.get_width() + 20

    def _draw_right_panel(self, start_x, start_y):
        # 1. Local View
        card_rect1 = pygame.Rect(start_x, start_y, 410, 165)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect1, border_radius=8)

        txt_title1 = self.font_title.render("AGENT LOCAL VISION", True, COLOR_TEXT)
        self.screen.blit(txt_title1, (start_x + 15, start_y + 12))

        local_grid = self.world.get_local_view(self.agent.position, self.agent.facing)
        vh, vw = local_grid.shape            # rows = VIEW_H, cols = VIEW_W
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

        # 2. Q-value telemetry
        card_rect2 = pygame.Rect(start_x, start_y + 180, 410, 205)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect2, border_radius=8)

        txt_title2 = self.font_title.render("Q-VALUE TELEMETRY Q(s, a)", True, COLOR_TEXT)
        self.screen.blit(txt_title2, (start_x + 15, start_y + 192))

        chart_x = start_x + 25
        chart_y = start_y + 230
        max_q = max(1.0, np.max(np.abs(self.last_q_values)))

        for i, (name, val) in enumerate(zip(ACTION_NAMES, self.last_q_values)):
            bx = chart_x + i * 75
            by = chart_y + 110

            is_chosen = (i == self.last_action)
            bar_color = COLOR_HIGHLIGHT if is_chosen else COLOR_AGENT

            h = int((val / (max_q * 1.2)) * 80)
            if h >= 0:
                bar_rect = pygame.Rect(bx + 15, by - h, 35, max(4, h))
            else:
                bar_rect = pygame.Rect(bx + 15, by, 35, min(-4, -h))

            pygame.draw.rect(self.screen, COLOR_BAR_BG, (bx + 15, by - 80, 35, 80), border_radius=4)
            pygame.draw.rect(self.screen, bar_color, bar_rect, border_radius=4)

            t_name = self.font_bold.render(name, True, COLOR_HIGHLIGHT if is_chosen else COLOR_TEXT)
            self.screen.blit(t_name, (bx + 15, by + 8))

            t_val = self.font_small.render(f"{val:+.2f}", True, COLOR_TEXT_DIM)
            self.screen.blit(t_val, (bx + 12, by + 26))

        # 3. Lifetime & body telemetry
        card_rect3 = pygame.Rect(start_x, start_y + 395, 410, 175)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect3, border_radius=8)

        bar_w = 175
        bar_h = 16

        e_bar_x = start_x + 20
        e_bar_y = start_y + 425
        pygame.draw.rect(self.screen, COLOR_BAR_BG, (e_bar_x, e_bar_y, bar_w, bar_h), border_radius=8)
        energy_pct = max(0.0, min(1.0, self.agent.energy / Agent.MAX_ENERGY))
        e_color = COLOR_ENERGY_GOOD if energy_pct > 0.5 else (COLOR_ENERGY_WARN if energy_pct > 0.2 else COLOR_ENERGY_CRIT)
        if energy_pct > 0:
            pygame.draw.rect(self.screen, e_color, (e_bar_x, e_bar_y, int(bar_w * energy_pct), bar_h), border_radius=8)
        txt_e = self.font_bold.render(f"ENERGY: {self.agent.energy:.0f}/{Agent.MAX_ENERGY:.0f}", True, COLOR_TEXT)
        self.screen.blit(txt_e, (e_bar_x, start_y + 404))

        h_bar_x = start_x + 215
        h_bar_y = start_y + 425
        pygame.draw.rect(self.screen, COLOR_BAR_BG, (h_bar_x, h_bar_y, bar_w, bar_h), border_radius=8)
        health_pct = max(0.0, min(1.0, self.agent.health / Agent.MAX_HEALTH))
        h_color = COLOR_HEALTH_GOOD if health_pct > 0.5 else (COLOR_HEALTH_WARN if health_pct > 0.2 else COLOR_HEALTH_CRIT)
        if health_pct > 0:
            pygame.draw.rect(self.screen, h_color, (h_bar_x, h_bar_y, int(bar_w * health_pct), bar_h), border_radius=8)
        txt_h = self.font_bold.render(f"HEALTH: {self.agent.health:.0f}/{Agent.MAX_HEALTH:.0f}", True, COLOR_TEXT)
        self.screen.blit(txt_h, (h_bar_x, start_y + 404))

        m_x = start_x + 20
        m_y = start_y + 446

        last_event_str = "-"
        if self.last_event is not None:
            last_event_str = self.last_event["type"]

        movement_str = "FREE" if self.agent.is_free else f"cooldown ({self.agent.busy_ticks_remaining})"
        movement_col = COLOR_ENERGY_GOOD if self.agent.is_free else COLOR_HIGHLIGHT

        info_items = [
            (f"Lifetime: #{self.lifetime_count}", COLOR_TEXT),
            (f"Age: {self.agent.age} / {self.agent.max_age} (ticks)", COLOR_TEXT),
            (f"Food Eaten: {self.total_food_eaten}", COLOR_FOOD_LOW),
            (f"Hazard Hits: {self.total_hazard_hits}", COLOR_HAZARD),
            (f"Facing: {FACING_NAMES[self.agent.facing]}", COLOR_TEXT),
            (f"Movement: {movement_str}", movement_col),
            (f"Last Event: {last_event_str}", COLOR_TEXT_DIM),
            (f"Epsilon (\u03b5): {self.policy.epsilon:.3f}", COLOR_TEXT_DIM),
            (f"Last Reward: {self.last_reward:+.3f}", COLOR_ENERGY_GOOD if self.last_reward > 0 else COLOR_TEXT_DIM),
        ]

        for idx, (label_txt, col) in enumerate(info_items):
            col_idx = idx % 2
            row_idx = idx // 2
            t = self.font_main.render(label_txt, True, col)
            self.screen.blit(t, (m_x + col_idx * 195, m_y + row_idx * 18))

    def _draw_control_bar(self, start_x, start_y):
        bar_rect = pygame.Rect(start_x, start_y, 980, 30)
        pygame.draw.rect(self.screen, COLOR_CARD, bar_rect, border_radius=6)

        status_str = (f"STATUS: {'PAUSED' if self.paused else 'RUNNING'} | "
                       f"SPEED: {self.target_fps} FPS | "
                       f"TRAINING: {'ON' if self.training_enabled else 'OFF'}")
        t_status = self.font_bold.render(status_str, True, COLOR_HIGHLIGHT if self.paused else COLOR_ENERGY_GOOD)
        self.screen.blit(t_status, (start_x + 15, start_y + 5))

        help_str = "[SPACE] Pause | [->] Step | [UP/DN] Speed | [R] Reset | [T] Training | [S] Save weights"
        t_help = self.font_small.render(help_str, True, COLOR_TEXT_DIM)
        self.screen.blit(t_help, (start_x + 480, start_y + 7))

    def save_weights(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_lstm_weights(self.net, self.weights_path)
        print(f"Weights saved to {self.weights_path}")

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
                        self._reset_episode()
                    elif event.key == pygame.K_t:
                        self.training_enabled = not self.training_enabled
                        print(f"Online training: {'ON' if self.training_enabled else 'OFF'}")
                    elif event.key == pygame.K_s:
                        self.save_weights()

            if not self.paused:
                self.step_simulation()

            self.draw()
            self.clock.tick(self.target_fps)

        # Auto-save on exit if training was enabled
        if self.training_enabled:
            print("Auto-saving weights before exit...")
            self.save_weights()
        pygame.quit()


if __name__ == "__main__":
    viewer = WorldViewer2D()
    viewer.run()
