import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pygame

# Set up paths
RL_DIR      = Path(__file__).resolve().parent
ROOT_DIR    = RL_DIR.parent.parent
BRAIN_DIR   = ROOT_DIR / "src" / "brain"
WORLD_DIR   = ROOT_DIR / "src" / "world"
SIM_DIR     = ROOT_DIR / "src" / "simulation"
RESULTS_DIR = ROOT_DIR / "results"
SRC_DIR     = ROOT_DIR / "src"

for d in (RL_DIR, BRAIN_DIR, WORLD_DIR, SIM_DIR, SRC_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from world import World
from simulation.session import WorldSession

# ── Viewer defaults ────────────────────────────────────────────────────────
VIEWER_WORLD_SIZE     = 50
VIEWER_NUM_FOOD_LOW   = 142
VIEWER_NUM_FOOD_HIGH  = 114
VIEWER_NUM_HAZARDS    = 75
VIEWER_TARGET_FPS     = 2      # slow/contemplative by default, adjustable live
VIEWER_INITIAL_POPS   = 4      # multiple initial residents so fresh start is lively
WORLD_SAVE_PATH       = RESULTS_DIR / "world_save.json"

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

def _creature_color(wellbeing_pct):
    if wellbeing_pct > 0.6:
        return (46, 204, 113)    # thriving — green
    if wellbeing_pct > 0.3:
        return (241, 196, 15)    # getting by — amber
    return (231, 76, 60)         # struggling — red

CELL_COLORS = {
    World.CELL_UNKNOWN: (28, 30, 36),
    World.CELL_WALL:    COLOR_WALL,
    World.CELL_WATER:   COLOR_WATER,
    World.CELL_SOIL:    COLOR_SOIL,
    World.CELL_GRASS:   COLOR_GRASS,
    5: COLOR_FOOD_LOW,
    6: COLOR_FOOD_HIGH,
    7: COLOR_HAZARD,
    8: COLOR_FOOD_ROTTEN,
}

ACTION_NAMES  = ["Đứng yên", "Tiến", "Lùi", "Quay trái", "Quay phải"]
FACING_NAMES  = ["Lên ↑", "Phải →", "Xuống ↓", "Trái ←"]
FACING_ANGLES = {0: -np.pi/2, 1: 0, 2: np.pi/2, 3: np.pi}


def _mood_label(energy_pct, health_pct):
    worst = min(energy_pct, health_pct)
    if worst > 0.7:
        return "Khỏe mạnh, đang sống tốt", COLOR_ENERGY_GOOD
    if worst > 0.4:
        return "Ổn, nhưng nên tìm thêm ăn", COLOR_ENERGY_WARN
    if worst > 0.15:
        return "Đang đói / yếu, khá nguy hiểm", COLOR_ENERGY_CRIT
    return "Sắp kiệt sức!", COLOR_ENERGY_CRIT


class WorldViewer2D:
    """
    Interactive Pygame viewer for Virtual Lifetime.

    Adheres strictly to the product vision in README.md & AGENTS.md:
      - The living world and its residents are the primary product.
      - Viewer only displays simulation state and conveys user actions.
      - Stepping logic is managed by WorldSession, decoupled from Pygame.
      - Watching the world never silently overwrites model checkpoints.
    """
    def __init__(self,
                 session: Optional[WorldSession] = None,
                 world_size: int = VIEWER_WORLD_SIZE,
                 num_food_low: int = VIEWER_NUM_FOOD_LOW,
                 num_food_high: int = VIEWER_NUM_FOOD_HIGH,
                 num_hazards: int = VIEWER_NUM_HAZARDS,
                 trained_weights_path: Optional[Path] = None):
        pygame.init()
        pygame.font.init()

        self.width = 1260
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Thế Giới Ảo — Virtual Lifetime")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_main  = pygame.font.SysFont("segoeui", 14)
        self.font_bold  = pygame.font.SysFont("segoeui", 14, bold=True)
        self.font_small = pygame.font.SysFont("segoeui", 12)

        if trained_weights_path is None:
            default_weights = RESULTS_DIR / "best_model.pt"
            trained_weights_path = default_weights if default_weights.exists() else None

        if session is not None:
            self.session = session
        else:
            controller_mode = "neural" if trained_weights_path is not None else "rules"
            self.session = WorldSession(
                world_size=world_size,
                num_food_low=num_food_low,
                num_food_high=num_food_high,
                num_hazards=num_hazards,
                initial_population=VIEWER_INITIAL_POPS,
                weights_path=trained_weights_path,
                target_fps=VIEWER_TARGET_FPS,
                enable_learning=False,
                controller_mode=controller_mode
            )

        self.spotlight_id = self.session.population[0].id if self.session.population else None
        self.save_path = WORLD_SAVE_PATH

        # Grid layout cache
        self._grid_offset_x = 30
        self._grid_offset_y = 30
        self._grid_pixel_size = 750
        self._cell_size = max(1, self._grid_pixel_size // self.session.world_size)

    @property
    def world(self):
        return self.session.world

    @property
    def population(self):
        return self.session.population

    @property
    def world_size(self):
        return self.session.world_size

    def _spotlight(self):
        if not self.population:
            return None
        current = next((ind for ind in self.population if ind.id == self.spotlight_id), None)
        if current is not None:
            return current
        oldest = max(self.population, key=lambda ind: ind.agent.age)
        self.spotlight_id = oldest.id
        return oldest

    def _cycle_spotlight(self, step):
        if not self.population:
            return
        ids = sorted(ind.id for ind in self.population)
        if self.spotlight_id not in ids:
            self.spotlight_id = ids[0]
            return
        idx = ids.index(self.spotlight_id)
        self.spotlight_id = ids[(idx + step) % len(ids)]

    def _select_at_screen_pos(self, mx, my):
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
        self.session.step()

    def save_world(self):
        self.session.save(self.save_path)

    def load_world(self):
        self.session = WorldSession.load(self.save_path, enable_learning=False)
        self.spotlight_id = self.session.population[0].id if self.session.population else None
        self._cell_size = max(1, self._grid_pixel_size // self.session.world_size)

    def draw(self):
        self.screen.fill(COLOR_BG)
        self._draw_world_grid(offset_x=30, offset_y=30, grid_pixel_size=750)
        self._draw_right_panel(start_x=810, start_y=30)
        self._draw_control_bar(start_x=30, start_y=860)
        pygame.display.flip()

    def _draw_world_grid(self, offset_x, offset_y, grid_pixel_size):
        cell_size = max(1, grid_pixel_size // self.world_size)
        self._grid_offset_x, self._grid_offset_y = offset_x, offset_y
        self._grid_pixel_size, self._cell_size = grid_pixel_size, cell_size

        grid_rect = pygame.Rect(offset_x, offset_y, grid_pixel_size, grid_pixel_size)
        pygame.draw.rect(self.screen, COLOR_GRID_BG, grid_rect, border_radius=8)

        # Terrain background
        for tx in range(self.world_size):
            for ty in range(self.world_size):
                terrain_code = int(self.world.terrain[tx, ty])
                color = CELL_COLORS.get(terrain_code, COLOR_GRID_BG)
                rect = pygame.Rect(offset_x + tx * cell_size,
                                   offset_y + (self.world_size - 1 - ty) * cell_size,
                                   cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect)

        if cell_size >= 8:
            for i in range(self.world_size + 1):
                pygame.draw.line(self.screen, COLOR_GRID_LINE,
                                 (offset_x + i * cell_size, offset_y),
                                 (offset_x + i * cell_size, offset_y + grid_pixel_size), 1)
                pygame.draw.line(self.screen, COLOR_GRID_LINE,
                                 (offset_x, offset_y + i * cell_size),
                                 (offset_x + grid_pixel_size, offset_y + i * cell_size), 1)

        # Seeds
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

        # Residents
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

        if self.session.extinct:
            msg = self.font_title.render("THẾ GIỚI ĐÃ TUYỆT DIỆT — nhấn R để bắt đầu lại", True, COLOR_EXTINCT)
            self.screen.blit(msg, (offset_x + 30, offset_y + grid_pixel_size // 2 - 10))

        # Legend
        legend_y = offset_y + grid_pixel_size + 8
        items = [("Thức ăn nhỏ", COLOR_FOOD_LOW), ("Thức ăn to", COLOR_FOOD_HIGH),
                ("Thức ăn khởi đầu", COLOR_FOOD_STARTER), ("Đã thối", COLOR_FOOD_ROTTEN),
                ("Hạt giống", COLOR_SEED), ("Nguy hiểm", COLOR_HAZARD),
                ("Tường", COLOR_WALL), ("Nước", COLOR_WATER)]
        lx = offset_x
        for label, color in items:
            pygame.draw.circle(self.screen, color, (lx + 6, legend_y + 6), 6)
            t = self.font_small.render(label, True, COLOR_TEXT_DIM)
            self.screen.blit(t, (lx + 16, legend_y))
            lx += 16 + t.get_width() + 20

        hint = self.font_small.render(
            "Màu cư dân: xanh = khỏe mạnh, vàng = tạm ổn, đỏ = yếu / nguy hiểm",
            True, COLOR_TEXT_DIM)
        self.screen.blit(hint, (offset_x, legend_y + 22))

    def _draw_right_panel(self, start_x, start_y):
        spot = self._spotlight()

        # 1. Local view of spotlight resident
        card_rect1 = pygame.Rect(start_x, start_y, 410, 165)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect1, border_radius=8)
        title = "TẦM NHÌN CỦA CƯ DÂN"
        if spot is not None:
            title = f"TẦM NHÌN CỦA CƯ DÂN #{spot.id} (thế hệ {spot.generation})"
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
                "Vùng quan sát trực tiếp của cư dân",
                "Hàng trên: xa nhất phía trước mặt",
                "Hàng dưới: ô gần nhất phía trước mặt",
                "Màu sắc tương ứng bản đồ chính",
            ]
            for idx, txt in enumerate(lbls):
                t = self.font_small.render(txt, True, COLOR_TEXT_DIM)
                self.screen.blit(t, (legend_x, legend_y + idx * 22))
        else:
            t = self.font_main.render("Không còn cư dân nào để theo dõi.", True, COLOR_TEXT_DIM)
            self.screen.blit(t, (start_x + 20, start_y + 60))

        # 2. Behavior intuition & cooldown status
        card_rect2 = pygame.Rect(start_x, start_y + 180, 410, 205)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect2, border_radius=8)
        txt_title2 = self.font_title.render("ĐÁNH GIÁ TRỰC GIÁC (Q-VALUES)", True, COLOR_TEXT)
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

            if not spot.agent.is_free:
                status_txt = self.font_bold.render(
                    f"→ Đang thực hiện (chờ: {spot.agent.busy_ticks_remaining} tick)", True, COLOR_ENERGY_WARN)
            else:
                status_txt = self.font_bold.render(
                    f"→ Ưu tiên nhất: {ACTION_NAMES[chosen]}", True, COLOR_HIGHLIGHT)
            self.screen.blit(status_txt, (start_x + 20, start_y + 355))

            explore_pct = spot.brain.policy.epsilon * 100
            explore_txt = self.font_small.render(
                f"(thử nghiệm: {explore_pct:.0f}%)", True, COLOR_TEXT_DIM)
            self.screen.blit(explore_txt, (start_x + 270, start_y + 358))

        # 3. World Status & Spotlight Telemetry
        card_rect3 = pygame.Rect(start_x, start_y + 395, 410, 175)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect3, border_radius=8)
        txt_title3 = self.font_title.render("TRẠNG THÁI CƯ DÂN & THẾ GIỚI", True, COLOR_TEXT)
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
            txt_e = self.font_small.render(f"Năng lượng: {spot.agent.energy:.0f}/{spot.agent.MAX_ENERGY:.0f}", True, COLOR_TEXT)
            self.screen.blit(txt_e, (e_bar_x, start_y + 430))

            h_bar_x = start_x + 215
            h_bar_y = start_y + 448
            pygame.draw.rect(self.screen, COLOR_BAR_BG, (h_bar_x, h_bar_y, bar_w, bar_h), border_radius=8)
            health_pct = max(0.0, min(1.0, spot.agent.health / spot.agent.MAX_HEALTH))
            h_color = COLOR_HEALTH_GOOD if health_pct > 0.5 else (COLOR_HEALTH_WARN if health_pct > 0.2 else COLOR_HEALTH_CRIT)
            if health_pct > 0:
                pygame.draw.rect(self.screen, h_color, (h_bar_x, h_bar_y, int(bar_w * health_pct), bar_h), border_radius=8)
            txt_h = self.font_small.render(f"Sức khỏe: {spot.agent.health:.0f}/{spot.agent.MAX_HEALTH:.0f}", True, COLOR_TEXT)
            self.screen.blit(txt_h, (h_bar_x, start_y + 430))

            mood_txt, mood_color = _mood_label(energy_pct, health_pct)
            t_mood = self.font_bold.render(f"{mood_txt} (Tuổi: {spot.agent.age}/{spot.agent.max_age})", True, mood_color)
            self.screen.blit(t_mood, (start_x + 20, start_y + 470))

            direction = FACING_NAMES[spot.agent.facing]
            action = ACTION_NAMES[spot.last_action]
            activity = self.font_small.render(
                f"Hướng: {direction} | Gần nhất: {action}", True, COLOR_TEXT_DIM)
            self.screen.blit(activity, (start_x + 20, start_y + 488))

        oldest = self.session.get_oldest_resident()
        oldest_str = f"#{oldest.id} ({oldest.agent.age} tick)" if oldest else "-"

        m_x = start_x + 20
        m_y = start_y + 505
        info_items = [
            (f"Số cư dân đang sống: {len(self.population)}", COLOR_TEXT),
            (f"Thời gian: {self.session.tick_count} tick", COLOR_TEXT),
            (f"Sinh ra: {self.session.session_births}", COLOR_ENERGY_GOOD),
            (f"Mất đi: {self.session.session_deaths}", COLOR_HAZARD),
            (f"Thế hệ cao nhất: {max((i.generation for i in self.population), default='-')}", COLOR_TEXT),
            (f"Cư dân cao tuổi nhất: {oldest_str}", COLOR_TEXT_DIM),
        ]
        for idx, (label_txt, col) in enumerate(info_items):
            col_idx = idx % 2
            row_idx = idx // 2
            t = self.font_main.render(label_txt, True, col)
            self.screen.blit(t, (m_x + col_idx * 195, m_y + row_idx * 18))

        # 4. Resident history + recent world events
        card_rect4 = pygame.Rect(start_x, start_y + 580, 410, 265)
        pygame.draw.rect(self.screen, COLOR_CARD, card_rect4, border_radius=8)
        txt_title4 = self.font_title.render("LỊCH SỬ & NHẬT KÝ THẾ GIỚI", True, COLOR_TEXT)
        self.screen.blit(txt_title4, (start_x + 15, start_y + 590))

        resident_events = list(spot.event_history[-3:]) if spot is not None else []
        world_events = self.session.recent_events[-4:]
        events_to_show = ([f"Cư dân: {event}" for event in resident_events]
                          + [f"World: {event}" for event in world_events])[-7:]
        ey = start_y + 625
        if not events_to_show:
            t = self.font_small.render("Chưa có biến cố nào diễn ra.", True, COLOR_TEXT_DIM)
            self.screen.blit(t, (start_x + 20, ey))
        else:
            for ev in events_to_show:
                t = self.font_small.render(ev, True, COLOR_TEXT_DIM)
                self.screen.blit(t, (start_x + 20, ey))
                ey += 24

    def _draw_control_bar(self, start_x, start_y):
        bar_rect = pygame.Rect(start_x, start_y, self.width - 2 * start_x, 30)
        pygame.draw.rect(self.screen, COLOR_CARD, bar_rect, border_radius=6)

        status = "TUYỆT DIỆT" if self.session.extinct else ("TẠM DỪNG" if self.session.paused else "ĐANG CHẠY")
        status_color = COLOR_EXTINCT if self.session.extinct else (COLOR_HIGHLIGHT if self.session.paused else COLOR_ENERGY_GOOD)
        status_str = f"{status}  |  Tốc độ: {self.session.target_fps} fps  |  Thế giới: {self.world_size}x{self.world_size}"
        t_status = self.font_bold.render(status_str, True, status_color)
        self.screen.blit(t_status, (start_x + 15, start_y + 5))

        help_str = ("[Click] Chọn cư dân | [Space] Tạm dừng | [Enter] Bước tick | [←/→] Đổi cư dân "
                    "| [↑/↓] Tốc độ | [S] Lưu | [L] Mở | [R] Khởi tạo lại")
        t_help = self.font_small.render(help_str, True, COLOR_TEXT_DIM)
        self.screen.blit(t_help, (start_x + 480, start_y + 7))

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
                        self.session.toggle_pause()
                    elif event.key == pygame.K_RETURN and self.session.paused:
                        self.session.step()
                    elif event.key == pygame.K_UP:
                        self.session.adjust_speed(1)
                    elif event.key == pygame.K_DOWN:
                        self.session.adjust_speed(-1)
                    elif event.key == pygame.K_LEFT:
                        self._cycle_spotlight(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._cycle_spotlight(1)
                    elif event.key == pygame.K_r:
                        self.session.reset()
                        self.spotlight_id = self.session.population[0].id if self.session.population else None
                    elif event.key == pygame.K_s:
                        self.save_world()
                    elif event.key == pygame.K_l and self.save_path.exists():
                        self.load_world()

            if not self.session.paused:
                self.session.step()

            self.draw()
            self.clock.tick(self.session.target_fps)

        pygame.quit()


if __name__ == "__main__":
    viewer = WorldViewer2D()
    viewer.run()
