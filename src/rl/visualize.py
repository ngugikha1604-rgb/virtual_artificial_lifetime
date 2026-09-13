import sys
from pathlib import Path

RL_DIR    = Path(__file__).resolve().parent
WORLD_DIR = RL_DIR.parent / "world"
for _dir in (RL_DIR, WORLD_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from world import World

FACING_MARKERS = {0:"^", 1:">", 2:"v", 3:"<"}

# Terrain code -> RGB (0-1 floats), indexed by World.CELL_WALL/WATER/SOIL/GRASS
# (1..4). Kept as its own small table here rather than importing colors from
# live_viewer.py (that file is Pygame-specific / int 0-255 colors) — same
# duplication tradeoff already accepted for world.visible_entity_positions()
# vs get_local_view(): keep both in sync by hand if the terrain codes change.
TERRAIN_COLORS = {
    World.CELL_WALL:  (0.35, 0.37, 0.41),
    World.CELL_WATER: (0.36, 0.68, 0.91),
    World.CELL_SOIL:  (0.59, 0.44, 0.20),
    World.CELL_GRASS: (0.64, 0.82, 0.51),
}


def snapshot(world, agent, step, event):
    """One frame's worth of drawable state. Public (not `_snapshot`) so
    run_episode.py can reuse it to record an ACTUAL training lifetime
    in-place (see run_episode.run_episode(..., record=True)) instead of
    only being able to record a separate, untracked demo episode."""
    return {
        "step": step, "position": agent.position, "facing": agent.facing,
        "food_low_positions":     list(world.positions_by_type("food_low")),
        "food_high_positions":    list(world.positions_by_type("food_high")),
        "food_starter_positions": list(world.positions_by_type("food_starter")),
        "hazard_positions":       list(world.positions_by_type("hazard")),
        "seed_positions":         list(world.seeds.keys()),
        "energy": agent.energy, "health": agent.health, "event": event,
        "busy": not agent.is_free,
    }


def save_lifetime_gif(frames, width, height, output_path, fps=5, terrain=None):
    fig, (ax_world, ax_bars) = plt.subplots(1, 2, figsize=(10, 5),
                                             gridspec_kw={"width_ratios":[1,1]})
    fig.subplots_adjust(top=0.85); fig.suptitle("Virtual Lifetime", y=0.97)

    ax_world.set_xlim(-0.5,width-0.5); ax_world.set_ylim(-0.5,height-0.5)
    ax_world.set_xticks(range(width)); ax_world.set_yticks(range(height))
    ax_world.grid(True,linewidth=0.5,color="lightgray"); ax_world.set_aspect("equal")
    ax_world.set_title("World",fontsize=10)

    if terrain is not None:
        # terrain[x, y] -> image array indexed [row=y, col=x] for imshow's
        # (rows, cols) convention, origin="lower" so y increases upward same
        # as the scatter markers below. Drawn ONCE (terrain is static for the
        # whole lifetime) rather than per-frame — much cheaper than redrawing
        # a full grid of colored cells on every one of ~100-250 frames.
        img = np.array([[TERRAIN_COLORS.get(int(terrain[x, y]), (1, 1, 1))
                         for x in range(width)] for y in range(height)])
        ax_world.imshow(img, extent=(-0.5, width-0.5, -0.5, height-0.5),
                        origin="lower", zorder=0)
    agent_dot,=ax_world.plot([],[],marker="^",color="royalblue",markersize=18,label="Agent")
    seed_dots,=ax_world.plot([],[],marker=".",color="yellowgreen",markersize=8,
                              linestyle="None",label="Seed (growing)")
    food_low_dots,=ax_world.plot([],[],marker="*",color="orange",markersize=14,
                                  linestyle="None",label="Food (low)")
    food_high_dots,=ax_world.plot([],[],marker="*",color="gold",markersize=22,
                                   linestyle="None",label="Food (high)")
    food_starter_dots,=ax_world.plot([],[],marker="D",color="mediumvioletred",markersize=10,
                                      linestyle="None",label="Food (starter)")
    hazard_dots,=ax_world.plot([],[],marker="X",color="crimson",markersize=16,
                                linestyle="None",label="Hazard")
    ax_world.legend(loc="upper right",fontsize=7)
    step_text=ax_world.text(0.02,1.03,"",transform=ax_world.transAxes,fontsize=9)

    ax_bars.set_xlim(0,2); ax_bars.set_ylim(0,100)
    ax_bars.set_title("Body",fontsize=10); ax_bars.set_xticks([0.5,1.5])
    ax_bars.set_xticklabels(["Energy","Health"]); ax_bars.set_ylabel("value")
    e_bar=ax_bars.bar([0.5],[0],width=0.6,color="mediumseagreen")[0]
    h_bar=ax_bars.bar([1.5],[0],width=0.6,color="steelblue")[0]
    e_txt=ax_bars.text(0.5,3,"",ha="center",fontsize=9,color="white",fontweight="bold")
    h_txt=ax_bars.text(1.5,3,"",ha="center",fontsize=9,color="white",fontweight="bold")

    def update(i):
        fr = frames[i]
        event      = fr["event"]
        ate_food   = event is not None and event["category"] == "food"
        hit_hazard = event is not None and event["category"] == "hazard"

        bg = "#e8ffe8" if ate_food else ("#ffe8e8" if hit_hazard else ("#f0f0f5" if fr["busy"] else "white"))
        ax_world.set_facecolor(bg)
        agent_dot.set_data([fr["position"][0]],[fr["position"][1]])
        agent_dot.set_marker(FACING_MARKERS[fr["facing"]])

        food_low_dots.set_data([p[0] for p in fr["food_low_positions"]],
                                [p[1] for p in fr["food_low_positions"]])
        seed_dots.set_data([p[0] for p in fr["seed_positions"]],
                            [p[1] for p in fr["seed_positions"]])
        food_high_dots.set_data([p[0] for p in fr["food_high_positions"]],
                                 [p[1] for p in fr["food_high_positions"]])
        food_starter_dots.set_data([p[0] for p in fr["food_starter_positions"]],
                                    [p[1] for p in fr["food_starter_positions"]])
        hazard_dots.set_data([p[0] for p in fr["hazard_positions"]],
                              [p[1] for p in fr["hazard_positions"]])

        label = "ATE FOOD!" if ate_food else ("HAZARD!" if hit_hazard else ("cooldown..." if fr["busy"] else ""))
        step_text.set_text(f"tick {fr['step']}  {label}")
        e=max(fr["energy"],0); hp=max(fr["health"],0)
        e_bar.set_height(e); e_bar.set_color("mediumseagreen" if e>30 else "tomato")
        h_bar.set_height(hp); h_bar.set_color("steelblue" if hp>50 else "darkorange")
        e_txt.set_text(f"{e:.0f}"); h_txt.set_text(f"{hp:.0f}")
        return (agent_dot, seed_dots, food_low_dots, food_high_dots, food_starter_dots, hazard_dots,
                step_text, e_bar, h_bar, e_txt, h_txt)

    anim=FuncAnimation(fig,update,frames=len(frames),interval=1000/fps,blit=False)
    anim.save(str(output_path),writer=PillowWriter(fps=fps))
    plt.close(fig)
