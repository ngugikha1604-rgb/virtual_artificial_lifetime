"""
config.py — single source of truth for all global constants.

Every entry point (experiments/phase5_online_qlearning/run_experiment.py,
src/rl/live_viewer.py, src/rl/run_episode.py, ...) reads its world + RL
settings from here instead of re-declaring them, so the world an agent is
trained on always matches the world it is then visualized in.

Rationale / earlier duplication:
  - run_experiment.py declared world_size=10, food 2/2, hazard 3, max_age 400
  - visualize_lifetime.py declared TRAIN_LIFETIMES=1000 subset
  - live_viewer.py hard-coded the SAME world_size=10 / 2 / 2 / 3 / 400 and RL
    hyperparameters in its __init__ signature defaults
If these ever drifted, a trained brain would be framed against a different
world than it was trained on. Now they all point here.

The project philosophy is <<understand > abstraction>>, so this file stays a
plain, readable set of module-level constants (no nested config loader, no env
files, no magic). Searching for a value means reading this one file.

Agent body constants live here too (energy/health/damage...) so reward / damage
tuning experiments (see progress.md: "tinh chỉnh reward/damage constants") are a
one-file edit — src/world/agent.py no longer hard-codes them.
"""

# ── World layout (shared by training + live viewer + demo recording) ────────
WORLD_SIZE      = 10      # square grid (world default in __init__ used elsewhere)
NUM_FOOD_LOW    = 5
NUM_FOOD_HIGH   = 4
NUM_HAZARDS     = 3
# A single food that (a) never respawns once eaten and (b) always spawns
# exactly one cell in front of the agent's spawn point (World.spawn_point()) —
# see World's docstring / _init_entities. Purpose: guarantee a newborn agent
# can succeed at eating something immediately, rather than depending on
# wandering into the zone-distributed food below before it happens to notice
# food exists at all. NUM_FOOD_LOW + NUM_FOOD_HIGH must be >= the number of
# 3x3 zones ((WORLD_SIZE // World.ZONE_SIZE) ** 2 == 9 at these defaults) for
# World._init_entities()'s "at least one food per zone" guarantee to hold —
# 5+4=9 exactly covers it. If you change WORLD_SIZE or World.ZONE_SIZE,
# recompute this and bump the counts to match.
NUM_FOOD_STARTER = 1
MAX_AGE         = 400     # ticks a lifetime may last at most

# ── Terrain (background layer, separate from food/hazard entities) ──────────
# Terrain is a persistent per-cell background (generated once at World.__init__,
# never changes during a lifetime) that entities sit ON TOP of — unlike food/
# hazard, terrain is never eaten/consumed/respawned. Kept as its own grid
# (World.terrain) rather than being modeled as entities because its lifecycle
# is completely different (static vs. ephemeral) — see progress.md.
#
# Codes share ONE alphabet with entities in the local-view grid (World's
# get_local_view() shows whichever is present in a cell — entity takes
# priority, else terrain). See World's docstring for the authoritative table.
#
# grass/soil: both freely walkable, currently mechanically identical (a hook
#   for later, e.g. biasing food spawns toward one) — SOIL_FRACTION is the
#   fraction of the default ground fill that becomes soil vs grass.
# wall (interior, not just the world boundary): blocks movement outright —
#   World.step() reverts the move (agent "bumps" and stays put, still paying
#   the tick's energy cost) if the destination cell is a wall.
# water: walkable, but costs WATER_DURATION_MULTIPLIER x as many busy-ticks
#   for a forward/backward move that lands on it (Agent.commit_action) —
#   since every tick, free or busy, still costs ENERGY_COST regardless of
#   action, a longer duration is also a proportionally larger energy cost,
#   so one multiplier models both "wading takes longer" and "wading tires
#   you out more" without needing two separate mechanisms.
#
# Generation carves a few small WATER patches + WALL line segments out of a
# random grass/soil base fill, keeping counts/sizes modest relative to a
# 10x10 world so they add texture without much risk of sealing off an area
# (see progress.md for the accepted simplification: no full reachability/
# connectivity check is done — revisit if the agent starts getting stuck).
# The world-center spawn cell (see Agent.__init__) is always kept clear of
# wall/water so the agent never spawns already blocked in.
NUM_WATER_PATCHES = 2      # count of water patches carved per world
WATER_PATCH_SIZE  = 2      # each patch is WATER_PATCH_SIZE x WATER_PATCH_SIZE
NUM_WALL_SEGMENTS = 2      # count of wall line-segments carved per world
WALL_SEGMENT_LEN  = 3      # length of each wall segment (straight, random orientation)
SOIL_FRACTION     = 0.30   # fraction of the default ground fill that's soil (rest grass)
WATER_DURATION_MULTIPLIER = 2   # x ACTION_DURATION for a move landing on water

# Local-view geometry — the source of truth; World.__init__ defaults mirror it.
# The agent sees VISION_RANGE squares straight ahead, VISION_WIDTH columns wide,
# plus BEHIND_ROWS squares behind it. The observation grid World.get_local_view()
# produces has shape (VIEW_H, VIEW_W) = (VISION_RANGE + BEHIND_ROWS, VISION_WIDTH).
# These drive the network's input size (per-cell one-hot classes + conv), so
# changing VISION_RANGE here re-sizes the Conv-LSTM and invalidates any saved
# weights (model_io refuses the shape mismatch and falls back to a fresh net).
VISION_RANGE   = 4        # how many cells ahead the agent can see (was 2)
VISION_WIDTH   = 4        # column width of the vision cone
BEHIND_ROWS    = 2        # rows behind the agent (kept unknown-ish constant)
VIEW_H         = VISION_RANGE + BEHIND_ROWS   # observation grid rows  (2+4 = 6)
VIEW_W         = VISION_WIDTH                 # observation grid columns (4)

# ── RL / network hyper-parameters ─────────────────────────────────────────────
NUM_ACTIONS     = 5       # 0=stay,1=fwd,2=bwd,3=turnL,4=turnR
HIDDEN_SIZE     = 64      # LSTM hidden size

# One-hot channels: one binary channel per cell code 0..7 (unknown, wall,
# water, soil, grass, food_low, food_high, hazard). food_starter (see
# NUM_FOOD_STARTER) deliberately reuses food_low's code/channel rather than
# getting its own — see World's docstring — so this does NOT need bumping to
# 9 for it. Keeping the binary-per-content-class mapping rather than a single
# scalar magnitude removes the implicit "better/worse" ordering a magnitude
# encoding would impose on discrete object/terrain classes. The network
# learns each channel's importance from data.
NUM_CELL_CLASSES = 8      # len of World cell-code alphabet (0..7 incl. unknown)

# Conv2D that maps the multi-channel local view down to spatial features.
# Same-pad convolution preserves the input spatial size, so the feature map is
# (CONV_FILTERS, VIEW_H, VIEW_W) for any (non-square) view.
CONV_FILTERS    = 8       # number of output filters
CONV_KERNEL     = 3       # square kernel edge length
CONV_PAD        = 1       # "same" padding keeps the spatial size unchanged
# Flattened conv feature length = CONV_FILTERS * VIEW_H * VIEW_W.
CONV_OUT        = CONV_FILTERS * VIEW_H * VIEW_W

# LSTM input = flattened conv output + 3 body-state (energy, health, age).
INPUT_SIZE      = CONV_OUT + 3

# The observation vector an episode stores in replay is the one-hot grid +
# body state (conv weights live in the agent and run per forward, not on the
# stored vector). Its length is what run_episode/live_viewer push as x.
OBS_GRID_FLAT   = NUM_CELL_CLASSES * VIEW_H * VIEW_W   # one-hot, flattened
OBS_SIZE        = OBS_GRID_FLAT + 3

LEARNING_RATE   = 1e-3    # Adam + DQN: 0.01 is far too hot (Q-values oscillate
                          # / "moving target" overestimation). 1e-3..1e-4 is the
                          # standard stable range; 1e-3 chosen as the aggressive
                          # end (still well below the old 0.01).
GAMMA           = 0.9
# REPLAY_CAPACITY counts EPISODES now, not transitions. 20_000 episodes would
# hold 4-8M ticks (~obs 147 + hidden 64 + cell 64 floats each) -> multiple GB of
# RAM and, over 4000 lifetimes, the buffer never evicts. A few hundred episodes
# is plenty (a lifetime is ~200 ticks, so 200 episodes ≈ 40k ticks).
REPLAY_CAPACITY = 200     # max number of stored EPISODES

# Phase B — truncated BPTT over replay windows.
#  - BATCH_SIZE  : number of episode-windows sampled per learn step.
#  - WINDOW_N    : max ticks unrolled through the LSTM per window in one
#                  autograd graph (truncated BPTT; gradient flows back <= N steps).
#  - MIN_EPISODES: don't learn until this many completed episodes (any length)
#                  are buffered.
BATCH_SIZE      = 8
WINDOW_N        = 16
MIN_EPISODES    = 10
LEARN_EVERY     = 4             # learn every N ticks of the live episode

# Target network (classic DQN fix, ported to the windowed/recurrent setting):
# learn_windows() bootstraps step k's target from a SEPARATE, periodically-
# frozen copy of the net (its own hidden-state rollout, not borrowed from the
# online unroll) instead of the live weights being updated this very step.
# Without this, the bootstrap target moves every single Adam step ("chasing
# a moving target"), which is a known source of Q-value oscillation/instability.
# Unit is LEARN CALLS (a learn_windows() invocation), not ticks or lifetimes:
# at LEARN_EVERY=4 and a ~150-250 tick lifetime there are roughly 40-60 learn
# calls per lifetime, so 200 syncs the target net every ~4-5 lifetimes — frozen
# long enough to give a stable bootstrap, refreshed often enough to track the
# online net without lagging for an entire long run.
TARGET_SYNC_EVERY = 200

EPSILON_START   = 1.0
# A permanent 15% exploit noise (the old default) meant the agent never behaved
# near-greedily late in training — one in ~7 actions stayed random forever. 0.05
# is a standard exploration floor: still enough to escape local optima / loops,
# low enough that the final policy is mostly exploitative after epsilon decays.
EPSILON_MIN     = 0.05
EPSILON_DECAY   = 0.9995  # per-lifetime decay; hits eps_min late in a 4000-lt run

# ── Agent body mechanics ──────────────────────────────────────────────────────
MAX_ENERGY      = 100.0
START_ENERGY    = 70.0
MAX_HEALTH      = 100.0
START_HEALTH    = 100.0
STARVATION_DAMAGE = 2.0
HAZARD_DAMAGE     = 5.0
DEFAULT_MAX_AGE   = 200    # fallback when no max_age passed to Agent (world default)
ENERGY_COST       = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
FOOD_EFFECTS      = {
    "food_low":     {"energy": 30.0, "health": 10.0},
    "food_high":    {"energy": 60.0, "health": 20.0},
    # Same nutrition as food_low — its whole purpose is WHEN/WHERE it appears
    # (guaranteed at birth), not being extra nutritious. See NUM_FOOD_STARTER.
    "food_starter": {"energy": 30.0, "health": 10.0},
}
# ticks each action occupies the agent before it can pick a new one; turning /
# staying are instantaneous (duration 1). See Agent.ACTION_DURATION.
ACTION_DURATION   = {0: 1, 1: 3, 2: 3, 3: 1, 4: 1}

# ── Reward shaping ─────────────────────────────────────────────────────────────
# Values are DESIGNED, not yet calibrated by a long experiment run — re-tune
# after you get a real 4000-lifetime benchmark (progress.md notes these were
# placeholders).
#
# Why these numbers:
#  - SURVIVAL_BONUS 0.001 (was 0.01): a *constant* per-tick positive paid for
#    every state, including idling, biases every Q-value upward uniformly and
#    adds a noise floor that can exceed a single shaping step (max ~0.01/tick
#    with the old 0.05 weight over max_dist). Cut 10x so it no longer drowns
#    the movement guidance while still gently favouring staying alive.
#  - SHAPE_WEIGHT_* doubled: horizontal distance is normalized by max_dist, so
#    each step's shaping contribution is small; potential-based shaping is what
#    actually steers the agent toward food / away from hazard between the rare
#    eat events. Weakening it relative to survival would let pure idle-noise
#    dominate the online gradient. Raised to 0.1 (still well below an eat bonus)
#    so it guides without overriding FOOD_BONUS / HAZARD_PENALTY.
#  - FOOD_BONUS / HAZARD_PENALTY unchanged: these are event-level (eat 3/5,
#    hazard -1) and dominate whenever an interaction happens, as intended.
SURVIVAL_BONUS      = 0.001
FOOD_BONUS          = {"food_low": 3.00, "food_high": 5.00, "food_starter": 3.00}
HAZARD_PENALTY      = 1.00
# Starvation (energy hits 0 -> STARVATION_DAMAGE to health every tick, see
# Agent.apply_energy_cost) used to have NO direct reward penalty, unlike
# hazard — both are lethal-if-sustained sources of health loss, but only one
# was ever penalised explicitly. The other signal the agent had for starving
# was purely indirect (dying sooner = fewer future SURVIVAL_BONUS/food ticks),
# a much weaker gradient than a flat per-tick penalty. Same magnitude as
# HAZARD_PENALTY by default since both are "actively taking lethal damage
# right now" events of comparable severity — re-tune independently once you
# have baseline data on how often each actually triggers.
STARVATION_PENALTY  = 1.00
SHAPE_WEIGHT_FOOD   = 0.10
SHAPE_WEIGHT_HAZARD = 0.10

# ── Checkpointing (best-model tracking + periodic saves) ─────────────────────
# Training used to save weights ONLY once, after the entire run finished:
# (1) a crash/interrupt mid-run lost all progress, and (2) the final weights
# are whatever the policy happened to be doing on the LAST few lifetimes,
# which in RL can be WORSE than an earlier point (Q-value drift/oscillation
# late in training is a known failure mode, not just theoretical — overfitting
# to a recent unlucky/lucky streak of lifetimes is a real risk here since there
# is no held-out validation set, just "whatever happened most recently"). Now,
# every SAVE_EVERY lifetimes OR whenever the ROLLING MEAN total_reward over the
# last BEST_METRIC_WINDOW lifetimes beats the best ever seen, results/best_model.pt
# is re-saved (see run_experiment.py / training_state.py) — there is only ONE
# model file (shared by run_experiment.py and live_viewer.py, see
# run_experiment.py's module docstring for why), so this protects against
# losing an interrupted run's progress but, unlike an earlier design with a
# separate best-only file, does NOT protect against a later save overwriting
# it with weights that turn out worse.
SAVE_EVERY          = 200   # lifetimes between periodic checkpoint saves (see below)
BEST_METRIC_WINDOW  = 50    # lifetimes averaged for the "is this a new best" check

# ── Experiment run configuration ──────────────────────────────────────────────
NUM_LIFETIMES  = 4000     # how many lifetimes to train across
PRINT_EVERY    = 100      # progress line frequency during training
SEED           = 42
DEMO_EPSILON   = 0.1      # demo-recording exploration (mostly-greedy; a touch
                          # of noise lets a greedy agent escape deterministic
                          # oscillation between two states — see visualize note)
DEMO_FPS       = 5
