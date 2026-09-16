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
# Codes share ONE alphabet with entities in the local-view grid. World has TWO
# ways to read a cell: get_local_view() collapses to one representative code
# per cell (entity wins over terrain — used for human-facing display only),
# while get_local_view_layers() keeps terrain and entity as independent bits
# that the network's multi-hot observation encoding can both set at once (see
# NUM_CELL_CLASSES above and World's docstring for the authoritative table).
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

# ── Food growth (replaces instant "teleport respawn" with an organic sim) ─────
# food_low/food_high used to instantly reappear elsewhere the moment one was
# eaten (World._respawn_entity, zone-balanced). Now they're removed for good
# on eat ("respawns": False, same as food_starter) and NEW food only appears
# through this per-tick growth simulation — chosen because instant magic
# relocation never fit "world hợp lý hơn" (a more believable/organic world),
# and because it's what finally makes soil vs grass mechanically different,
# not just differently colored (see SOIL_FRACTION above, previously just a
# hook for this).
#
# TWO-STAGE growth, not "roll dice -> food appears" in one step:
#   1. SEEDING (stochastic): each tick, every eligible empty cell (grass or
#      soil, no entity, not already seeded) has a small chance of becoming a
#      seed (World.seeds: {(x,y): ticks_remaining}). Chance =
#          FOOD_SEED_BASE_RATE[terrain]
#        + FOOD_SEED_SPREAD_BONUS[terrain]  (only if an orthogonally-adjacent
#                                             cell currently has a MATURE food
#                                             item — "reproduction"/spreading
#                                             from existing food, not other
#                                             seeds)
#      Soil seeds spontaneously FASTER (base rate) but spreads slower;
#      grass is the opposite — spreads fast near existing food but rarely
#      starts one on its own. (Khanh's choice — "soil mọc nền nhanh hơn, grass
#      lan truyền nhanh hơn".)
#   2. MATURATION (deterministic): once seeded, a fixed countdown (randomized
#      per seed between MIN/MAX below) ticks down; at 0 the seed becomes an
#      actual food entity. This is what makes growth genuinely feel like
#      "planted, then takes time to grow" instead of an instant probability
#      roll — a seed's arrival is stochastic, but once it exists, its
#      maturity time is not, so growth is visibly staged over time rather
#      than popping food into existence at random.
# Seeds are NOT visible to the agent (no observation channel — kept out of
# NUM_CELL_CLASSES on purpose, so this doesn't re-invalidate the network); a
# human watching the GIF/live_viewer CAN see them (small distinct marker) —
# this can change later if "agent waits near a maturing seed" behavior is
# wanted, at the cost of another observation channel.
#
# Food TYPE on maturation is a fixed global split, independent of terrain or
# the neighbor that triggered the spread (Khanh's choice — simplest option).
FOOD_SEED_BASE_RATE   = {"soil": 0.003, "grass": 0.001}
FOOD_SEED_SPREAD_BONUS = {"soil": 0.005, "grass": 0.012}
FOOD_SEED_MATURATION_MIN = 15   # ticks a seed takes to become food (fastest)
FOOD_SEED_MATURATION_MAX = 30   # ticks a seed takes to become food (slowest)
FOOD_GROWTH_TYPE_SPLIT = {"food_low": 0.7, "food_high": 0.3}
# Cap on (mature food + pending seeds) combined — growth simply stops
# attempting NEW seeds once reached (existing seeds already "in progress"
# still mature normally, so actual food count can briefly exceed this right
# after several mature at once). Prevents the spreading/reproduction bonus
# from compounding unbounded and carpeting the whole 10x10 world; think of it
# as the world's food carrying capacity. (2026-09: lowered 20->14 + halved the
# rates above — Khanh found the world felt too food-dense; still comfortably
# above the initial NUM_FOOD_LOW+NUM_FOOD_HIGH+NUM_FOOD_STARTER=10 so there's
# some room to regrow, just less of it.)
MAX_FOOD_ON_WORLD = 14

# ── Food aging / expiration ────────────────────────────────────────────────────
# Once a seed matures into food_low, it ages through a fixed lifecycle:
#   food_low  -> food_high  (after FOOD_AGE_STAGE_TICKS ticks)
#   food_high -> food_low   (after FOOD_AGE_STAGE_TICKS ticks)
#   food_low  -> rotten     (after FOOD_AGE_STAGE_TICKS ticks)
#   rotten    -> gone       (after FOOD_AGE_STAGE_TICKS ticks, removed from world)
# food_starter is EXEMPT — it exists until eaten (its whole purpose is a
# guaranteed early win; making it rot would undermine that).
# rotten_food is a distinct entity type with its own observation channel (8)
# so the network can learn "that triangle is food I should avoid".
# Eating rotten_food applies a negative energy effect (see FOOD_EFFECTS above).
FOOD_AGE_STAGE_TICKS = 5    # ticks each stage lasts before advancing

# Local-view geometry — the source of truth; World.__init__ defaults mirror it.
# The agent sees VISION_RANGE squares straight ahead, VISION_WIDTH columns wide,
# plus BEHIND_ROWS squares behind it. The observation grid World.get_local_view()
# produces has shape (VIEW_H, VIEW_W) = (VISION_RANGE + BEHIND_ROWS, VISION_WIDTH).
# These drive the network's input size (per-cell multi-hot classes + conv), so
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

# Multi-hot channels: one binary channel per cell code 0..9:
#   0=unknown, 1=wall, 2=water, 3=soil, 4=grass,
#   5=food_low, 6=food_high, 7=hazard, 8=rotten_food, 9=other_agent.
# food_starter reuses food_low's code 5 (see World docstring).
# rotten_food gets its OWN channel (8) so the network can distinguish
# "food that will hurt me" from "food that will help me".
# other_agent (9) is a binary channel: 1 wherever another living agent
# occupies a cell inside the vision cone, 0 everywhere else. All agents
# are treated as the same type — the network sees only "someone is there",
# not who or what generation. Behind-agent rows are always 0 (same as other
# channels: unknown territory). Injected by world_tick, not World, since
# World has no concept of "other agents" (see population.py's module doc).
# NOTE: bumping this from 9 to 10 changes OBS_GRID_FLAT / OBS_SIZE /
# INPUT_SIZE / CONV_OUT — all saved weights (best_model.pt) are now
# incompatible and must be retrained from scratch.
NUM_CELL_CLASSES = 10     # len of World cell-code alphabet (0..9)

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

# The observation vector an episode stores in replay is the multi-hot grid +
# body state (conv weights live in the agent and run per forward, not on the
# stored vector). Its length is what run_episode/live_viewer push as x.
OBS_GRID_FLAT   = NUM_CELL_CLASSES * VIEW_H * VIEW_W   # multi-hot, flattened
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
    # Rotten food: eating it costs energy (about 1/3 of food_low's gain).
    # No health change — it's mildly unpleasant, not lethal.
    "rotten_food":  {"energy": -10.0, "health": 0.0},
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
FOOD_BONUS          = {"food_low": 3.00, "food_high": 5.00, "food_starter": 3.00,
                       "rotten_food": -1.50}   # eating rotten hurts
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

# Terminal (once-per-lifetime) reward, paid on the tick the episode ends,
# regardless of cause — previously the ONLY signal tied to "how long did you
# survive" was SURVIVAL_BONUS (0.001/tick) accumulating, which over a typical
# ~150-400 tick lifetime totals only ~0.15-0.4 — negligible next to a single
# FOOD_BONUS (3-5) or HAZARD_PENALTY (1). This makes surviving longer a much
# more explicit, comparable-scale signal instead of an easily-drowned-out one.
#   AGE_BONUS_PER_TICK: paid ONCE at episode end, proportional to the final
#     age reached (not accumulated every tick like SURVIVAL_BONUS) — 0.05 so
#     a lifetime of ~200 ticks earns ~10 (a couple of FOOD_BONUS-eats' worth),
#     and a full ~400-tick run earns ~20.
#   MAX_AGE_SURVIVAL_BONUS: an ADDITIONAL flat bonus, paid only if the episode
#     ended by reaching MAX_AGE with health still > 0 (i.e. old age, not a
#     hazard/starvation death) — a clear "completed a full life" signal,
#     distinct from merely having racked up a lot of ticks before dying.
AGE_BONUS_PER_TICK      = 0.05
MAX_AGE_SURVIVAL_BONUS  = 10.0

# ── Reproduction / ecosystem (2026-09) ───────────────────────────────
# Hybrid nature+nurture (Khanh's choice over pure forward-only evolution):
# individuals keep learning via backprop during their life exactly as before
# (TorchQAgent.learn_windows, completely unchanged), AND ALSO pass down
# (mutated) weights to offspring — both mechanisms run side by side, not one
# replacing the other.
#
# Trigger is fully automatic (no new "reproduce" action / no NUM_ACTIONS
# change — Khanh's choice), so this never touches the network's output layer
# or invalidates existing weights that way: whenever a living individual's
# energy is >= ENERGY_TO_REPRODUCE at the end of a tick, it reproduces if
# there's still room under MAX_POPULATION and a free cell exists next to it
# (see World.random_adjacent_cell). See src/rl/population.py for the actual
# mechanics — kept as a NEW module layered on top of World/Agent/world_tick/
# TorchQAgent/compute_reward without modifying any of them (they turned out
# to already be multi-agent-safe: every call takes agent/brain/state as
# explicit parameters, nothing is stored as a single global "the agent").
ENERGY_TO_REPRODUCE      = 75.0   # of MAX_ENERGY=100 — comfortably fed, not starving
REPRODUCTION_ENERGY_COST = 40.0   # deducted from parent on an actual birth; the
                                  # ONLY throttle on how often one individual can
                                  # reproduce (energy must climb back above the
                                  # threshold again — no separate cooldown timer)
REPRODUCE_BONUS          = 4.0    # reward bonus for the PARENT the tick it
                                  # reproduces (comparable to eating food_high) —
                                  # gives backprop a direct incentive to reach
                                  # reproductive fitness, not just relying on
                                  # cross-generation selection to notice it works
MUTATION_STD             = 0.05   # std of Gaussian noise added to EVERY weight
                                  # of a COPY of the parent's network ("biến dị
                                  # gen") — small relative to typical Conv2d/
                                  # Linear init scales, so children start close
                                  # to parent behavior, not randomized from scratch
MAX_POPULATION            = 12    # hard cap on simultaneous living individuals —
                                  # same purpose as MAX_FOOD_ON_WORLD: N separate
                                  # networks means N forward passes/tick, so this
                                  # also directly bounds per-tick compute cost
INITIAL_POPULATION        = 1     # how many founder individuals start a run
CHILD_INITIAL_EPSILON     = 0.3   # child's exploration starts here, not
                                  # EPSILON_START=1.0 — it inherits reasonable
                                  # starting behavior from its parent's weights,
                                  # so re-exploring from total randomness would
                                  # waste that; matches the "resuming training"
                                  # default (0.3) used elsewhere in the project

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

# ── Ecosystem run configuration (experiments/phase6_ecosystem/run_ecosystem.py) ──
NUM_ECOSYSTEM_TICKS     = 3000   # default length of an ecosystem run
ECOSYSTEM_SNAPSHOT_EVERY = 10    # record 1 GIF frame every N ticks (recording
                                 # every single tick over thousands of ticks
                                 # would make an unreasonably large/slow GIF)
ECOSYSTEM_LOG_EVERY      = 50    # console/CSV progress line frequency
ECOSYSTEM_GIF_FPS        = 10
