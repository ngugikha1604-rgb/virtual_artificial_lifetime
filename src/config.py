# Shared tuning values for the world, residents, viewer, and optional brain.

# ── World layout ─────────────────────────────────────────────────────────────
WORLD_SIZE      = 10      # square grid (world default in __init__ used elsewhere)
NUM_FOOD_LOW    = 5
NUM_FOOD_HIGH   = 4
NUM_HAZARDS     = 3
# Starter food guarantees an immediately reachable first meal.
NUM_FOOD_STARTER = 1
MAX_AGE         = 400     # ticks a lifetime may last at most

# ── Terrain ───────────────────────────────────────────────────────────────────
# Terrain is persistent background; food and hazards are separate entities.
# Soil/grass are walkable, water slows movement, and walls block movement.
NUM_WATER_PATCHES = 2      # count of water patches carved per world
WATER_PATCH_SIZE  = 2      # each patch is WATER_PATCH_SIZE x WATER_PATCH_SIZE
NUM_WALL_SEGMENTS = 2      # count of wall line-segments carved per world
WALL_SEGMENT_LEN  = 3      # length of each wall segment (straight, random orientation)
SOIL_FRACTION     = 0.30   # fraction of the default ground fill that's soil (rest grass)
WATER_DURATION_MULTIPLIER = 2   # x ACTION_DURATION for a move landing on water

# ── Food growth ───────────────────────────────────────────────────────────────
# Eaten food is removed; new food grows through seed -> maturation.
#
# Seeds mature into food after a randomized 15-30 tick countdown.
FOOD_SEED_BASE_RATE   = {"soil": 0.003, "grass": 0.001}
FOOD_SEED_SPREAD_BONUS = {"soil": 0.005, "grass": 0.012}
FOOD_SEED_MATURATION_MIN = 15   # ticks a seed takes to become food (fastest)
FOOD_SEED_MATURATION_MAX = 30   # ticks a seed takes to become food (slowest)
FOOD_GROWTH_TYPE_SPLIT = {"food_low": 0.7, "food_high": 0.3}
# Maximum mature food + pending seeds. Growth pauses at this capacity.
MAX_FOOD_ON_WORLD = 14

# ── Food aging / expiration ────────────────────────────────────────────────────
# food_low -> food_high -> food_low -> rotten -> gone.
# Starter food is permanent; rotten food remains visible and harmful to eat.
FOOD_AGE_STAGE_TICKS = 25   # 4 stages; roughly 100 ticks fresh-to-gone

# Local-view geometry — the source of truth; World.__init__ defaults mirror it.
# The agent sees VISION_RANGE squares straight ahead, VISION_WIDTH columns wide,
# plus BEHIND_ROWS squares behind it. The observation grid World.get_local_view()
# produces has shape (VIEW_H, VIEW_W) = (VISION_RANGE + BEHIND_ROWS, VISION_WIDTH).
# These drive the network's input size (per-cell multi-hot classes + conv), so
# changing VISION_RANGE here re-sizes the Conv-LSTM and invalidates any saved
# weights (model_io refuses the shape mismatch and falls back to a fresh net).
VISION_RANGE   = 4        # how many cells ahead the agent can see (was 2)
VISION_WIDTH   = 4        # column width of the vision cone
BEHIND_ROWS    = 0        # rows behind the agent — 0: agent only sees forward
                          # (was 2 before 2026-09: those 2 rows were always
                          # CELL_UNKNOWN=0, never containing real information,
                          # but still wasted 80 input dimensions and made the
                          # conv + obs vector larger than needed. Set to 0 to
                          # match the intended "human-like forward-only vision"
                          # design; all code in world.py/world_tick.py/
                          # lstm_q_network.py reads world.behind_rows at
                          # runtime so no other file needs changing. NOTE:
                          # changing this invalidates best_model.pt — the
                          # saved weights have a different shape and model_io
                          # will fall back to a fresh random brain.)
VIEW_H         = VISION_RANGE + BEHIND_ROWS   # observation grid rows  (= 4)
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
BATCH_SIZE               = 8
BATCH_SIZE_ECOSYSTEM     = 4       # windows sampled per learn step for an ecosystem individual
BURN_IN_N                = 16      # hard cap on max ticks burned-in before a window
WINDOW_N                 = 16
MIN_EPISODES             = 10
LEARN_EVERY              = 4             # learn every N ticks of the live episode

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

# How often (in ecosystem ticks, NOT learn-call count) the ecosystem prints a
# one-line learn-stats summary to stdout — loss, Q-range, gradient norm, number
# of learn calls in the interval.  Set to 0 to silence all learn logging.
# At LEARN_EVERY=4 one ecosystem tick fires a learn call every 4 ticks, so
# LOG_LEARN_EVERY=200 means a log line every ~50 learn calls — visible in a
# terminal at 2 fps but not flooding faster than you can read it.
LOG_LEARN_EVERY = 200

EPSILON_START   = 0.5
# A permanent 15% exploit noise (the old default) meant the agent never behaved
# near-greedily late in training — one in ~7 actions stayed random forever.
# 0.03 is a standard exploration floor: still enough to escape local optima /
# loops, low enough that the final policy is mostly exploitative after epsilon
# decays.
EPSILON_MIN     = 0.03
EPSILON_DECAY   = 0.9995  # per-lifetime decay; hits eps_min late in a 4000-lt run

# Within-lifetime decay (2026-09) — a SEPARATE, much faster decay rate than
# EPSILON_DECAY above, applied every TICK during an individual's own life
# (population.py's ecosystem_step), not once between lifetimes. Discovered
# during a stability audit: EPSILON_DECAY/policy.decay() was only ever called
# once, right as an individual dies — a no-op for that individual's OWN
# behavior (it's discarded immediately after) — so no individual, founder or
# child, ever actually became less random over the course of its own life;
# every ecosystem individual acted at a constant exploration rate for its
# entire lifetime. Khanh wants an individual to progressively rely less on
# random exploration and more on what it's actually learned as its own life
# goes on, not just across separate lifetimes.
# Tuned (2026-09, second pass) so EPSILON_START=0.5 decays to EPSILON_MIN=0.03
# by roughly tick 100 (0.97**100 ~= 0.0476, close enough) — the FIRST version
# of this (rate 0.985) only reached the floor by ~tick 186, but a separate
# comment elsewhere in this file (TARGET_SYNC_EVERY) estimates a typical
# lifetime at only ~150-250 ticks, meaning many individuals were dying before
# ever reaching a real "exploit" phase — the whole point of this mechanism.
# 0.97 leaves at least half of even a short (~200-tick) life mostly greedy.
# Applies to BOTH live_viewer.py AND run_experiment.py (population.py is
# shared code, not viewer-only) — unlike LIVE_VIEWER_START_EPSILON below,
# which Khanh wants viewer-only.
INLIFE_EPSILON_DECAY = 0.97

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
#  - SURVIVAL_BONUS (2026-09: changed from a flat per-tick constant to a
#    continuous energy/health-based formula — was 0.001 paid identically every
#    tick regardless of body state, which gave zero training signal for
#    actually managing energy/health well vs. barely staying alive, and, once
#    the ecosystem reliably produces individuals that live to MAX_AGE, left
#    check_and_save_best's fitness (total_reward) collapsing to near-identical
#    values across most of the population — the only large reward component
#    left to discriminate them was the once-per-lifetime terminal bonus below,
#    which is IDENTICAL (30.0) for every individual that survives the full
#    lifespan. Now: reward_per_tick = SURVIVAL_BONUS * (energy/MAX_ENERGY) *
#    (health/MAX_HEALTH) — see compute_reward() in run_episode.py. PRODUCT (not
#    average) of the two fractions so an individual needs BOTH energy AND
#    health healthy to earn the full bonus; letting either drop crashes this
#    term toward 0 well before starvation/hazard damage actually kicks in, so
#    it doubles as an early continuous warning gradient. Raised back to 0.01
#    (was cut to 0.001 when it was flat noise — now that it's a genuine,
#    state-dependent signal and the main differentiator among full-lifespan
#    individuals, drowning it out would defeat the point of the change).
#  - SHAPE_WEIGHT_* raised 0.10 -> 1.0 (2026-09, second pass): a stability
#    audit found a "walks straight into a wall and just sits there" policy
#    could still score 100+ purely from FOOD_BONUS (incidental eating) +
#    REPRODUCE_BONUS + the terminal age bonus below — i.e. the ONLY
#    component that actually rewards DIRECTED, purposeful foraging (moving
#    toward food you can see / away from hazard you can see) was so small
#    relative to those "passive" rewards that being smart barely paid better
#    than being lucky. At WORLD_SIZE=10 (max_dist=18, what training actually
#    runs at), 1.0 means a single well-directed step now contributes ~0.056 —
#    still small per tick (so it can't override an actual FOOD_BONUS/eat
#    event, still "guides" rather than dominates) but a sustained efficient
#    approach now sums to something comparable to an eat bonus, instead of
#    being lost in noise. NOTE: this is normalized by max_dist = width+height-2,
#    which is world-size-dependent — live_viewer.py's much bigger 50x50
#    viewer world (max_dist=98) dilutes this ~5.4x more than the 10x10
#    training world; tuned for the world training ACTUALLY happens in, not
#    the viewer's, since the viewer is for watching the trained brain, not
#    itself the thing being optimized for.
#  - FOOD_BONUS / HAZARD_PENALTY unchanged: these are event-level (eat 3/5,
#    hazard -1) and dominate whenever an interaction happens, as intended.
#  - REPRODUCE_BONUS lowered 4.0 -> 1.0 (2026-09, second pass): reproduction
#    is fully automatic (fires whenever energy >= ENERGY_TO_REPRODUCE, no
#    "decision" the network makes), so rewarding it at a FOOD_HIGH-comparable
#    magnitude meant an individual that merely survives long enough to get
#    well-fed collects a large, skill-independent reward simply for existing
#    near abundant food — exactly the "lucky, not smart" inflation the audit
#    found. Lowered to a token nudge (comparable to HAZARD_PENALTY) — still
#    gives backprop SOME direct signal toward reproductive fitness (the
#    original intent), just no longer large enough to dominate total_reward
#    on its own.
SURVIVAL_BONUS      = 0.01
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
SHAPE_WEIGHT_FOOD   = 1.0
SHAPE_WEIGHT_HAZARD = 1.0

# Terminal (once-per-lifetime) reward, paid on the tick the episode ends,
# regardless of cause — previously the ONLY signal tied to "how long did you
# survive" was SURVIVAL_BONUS (0.001/tick) accumulating, which over a typical
# ~150-400 tick lifetime totals only ~0.15-0.4 — negligible next to a single
# FOOD_BONUS (3-5) or HAZARD_PENALTY (1). This makes surviving longer a much
# more explicit, comparable-scale signal instead of an easily-drowned-out one.
#   AGE_BONUS_PER_TICK: paid ONCE at episode end, proportional to the final
#     age reached (not accumulated every tick like SURVIVAL_BONUS) — 0.025
#     (2026-09, halved from 0.05: a full ~400-tick life used to earn a flat
#     20 here regardless of HOW that life was spent, dwarfing the shaping
#     signal above even after raising it; a full life is now worth ~10, still
#     a meaningful "you survived" signal, just no longer big enough to make
#     mere longevity out-earn directed foraging skill on its own).
#   MAX_AGE_SURVIVAL_BONUS: an ADDITIONAL flat bonus, paid only if the episode
#     ended by reaching MAX_AGE with health still > 0 (i.e. old age, not a
#     hazard/starvation death) — a clear "completed a full life" signal,
#     distinct from merely having racked up a lot of ticks before dying.
#     Halved 10.0 -> 5.0 alongside AGE_BONUS_PER_TICK for the same reason.
AGE_BONUS_PER_TICK      = 0.025
MAX_AGE_SURVIVAL_BONUS  = 5.0

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
REPRODUCE_BONUS          = 1.0    # reward bonus for the PARENT the tick it
                                  # reproduces (2026-09: lowered from 4.0 —
                                  # see the "Reward shaping" section above for
                                  # why; still a token direct incentive for
                                  # backprop, just no longer large enough to
                                  # dominate total_reward on its own since
                                  # reproduction itself requires no skill,
                                  # only enough energy)
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
# CHILD_INITIAL_EPSILON removed (2026-09 audit): was superseded when Khanh
# decided children should start at EPSILON_START=1.0 instead (full re-explore
# per generation — see progress.md item 10), leaving this constant unused by
# any code (reproduce() in population.py passes config.EPSILON_START, not
# this). Already flagged as safe-to-delete in progress.md's own roadmap.

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
# is re-saved (see pretrain_single_agent.py / training_state.py) — there is only
# ONE model file (shared by every entry point, see run_experiment.py's module
# docstring for why), so this protects against losing an interrupted run's
# progress but, unlike an earlier design with a separate best-only file, does
# NOT protect against a later save overwriting it with weights that turn out
# worse.
SAVE_EVERY          = 200   # LIFETIMES between periodic checkpoint saves — used by
                            # pretrain_single_agent.py's single-persistent-brain
                            # loop, where "lifetime" is the natural unit of
                            # progress (one full episode of the one brain being
                            # trained).
# The ecosystem (run_experiment.py / live_viewer.py) has no single equivalent
# "lifetime" to count between saves — many individuals live and die
# concurrently — so its periodic safety-net save is measured in ECOSYSTEM
# TICKS instead. 2026-09: this used to just reuse SAVE_EVERY (both happened to
# be 200, but one counted lifetimes and the other ticks — same number,
# different units, easy to misread while editing either loop). Split into its
# own constant so "200" always means one specific thing wherever it's read.
ECOSYSTEM_SAVE_EVERY_TICKS = 200   # ticks between periodic checkpoint saves
                                   # in ecosystem_step-driven loops
BEST_METRIC_WINDOW  = 50    # lifetimes averaged for the "is this a new best" check

# ── Experiment run configuration ──────────────────────────────────────────────
NUM_LIFETIMES  = 1000     # how many lifetimes to train across
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

# ── live_viewer.py-only overrides (2026-09) ──────────────────────────────────
# Deliberately NOT used by run_experiment.py / population.py's defaults —
# Khanh wants the actual training run's exploration schedule (EPSILON_START,
# whatever training_state.json has saved) left completely untouched; this is
# purely so the interactive viewer doesn't spend most of a session showing
# individuals that are almost entirely random (training_state.json's saved
# epsilon can be anywhere up to 1.0 depending how early training still is —
# see EPSILON_START/INLIFE_EPSILON_DECAY above for why that's not itself a
# bug). live_viewer.py passes this explicitly to spawn_founder() (in place of
# the loaded training_state.json epsilon) and to ecosystem_step()'s
# child_epsilon= (in place of reproduce()'s EPSILON_START default) — nothing
# in population.py defaults to this value, so run_experiment.py is unaffected.
LIVE_VIEWER_START_EPSILON = 0.4   # combined with INLIFE_EPSILON_DECAY=0.97,
                                  # reaches EPSILON_MIN by ~tick 87 of an
                                  # individual's life
