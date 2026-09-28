# Instructions for coding agents

Before proposing or changing code, read `README.md`. It is the source of truth for the product goal and the refactor direction. `progress.md` is a historical engineering log; older training-centered plans and comments there do not override the current goal in README.

## Product goal

Virtual Lifetime is a small living world that the user opens to observe for enjoyment. The world and its residents are the product. Training metrics, reward maximization, and neural-network sophistication are implementation details, not success criteria.

## Priorities

- Make the live world engaging and understandable from a fresh start, without requiring pretraining.
- Treat residents as persistent inhabitants with identity, needs, life history, and observable events.
- Keep simulation rules independent of the Pygame viewer; the viewer displays and controls the simulation.
- Persist the world and its residents independently from any optional controller/model checkpoint.
- Keep behavior controllers replaceable. A simple rule-based controller is acceptable and useful as a baseline.
- Prefer small vertical changes that keep the viewer runnable and make a visible improvement.
- Explain code changes in terms of what changes for someone watching the world.

Do not assume that the goal is to improve RL performance, maximize fitness/reward, increase training throughput, or add model complexity. Only retain or extend learning when it improves the observable world experience. Do not make the live viewer silently overwrite a globally shared "best model" as a side effect of being used; model persistence must be an explicit, separate concern.

## Refactor sequence

Follow the staged roadmap in README: (1) make the viewer the primary entry point and separate simulation stepping from rendering, (2) save and resume the world, (3) make resident life and events legible, (4) isolate replaceable behavior controllers and make learning optional, then (5) add depth to the world in small observable increments.

Do not perform a broad rewrite just to match the conceptual folder layout. Preserve useful existing mechanics and refactor one runnable slice at a time.
