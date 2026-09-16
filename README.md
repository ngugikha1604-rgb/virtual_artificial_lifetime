# Virtual Lifetime

> A simulation of a small 2D creature that lives, learns, and develops inside a persistent
> virtual world — built to be watched and understood, not just measured.

## 1. Why this project exists

**This project's purpose changed.** It began as a from-scratch learning exercise (implement
neural networks, backprop, LSTM, Conv2D by hand in NumPy to understand every mechanism). That
phase is over and its code has been retired. The project now uses PyTorch for the neural network
plumbing, and the goal is different:

> Build a small persistent 2D world containing a creature, and watch how its **behavior** —
> not its weights — changes as it lives through a simulated lifetime and learns from experience.

This can serve two purposes, not mutually exclusive:

- A **visualization / observation tool**: something to run and watch, to build intuition for how
  a simple learning creature explores, finds food, avoids hazards, and how that behavior evolves
  across a lifetime and across generations of training.
- The seed of a **small game**: a persistent creature/world that a person could eventually
  interact with, not only observe.

Understanding the underlying ML mechanics is no longer the point of this project — using a
standard framework (PyTorch) to get a working, trainable brain quickly is the right tradeoff now.
What still matters is that the **world, the agent's body, and its behavior over time** stay
something we can inspect, reason about, and explain.

---

## 2. Core philosophy

The project favors:

**observable, explainable behavior > architectural novelty > raw performance**

A creature whose behavior changes in ways we can watch, describe, and reason about ("it used to
wander randomly, now it beelines for food and skirts hazards") is more valuable to this project
than a marginally higher benchmark score in a black box.

The high-level loop the project is built around:

```text
Simple neural network (PyTorch)
        ↓
Learning (online, during the lifetime)
        ↓
Agent (body + brain)
        ↓
Environment (2D grid world)
        ↓
Experience
        ↓
Changing weights
        ↓
Lifetime
        ↓
Observable, evolving behavior
```

This is **inspired by biological development**, but it is not intended to be a faithful
simulation of a human brain, and it is not trying to be a general "AI agent" wrapped around an
LLM.

---

## 3. Long-term goal

Create a small artificial creature that exists inside a persistent simulated 2D world and goes
through something resembling a lifetime, in a way that is fun/interesting to actually watch.

```text
                         VIRTUAL WORLD
                              │
                ┌─────────────┴─────────────┐
                │                           │
           Environment                    Agent
                │                           │
        ┌───────┼────────┐          ┌───────┴────────┐
        │       │        │          │                │
       Time   Objects   Events     Body             Brain
                                      │                │
                                ┌─────┼─────┐          │
                                │     │     │          │
                              Energy Age   Alive    Weights (torch)
                                                     │
                                                     ↓
                                                Learning
```

The agent:

1. is created/born with a minimally initialized brain;
2. perceives part of its environment (a local view around itself);
3. chooses actions;
4. receives consequences from the environment (food, hazards, energy cost);
5. learns from experience, online, during its lifetime;
6. modifies its own neural-network weights while alive;
7. accumulates internal state/memory (LSTM hidden state) across its lifetime;
8. ages;
9. eventually dies (health or age runs out) or reaches the end of an episode.

The interesting question driving this project:

> **How does behavior visibly change as the same creature experiences a sequence of events
> throughout a simulated lifetime, and as training accumulates across many lifetimes?**

Concretely, this is the kind of thing worth watching for: does the creature learn to cut through
a costly patch of terrain (e.g. water, which is slower/more tiring to cross) when it's a genuine
shortcut, but go around it otherwise? That's a real behavioral trade-off, not a metric — exactly
the sort of thing this project exists to make visible.

Weight values, gradients, and loss curves are implementation detail in service of that question —
not the thing being studied for its own sake anymore.

---

## 4. Framework use

PyTorch is used for the neural network (Conv2d + LSTMCell + Linear, trained with
`torch.optim.Adam` and autograd). This is a deliberate change from the project's original
NumPy-from-scratch phase.

What is kept from the original constraint:

> **The brain is trained from scratch on this project's own experience — no pretrained weights,
> no pretrained model is loaded as the agent's brain.**

What is dropped:

> ~~Implement the neural network math (forward/backward/optimizer) ourselves.~~ — no longer a
> goal. Using PyTorch's autograd and layers is the right tool now that the point is the world and
> the behavior, not re-deriving backprop.

The NumPy-from-scratch implementation (Dense layers, hand-written LSTM cell, hand-written Conv2D
via im2col, a manual gradient checker) existed and worked during the earlier phase of this
project; see `progress.md` for that history. It has since been removed from the codebase in favor
of the PyTorch version — it is not being kept around as dead/legacy code to maintain.

---

## 5. Working with an AI assistant

Claude (or another AI assistant) acts as **implementer and technical reviewer** for this project:
reading the codebase, proposing designs, prototyping/testing changes in a sandbox before touching
the real project files, and writing to the project on request. The person reviews and confirms
before changes land on disk.

This is different from the project's original "AI as teacher only, human writes every line"
workflow from the NumPy phase — that workflow made sense when the goal was to personally
internalize every mechanism. Now that the goal is a working, observable simulation, delegating
implementation work to an assistant (with review) is the more effective way to make progress.

What's still expected of an assistant working on this project:

- Keep world/agent/brain concerns cleanly separated (see architecture below) so behavior stays
  inspectable and a change to one layer doesn't silently break another.
- Verify non-trivial refactors behave identically to what they replaced (fixed seeds, before/after
  comparison) before treating them as done.
- Prefer code that's easy to reason about over cleverness, since "can I explain why the creature
  is behaving this way" is still a project value even though "did I derive the math myself" is
  not.

---

## 6. Current architecture (high level — see `progress.md` for full detail)

```text
World (2D grid: terrain layer [wall/water/soil/grass] + entities [food_low/food_high/food_starter/rotten_food/hazard])
   ↓ local view (NumPy)
Observation encoder (NumPy: multi-hot grid + body state)
   ↓
ConvLSTMDQN (PyTorch nn.Module: Conv2d+ReLU → LSTMCell → Linear)
   ↓
Epsilon-greedy policy → action
   ↓
World.step() → new state + event (ate food / hit hazard / blocked by wall / nothing)
   ↓
Agent body update (energy/health/age; crossing water costs extra time+energy)
   ↓
Episode replay buffer → windowed truncated-BPTT training (PyTorch autograd + Adam,
                         with a periodically-synced target network)
```

Entry points:

- `experiments/run_experiment.py` — multi-agent ecosystem training: runs the population simulation,
  checkpoints the best individual to `results/best_model.pt`, renders `ecosystem_lifetime.gif`,
  and logs `ecosystem_log.csv`, all in one command.
- `src/rl/live_viewer.py` — interactive Pygame 2D viewer; can run with or without online training.
- `src/rl/run_episode.py` — the training loop for a single lifetime, importable/reusable.

All entry points load from and save to the same single `results/best_model.pt` — there is
deliberately only ever one "the model" file, so training-by-command and watching-interactively are
always looking at/building on the same brain, never two silently diverging copies.

All tunable constants (world size, terrain, entity counts, reward shaping, network sizes, training
hyperparameters) live in `src/config.py`.

---

## 7. What this project is NOT

This project is not intended to:

- claim to be AGI;
- reproduce the human brain;
- create a competitive LLM or general-purpose agent;
- be a rigorous RL research benchmark;
- immediately solve language understanding.

It also is **no longer** primarily:

- a from-scratch deep-learning tutorial project (that was its original purpose; see `progress.md`
  for that history — the code from that phase has been retired).

It is an experimental, watchable simulation of a simple learning creature. If it also ends up fun
to watch or play with as a small game, that's the point, not a bonus.

---

## 8. Definition of success

The project is successful if someone (including the person building it) can watch a creature's
lifetime — live in the viewer or as a recorded GIF — and say something like:

> "Early in training it wandered and bumped into hazards; by the end it clearly beelines for food
> and swerves around hazards it can see. I can point at *when* and *how* that changed."

That is the core experiment now: not "I understand every gradient," but "I can observe and explain
how this creature's behavior developed."

```text
small persistent world
    ↓
creature with a trainable brain
    ↓
lifetime of experience
    ↓
observable behavior change
    ↓
repeat across lifetimes / world variations
    ↓
a simulation that's actually worth watching
```
