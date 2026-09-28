import sys
import copy
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple
import numpy as np
import torch

# Set up paths so imports work smoothly
SIM_DIR   = Path(__file__).resolve().parent
SRC_DIR   = SIM_DIR.parent
RL_DIR    = SRC_DIR / "rl"
WORLD_DIR = SRC_DIR / "world"
BRAIN_DIR = SRC_DIR / "brain"

for d in (SIM_DIR, SRC_DIR, RL_DIR, WORLD_DIR, BRAIN_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from world import World
from population import spawn_founder, ecosystem_step, Individual
from behavior import RuleBasedController
from lstm_q_network import build_lstm_network
from model_io import load_weights_or_fresh
import config

FOOD_DISPLAY_NAMES = {
    "food_low": f"thức ăn nhỏ (+{int(config.FOOD_EFFECTS['food_low']['energy'])} NL)",
    "food_high": f"thức ăn lớn (+{int(config.FOOD_EFFECTS['food_high']['energy'])} NL)",
    "food_starter": "thức ăn khởi đầu",
    "rotten_food": "thức ăn ôi thiu (-NL)",
}


def _find_safe_spawn_cell(world: World,
                          avoid_cells: Optional[Set[Tuple[int, int]]] = None,
                          max_attempts: int = 200) -> Tuple[int, int]:
    """
    Find a random coordinate in the world that is not a wall, not a hazard,
    and not already in avoid_cells (to avoid stacking founders on top of each other).
    """
    hazard_positions = {
        e["pos"] for e in world.entities
        if World.ENTITY_SPECS.get(e["type"], {}).get("category") == "hazard"
    }
    avoid = set(avoid_cells) if avoid_cells else set()

    for _ in range(max_attempts):
        x = int(np.random.randint(0, world.width))
        y = int(np.random.randint(0, world.height))
        pos = (x, y)
        if (world.terrain_at(pos) != World.CELL_WALL
                and pos not in hazard_positions
                and pos not in avoid):
            return pos

    # Fallback if world is very crowded: try any cell not in avoid_cells
    for _ in range(max_attempts):
        x = int(np.random.randint(0, world.width))
        y = int(np.random.randint(0, world.height))
        pos = (x, y)
        if world.terrain_at(pos) != World.CELL_WALL and pos not in avoid:
            return pos

    return world.spawn_point()[0]


class WorldSession:
    """
    Simulation session that owns the world, residents, simulation clock,
    and event log, completely decoupled from any rendering or Pygame viewer.

    Key design principles:
      - Each resident has an independent brain (deep copied parameters) and body.
      - Simulation stepping is decoupled from viewer rendering.
      - Learning is optional (enable_learning=False by default for live observation),
        preventing silent training or drift while watching.
      - Rich event recording captures daily life (eating, hazards, births, deaths).
    """

    def __init__(self,
                 world_size: int = 50,
                 num_food_low: Optional[int] = None,
                 num_food_high: Optional[int] = None,
                 num_hazards: Optional[int] = None,
                 initial_population: int = 4,
                 weights_path: Optional[Path] = None,
                 target_fps: int = 2,
                 enable_learning: bool = False,
                 controller_mode: str = "neural"):
        self.world_size = world_size

        # Auto-scale entity counts based on world area if not explicitly specified
        n_zones = max(1, (world_size // 3) ** 2)
        if num_food_low is None:
            num_food_low = max(1, int(round(n_zones * (5.0 / 9.0))))
        if num_food_high is None:
            num_food_high = max(1, int(round(n_zones * (4.0 / 9.0))))
        if num_hazards is None:
            num_hazards = max(1, int(round(3 * (world_size / 10.0) ** 2)))

        self.num_food_low = num_food_low
        self.num_food_high = num_food_high
        self.num_hazards = num_hazards
        self.initial_population_count = max(1, initial_population)
        self.weights_path = Path(weights_path) if weights_path else None
        if controller_mode not in {"neural", "rules"}:
            raise ValueError("controller_mode must be 'neural' or 'rules'")
        self.controller_mode = controller_mode
        self.target_fps = target_fps
        self.enable_learning = enable_learning

        self.paused = False
        self.tick_count = 0
        self.session_births = 0
        self.session_deaths = 0
        self.extinct = False
        self.recent_events: List[str] = []

        self.world: Optional[World] = None
        self.population: List = []

        self.reset()

    def _record_event(self, text: str, max_history: int = 25):
        self.recent_events.append(text)
        if len(self.recent_events) > max_history:
            self.recent_events.pop(0)

    def _load_initial_net(self):
        """Loads optional template weights if available; falls back to a fresh random brain."""
        if self.weights_path and self.weights_path.exists():
            net, loaded = load_weights_or_fresh(build_lstm_network, self.weights_path, verbose=False)
            if loaded:
                return net, True
        return build_lstm_network(), False

    def reset(self):
        """Creates a fresh world and initializes distinct starting residents."""
        self.world = World(
            width=self.world_size,
            height=self.world_size,
            num_food_low=self.num_food_low,
            num_food_high=self.num_food_high,
            num_hazards=self.num_hazards
        )

        template_net, loaded = self._load_initial_net()
        epsilon = config.LIVE_VIEWER_START_EPSILON

        self.population = []
        occupied_cells: Set[Tuple[int, int]] = set()

        # First founder at the designated world spawn point
        first_pos, first_facing = self.world.spawn_point()
        occupied_cells.add(first_pos)
        founder = spawn_founder(
            self.world,
            epsilon=epsilon,
            generation=0,
            net=copy.deepcopy(template_net),
            position=first_pos,
            facing=first_facing,
            controller=self.controller_mode
        )
        self.population.append(founder)

        # Additional founders scattered safely at UNIQUE cells across the world
        for _ in range(1, self.initial_population_count):
            pos = _find_safe_spawn_cell(self.world, avoid_cells=occupied_cells)
            occupied_cells.add(pos)
            facing = int(np.random.randint(0, 4))
            ind = spawn_founder(
                self.world,
                epsilon=epsilon,
                generation=0,
                net=copy.deepcopy(template_net),
                position=pos,
                facing=facing,
                controller=self.controller_mode
            )
            self.population.append(ind)

        self.tick_count = 0
        self.session_births = 0
        self.session_deaths = 0
        self.extinct = False
        self.recent_events.clear()
        self._record_event(
            f"Thế giới mới ({self.world_size}x{self.world_size}) với {len(self.population)} cư dân độc lập."
        )

    @staticmethod
    def _json_random_state(state):
        """Make Python's nested tuple RNG state JSON-compatible."""
        if isinstance(state, tuple):
            return [WorldSession._json_random_state(item) for item in state]
        return state

    @staticmethod
    def _tuple_random_state(state):
        if isinstance(state, list):
            return tuple(WorldSession._tuple_random_state(item) for item in state)
        return state

    def save(self, path):
        """Save the complete running world to ``path``.

        The JSON file is the world save; resident neural weights and optimizer
        tensors live in a sidecar file next to it. This is deliberately separate
        from ``results/best_model.pt``: observing a world never changes the
        global model checkpoint, and a world can still be inspected with the
        rule controller if its optional sidecar is unavailable.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        brain_path = path.with_suffix(path.suffix + ".brains.pt")

        entity_ages = []
        for index, entity in enumerate(self.world.entities):
            age = self.world.food_ages.get(id(entity))
            if age is not None:
                entity_ages.append({"entity_index": index, "ticks": age[0],
                                    "stage": age[1]})

        brains = {}
        residents = []
        for ind in self.population:
            is_rules = isinstance(ind.brain, RuleBasedController)
            controller = "rules" if is_rules else "neural"
            if not is_rules:
                brains[str(ind.id)] = {
                    "net": {key: value.detach().cpu() for key, value
                            in ind.brain.net.state_dict().items()},
                    "optimizer": ind.brain.optimizer.state_dict(),
                    "learn_calls": ind.brain._learn_calls,
                }
            residents.append({
                "id": ind.id,
                "generation": ind.generation,
                "parent_id": ind.parent_id,
                "total_reward": ind.total_reward,
                "steps": ind.steps,
                "controller": controller,
                "epsilon": getattr(ind.brain.policy, "epsilon", 0.0),
                "position": list(ind.agent.position),
                "facing": ind.agent.facing,
                "energy": ind.agent.energy,
                "health": ind.agent.health,
                "age": ind.agent.age,
                "alive": ind.agent.alive,
                "busy_ticks_remaining": ind.agent.busy_ticks_remaining,
                "event_history": ind.event_history,
                "last_action": ind.last_action,
                "last_position": (list(ind.last_position)
                                  if ind.last_position is not None else None),
                "last_event": ind.last_event,
                "x": ind.x.tolist(),
                "h": ind.h.tolist(),
                "c": ind.c.tolist(),
            })

        world_data = {
            "width": self.world.width,
            "height": self.world.height,
            "vision_range": self.world.vision_range,
            "behind_rows": self.world.behind_rows,
            "vision_width": self.world.vision_width,
            "num_water_patches": self.world.num_water_patches,
            "water_patch_size": self.world.water_patch_size,
            "num_wall_segments": self.world.num_wall_segments,
            "wall_segment_len": self.world.wall_segment_len,
            "soil_fraction": self.world.soil_fraction,
            "entity_counts": self.world.entity_counts,
            "time": self.world.time,
            "terrain": self.world.terrain.tolist(),
            "entities": copy.deepcopy(self.world.entities),
            "seeds": [{"position": list(pos), "ticks": ticks}
                      for pos, ticks in self.world.seeds.items()],
            "food_ages": entity_ages,
            "spawn_point": [list(self.world._spawn_point[0]),
                            self.world._spawn_point[1]],
            "reserved_cells": [list(pos) for pos in self.world._reserved_cells],
        }
        payload = {
            "version": 1,
            "world": world_data,
            "session": {
                "world_size": self.world_size,
                "num_food_low": self.num_food_low,
                "num_food_high": self.num_food_high,
                "num_hazards": self.num_hazards,
                "initial_population_count": self.initial_population_count,
                "target_fps": self.target_fps,
                "enable_learning": self.enable_learning,
                "controller_mode": self.controller_mode,
                "paused": self.paused,
                "tick_count": self.tick_count,
                "session_births": self.session_births,
                "session_deaths": self.session_deaths,
                "extinct": self.extinct,
                "recent_events": self.recent_events,
            },
            "residents": residents,
            "rng": {
                "python": self._json_random_state(random.getstate()),
                "numpy": [np.random.get_state()[0],
                          np.random.get_state()[1].tolist(),
                          np.random.get_state()[2], np.random.get_state()[3],
                          np.random.get_state()[4]],
                "torch": torch.get_rng_state().tolist(),
            },
        }

        brain_tmp = brain_path.with_suffix(brain_path.suffix + ".tmp")
        json_tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"brains": brains}, brain_tmp)
        with json_tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(brain_tmp, brain_path)
        os.replace(json_tmp, path)
        return path

    @classmethod
    def load(cls, path, enable_learning=None):
        """Restore a world saved by :meth:`save`.

        If the optional brain sidecar is missing or corrupt, neural residents
        are safely replaced with rule residents so the world remains viewable.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != 1:
            raise ValueError("Unsupported world save version")

        session_data = payload["session"]
        world_data = payload["world"]
        session = cls.__new__(cls)
        session.world_size = session_data["world_size"]
        session.num_food_low = session_data["num_food_low"]
        session.num_food_high = session_data["num_food_high"]
        session.num_hazards = session_data["num_hazards"]
        session.initial_population_count = session_data["initial_population_count"]
        session.target_fps = session_data["target_fps"]
        session.enable_learning = (session_data["enable_learning"]
                                   if enable_learning is None else enable_learning)
        session.controller_mode = session_data["controller_mode"]
        session.paused = session_data["paused"]
        session.tick_count = session_data["tick_count"]
        session.session_births = session_data["session_births"]
        session.session_deaths = session_data["session_deaths"]
        session.extinct = session_data["extinct"]
        session.recent_events = list(session_data["recent_events"])
        session.weights_path = None

        session.world = World(
            width=world_data["width"], height=world_data["height"],
            vision_range=world_data["vision_range"],
            behind_rows=world_data["behind_rows"],
            vision_width=world_data["vision_width"],
            num_food_low=world_data["entity_counts"].get("food_low", 0),
            num_food_high=world_data["entity_counts"].get("food_high", 0),
            num_hazards=world_data["entity_counts"].get("hazard", 0),
            num_food_starter=world_data["entity_counts"].get("food_starter", 0),
            num_water_patches=world_data["num_water_patches"],
            water_patch_size=world_data["water_patch_size"],
            num_wall_segments=world_data["num_wall_segments"],
            wall_segment_len=world_data["wall_segment_len"],
            soil_fraction=world_data["soil_fraction"],
        )
        session.world.time = world_data["time"]
        session.world.terrain = np.asarray(world_data["terrain"], dtype=int)
        session.world.entities = [
            {"type": entity["type"], "pos": tuple(entity["pos"])}
            for entity in world_data["entities"]
        ]
        session.world.seeds = {
            tuple(item["position"]): item["ticks"]
            for item in world_data["seeds"]
        }
        session.world.food_ages = {}
        for item in world_data["food_ages"]:
            entity = session.world.entities[item["entity_index"]]
            session.world.food_ages[id(entity)] = (item["ticks"], item["stage"])
        spawn = world_data["spawn_point"]
        session.world._spawn_point = (tuple(spawn[0]), spawn[1])
        session.world._reserved_cells = {
            tuple(position) for position in world_data["reserved_cells"]
        }

        brains = {}
        brain_path = path.with_suffix(path.suffix + ".brains.pt")
        if brain_path.exists():
            try:
                brains = torch.load(brain_path, map_location="cpu", weights_only=False)["brains"]
            except Exception:
                brains = {}

        session.population = []
        for record in payload["residents"]:
            brain_record = brains.get(str(record["id"]))
            controller = record["controller"]
            net = None
            if controller == "neural" and brain_record is not None:
                net = build_lstm_network()
                net.load_state_dict(brain_record["net"])
            else:
                controller = "rules"
            ind = spawn_founder(
                session.world, epsilon=record["epsilon"],
                generation=record["generation"], parent_id=record["parent_id"],
                net=net, position=tuple(record["position"]),
                facing=record["facing"], controller=controller,
            )
            ind.id = record["id"]
            ind.total_reward = record["total_reward"]
            ind.steps = record["steps"]
            ind.agent.energy = record["energy"]
            ind.agent.health = record["health"]
            ind.agent.age = record["age"]
            ind.agent.alive = record["alive"]
            ind.agent.busy_ticks_remaining = record["busy_ticks_remaining"]
            ind.event_history = list(record.get("event_history", []))[-20:]
            ind.last_action = record.get("last_action", 0)
            last_position = record.get("last_position")
            ind.last_position = (tuple(last_position)
                                 if last_position is not None else None)
            ind.last_event = record.get("last_event")
            ind.x = np.asarray(record["x"], dtype=np.float32)
            ind.h = np.asarray(record["h"], dtype=np.float32)
            ind.c = np.asarray(record["c"], dtype=np.float32)
            if not isinstance(ind.brain, RuleBasedController) and brain_record is not None:
                ind.brain.optimizer.load_state_dict(brain_record["optimizer"])
                ind.brain._learn_calls = brain_record["learn_calls"]
            session.population.append(ind)

        Individual._next_id = max(
            [record["id"] + 1 for record in payload["residents"]] + [Individual._next_id]
        )
        rng = payload["rng"]
        random.setstate(cls._tuple_random_state(rng["python"]))
        numpy_state = rng["numpy"]
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                             numpy_state[2], numpy_state[3], numpy_state[4]))
        torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))
        return session

    def _on_resident_event(self, ind, event: dict):
        """Callback to capture frequent daily life events (eating, hazards, exhaustion)."""
        category = event.get("category")
        event_type = event.get("type")
        event_tick = self.tick_count + 1

        if category == "food":
            food_label = FOOD_DISPLAY_NAMES.get(event_type, event_type)
            text = f"Tick {event_tick}: Cư dân #{ind.id} ăn {food_label}"
            self._record_event(text)
        elif category == "hazard":
            text = f"Tick {event_tick}: Cư dân #{ind.id} dẫm phải bẫy nguy hiểm!"
            self._record_event(text)
        elif category == "starvation":
            text = f"Tick {event_tick}: Cư dân #{ind.id} bị đói / kiệt sức!"
            self._record_event(text)
        else:
            return

        ind.event_history.append(text)
        ind.event_history = ind.event_history[-20:]

    def step(self):
        """
        Advances the world by exactly one tick.
        Does NOT silently overwrite any model checkpoint or training state.
        """
        if self.extinct or self.world is None:
            return

        births, deaths = ecosystem_step(
            self.world,
            self.population,
            auto_reseed=False,
            child_epsilon=config.LIVE_VIEWER_START_EPSILON,
            tick_count=self.tick_count + 1,
            enable_learning=self.enable_learning,
            on_event=self._on_resident_event
        )

        self.session_births += len(births)
        self.session_deaths += len(deaths)
        self.tick_count += 1

        for child in births:
            parent_info = f"#{child.parent_id}" if child.parent_id is not None else "?"
            text = (f"Tick {self.tick_count}: Cư dân #{child.id} chào đời "
                    f"(mẹ: {parent_info}, thế hệ {child.generation})")
            self._record_event(text)
            child.event_history.append(text)
            child.event_history = child.event_history[-20:]

        for d in deaths:
            if d.agent.age >= d.agent.max_age:
                cause = "tuổi già"
            elif d.agent.energy <= 0:
                cause = "kiệt sức vì đói"
            else:
                cause = "mất máu"
            text = (f"Tick {self.tick_count}: Cư dân #{d.id} qua đời vì {cause} "
                    f"(thọ {d.agent.age} tick)")
            self._record_event(text)
            d.event_history.append(text)
            d.event_history = d.event_history[-20:]

        if not self.population:
            self.extinct = True
            self._record_event(f"Tick {self.tick_count}: Quần thể đã tuyệt diệt. Nhấn R để tái tạo thế giới.")

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def adjust_speed(self, delta: int, min_fps: int = 1, max_fps: int = 60):
        self.target_fps = max(min_fps, min(max_fps, self.target_fps + delta))
        return self.target_fps

    def get_oldest_resident(self):
        """Returns the oldest living resident, or None if extinct."""
        if not self.population:
            return None
        return max(self.population, key=lambda ind: ind.agent.age)
