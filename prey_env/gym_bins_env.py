import collections
import math
import random

import numpy as np
from cellworld import World, Location, Location_list, Cell_group, Paths_builder, angle_difference
from gym import Env, spaces

from Agent import AgentAction
from AmbushPredator import AmbushPredator, zone_cells, attack_radius, HISTORY_TRIALS, DEFAULT_THETA
from Model import Model
from Predator import Predator, decoupled_cells
from geometry import heading


class Environment(Env):
    """
    Prey navigation task with a scripted predator (Mice-aligned).

    One RL step lasts `time_step` s, run as `time_step * freq` physics ticks. Default
    0.25 s = 10 ticks of 0.025 s. Action: 10x10 grid of (speed in [0, 1], turning in
    [-1, 1]); speed 0 is a pause. Reward: `goal_reward` (+1) at the goal, `capture_penalty`
    (-1) per capture, `step_penalty` per step (0 in Mice). Captures never end the episode;
    each starts a `capture_cooldown`. The episode ends at the goal or after `max_step`.

    Predator strategy (`predator_strategy`):
      "default": decoupled robot wandering between random cells, no coupling to the prey.
      "none":    no predator.
      "ambush":  waits at one of five fixed cells, pursues the prey inside its 3-cell region,
                 returns. The cell is sampled every reset with weight 1 + number of the last
                 HISTORY_TRIALS episodes in which the prey approached it (window not checkpointed).
    """

    def __init__(self,
                 world_name: str = "21_05",
                 freq: int = 40,
                 time_step: float = 0.25,
                 has_predator: bool = True,
                 max_step: int = 300,
                 prey_max_speed: float = 0.5,
                 predator_prey_forward_speed_ratio: float = 0.3,
                 capture_cooldown: float = 0.5,
                 goal_reward: float = 1.0,
                 capture_penalty: float = -1.0,
                 step_penalty: float = 0.0,
                 goal_threshold: float = 0.1,
                 capture_threshold: float = 0.1,
                 predator_strategy: str = "default",
                 predator_surge_speed: float = 1.0):
        if predator_strategy not in ("default", "ambush", "none"):
            raise ValueError(f"unknown predator_strategy {predator_strategy!r}; use 'default', 'ambush' or 'none'")
        self.world_name = world_name
        self.world = World.get_from_parameters_names("hexagonal", "canonical", world_name)
        self.paths_builder = Paths_builder.get_from_name("hexagonal", world_name)
        self.freq = freq
        self.time_step = time_step
        self.ticks_per_step = max(1, int(round(time_step * freq)))
        self.model = Model(self.world, freq=freq)
        self.goal_location = Location(1, .5)
        self.start_location = Location(.05, .5)
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        self.action_space = spaces.Discrete(100)
        self.max_step = max_step
        self.current_step = 0
        self.tick_callback = None   # called after every physics tick (video recording)
        self.prey_max_speed = prey_max_speed
        self.predator_max_speed = prey_max_speed * predator_prey_forward_speed_ratio
        self.capture_cooldown_ticks = int(round(capture_cooldown * freq))
        self.goal_reward = goal_reward
        self.capture_penalty = capture_penalty
        self.step_penalty = step_penalty
        self.goal_threshold = goal_threshold
        self.capture_threshold = capture_threshold
        self.capture_count = 0
        self.capture_cooldown_remaining = 0
        self.predator_strategy = predator_strategy
        self.predator_surge_speed = predator_surge_speed
        self.has_predator = has_predator and predator_strategy != "none"
        self.predator = None
        # Ambush bookkeeping (empty for the other strategies).
        self.ambush_zones = Cell_group()
        self.attack_radius = 0.0
        self.zone_history = collections.deque(maxlen=HISTORY_TRIALS)   # per episode: prey entered zone k?
        self.zone_entered = []
        self.ambush_zone = -1
        self.ambush_zone_weights = []
        if predator_strategy == "ambush":
            self.ambush_zones = zone_cells(self.world, world_name)
            self.attack_radius = attack_radius(self.world)
        # Decoupled robot spawns: hidden from the prey entrance, away from obstacles.
        self.spawn_locations = Location_list([c.location for c in decoupled_cells(self.world)[0]
                                              if not self.model.visibility.is_visible(self.start_location, c.location)])

    def is_goal_reached(self, prey_location: Location):
        return prey_location.dist(self.goal_location) <= self.goal_threshold

    def is_prey_captured(self, prey_location: Location, predator_location: Location):
        return prey_location.dist(predator_location) <= self.capture_threshold

    def _ambush_surges(self) -> int:
        return self.predator.surges if isinstance(self.predator, AmbushPredator) else 0

    def _sample_ambush_zone(self) -> int:
        """Weight 1 + number of remembered episodes in which the prey approached the cell."""
        self.ambush_zone_weights = [1 + sum(entered[k] for entered in self.zone_history)
                                    for k in range(len(self.ambush_zones))]
        return random.choices(range(len(self.ambush_zones)), weights=self.ambush_zone_weights)[0]

    def _track_zone_entry(self) -> None:
        if not len(self.ambush_zones):
            return
        prey_location = self.model.agents_data["prey"].location
        for k, c in enumerate(self.ambush_zones):
            if not self.zone_entered[k] and prey_location.dist(c.location) <= self.attack_radius:
                self.zone_entered[k] = True

    def get_observation(self) -> tuple:
        """(prey_location, prey_theta, predator_location or None, predator_theta or None,
        goal_reached, [(distance, bearing)] of the 3 closest occlusions)."""
        o = self.model.get_observation("prey")
        prey = o["prey"]
        occlusions = sorted(((c.location.dist(prey.location), c.location) for c in self.world.cells if c.occluded),
                            key=lambda t: t[0])
        closest = [(d, math.prod(angle_difference(heading(prey.location, loc), prey.theta)))
                   for d, loc in occlusions[:3]]
        closest += [(2.0, 0.0)] * (3 - len(closest))   # open worlds: "far away" sentinels
        predator = o.get("predator") if self.has_predator else None
        if predator is None:
            return prey.location, prey.theta, None, None, self.is_goal_reached(prey.location), closest
        return prey.location, prey.theta, predator.location, predator.theta, self.is_goal_reached(prey.location), closest

    def _build_observation(self, speed: float, turning: float):
        """The 14-float observation vector. Returns (obs, goal_reached)."""
        prey_location, prey_theta, pred_location, pred_theta, goal_reached, closest = self.get_observation()
        pred = (pred_location.x, pred_location.y, pred_theta) if pred_location is not None else (-1.0, -1.0, 0.0)
        obs = np.array([prey_location.x, prey_location.y, prey_theta, speed, turning, *pred,
                        *closest[0], *closest[1], *closest[2]], dtype=np.float32)
        return obs, goal_reached

    def start(self) -> None:
        self.model.set_agent_position("prey", self.start_location, 0.0)   # facing the goal (+x)
        if self.has_predator:
            if self.predator_strategy == "ambush":
                location, theta = self.ambush_zones[self.ambush_zone].location, DEFAULT_THETA
            else:
                location, theta = random.choice(self.spawn_locations), math.pi * 2 * random.random()
            self.model.set_agent_position("predator", location, theta)
        for name, a in self.model.agents_data.items():
            assert self.model.is_valid_pose(a.body, a.location, a.body_polygon), f"{name} starts in an invalid pose"

    def map_discrete_to_continuous(self, discrete_val, n_bins=10):
        """10x10 grid -> (speed in [0, 1], turning in [-1, 1]). Forward only, as the Mice prey;
        speed 0 is the pause action."""
        row, col = divmod(discrete_val, n_bins)
        return row / (n_bins - 1), -1 + col * 2.0 / (n_bins - 1)

    def _check_capture(self) -> bool:
        """Mice puff rule at this tick: distance <= capture_threshold and mutual line of sight,
        outside the cooldown. Returns True on a new capture."""
        if not self.has_predator:
            return False
        if self.capture_cooldown_remaining > 0:
            self.capture_cooldown_remaining -= 1
            return False
        prey, predator = self.model.agents_data["prey"], self.model.agents_data["predator"]
        if not self.is_prey_captured(prey.location, predator.location):
            return False
        if self.model.sees("prey", "predator") and self.model.sees("predator", "prey"):
            self.capture_count += 1
            self.capture_cooldown_remaining = self.capture_cooldown_ticks
            return True
        return False

    def step(self, action):
        speed, turning = self.map_discrete_to_continuous(action)
        self.model.set_agent_action("prey", AgentAction(speed * self.prey_max_speed, turning))

        # One RL step = ticks_per_step physics ticks; the action is held for all of them.
        captures_this_step = 0
        surges_before = self._ambush_surges()
        for _ in range(self.ticks_per_step):
            self.model.step()
            captures_this_step += self._check_capture()
            self._track_zone_entry()
            if self.tick_callback:
                self.tick_callback()
            if self.is_goal_reached(self.model.agents_data["prey"].location):
                break

        obs, goal_reached = self._build_observation(speed, turning)
        reward = self.step_penalty + self.goal_reward * goal_reached + self.capture_penalty * captures_this_step
        self.current_step += 1
        truncated = self.current_step > self.max_step
        info = {"is success": goal_reached,
                "survived": goal_reached and self.capture_count == 0,
                "captures": self.capture_count,
                "captured_this_step": captures_this_step,
                "is truncated": truncated,
                "is_terminal": goal_reached,   # only the goal is terminal; timeouts bootstrap
                "ambush_zone": self.ambush_zone,
                "surges_this_step": self._ambush_surges() - surges_before}
        if (goal_reached or truncated) and len(self.ambush_zones):
            self.zone_history.append(list(self.zone_entered))
        return obs, reward, goal_reached or truncated, info

    def reset(self):
        self.model = Model(self.world, freq=self.freq)
        self.model.add_agent("prey", None, Location(0, 0), 0, pauto_update=False)
        self.capture_count = 0
        self.capture_cooldown_remaining = 0
        self.zone_entered = [False] * len(self.ambush_zones)
        self.ambush_zone = -1
        self.predator = None
        if self.has_predator:
            gains = dict(pP_value=2, pI_value=0, pD_value=0,
                         pmax_speed=self.predator_max_speed, pmax_turning_speed=math.pi)
            if self.predator_strategy == "ambush":
                self.ambush_zone = self._sample_ambush_zone()
                self.predator = AmbushPredator(self.world, self.paths_builder, self.model.visibility,
                                               pzone_cell=self.world.cells[self.ambush_zones[self.ambush_zone].id],
                                               pprey_locator=lambda: self.model.agents_data["prey"].location,
                                               psurge_speed=self.predator_surge_speed, **gains)
            else:
                self.predator = Predator(self.world, self.paths_builder, self.model.visibility, **gains)
            self.model.add_agent("predator", self.predator, Location(0, 0), 0)
        self.start()
        obs, _ = self._build_observation(0.0, 0.0)
        self.current_step = 1
        return obs

    def seed(self, seed=123):
        self.np_random = np.random.default_rng(seed)
