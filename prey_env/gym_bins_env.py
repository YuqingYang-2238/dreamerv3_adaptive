from Predator import Predator
from Agent import *
from Model import Model
from myPaths import *
from cellworld import *
import random
from gym import Env
from gym import spaces
import numpy as np


class Environment(Env):
    """
    Prey navigation task with a scripted predator.

    Timing follows the Mice/SAC setup: one RL step lasts `time_step` simulated
    seconds and is executed as `time_step * freq` physics ticks of 1/freq s.
    Default: 0.25 s per step = 10 ticks of 0.025 s.

    Reward is sparse, as in Mice/SAC: `goal_reward` (+1) when the goal is reached,
    `capture_penalty` (-1) per capture, 0 otherwise.

    Capture ("puff") never ends the episode. Each capture starts a
    `capture_cooldown` window during which the prey cannot be captured again.
    The episode ends only when the goal is reached (terminal) or `max_step`
    is exceeded (truncation).

    World selection: pass `world_name` (e.g. "21_05", "clump01_05", "00_00") to
    train on one fixed world as Mice does. Leave it None to sample a random
    world from family `e` every episode (train: 00..10, test: 11..19).
    """

    def __init__(self, e: int = 5,
                 world_name: str = None,
                 freq: int = 40,
                 time_step: float = 0.25,
                 has_predator=False,
                 real_time: bool = False,
                 prey_agent: Agent = None,
                 max_step: int = 300,
                 prey_max_speed: float = 0.5,
                 predator_prey_forward_speed_ratio: float = 0.3,
                 capture_cooldown: float = 0.5,
                 goal_reward: float = 1.0,
                 capture_penalty: float = -1.0,
                 goal_threshold: float = 0.1,
                 capture_threshold: float = 0.1,
                 env_type: str = "train",
                 env_random: bool = False):
        self.fixed_world_name = world_name
        self.e = e
        self.env_type = env_type
        world_name = self._pick_world_name()
        self.freq = freq                      # physics ticks per simulated second
        self.time_step = time_step            # simulated seconds per RL step
        self.ticks_per_step = max(1, int(round(time_step * freq)))
        self.real_time = real_time
        self.prey_agent = prey_agent
        self.env_random = env_random
        self.world = World.get_from_parameters_names("hexagonal", "canonical", world_name)
        self.model = Model(pworld=self.world, freq=self.freq, real_time=self.real_time)
        self.goal_location = Location(1, .5)
        self.start_location = Location(.05, .5)   # inside the arena, as in Mice
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        self.action_space = spaces.Discrete(100)
        self.has_predator = has_predator
        self.max_step = max_step
        self.current_step = 0
        self.episode_reward_history = []
        self.current_episode_reward = 0
        self.predator = None
        # Speeds: prey max speed in habitat lengths / s at action value +-1.
        # Predator max speed is a ratio of the prey max speed (as in Mice/SAC).
        self.prey_max_speed = prey_max_speed
        self.predator_prey_forward_speed_ratio = predator_prey_forward_speed_ratio
        self.predator_max_speed = prey_max_speed * predator_prey_forward_speed_ratio
        # Capture bookkeeping (non-terminal captures with cooldown).
        self.capture_cooldown_ticks = int(round(capture_cooldown * freq))
        self.goal_reward = goal_reward
        self.capture_penalty = capture_penalty
        self.capture_count = 0
        self.capture_cooldown_remaining = 0
        # Absolute radii in habitat lengths (Mice: goal 0.1, puff 0.1).
        self.goal_threshold = goal_threshold
        self.capture_threshold = capture_threshold
        self.goal_area = self.model.display.circle(location=self.goal_location,
                                                   color="g",
                                                   alpha=.5,
                                                   radius=self.goal_threshold)

    def _pick_world_name(self) -> str:
        if self.fixed_world_name is not None:
            return self.fixed_world_name
        if self.env_type == "train":
            return "%02i_%02i" % (random.randint(0, 10), self.e)
        return "%02i_%02i" % (random.randint(11, 19), self.e)

    def is_goal_reached(self, prey_location: Location):
        return prey_location.dist(self.goal_location) <= self.goal_threshold

    def is_prey_captured(self, prey_location: Location, predator_location: Location):
        return prey_location.dist(predator_location) <= self.capture_threshold

    def get_observation(self) -> tuple:
        """
        Returns (prey_location, prey_theta, goal_location,
                 predator_location or None, predator_theta or None,
                 goal_reached, closest_occlusions).
        Predator fields are None when there is no predator or it is not visible.
        """
        o = self.model.get_observation("prey")
        prey = o["prey"]
        goal_reached = self.is_goal_reached(prey.location)
        if goal_reached:
            self.complete = True

        occlusions = Location_list([(c.location.dist(prey.location), c.location) for c in self.world.cells if c.occluded])
        occlusions.sort(key=lambda a: a[0])

        def dif_dir(a, b):
            diff, direc = angle_difference(a, b)
            return diff * direc

        max_number_of_occlusions_in_observation = 3
        closest_occlusions = [(c[0], dif_dir(prey.location.atan(c[1]), prey.theta))
                              for c in occlusions[:min(len(occlusions), max_number_of_occlusions_in_observation)]]
        # Worlds with fewer than 3 occlusions (e.g. open arena "00_00"): pad with "far away" sentinels.
        while len(closest_occlusions) < max_number_of_occlusions_in_observation:
            closest_occlusions.append((2.0, 0.0))

        predator = o["predator"] if (self.has_predator and "predator" in o) else None
        if predator is not None:
            return prey.location, prey.theta, self.goal_location, \
                   predator.location, predator.theta, goal_reached, closest_occlusions
        return prey.location, prey.theta, self.goal_location, None, None, goal_reached, closest_occlusions

    def _build_observation(self, speed: float, turning: float):
        """Builds the 14-float observation vector. Returns (obs, prey_location, goal_reached)."""
        prey_location, prey_theta, goal_location, pred_location, pred_theta, \
            goal_reached, closest_occlusions = self.get_observation()
        closest_distance, closest_angle = closest_occlusions[0]
        se_closest_distance, se_closest_angle = closest_occlusions[1]
        th_closest_distance, th_closest_angle = closest_occlusions[2]
        if pred_location is not None:
            pred_x, pred_y, pred_th = pred_location.x, pred_location.y, pred_theta
        else:
            pred_x, pred_y, pred_th = -1.0, -1.0, 0.0
        obs = np.array(
            [prey_location.x, prey_location.y, prey_theta,
             speed, turning,
             pred_x, pred_y, pred_th,
             closest_distance, closest_angle,
             se_closest_distance, se_closest_angle,
             th_closest_distance, th_closest_angle], dtype=np.float32)
        return obs, prey_location, goal_reached

    def is_complete(self):
        return self.complete

    def run(self) -> None:
        self.start()
        self.model.run()

    def start(self) -> None:
        self.complete = False
        self.model.set_agent_position("prey", self.start_location, math.pi / 2)
        if self.has_predator:
            predator_location = random.choice(self.spawn_locations)
            predator_theta = math.pi * 2 * random.random()
            self.model.set_agent_position("predator", predator_location, predator_theta)

    def map_discrete_to_continuous(self, discrete_val, n_bins=10):
        # Convert the single discrete value into two discrete values
        # assuming a 10x10 grid
        row = discrete_val // n_bins  # Integer division to get the row
        col = discrete_val % n_bins  # Modulus to get the column
        # Now map each discrete value to the continuous range [-1, 1]
        bin_width = 2.0 / (n_bins - 1)
        continuous_row = -1 + row * bin_width
        continuous_col = -1 + col * bin_width
        noise_level, action_shape = 0, 2
        rand_noise = np.random.randn(action_shape) * noise_level
        continuous_row += rand_noise[0]
        continuous_col += rand_noise[1]
        return continuous_row, continuous_col

    def set_action(self, speed: float, turning: float) -> None:
        action = AgentAction(speed, turning)
        self.model.set_agent_action("prey", action)

    def _check_capture(self) -> bool:
        """One capture check at the current tick. Honors the cooldown. Returns True on a new capture."""
        if not self.has_predator:
            return False
        if self.capture_cooldown_remaining > 0:
            self.capture_cooldown_remaining -= 1
            return False
        o = self.model.get_observation("prey")
        predator = o.get("predator")
        if predator is None:
            return False
        if self.is_prey_captured(o["prey"].location, predator.location):
            self.capture_count += 1
            self.capture_cooldown_remaining = self.capture_cooldown_ticks
            return True
        return False

    def step(self, action):
        done, truncated = False, False
        speed, turning = self.map_discrete_to_continuous(action)
        self.set_action(speed * self.prey_max_speed, turning)

        # One RL step = ticks_per_step physics ticks. The action is held for all of them.
        captures_this_step = 0
        for _ in range(self.ticks_per_step):
            self.model.step()
            if self._check_capture():
                captures_this_step += 1
            if self.is_goal_reached(self.model.agents_data["prey"].location):
                break

        obs, prey_location, goal_reached = self._build_observation(speed, turning)

        # Sparse reward (Mice custom_reward): +goal_reward at the goal, capture_penalty per capture.
        reward = 0.0
        if goal_reached:
            reward += self.goal_reward
            done = True
        reward += self.capture_penalty * captures_this_step

        self.current_step += 1
        if self.current_step > self.max_step:
            truncated = True

        info = {"is success": done,
                "survived": done and self.capture_count == 0,
                "captures": self.capture_count,
                "captured_this_step": captures_this_step,
                "is truncated": truncated,
                # Only the goal is a true terminal state; timeouts should bootstrap.
                "is_terminal": done}

        self.current_episode_reward += reward
        if done or truncated:
            self.episode_reward_history.append(self.current_episode_reward)
            self.current_episode_reward = 0
        done = done or truncated
        return obs, reward, done, info

    def reset(self):
        import matplotlib.pyplot as plt
        plt.close('all')
        world_name = self._pick_world_name()
        self.world_name = world_name
        self.world = World.get_from_parameters_names("hexagonal", "canonical", world_name)
        self.model = Model(pworld=self.world, freq=self.freq, real_time=self.real_time)
        self.goal_location = Location(1, .5)
        self.start_location = Location(.05, .5)
        self.model.add_agent("prey",
                             self.prey_agent,
                             Location(0, 0), 0, "b",
                             pauto_update=self.prey_agent is not None)
        self.goal_area = self.model.display.circle(location=self.goal_location,
                                                   color="g",
                                                   alpha=.5,
                                                   radius=self.goal_threshold)
        self.capture_count = 0
        self.capture_cooldown_remaining = 0
        if self.env_random:
            random_number = random.random()
            self.has_predator = random_number > 0.5
        if self.has_predator:
            paths_builder = Paths_builder.get_from_name("hexagonal", world_name)
            self.predator = Predator(self.world,
                                     ppath_builder=paths_builder,
                                     pvisibility=self.model.visibility,
                                     pP_value=2,
                                     pI_value=0,
                                     pD_value=0,
                                     pmax_speed=self.predator_max_speed,
                                     pmax_turning_speed=math.pi)

            self.spawn_locations = Location_list()
            for c in self.world.cells.free_cells():
                if not self.model.visibility.is_visible(self.start_location, c.location):
                    self.spawn_locations.append(c.location)
            if len(self.spawn_locations) == 0:
                # Open arenas (e.g. "00_00") have no cell hidden from the start:
                # fall back to free cells at least 0.5 habitat lengths from the prey.
                for c in self.world.cells.free_cells():
                    if c.location.dist(self.start_location) >= 0.5:
                        self.spawn_locations.append(c.location)
            self.model.add_agent("predator", self.predator, Location(0, 0), 0, "r")
            self.predator_destination = self.model.display.circle(location=Location(0, 0),
                                                                  color="b",
                                                                  alpha=.5,
                                                                  radius=0.01)
            self.predator_destination_cell = self.model.display.circle(location=Location(0, 0),
                                                                       color="g",
                                                                       alpha=.5,
                                                                       radius=0.02)
            self.predator_capture_area = self.model.display.circle(location=Location(0, 0),
                                                                   color="r",
                                                                   alpha=.5,
                                                                   radius=self.capture_threshold)
        # start has to be here, after all the setup()
        self.start()
        obs, _, _ = self._build_observation(0.0, 0.0)
        self.current_step = 1
        self.stop()
        return obs

    def stop(self) -> None:
        self.model.stop()

    def show(self) -> None:
        if self.predator:
            if self.predator.destination:
                self.predator_destination.set(center=(self.predator.destination.x, self.predator.destination.y), color="b")
            if self.predator.destination_cell:
                self.predator_destination_cell.set(center=(self.predator.destination_cell.location.x, self.predator.destination_cell.location.y), color="g")
            self.predator_capture_area.set(center=(self.model.agents_data["predator"].location.x, self.model.agents_data["predator"].location.y), color="r")
        self.model.show()

    def seed(self, seed=123):
        self.np_random = np.random.default_rng(seed)
