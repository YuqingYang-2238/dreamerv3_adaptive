import math

from cellworld import World, Location, normalize

from Agent import Agent, AgentData, AgentAction
from geometry import Geometry, Body, MOUSE, ROBOT


class Model:
    """Agents with bodies in one world: movement with collision, line of sight (after cellworld_game)."""

    def __init__(self, pworld: World, freq: int = 100):
        self.world = pworld
        self.agents = dict()
        self.agents_data = dict()
        self.interval = 1 / freq
        self.geometry = Geometry(self.world)
        self.visibility = self.geometry   # is_visible(src, dst), as cellworld's Location_visibility
        self.blocked = dict()             # per agent: ticks where the full move was rejected

    def add_agent(self, pagent_name: str, pagent: Agent, plocation: Location, ptheta: float,
                  pauto_update: bool = True, pbody: Body = None):
        # Mice bodies. Predator: no occlusion collision, as in Mice. Prey: collides as a disc of
        # half its length (the polygon rule wedges a forward-only mouse against walls).
        if pbody is None:
            pbody = Body(ROBOT, collision=False) if pagent_name == "predator" else Body(MOUSE, collision_radius=.015)
        self.agents[pagent_name] = pagent
        self.agents_data[pagent_name] = AgentData(plocation, ptheta, 0, 0, pbody, pauto_update=pauto_update)
        self.blocked[pagent_name] = 0

    def set_agent_position(self, pagent_name: str, plocation: Location, ptheta: float):
        self.agents_data[pagent_name].set_pose(plocation, ptheta)

    def set_agent_action(self, agent_name: str, action: AgentAction):
        self.agents_data[agent_name].speed = action.speed
        self.agents_data[agent_name].turning_speed = action.turning_speed

    def is_valid_pose(self, body: Body, location: Location, polygon) -> bool:
        if body.collision_radius:
            return self.geometry.is_valid_disc((location.x, location.y), body.collision_radius)
        return self.geometry.is_valid(polygon, body.collision)

    def __move_agent__(self, agent_name):
        agent = self.agents_data[agent_name]
        rotation = agent.turning_speed * self.interval
        distance = agent.speed * self.interval
        # Mice Model.step: full move, else rotation only, else translation only, else stay.
        for i, (rot, dist) in enumerate(((rotation, distance), (rotation, 0), (0, distance))):
            theta = normalize(agent.theta + rot)
            location = Location(agent.location.x + math.cos(theta) * dist,
                                agent.location.y + math.sin(theta) * dist)
            polygon = agent.body.polygon(location.x, location.y, theta)
            if self.is_valid_pose(agent.body, location, polygon):
                agent.set_pose(location, theta, polygon)
                break
            if i == 0:
                self.blocked[agent_name] += 1

    def step(self):
        for agent_name, agent in self.agents.items():
            if self.agents_data[agent_name].auto_update:
                self.set_agent_action(agent_name, agent.get_action(self.get_observation(agent_name)))
        for agent_name in self.agents:
            self.__move_agent__(agent_name)

    def sees(self, agent_name: str, other_name: str) -> bool:
        """Mice line of sight: any part of the other body is visible from this agent's centre."""
        src = self.agents_data[agent_name].location
        return self.geometry.body_visible((src.x, src.y), self.agents_data[other_name].los_points)

    def get_observation(self, agent_name: str) -> dict:
        """{name: AgentData or None if not in line of sight}."""
        return {name: data if name == agent_name or self.sees(agent_name, name) else None
                for name, data in self.agents_data.items()}
