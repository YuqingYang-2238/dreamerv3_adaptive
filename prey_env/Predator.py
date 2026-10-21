from cellworld import Location
from myPaths import *
from random import choice
from Agent import *
from geometry import Geometry, heading, ROBOT

CLEARANCE = max(abs(y) for _, y in ROBOT)   # body half-width kept off the obstacles when choosing waypoints


def neighbours(world: World) -> dict:
    """{cell id: cells sharing an edge with it} on the hexagonal grid (6 for interior cells)."""
    cells = list(world.cells)
    spacing = min(cells[0].location.dist(c.location) for c in cells[1:])
    return {c.id: [d for d in cells if d is not c and c.location.dist(d.location) <= 1.2 * spacing]
            for c in cells}


_DECOUPLED_CELLS = {}  # cached per occlusion layout: the world object is rebuilt every episode


def decoupled_cells(world: World) -> tuple:
    """(spawn candidates, destination candidates) of the paper's Decoupled robot:
    free cells not adjacent to an obstacle; destinations also not adjacent to a wall."""
    key = tuple(c.id for c in world.cells.occluded_cells())
    if key not in _DECOUPLED_CELLS:
        near = neighbours(world)
        clear = [c.id for c in world.cells.free_cells() if not any(d.occluded for d in near[c.id])]
        _DECOUPLED_CELLS[key] = (clear, [i for i in clear if len(near[i]) == 6])
    clear, interior = _DECOUPLED_CELLS[key]
    return [world.cells[i] for i in clear], [world.cells[i] for i in interior]


class Predator(Agent):
    """Default scripted predator, the "Decoupled" robot of the mouse experiments: it moves
    continuously to randomly selected free cells not adjacent to walls or obstacles, with no
    coupling to the prey's position. Captures happen only when the prey comes close by chance."""

    def __init__(self,
                 pworld: World,
                 ppath_builder: Paths_builder,
                 pvisibility: Geometry,
                 pP_value: float = .1,
                 pI_value: float = .1,
                 pD_value: float = .1,
                 pmax_speed: float = .4,
                 pmax_turning_speed: float = .2):
        self.world = pworld
        self.paths = myPaths(ppath_builder, pworld)
        self.visibility = pvisibility
        self.destination = None
        self.destination_cell = None
        self.path = Cell_group()
        self.P_value = pP_value
        self.I_value = pI_value
        self.D_value = pD_value
        self.last_theta = None
        self.accum_theta_error = 0
        self.max_speed = pmax_speed
        self.max_turning_speed = pmax_turning_speed
        self.destination_cells = decoupled_cells(pworld)[1]

    def _follow_path(self, predator_location: Location, destination_cell: Cell) -> None:
        """Plans a path to destination_cell and aims at the furthest path cell reachable in a
        straight line with the body's clearance (so the body does not cut corners)."""
        self.destination_cell = destination_cell
        predator_cell = self.world.cells[self.world.cells.find(predator_location)]
        self.path = self.paths.get_path(predator_cell, destination_cell)
        self.destination = self.path[min(1, len(self.path) - 1)].location
        for cd in self.path:
            if self.visibility.is_clear(predator_location, cd.location, CLEARANCE):
                self.destination = cd.location

    def __update_destination__(self, predator_location: Location):
        if self.destination_cell and \
                predator_location.dist(self.destination_cell.location) < self.world.implementation.cell_transformation.size / 2:
            self.destination_cell = None
        if self.destination_cell is None:
            self.destination_cell = choice(self.destination_cells)
        self._follow_path(predator_location, self.destination_cell)

    @staticmethod
    def normalized_error(theta_error: float):
        pi_err = math.pi * theta_error / 2
        return 1 / (pi_err * pi_err + 1)

    def _steer(self, predator: AgentData) -> AgentAction:
        """PID heading control toward self.destination; speed drops with heading error."""
        if self.destination is None:
            return AgentAction(0, 0)
        desired_theta = heading(predator.location, self.destination)
        theta_error, direction = angle_difference(predator.theta, desired_theta)
        dist_error = predator.location.dist(self.destination)
        self.accum_theta_error += theta_error
        turning_speed_P = theta_error * self.P_value
        turning_speed_D = 0
        if self.last_theta is not None:
            turning_speed_D = (self.last_theta - predator.theta) * self.D_value
        turning_speed_I = self.accum_theta_error * self.I_value
        turning_speed = turning_speed_P - turning_speed_D + turning_speed_I
        if turning_speed > self.max_turning_speed:
            turning_speed = self.max_turning_speed
        turning_speed = turning_speed * (-direction)
        speed = self.normalized_error(theta_error) * (1 + dist_error)
        if speed > self.max_speed:
            speed = self.max_speed
        self.last_theta = predator.theta
        return AgentAction(speed, turning_speed)

    def get_action(self, observation: dict) -> AgentAction:
        predator = observation["predator"]
        self.__update_destination__(predator.location)
        return self._steer(predator)
