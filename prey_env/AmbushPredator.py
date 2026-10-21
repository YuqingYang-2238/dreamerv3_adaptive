"""Ambush predator strategy, as in the mouse experiments.

The robot starts the trial at the selected ambush cell in a default orientation
and does not move. Its actions are coupled to the prey's true position (the real
robot was driven by the overhead tracker): when the prey enters the cell's ambush
region (ATTACK_CELLS), it pursues the prey for as long as the prey stays inside the
region, then navigates back to the ambush cell and waits again.

Which ambush cell is used in a given episode is sampled by gym_bins_env.Environment
with probability weighted by how often the prey approached each cell within
ATTACK_CELLS on recent episodes; that bookkeeping lives there because the
predator object is rebuilt every episode.
"""
from cellworld import *
from Predator import Predator
from Agent import AgentAction, AgentData
from geometry import Geometry

# Waiting cells per world, designed by hand (scores/plot_ambush_zones.py draws them).
# Each is a free cell hidden from the prey entry (0.05, 0.5), directly behind an
# occluded cell on the prey's approach (-x) side, and the five are spread over the arena.
#   zone  cell  (x, y)          sheltering obstacle(s), hex coords           place
#   0     103   (0.359, 0.581)  (-8, 2) left, (-7, 3) upper-left             top-left
#   1     251   (0.688, 0.662)  (6, 4) left                                  top-right
#   2     144   (0.453, 0.500)  (-4, 0) left, blocks the straight route      centre
#   3     152   (0.477, 0.297)  (-3, -5) left, (-2, -6), (0, -4)             bottom-left
#   4     258   (0.711, 0.378)  (7, -3) left, (8, -2) upper-left             bottom-right
AMBUSH_ZONES = {
    "21_05": [103, 251, 144, 152, 258],
}
ATTACK_CELLS = 3      # ambush region radius in cell sizes (38 cm = 3 cells in the paper); also the approach threshold of the selection bias
HISTORY_TRIALS = 10   # episodes remembered for the bias
DEFAULT_THETA = math.pi   # start orientation: facing the entrance, -x (the paper's Fig. 2C default)


def attack_radius(world: World) -> float:
    return ATTACK_CELLS * world.implementation.cell_transformation.size


def zone_cells(world: World, world_name: str) -> Cell_group:
    """The waiting cells designed for world_name, in zone order."""
    if world_name not in AMBUSH_ZONES:
        raise ValueError(f"no ambush zones designed for world {world_name!r}; known worlds: {sorted(AMBUSH_ZONES)}")
    cells = Cell_group()
    for cell_id in AMBUSH_ZONES[world_name]:
        cells.append(world.cells[cell_id])
    return cells


class AmbushPredator(Predator):
    """Ambush robot. States: WAIT (static at the ambush cell), ATTACK (pursue the
    prey inside the ambush region), RETURN (walk back)."""

    WAIT, ATTACK, RETURN = "WAIT", "ATTACK", "RETURN"

    def __init__(self,
                 pworld: World,
                 ppath_builder: Paths_builder,
                 pvisibility: Geometry,
                 pzone_cell: Cell,
                 pprey_locator,
                 pP_value: float = .1,
                 pI_value: float = .1,
                 pD_value: float = .1,
                 pmax_speed: float = .4,
                 pmax_turning_speed: float = .2,
                 psurge_speed: float = 1.0):
        Predator.__init__(self, pworld, ppath_builder, pvisibility,
                          pP_value, pI_value, pD_value, pmax_speed, pmax_turning_speed)
        self.surge_speed = psurge_speed   # attack speed as a multiple of pmax_speed
        self.zone_cell = pzone_cell
        self.prey_locator = pprey_locator   # returns the prey's true Location
        self.attack_radius = attack_radius(pworld)
        self.arrival_radius = pworld.implementation.cell_transformation.size / 2
        self.base_speed = pmax_speed
        self.state = self.WAIT
        self.surges = 0
        self.destination = pzone_cell.location
        self.destination_cell = pzone_cell

    def get_action(self, observation: dict) -> AgentAction:
        predator = observation["predator"]
        location = predator.location
        prey_location = self.prey_locator()
        prey_distance = prey_location.dist(self.zone_cell.location)

        if self.state != self.ATTACK and prey_distance <= self.attack_radius:
            self.state = self.ATTACK
            self.surges += 1
        elif self.state == self.ATTACK and prey_distance > self.attack_radius:
            self.state = self.RETURN

        self.max_speed = self.base_speed * (self.surge_speed if self.state == self.ATTACK else 1.0)
        if self.state == self.ATTACK:
            target = self.world.cells[self.world.cells.find(prey_location)]
        else:
            target = self.zone_cell

        if location.dist(target.location) < self.arrival_radius:
            if self.state == self.RETURN:
                self.state = self.WAIT
            self.destination = target.location
            self.destination_cell = target
            return AgentAction(0, 0)
        self._follow_path(location, target)
        return self._steer(predator)
