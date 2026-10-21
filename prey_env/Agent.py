from cellworld import *
from shapely.geometry import Polygon
from geometry import Body


class AgentAction(JsonObject):
    def __init__(self, speed: float = 0, turning_speed: float = 0):
        self.speed = speed
        self.turning_speed = turning_speed*10


class AgentData(JsonObject):

    def __init__(self,
                 plocation: Location,
                 ptheta: float,
                 pspeed: float,
                 pturning_speed: float,
                 pbody: Body,
                 pauto_update=True):
        self.body = pbody
        self.set_pose(plocation, ptheta)
        self.speed = pspeed
        self.turning_speed = pturning_speed
        self.auto_update = pauto_update

    def set_pose(self, location: Location, theta: float, polygon: Polygon = None):
        """Pose plus the body in the world frame (polygon for collision, points for line of sight)."""
        self.location = location
        self.theta = theta
        self.body_points = self.body.points(location.x, location.y, theta)
        self.body_polygon = polygon or Polygon(self.body_points)
        self.los_points = self.body.points(location.x, location.y, theta, self.body.los_local)


class Agent:

    def get_action(self, observation: dict) -> AgentAction:
        return AgentAction(0, 0)
