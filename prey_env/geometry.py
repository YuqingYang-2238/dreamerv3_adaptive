"""Bodies, collision and line of sight (shapely), after cellworld_game (the Mice engine)."""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from cellworld import Location, World

# Body polygons from cellworld_game mouse.py / robot.py; +x is forward.
MOUSE = [(.015, 0), (0, .005), (-.015, 0), (0, -.005)]
ROBOT = [(.02, .013), (-.02, .013), (-.02, -.013), (.02, -.013), (.025, -.01), (.025, .01)]


def heading(src: Location, dst: Location) -> float:
    """Bearing src -> dst in radians, 0 = +x, counter-clockwise."""
    return math.atan2(dst.y - src.y, dst.x - src.x)


def hexagon(center: Location, radius: float, rotation: float) -> Polygon:
    """Same vertices as cellworld.Polygon(center, 6, radius, rotation)."""
    th = math.radians(rotation)
    return Polygon([(center.x + radius * math.sin(th + i * math.pi / 3),
                     center.y + radius * math.cos(th + i * math.pi / 3)) for i in range(6)])


class Body:
    """Agent body in its own frame. `collision`: test against occlusions (Mice: off for the robot).
    `collision_radius`: collide as a disc of that radius instead of the polygon (free rotation)."""

    def __init__(self, vertices, collision: bool = True, collision_radius: float = None):
        self.local = np.array(vertices, dtype=float)
        self.collision = collision
        self.collision_radius = collision_radius
        mids = (self.local + np.roll(self.local, -1, axis=0)) / 2
        self.los_local = np.vstack([[0, 0], self.local, mids])   # points to test for visibility

    def points(self, x: float, y: float, theta: float, local=None) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        return (self.local if local is None else local) @ np.array([[c, s], [-s, c]]) + (x, y)

    def polygon(self, x: float, y: float, theta: float) -> Polygon:
        return Polygon(self.points(x, y, theta))


class Geometry:
    """Arena, occlusions and visibility of one world."""

    def __init__(self, world: World):
        imp = world.implementation
        self.arena = hexagon(imp.space.center, imp.space.transformation.size / 2,
                             imp.space.transformation.rotation)
        rotation = imp.space.transformation.rotation + imp.cell_transformation.rotation  # the drawn tiling
        self.occlusion_cells = [hexagon(c.location, imp.cell_transformation.size / 2, rotation)
                                for c in world.cells.occluded_cells()]
        self.occlusions = unary_union(self.occlusion_cells)
        self.arena_edge = self.arena.exterior
        for g in (self.arena, self.arena_edge, self.occlusions):
            shapely.prepare(g)
        # Wall segments (occlusion and arena edges), shape (n, 2, 2), for the visibility polygon.
        rings = shapely.get_parts(unary_union([self.occlusions.boundary, self.arena.boundary]))
        coords = [shapely.get_coordinates(r) for r in rings]
        self.walls = np.concatenate([np.stack([c[:-1], c[1:]], axis=1) for c in coords])
        self._wall_vertices = np.unique(np.concatenate([c[:-1] for c in coords]), axis=0)

    def is_valid(self, body: Polygon, collision: bool = True) -> bool:
        """Mice Model.is_valid_state: inside the arena and, if collision, off the occlusions."""
        return self.arena.contains(body) and not (collision and self.occlusions.intersects(body))

    def is_valid_disc(self, xy, radius: float) -> bool:
        """Disc of `radius` at xy inside the arena and off the occlusions."""
        p = Point(xy)
        return self.arena.contains(p) and not self.arena_edge.dwithin(p, radius) and not self.occlusions.dwithin(p, radius)

    def is_visible(self, src: Location, dst: Location) -> bool:
        return not self.occlusions.intersects(LineString([(src.x, src.y), (dst.x, dst.y)]))

    def is_clear(self, src: Location, dst: Location, margin: float) -> bool:
        """Segment src -> dst stays `margin` away from every occlusion."""
        return not self.occlusions.dwithin(LineString([(src.x, src.y), (dst.x, dst.y)]), margin)

    def body_visible(self, src_xy, points: np.ndarray) -> bool:
        """Mice rule: seen if any part of the body is in line of sight of src."""
        rays = shapely.linestrings(np.stack([np.broadcast_to(src_xy, points.shape), points], axis=1))
        return not shapely.intersects(self.occlusions, rays).all()

    def visibility_polygon(self, src_xy) -> Polygon:
        """Area visible from src (360 degrees), for rendering."""
        src = np.asarray(src_xy, dtype=float)
        angles = np.arctan2(*(self._wall_vertices - src).T[::-1])
        angles = np.sort(np.concatenate([angles - 1e-3, angles, angles + 1e-3]))
        d = np.stack([np.cos(angles), np.sin(angles)], axis=1)          # (r, 2) ray directions
        p, e = self.walls[:, 0], self.walls[:, 1] - self.walls[:, 0]    # (w, 2) segment start, direction
        cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        denom = cross(d[:, None], e[None])                               # (r, w)
        ps = (p - src)[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross(ps, e[None]) / denom                               # distance along the ray
            u = cross(ps, d[:, None]) / denom                            # position along the segment
        t = np.where((denom != 0) & (t > 0) & (u >= -1e-9) & (u <= 1 + 1e-9), t, np.inf).min(axis=1)
        hit = np.isfinite(t)   # a ray through a corner can miss both edges by rounding
        return Polygon(src + t[hit, None] * d[hit])
