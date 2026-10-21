"""Mice-look frames (cellworld_game View / BotEvade render steps) on an offscreen pygame surface."""
import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

WIDTH = 800
HEIGHT = int(math.sqrt(3) / 2 * WIDTH)
FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
# Mice colours.
ARENA, OCCLUSION, VISIBLE, HIDDEN = (210, 210, 210), (50, 50, 50), (255, 255, 255), (210, 210, 210, 255)
BODY = {"prey": (20, 90, 20), "predator": (90, 20, 20)}
SPRITES = {"prey": ("prey.png", .04 * 2.0), "predator": ("predator.png", .05 * 1.9)}   # file, size x scale


class Renderer:
    """pov 'prey': white visible area, hidden area masked, as Mice's SAC_test; 'top': everything + puff circle."""

    def __init__(self, env, pov: str = "prey", sprites: bool = True):
        self.env, self.pov = env, pov
        self.surface = pygame.Surface((WIDTH, HEIGHT))
        self.images = {}
        if sprites:
            for name, (file, size) in SPRITES.items():
                image = pygame.transform.rotate(pygame.image.load(os.path.join(FILES, file)), 270)
                self.images[name] = pygame.transform.smoothscale(image, (int(size * WIDTH),) * 2)

    @staticmethod
    def px(xy):
        """Mice CoordinateConverter: canonical (x, y) to screen, y flipped."""
        return xy[0] * WIDTH, (1 - xy[1]) * WIDTH - (WIDTH - HEIGHT) / 2

    def polygon(self, coords, color, surface=None):
        pygame.draw.polygon(surface or self.surface, color, [self.px(c) for c in coords])

    def circle(self, location, radius, color, width=0):
        r = radius * WIDTH
        if len(color) == 4:   # alpha fill needs its own surface
            disc = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
            pygame.draw.circle(disc, color, (r, r), r)
            cx, cy = self.px((location.x, location.y))
            self.surface.blit(disc, (cx - r, cy - r))
        else:
            pygame.draw.circle(self.surface, color, self.px((location.x, location.y)), r, width)

    def occlusions(self):
        for cell in self.env.model.geometry.occlusion_cells:
            self.polygon(cell.exterior.coords, OCCLUSION)

    def agent(self, name, data):
        if name in self.images:
            image = pygame.transform.rotate(self.images[name], math.degrees(data.theta))
            cx, cy = self.px((data.location.x, data.location.y))
            self.surface.blit(image, (cx - image.get_width() / 2, cy - image.get_height() / 2))
        else:
            self.polygon(data.body_points, BODY.get(name, (0, 80, 120)))

    def frame(self) -> np.ndarray:
        env, geometry, agents = self.env, self.env.model.geometry, self.env.model.agents_data
        self.surface.fill((0, 0, 0))
        self.polygon(geometry.arena.exterior.coords, ARENA)
        if self.pov == "prey":
            prey = agents["prey"].location
            visible = geometry.visibility_polygon((prey.x, prey.y))
            self.polygon(visible.exterior.coords, VISIBLE)
        else:
            self.occlusions()
            self.circle(env.goal_location, env.goal_threshold, (0, 200, 0, 80))
            for k, cell in enumerate(env.ambush_zones):
                self.circle(cell.location, env.attack_radius,
                            (255, 0, 255, 60) if k == env.ambush_zone else (120, 120, 120, 25))
            if "predator" in agents:   # Mice render_puff_area
                predator = agents["predator"].location
                self.circle(predator, env.capture_threshold,
                            (255, 0, 0, 60) if env.capture_cooldown_remaining > 0 else (0, 0, 255, 60))
                self.circle(predator, env.capture_threshold, (0, 0, 255), 2)
        for name, data in agents.items():
            self.agent(name, data)
        if self.pov == "prey":   # Mice render_hidden_area, then occlusions on top
            mask = pygame.Surface(self.surface.get_size(), pygame.SRCALPHA)
            mask.fill((0, 0, 0, 255))
            self.polygon(geometry.arena.exterior.coords, HIDDEN, mask)
            self.polygon(visible.exterior.coords, (0, 0, 0, 0), mask)
            self.surface.blit(mask, (0, 0))
            self.occlusions()
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))
