"""Draws the fixed ambush zones of a world and checks that the waiting cells are
hidden from the prey entry.

  python scores/plot_ambush_zones.py [--world 21_05] [--out logdir/ambush_zones_21_05.png]
"""
import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('CELLWORLD_CACHE', str(ROOT / 'data' / '2105_world'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path[:0] = [str(ROOT), str(ROOT / 'prey_env')]

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402
from cellworld import Display, Location, Location_visibility, Polygon_list, World  # noqa: E402
from prey_env.AmbushPredator import AMBUSH_ZONES, ATTACK_CELLS, attack_radius  # noqa: E402

COLORS = ['#1f77b4', '#9467bd', '#2ca02c', '#d62728', '#ff7f0e']


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--world', default='21_05')
  parser.add_argument('--out', default=None)
  args = parser.parse_args()
  out = args.out or str(ROOT / 'logdir' / f'ambush_zones_{args.world}.png')
  world = World.get_from_parameters_names('hexagonal', 'canonical', args.world)
  size = world.implementation.cell_transformation.size
  occluded = list(world.cells.occluded_cells())
  # Same visibility model as prey_env/Model.py.
  polygons = Polygon_list.get_polygons(
      [c.location for c in occluded], 6, size / 2 * 1.05,
      world.implementation.cell_transformation.rotation)
  visibility = Location_visibility(occlusions=polygons)
  entry = Location(.05, .5)
  radius = attack_radius(world)

  display = Display(world, fig_size=(10, 10))
  free = list(world.cells.free_cells())
  visible_from_entry = [c for c in free if visibility.is_visible(entry, c.location)]
  for c in visible_from_entry:
    display.cell(cell=c, color='#fff1a8')
  for k, cell_id in enumerate(AMBUSH_ZONES[args.world]):
    c, color = world.cells[cell_id], COLORS[k % len(COLORS)]
    visible = visibility.is_visible(entry, c.location)
    display.cell(cell=c, color=color)
    display.ax.add_patch(Circle((c.location.x, c.location.y), radius, fill=False, ls='--', lw=1.6, ec=color, zorder=5))
    display.ax.text(c.location.x, c.location.y + 0.035, f'Z{k} cell {cell_id}', ha='center', fontsize=10,
                    color=color, weight='bold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=.8))
    display.ax.plot([entry.x, c.location.x], [entry.y, c.location.y],
                    color='green' if visible else 'red', lw=1.2, alpha=.7, zorder=4)
    theta, dist = entry.atan(c.location), entry.dist(c.location)
    blocker = next((occluded[i] for i, p in enumerate(polygons) if p.is_between(entry, theta=theta, dist=dist)), None)
    if blocker is not None:
      display.ax.add_patch(Circle((blocker.location.x, blocker.location.y), size / 2 * 1.05,
                                  fill=False, lw=2.2, ec='red', zorder=6))
    region = [x for x in free if x.id != cell_id and x.location.dist(c.location) <= radius]
    seen = sum(visibility.is_visible(x.location, c.location) for x in region)
    print(f'Z{k} cell {cell_id:3d} at ({c.location.x:.3f}, {c.location.y:.3f}): visible from entry = {visible}; '
          f'blocked by occluded cell {blocker.id if blocker else None}; '
          f'seen from {seen}/{len(region)} cells of its attack region')
  display.ax.plot([entry.x], [entry.y], marker='>', color='black', markersize=13, zorder=8)
  display.ax.text(entry.x + 0.02, entry.y - 0.05, 'mouse entry\n(0.05, 0.50)', fontsize=9, zorder=8)
  display.ax.add_patch(Circle((1.0, 0.5), 0.1, fc='#b7e4c7', ec='green', alpha=.5, zorder=-5))
  display.ax.text(0.97, 0.41, 'goal', fontsize=9, color='green', ha='center')
  handles = [
      Patch(fc='#fff1a8', ec='lightgray', label=f'cell visible from the mouse entry ({len(visible_from_entry)} of {len(free)} free cells)'),
      Patch(fc='black', label='occlusion'),
      Patch(fc='gray', label='ambush waiting cell (filled)'),
      Line2D([0], [0], ls='--', color='gray', label=f'ambush region: {ATTACK_CELLS} cells = {radius:.3f} (surge trigger and selection bias)'),
      Line2D([0], [0], color='red', label='sightline from entry, red = blocked (circled obstacle)')]
  display.ax.legend(handles=handles, loc='upper left', fontsize=9, framealpha=.95)
  display.ax.set_title(f'World {args.world}: fixed ambush zones', fontsize=11)
  pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
  display.fig.savefig(out, dpi=130, bbox_inches='tight')
  print('saved', out)


if __name__ == '__main__':
  main()
