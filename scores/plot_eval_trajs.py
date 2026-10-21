"""Prey trajectories of evaluation phases: one panel per (run, steps since branch).

  python scores/plot_eval_trajs.py logdir/cont_s1_default logdir/cont_s1_ambush [--x 0 80000 150000] [--out PNG]

Reads eval_trajs/step_*.npz (the phase nearest to each x) and draws the world
with its occlusions, the ambush zones for ambush runs, every episode's prey
path coloured by outcome, the predator where the prey saw it, and captures.
A logdir without branch.json (a trunk) is plotted at absolute steps.
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('CELLWORLD_CACHE', str(ROOT / 'data' / '2105_world'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path[:0] = [str(ROOT), str(ROOT / 'prey_env')]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cellworld import Display, World  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from main_train import ENV_KWARGS  # noqa: E402
from prey_env.AmbushPredator import AMBUSH_ZONES, attack_radius  # noqa: E402

OUTCOME = {'success': '#1baf7a', 'captured': '#eb6834', 'timeout': '#888888'}


def load_run(logdir):
  logdir = pathlib.Path(logdir)
  branch = logdir / 'branch.json'
  s0 = json.load(open(branch))['step'] if branch.exists() else 0
  config = json.load(open(logdir / 'config.json'))
  strategy = config.get('env', {}).get('prey', {}).get('predator_strategy', 'default')
  phases = sorted((int(p.stem.split('_')[1]), p) for p in (logdir / 'eval_trajs').glob('step_*.npz'))
  return dict(name=logdir.name, s0=s0, strategy=strategy, phases=phases)


def episodes(data):
  """Per-episode dicts of the arrays stored by run_eval / main_eval."""
  offsets = data['ep_offsets']
  for i in range(len(offsets) - 1):
    sl = slice(offsets[i], offsets[i + 1])
    captures = int(data['ep_captures'][i])
    outcome = 'success' if data['ep_success'][i] else 'captured' if captures else 'timeout'
    yield dict(outcome=outcome, captures=captures, prey=data['prey_xy'][sl],
               pred=data['pred_xy'][sl], visible=data['pred_visible'][sl], captured=data['captured'][sl])


def draw(ax, world, data, strategy, title):
  Display(world, fig=ax.figure, ax=ax)
  eps = list(episodes(data))
  for ep in eps:
    ax.plot(ep['prey'][:, 0], ep['prey'][:, 1], color=OUTCOME[ep['outcome']], lw=1, alpha=0.7)
    seen = ep['visible'] & (ep['pred'][:, 0] >= 0)
    ax.scatter(ep['pred'][seen, 0], ep['pred'][seen, 1], s=3, color='red', alpha=0.3, lw=0)
    hit = ep['captured'] > 0
    ax.plot(ep['prey'][hit, 0], ep['prey'][hit, 1], 'x', color='red', ms=5, mew=1.2)
  if strategy == 'ambush':
    for cell_id in AMBUSH_ZONES[ENV_KWARGS['world_name']]:
      c = world.cells[cell_id].location
      ax.add_patch(Circle((c.x, c.y), attack_radius(world), fill=False, ls='--', lw=1, ec='#9467bd'))
  ax.add_patch(Circle((1.0, 0.5), 0.1, fc='#b7e4c7', ec='green', alpha=0.4))
  ax.plot([0.05], [0.5], marker='>', color='black', ms=8)
  n = len(eps)
  success = sum(e['outcome'] == 'success' for e in eps)
  captures = sum(e['captures'] for e in eps)
  ax.set_title(f'{title}\nsuccess {success}/{n}, captures {captures}', fontsize=9)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('logdirs', nargs='+')
  parser.add_argument('--x', type=int, nargs='+', default=[0, 80000, 150000],
                      help='Steps since the branch of the phases to show.')
  parser.add_argument('--out', default=None, help='Output PNG (default: next to the first logdir).')
  args = parser.parse_args()
  runs = [load_run(d) for d in args.logdirs]
  world = World.get_from_parameters_names('hexagonal', 'canonical', ENV_KWARGS['world_name'])
  fig, axes = plt.subplots(len(runs), len(args.x), figsize=(4.2 * len(args.x), 4.4 * len(runs)), squeeze=False)
  for row, run in zip(axes, runs):
    for ax, x in zip(row, args.x):
      if not run['phases']:
        ax.set_axis_off()
        continue
      step, path = min(run['phases'], key=lambda p: abs(p[0] - run['s0'] - x))
      draw(ax, world, np.load(path), run['strategy'],
           f"{run['name']} ({run['strategy']}) at x = {(step - run['s0']) / 1e3:g}k")
  handles = [Line2D([0], [0], color=c, label=k) for k, c in OUTCOME.items()]
  handles += [Line2D([0], [0], ls='', marker='o', color='red', ms=3, label='predator seen by the prey'),
              Line2D([0], [0], ls='', marker='x', color='red', label='capture')]
  fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, frameon=False)
  fig.tight_layout(rect=(0, 0.03, 1, 1))
  out = pathlib.Path(args.out or pathlib.Path(args.logdirs[0]) / 'eval_trajs.png')
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=150)
  print('Saved', out)


if __name__ == '__main__':
  main()
