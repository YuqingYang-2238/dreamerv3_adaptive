"""Default -> Default vs Default -> Ambush continuation runs, aligned at the branch.

  python scores/plot_branches.py logdir/cont_s{1,2,3}_default logdir/cont_s{1,2,3}_ambush [--out PNG]

Six panels against env steps since the branch (x = 0): training survival,
the value-change marker, the same marker on recent data only, the current
critic's mean value, the world model's dynamics KL before the free-bits clip,
and the evaluation success per phase. Thin lines are single runs, thick lines
the mean per predator strategy; the gray band is the trunk's settled marker floor.
"""
import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_learning import load_records, series, smooth, style  # noqa: E402

COLORS = {'default': '#2a78d6', 'ambush': '#eb6834'}
BIN = 5000      # env steps per bin for training survival
SMOOTH = 5      # markers in the centered moving average
FLOOR = 27000   # trunk steps before the branch that define the settled floor


def load_run(logdir):
  logdir = pathlib.Path(logdir)
  branch = json.load(open(logdir / 'branch.json'))
  config = json.load(open(logdir / 'config.json'))
  return dict(strategy=config['env']['prey']['predator_strategy'], s0=branch['step'],
              source=branch['source'], records=load_records(logdir))


def survival(run):
  """Fraction of training episodes reaching the goal without capture, per BIN."""
  x, score = series(run['records'], 'episode/success' if any('episode/success' in r for r in run['records']) else 'episode/score')
  x -= run['s0']
  edges = np.arange(0, x.max() + BIN, BIN)
  rate = [np.mean(score[(x > a) & (x <= b)] == 1) for a, b in zip(edges[:-1], edges[1:])]
  return edges[1:] - BIN / 2, np.array(rate)


def marker(run, key):
  x, v = series(run['records'], key)
  return x - run['s0'], smooth(v, SMOOTH)


def binned(run, key, width=2000):
  """Mean of a train/* metric per `width` env steps after the branch."""
  x, v = series(run['records'], key)
  x = x - run['s0']
  edges = np.arange(0, x.max() + width, width)
  means = [v[(x > a) & (x <= b)].mean() if ((x > a) & (x <= b)).any() else np.nan
           for a, b in zip(edges[:-1], edges[1:])]
  return edges[1:] - width / 2, np.array(means)


def evaluation(run, key='eval/success_rate'):
  x, v = series(run['records'], key)
  return x - run['s0'], v


def draw(ax, runs, curve, dots=False):
  style = dict(marker='o', ms=4) if dots else {}
  for strategy, color in COLORS.items():
    curves = [curve(r) for r in runs if r['strategy'] == strategy]
    if not curves:
      continue
    for x, y in curves:
      ax.plot(x / 1e3, y, color=color, lw=0.8, alpha=0.35, **style)
    n = min(len(x) for x, _ in curves)
    ax.plot(curves[0][0][:n] / 1e3, np.nanmean([y[:n] for _, y in curves], 0),
            color=color, lw=2.2, label=f'Default → {strategy.capitalize()} (n={len(curves)})', **style)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('logdirs', nargs='+')
  parser.add_argument('--out', default=None, help='Output PNG (default: logdir/branches.png).')
  args = parser.parse_args()
  runs = [load_run(d) for d in args.logdirs]
  trunk = load_records(runs[0]['source'])
  x, v = series(trunk, 'learning/value_change')
  floor = v[(x > runs[0]['s0'] - FLOOR) & (x <= runs[0]['s0'])]

  disag = any('learning/disag' in r for r in runs[0]['records'])   # Instance 2 adds a third row
  fig, axes = plt.subplots(3 if disag else 2, 3, figsize=(15, 10.5 if disag else 7), sharex=True)
  panels = [
      (axes[0, 0], survival, 'Training survival\n(goal without capture)', False),
      (axes[0, 1], lambda r: marker(r, 'learning/value_change'), 'Value change, all replay\n' + r'mean $|V_k - V_{k-2000}|$', False),
      (axes[0, 2], lambda r: marker(r, 'learning/value_change_recent'), 'Value change, last 20k steps\n' + r'mean $|V_k - V_{k-2000}|$', False),
      (axes[1, 0], lambda r: marker(r, 'learning/value_new_mean'), 'Critic mean value\n(reward units)', False),
      (axes[1, 1], lambda r: binned(r, 'train/dyn_kl_mean'), 'Dynamics KL(posterior ‖ prior)\nbefore the free-bits clip (nats, 2k bins)', False),
      (axes[1, 2], evaluation, 'Evaluation success\n(scored episodes per phase)', True),
  ]
  if disag:
    panels += [
        (axes[2, 0], lambda r: marker(r, 'learning/disag'), 'Ensemble disagreement $D$ (mean)\nall replay', False),
        (axes[2, 1], lambda r: marker(r, 'learning/disag_var'), 'Variance of $D$ over pairs\nall replay', False),
        (axes[2, 2], lambda r: evaluation(r, 'eval/disag_mean'), 'Evaluation disagreement\n(episode mean, frozen agent)', True),
    ]
  for ax, curve, label, dots in panels:
    draw(ax, runs, curve, dots)
    ax.set_ylabel(label)
    ax.axvline(0, color='black', lw=0.8, ls=':')
    style(ax)
  axes[0, 1].axhspan(floor.mean() - 2 * floor.std(), floor.mean() + 2 * floor.std(),
                     color='#888888', alpha=0.15, lw=0, label='trunk floor ± 2 SD')
  for ax in axes[0, 0], axes[1, 2]:
    ax.set_ylim(0, 1.02)
  for ax in axes[0, 1], axes[0, 2]:
    ax.set_ylim(bottom=0)
  for ax in axes[-1]:
    ax.set_xlabel('Env steps since the predator switch (k)')
  axes[0, 1].legend(fontsize='small', frameon=False, loc='upper right')
  fig.suptitle(f'Continuation from {pathlib.Path(runs[0]["source"]).name} at step {runs[0]["s0"]:,}'
               f' ({SMOOTH}-marker moving average)', fontsize='medium')
  fig.tight_layout()
  out = pathlib.Path(args.out or 'logdir/branches.png')
  fig.savefig(out, dpi=200)
  print('Saved', out)


if __name__ == '__main__':
  main()
