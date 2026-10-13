"""Plot the learning curve and the value-change marker of one DreamerV3 run.

Usage:
  python scores/plot_learning.py LOGDIR [--out PATH]

Reads <logdir>/metrics.jsonl and <logdir>/eval_trajs/step_*.npz and draws
three panels on a shared env-step axis: evaluation rates, episode return,
and the critic value-change marker (raw points plus a smoothed line).
Saves to <logdir>/learning_curve.png unless --out is given.
"""

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
BIN = 2000          # env steps per bin for the training return
SMOOTH = 5          # window of the centered moving average for value change


def load_records(logdir):
  """Returns the metrics.jsonl records sorted by step, last record per step."""
  filename = pathlib.Path(logdir) / 'metrics.jsonl'
  if not filename.exists():
    sys.exit(f'No metrics.jsonl in {logdir}')
  bystep = {}
  with open(filename) as f:
    for line in f:
      if line.strip():
        record = json.loads(line)
        bystep.setdefault(record['step'], {}).update(record)
  return [bystep[step] for step in sorted(bystep)]


def series(records, key):
  """(steps, values) arrays for one metric key."""
  pairs = [(r['step'], r[key]) for r in records if key in r]
  if not pairs:
    return np.array([]), np.array([])
  steps, values = zip(*pairs)
  return np.asarray(steps, np.float64), np.asarray(values, np.float64)


def strict_success(logdir, records):
  """Eval success = goal reached with zero captures, per eval phase.

  Newer runs log it as eval/success_rate (with eval/goal_rate alongside);
  older runs logged goal-reached there, so recompute from the trajectories.
  """
  steps, logged = series(records, 'eval/success_rate')
  if any('eval/goal_rate' in r for r in records):
    return steps, logged, series(records, 'eval/goal_rate')[1]
  strict = []
  for step in steps:
    d = np.load(pathlib.Path(logdir) / 'eval_trajs' / f'step_{int(step):08d}.npz')
    strict.append((d['ep_success'] & (d['ep_captures'] == 0)).mean())
  return steps, np.asarray(strict), logged


def binning(xs, ys, borders):
  """Mean of ys within each (start, stop] bin; nan for empty bins."""
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  binned = []
  for start, stop in zip(borders[:-1], borders[1:]):
    left, right = (xs <= start).sum(), (xs <= stop).sum()
    binned.append(ys[left:right].mean() if left < right else np.nan)
  return borders[1:], np.asarray(binned)


def smooth(values, window):
  """Centered moving average with shrinking windows at the edges."""
  half = window // 2
  return np.array([
      values[max(0, i - half):i + half + 1].mean() for i in range(len(values))])


def style(ax):
  ax.grid(which='major', color='#eeeeee')
  ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
  ax.tick_params(axis='both', which='major', labelsize='small', pad=1, length=1)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)


def legend(ax):
  leg = ax.legend(fontsize='small', numpoints=1, labelspacing=0.2,
                  handlelength=1.5, handletextpad=0.5, loc='best')
  leg.get_frame().set_edgecolor('white')


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('logdir')
  parser.add_argument('--out', default=None, help='Output PNG path.')
  args = parser.parse_args()
  records = load_records(args.logdir)
  name = pathlib.Path(args.logdir).name

  fig, axes = plt.subplots(3, 1, figsize=(6, 7.5), sharex=True)

  # 1. Evaluation rates.
  ax = axes[0]
  steps, strict, goal = strict_success(args.logdir, records)
  ax.plot(steps, goal, color=ORANGE, lw=2, marker='o', ms=4, label='goal reached')
  ax.plot(steps, strict, color=BLUE, lw=2, marker='o', ms=4, label='success (goal, no capture)')
  csteps, capture = series(records, 'eval/capture_rate')
  ax.plot(csteps, capture, color=AQUA, lw=2, marker='o', ms=4, label='captured at least once')
  ax.set_ylim(-0.03, 1.03)
  ax.set_ylabel('Eval rate (20 episodes)')
  ax.set_title(f'{name}: learning curve and critic value change', fontsize='medium')
  legend(ax)
  style(ax)

  # 2. Episode return.
  ax = axes[1]
  tsteps, treturn = series(records, 'episode/score')
  ax.scatter(tsteps, treturn, s=4, color=BLUE, alpha=0.15, lw=0, label='train episode')
  borders = np.arange(0, tsteps.max() + BIN, BIN)
  centers, binned = binning(tsteps, treturn, borders)
  ax.plot(centers - BIN / 2, binned, color=BLUE, lw=2, label=f'train mean per {BIN // 1000}k steps')
  esteps, ereturn = series(records, 'eval/return_mean')
  ax.plot(esteps, ereturn, color=ORANGE, lw=2, marker='o', ms=4, label='eval mean')
  ax.set_ylabel('Episode return')
  legend(ax)
  style(ax)

  # 3. Value-change marker: raw points and a smoothed line.
  ax = axes[2]
  vsteps, vchange = series(records, 'learning/value_change')
  ax.plot(vsteps, vchange, color=BLUE, lw=0, marker='o', ms=4, alpha=0.5, label='raw (every 2k steps)')
  ax.plot(vsteps, smooth(vchange, SMOOTH), color=BLUE, lw=2, label=f'{SMOOTH}-point moving average')
  ax.set_ylabel(r'Mean $|V_k - V_{k-\Delta}|$')
  ax.set_xlabel('Env steps')
  ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
  ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
  legend(ax)
  style(ax)

  fig.tight_layout()
  out = pathlib.Path(args.out or pathlib.Path(args.logdir) / 'learning_curve.png')
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=200)
  print('Saved', out)


if __name__ == '__main__':
  main()
