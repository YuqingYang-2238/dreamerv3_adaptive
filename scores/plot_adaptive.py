"""Fixed vs signal-adaptive imagination: learning curves, controller trace, compute table.

  python scores/plot_adaptive.py logdir/adapt_fixed logdir/adapt_adaptive [--out PNG] [--table MD]

x = env steps since the branch. Panels: eval success and capture rate, eval
return, training survival (5k bins), the recent-experience value change
(raw and controller-smoothed, with the reference c), the adaptive probability
p_extra with the realized extra-update fraction, and cumulative imagined
trajectories. The table compares performance and computation per condition.
"""
import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_learning import load_records, series, style  # noqa: E402

COLORS = {'fixed': '#2a78d6', 'adaptive': '#eb6834'}  # fixed order, never cycled
BIN = 5000       # env steps per bin for training survival
LAST = 30000     # window for the "late" performance summary


def load_run(logdir):
  logdir = pathlib.Path(logdir)
  branch = json.load(open(logdir / 'branch.json'))
  config = json.load(open(logdir / 'config.json'))
  info = [json.loads(l) for l in open(logdir / 'run_info.jsonl') if l.strip()][-1]
  records = load_records(logdir)
  name = 'adaptive' if config['run']['adaptive'] else 'fixed'
  return dict(name=name, s0=branch['step'], records=records, c=config['run']['adaptive_c'],
              info=info, logdir=str(logdir))


def rel(run, key, lo=None):
  x, y = series(run['records'], key)
  x = x - run['s0']
  keep = x >= 0 if lo is None else x > lo
  return x[keep], y[keep]


def survival(run):
  x, y = rel(run, 'episode/success')
  edges = np.arange(0, x.max() + BIN, BIN)
  idx = np.digitize(x, edges) - 1
  return edges[1:], np.array([y[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(edges) - 1)])


def last(run, key):
  x, y = rel(run, key)
  return y[-1] if len(y) else np.nan


def summary(run):
  es, succ = rel(run, 'eval/success_rate')
  _, cap = rel(run, 'eval/capture_rate')
  _, ret = rel(run, 'eval/return_mean')
  sx, sv = rel(run, 'episode/success')
  late = sx > sx.max() - LAST
  n_eval = len(es)
  wall = last(run, 'adaptive/wall_seconds')
  return {
      'condition': run['name'],
      'env steps': int(sx.max()),
      'eval phases': n_eval,
      'final eval success': succ[-1],
      'last-3 eval success': succ[-3:].mean(),
      'final eval capture rate': cap[-1],
      'last-3 eval capture rate': cap[-3:].mean(),
      'final eval return': ret[-1],
      'last-3 eval return': ret[-3:].mean(),
      f'training survival (last {LAST // 1000}k)': sv[late].mean(),
      'training episodes': len(sx),
      'actor-critic updates': int(last(run, 'adaptive/updates')),
      'world-model updates': int(last(run, 'adaptive/wm_updates')),
      'extra-budget updates': int(last(run, 'adaptive/updates_extra')),
      'extra fraction': last(run, 'adaptive/extra_frac_total'),
      'imagined trajectories': int(last(run, 'adaptive/imag_trajs')),
      'imagined transitions': int(last(run, 'adaptive/imag_transitions')),
      'training wall time (h)': wall / 3600,
      'c': run['c'] if run['name'] == 'adaptive' else None,
  }


def table(runs):
  rows = [summary(r) for r in runs]
  keys = [k for k in rows[0] if k != 'condition']
  fmt = lambda v: '-' if v is None else (f'{v:,}' if isinstance(v, int) else f'{v:.3f}')
  lines = ['| metric | ' + ' | '.join(r['condition'] for r in rows) + ' |',
           '|---|' + '---|' * len(rows)]
  for k in keys:
    lines.append(f'| {k} | ' + ' | '.join(fmt(r[k]) for r in rows) + ' |')
  return '\n'.join(lines)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('runs', nargs=2)
  parser.add_argument('--out', default='logdir/adaptive_pilot.png')
  parser.add_argument('--table', default='logdir/adaptive_pilot.md')
  args = parser.parse_args()
  runs = sorted([load_run(d) for d in args.runs], key=lambda r: r['name'] != 'fixed')
  adaptive = next(r for r in runs if r['name'] == 'adaptive')
  c = adaptive['c']

  fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
  (ax_succ, ax_ret, ax_surv), (ax_u, ax_p, ax_traj) = axes
  k = lambda v: v / 1000
  for run in runs:
    col, name = COLORS[run['name']], run['name']
    x, y = rel(run, 'eval/success_rate')
    ax_succ.plot(k(x), y, '-o', color=col, lw=2, ms=4, label=f'{name}: success')
    x, y = rel(run, 'eval/capture_rate')
    ax_succ.plot(k(x), y, '--', color=col, lw=1.2, label=f'{name}: capture rate')
    x, y = rel(run, 'eval/return_mean')
    ax_ret.plot(k(x), y, '-o', color=col, lw=2, ms=4, label=name)
    x, y = survival(run)
    ax_surv.plot(k(x), y, '-', color=col, lw=2, label=name)
    x, y = rel(run, 'adaptive/u_raw')
    ax_u.plot(k(x), y, '-', color=col, lw=1, alpha=0.45, label=f'{name}: raw')
    x, y = rel(run, 'adaptive/u_smoothed')
    ax_u.plot(k(x), y, '-', color=col, lw=2, label=f'{name}: smoothed')
    x, y = rel(run, 'adaptive/imag_trajs')
    ax_traj.plot(k(x), y / 1e6, '-', color=col, lw=2, label=name)
  ax_u.axhline(c, color='#555555', lw=1, ls=':', label=f'c = {c:.4f}')
  x, y = rel(adaptive, 'adaptive/p_extra')
  ax_p.step(k(x), y, where='post', color=COLORS['adaptive'], lw=2, label='p_extra (held)')
  x, y = rel(adaptive, 'adaptive/extra_frac_interval')
  ax_p.plot(k(x), y, 'o', color=COLORS['adaptive'], ms=4, alpha=0.6, label='realized extra fraction')
  ax_p.set_ylim(-0.02, 1.02)

  titles = ['Evaluation: success (goal, no capture) and capture rate', 'Evaluation: mean return',
            f'Training survival ({BIN // 1000}k-step bins)',
            'Recent-experience value change (controller signal)',
            'Adaptive condition: extra-imagination probability', 'Cumulative imagined trajectories (millions)']
  for ax, title in zip(axes.flat, titles):
    ax.set_title(title, fontsize='small', loc='left')
    ax.set_xlabel('env steps since branch (k)', fontsize='small')
    style(ax)
    ax.legend(fontsize='x-small', frameon=False)
  fig.suptitle('Fixed vs signal-adaptive imagination against the ambush predator (one seed)', fontsize='medium')
  fig.tight_layout()
  fig.savefig(args.out, dpi=150)
  md = table(runs)
  pathlib.Path(args.table).write_text(md + '\n')
  print(md)
  print(f'\nfigure: {args.out}\ntable:  {args.table}')


if __name__ == '__main__':
  main()
