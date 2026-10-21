"""Per-step signals of a frozen evaluation, split by whether the predator is visible to the prey.

Reads main_eval output dirs (trajs.npz) and prints, per strategy, the mean of each per-step
signal on visible steps, hidden steps and steps with the predator within --near, with the
per-seed values. The first step of an episode and the last (zeroed by the driver) are dropped.

  python scores/split_visible.py $EXP/trunk_s0/frozenerr_{default,ambush}_s{1,2,3} --keys kl disag disag_err
"""
import argparse
import json
import pathlib

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('evaldirs', nargs='+')
  parser.add_argument('--keys', nargs='+', default=['kl', 'disag', 'disag_err'])
  parser.add_argument('--near', type=float, default=0.2, help='distance threshold (arena widths)')
  args = parser.parse_args()
  runs = {}
  for d in args.evaldirs:
    strategy = json.load(open(pathlib.Path(d) / 'summary.json'))['strategy']
    runs.setdefault(strategy, []).append(np.load(pathlib.Path(d) / 'trajs.npz'))
  for key in args.keys:
    print(f'=== {key} ===')
    means = {}
    for strategy, group in runs.items():
      seeds = []
      for t in group:
        if key not in t.files:
          continue
        off = t['ep_offsets']
        vis, hid, near = [], [], []
        for a, b in zip(off[:-1], off[1:]):
          x, v = t[key][a + 1:b - 1], t['pred_visible'][a + 1:b - 1]
          d = np.linalg.norm(t['prey_xy'][a + 1:b - 1] - t['pred_xy'][a + 1:b - 1], axis=1)
          vis += list(x[v]); hid += list(x[~v]); near += list(x[d < args.near])
        seeds.append((np.mean(vis), np.mean(hid), np.mean(near) if near else np.nan))
      if not seeds:
        print(f'  {strategy:8s} not logged')
        continue
      means[strategy] = np.mean(seeds, 0)
      per_seed = ', '.join(f'({v:.4g}, {h:.4g})' for v, h, _ in seeds)
      print(f'  {strategy:8s} visible {means[strategy][0]:.4g}  hidden {means[strategy][1]:.4g}  '
            f'within {args.near} {means[strategy][2]:.4g}   per seed (visible, hidden): {per_seed}')
    if {'default', 'ambush'} <= set(means):
      rel = means['ambush'] / means['default'] - 1
      print(f'  ambush vs default: visible {rel[0]:+.0%}, hidden {rel[1]:+.0%}, within {args.near} {rel[2]:+.0%}')


if __name__ == '__main__':
  main()
