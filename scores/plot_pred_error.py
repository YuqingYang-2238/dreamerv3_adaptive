"""Ensemble prediction error (learning/disag_mse) from the trunk through the continuations.

Both batches: all replay and the recent-data buffer (last 20k steps). Spikes = markers more than
`--ratio` x the median of the 5 preceding markers. Training survival underneath for the stage context.

  python scores/plot_pred_error.py --trunk $EXP/trunk_s0 --runs $EXP/cont2_s{1,2,3}_{default,ambush} \
      --out $EXP/pred_error.png
"""
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_learning import load_records, series, smooth, style  # noqa: E402

COLORS = {'default': '#2a78d6', 'ambush': '#eb6834'}
STAGES = [(0, 100000, '#f2f2f2', 'trunk: no predator'),
          (100000, 200000, '#e8e8e8', 'trunk: decoupled predator'),
          (200000, 350000, '#fbeee6', 'continuation')]


def spikes(x, v, ratio):
  out = []
  for i in range(5, len(v)):
    m = np.median(v[i - 5:i])
    if v[i] > ratio * m:
      out.append((x[i], v[i], v[i] / m))
  return out


def survival(records, lo=0, width=2000):
  x, s = series(records, 'episode/success')
  edges = np.arange(lo, x.max() + width, width)
  rate = [s[(x > a) & (x <= b)].mean() if ((x > a) & (x <= b)).any() else np.nan for a, b in zip(edges[:-1], edges[1:])]
  return edges[1:] - width / 2, np.array(rate)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--trunk', required=True)
  parser.add_argument('--runs', nargs='+', required=True)
  parser.add_argument('--ratio', type=float, default=1.5)
  parser.add_argument('--out', default='logdir/pred_error.png')
  args = parser.parse_args()
  trunk = load_records(args.trunk)
  runs = {'default': [], 'ambush': []}
  for d in args.runs:
    runs[json.load(open(pathlib.Path(d) / 'config.json'))['env']['prey']['predator_strategy']].append(load_records(d))

  fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
  for a in (ax, ax2):
    for lo, hi, color, label in STAGES:
      a.axvspan(lo / 1e3, hi / 1e3, color=color, lw=0, zorder=-1)
    a.axvline(200, color='black', lw=0.8, ls=':')
  for lo, hi, _, label in STAGES:
    ax.text((lo + hi) / 2e3, 0.98, label, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9, color='#555')

  found = []
  for key, ls, name in [('learning/disag_mse', '-', 'all replay'), ('learning/disag_mse_recent', '--', 'recent data (last 20k steps)')]:
    x, v = series(trunk, key)
    ax.plot(x / 1e3, v, 'o', color='#555', ms=2, alpha=0.3)
    ax.plot(x / 1e3, smooth(v, 3), ls, color='#333', lw=1.8, label=f'trunk, {name}')
    found += [(name, *s) for s in spikes(x, v, args.ratio)]
    for strategy, color in COLORS.items():
      curves = [series(r, key) for r in runs[strategy]]
      if not curves:
        continue
      n = min(len(c[0]) for c in curves)
      mean = np.mean([c[1][:n] for c in curves], 0)
      for cx, cv in curves:
        ax.plot(cx / 1e3, smooth(cv, 3), ls, color=color, lw=0.7, alpha=0.3)
      ax.plot(curves[0][0][:n] / 1e3, smooth(mean, 3), ls, color=color, lw=2,
              label=f'{strategy} continuation (n={len(curves)}), {name}')
      for cx, cv in curves:
        found += [(f'{strategy} {name}', *s) for s in spikes(cx, cv, args.ratio)]
  for name, sx, sv, r in found:
    ax.annotate(f'{sx / 1e3:.0f}k\nx{r:.1f}', (sx / 1e3, sv), xytext=(6, 4), textcoords='offset points',
                fontsize=8, color='#b00020')
    ax.plot(sx / 1e3, sv, 'v', color='#b00020', ms=7)
  x, v = series(trunk, 'learning/disag_mse')
  peak = int(np.argmax(smooth(v, 3)))
  ax.annotate(f'peak {x[peak] / 1e3:.0f}k\n(learning the goal)', (x[peak] / 1e3, smooth(v, 3)[peak]),
              xytext=(14, -28), textcoords='offset points', fontsize=8, color='#333')
  ax.set_ylabel('Ensemble prediction error\nMSE per dim of mean prediction vs $e_{t+1}$')
  ax.set_ylim(bottom=0)
  ax.legend(fontsize='small', frameon=False, loc='upper right', bbox_to_anchor=(1, 0.9), ncol=2)
  style(ax)

  cx, cv = survival(trunk)
  ax2.plot(cx / 1e3, smooth(np.nan_to_num(cv), 3), color='#333', lw=1.8, label='trunk')
  for strategy, color in COLORS.items():
    curves = [survival(r, lo=200000) for r in runs[strategy]]
    if curves:
      n = min(len(c[0]) for c in curves)
      ax2.plot(curves[0][0][:n] / 1e3, smooth(np.nanmean([c[1][:n] for c in curves], 0), 3), color=color, lw=2, label=strategy)
  ax2.set_ylabel('Training survival\n(per 2k steps)')
  ax2.set_ylim(-0.02, 1.02)
  ax2.set_xlabel('Env steps (k)')
  ax2.legend(fontsize='small', frameon=False, loc='lower right')
  style(ax2)
  fig.suptitle('Ensemble prediction error from pretraining to the strategy switch (spikes: > '
               f'{args.ratio}x the median of the 5 preceding markers)')
  fig.tight_layout()
  fig.savefig(args.out, dpi=170)
  print('Saved', args.out)
  for name, sx, sv, r in found:
    print(f'spike: {name:40s} at {sx / 1e3:.0f}k  value {sv:.4f}  x{r:.2f} of the preceding median')


if __name__ == '__main__':
  main()
