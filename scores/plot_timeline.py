"""One metric over the whole history: the pretraining trunk in gray, then each
continuation strategy in its own color (thin lines = runs, thick = mean), log scale.

  python scores/plot_timeline.py --metric kl --trunk logdir/pretrain_nopred_s0 logdir/pretrain_warm_s0 \
      --runs logdir/cont2_s{1,2,3}_default logdir/cont2_s{1,2,3}_ambush [--out PNG]

--metric kl: train/dyn_kl_mean (KL before the free-bits clip), averaged per 2k steps.
--metric value_change: learning/value_change (one marker per 2k steps), 5-marker moving average.
"""
import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_learning import load_records, series, smooth, style  # noqa: E402

METRICS = {
    'kl': ('train/dyn_kl_mean', 'Dynamics KL(posterior ‖ prior)\nbefore the free-bits clip (nats, log scale)', [2, 2.5, 3, 4, 6, 8]),
    'value_change': ('learning/value_change', r'Value change $|V_k - V_{k-2000}|$' + '\n(5-marker moving average, log scale)', [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
    # Instance 2: ensemble disagreement; None = automatic log ticks.
    'disag': ('learning/disag', 'Ensemble disagreement $D$ (mean over batch)\n(5-marker moving average, log scale)', None),
    'disag_var': ('learning/disag_var', 'Variance of $D$ over batch pairs\n(5-marker moving average, log scale)', None),
    'disag_mse': ('learning/disag_mse', 'Ensemble-mean prediction MSE\n(5-marker moving average, log scale)', None),
    # Same, scored on the recent-data batch (last 20k steps; the buffer restarts empty at every resume).
    'disag_recent': ('learning/disag_recent', 'Ensemble disagreement $D$ (mean), recent-data batch\n(5-marker moving average, log scale)', None),
    'disag_var_recent': ('learning/disag_var_recent', 'Variance of $D$ over pairs, recent-data batch\n(5-marker moving average, log scale)', None),
    'disag_mse_recent': ('learning/disag_mse_recent', 'Ensemble-mean prediction MSE, recent-data batch\n(5-marker moving average, log scale)', None),
    'disag_visible_recent': ('learning/disag_visible_recent', 'Disagreement $D$ on predator-visible pairs, recent-data batch\n(5-marker moving average, log scale)', None),
    'disag_hidden_recent': ('learning/disag_hidden_recent', 'Disagreement $D$ on predator-hidden pairs, recent-data batch\n(5-marker moving average, log scale)', None),
    'kl_marker': ('learning/dyn_kl', 'World-model KL(posterior ‖ prior), replay batch\n(5-marker moving average, log scale)', None),
    'kl_recent': ('learning/dyn_kl_recent', 'World-model KL(posterior ‖ prior), recent-data batch\n(5-marker moving average, log scale)', None),
}
BIN = 2000
COLORS = {'default': '#2a78d6', 'ambush': '#eb6834'}
PHASES = [(0, 100000, '#f2f2f2', 'pretraining: no predator'),
          (100000, 200000, '#e8e8e8', 'pretraining: decoupled predator')]


def curve(records, key, lo=0):
  """Markers are already one per 2k steps (smoothed); train/* metrics are averaged per 2k."""
  x, v = series(records, key)
  x, v = x[x > lo], v[x > lo]
  if key.startswith('learning/'):
    return x, smooth(v, 5)
  edges = np.arange(lo, x.max() + BIN, BIN)
  centers, means = [], []
  for a, b in zip(edges[:-1], edges[1:]):
    m = (x > a) & (x <= b)
    if m.any():
      centers.append(b - BIN / 2)
      means.append(v[m].mean())
  return np.array(centers), np.array(means)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--trunk', nargs='+', required=True, help='trunk logdirs in training order')
  parser.add_argument('--runs', nargs='+', required=True)
  parser.add_argument('--metric', choices=sorted(METRICS), default='kl')
  parser.add_argument('--out', default=None)
  args = parser.parse_args()
  key, ylabel, ticks = METRICS[args.metric]
  bystep = {}
  for d in args.trunk:
    for r in load_records(d):
      bystep[r['step']] = r
  trunk = [bystep[s] for s in sorted(bystep)]
  s0 = json.load(open(pathlib.Path(args.runs[0]) / 'branch.json'))['step']

  fig, ax = plt.subplots(figsize=(11, 4.5))
  for lo, hi, color, label in PHASES:
    ax.axvspan(lo / 1e3, hi / 1e3, color=color, lw=0, zorder=-1)
    ax.text((lo + hi) / 2e3, 0.97, label, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9, color='#555')
  if key.startswith('learning/'):
    rx, rv = series(trunk, key)
    ax.plot(rx / 1e3, rv, 'o', color='#555555', ms=2.5, alpha=0.35)
  x, v = curve(trunk, key)
  ax.plot(x / 1e3, v, color='#555555', lw=2, label='pretrained trunk')
  for strategy, color in COLORS.items():
    runs = [d for d in args.runs if json.load(open(pathlib.Path(d) / 'config.json'))['env']['prey']['predator_strategy'] == strategy]
    curves = [curve(load_records(d), key, lo=s0) for d in runs]
    if not curves:
      continue
    for cx, cv in curves:
      ax.plot(cx / 1e3, cv, color=color, lw=0.8, alpha=0.35)
    n = min(len(c[0]) for c in curves)
    ax.plot(curves[0][0][:n] / 1e3, np.mean([c[1][:n] for c in curves], 0), color=color, lw=2.2,
            label=f'continuation, {"default (decoupled)" if strategy == "default" else "ambush"} predator (n={len(runs)})')
  ax.axvline(s0 / 1e3, color='black', lw=0.8, ls=':')
  ax.text(s0 / 1e3, 0.90, '  strategy switch', transform=ax.get_xaxis_transform(), fontsize=9)
  ax.set_yscale('log')
  ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
  ax.yaxis.set_minor_locator(ticker.NullLocator())
  ax.set_ylabel(ylabel)
  ax.set_xlabel('Env steps (k)')
  style(ax)
  if ticks:
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
  ax.legend(fontsize='small', frameon=False, loc='upper right', bbox_to_anchor=(1, 0.86))
  fig.tight_layout()
  out = args.out or f'logdir/{args.metric}_timeline.png'
  fig.savefig(out, dpi=200)
  print('Saved', out)


if __name__ == '__main__':
  main()
