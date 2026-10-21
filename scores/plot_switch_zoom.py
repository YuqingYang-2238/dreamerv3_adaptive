"""Zoom on the first 20k continuation steps: every value-change marker (one per
2k steps, unsmoothed) of every run, scored on the usual replay batch and on the
recent-data batch, with the trunk's last 20k steps in gray for reference. A
second figure (<out>_paired.png) shows the paired differences per seed,
D_i(t) = ambush_i(t) - default_i(t), summarized by one mean per seed. A third
(<out>_kl.png) does the same for the unclamped dynamics KL, which is logged at
every ~500 steps (wall-clock cadence), so its paired differences use 2k bins.

  python scores/plot_switch_zoom.py --trunk logdir/pretrain_warm_s0 \
      --runs logdir/cont2_s{1,2,3}_default logdir/cont2_s{1,2,3}_ambush [--out PNG]

The usual batch is drawn uniformly from the whole replay, so only a small share
of its steps were collected after the switch (printed under the x axis). The
recent batch is drawn from the last 20k steps of experience, which after the
switch are post-switch steps only.
"""
import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_learning import load_records, series, style  # noqa: E402

WINDOW = 20000
COLORS = {'default': '#2a78d6', 'ambush': '#eb6834'}
PANELS = [  # (key, y label, short name); left column: usual batch, right column: recent batch
    ('learning/value_change', 'Value change $|V_k - V_{k-2000}|$\nusual replay batch', 'value change, usual batch'),
    ('learning/value_change_recent', 'Value change $|V_k - V_{k-2000}|$\nrecent-data batch (post-switch steps only)', 'value change, recent batch'),
    ('learning/value_new_mean', 'Critic mean value $V_k$\nusual replay batch', 'critic mean value, usual batch'),
    ('learning/value_new_mean_recent', 'Critic mean value $V_k$\nrecent-data batch', 'critic mean value, recent batch'),
    # Instance 2: Plan2Explore ensemble disagreement, signal 1 (mean over pairs) and signal 2 (variance over pairs).
    ('learning/disag', 'Ensemble disagreement $D$ (mean)\nusual replay batch', 'disagreement mean, usual batch'),
    ('learning/disag_recent', 'Ensemble disagreement $D$ (mean)\nrecent-data batch', 'disagreement mean, recent batch'),
    ('learning/disag_var', 'Variance of $D$ over pairs\nusual replay batch', 'disagreement variance, usual batch'),
    ('learning/disag_var_recent', 'Variance of $D$ over pairs\nrecent-data batch', 'disagreement variance, recent batch'),
    # Instance 2b: split of the recent-batch disagreement by what the prey sees.
    ('learning/disag_visible_recent', 'Disagreement $D$, predator visible\nrecent-data batch', 'disagreement visible, recent batch'),
    ('learning/disag_hidden_recent', 'Disagreement $D$, predator hidden\nrecent-data batch', 'disagreement hidden, recent batch'),
]
KL = 'train/dyn_kl_mean'
XLABEL = 'Env steps since the switch (k)\nshare of the replay collected after the switch'


def window(records, key, s0, lo, hi):
  """(x, v) of the markers with lo < x <= hi, where x = step - s0."""
  x, v = series(records, key)
  keep = (x > s0 + lo) & (x <= s0 + hi)
  return x[keep] - s0, v[keep]


def binned(records, key, s0, lo, hi, width=2000):
  """(bin centers, bin means) of a metric logged at irregular steps."""
  x, v = window(records, key, s0, lo, hi)
  edges = np.arange(lo, hi + 1, width)
  idx = np.digitize(x, edges[1:], right=True)
  means = [v[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(edges) - 1)]
  return edges[:-1] + width / 2, np.array(means)


def label_x(axes, share, lo):
  """Ticks every 4k steps from lo; positive ticks also show the post-switch share of the replay."""
  ticks = np.arange(lo, WINDOW + 1, 4000)
  labels = [f'{t // 1000}' + (f'\n{share[t]:.0%}' if t > 0 else '\n0%' if t == 0 else '')
            for t in ticks]
  for ax in axes:
    ax.set_xticks(ticks / 1e3)
    ax.set_xticklabels(labels)
    ax.set_xlabel(XLABEL)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('--trunk', required=True, help='logdir the runs were branched from')
  parser.add_argument('--runs', nargs='+', required=True)
  parser.add_argument('--out', default='logdir/switch_zoom.png')
  args = parser.parse_args()
  s0 = json.load(open(pathlib.Path(args.runs[0]) / 'branch.json'))['step']
  trunk = load_records(args.trunk)
  runs = {strategy: {} for strategy in COLORS}  # strategy -> seed -> records
  for d in args.runs:
    config = json.load(open(pathlib.Path(d) / 'config.json'))
    runs[config['env']['prey']['predator_strategy']][config['seed']] = load_records(d)
  seeds = sorted(set(runs['default']) & set(runs['ambush']))
  share = dict(zip(*window(runs['default'][seeds[0]], 'learning/replay_new_frac', s0, 0, WINDOW)))
  out = pathlib.Path(args.out)

  # Figure 1: levels, trunk and all runs.
  rows = len(PANELS) // 2
  fig, axes = plt.subplots(rows, 2, figsize=(10.5, 3.25 * rows), sharex=True, sharey='row')
  print(f'Means over the first {WINDOW // 1000}k steps (trunk: over its last {WINDOW // 1000}k):')
  for ax, (key, ylabel, _) in zip(axes.flat, PANELS):
    x, v = window(trunk, key, s0, -WINDOW, 0)
    if not len(v) and not any(len(window(r, key, s0, 0, WINDOW)[1]) for r in runs['default'].values()):
      ax.set_ylabel(ylabel); ax.text(0.5, 0.5, 'not logged', transform=ax.transAxes, ha='center'); continue
    ax.plot(x / 1e3, v, 'o-', color='#555555', lw=1, ms=3, label='pretrained trunk')
    means = {}
    for strategy, color in COLORS.items():
      curves = [window(r, key, s0, 0, WINDOW) for r in runs[strategy].values()]
      for cx, cv in curves:
        ax.plot(cx / 1e3, cv, 'o-', color=color, lw=0.8, ms=3, alpha=0.45)
      n = min(len(c[0]) for c in curves)
      ax.plot(curves[0][0][:n] / 1e3, np.mean([c[1][:n] for c in curves], 0), color=color,
              lw=2.2, label=f'Default → {strategy.capitalize()} (n={len(curves)}), mean')
      means[strategy] = np.mean([c[1].mean() for c in curves])
    ax.axvline(0, color='black', lw=0.8, ls=':')
    ax.set_ylabel(ylabel)
    style(ax)
    print(f"  {key.split('/')[1]:<22} trunk {v.mean():.4f}  default {means['default']:.4f}  "
          f"ambush {means['ambush']:.4f}  ambush - default {means['ambush'] - means['default']:+.4f}")
  axes[0, 0].set_ylim(bottom=0)
  label_x(axes[-1], share, -WINDOW)
  handles, names = axes[0, 0].get_legend_handles_labels()
  fig.legend(handles, names, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.95))
  fig.suptitle(f'First {WINDOW // 1000}k steps after the predator switch at step {s0:,} '
               '(unsmoothed, one marker per 2k steps)', y=0.99)
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  fig.savefig(out, dpi=200)
  print('Saved', out)

  # Figure 2: paired differences (ambush - default), mean over the seeds.
  bases = [('learning/value_change', 'Value change'), ('learning/value_new_mean', 'Critic mean value'),
           ('learning/disag', 'Disagreement mean'), ('learning/disag_var', 'Disagreement variance')]
  bases = [b for b in bases if any(b[0] in r for r in runs['default'][seeds[0]])]   # skip signals not logged
  fig, axes = plt.subplots(1, len(bases), figsize=(5 * len(bases), 4))
  print(f'Ambush - default, mean over seeds, first {WINDOW // 1000}k steps:')
  for ax, (base, ylabel) in zip(np.atleast_1d(axes), bases):
    for suffix, name, color in [('', 'usual replay batch', '#6a5acd'), ('_recent', 'recent-data batch', '#2a9d8f')]:
      diffs = []
      for seed in seeds:
        x, a = window(runs['ambush'][seed], base + suffix, s0, 0, WINDOW)
        _, d = window(runs['default'][seed], base + suffix, s0, 0, WINDOW)
        diffs.append(a - d)
      mean = np.mean(diffs, 0)
      ax.plot(x / 1e3, mean, 'o-', color=color, lw=2, ms=4, label=name)
      print(f'  {base.split("/")[1]:<18}{name:<20} {mean.mean():+.4f}')
    ax.axhline(0, color='black', lw=0.8, ls=':')
    ax.set_xlabel('Steps since switch (k)')
    ax.set_ylabel(f'{ylabel}: Ambush − Default')
    ax.set_xticks(np.arange(0, WINDOW + 1, 4000) / 1e3)
    style(ax)
  np.atleast_1d(axes)[0].legend(frameon=False, fontsize='small')
  fig.tight_layout()
  paired = out.with_name(out.stem + '_paired' + out.suffix)
  fig.savefig(paired, dpi=200)
  print('Saved', paired)

  # Figure 3: dynamics KL, every log point per run, and its paired differences on 2k bins.
  fig, (ax, axd) = plt.subplots(1, 2, figsize=(10.5, 4.6))
  x, v = window(trunk, KL, s0, -WINDOW, 0)
  ax.plot(x / 1e3, v, 'o-', color='#555555', lw=0.8, ms=2.5, label='pretrained trunk')
  for strategy, color in COLORS.items():
    for seed, r in sorted(runs[strategy].items()):
      x, v = window(r, KL, s0, 0, WINDOW)
      ax.plot(x / 1e3, v, 'o-', color=color, lw=0.8, ms=2.5, alpha=0.6,
              label=f'Default → {strategy.capitalize()} (n={len(runs[strategy])})' if seed == seeds[0] else None)
  ax.axvline(0, color='black', lw=0.8, ls=':')
  ax.set_ylabel('Dynamics KL(posterior ‖ prior), nats, unclamped\nevery log point (≈500 steps, ~3 batches)')
  style(ax)
  label_x([ax], share, -WINDOW)
  means = []
  for seed in seeds:
    cx, a = binned(runs['ambush'][seed], KL, s0, 0, WINDOW)
    _, b = binned(runs['default'][seed], KL, s0, 0, WINDOW)
    axd.plot(cx / 1e3, a - b, 'o-', lw=0.9, ms=3, label=f'seed {seed}')
    means.append(np.nanmean(a - b))
  m = np.mean(means)
  half = stats.t.ppf(0.975, len(means) - 1) * np.std(means, ddof=1) / np.sqrt(len(means))
  axd.axhline(0, color='black', lw=0.8, ls=':')
  axd.set_title(f'{WINDOW // 1000}k-step mean per seed: {np.round(means, 3).tolist()}\n'
                f'mean {m:+.3f}, 95% CI [{m - half:+.3f}, {m + half:+.3f}]', fontsize=8.5, loc='left')
  axd.set_ylabel('Ambush − Default\ndynamics KL, 2k-step bin means')
  axd.legend(fontsize='small', frameon=False)
  style(axd)
  label_x([axd], share, 0)
  handles, names = ax.get_legend_handles_labels()
  fig.legend(handles, names, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
  fig.suptitle(f'Dynamics KL in the first {WINDOW // 1000}k steps after the switch (unsmoothed)', y=0.99)
  fig.tight_layout(rect=(0, 0, 1, 0.9))
  kl = out.with_name(out.stem + '_kl' + out.suffix)
  fig.savefig(kl, dpi=200)
  print('Saved', kl)
  print(f"  dyn_kl per seed {np.round(means, 4).tolist()}  mean {m:+.4f}  CI [{m - half:+.4f}, {m + half:+.4f}]")


if __name__ == '__main__':
  main()
