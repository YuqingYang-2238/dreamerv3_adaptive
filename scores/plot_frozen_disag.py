"""Figure 2 of the disagreement experiment: episode-level disagreement of the frozen trunk
under the Default vs the Ambush predator (no learning), from main_eval summary.json files.

  python scores/plot_frozen_disag.py $EXP/trunk_s0/eval0_default_s{1,2,3} $EXP/trunk_s0/eval0_ambush_s{1,2,3} \
      --out $EXP/frozen_disag.png

Raw points = episodes, bars = per-seed means; one panel per signal (mean and variance of D over
the episode's steps). Seeds are the replicates, episodes within a seed are not.
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {'default': '#2a78d6', 'ambush': '#eb6834'}
SIGNALS = [('disag_mean', 'Episode mean disagreement $D$'), ('disag_var', 'Episode variance of $D$'),
           ('kl_mean', 'Episode mean world-model KL\n(posterior || prior, nats)')]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('evaldirs', nargs='+', help='main_eval output dirs (summary.json inside)')
  parser.add_argument('--out', default='logdir/frozen_disag.png')
  args = parser.parse_args()
  runs = []
  for d in args.evaldirs:
    s = json.load(open(pathlib.Path(d) / 'summary.json'))
    runs.append(dict(strategy=s['strategy'], seed=s.get('seed', len(runs)),
                     **{k: np.array([e.get(k, np.nan) for e in s['per_episode']], float) for k, _ in SIGNALS}))
  signals = [(k, l) for k, l in SIGNALS if any(np.isfinite(r[k]).any() for r in runs)]   # skip signals not logged
  fig, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 4.2))
  for ax, (key, label) in zip(np.atleast_1d(axes), signals):
    for i, strategy in enumerate(('default', 'ambush')):
      group = [r for r in runs if r['strategy'] == strategy]
      for j, r in enumerate(group):
        x = i * 4 + j
        ax.plot(x + np.random.uniform(-0.25, 0.25, len(r[key])), r[key], 'o',
                color=COLORS[strategy], ms=3, alpha=0.35)
        ax.plot([x - 0.3, x + 0.3], [r[key].mean()] * 2, color='black', lw=2)
      means = [r[key].mean() for r in group]
      if means:
        ax.text(i * 4 + (len(group) - 1) / 2, ax.get_ylim()[1] if False else np.nanmax([r[key].max() for r in group]),
                f'{strategy}: {np.mean(means):.3g} ± {np.std(means, ddof=1) if len(means) > 1 else 0:.2g} (seeds)',
                ha='center', va='bottom', fontsize=9, color=COLORS[strategy])
    ticks = [i * 4 + j for i in range(2) for j in range(max(1, len(runs) // 2))]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{s[0].upper()}{j + 1}" for s in ('default', 'ambush') for j in range(len(ticks) // 2)])
    ax.set_ylabel(label)
    ax.set_xlabel('frozen trunk vs predator, one column per seed (D = Default, A = Ambush)')
    ax.spines[['top', 'right']].set_visible(False)
  fig.suptitle('Frozen agent, no learning: signals per evaluation episode')
  fig.tight_layout()
  pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=150)
  print('->', args.out)


if __name__ == '__main__':
  main()
