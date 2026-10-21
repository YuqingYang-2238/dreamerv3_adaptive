"""Learning curve of a pretraining run that spans several logdirs (e.g. a warm start).

  python scores/plot_pretrain.py logdir/pretrain_nopred_s0 logdir/pretrain_warm_s0 \
      --phases "0:80000:no predator, -0.02/step" "80000:123400:decoupled predator, -0.02/step" \
               "123400:200000:decoupled predator, no time cost" [--out PNG]

Logdirs are given in training order; for steps logged in several of them the later
logdir wins. Panels: training survival and goal rate (2k bins) with evaluation
success, captures and episode length, the value-change marker, and the world
model's dynamics KL before the free-bits clip.
"""
import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_learning import load_records, series, smooth, style  # noqa: E402

BIN = 2000
SHADES = ['#f2f2f2', '#fde9dc', '#e3eefa']
BLUE, ORANGE, GREEN, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#666666'


def merged(logdirs):
  bystep = {}
  for logdir in logdirs:
    for record in load_records(logdir):
      bystep[record['step']] = record   # later logdirs override overlapping steps
  return [bystep[s] for s in sorted(bystep)]


def binned(x, y, end, width=BIN):
  edges = np.arange(0, end + width, width)
  mean = [y[(x > a) & (x <= b)].mean() if ((x > a) & (x <= b)).any() else np.nan
          for a, b in zip(edges[:-1], edges[1:])]
  return edges[1:] - width / 2, np.array(mean)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('logdirs', nargs='+')
  parser.add_argument('--phases', nargs='*', default=[], help='start:end:label, shaded in order')
  parser.add_argument('--out', default='logdir/pretrain_curve.png')
  args = parser.parse_args()
  records = merged(args.logdirs)
  end = records[-1]['step']
  episodes = [r for r in records if 'episode/length' in r]
  x = np.array([r['step'] for r in episodes], float)
  length = np.array([r['episode/length'] for r in episodes], float)
  goal = (length < 300).astype(float)
  success = np.array([r.get('episode/success', r['episode/score'] == 1) for r in episodes], float)
  cx, captures = series(records, 'stats/sum_log_captured')
  ex, eval_success = series(records, 'eval/success_rate')
  vx, vc = series(records, 'learning/disag')          # ensemble disagreement, signal 1
  mx, mse = series(records, 'learning/disag_mse')     # ensemble-mean prediction error
  wx, dv = series(records, 'learning/disag_var')      # signal 2
  kx, kl = series(records, 'train/dyn_kl_mean')

  fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
  ax = axes.flat
  c, v = binned(x, success, end); ax[0].plot(c / 1e3, v, color=BLUE, lw=2, label='survival (goal, no capture)')
  c, v = binned(x, goal, end); ax[0].plot(c / 1e3, v, color=GREEN, lw=1.5, label='goal reached')
  ax[0].plot(ex / 1e3, eval_success, 'o', color=ORANGE, ms=5, label='eval success')
  ax[0].set_ylim(-0.02, 1.02); ax[0].set_ylabel('Rate per 2k steps')
  c, v = binned(cx, captures, end); ax[1].plot(c / 1e3, v, color=ORANGE, lw=2)
  ax[1].set_ylabel('Captures per episode')
  c, v = binned(x, length, end); ax[2].plot(c / 1e3, v, color=GRAY, lw=2)
  ax[2].set_ylabel('Episode length (steps)'); ax[2].set_ylim(0, 310)
  ax[3].plot(vx / 1e3, vc, 'o', color=BLUE, ms=2.5, alpha=0.4)
  ax[3].plot(vx / 1e3, smooth(vc, 5), color=BLUE, lw=2, label='disagreement $D$ (mean)')
  ax[3].set_ylabel('Ensemble disagreement $D$ (mean over batch)'); ax[3].set_ylim(bottom=0)
  twin = ax[3].twinx()
  twin.plot(mx / 1e3, smooth(mse, 5), color=ORANGE, lw=1.5, label='prediction MSE')
  twin.set_ylabel('Ensemble-mean prediction MSE', color=ORANGE); twin.set_ylim(bottom=0)
  c, v = binned(kx, kl, end); ax[4].plot(c / 1e3, v, color=GRAY, lw=2)
  ax[4].set_ylabel('Dynamics KL before the clip (nats)')
  ax[5].plot(wx / 1e3, dv, 'o', color=BLUE, ms=2.5, alpha=0.4)
  ax[5].plot(wx / 1e3, smooth(dv, 5), color=BLUE, lw=2)
  ax[5].set_ylabel('Variance of $D$ over batch pairs'); ax[5].set_ylim(bottom=0)
  for a in ax:
    for (spec, shade) in zip(args.phases, SHADES):
      lo, hi, _ = spec.split(':', 2)
      a.axvspan(float(lo) / 1e3, float(hi) / 1e3, color=shade, lw=0, zorder=-1)
    style(a)
  for a in axes[1]:
    a.set_xlabel('Env steps (k)')
  ax[0].legend(fontsize='small', frameon=False, loc='lower right')
  handles = [plt.Rectangle((0, 0), 1, 1, color=s) for s in SHADES[:len(args.phases)]]
  fig.legend(handles, [p.split(':', 2)[2] for p in args.phases], loc='upper center', ncol=3,
             frameon=False, fontsize='medium')
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=200)
  print('Saved', args.out)


if __name__ == '__main__':
  main()
