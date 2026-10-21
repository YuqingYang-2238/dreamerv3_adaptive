"""Plot the 'learning/value_change' marker from one or more DreamerV3 logdirs.
"""

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


METRIC = 'learning/value_change'

COLORS = (
    '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
    '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
)


def load(logdir):
  """Return sorted (steps, values) arrays for METRIC, or (None, reason)."""
  filename = pathlib.Path(logdir) / 'metrics.jsonl'
  if not filename.exists():
    return None, f'no metrics.jsonl in {logdir}'
  steps, values = [], []
  with open(filename) as f:
    for lineno, line in enumerate(f, 1):
      line = line.strip()
      if not line:
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError:
        print(f'Skipping unparsable line {lineno} in {filename}')
        continue
      if METRIC in record and 'step' in record:
        steps.append(record['step'])
        values.append(record[METRIC])
  if not steps:
    return None, f'no {METRIC!r} entries in {filename}'
  steps = np.asarray(steps, dtype=np.float64)
  values = np.asarray(values, dtype=np.float64)
  order = np.argsort(steps, kind='stable')
  return (steps[order], values[order]), None


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('logdirs', nargs='+', help='One or more logdir paths.')
  parser.add_argument(
      '--logy', action='store_true', help='Use a log scale on the y axis.')
  parser.add_argument(
      '--out', default=None,
      help='Output PNG path (default: <first logdir>/value_change.png).')
  args = parser.parse_args()

  series = []
  for logdir in args.logdirs:
    data, reason = load(logdir)
    if data is None:
      print(f'Skipping {logdir}: {reason}')
      continue
    series.append((logdir, *data))
    print(f'{logdir}: {len(data[0])} entries')

  if not series:
    print(f'Nothing to plot: none of the logdirs contain {METRIC!r}.')
    return 1

  fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
  ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
  if not args.logy:
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
  ax.grid(which='both', color='#eeeeee')
  ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(1))
  ax.tick_params(which='minor', length=0)

  for i, (logdir, steps, values) in enumerate(series):
    finite = np.isfinite(values)
    if args.logy:
      finite &= values > 0
    ax.plot(
        steps[finite], values[finite], marker='o', markersize=3,
        label=pathlib.Path(logdir).name, zorder=1000 - i,
        color=COLORS[i % len(COLORS)])

  if args.logy:
    ax.set_yscale('log')
  ax.set_xlabel('Env steps')
  ax.set_ylabel(r'Mean $|V_k - V_{k-\Delta}|$')
  ax.set_title('Critic value change')
  ax.tick_params(
      axis='both', which='major', labelsize='small', pad=1, length=1)
  ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
  if len(series) > 1:
    leg = ax.legend(
        fontsize='small', numpoints=1, labelspacing=0.2, handlelength=1.5,
        handletextpad=0.5)
    leg.get_frame().set_edgecolor('white')
  fig.tight_layout()

  out = args.out
  if out is None:
    out = pathlib.Path(series[0][0]) / 'value_change.png'
  out = pathlib.Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=300)
  print('Saved', out)
  return 0


if __name__ == '__main__':
  sys.exit(main())
