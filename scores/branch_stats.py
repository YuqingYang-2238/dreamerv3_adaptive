"""Pre-registered statistics for continuation runs branched from one checkpoint.

  python scores/branch_stats.py logdir/cont2_s{1,2,3}_default logdir/cont2_s{1,2,3}_ambush [--out DIR]

For the value-change marker (all replay and recent data), value_delta_std and the
unclamped dynamics KL: per-run window means over 0-30k (primary), 0-10k and
120-150k steps after the branch; the ambush-minus-default difference with complete
separation, an exact permutation p, a Welch CI and the control-noise reference (the
largest change between neighbouring same-size windows within a default run). Also
training survival and T50, the branch-point evaluations, and the pipeline checks.
Writes branch_stats.json and branch_stats.md.
"""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_learning import load_records, series, smooth  # noqa: E402

WINDOWS = {'early': (0, 30000), 'first10k': (0, 10000), 'late': (120000, 150000)}  # lo < x <= hi
QUANTITIES = {  # name: (metric key, series type)
    'value_change': ('learning/value_change', 'marker'),
    'value_change_recent': ('learning/value_change_recent', 'marker'),
    'value_delta_std': ('learning/value_delta_std', 'marker'),
    'dyn_kl': ('train/dyn_kl_mean', 'binned'),
    # Instance 2: Plan2Explore ensemble disagreement (signal 1: mean, signal 2: variance over pairs).
    'disag': ('learning/disag', 'marker'),
    'disag_recent': ('learning/disag_recent', 'marker'),
    'disag_var': ('learning/disag_var', 'marker'),
    'disag_var_recent': ('learning/disag_var_recent', 'marker'),
    'disag_mse': ('learning/disag_mse', 'marker'),
    'disag_mse_recent': ('learning/disag_mse_recent', 'marker'),
    # Visible-step split (Instance 2b): pairs where the prey sees the predator / where it does not.
    'disag_visible_recent': ('learning/disag_visible_recent', 'marker'),
    'disag_hidden_recent': ('learning/disag_hidden_recent', 'marker'),
    # World-model dynamics KL scored at the marker on the same batches (training batches give dyn_kl above).
    'kl_marker': ('learning/dyn_kl', 'marker'),
    'kl_marker_recent': ('learning/dyn_kl_recent', 'marker'),
}
MIN_AGE = 120    # reference age in updates a marker must have
BIN = 2000       # env steps per bin for train/* and training-episode series
FLOOR = 27000    # trunk steps before the branch that define its settled marker floor


def nanmean(values):
  values = np.asarray(values, np.float64)
  return float(values.mean()) if len(values) else float('nan')


def load_run(logdir):
  logdir = pathlib.Path(logdir)
  branch = json.load(open(logdir / 'branch.json'))
  config = json.load(open(logdir / 'config.json'))
  return dict(name=logdir.name, strategy=config['env']['prey']['predator_strategy'],
              seed=config['seed'], s0=branch['step'], source=branch['source'],
              records=load_records(logdir))


def marker_series(records, s0, key):
  rows = [(r['step'] - s0, r[key]) for r in records
          if key in r and r.get('learning/ref_age_updates', 0) >= MIN_AGE]
  if not rows:
    return np.array([]), np.array([])
  x, v = (np.asarray(c, np.float64) for c in zip(*rows))
  return x, v


def binned_series(records, s0, key, width=BIN):
  x, v = series(records, key)
  x = x - s0
  edges = np.arange(0, x.max() + width, width) if len(x) else np.array([0.0])
  centers, means = [], []
  for a, b in zip(edges[:-1], edges[1:]):
    m = (x > a) & (x <= b)
    if m.any():
      centers.append(b - width / 2)
      means.append(v[m].mean())
  return np.asarray(centers), np.asarray(means)


def quantity_series(run, name):
  key, kind = QUANTITIES[name]
  if kind == 'marker':
    return marker_series(run['records'], run['s0'], key)
  return binned_series(run['records'], run['s0'], key)


def window_mean(x, v, lo, hi):
  m = (x > lo) & (x <= hi)
  return nanmean(v[m]), int(m.sum())


def noise_reference(curves, width):
  """Largest |difference| between neighbouring non-overlapping windows of `width`
  points within any one curve (x > 0 only): the change a stable run shows."""
  jumps = []
  for x, v in curves:
    v = v[x > 0]
    means = [v[i:i + width].mean() for i in range(0, len(v) - width + 1, width)]
    jumps += list(np.abs(np.diff(means)))
  return float(max(jumps)) if jumps else float('nan')


def welch_ci(a, d):
  """95% CI of mean(a) - mean(d) (Welch), nan when a group has fewer than 2 values."""
  if len(a) < 2 or len(d) < 2:
    return [float('nan'), float('nan')]
  va, vd = a.var(ddof=1) / len(a), d.var(ddof=1) / len(d)
  se = np.sqrt(va + vd)
  if se == 0:
    return [float(a.mean() - d.mean())] * 2
  df = (va + vd) ** 2 / (va ** 2 / (len(a) - 1) + vd ** 2 / (len(d) - 1))
  half = stats.t.ppf(0.975, df) * se
  return [float(a.mean() - d.mean() - half), float(a.mean() - d.mean() + half)]


def compare(a, d, reference=float('nan')):
  """Ambush (a) vs default (d) per-run values."""
  diff = float(a.mean() - d.mean())
  pooled = np.concatenate([a, d])
  diffs = [pooled[list(i)].mean() - np.delete(pooled, list(i)).mean()
           for i in itertools.combinations(range(len(pooled)), len(a))]
  return dict(
      ambush=a.tolist(), default=d.tolist(), diff=diff,
      separation=bool(a.min() > d.max()),
      p_perm_one_sided=float(np.mean(np.asarray(diffs) >= diff - 1e-12)),
      welch_ci95=welch_ci(a, d), reference=reference,
      above_reference=bool(abs(diff) > reference) if reference == reference else None)


def survival(run):
  """Training survival (goal without capture) per 2k bin, its 10k means, and T50."""
  x, score = series(run['records'], 'episode/success' if any('episode/success' in r for r in run['records']) else 'episode/score')
  x = x - run['s0']
  keep = x > 0
  edges = np.arange(0, 150001, BIN)
  rate = np.array([(score[keep & (x > a) & (x <= b)] == 1).mean() if (keep & (x > a) & (x <= b)).any() else np.nan
                   for a, b in zip(edges[:-1], edges[1:])])
  centers = edges[1:] - BIN / 2
  per10k = [nanmean(rate[(centers > a) & (centers <= a + 10000)][~np.isnan(rate[(centers > a) & (centers <= a + 10000)])])
            for a in range(0, 150000, 10000)]
  level0, level1 = nanmean(rate[centers <= 10000]), nanmean(rate[centers > 120000])
  reached = np.flatnonzero((smooth(np.nan_to_num(rate, nan=level0), 5) >= (level0 + level1) / 2) & (centers > 10000))
  t50 = float(centers[reached[0]]) if len(reached) else None
  return dict(per10k=per10k, first10k=level0, last30k=level1, T50=t50)


def evals(run):
  rows = [r for r in run['records'] if 'eval/success_rate' in r]
  at = lambda r: {k.split('/')[1]: r[k] for k in r if k.startswith('eval/')}
  x = [r['step'] - run['s0'] for r in rows]
  return dict(x=x, success=[r['eval/success_rate'] for r in rows],
              first=at(rows[0]) if rows else {}, last=at(rows[-1]) if rows else {})


def pipeline(run):
  records, s0 = run['records'], run['s0']
  sizes = [r['replay/size'] for r in records if 'replay/size' in r and r['step'] > s0]
  ages = {r['learning/ref_age_steps'] for r in records if 'learning/ref_age_steps' in r and r['step'] > s0}
  markers = sum('learning/value_change' in r and r['step'] > s0 for r in records)
  return dict(replay_first=sizes[0] if sizes else None, replay_expected=s0 - 63,
              replay_ok=bool(sizes) and sizes[0] == s0 - 63, ref_ages=sorted(ages),
              markers=markers, eval_phases=len([r for r in records if 'eval/episodes' in r]))


def trunk_floor(source, s0):
  x, v = marker_series(load_records(source), 0, 'learning/value_change')
  v = v[(x > s0 - FLOOR) & (x <= s0)]
  return dict(mean=nanmean(v), sd=float(v.std()) if len(v) else float('nan'), n=int(len(v)))


def fmt(v, digits=4):
  if v is None:
    return 'none'
  if isinstance(v, bool):
    return 'yes' if v else 'no'
  if isinstance(v, float):
    return 'nan' if v != v else f'{v:.{digits}f}'
  if isinstance(v, list):
    return '[' + ', '.join(fmt(u, digits) for u in v) + ']'
  return str(v)


def markdown(result):
  lines = ['| quantity | window | ambush | default | diff | reference | separated | p | 95% CI |',
           '|---|---|---|---|---|---|---|---|---|']
  for q, windows in result['comparisons'].items():
    for w, c in windows.items():
      if 'diff' not in c:   # 'not logged' or 'incomplete'
        lines.append(f"| {q} | {w} | {c.get('note', '')} | | | | | | |")
        continue
      lines.append(f"| {q} | {w} | {fmt(c['ambush'])} | {fmt(c['default'])} | {fmt(c['diff'])} | "
                   f"{fmt(c['reference'])} ({'above' if c['above_reference'] else 'within'}) | "
                   f"{fmt(c['separation'])} | {c['p_perm_one_sided']:.2f} | {fmt(c['welch_ci95'])} |")
  lines += ['', '| run | strategy | survival first 10k | last 30k | T50 | eval success x=0 -> end | replay ok | marker ages | markers |',
            '|---|---|---|---|---|---|---|---|---|']
  for r in result['runs']:
    lines.append(f"| {r['name']} | {r['strategy']} | {fmt(r['survival']['first10k'], 2)} | "
                 f"{fmt(r['survival']['last30k'], 2)} | {fmt(r['survival']['T50'], 0)} | "
                 f"{fmt(r['evals']['first'].get('success_rate'), 2)} -> {fmt(r['evals']['last'].get('success_rate'), 2)} | "
                 f"{fmt(r['pipeline']['replay_ok'])} | {r['pipeline']['ref_ages']} | {r['pipeline']['markers']} |")
  b = result['branch_point']
  lines += ['', f"branch point (x = 0): success ambush - default = {fmt(b['success']['diff'], 3)} "
            f"CI {fmt(b['success']['welch_ci95'], 3)}; first-10k training survival diff = "
            f"{fmt(b['survival_first10k']['diff'], 3)} CI {fmt(b['survival_first10k']['welch_ci95'], 3)}",
            f"trunk floor (last {FLOOR // 1000}k before the branch): {fmt(result['floor']['mean'])} +- {fmt(result['floor']['sd'])}"]
  return '\n'.join(lines)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('logdirs', nargs='+')
  parser.add_argument('--out', default=None, help='Output directory (default: logdir/branch_stats).')
  args = parser.parse_args()
  runs = [load_run(d) for d in args.logdirs]
  group = lambda s: [r for r in runs if r['strategy'] == s]
  floor = trunk_floor(runs[0]['source'], runs[0]['s0'])
  for r in runs:
    r['survival'], r['evals'], r['pipeline'] = survival(r), evals(r), pipeline(r)
    x, v = quantity_series(r, 'value_change')
    r['artifact_ok'] = bool(abs(window_mean(x, v, *WINDOWS['early'])[0] - floor['mean']) <= 2 * floor['sd']) \
        if r['strategy'] == 'default' else None
  comparisons = {}
  for q in QUANTITIES:
    comparisons[q] = {}
    curves = {r['name']: quantity_series(r, q) for r in runs}
    if any(len(x) == 0 for x, _ in curves.values()):   # signal not logged in these runs
      comparisons[q] = {w: dict(note='not logged') for w in WINDOWS}
      continue
    for w, (lo, hi) in WINDOWS.items():
      means = {n: window_mean(x, v, lo, hi) for n, (x, v) in curves.items()}
      width = max(n for _, n in means.values())
      ref = noise_reference([curves[r['name']] for r in group('default')], width)
      a = np.array([means[r['name']][0] for r in group('ambush')])
      d = np.array([means[r['name']][0] for r in group('default')])
      comparisons[q][w] = compare(a, d, ref) if len(a) and len(d) and not np.isnan(np.concatenate([a, d])).any() \
          else dict(ambush=a.tolist(), default=d.tolist(), note='incomplete')
  a_succ = np.array([r['evals']['first'].get('success_rate', np.nan) for r in group('ambush')])
  d_succ = np.array([r['evals']['first'].get('success_rate', np.nan) for r in group('default')])
  a_surv = np.array([r['survival']['first10k'] for r in group('ambush')])
  d_surv = np.array([r['survival']['first10k'] for r in group('default')])
  branch_point = dict(success=compare(a_succ, d_succ), survival_first10k=compare(a_surv, d_surv))
  result = dict(windows=WINDOWS, floor=floor, comparisons=comparisons, branch_point=branch_point,
                runs=[{k: v for k, v in r.items() if k != 'records'} for r in runs])
  out = pathlib.Path(args.out or pathlib.Path(args.logdirs[0]).parent / 'branch_stats')
  out.mkdir(parents=True, exist_ok=True)
  json.dump(result, open(out / 'branch_stats.json', 'w'), indent=1, default=float)
  text = markdown(result)
  (out / 'branch_stats.md').write_text(text + '\n')
  print(text)
  print('Saved', out / 'branch_stats.json')


if __name__ == '__main__':
  main()
