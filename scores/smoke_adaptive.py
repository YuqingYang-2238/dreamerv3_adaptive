"""Smoke check of the adaptive-imagination controller on the local CPU.

  python scores/smoke_adaptive.py [--platform gpu] [--trunk logdir/baseline_21_05_seed0] [--out logdir/smoke_adapt]

Branches the trunk twice and trains a few thousand steps against the ambush
predator: one branch with c = 1e9 (p_extra stays 0, baseline budget) and one
with c = 1e-6 (p_extra = 1 after the first signal measurement, doubled budget),
then resumes the second one from its checkpoint. Checks, from metrics.jsonl and
the checkpoints: p=0 uses N0 trajectories and no extra updates; p=1 uses 2 N0
with the same horizon and the two repeats agree only as often as independent
draws should; optimizer step counts stay one per update for WM, actor and
critic; losses are finite; the resume restores the controller (counters, p,
RNG stream) and the critic snapshot (reference age 125 updates). Prints a
pass/fail table and exits non-zero on failure. Runs the two branches
concurrently; ~30-45 min at 3-4 fps.
"""
import argparse
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import main_branch  # noqa: E402

COMMON = [
    '--seed', '1', '--env.prey.predator_strategy', 'ambush',
    '--run.eval_eps', '2', '--run.eval_burn_in', '1', '--run.log_every', '10',
    '--run.adaptive', 'True']
N0, HORIZON = 1024, 15
MEASURE = 144000  # first signal measurement after branching at 141,300


def records(logdir):
  rows = [json.loads(l) for l in open(logdir / 'metrics.jsonl') if l.strip()]
  return rows


def launch(python, logdir, steps, c, log, platform):
  cmd = [python, 'main_train.py', '--logdir', str(logdir), '--run.steps', str(steps),
         '--run.adaptive_c', str(c), '--jax.platform', platform, *COMMON]
  env = dict(os.environ, MPLBACKEND='Agg')
  return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log, 'a'), stderr=subprocess.STDOUT)


def wait(procs):
  for p in procs:
    assert p.wait() == 0, f'training process failed, see its log ({p.args[3]})'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--trunk', default='logdir/baseline_21_05_seed0')
  parser.add_argument('--out', default='logdir/smoke_adapt')
  parser.add_argument('--python', default=sys.executable)
  parser.add_argument('--platform', default='cpu', help='jax platform (gpu on Lambda: ~5 min)')
  parser.add_argument('--skip-train', action='store_true', help='only run the checks')
  args = parser.parse_args()
  p0, p1 = pathlib.Path(args.out + '_p0'), pathlib.Path(args.out + '_p1')
  if not args.skip_train:
    for d in (p0, p1):
      shutil.rmtree(d, ignore_errors=True)
      main_branch.main(args.trunk, str(d))
    procs = [launch(args.python, p0, 145000, 1e9, p0 / 'train.log', args.platform),
             launch(args.python, p1, 145000, 1e-6, p1 / 'train.log', args.platform)]
    wait(procs)
    ckpt1 = pickle.load(open(p1 / 'checkpoint.ckpt', 'rb'))
    pickle.dump({k: ckpt1[k] for k in ('controller', 'value_ref', 'step', 'updates')},
                open(p1 / 'ckpt_segment1.pkl', 'wb'))
    wait([launch(args.python, p1, 146000, 1e-6, p1 / 'train_resume.log', args.platform)])

  results = []
  check = lambda name, ok, detail='': results.append((name, bool(ok), detail))
  r0, r1 = records(p0), records(p1)
  get = lambda rows, key, lo=0: np.array([r[key] for r in rows if key in r and r['step'] > lo])
  fin = lambda x: np.all(np.isfinite(x))

  # 1. p_extra = 0 -> baseline budget.
  t0 = get(r0, 'train/imag_trajs')
  check('p0: train/imag_trajs == 1024 in every record', len(t0) and np.all(t0 == N0), f'{set(t0)}')
  check('p0: adaptive/rollouts == 1 in every record', np.all(get(r0, 'adaptive/rollouts') == 1))
  check('p0: adaptive/updates_extra == 0 at the end', get(r0, 'adaptive/updates_extra')[-1] == 0)
  check('p0: adaptive/p_extra == 0 after the measurement', np.all(get(r0, 'adaptive/p_extra', MEASURE - 1) == 0))
  u0 = get(r0, 'adaptive/u_raw', MEASURE - 1)
  check('p0: signal u_raw measured and finite', len(u0) and fin(u0), f'u={u0[:1]}')

  # 2. p_extra = 1 -> doubled budget, same horizon, independent repeats.
  after = MEASURE + 200  # records whose whole log window lies after the measurement
  t1 = get(r1, 'train/imag_trajs', after)
  check('p1: adaptive/p_extra == 1 after the measurement', np.all(get(r1, 'adaptive/p_extra', MEASURE - 1) == 1))
  check('p1: train/imag_trajs == 2048 after the measurement', len(t1) and np.all(t1 == 2 * N0), f'{set(t1)}')
  check('p1: train/imag_trajs == 1024 before the measurement', np.all(get(r1, 'train/imag_trajs')[:3] == N0))
  check('p1: adaptive/rollouts == 2 after the measurement', np.all(get(r1, 'adaptive/rollouts', after) == 2))
  h = np.concatenate([get(r0, 'train/imag_horizon'), get(r1, 'train/imag_horizon')])
  check('horizon == 15 in every record of both runs', np.all(h == HORIZON), f'{set(h)}')
  m, e = get(r1, 'train/imag_half_match', after), get(r1, 'train/imag_half_match_expected', after)
  check('p1: repeats agree like independent draws (|match - expected| < 0.03, match < 1)',
        len(m) and np.all(np.abs(m - e) < 0.03) and np.all(m < 1),
        f'match {m.mean():.3f} expected {e.mean():.3f}' if len(m) else 'no records')
  ex = get(r1, 'adaptive/extra_frac_interval')
  check('p1: realized extra fraction after the measurement == 1', len(ex) >= 2 and ex[-1] == 1.0, f'{ex}')

  # 3. One optimizer step per update for WM, actor and critic (exact, from the
  # checkpoint's optimizer step counters vs. the controller's update count).
  for name, rows, d in (('p0', r0, p0), ('p1', r1, p1)):
    ck = pickle.load(open(d / 'checkpoint.ckpt', 'rb'))
    steps = [int(ck['agent'][f'agent/{k}/step/value']) for k in (
        'wm/model_opt', 'task_behavior/ac/actor_opt', 'task_behavior/critic/critic_opt')]
    expected = 8763 + ck['controller']['state']['updates']  # trunk checkpoint: 8763 updates
    check(f'{name}: WM/actor/critic optimizer steps == trunk + controller updates',
          all(st == expected for st in steps), f'{steps} vs {expected}')
    keys = sorted({k for r in rows for k in r if k.startswith('train/') and 'loss' in k})
    bad = [k for k in keys if not fin(get(rows, k))]
    check(f'{name}: all train/*loss* finite ({len(keys)} keys)', keys and not bad, f'{bad}')

  # 4. Resume restores controller and critic-history state.
  seg1 = pickle.load(open(p1 / 'ckpt_segment1.pkl', 'rb'))
  ckpt2 = pickle.load(open(p1 / 'checkpoint.ckpt', 'rb'))
  c1, c2 = seg1['controller'], ckpt2['controller']
  check('resume: checkpoint holds controller and value_ref', 'controller' in ckpt2 and ckpt2['value_ref'] is not None)
  check('resume: controller counters continue (updates grew, none reset)',
        c2['state']['updates'] > c1['state']['updates'] and c2['state']['updates_extra'] > c1['state']['updates_extra']
        and c2['state']['measurements'] >= c1['state']['measurements'] and c2['state']['wall_seconds'] > c1['state']['wall_seconds'],
        f"{c1['state']['updates']} -> {c2['state']['updates']}")
  rng = np.random.default_rng(0)
  rng.bit_generator.state = c1['rng']
  rng.random(c2['state']['updates'] - c1['state']['updates'])
  check('resume: controller RNG stream continued exactly (no re-seed)', rng.bit_generator.state == c2['rng'])
  res = [r for r in r1 if r['step'] > int(seg1['step']) and 'adaptive/p_extra' in r]
  check('resume: p_extra == 1 right after resume (restored, before any new measurement)',
        res and res[0]['adaptive/p_extra'] == 1.0 and res[0]['adaptive/updates'] >= c1['state']['updates'])
  age = [(r['learning/ref_age_updates'], r['learning/ref_age_steps']) for r in r1 if r['step'] == 146000 and 'learning/ref_age_updates' in r]
  # 126: the train-ratio scheduler fires one extra update on its first call after
  # any (re)start (upstream behaviour, same for both conditions at the branch).
  check('resume: marker at 146000 uses the restored snapshot (age 125-126 updates / 2000 steps)',
        age and age[0][0] in (125.0, 126.0) and age[0][1] == 2000.0, f'{age}')
  check('resume: value_ref snapshot at 146000 taken after resume', ckpt2['value_ref']['step'] == 146000, f"{ckpt2['value_ref']['step']}")

  width = max(len(n) for n, _, _ in results)
  for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
  failed = [n for n, ok, _ in results if not ok]
  print(f'\n{len(results) - len(failed)}/{len(results)} checks passed')
  sys.exit(1 if failed else 0)


if __name__ == '__main__':
  main()
