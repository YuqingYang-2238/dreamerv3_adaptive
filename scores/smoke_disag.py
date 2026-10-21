"""Smoke test of the measured-only Plan2Explore ensemble (disag_measure), CPU.

Checks: the ensemble loss is finite and falls on repeated identical batches; disag_measure
returns finite values with D >= 0; policy returns log_disag of shape [B] in all modes; the
checkpoint holds the ensemble and its optimizer; with disag_measure off nothing new appears.

  python scores/smoke_disag.py
"""
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('CELLWORLD_CACHE', str(ROOT / 'data' / '2105_world'))
sys.path[:0] = [str(ROOT), str(ROOT / 'prey_env')]

import jax  # noqa: E402
import numpy as np  # noqa: E402

import dreamerv3  # noqa: E402
from dreamerv3 import embodied  # noqa: E402
from main_train import make_env  # noqa: E402


def build(disag_measure):
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'disag_measure': disag_measure, 'disag_models': 5, 'disag_head.layers': 2,
      'disag_target': ['embed'], 'batch_size': 4, 'batch_length': 16, 'jax.prealloc': False,
      'encoder.mlp_keys': '.*', 'decoder.mlp_keys': '.*', 'encoder.cnn_keys': '$^',
      'decoder.cnn_keys': '$^', 'jax.platform': 'cpu', 'jax.precision': 'float32',
      'env.prey.predator_strategy': 'default'})
  env = make_env(config)
  agent = dreamerv3.Agent(env.obs_space, env.act_space, embodied.Counter(), config)
  replay = embodied.replay.Uniform(config.batch_length, 1000)
  driver = embodied.Driver(env)
  driver.on_step(replay.add)
  driver(embodied.RandomAgent(env.act_space).policy, steps=150)
  batch = next(agent.dataset(replay.dataset, workers=0))
  return config, env, agent, batch


def main():
  random.seed(0)
  np.random.seed(0)
  failures = []
  check = lambda ok, msg: (print(('ok   ' if ok else 'FAIL ') + msg), failures.append(msg) if not ok else None)

  config, env, agent, batch = build(True)
  losses = []
  state = None
  for _ in range(25):   # metrics come back every jax.metrics_every (10) updates
    _, state, mets = agent.train(batch, state)
    if 'disag_opt_loss' in mets:
      losses.append(float(mets['disag_opt_loss']))
  check(len(losses) >= 2 and all(np.isfinite(losses)), f'ensemble loss finite: {np.round(losses, 4)}')
  check(len(losses) >= 2 and losses[-1] < losses[0], 'ensemble loss falls on repeated identical batches')
  m = agent.disag_measure(batch)
  check(all(np.isfinite(list(m.values()))), f'disag_measure finite: { {k: round(v, 5) for k, v in m.items()} }')
  check(m['disag'] >= 0 and m['disag_var'] >= 0 and m['disag_mse'] >= 0, 'disag, disag_var, disag_mse >= 0')
  vis = np.asarray(jax.device_get(batch['vector']))[..., 5] >= 0
  lo, hi = sorted([m['disag_visible'], m['disag_hidden']])
  check(all(np.isfinite([m['disag_visible'], m['disag_hidden'], m['disag_mse_visible']])),
        f"visible/hidden split finite: visible {m['disag_visible']:.5f} hidden {m['disag_hidden']:.5f} (visible share {vis.mean():.0%})")
  check(not (0 < vis.mean() < 1) or lo <= m['disag'] + 1e-9 and m['disag'] <= hi + 1e-9,
        'overall disagreement lies between the visible and hidden means')
  k = agent.kl_measure(batch)
  check(np.isfinite(k['dyn_kl_visible']), f"dyn_kl_visible finite: {k['dyn_kl_visible']:.4f}")
  obs = env.step({'action': np.zeros((1,) + env.act_space['action'].shape, np.float32), 'reset': np.ones(1, bool)})
  k = agent.kl_measure(batch)
  check(np.isfinite(k['dyn_kl']) and k['dyn_kl'] >= 0, f"kl_measure finite and >= 0: {k['dyn_kl']:.4f}")
  for mode in ('eval', 'explore', 'train'):
    outs, _ = agent.policy(obs, None, mode=mode)
    check('log_kl' in outs and outs['log_kl'].shape == (1,) and (outs['log_kl'] >= 0).all(),
          f'policy returns log_kl of shape [B] in mode {mode}: {outs.get("log_kl")}')
    check('log_disag_err' in outs and outs['log_disag_err'].shape == (1,) and (outs['log_disag_err'] >= 0).all(),
          f'policy returns log_disag_err of shape [B] in mode {mode}: {outs.get("log_disag_err")}')
    check('log_disag' in outs and outs['log_disag'].shape == (1,) and np.isfinite(outs['log_disag']).all(),
          f'policy returns finite log_disag of shape [B] in mode {mode}: {outs.get("log_disag")}')
  keys = list(agent.save())
  heads = [k for k in keys if k.startswith('agent/disag/disag')]
  check(len(heads) > 0, f'checkpoint has ensemble head parameters ({len(heads)} entries)')
  check(any('disag_opt' in k for k in keys), 'checkpoint has the ensemble optimizer state')
  check(agent.measures_disag, 'agent.measures_disag is True')

  config, env, agent, batch = build(False)
  _, _, mets = agent.train(batch, None)
  outs, _ = agent.policy(obs, None, mode='train')
  check(not any(k.startswith('disag_') for k in mets), 'no disag metrics when disag_measure is off')
  check('log_disag' not in outs, 'no log_disag output when disag_measure is off')
  check(not any('disag' in k for k in agent.save()), 'no ensemble parameters when disag_measure is off')
  check(not agent.measures_disag, 'agent.measures_disag is False')

  print(f'\n{len(failures)} failures' if failures else '\nall checks passed')
  sys.exit(1 if failures else 0)


if __name__ == '__main__':
  main()
