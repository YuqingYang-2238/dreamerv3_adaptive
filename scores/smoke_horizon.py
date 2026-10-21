"""Smoke test of the MACURA-style imagination cutoff (imag_adaptive), CPU.

Checks: (1) all-accepted mask reproduces the fixed targets and losses; (2) a cut removes
later rewards; (3) the target before a cut bootstraps from the critic; (4) all-rejected
rollouts give finite zero losses; kappa follows the MACURA code; the threshold state is
checkpointed and an older checkpoint loads.

  python scores/smoke_horizon.py
"""
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('CELLWORLD_CACHE', str(ROOT / 'data' / '2105_world'))
sys.path[:0] = [str(ROOT), str(ROOT / 'prey_env')]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import dreamerv3  # noqa: E402
from dreamerv3 import embodied, jaxutils  # noqa: E402
from main_train import make_env  # noqa: E402

FAILS = []


def check(ok, msg):
  print(('ok   ' if ok else 'FAIL ') + msg)
  ok or FAILS.append(msg)


def original_return(rew, disc, value, lam):
  """VFunction.score before this change, verbatim."""
  vals = [value[-1]]
  interm = rew + disc * value[1:] * (1 - lam)
  for t in reversed(range(len(disc))):
    vals.append(interm[t] + disc[t] * lam * vals[-1])
  return jnp.stack(list(reversed(vals))[:-1])


def macura_kappas(bs, W):
  """Threshold sequence of macura.py (rollout_model_and_populate_sac_buffer + caller)."""
  pos, count, est, full, lst, out = 0, 0, 0.0, False, np.empty(W), []
  for b in bs:
    out.append(1 / (count + 1) * b + count / (count + 1) * est)
    lst[pos] = b
    if not full:
      if pos == W - 1:
        full, count = True, W - 1
      else:
        count = pos + 1
    pos = (pos + 1) % W
    est = np.mean(lst) if full else np.mean(lst[0:pos])
  return np.array(out)


def unit_checks():
  rng = np.random.default_rng(0)
  H, B, lam = 15, 64, 0.95
  rew = jnp.asarray(rng.normal(size=(H, B)), jnp.float32)
  disc = jnp.asarray(rng.uniform(0.9, 1.0, (H, B)), jnp.float32)
  value = jnp.asarray(rng.normal(size=(H + 1, B)), jnp.float32)
  ref = original_return(rew, disc, value, lam)
  check(bool((jaxutils.lambda_return(rew, disc, value, lam) == ref).all()), '(1) unmasked return equals the original recursion bitwise')
  ones = jnp.ones((H, B), jnp.float32)
  check(bool((jaxutils.lambda_return(rew, disc, value, lam, ones) == ref).all()), '(1) all-ones mask equals the original recursion bitwise')
  k = 6  # transitions 0..k-1 accepted, k rejected
  acc = (np.arange(H) < k).astype(np.float32)[:, None].repeat(B, 1)
  mnext = jnp.asarray(np.concatenate([acc[1:], np.zeros((1, B), np.float32)]))
  ret = jaxutils.lambda_return(rew, disc, value, lam, mnext)
  rew2 = rew.at[k:].add(100.0)
  value2 = value.at[k + 1:].add(100.0)
  ret2 = jaxutils.lambda_return(rew2, disc, value2, lam, mnext)
  check(bool(jnp.allclose(ret[:k], ret2[:k])), '(2) targets of accepted states ignore rewards and values after the cut')
  check(bool(jnp.allclose(ret[k - 1], rew[k - 1] + disc[k - 1] * value[k])), '(3) last accepted state bootstraps: R = r + disc * v(next)')
  check(bool(jnp.allclose(ret[:k - 1], original_return(rew[:k], disc[:k], value[:k + 1], lam)[:k - 1])),
        '(3) earlier targets equal a lambda-return over the accepted segment with bootstrap at the cut')


def build(**over):
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'disag_measure': True, 'disag_models': 5, 'disag_head.layers': 2,
      'disag_target': ['embed'], 'batch_size': 4, 'batch_length': 16, 'jax.prealloc': False,
      'encoder.mlp_keys': '.*', 'decoder.mlp_keys': '.*', 'encoder.cnn_keys': '$^',
      'decoder.cnn_keys': '$^', 'jax.platform': 'cpu', 'jax.precision': 'float32',
      'jax.metrics_every': 1, 'env.prey.predator_strategy': 'default', **over})
  env = make_env(config)
  agent = dreamerv3.Agent(env.obs_space, env.act_space, embodied.Counter(), config)
  return config, env, agent


def batch_for(env, agent, config):
  random.seed(0)
  np.random.seed(0)
  replay = embodied.replay.Uniform(config.batch_length, 1000)
  driver = embodied.Driver(env)
  driver.on_step(replay.add)
  driver(embodied.RandomAgent(env.act_space).policy, steps=150)
  return next(agent.dataset(replay.dataset, workers=0))


def train(agent, batch, n):
  state, out = None, []
  for _ in range(n):
    _, state, mets = agent.train(batch, state)
    out.append({k: np.asarray(v) for k, v in mets.items()})
  return out


def agent_checks():
  keys = ('actor_opt_loss', 'extr_critic_critic_opt_loss')
  config, env, fixed = build(imag_adaptive=False)
  batch = batch_for(env, fixed, config)
  mf = train(fixed, batch, 3)
  check(all(k in mf[0] for k in keys) and 'imag_kappa' in mf[0], f'metric keys present (fixed logs the cutoff diagnostics too)')
  _, _, wide = build(imag_adaptive=True, imag_xi=1e9)
  mw = train(wide, batch, 3)
  same = all(np.allclose(a[k], b[k], rtol=1e-6, atol=0) for a, b in zip(mf, mw) for k in keys)
  check(same and all(m['imag_frac_full'] == 1 for m in mw),
        f"(1) kappa = inf accepts every step and reproduces the fixed losses: "
        f"{[(float(a['actor_opt_loss']), float(b['actor_opt_loss'])) for a, b in zip(mf, mw)]}")

  _, _, tight = build(imag_adaptive=True, imag_xi=1e-9)
  mt = train(tight, batch, 3)
  finite = all(np.isfinite(m[k]) for m in mt for k in mt[0] if m[k].dtype.kind == 'f')
  check(finite and all(m['imag_frac_zero'] == 1 and m['actor_opt_loss'] == 0 and m['extr_critic_critic_opt_loss'] == 0 for m in mt),
        '(4) kappa ~ 0 rejects every rollout at step 0: losses exactly 0, all metrics finite')
  check(all(np.isfinite(m['actor_opt_grad_norm']) and np.isfinite(m['extr_critic_critic_opt_grad_norm']) for m in mt),
        '(4) gradient norms finite with zero accepted steps')

  W = 3
  _, _, mid = build(imag_adaptive=True, imag_kappa_history=W)
  mm = train(mid, batch, 8)
  bs = np.array([float(m['imag_b']) for m in mm])
  ks = np.array([float(m['imag_kappa']) for m in mm])
  check(np.allclose(ks, macura_kappas(bs, W), rtol=1e-5), f'kappa equals the MACURA code over {len(bs)} rounds with history {W} (wrap-around)')
  m = mm[-1]
  print('     last round: kappa %.4g, b %.4g, accepted mean %.2f, zero %.2f, full %.2f, D by depth %s' % (
      m['imag_kappa'], m['imag_b'], m['imag_accepted_mean'], m['imag_frac_zero'], m['imag_frac_full'],
      [round(float(m[k]), 5) for k in sorted(m) if k.startswith('imag_disag_d')]))

  state = mid.save()
  kk = [k for k in state if '/kappa_' in k]
  check(len(kk) == 2 and int(state[[k for k in kk if 'rounds' in k][0]]) == 8, f'threshold state in checkpoint: {kk}')
  old = {k: v for k, v in state.items() if k not in kk}
  mid.load(old)
  m = train(mid, batch, 1)[0]
  check(float(m['imag_kappa_rounds']) == 1 and np.isclose(m['imag_kappa'], m['imag_b']),
        'checkpoint without the threshold state loads; kappa restarts from b')


if __name__ == '__main__':
  unit_checks()
  agent_checks()
  print('\nALL OK' if not FAILS else f'\n{len(FAILS)} FAILED')
  sys.exit(1 if FAILS else 0)
