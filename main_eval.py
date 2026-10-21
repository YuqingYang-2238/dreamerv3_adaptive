"""Evaluates a trained DreamerV3 agent in the prey/predator env.

Rebuilds the agent from a run directory (config.json + checkpoint.ckpt), runs
it against the chosen predator strategy for N episodes, records every episode
as an MP4 (prey_env/video.py), and writes eval metrics in the same form as the
training-time evaluations (dreamerv3/embodied/run/train.py::run_eval).

  python main_eval.py --logdir logdir/baseline_21_05_seed0 --strategy ambush --episodes 10
  python main_eval.py --logdir logdir/baseline_21_05_seed0 --strategy default --episodes 20 --no-video
  python main_eval.py --logdir logdir/ambush_21_05_seed0 --checkpoint path/to/other.ckpt --out logdir/compare

Outputs in --out (default <logdir>/eval_<strategy>): episode_NNN.mp4,
all_episodes.mp4 (all of them back to back), summary.json (metrics + per-episode records), trajs.npz (eval_trajs layout).
Actions are sampled from the actor (DreamerV3 eval mode), as in training evals.
--burn-in N runs N unscored episodes first, so the ambush predator's zone window
is filled the same way as in the training-time evaluations.

From another script:
  from main_eval import load_agent, evaluate
  agent, env, raw = load_agent('logdir/baseline_21_05_seed0', strategy='ambush')
  summary = evaluate(agent, env, raw, episodes=10, out='logdir/x')
"""
import argparse
import json
import os
import pathlib
import random
import sys

# World data for the fixed Mice world, read by cellworld instead of GitHub.
os.environ.setdefault(
    'CELLWORLD_CACHE', str(pathlib.Path(__file__).parent / 'data' / '2105_world'))
os.environ.setdefault('MPLBACKEND', 'Agg')  # headless rendering for the videos

import numpy as np

from main_train import ENV_KWARGS
from prey_env import gym_Environment_D as Environment
from prey_env.embodied_env import PreyFromGym
from prey_env.video import EpisodeRecorder, concatenate_videos


def episode_disag(ep):
  """Mean and variance over steps of the per-step ensemble disagreement (nan without it)."""
  if 'log_disag' not in ep or len(ep['log_disag']) < 3:
    return float('nan'), float('nan')
  d = np.asarray(ep['log_disag'], np.float64).reshape(-1)[:-1]   # the driver zeroes the last step
  return float(d.mean()), float(d.var(ddof=1))


def episode_kl(ep, key='log_kl'):
  """Mean over steps of a per-step policy output (default: world-model KL), first and last
  step excluded (no valid previous step / zeroed by the driver). nan when not logged."""
  if key not in ep or len(ep[key]) < 3:
    return float('nan')
  return float(np.asarray(ep[key], np.float64).reshape(-1)[1:-1].mean())


def step_values(ep, key):
  """Per-step values of a policy output, nan when it was not logged."""
  return np.asarray(ep.get(key, np.full(len(ep['reward']), np.nan)), np.float64).reshape(-1)


def load_agent(logdir, strategy='default', checkpoint=None, platform='cpu', seed=0, predator_surge_speed=1.0):
  """Rebuilds the trained agent and an env with the given predator strategy.

  Returns (agent, env, raw): `env` is the DreamerV3-wrapped batch env the agent
  acts in, `raw` the underlying prey_env Environment (for videos and ambush state).
  """
  import dreamerv3
  from dreamerv3 import embodied
  logdir = pathlib.Path(logdir)
  saved = json.load(open(logdir / 'config.json'))
  saved.get('run', {}).pop('from_checkpoint', None)  # option removed after older runs were saved
  config = embodied.Config(dreamerv3.configs['defaults']).update(saved).update({
      'jax.platform': platform, 'jax.prealloc': False,
      'env.prey.predator_strategy': strategy, 'env.prey.predator_surge_speed': predator_surge_speed})
  random.seed(seed)      # env: zone sampling, spawn, wander
  np.random.seed(seed)
  raw = Environment(**ENV_KWARGS, predator_strategy=strategy, predator_surge_speed=predator_surge_speed)
  env = dreamerv3.wrap_env(PreyFromGym(raw, obs_key='vector'), config)
  env = embodied.BatchEnv([env], parallel=False)
  agent = dreamerv3.Agent(env.obs_space, env.act_space, embodied.Counter(), config)
  ckpt = embodied.Checkpoint()
  ckpt.agent = agent
  ckpt.load(str(checkpoint or logdir / 'checkpoint.ckpt'), keys=['agent'])
  return agent, env, raw


def evaluate(agent, env, raw, episodes, out, video=True, mode='eval', burn_in=0):
  """Runs `burn_in` unscored then `episodes` scored episodes; writes videos/summary/trajs into `out`.
  Videos: Mice look from the prey's point of view, one frame per physics tick (prey_env/video.py)."""
  from dreamerv3 import embodied
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  recorder = EpisodeRecorder(raw) if video else None
  records, trajs = [], []
  scoring = [burn_in == 0]

  def on_step(tran, worker):
    if recorder:
      recorder.active = scoring[0]
      scoring[0] and tran['is_first'] and recorder.grab()   # frame of the start pose

  def on_episode(ep, worker):
    if not scoring[0]:
      return
    i = len(records)
    goal = bool(ep['log_goal'].any())
    rec = dict(
        episode=i,
        length=int(len(ep['reward']) - 1),
        score=float(ep['reward'].astype(np.float64).sum()),
        goal=goal,
        success=bool(ep['log_success'].any()),  # goal reached, never captured
        timeout=bool(ep['log_timeout'].any()) and not goal,
        captures=int(ep['log_captured'].sum()),
        zone=int(ep['log_zone'].max()),         # -1 for the default predator
        surges=int(ep['log_surge'].sum()),
        zone_weights=[int(w) for w in getattr(raw, 'ambush_zone_weights', [])],
        zones_entered=[k for k, e in enumerate(getattr(raw, 'zone_entered', [])) if e])
    rec['disag_mean'], rec['disag_var'] = episode_disag(ep)   # nan without the ensemble
    rec['kl_mean'] = episode_kl(ep)
    rec['disag_err_mean'] = episode_kl(ep, 'log_disag_err')   # ensemble prediction error
    if recorder:
      rec['video'] = str(out / f'episode_{i:03d}.mp4')
      rec['frames'] = recorder.save(rec['video'])
    records.append(rec)
    vector = ep['vector']
    trajs.append(dict(
        prey_xy=vector[:, 0:2], pred_visible=vector[:, 5] >= 0.0,
        pred_xy=np.stack([ep['log_pred_x'].reshape(-1), ep['log_pred_y'].reshape(-1)], -1),
        captured=ep['log_captured'], reward=ep['reward'],
        kl=step_values(ep, 'log_kl'), disag_err=step_values(ep, 'log_disag_err'),
        disag=np.asarray(ep.get('log_disag', np.full(len(ep['reward']), np.nan)), np.float64).reshape(-1)))
    outcome = ('success' if rec['success'] else 'goal (captured)' if goal
               else 'timeout' if rec['timeout'] else 'ended')
    print(f"episode {i}: {outcome} after {rec['length']} steps, return {rec['score']:.1f}, "
          f"captures {rec['captures']}, zone {rec['zone']} (weights {rec['zone_weights']}), "
          f"surges {rec['surges']}, zones entered {rec['zones_entered']}", flush=True)

  driver = embodied.Driver(env)
  driver.on_step(on_step)
  driver.on_episode(on_episode)
  policy = lambda *a: agent.policy(*a, mode=mode)
  if burn_in:
    driver(policy, episodes=burn_in)
    scoring[0] = True
  driver(policy, episodes=episodes)

  stack = lambda key: np.array([r[key] for r in records])
  goal, success, timeout = stack('goal'), stack('success'), stack('timeout')
  captures, score, length, surges = stack('captures'), stack('score'), stack('length'), stack('surges')
  summary = dict(
      episodes=len(records), burn_in=burn_in, strategy=raw.predator_strategy, mode=mode,
      success_rate=float(success.mean()), goal_rate=float(goal.mean()),
      timeout_rate=float(timeout.mean()), capture_rate=float((captures > 0).mean()),
      captures_mean=float(captures.mean()), surge_rate=float((surges > 0).mean()),
      surges_mean=float(surges.mean()), return_mean=float(score.mean()),
      return_std=float(score.std()), length_mean=float(length.mean()),
      disag_mean=float(np.nanmean(stack('disag_mean'))), disag_var=float(np.nanmean(stack('disag_var'))),
      kl_mean=float(np.nanmean(stack('kl_mean'))),
      disag_err_mean=float(np.nanmean(stack('disag_err_mean'))),
      env=dict(ENV_KWARGS, predator_strategy=raw.predator_strategy),
      ambush=None if raw.predator_strategy != 'ambush' else dict(
          zones=[c.id for c in raw.ambush_zones], attack_radius=raw.attack_radius,
          history_trials=raw.zone_history.maxlen, predator_surge_speed=raw.predator_surge_speed),
      per_episode=records)
  if recorder and records:
    # One file with all episodes back to back, easier to review than N files.
    concatenate_videos([r['video'] for r in records], out / 'all_episodes.mp4')
    summary['video'] = str(out / 'all_episodes.mp4')
  json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
  concat = lambda key: np.concatenate([t[key] for t in trajs])
  np.savez_compressed(
      str(out / 'trajs.npz'),
      ep_offsets=np.cumsum([0] + [len(t['reward']) for t in trajs]),
      ep_goal=goal, ep_success=success, ep_timeout=timeout, ep_captures=captures,
      ep_return=score, ep_length=length, ep_zone=stack('zone'), ep_surges=surges,
      ep_disag_mean=stack('disag_mean'), ep_disag_var=stack('disag_var'), ep_kl_mean=stack('kl_mean'),
      ep_disag_err_mean=stack('disag_err_mean'),
      prey_xy=concat('prey_xy'), pred_xy=concat('pred_xy'), pred_visible=concat('pred_visible'),
      captured=concat('captured'), reward=concat('reward'), disag=concat('disag'), kl=concat('kl'),
      disag_err=concat('disag_err'))
  print(f"\n{raw.predator_strategy} predator, {len(records)} episodes: "
        f"success {summary['success_rate']:.2f}, goal {summary['goal_rate']:.2f}, "
        f"capture_rate {summary['capture_rate']:.2f}, captures/ep {summary['captures_mean']:.1f}, "
        f"surge_rate {summary['surge_rate']:.2f}, surges/ep {summary['surges_mean']:.1f}, "
        f"return {summary['return_mean']:.2f} +- {summary['return_std']:.2f}, "
        f"length {summary['length_mean']:.0f}\n-> {out}")
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('--logdir', required=True, help='run directory with config.json and checkpoint.ckpt')
  parser.add_argument('--checkpoint', default=None, help='checkpoint file; default <logdir>/checkpoint.ckpt')
  parser.add_argument('--strategy', default='ambush', choices=['default', 'ambush', 'none'], help='predator strategy')
  parser.add_argument('--episodes', type=int, default=10)
  parser.add_argument('--burn-in', type=int, default=0, help='unscored episodes before the scored ones')
  parser.add_argument('--predator-surge-speed', type=float, default=1.0, help='ambush attack speed, multiple of the predator max speed')
  parser.add_argument('--out', default=None, help='output directory; default <logdir>/eval_<strategy>')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--no-video', action='store_true', help='skip the MP4s (much faster)')
  parser.add_argument('--jax-platform', default='cpu', help="'cpu' or 'gpu'")
  args = parser.parse_args()
  out = args.out or str(pathlib.Path(args.logdir) / f'eval_{args.strategy}')
  agent, env, raw = load_agent(args.logdir, args.strategy, args.checkpoint, args.jax_platform, args.seed, args.predator_surge_speed)
  summary = evaluate(agent, env, raw, args.episodes, out, video=not args.no_video, burn_in=args.burn_in)
  summary.update(logdir=args.logdir, checkpoint=args.checkpoint or str(pathlib.Path(args.logdir) / 'checkpoint.ckpt'), seed=args.seed)
  json.dump(summary, open(pathlib.Path(out) / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
  main()
