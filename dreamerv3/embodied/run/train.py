import copy
import random
import re
import threading
import time

import embodied
import numpy as np


class ValueRef:
  """Critic reference for the value-change marker; checkpointed with its (updates, step)."""

  def __init__(self):
    self.data = None  # {'params': host arrays, 'updates': int, 'step': int}

  def take(self, agent, updates, step):
    # One atomic rebind: save() may run concurrently on the checkpoint thread.
    self.data = dict(
        params=agent.snapshot_critic(), updates=int(updates), step=int(step))

  def save(self):
    return self.data

  def load(self, data):
    self.data = data


class ImagController:
  """Signal-adaptive imagination budget (Bernoulli gate per actor-critic update).

  Every signal measurement u (recent-experience value change) updates
  smoothed = 0.8 * smoothed + 0.2 * u and p_extra = clip((smoothed - c) / c, 0, 1),
  held until the next measurement. gate() draws Bernoulli(p_extra) from its own
  RNG stream and returns the number of imagined rollouts per start state for
  the next update (1 = baseline N0, `extra` = the larger budget). With
  enabled=False it never draws and always returns 1 (the fixed condition), but
  it still counts updates, imagined trajectories and wall time for both.
  All state sits in one dict behind a lock because the checkpoint thread
  calls save() while training continues.
  """

  def __init__(self, enabled, c, seed, n0, horizon, extra=2):
    enabled = bool(enabled)
    if enabled and not (np.isfinite(c) and c > 0):
      raise ValueError(
          f'--run.adaptive_c must be finite and positive, got {c!r}. '
          'Calibrate it as the median recent-experience value change of the '
          'pretraining run and pass it explicitly.')
    self.enabled, self.c, self.n0, self.horizon, self.extra = (
        enabled, float(c), int(n0), int(horizon), int(extra))
    self._lock = threading.Lock()
    self.rng = np.random.default_rng([int(seed), 7])  # controller-only stream
    self.s = dict(
        smoothed=float(c), p_extra=0.0, u_raw=float('nan'), measurements=0,
        updates=0, updates_extra=0, interval_updates=0, interval_extra=0,
        imag_trajs=0, imag_transitions=0, wall_seconds=0.0)
    self._tick = None

  def gate(self):
    """Rollouts for the next update; call exactly once per update."""
    with self._lock:
      extra = self.enabled and bool(self.rng.random() < self.s['p_extra'])
      rollouts = self.extra if extra else 1
      self.s['updates'] += 1
      self.s['interval_updates'] += 1
      self.s['updates_extra'] += int(extra)
      self.s['interval_extra'] += int(extra)
      self.s['imag_trajs'] += rollouts * self.n0
      self.s['imag_transitions'] += rollouts * self.n0 * self.horizon
      return rollouts

  def update(self, u):
    """New signal measurement; returns the metrics to log."""
    with self._lock:
      s = self.s
      s['u_raw'] = float(u)
      s['measurements'] += 1
      s['smoothed'] = 0.8 * s['smoothed'] + 0.2 * float(u)
      if self.enabled:
        s['p_extra'] = float(np.clip(
            (s['smoothed'] - self.c) / (self.c + 1e-8), 0.0, 1.0))
      frac = s['interval_extra'] / max(1, s['interval_updates'])
      s['interval_updates'] = s['interval_extra'] = 0
      return {**self._stats(), 'extra_frac_interval': frac}

  def tick(self):
    """Accumulates training wall time between calls (not the eval time)."""
    now = time.time()
    with self._lock:
      if self._tick is not None:
        self.s['wall_seconds'] += now - self._tick
      self._tick = now

  def pause(self):
    with self._lock:
      if self._tick is not None:
        self.s['wall_seconds'] += time.time() - self._tick
      self._tick = None

  def stats(self):
    with self._lock:
      return self._stats()

  def _stats(self):
    s = self.s
    return dict(
        enabled=float(self.enabled), c=self.c, u_raw=s['u_raw'],
        u_smoothed=s['smoothed'], p_extra=s['p_extra'],
        measurements=s['measurements'], updates=s['updates'],
        wm_updates=s['updates'], updates_extra=s['updates_extra'],
        extra_frac_total=s['updates_extra'] / max(1, s['updates']),
        imag_trajs=s['imag_trajs'], imag_transitions=s['imag_transitions'],
        wall_seconds=s['wall_seconds'])

  def save(self):
    with self._lock:
      return dict(state=copy.deepcopy(self.s), rng=self.rng.bit_generator.state)

  def load(self, data):
    with self._lock:
      self.s.update(data['state'])
      self.rng.bit_generator.state = data['rng']


def episode_disag(ep):
  """Mean and variance over an episode's steps of the per-step ensemble disagreement
  (nan when the agent has no ensemble). The driver zeroes the last step's outputs."""
  if 'log_disag' not in ep or len(ep['log_disag']) < 3:
    return float('nan'), float('nan')
  d = np.asarray(ep['log_disag'], np.float64).reshape(-1)[:-1]
  return float(d.mean()), float(d.var(ddof=1))


def episode_disag_visible(ep):
  """Mean of the per-step disagreement over the steps where the prey saw the predator
  (obs entry 5 = pred_x, sentinel -1); the driver-zeroed last step is excluded."""
  if 'log_disag' not in ep or len(ep['log_disag']) < 2:
    return float('nan')
  d = np.asarray(ep['log_disag'], np.float64).reshape(-1)
  # An episode that started under the prefill random agent has fewer signal entries than
  # observations: the signal covers the episode's last len(d) steps.
  vis = np.asarray(ep['vector'])[-len(d):, 5] >= 0
  d, vis = d[:-1], vis[:-1]
  return float(d[vis].mean()) if vis.any() else float('nan')


def episode_kl(ep, key='log_kl'):
  """Mean over an episode of a per-step policy output (default: world-model KL(posterior ||
  prior)); the first step (no valid previous step) and the last (zeroed by the driver) are
  excluded. nan when not logged."""
  if key not in ep or len(ep[key]) < 3:
    return float('nan')
  return float(np.asarray(ep[key], np.float64).reshape(-1)[1:-1].mean())


def train(agent, env, replay, logger, args, eval_env=None):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_keep = embodied.when.Every(args.keep_every, initial=False)
  should_sync = embodied.when.Every(args.sync_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  # initial=False: the first firing installs the reference on the grid, so
  # every measurement has an age of exactly value_change_every steps.
  should_value = embodied.when.Every(int(args.value_change_every), initial=False)
  value_ref = ValueRef()
  controller = ImagController(
      args.adaptive, args.adaptive_c, args.seed, args.batch_steps,
      args.imag_horizon, args.imag_extra_rollouts)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  loaded = len(replay)  # sequences read from disk (0 on a fresh run)
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, [
      'policy', 'train', 'report', 'save', 'value_change', 'disag_measure', 'kl_measure'])
  measures_disag = getattr(agent, 'measures_disag', False)
  timer.wrap('env', env, ['step'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
    logger.add({
        'length': length,
        'score': score,
        'success': float(ep['log_success'].any()),  # goal reached without capture
        'sum_abs_reward': sum_abs_reward,
        'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')
    logger.add({'kl_mean': episode_kl(ep)}, prefix='episode')
    if measures_disag:
      dmean, dvar = episode_disag(ep)
      logger.add({'disag_mean': dmean, 'disag_var': dvar,
                  'disag_visible_mean': episode_disag_visible(ep)}, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(replay.add)
  # Last ~20k steps only, never saved: scores the marker on post-branch data.
  recent = embodied.replay.Uniform(replay.length, 20000)
  driver.on_step(recent.add)

  print('Prefill train dataset.')
  random_agent = embodied.RandomAgent(env.act_space)
  while len(replay) < max(args.batch_steps, args.train_fill):
    driver(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset = agent.dataset(replay.dataset)
  dataset_recent = agent.dataset(recent.dataset, workers=0)  # sampled on demand
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset'):
        batch[0] = next(dataset)
      rollouts = controller.gate()  # 1 = N0 trajectories, 2 = 2 N0
      outs, state[0], mets = agent.train(batch[0], state[0], rollouts=rollouts)
      metrics.add(mets, prefix='train')
      metrics.add({'rollouts': rollouts}, prefix='adaptive')
      if 'priority' in outs:
        replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_value(step):
      # Learning marker: critic change since the reference snapshot, measured
      # on a fresh replay batch and on a batch of recent steps.
      ref = value_ref.data
      if ref is not None:
        with timer.scope('dataset'):
          vbatch = next(dataset)
        mets = agent.value_change(vbatch, ref['params'])
        mets.update(agent.kl_measure(vbatch))   # world-model KL on the same batch
        if measures_disag:   # ensemble disagreement on the same batch
          mets.update(agent.disag_measure(vbatch))
        mets.update(
            updates=int(updates),
            ref_age_updates=int(updates) - ref['updates'],
            ref_age_steps=int(step) - ref['step'],
            replay_new_frac=(len(replay) - loaded) / len(replay))
        if len(recent):
          rbatch = next(dataset_recent)
          rmets = agent.value_change(rbatch, ref['params'])
          rmets.update(agent.kl_measure(rbatch))
          if measures_disag:
            rmets.update(agent.disag_measure(rbatch))
          mets.update({f'{k}_recent': v for k, v in rmets.items()})
          # The controller's signal u_t is the recent-experience value change.
          logger.add(controller.update(rmets['value_change']), prefix='adaptive')
        logger.add(mets, prefix='learning')
        print(
            f"Value change at step {int(step)}: {mets['value_change']:.4f} "
            f"(reference age {mets['ref_age_updates']} updates)")
      value_ref.take(agent, updates, step)
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      agg = metrics.result()
      # No batch yet with train_ratio 0 (collection-only runs): skip the report.
      report = agent.report(batch[0]) if batch[0] is not None else {}
      report = {k: v for k, v in report.items() if 'train/' + k not in agg}
      logger.add(agg)
      controller.tick()
      logger.add(controller.stats(), prefix='adaptive')
      logger.add(report, prefix='report')
      logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver.on_step(train_step)

  # Evaluation: separate env and driver. Nothing here increments the step
  # counter, inserts into the replay, trains, or touches the value reference.
  eval_episodes = []
  def per_eval_episode(ep):
    vector = ep['vector']
    goal = bool(ep['log_goal'].any())
    dmean, dvar = episode_disag(ep)
    eval_episodes.append(dict(
        disag_mean=dmean, disag_var=dvar, kl_mean=episode_kl(ep),
        disag_err_mean=episode_kl(ep, 'log_disag_err'),
        disag_visible_mean=episode_disag_visible(ep),
        length=len(ep['reward']) - 1,
        score=float(ep['reward'].astype(np.float64).sum()),
        goal=goal,
        success=bool(ep['log_success'].any()),  # goal reached, never captured
        timeout=bool(ep['log_timeout'].any()) and not goal,
        captures=int(ep['log_captured'].sum()),
        zone=int(ep['log_zone'].max()),        # ambush zone (-1: default predator)
        surges=int(ep['log_surge'].sum()),     # ambush surges in the episode
        reward=ep['reward'],
        captured=ep['log_captured'],
        prey_xy=vector[:, 0:2],
        pred_visible=vector[:, 5] >= 0.0,  # The obs sentinel is (-1, -1).
        pred_xy=np.stack([
            ep['log_pred_x'].reshape(-1), ep['log_pred_y'].reshape(-1)], -1),
    ))
  if eval_env is not None:
    driver_eval = embodied.Driver(eval_env)
    driver_eval.on_episode(lambda ep, worker: per_eval_episode(ep))
    policy_eval = lambda *args: agent.policy(*args, mode='eval')
    (logdir / 'eval_trajs').mkdirs()

  def run_eval():
    if eval_env is None:
      return
    print(f'Evaluation at step {int(step)}: {int(args.eval_eps)} episodes '
          f'after {int(args.eval_burn_in)} unscored warm-up episodes.')
    # Training must not see the evaluation: the RNG streams are restored
    # afterwards, and every phase starts from the same predator memory.
    controller.pause()
    rng_state = agent.rng.bit_generator.state
    py_state, np_state = random.getstate(), np.random.get_state()
    # Matched initial conditions across runs: the env draws its spawns and
    # the predator's choices from `random`, actions come from agent.rng, so
    # phases at the same step see the same draws given the same policy.
    random.seed(f'{int(args.seed)}-{int(step)}')
    np.random.seed((int(args.seed) * 1_000_003 + int(step)) % 2 ** 32)
    agent.rng = np.random.default_rng([int(args.seed), int(step)])
    eval_env._envs[0].reset_memory()
    driver_eval.reset()
    driver_eval(policy_eval, episodes=int(args.eval_burn_in))
    eval_episodes.clear()
    driver_eval(policy_eval, episodes=int(args.eval_eps))
    agent.rng.bit_generator.state = rng_state
    random.setstate(py_state)
    np.random.set_state(np_state)
    controller.tick()
    eps = list(eval_episodes)
    stack = lambda key: np.array([e[key] for e in eps])
    concat = lambda key: np.concatenate([e[key] for e in eps])
    goal, success, timeout = stack('goal'), stack('success'), stack('timeout')
    captures, score, length = stack('captures'), stack('score'), stack('length')
    zone, surges = stack('zone'), stack('surges')
    dmean, dvar, klm = stack('disag_mean'), stack('disag_var'), stack('kl_mean')
    logger.add({'kl_mean': np.nanmean(klm)}, prefix='eval')
    if measures_disag:
      logger.add({'disag_mean': dmean.mean(), 'disag_var': dvar.mean(),
                  'disag_err_mean': np.nanmean(stack('disag_err_mean')),
                  'disag_visible_mean': np.nanmean(stack('disag_visible_mean'))}, prefix='eval')
    logger.add({
        'success_rate': success.mean(),
        'goal_rate': goal.mean(),
        'timeout_rate': timeout.mean(),
        'capture_rate': (captures > 0).mean(),
        'captures_mean': captures.mean(),
        'surge_rate': (surges > 0).mean(),  # episodes with >= 1 ambush surge
        'surges_mean': surges.mean(),
        'return_mean': score.mean(),
        'return_std': score.std(),
        'length_mean': length.mean(),
        'episodes': len(eps),
        'updates': int(updates),
    }, prefix='eval')
    np.savez_compressed(
        str(logdir / 'eval_trajs' / f'step_{int(step):08d}.npz'),
        step=int(step), updates=int(updates),
        ep_offsets=np.cumsum([0] + [len(e['reward']) for e in eps]),
        ep_goal=goal, ep_success=success, ep_timeout=timeout, ep_captures=captures,
        ep_return=score, ep_length=length, ep_zone=zone, ep_surges=surges,
        ep_disag_mean=dmean, ep_disag_var=dvar, ep_kl_mean=klm,
        prey_xy=concat('prey_xy'), pred_xy=concat('pred_xy'),
        pred_visible=concat('pred_visible'),
        captured=concat('captured'), reward=concat('reward'))
    logger.write()

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  timer.wrap('checkpoint', checkpoint, ['save', 'load'])
  checkpoint.step = step
  checkpoint.updates = updates
  checkpoint.agent = agent
  checkpoint.replay = replay
  checkpoint.value_ref = value_ref
  checkpoint.controller = controller
  checkpoint.load_or_save()  # Continue a run: main_branch.py prepares the logdir.
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  controller.tick()
  while step < args.steps:
    if should_eval(step):
      run_eval()
    driver(policy, steps=100)
    controller.tick()
    if should_keep(step):
      checkpoint.save(logdir / f'checkpoint_{int(step)}.ckpt')
    if should_save(step):
      checkpoint.save()
  run_eval()
  checkpoint.save()
  checkpoint.wait()  # The process exits next; the save thread must finish first.
  logger.write()
