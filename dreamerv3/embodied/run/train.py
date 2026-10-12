import re

import embodied
import numpy as np


def train(agent, env, replay, logger, args, eval_env=None):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_value = embodied.when.Every(int(args.value_change_every))
  value_snap = [None]  # Critic snapshot from delta steps ago; not checkpointed.
  snap_meta = [None]   # (updates, step) at which value_snap was taken.
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, [
      'policy', 'train', 'report', 'save', 'value_change'])
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
        'sum_abs_reward': sum_abs_reward,
        'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')
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

  print('Prefill train dataset.')
  random_agent = embodied.RandomAgent(env.act_space)
  while len(replay) < max(args.batch_steps, args.train_fill):
    driver(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset = agent.dataset(replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset'):
        batch[0] = next(dataset)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_value(step):
      # Learning marker: critic change since the reference snapshot, measured
      # on a fresh replay batch. The first firing (and the first one after a
      # resume) only installs the reference.
      if value_snap[0] is not None:
        with timer.scope('dataset'):
          vbatch = next(dataset)
        mets = agent.value_change(vbatch, value_snap[0])
        snap_updates, snap_step = snap_meta[0]
        mets.update(
            updates=int(updates),
            ref_age_updates=int(updates) - snap_updates,
            ref_age_steps=int(step) - snap_step)
        logger.add(mets, prefix='learning')
        print(
            f"Value change at step {int(step)}: {mets['value_change']:.4f} "
            f"(reference age {mets['ref_age_updates']} updates)")
      value_snap[0] = agent.snapshot_critic()
      snap_meta[0] = (int(updates), int(step))
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      agg = metrics.result()
      report = agent.report(batch[0])
      report = {k: v for k, v in report.items() if 'train/' + k not in agg}
      logger.add(agg)
      logger.add(report, prefix='report')
      logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver.on_step(train_step)

  # Evaluation: separate env and driver. Nothing here increments the step
  # counter, inserts into the replay, trains, or touches the value snapshot.
  eval_episodes = []
  def per_eval_episode(ep):
    vector = ep['vector']
    goal = bool(ep['log_goal'].any())
    eval_episodes.append(dict(
        length=len(ep['reward']) - 1,
        score=float(ep['reward'].astype(np.float64).sum()),
        goal=goal,
        success=bool(ep['log_success'].any()),  # goal reached, never captured
        timeout=bool(ep['log_timeout'].any()) and not goal,
        captures=int(ep['log_captured'].sum()),
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
    print(f'Evaluation at step {int(step)}: {int(args.eval_eps)} episodes.')
    eval_episodes.clear()
    driver_eval.reset()
    driver_eval(policy_eval, episodes=int(args.eval_eps))
    eps = list(eval_episodes)
    stack = lambda key: np.array([e[key] for e in eps])
    concat = lambda key: np.concatenate([e[key] for e in eps])
    goal, success, timeout = stack('goal'), stack('success'), stack('timeout')
    captures, score, length = stack('captures'), stack('score'), stack('length')
    logger.add({
        'success_rate': success.mean(),
        'goal_rate': goal.mean(),
        'timeout_rate': timeout.mean(),
        'capture_rate': (captures > 0).mean(),
        'captures_mean': captures.mean(),
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
        ep_return=score, ep_length=length,
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
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  while step < args.steps:
    if should_eval(step):
      run_eval()
    driver(policy, steps=100)
    if should_save(step):
      checkpoint.save()
  run_eval()
  checkpoint.save()
  logger.write()
