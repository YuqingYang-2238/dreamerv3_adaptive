import json
import os
import pathlib
import random
import subprocess
import sys
import time

import numpy as np

# World data for the fixed Mice world, read by cellworld instead of GitHub.
# Must be set before cellworld is imported (it reads the variable at import).
os.environ.setdefault(
    'CELLWORLD_CACHE', str(pathlib.Path(__file__).parent / 'data' / '2105_world'))

from prey_env import gym_Environment_D as Environment
from prey_env.embodied_env import PreyFromGym

# Mice-aligned task settings, shared by the training and the evaluation env.
ENV_KWARGS = dict(
    world_name='21_05',      # fixed Mice world ('21_05' scatter, 'clump01_05' clumpy, '00_00' open)
    has_predator=True,
    max_step=300,
    time_step=0.25, freq=40,  # 0.25 s per RL step = 10 ticks of 0.025 s
    prey_max_speed=0.5,       # Mice prey speed
    predator_prey_forward_speed_ratio=0.4,  # predator max speed = 0.4 x prey = 0.2 (both strategies)
)


def env_kwargs(config):
  """ENV_KWARGS plus the options selectable from the command line (--env.prey.*)."""
  return dict(ENV_KWARGS, predator_strategy=config.env.prey.predator_strategy,
              predator_surge_speed=config.env.prey.predator_surge_speed)


def ambush_info(config):
  """What the ambush predator does, for run_info.jsonl (None for the default predator)."""
  if config.env.prey.predator_strategy != 'ambush':
    return None
  from prey_env.AmbushPredator import AMBUSH_ZONES, ATTACK_CELLS, HISTORY_TRIALS
  return dict(
      zones=AMBUSH_ZONES[ENV_KWARGS['world_name']],
      attack_cells=ATTACK_CELLS,
      history_trials=HISTORY_TRIALS,
      predator_surge_speed=config.env.prey.predator_surge_speed,
      trigger='true prey position (god view), like the overhead tracker in the experiment',
      bias='zone weight = 1 + episodes among the last history_trials in which the prey entered the zone',
      window='per env object (train and eval separate), not checkpointed')


def make_env(config):
  """One prey/predator env wrapped for DreamerV3 (vector obs, log_* stats)."""
  import dreamerv3
  from dreamerv3 import embodied
  env = PreyFromGym(Environment(**env_kwargs(config)), obs_key='vector')
  env = dreamerv3.wrap_env(env, config)
  return embodied.BatchEnv([env], parallel=False)


def write_run_info(logdir, config):
  """Appends what is needed to reproduce the run: seed, code version, config."""
  git = lambda *cmd: subprocess.run(
      ['git', *cmd], capture_output=True, text=True).stdout.strip()
  info = dict(
      started=time.strftime('%Y-%m-%d %H:%M:%S'),
      argv=sys.argv,
      logdir=str(logdir),
      seed=config.seed,
      git_hash=git('rev-parse', 'HEAD'),
      git_dirty=bool(git('status', '--porcelain')),
      jax=dict(platform=config.jax.platform, precision=config.jax.precision),
      env=env_kwargs(config),
      ambush=ambush_info(config),
      eval=dict(
          every=config.run.eval_every,
          episodes=config.run.eval_eps,
          burn_in=config.run.eval_burn_in,
          initial=config.run.eval_initial,
          env='separate Environment with the same kwargs as training',
          action_selection='sampled from the actor (DreamerV3 eval mode)',
          replay='not inserted', updates='none', step_counter='not advanced',
          rng='agent and global RNG states restored afterwards',
          predator_memory='eval env window cleared before every evaluation'),
      imagination=dict(
          horizon=config.imag_horizon,
          adaptive=config.imag_adaptive,
          cutoff=None if not config.disag_measure else dict(
              rule='MACURA: accept imagined transition t while ensemble disagreement D_s < kappa '
                   'for all s <= t; rejected steps excluded from targets and losses',
              xi=config.imag_xi, zeta=config.imag_zeta, history=config.imag_kappa_history,
              kappa='(b + c * mean(last stored b)) / (c + 1), b = xi * zeta-percentile of D '
                    'at the start states, c = min(rounds, history - 1)',
              applied=config.imag_adaptive),
          batch_size=config.batch_size,
          batch_length=config.batch_length,
          train_ratio=config.run.train_ratio),
      value_change_every=config.run.value_change_every,
      disagreement=None if not config.disag_measure else dict(
          models=config.disag_models, layers=config.disag_head.layers,
          units=config.disag_head.units, inputs=['deter', 'stoch', 'action'],
          target='embed (next observation embedding)',
          bootstrap='none: members differ by initialisation only',
          loss='per-dimension MSE, pairs crossing an episode reset excluded, own optimizer',
          optimizer=dict(config.expl_opt),
          signal='learning/disag = mean over the batch pairs of the unbiased variance '
                 'across heads (mean over dims); learning/disag_var = variance of that '
                 'over pairs; same cadence and batches as value_change; per episode '
                 'episode/disag_{mean,var} from the per-step log_disag',
          visible_split='learning/disag_{visible,hidden}(_recent), disag_mse_visible, dyn_kl_visible '
                        'and episode/eval disag_visible_mean: masked by obs entry 5 (pred_x) >= 0, '
                        'i.e. the predator is in the prey\'s line of sight at that step'),
      adaptive=dict(
          enabled=config.run.adaptive,
          c=config.run.adaptive_c,
          c_source='pooled median of learning/value_change_recent over the '
                   'first 20k steps of cont_s{1,2,3}_default (same checkpoint, '
                   'same protocol); the trunk logged no recent marker',
          n0=config.batch_size * config.batch_length,
          extra_rollouts=config.run.imag_extra_rollouts,
          signal='learning/value_change_recent: mean |V_now - V_snapshot| on '
                 'one batch from the last 20000 replay sequences, both '
                 'critics on the same batch and the same sampled posterior '
                 'latents; snapshot age = value_change_every env steps',
          smoothing='smoothed = 0.8 smoothed + 0.2 u, p = clip((smoothed-c)/c, 0, 1)',
          gate='Bernoulli(p_extra) per actor-critic update, controller RNG '
               'seeded by [seed, 7]; always one WM + one actor + one critic step',
          eval_seeding='random, np.random and agent.rng reseeded from '
                       '(seed, step) before every evaluation, restored after'),
  )
  (logdir / 'run_info.jsonl').write(json.dumps(info) + '\n', mode='a')  # one line per (re)start
  diff = git('diff', 'HEAD')
  diff and (logdir / 'git_diff.patch').write(diff + '\n')
  config.save(logdir / 'config.json')


def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  ## For non-image
  config = config.update({
      'logdir': f'logdir/{int(time.time())}',  # pass --logdir to name or resume a run
      'run.steps': 150000,       # total environment steps
      'run.train_ratio': 64,
      'run.log_every': 30,
      'run.eval_every': 20000,   # env steps between evaluations
      'run.eval_eps': 20,        # episodes per evaluation
      'run.eval_initial': True,  # also evaluate the untrained agent
      'run.eval_burn_in': 10,    # unscored episodes first: the ambush bias window refills
      'run.log_keys_sum': 'log_captured|log_surge',   # -> stats/sum_log_{captured,surge}
      'run.log_keys_max': 'log_success|log_goal|log_timeout',  # -> stats/max_log_{success,goal,timeout}
      'run.log_zeros': True,     # keep all-zero episodes in those stats
      'run.log_keys_mean': '(log_entropy|log_disag)',  # -> stats/mean_log_disag (per-step ensemble disagreement)
      # Measured-only Plan2Explore ensemble (on with --disag_measure True). Set here, not in
      # configs.yaml defaults: the 'medium' preset above rewrites every *.layers / *.units.
      'disag_models': 5, 'disag_head.layers': 2, 'disag_target': ['embed'],
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '.*',
      'decoder.mlp_keys': '.*',
      'encoder.cnn_keys': '$^',
      'decoder.cnn_keys': '$^',
      'jax.platform': 'cpu',  # pass --jax.platform gpu on a GPU machine
      'jax.precision': 'float32',  # float16 (the default) is very slow on CPU
  })
  config = embodied.Flags(config).parse()
  random.seed(config.seed)      # env: world pick, predator spawn and wander
  np.random.seed(config.seed)   # random prefill actions

  logdir = embodied.Path(config.logdir)
  logdir.mkdirs()
  write_run_info(logdir, config)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  env = make_env(config)
  eval_env = make_env(config)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay',
      seed=config.seed)
  args = embodied.Config(
      **config.run, logdir=config.logdir, seed=config.seed,
      batch_steps=config.batch_size * config.batch_length,
      imag_horizon=config.imag_horizon)
  embodied.run.train(agent, env, replay, logger, args, eval_env=eval_env)


if __name__ == '__main__':
  main()
