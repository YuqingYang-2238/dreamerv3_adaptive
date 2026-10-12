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
    predator_prey_forward_speed_ratio=0.3,  # predator max speed = 0.3 x prey = 0.15
)


def make_env(config):
  """One prey/predator env wrapped for DreamerV3 (vector obs, log_* stats)."""
  import dreamerv3
  from dreamerv3 import embodied
  env = PreyFromGym(Environment(**ENV_KWARGS), obs_key='vector')
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
      env=ENV_KWARGS,
      eval=dict(
          every=config.run.eval_every,
          episodes=config.run.eval_eps,
          initial=config.run.eval_initial,
          env='separate Environment with the same kwargs as training',
          action_selection='sampled from the actor (DreamerV3 eval mode)',
          replay='not inserted', updates='none', step_counter='not advanced'),
      imagination=dict(
          horizon=config.imag_horizon,
          batch_size=config.batch_size,
          batch_length=config.batch_length,
          train_ratio=config.run.train_ratio),
      value_change_every=config.run.value_change_every,
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
      'run.log_keys_sum': 'log_captured',             # -> stats/sum_log_captured
      'run.log_keys_max': 'log_success|log_goal|log_timeout',  # -> stats/max_log_{success,goal,timeout}
      'run.log_zeros': True,     # keep all-zero episodes in those stats
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
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args, eval_env=eval_env)


if __name__ == '__main__':
  main()
