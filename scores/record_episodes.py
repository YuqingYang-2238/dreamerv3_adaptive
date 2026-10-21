"""Records episodes of the prey/predator env as MP4 videos, one file per episode.

Default: a random prey (uniform over the 100 discrete actions, the same
distribution DreamerV3 uses for its prefill) against the ambush predator.

  python scores/record_episodes.py --strategy ambush --episodes 3 --out logdir/videos/ambush_random
  python scores/record_episodes.py --strategy default --episodes 1 --out logdir/videos/default_random
"""
import argparse
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('CELLWORLD_CACHE', str(ROOT / 'data' / '2105_world'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path[:0] = [str(ROOT), str(ROOT / 'prey_env')]

from main_train import ENV_KWARGS  # noqa: E402
from prey_env import gym_Environment_D as Environment  # noqa: E402
from prey_env.video import record_episode  # noqa: E402


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--strategy', default='ambush', choices=['default', 'ambush'])
  parser.add_argument('--episodes', type=int, default=3)
  parser.add_argument('--out', default='logdir/videos')
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  random.seed(args.seed)
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  env = Environment(**ENV_KWARGS, predator_strategy=args.strategy)
  act = lambda obs: random.randrange(env.action_space.n)  # random prey
  for i in range(args.episodes):
    summary = record_episode(env, act, str(out / f'episode_{i:03d}.mp4'))
    weights = getattr(env, 'ambush_zone_weights', [])
    print(f"episode {i}: zone {summary['zone']} (weights at reset {weights}), "
          f"surges {summary['surges']}, captures {summary['captures']}, "
          f"{'goal' if summary['goal'] else 'timeout' if summary['timeout'] else 'ended'} "
          f"after {summary['length']} steps, return {summary['ret']:.1f} -> {summary['path']}")
    if args.strategy == 'ambush':
      print(f"  zones entered this episode: {[k for k, e in enumerate(env.zone_history[-1]) if e]}")


if __name__ == '__main__':
  main()
