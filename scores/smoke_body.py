"""Body/collision smoke test: random prey, both predator strategies, headless.

Checks every physics tick that both bodies are inside the arena and the prey body never touches
an occlusion; reports predator/occlusion overlap, blocked moves, predator stalls and speed.

  python scores/smoke_body.py --episodes 10
"""
import argparse
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('CELLWORLD_CACHE', str(ROOT / 'data' / '2105_world'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path[:0] = [str(ROOT), str(ROOT / 'prey_env')]

import numpy as np  # noqa: E402

from main_train import ENV_KWARGS  # noqa: E402
from prey_env import gym_Environment_D as Environment  # noqa: E402


def run(strategy, episodes, seed):
  random.seed(seed)
  np.random.seed(seed)
  env = Environment(**ENV_KWARGS, predator_strategy=strategy)
  stats = dict(ticks=0, prey_hit=0, pred_hit=0, outside=0, stall=0, pred_moving=0)
  last = {}

  def on_tick():
    g, agents = env.model.geometry, env.model.agents_data
    stats['ticks'] += 1
    for name, a in agents.items():
      stats['outside'] += not g.arena.contains(a.body_polygon)
      hit = g.occlusions.intersects(a.body_polygon)
      stats['prey_hit' if name == 'prey' else 'pred_hit'] += hit
      assert not (name == 'prey' and hit), f'prey body inside an occlusion at {a.location}'
    p = agents['predator']
    if env.predator.destination is not None and p.speed > 0:
      stats['pred_moving'] += 1
      stats['stall'] += last.get('xy') == (p.location.x, p.location.y)
    last['xy'] = (p.location.x, p.location.y)

  env.tick_callback = on_tick
  steps, goals, captures, wedged, t0 = 0, 0, 0, 0, time.time()
  for _ in range(episodes):
    env.reset()
    last.clear()
    done, xy = False, []
    while not done:
      _, _, done, info = env.step(random.randrange(env.action_space.n))
      steps += 1
      xy.append((env.model.agents_data['prey'].location.x, env.model.agents_data['prey'].location.y))
    goals += info['is success']
    captures += info['captures']
    moved = np.hypot(*np.diff(np.array(xy[-40:]), axis=0).T)   # last 10 s
    wedged += len(moved) == 39 and (moved < 1e-3).all()
  n = stats['ticks']
  print(f"{strategy}: {episodes} episodes, {steps} steps, {n} ticks, {1e3 * (time.time() - t0) / steps:.1f} ms/step, "
        f"goal {goals / episodes:.2f}, captures/ep {captures / episodes:.1f}, wedged at the end {wedged}/{episodes}")
  print(f"  prey body in occlusion {stats['prey_hit'] / n:.4f}  predator body in occlusion {stats['pred_hit'] / n:.4f}  "
        f"body outside arena {stats['outside'] / n:.4f}")
  print(f"  prey moves blocked {env.model.blocked['prey'] / n:.3f}  predator moves blocked {env.model.blocked['predator'] / n:.3f}  "
        f"predator stalled {stats['stall'] / max(1, stats['pred_moving']):.4f}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--episodes', type=int, default=10)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  for strategy in ('default', 'ambush'):
    run(strategy, args.episodes, args.seed)
