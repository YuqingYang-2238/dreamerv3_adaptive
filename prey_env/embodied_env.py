"""DreamerV3 (embodied) wrapper for the prey/predator gym env.

Adds log-only observation keys that carry the env's episode outcome into the
DreamerV3 logs. Keys starting with 'log_' are skipped by Agent.preprocess,
excluded from the encoder/decoder, and stripped by the replay, so the agent's
inputs stay the 14-float 'vector'.
"""

import functools

import numpy as np

import dreamerv3  # noqa: F401  Puts the bundled 'embodied' package on sys.path.
import embodied
from embodied.envs import from_gym


class PreyFromGym(from_gym.FromGym):

  LOG_SPACES = {
      'log_captured': embodied.Space(np.int32),  # captures during this step
      'log_goal': embodied.Space(bool),          # goal reached this step
      'log_success': embodied.Space(bool),       # goal reached with zero captures
      'log_timeout': embodied.Space(bool),       # truncated this step (no goal)
      'log_pred_x': embodied.Space(np.float32),  # true predator position,
      'log_pred_y': embodied.Space(np.float32),  # visible or not (for plots)
      'log_zone': embodied.Space(np.int32),      # ambush zone of this episode (-1: default predator)
      'log_surge': embodied.Space(np.int32),     # ambush surges started during this step
  }

  @functools.cached_property
  def obs_space(self):
    base = from_gym.FromGym.obs_space.func(self)
    return {**base, **self.LOG_SPACES}

  def _obs(self, obs, reward, is_first=False, **kwargs):
    out = super()._obs(obs, reward, is_first=is_first, **kwargs)
    info = {} if is_first else self._info
    goal = bool(info.get('is success', False))
    predator = self._env.model.agents_data.get('predator')
    out.update(
        log_captured=np.int32(info.get('captured_this_step', 0)),
        log_goal=goal,
        log_success=bool(info.get('survived', False)),
        log_timeout=bool(info.get('is truncated', False)) and not goal,
        log_pred_x=np.float32(predator.location.x if predator else -1.0),
        log_pred_y=np.float32(predator.location.y if predator else -1.0),
        log_zone=np.int32(getattr(self._env, 'ambush_zone', -1)),
        log_surge=np.int32(info.get('surges_this_step', 0)))
    return out

  def reset_memory(self):
    """Clears the ambush zone window (no-op for the default predator)."""
    self._env.zone_history.clear()
