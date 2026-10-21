import jax
import jax.numpy as jnp
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import nets
from . import jaxutils
from . import ninjax as nj


class Disag(nj.Module):
  """Plan2Explore ensemble of one-step predictors (latent disagreement, Sekar et al. 2020).

  K independently initialised MLP heads map [h_t, z_t, a_t] to a target at t+1 and train with
  their own optimizer on stop-gradient inputs and targets, so nothing else in the agent is
  affected. `__call__` is the exploration reward used by behaviors.Explore; `disagreement` is
  the measured signal: unbiased variance across heads, averaged over target dims.
  """

  def __init__(self, wm, act_space, config):
    self.config = config.update({'disag_head.inputs': ['tensor']})
    self.opt = jaxutils.Optimizer(name='disag_opt', **config.expl_opt)
    self.inputs = nets.Input(config.disag_head.inputs, dims='deter')
    self.target = nets.Input(self.config.disag_target, dims='deter')
    # Fixed output width when predicting the embedding, so policy() and measurements
    # work before train() was traced. Other targets keep the shape patched in loss().
    shape = (config.encoder.mlp_units,) if list(config.disag_target) == ['embed'] else None
    self.nets = [
        nets.MLP(shape=shape, **self.config.disag_head, name=f'disag{i}')
        for i in range(self.config.disag_models)]

  def __call__(self, traj):
    return self.predictions(traj).std(0).mean(-1)[1:]

  def predictions(self, inp):
    """Stacked head predictions [K, ..., d] (float32) for the same states and actions."""
    inp = self.inputs(inp)
    return jnp.stack([net(inp).mode() for net in self.nets])

  def disagreement(self, inp):
    """D = mean over dims of the unbiased variance across heads: one value per (state, action)."""
    return self.predictions(inp).var(0, ddof=1).mean(-1)

  def train(self, data):
    return self.opt(self.nets, self.loss, data)

  def loss(self, data):
    inp = sg(self.inputs(data)[:, :-1])
    tar = sg(self.target(data)[:, 1:]).astype(jnp.float32)
    mask = 1.0 - data['is_first'][:, 1:].astype(jnp.float32)   # drop pairs crossing a reset
    losses = []
    for net in self.nets:
      if net._shape is None:
        net._shape = tar.shape[2:]
      err = ((net(inp).mode() - tar) ** 2).mean(-1)   # per-dimension MSE
      losses.append((err * mask).sum() / jnp.maximum(mask.sum(), 1.0))
    return jnp.array(losses).sum()
