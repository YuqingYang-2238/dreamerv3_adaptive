import os

import embodied
import jax
import jax.numpy as jnp
import numpy as np

from . import jaxutils
from . import ninjax as nj

tree_map = jax.tree_util.tree_map
tree_flatten = jax.tree_util.tree_flatten


def Wrapper(agent_cls):
  class Agent(JAXAgent):
    configs = agent_cls.configs
    inner = agent_cls
    def __init__(self, *args, **kwargs):
      super().__init__(agent_cls, *args, **kwargs)
  return Agent


class JAXAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, step, config):
    self.config = config.jax
    self.batch_size = config.batch_size
    self.batch_length = config.batch_length
    self.data_loaders = config.data_loaders
    self._setup()
    self.agent = agent_cls(obs_space, act_space, step, config, name='agent')
    self.rng = np.random.default_rng(config.seed)

    available = jax.devices(self.config.platform)
    self.policy_devices = [available[i] for i in self.config.policy_devices]
    self.train_devices = [available[i] for i in self.config.train_devices]
    self.single_device = (self.policy_devices == self.train_devices) and (
        len(self.policy_devices) == 1)
    print(f'JAX devices ({jax.local_device_count()}):', available)
    print('Policy devices:', ', '.join([str(x) for x in self.policy_devices]))
    print('Train devices: ', ', '.join([str(x) for x in self.train_devices]))

    self._once = True
    self._updates = embodied.Counter()
    self._should_metrics = embodied.when.Every(self.config.metrics_every)
    self._transform()
    self.varibs = self._init_varibs(obs_space, act_space)
    self.sync()

  def policy(self, obs, state=None, mode='train'):
    obs = obs.copy()
    obs = self._convert_inps(obs, self.policy_devices)
    rng = self._next_rngs(self.policy_devices)
    varibs = self.varibs if self.single_device else self.policy_varibs
    if state is None:
      state, _ = self._init_policy(varibs, rng, obs['is_first'])
    else:
      state = tree_map(
          np.asarray, state, is_leaf=lambda x: isinstance(x, list))
      state = self._convert_inps(state, self.policy_devices)
    (outs, state), _ = self._policy(varibs, rng, obs, state, mode=mode)
    outs = self._convert_outs(outs, self.policy_devices)
    # TODO: Consider keeping policy states in accelerator memory.
    state = self._convert_outs(state, self.policy_devices)
    return outs, state

  def train(self, data, state=None, rollouts=1):
    # rollouts is a static jit argument: each value is its own compiled
    # program, so only the requested imagination budget is ever generated.
    rng = self._next_rngs(self.train_devices)
    if state is None:
      state, self.varibs = self._init_train(self.varibs, rng, data['is_first'])
    (outs, state, mets), self.varibs = self._train(
        self.varibs, rng, data, state, rollouts=int(rollouts))
    outs = self._convert_outs(outs, self.train_devices)
    self._updates.increment()
    if self._should_metrics(self._updates):
      mets = self._convert_mets(mets, self.train_devices)
    else:
      mets = {}
    if self._once:
      self._once = False
      assert jaxutils.Optimizer.PARAM_COUNTS
      for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
        mets[f'params_{name}'] = float(count)
    return outs, state, mets

  def report(self, data):
    rng = self._next_rngs(self.train_devices)
    mets, _ = self._report(self.varibs, rng, data)
    mets = self._convert_mets(mets, self.train_devices)
    return mets

  # Learning marker: change of the critic's value estimates over time.
  CRITIC_PREFIX = 'agent/task_behavior/critic/net/'

  def snapshot_critic(self):
    """Host copy of the current critic-MLP parameters (picklable)."""
    snap = {
        k: v for k, v in self.varibs.items()
        if k.startswith(self.CRITIC_PREFIX)}
    assert snap, f'No parameters found under {self.CRITIC_PREFIX}'
    if len(self.train_devices) > 1:
      snap = tree_map(lambda x: x[0], snap)
    return tree_map(np.asarray, jax.device_get(snap))

  def value_change(self, data, snapshot):
    """Critic change between the snapshot and now on one replay batch.

    Both evaluations use the current world model and the same rng, so they
    score exactly the same latent states; only the critic parameters differ.
    Returns 'value_change' = mean |V_now - V_snapshot| (the marker), the
    std of V_now - V_snapshot (change beyond a uniform shift) and the mean
    of each critic on that batch, all in reward units.
    """
    if len(self.train_devices) == 1:
      snapshot = jax.device_put(snapshot, self.train_devices[0])
    else:
      snapshot = jax.device_put_replicated(snapshot, self.train_devices)
    rng = self._next_rngs(self.train_devices)  # one rng for both calls
    new, _ = self._value(self.varibs, rng, data)
    old, _ = self._value({**self.varibs, **snapshot}, rng, data)
    missing = set(snapshot) - self._value.keys
    assert not missing, f'Snapshot keys not used by value(): {missing}'
    new, old = self._convert_outs((new, old), self.train_devices)
    new, old = new.astype(np.float64), old.astype(np.float64)
    return {
        'value_change': float(np.abs(new - old).mean()),
        'value_delta_std': float((new - old).std()),
        'value_new_mean': float(new.mean()),
        'value_old_mean': float(old.mean()),
        'value_new_abs_mean': float(np.abs(new).mean()),
    }

  @property
  def measures_disag(self):
    """True when the agent trains the measured-only Plan2Explore ensemble."""
    return hasattr(self.agent, 'disag')

  def kl_measure(self, data):
    """World-model dynamics KL on one batch (episode starts excluded)."""
    rng = self._next_rngs(self.train_devices)
    mets, _ = self._kl_measure(self.varibs, rng, data)
    mets = self._convert_mets(mets, self.train_devices)
    return {k: float(v) for k, v in mets.items()}

  def disag_measure(self, data):
    """Ensemble disagreement (mean, variance), prediction MSE and embed RMS on one batch."""
    rng = self._next_rngs(self.train_devices)
    mets, _ = self._disag_measure(self.varibs, rng, data)
    mets = self._convert_mets(mets, self.train_devices)
    return {k: float(v) for k, v in mets.items()}

  def dataset(self, generator, workers=None):
    batcher = embodied.Batcher(
        sources=[generator] * self.batch_size,
        workers=self.data_loaders if workers is None else workers,
        postprocess=lambda x: self._convert_inps(x, self.train_devices),
        prefetch_source=4, prefetch_batch=1)
    return batcher()

  def save(self):
    if len(self.train_devices) > 1:
      varibs = tree_map(lambda x: x[0], self.varibs)
    else:
      varibs = self.varibs
    varibs = jax.device_get(varibs)
    data = tree_map(np.asarray, varibs)
    return data

  def load(self, state):
    if self.measures_disag:   # otherwise policy() would silently initialise random heads
      assert any(k.startswith('agent/disag/') for k in state), \
          'disag_measure is on but the checkpoint has no ensemble parameters'
    # Older checkpoints lack the MACURA threshold state: start it empty.
    missing = [k for k in self.varibs if '/kappa_' in k and k not in state]
    if missing:
      print('Checkpoint has no', missing, '- starting them from zero.')
      init = self.save()
      state = {**state, **{k: init[k] for k in missing}}
    if len(self.train_devices) == 1:
      self.varibs = jax.device_put(state, self.train_devices[0])
    else:
      self.varibs = jax.device_put_replicated(state, self.train_devices)
    self.sync()

  def sync(self):
    if self.single_device:
      return
    if len(self.train_devices) == 1:
      varibs = self.varibs
    else:
      varibs = tree_map(lambda x: x[0].device_buffer, self.varibs)
    if len(self.policy_devices) == 1:
      self.policy_varibs = jax.device_put(varibs, self.policy_devices[0])
    else:
      self.policy_varibs = jax.device_put_replicated(
          varibs, self.policy_devices)

  def _setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
    except Exception as e:
      print('Could not disable TensorFlow devices:', e)
    if not self.config.prealloc:
      os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    xla_flags = []
    if self.config.logical_cpus:
      count = self.config.logical_cpus
      xla_flags.append(f'--xla_force_host_platform_device_count={count}')
    if xla_flags:
      os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
    jax.config.update('jax_platform_name', self.config.platform)
    jax.config.update('jax_disable_jit', not self.config.jit)
    jax.config.update('jax_debug_nans', self.config.debug_nans)
    jax.config.update('jax_transfer_guard', 'disallow')
    if self.config.platform == 'cpu':
      jax.config.update('jax_disable_most_optimizations', self.config.debug)
    jaxutils.COMPUTE_DTYPE = getattr(jnp, self.config.precision)

  def _transform(self):
    self._init_policy = nj.pure(lambda x: self.agent.policy_initial(len(x)))
    self._init_train = nj.pure(lambda x: self.agent.train_initial(len(x)))
    self._policy = nj.pure(self.agent.policy)
    self._train = nj.pure(self.agent.train)
    self._report = nj.pure(self.agent.report)
    self._value = nj.pure(self.agent.value)
    self._disag_measure = nj.pure(self.agent.disag_measure)
    self._kl_measure = nj.pure(self.agent.kl_measure)
    if len(self.train_devices) == 1:
      kw = dict(device=self.train_devices[0])
      self._init_train = nj.jit(self._init_train, **kw)
      self._train = nj.jit(self._train, static=['rollouts'], **kw)
      self._report = nj.jit(self._report, **kw)
      self._value = nj.jit(self._value, **kw)
      self._disag_measure = nj.jit(self._disag_measure, **kw)
      self._kl_measure = nj.jit(self._kl_measure, **kw)
    else:
      kw = dict(devices=self.train_devices)
      self._init_train = nj.pmap(self._init_train, 'i', **kw)
      self._train = nj.pmap(self._train, 'i', static=['rollouts'], **kw)
      self._report = nj.pmap(self._report, 'i', **kw)
      self._value = nj.pmap(self._value, 'i', **kw)
      self._disag_measure = nj.pmap(self._disag_measure, 'i', **kw)
      self._kl_measure = nj.pmap(self._kl_measure, 'i', **kw)
    if len(self.policy_devices) == 1:
      kw = dict(device=self.policy_devices[0])
      self._init_policy = nj.jit(self._init_policy, **kw)
      self._policy = nj.jit(self._policy, static=['mode'], **kw)
    else:
      kw = dict(devices=self.policy_devices)
      self._init_policy = nj.pmap(self._init_policy, 'i', **kw)
      self._policy = nj.pmap(self._policy, 'i', static=['mode'], **kw)

  def _convert_inps(self, value, devices):
    if len(devices) == 1:
      value = jax.device_put(value, devices[0])
    else:
      check = tree_map(lambda x: len(x) % len(devices) == 0, value)
      if not all(jax.tree_util.tree_leaves(check)):
        shapes = tree_map(lambda x: x.shape, value)
        raise ValueError(
            f'Batch must by divisible by {len(devices)} devices: {shapes}')
      # TODO: Avoid the reshape?
      value = tree_map(
          lambda x: x.reshape((len(devices), -1) + x.shape[1:]), value)
      shards = []
      for i in range(len(devices)):
        shards.append(tree_map(lambda x: x[i], value))
      value = jax.device_put_sharded(shards, devices)
    return value

  def _convert_outs(self, value, devices):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if len(devices) > 1:
      value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
    return value

  def _convert_mets(self, value, devices):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if len(devices) > 1:
      value = tree_map(lambda x: x[0], value)
    return value

  def _next_rngs(self, devices, mirror=False, high=2 ** 63 - 1):
    if len(devices) == 1:
      return jax.device_put(self.rng.integers(high), devices[0])
    elif mirror:
      return jax.device_put_replicated(
          self.rng.integers(high), devices)
    else:
      return jax.device_put_sharded(
          list(self.rng.integers(high, size=len(devices))), devices)

  def _init_varibs(self, obs_space, act_space):
    varibs = {}
    rng = self._next_rngs(self.train_devices, mirror=True)
    dims = (self.batch_size, self.batch_length)
    data = self._dummy_batch({**obs_space, **act_space}, dims)
    data = self._convert_inps(data, self.train_devices)
    state, varibs = self._init_train(varibs, rng, data['is_first'])
    varibs = self._train(varibs, rng, data, state, rollouts=1, init_only=True)
    # obs = self._dummy_batch(obs_space, (1,))
    # state, varibs = self._init_policy(varibs, rng, obs['is_first'])
    # varibs = self._policy(
    #     varibs, rng, obs, state, mode='train', init_only=True)
    return varibs

  def _dummy_batch(self, spaces, batch_dims):
    spaces = list(spaces.items())
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data
