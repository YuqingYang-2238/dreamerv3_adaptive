import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
masked_mean = lambda x, m: (x * m).sum() / jnp.maximum(m.sum(), 1.0)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import expl
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')
    if config.disag_measure:
      # Plan2Explore ensemble, measured only: never a reward, never a gradient to the agent.
      self.disag = expl.Disag(self.wm, self.act_space, self.config, name='disag')
    if config.imag_adaptive:
      assert config.disag_measure and list(config.disag_target) == ['embed'], \
          'imag_adaptive needs disag_measure True and disag_target [embed]'

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    latent, prior = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    # One-step surprise of the world model: KL(posterior || prior), as train/dyn_kl_mean.
    kl = self.wm.rssm.dyn_loss(latent, prior, impl='kl', free=0.0)
    self.expl_behavior.policy(latent, expl_state)
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    if mode == 'eval':
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore':
      outs = expl_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    outs['log_kl'] = kl.astype(jnp.float32)
    if self.config.disag_measure:   # D_t on the current posterior and the chosen action
      outs['log_disag'] = self.disag.disagreement({**latent, 'action': outs['action']})
      # Prediction error one step late: the heads' mean prediction of e_t from the previous
      # (h, z, a) against the actual e_t. Meaningless on an episode's first step (drop it there).
      pred = self.disag.predictions({**prev_latent, 'action': prev_action}).mean(0)
      outs['log_disag_err'] = ((pred - embed.astype(jnp.float32)) ** 2).mean(-1)
    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state, rollouts=1):
    # rollouts: imagined trajectories per starting state (static under jit;
    # 1 = the baseline budget, 2 = the adaptive controller's extra budget).
    self.config.jax.jit and print(f'Tracing train function (rollouts={rollouts}).')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    # Disagreement of every imagined step: logged always, cuts rollouts if imag_adaptive.
    cutoff = self.disag.disagreement if self.config.disag_measure else None
    _, mets = self.task_behavior.train(
        self.wm.imagine, start, context, rollouts=rollouts, cutoff=cutoff)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    if self.config.disag_measure:
      # Last on purpose: nj.grad draws an rng key, so the actor-critic draws stay
      # identical to a run without the ensemble.
      metrics.update(self.disag.train({**context, 'embed': wm_outs['embed']}))  # disag_opt_*
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def value(self, data):
    """Critic value V(h_t) for every timestep of a replay batch, shape [B, T].

    Uses the same latent-inference path as WorldModel.loss (encoder ->
    rssm.observe) and the task critic's mean prediction (reward scale).
    Reads only parameters that train() already created, so it never adds
    state entries. Posterior sampling in rssm.observe draws from nj.rng(),
    so two calls with the same rng and world-model params give identical
    latents.
    """
    self.config.jax.jit and print('Tracing value function.')
    data = self.preprocess(data)
    embed = self.wm.encoder(data)
    prev_latent, prev_action = self.wm.initial(len(data['is_first']))
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, _ = self.wm.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    critic = self.task_behavior.ac.critics['extr']
    return critic.net(post).mean().astype(jnp.float32)

  def kl_measure(self, data):
    """World-model dynamics KL(posterior || prior) on a replay batch, same latent path as value().

    Mean over the batch's steps, excluding episode starts (their prior is the initial state).
    """
    self.config.jax.jit and print('Tracing kl_measure function.')
    data = self.preprocess(data)
    embed = self.wm.encoder(data)
    prev_latent, prev_action = self.wm.initial(len(data['is_first']))
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.wm.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    kl = self.wm.rssm.dyn_loss(post, prior, impl='kl', free=0.0)
    mask = 1.0 - data['is_first'].astype(jnp.float32)
    vis = mask * self.predator_visible(data)
    return {'dyn_kl': masked_mean(kl, mask), 'dyn_kl_visible': masked_mean(kl, vis)}

  @staticmethod
  def predator_visible(data):
    """1 where the prey's observation holds the predator (obs entry 5 = pred_x, sentinel -1)."""
    return (data['vector'][..., 5] >= 0).astype(jnp.float32)

  def disag_measure(self, data):
    """Ensemble disagreement on a replay batch, same latent path as value().

    D per (state, action) = unbiased variance across heads, mean over embedding
    dims. Returns its mean and variance over the B x T pairs, the ensemble-mean
    prediction MSE per dim against e_{t+1} (reset-crossing pairs excluded) and
    the embedding RMS. Reads only parameters that train() created.
    """
    self.config.jax.jit and print('Tracing disag_measure function.')
    data = self.preprocess(data)
    embed = self.wm.encoder(data)
    prev_latent, prev_action = self.wm.initial(len(data['is_first']))
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, _ = self.wm.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    preds = self.disag.predictions({**post, 'action': data['action']})  # [K, B, T, d]
    disag = preds.var(0, ddof=1).mean(-1)                                # [B, T]
    embed = embed.astype(jnp.float32)
    mask = 1.0 - data['is_first'][:, 1:].astype(jnp.float32)
    err = ((preds.mean(0)[:, :-1] - embed[:, 1:]) ** 2).mean(-1)
    vis = self.predator_visible(data)   # split by what the prey sees at t
    return {
        'disag': disag.mean(),
        'disag_var': disag.var(ddof=1),
        'disag_visible': masked_mean(disag, vis),
        'disag_hidden': masked_mean(disag, 1.0 - vis),
        'disag_mse': masked_mean(err, mask),
        'disag_mse_visible': masked_mean(err, mask * vis[:, :-1]),
        'embed_rms': jnp.sqrt((embed ** 2).mean()),
    }

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon, rollouts=1):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    if rollouts > 1:
      # Same starting states, repeated: every sample below (first action,
      # prior stoch, later actions) is drawn per batch element from one key
      # per step, so each repeat is an independent rollout, not a copy.
      tile = lambda x: jnp.tile(x, (rollouts,) + (1,) * (x.ndim - 1))
      start = {k: tile(v) for k, v in start.items()}
      first_cont = tile(first_cont)
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['dyn_kl_mean'] = self.rssm.dyn_loss(post, prior, impl='kl', free=0.0).mean()
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)
    # MACURA threshold state: circular list of past start-state percentiles, rounds so far.
    self.kappa_hist = nj.Variable(
        jnp.zeros, (config.imag_kappa_history,), jnp.float32, name='kappa_hist')
    self.kappa_rounds = nj.Variable(jnp.zeros, (), jnp.int32, name='kappa_rounds')

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context, rollouts=1, cutoff=None):
    def loss(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon, rollouts)
      cmets = {}
      if cutoff is not None:
        mask, cmets = self.cutoff(traj, cutoff)
        if self.config.imag_adaptive:
          traj['mask'] = mask
      loss, metrics = self.loss(traj, rollouts)
      metrics.update(cmets)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def cutoff(self, traj, disagreement):
    """MACURA rollout adaption (macura.py rollout_model_and_populate_sac_buffer).
    Returns mask [H+1, B]: 1 while every disagreement up to transition t is below kappa;
    last row 0. Threshold from the start states, averaged over past rounds."""
    D = sg(disagreement({k: traj[k] for k in ('deter', 'stoch', 'action')})).astype(jnp.float32)
    H = D.shape[0] - 1
    W = self.config.imag_kappa_history
    hist, n = self.kappa_hist.read(), self.kappa_rounds.read()
    b = self.config.imag_xi * jnp.percentile(D[0], self.config.imag_zeta)
    count = jnp.where(n < W, n, W - 1).astype(jnp.float32)  # current_border_count
    est = jnp.where(n > 0, hist.sum() / jnp.maximum(jnp.minimum(n, W), 1), 0.0)
    kappa = (b + count * est) / (count + 1)
    self.kappa_hist.write(hist.at[n % W].set(b))
    self.kappa_rounds.write(n + 1)
    acc = jnp.cumprod((D[:-1] < kappa).astype(jnp.float32), 0)
    mask = jnp.concatenate([acc, jnp.zeros_like(acc[:1])], 0)
    length = acc.sum(0)
    mets = {
        'imag_kappa': kappa, 'imag_b': b, 'imag_kappa_rounds': (n + 1).astype(jnp.float32),
        'imag_accepted_mean': length.mean(), 'imag_accepted_total': acc.sum(),
        'imag_frac_zero': (length == 0).mean(), 'imag_frac_full': (length == H).mean(),
        'imag_frac_len_1_4': ((length >= 1) & (length <= 4)).mean(),
        'imag_frac_len_5_9': ((length >= 5) & (length <= 9)).mean(),
        'imag_frac_len_10_14': ((length >= 10) & (length < H)).mean(),
        'imag_disag_mean': D[:-1].mean()}
    mets.update({f'imag_disag_d{d}': D[d].mean() for d in sorted({0, H // 3, 2 * H // 3, H - 1})})
    return sg(mask), mets

  def loss(self, traj, rollouts=1):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    mask = traj['mask'][:-1] if 'mask' in traj else None  # state t valid iff transition t accepted
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret, mask)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv, rollouts))
    if mask is not None:  # mean over accepted steps; 0 if none
      return (loss * mask).sum() / jnp.maximum(mask.sum(), 1.0), metrics
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv, rollouts=1):
    metrics = {}
    # Imagination budget actually used by this update (static shapes).
    metrics['imag_trajs'] = float(traj['action'].shape[1])
    metrics['imag_horizon'] = float(len(traj['action']) - 1)
    if rollouts > 1 and self.act_space.discrete:
      # Agreement between the two repeats of each start state. Independent
      # draws agree with probability sum_a p1(a) p2(a) under the policies at
      # the two repeats' (diverging) states, so the empirical rate must track
      # imag_half_match_expected; a duplicated rollout would give exactly 1.
      n = traj['action'].shape[1] // rollouts
      act = jnp.argmax(traj['action'], -1)
      probs = policy.probs_parameter()
      metrics['imag_half_match'] = (act[:, :n] == act[:, n:2 * n]).mean()
      metrics['imag_half_match_expected'] = (
          probs[:, :n] * probs[:, n:2 * n]).sum(-1).mean()
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    if 'mask' in traj:  # mean over accepted steps; 0 if none
      loss = (loss * sg(traj['weight']) * traj['mask']).sum() / jnp.maximum(traj['mask'].sum(), 1.0)
    else:
      loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    mask = traj['mask'][1:] if 'mask' in traj else None  # rejected t+1 -> bootstrap at t+1
    ret = jaxutils.lambda_return(rew, disc, value, self.config.return_lambda, mask)
    return rew, ret, value[:-1]
