# Signal-learning experiment plan (living, reusable)

Question: does a learning signal measured during DreamerV3 training respond when the predator's
strategy switches (claim 1), and does it settle as performance recovers (claim 2)? Everything below
is signal-agnostic except **§6 Signal under test**: to try a new signal, keep the protocol and
replace that one section.

- Instance 1 (value change, Oct 13-14 2026): results and frozen as-planned copy in
  `logdir/1013_signal_learning/` (`1014_experiment_summary.md`, `testing_signal_learning_plan.md`).
- Instance 2 (Plan2Explore ensemble disagreement, Oct 2026): `logdir/1016_disagreement/`. A discovery
  test: the signal reductions are design choices, a flat result is a valid outcome.
- Instance 2b (settling pilot for the visible-step disagreement, Oct 2026): `logdir/1019_settling/`.
  Claim 2 only, from the Instance 2 trunk with a regenerated replay (§6).
- **Rule since Instance 2: score every signal on the recent-data batch** (`*_recent`, the last 20k
  steps). All-replay values are logged but not analysed: new data is a few percent of replay, so
  all-replay batches showed nothing at either predator switch in Instance 2 while the recent batch
  did (ensemble prediction error 0.086 -> 0.198 at the first predator appearance).
- Each new signal gets its own folder `logdir/<date>_<signal>/` with its own summary.
- `logdir/` is gitignored; this file and the code are what is versioned.
- Commands below use `EXP=logdir/<date>_<signal>` (set it once per experiment).

## 0. Design at a glance

| Item | Choice |
|---|---|
| Trunk | one pretrained agent, 200k steps in one logdir: 100k without predator, then 100k with the Default predator, no step cost (§2); every run below branches from it |
| Conditions | Default -> Default (control) and Default -> Ambush, 3 seeds each, 150k steps |
| Control of the comparison | identical except predator strategy; seeds paired across conditions |
| Frozen-agent check | trunk checkpoint evaluated against both strategies, no updates, 10 unscored + 50 scored episodes per strategy and seed (§3) |
| In-run frozen evaluations | x = 0 and every 2k continuation steps, 10 unscored + 10 scored episodes (Instance 2; Instance 1 used every 20k, 10 + 20) |
| Signal batch | recent-data batch (last 20k steps) is the primary; all-replay values are companions, not analysed (rule since Instance 2) |
| Predator speed | 0.2 for both strategies (`predator_prey_forward_speed_ratio` 0.4 x prey 0.5) |
| What 3 vs 3 can show | variation during continuation from one trunk, not across independently pretrained agents |

## 1. Environment (as implemented)

Shared: world 21_05; prey max speed 0.5; one RL step = 10 ticks of 0.025 s; 300-step limit; goal radius
0.1, +1, terminal; capture = mutual line of sight and distance <= 0.1: -1, 0.5 s cooldown, non-terminal;
14-float observation (predator fields only while visible). **Survival / success** = goal reached without
capture (`episode/success` in training, `eval/success_rate` in evaluation).
Since 2026-10-16 the agents have Mice bodies (`prey_env/geometry.py`): the prey collides as a disc of
radius 0.015 and never overlaps obstacles, the predator body is for drawing and line of sight only; line
of sight is "any part of the other body visible"; heading 0 = +x; speed bins 0-1 (forward only, bin 0 =
pause). Earlier checkpoints are not comparable with runs on this environment.

- **Default = the paper's Decoupled robot** (`prey_env/Predator.py`): spawns on a random free cell hidden
  from the entrance and not adjacent to an obstacle (114 cells), walks to random interior cells not
  adjacent to walls or obstacles (100 cells), redraws on arrival, never reads the prey.
- **Ambush, pursuit version** (`prey_env/AmbushPredator.py`): starts at one of five fixed cells
  `[103, 251, 144, 152, 258]` facing the entrance, still. Ambush region = 3 cells. Trigger = the prey's
  true position entering the region (god view); it then pursues while the prey is inside, returns when
  it leaves. Next episode's cell is drawn with weight 1 + (episodes among the last 10 in which the prey
  came within 3 cells of that cell). `predator_surge_speed` exists (default 1.0) and is not used.
- `--env.prey.predator_strategy {default, ambush, none}`; `none` removes the predator.

## 2. Step 0 - trunk (pretraining)

Two stages in one logdir, no step cost (body environment, verified 2026-10-16: goal rate 1.00 by 60k
without a predator, survival 0.75-0.82 at 200k with the Default predator). The signal under test is
trained and logged from step 0 (`--disag_measure True` for Instance 2).

| Stage | Steps | Predator | Logdir |
|---|---|---|---|
| 1 | 0-100k | none | `trunk_s0` |
| 2 | 100k-200k | Default | same logdir, relaunched with `--run.steps 200000` (resume loads checkpoint + replay) |

```bash
python main_train.py --logdir $EXP/trunk_s0 --seed 0 --jax.platform gpu --disag_measure True \
    --env.prey.predator_strategy none --run.steps 100000
python main_train.py --logdir $EXP/trunk_s0 --seed 0 --jax.platform gpu --disag_measure True \
    --env.prey.predator_strategy default --run.steps 200000
```

Checks at 200k: training survival over the last 10k steps >= 0.6; `checkpoint.ckpt` holds `value_ref`
with `step == 200000` and, for Instance 2, the `agent/disag/` ensemble parameters; `run_info.jsonl` has
one line per stage with the right `predator_strategy`.

Instance 1 (old point environment) needed three phases with a step cost of -0.02 that was removed at
~122k; that recipe is kept in `logdir/1013_signal_learning/testing_signal_learning_plan.md`. Why it was
needed there (all tried, same code, seed 0):
- Default predator from scratch, no cost: the agent hides near the entrance (0% goal rate, every episode
  a 300-step timeout); the reward head learns the rare +1 too late.
- Default predator from scratch with cost -0.005 or -0.02: the policy collapses before the reward head
  predicts the goal. The baseline's chasing predator had made the goal worth reaching as an escape; the
  decoupled one does not.
- No-predator warm start with cost -0.02 learns the goal in 30-50k steps (a straight dash).
- Adding the Default predator with the cost on plateaus at ~0.62 survival (the dash stays optimal);
  removing the cost lets the agent learn avoidance (training survival 0.75 at 200k).

**Archive the 200k `checkpoint.ckpt` and `replay/` and the trunk logdir before shutting the machine
down**: metrics files alone cannot test a new signal on this trunk.

## 3. Step 1 - frozen-agent evaluation

Trunk checkpoint vs both strategies, no learning, 10 unscored + 50 scored episodes, seeds 1-3
(episode-level signal values are kept in `summary.json`, Figure 2):

```bash
for s in 1 2 3; do for p in default ambush; do
  python main_eval.py --logdir $EXP/trunk_s0 --strategy $p --burn-in 10 --episodes 50 \
      --seed $s --no-video --jax-platform gpu --out $EXP/trunk_s0/eval0_${p}_s$s
done; done
```

Report success, capture rate, captures/episode and return per strategy and Ambush - Default with a
seed-level CI; no pass/fail threshold. Do not pool episodes across seeds for the CI (they are not
independent training seeds). Not implemented: world-model KL on the evaluation sequences.

## 4. Step 2 - six continuations

```bash
for s in 1 2 3; do for p in default ambush; do
  python main_branch.py $EXP/trunk_s0 $EXP/cont2_s${s}_$p
done; done
for s in 1 2 3; do for p in default ambush; do
  setsid nohup python main_train.py --logdir $EXP/cont2_s${s}_$p --seed $s \
      --jax.platform gpu --disag_measure True --run.steps 350000 --env.prey.predator_strategy $p \
      --run.eval_every 2000 --run.eval_eps 10 \
      > $EXP/cont2_s${s}_$p/train.log 2>&1 < /dev/null &
  sleep 30
done; done
```

`--run.steps` is absolute (branch step + 150000). Six concurrent runs fit one 23 GB A10 (about 14 GB).
`main_branch.py` copies the checkpoint and a step-exact replay cut, writes `branch.json`, refuses a
non-empty target. Only the seed (agent, env and replay sampling) and the strategy differ between runs.
In-run evaluations: x = 0 and every 2k steps, 10 unscored + 10 scored episodes, RNG restored, the eval
env's ambush window cleared (the 10 unscored episodes refill the Ambush predator's 10-episode cell-choice
window, so scored episodes face the adapted predator as in training). Budget on the A10: ~1 h of training
plus ~1.5 h of evaluations per run.

Machine notes: conda env `dreamerv3-prey` needs `jax[cuda12_pip]==0.4.20` and
`nvidia-cudnn-cu12==8.9.7.29`. rsync the repo **including `.git`**, otherwise `run_info.jsonl` records no
commit. Use `pgrep -f '[m]ain_train.py'` (a bare pattern matches its own ssh command line).

## 5. Logging common to every signal

| Quantity | Key | Cadence |
|---|---|---|
| Eval success, capture, captures/ep, return, surge rate | `eval/*` | x = 0, every 2k (Instance 2) |
| Training survival, length | `episode/{success,score,length}` | per episode |
| Ensemble disagreement, variance, prediction MSE, embed RMS (Instance 2) | `learning/disag`, `learning/disag_var`, `learning/disag_mse`, `learning/embed_rms` (+ `_recent`) | with the marker, every 2k |
| Per-episode disagreement (training and frozen eval) | `episode/disag_{mean,var}`, `eval/disag_{mean,var}` | per episode / per evaluation |
| World-model KL(posterior ‖ prior) on the marker batches (all runs, added 2026-10-19) | `learning/dyn_kl`, `learning/dyn_kl_recent` | with the marker, every 2k |
| Visible/hidden split of the marker signals: pairs where the prey's observation holds the predator (`vector[5] >= 0`) vs not | `learning/disag_{visible,hidden}(_recent)`, `learning/disag_mse_visible(_recent)`, `learning/dyn_kl_visible(_recent)` | with the marker, every 2k |
| Per-episode visible-step disagreement (training and frozen eval) | `episode/disag_visible_mean`, `eval/disag_visible_mean` | per episode / per evaluation |

Batch rule: analyse the `_recent` keys; the all-replay keys exist for completeness only.
| Per-step world-model KL from the policy, episode means (first step excluded) | `episode/kl_mean`, `eval/kl_mean`, `kl_mean` in `main_eval` summaries | per episode / per evaluation |
| Unclamped dynamics KL (companion diagnostic) | `train/dyn_kl_mean` | batch mean every 10th update, logged per ~30 s (~500 steps) |
| Clamped dynamics loss, model loss | `train/{dyn_loss_mean,model_loss_mean}` | same |
| Post-branch share of replay | `learning/replay_new_frac` | with the signal |

## 6. Signal under test (swap this section)

**Instance 2: Plan2Explore ensemble disagreement** (Sekar et al. 2020, Sec. 3.1; the `Disag` ensemble in
`dreamerv3/expl.py`, measured only, `--disag_measure True`). A discovery test: whether the ensemble's
spread carries information about the learning stage. No positive result is guaranteed.

- Ensemble: K = 5 independently initialised MLPs (2 hidden layers, 640 units) predicting the next
  observation embedding `e_{t+1} = Enc(o_{t+1})` (640-d) from `[h_t, flatten(z_t), a_t]`, the posterior
  state and the executed action. Trained on real replay transitions with their own optimizer
  (`expl_opt`), stop-gradient on inputs and target, per-dimension MSE, pairs crossing an episode reset
  excluded, no bootstrap masks. The encoder, RSSM, actor and critic never receive its gradients, and it
  never produces a reward; the ensemble update runs last in `Agent.train` so the actor-critic RNG draws
  equal those of a run without it.
- Per (state, action) pair: `D_t = 1/(d(K-1)) sum_i ||ê_{t+1}^(i) - mean_i ê_{t+1}^(i)||^2`, the unbiased
  variance across heads averaged over embedding dims. Disagreement at one checkpoint, not a change over time.
- Two candidate signals from the N = 16 x 64 pairs of a batch: **signal 1** `learning/disag` = mean of
  `D_n`; **signal 2** `learning/disag_var` = variance of `D_n` over the pairs ("variance of variance", how
  unevenly uncertainty is spread over states). Same marker cadence (every 2k steps) and the same replay
  and recent batches as value change, which stays logged as a companion.
- Companions on the same batch: `learning/disag_mse` (ensemble-mean prediction error per dim, to tell
  agreement from accuracy) and `learning/embed_rms` (embedding scale). Per training episode and per
  frozen-eval episode the policy's per-step `log_disag` gives `episode/disag_{mean,var}` and
  `eval/disag_{mean,var}` (Figure 2 uses the episode-level values of the frozen trunk).
- Readings: D rises after the switch in Ambush, not in controls, then falls with prediction error while
  survival recovers -> predictive learning (claims 1 and 2). D falls while error stays high -> shared
  mistakes, not learning. D flat -> the ensemble does not detect this change; still a result. Controls
  move as much as Ambush -> ordinary variability, no claim. Signal 2 responds where signal 1 does not
  (or the reverse) -> report both, claim neither alone. Embedding scale or visitation changes can move D
  by themselves: check `embed_rms` first; re-scoring fixed held-out sequences per checkpoint is the
  fallback (not implemented).
- To add a signal: compute it in `Agent.disag_measure`-style code (`dreamerv3/agent.py`, jitted in
  `jaxagent.py`), call it in the marker block of `dreamerv3/embodied/run/train.py` (same `mets` dict, so
  it inherits `ref_age_*` and the `_recent` variant), then register it in `QUANTITIES`
  (`scores/branch_stats.py`), `PANELS` (`scores/plot_switch_zoom.py`), `METRICS` (`scores/plot_timeline.py`)
  and the panels of `scores/plot_branches.py` / `scores/plot_pretrain.py`.

**Instance 2b: settling pilot for the visible-step disagreement** (`logdir/1019_settling/`).

Instance 2 found, after the fact, that the ensemble marks the Ambush strategy as unfamiliar only on
steps where the prey sees the predator: frozen-trunk visible-step disagreement +19/+38/+34 % per seed
under Ambush, hidden-step means equal, and 79 % of the top-10 % disagreement steps have the predator in
view under Ambush (base rate 47 %; Default 56 %). That answers claim 1 for a frozen agent. This pilot
tests claim 2 only: does `learning/disag_visible_recent` fall back to the control level while the
prey re-adapts?

- Signal: mean of `D_t` over the pairs of the recent batch where the predator is in the prey's
  observation (`vector[5] >= 0`); `disag_hidden_recent` is the control. Same ensemble as Instance 2.
- Trunk: the Instance 2 trunk checkpoint (200k steps, survival 0.70). Its replay was lost, so it is
  regenerated by running the frozen trunk policy for 100k steps on the Default predator with
  `--run.train_ratio 0` (no updates; 1,024 random prefill steps enter the replay). The branch point is
  therefore step 300k with the 200k parameters, and the replay holds only Default data from the frozen
  policy. Documented in `run_info.jsonl`.
- Six continuations of 150k steps (3 seeds x {Default, Ambush}), markers every 2k, in-run evaluations
  at x = 0 and every 50k (20 episodes).
- Pre-registered: (H3a) `disag_visible_recent` Ambush > Default in 0-10k in all three seed pairs,
  Welch CI excluding 0; (H3b) the Ambush excess returns to the control level by 120-150k while survival
  recovers; (control) `disag_hidden_recent` shows no Ambush-Default difference in any window. The
  noise reference is reported, not used as a criterion.
- Readings: H3a and H3b hold -> claim 2 supported for the visible-step signal. H3a fails -> the 2k
  marker is slower than the adaptation; shorten the interval. H3b fails with survival recovered ->
  the signal tracks the predator's presence, not learning.
- Deferred: a pre-registered enrichment test on fresh trunks, two-trunk replication, visible-only
  split logged in the in-run evaluation files.

```bash
EXP=logdir/1019_settling   # trunk_regen/ holds checkpoint.ckpt, config.json, run_info.jsonl of 1016 trunk_s0
python main_train.py --logdir $EXP/trunk_regen --seed 0 --jax.platform gpu --disag_measure True \
    --env.prey.predator_strategy default --run.steps 300000 --run.train_ratio 0 \
    --run.eval_every 1e9 --run.value_change_every 0 --run.eval_initial False
for s in 1 2 3; do for p in default ambush; do python main_branch.py $EXP/trunk_regen $EXP/cont_s${s}_$p; done; done
for s in 1 2 3; do for p in default ambush; do
  setsid nohup python main_train.py --logdir $EXP/cont_s${s}_$p --seed $s --jax.platform gpu \
      --disag_measure True --run.steps 450000 --env.prey.predator_strategy $p \
      --run.eval_every 50000 --run.eval_eps 20 > $EXP/cont_s${s}_$p/train.log 2>&1 < /dev/null & sleep 30; done; done
# analysis
python scores/branch_stats.py $EXP/cont_s{1,2,3}_default $EXP/cont_s{1,2,3}_ambush --out $EXP/branch_stats
for m in disag_visible_recent disag_hidden_recent; do
  python scores/plot_timeline.py --metric $m --trunk $EXP/trunk_regen --runs $EXP/cont_s{1,2,3}_default $EXP/cont_s{1,2,3}_ambush --out $EXP/timeline_$m.png; done
python scores/plot_switch_zoom.py --trunk $EXP/trunk_regen --runs $EXP/cont_s{1,2,3}_default $EXP/cont_s{1,2,3}_ambush --out $EXP/switch_zoom.png
```

Instance 1 (temporal value change): `learning/value_change` = mean |V_k - V_{k-2000}| over a 16 x 64
batch (same latents scored by the current and the 2,000-step-old critic), `_recent` variant from the
last-20k-steps buffer, companions `value_new_mean`, `value_delta_std`. Full text in
`logdir/1013_signal_learning/testing_signal_learning_plan.md`.

## 7. Analysis (signal-agnostic)

1. **Branch check:** every `replay/size` starts at S0 - 63 x streams; reference age exactly 2000 (or the
   chosen lag); no traceback; checkpoint written; for Instance 2 the branch checkpoint holds the
   `agent/disag/` ensemble parameters (`jaxagent.load` asserts it).
2. **Early response, per seed:** unsmoothed paired differences Ambush_i - Default_i over the first 20k
   steps (`plot_switch_zoom.py`); one 20k mean per seed is the replicate, CI by t over seeds (n = 3,
   wide). Never treat time points within a run as replicates. `branch_stats.py` adds the pre-registered
   windows 0-30k (primary), 0-10k and 120-150k, an exact permutation p (minimum 0.05 at 3 vs 3) and a
   Welch CI. The 20k paired view was added after a first look: exploratory.
3. **Noise reference:** the largest change between neighbouring same-size windows within a control run;
   a difference smaller than that is not read as a response. No fixed threshold.
4. **Settling (claim 2):** 20k-window means of the signal and of survival per seed, Ambush vs Default,
   0-150k; compare the signal's decline with the survival curve and with the same decline in the controls.
   Currently done ad hoc, not scripted.
5. **Performance:** training survival per 10k, T50 (first 2k bin where the 5-bin smoothed survival
   reaches the midpoint between the first 10k and the last 30k), eval success at x = 0 and the end.
6. **Companions:** critic mean and KL, same windows and paired differences.

| Pattern | Reading | Next |
|---|---|---|
| Early difference beyond the noise reference, settles with survival | claims 1 and 2 supported | independent replications (more pretrained trunks) |
| Survival recovers, signal indistinguishable from controls | signal insensitive in this setup | new signal, or a stronger switch |
| Signal up, no recovery by 150k | claim 1 only | extend both conditions to a common budget |
| Signal moves, control signal moves equally | ordinary learning variability | report; do not claim |
| Branch check fails or controls show a transient | resume artifact | fix before reading anything |

## 8. Commands (analysis, run from the repo root)

```bash
EXP=logdir/<date>_<signal>
python scores/branch_stats.py $EXP/cont2_s{1,2,3}_default $EXP/cont2_s{1,2,3}_ambush --out $EXP/branch_stats
python scores/plot_branches.py $EXP/cont2_s{1,2,3}_default $EXP/cont2_s{1,2,3}_ambush --out $EXP/branches.png
python scores/plot_switch_zoom.py --trunk $EXP/trunk_s0 \
    --runs $EXP/cont2_s{1,2,3}_default $EXP/cont2_s{1,2,3}_ambush --out $EXP/switch_zoom.png
for m in disag disag_var disag_mse; do
  python scores/plot_timeline.py --metric $m --trunk $EXP/trunk_s0 \
      --runs $EXP/cont2_s{1,2,3}_default $EXP/cont2_s{1,2,3}_ambush --out $EXP/timeline_$m.png; done
python scores/plot_pretrain.py $EXP/trunk_s0 --phases 0:100000:no_predator 100000:200000:decoupled_predator \
    --out $EXP/pretrain.png
python scores/plot_frozen_disag.py $EXP/trunk_s0/eval0_default_s{1,2,3} $EXP/trunk_s0/eval0_ambush_s{1,2,3} \
    --out $EXP/frozen_disag.png
python scores/plot_eval_trajs.py $EXP/cont2_s1_default $EXP/cont2_s1_ambush --x 0 80000 150000
```

Deliverables of Instance 2: `pretrain.png` (initial learning), `frozen_disag.png` (before adaptation),
`branches.png` + `switch_zoom.png` (continued learning, early response), `timeline_*.png`,
`branch_stats/branch_stats.md` (windows, permutation p, CI, noise reference), `eval_trajs_*.png`, and a
hand-written `summary.md` stating which reading of §6 each signal matched.

## 9. Checklist for a new signal

1. Implement and log the signal (§6); commit and tag the code; sync with `.git`.
2. Pretrain a new trunk (§2) or, if the 200k checkpoint + replay were archived, branch from it.
3. Frozen evaluation (§3), six continuations (§4), analysis (§7-8).
4. Write `logdir/<date>_<signal>/<summary>.md`; update this file only if the protocol itself changed.
