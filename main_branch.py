"""Creates a branch logdir that continues training from a finished run's checkpoint.

  python main_branch.py logdir/baseline_21_05_seed0 logdir/cont_s1_ambush

Copies checkpoint.ckpt and the replay chunks collected up to the checkpoint's
step (later chunks and partial re-saves are dropped; the chunk covering the
step is truncated), so the branch resumes with exactly the data the agent had
seen. The source is only read. Writes branch.json and prints the launch command.
"""
import json
import os
import pickle
import shutil
import sys

import numpy as np

ZERO_UUID = '0' * 22  # successor id of the newest chunk (embodied.uuid(0))


def chunk_files(directory):
  """{uuid: [(length, filename), ...]}, uuids in chunk creation order."""
  by_uuid = {}
  for name in sorted(os.listdir(directory)):
    if name.endswith('.npz'):
      _, uuid, _, length = name[:-4].split('-')
      by_uuid.setdefault(uuid, []).append((int(length), name))
  return by_uuid


def copy_replay(src, dst, steps):
  """Copies the chunks holding the first `steps` steps of src into dst."""
  os.makedirs(dst)
  kept, total = [], 0
  for uuid, files in chunk_files(src).items():
    if total >= steps:
      break
    files.sort()
    # Smallest re-save of this chunk that reaches `steps`, else the full chunk.
    length, name = next(((n, f) for n, f in files if total + n >= steps), files[-1])
    keep = min(length, steps - total)
    if keep == length:
      shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    else:
      with np.load(os.path.join(src, name)) as data:
        rows = {k: data[k][:keep] for k in data.files}
      name = f"{name.split('-')[0]}-{uuid}-{ZERO_UUID}-{keep}.npz"
      np.savez_compressed(os.path.join(dst, name), **rows)
    kept.append(name)
    total += keep
  # A regenerated replay (collection-only run after a lost buffer) holds fewer steps than the
  # checkpoint's step count; then every chunk is kept. More steps than the checkpoint is an error.
  assert total <= steps, f'replay holds {total} steps, checkpoint is at {steps}'
  return kept, total


def main(source, branch):
  assert not (os.path.isdir(branch) and os.listdir(branch)), f'{branch} is not empty'
  with open(os.path.join(source, 'checkpoint.ckpt'), 'rb') as f:
    ckpt = pickle.load(f)
  step, updates = int(ckpt['step']), int(ckpt['updates'])
  del ckpt
  kept, replay_steps = copy_replay(
      os.path.join(source, 'replay'), os.path.join(branch, 'replay'), step)
  shutil.copy2(
      os.path.join(source, 'checkpoint.ckpt'), os.path.join(branch, 'checkpoint.ckpt'))
  info = dict(source=source, step=step, updates=updates,
              replay_steps=replay_steps, replay_files=kept)
  with open(os.path.join(branch, 'branch.json'), 'w') as f:
    json.dump(info, f, indent=1)
  print(f'{branch}: checkpoint at step {step} ({updates} updates), '
        f'{len(kept)} replay chunks holding {replay_steps} steps')
  print(f'python main_train.py --logdir {branch} --seed <seed> --jax.platform gpu '
        f'--run.steps {step + 150000} --env.prey.predator_strategy <default|ambush>')


if __name__ == '__main__':
  main(*sys.argv[1:3])
