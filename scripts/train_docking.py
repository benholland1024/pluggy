"""Train the RL docking policy (milestone 6).

SAC over DockEnv: continuous wheel/lift/arm control from detector-box +
proprioception observations, rewarded by progress toward the seat and the
electrical charging criterion. See envs/dock_env.py for why the observation
is shaped the way it is (deployable sensors only, synthetic detector with
the real one's clipping bias, real 4 Hz detector cadence).

Usage:
  MUJOCO_GL=egl uv run python scripts/train_docking.py                # 500k steps
  MUJOCO_GL=egl uv run python scripts/train_docking.py --timesteps 200000
  MUJOCO_GL=egl uv run python scripts/train_docking.py --resume runs/docking/<run>/final.zip

Checkpoints land in runs/docking/<stamp>/: best.zip (highest eval success
rate; the one eval_docking.py picks up by default) and final.zip.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from pluggybot.envs.dock_env import DockEnv

N_ENVS = 4                # 6 cores: 4 samplers + learner + OS
EVAL_EVERY = 20_000       # env steps between eval sweeps
EVAL_EPISODES = 24        # SimNotes: a handful of poses is a smoke test --
                          # 24 fixed seeds is the smallest defensible sweep
EVAL_SEED0 = 10_000       # fixed seeds so successive evals are comparable


class SuccessEval(BaseCallback):
  """Deterministic eval on fixed seeds; keeps the best-by-success checkpoint.

  Success RATE is the metric that matters (the scripted baseline is 33 %) --
  mean reward would happily prefer a policy that always gets close and never
  seats.
  """

  def __init__(self, out_dir: Path):
    super().__init__()
    self.out_dir = out_dir
    self.eval_env = None
    self.next_eval = EVAL_EVERY
    self.best = -1.0

  def _run_eval(self) -> tuple[float, float]:
    if self.eval_env is None:
      self.eval_env = DockEnv()
    wins, dists = 0, []
    for ep in range(EVAL_EPISODES):
      obs, _ = self.eval_env.reset(seed=EVAL_SEED0 + ep)
      done = trunc = False
      while not (done or trunc):
        action, _ = self.model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = self.eval_env.step(action)
      wins += bool(info["success"])
      dists.append(info["distance"])
    return wins / EVAL_EPISODES, float(np.median(dists))

  def _on_step(self) -> bool:
    if self.num_timesteps >= self.next_eval:
      self.next_eval += EVAL_EVERY
      rate, med_dist = self._run_eval()
      self.logger.record("eval/success_rate", rate)
      self.logger.record("eval/median_final_dist", med_dist)
      print(f"[eval @ {self.num_timesteps:>8d}] success {rate:5.1%}  "
            f"median final dist {med_dist * 1000:6.1f} mm", flush=True)
      if rate >= self.best:
        self.best = rate
        self.model.save(self.out_dir / "best.zip")
    return True

  def _on_training_end(self) -> None:
    if self.eval_env is not None:
      self.eval_env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--timesteps", type=int, default=500_000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--resume", default=None, help="checkpoint .zip to continue from")
  parser.add_argument("--lr", type=float, default=None,
                      help="override learning rate (e.g. 1e-4 when resuming a "
                           "run that diverged -- the policy found the basin, a "
                           "hot lr kicked it back out)")
  parser.add_argument("--out", default=None, help="output dir (default: runs/docking/<stamp>)")
  args = parser.parse_args()

  out_dir = Path(args.out or f"runs/docking/{time.strftime('%Y%m%d_%H%M%S')}")
  out_dir.mkdir(parents=True, exist_ok=True)

  venv = VecMonitor(SubprocVecEnv([DockEnv for _ in range(N_ENVS)]))
  if args.resume:
    overrides = {"learning_rate": args.lr} if args.lr else {}
    model = SAC.load(args.resume, env=venv, device="cuda", **overrides)
    print(f"resumed from {args.resume}" + (f" (lr {args.lr})" if args.lr else ""))
  else:
    model = SAC(
      "MlpPolicy", venv,
      policy_kwargs=dict(net_arch=[256, 256]),
      learning_rate=3e-4,
      buffer_size=500_000,
      batch_size=256,
      gamma=0.99,
      tau=0.005,
      train_freq=1,          # one vec-step (N_ENVS transitions) per update pair
      gradient_steps=2,
      learning_starts=5_000,
      seed=args.seed,
      device="cuda",
      verbose=0,
    )

  t0 = time.time()
  model.learn(total_timesteps=args.timesteps, callback=SuccessEval(out_dir),
              reset_num_timesteps=not args.resume, progress_bar=False)
  model.save(out_dir / "final.zip")
  print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}/final.zip")


if __name__ == "__main__":
  main()
