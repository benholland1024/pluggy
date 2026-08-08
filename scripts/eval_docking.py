"""End-to-end docking eval in room_1: scripted DOCK vs RL policy, same trials.

The milestone-6 scoreboard. Each trial parks the robot at a docking hand-off
pose in front of a real room_1 outlet with the error statistics the mission
actually delivers (pose jitter vs the ideal standoff, a believed landmark
2 cm off truth, a whisper of odometry drift), then lets a controller try to
plug in with the REAL YOLO detector on dock_eye. Success is the electrical
charging criterion -- identical trials, identical sensors, identical verdict
for both controllers, so the numbers are comparable.

Usage:
  MUJOCO_GL=egl uv run python scripts/eval_docking.py --trials 24
  MUJOCO_GL=egl uv run python scripts/eval_docking.py --controller rl --policy runs/docking/<run>/best.zip
  MUJOCO_GL=egl uv run python scripts/eval_docking.py --controller scripted --trials 12
"""

import argparse
import glob
import importlib.util
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from pluggybot.docking.contact import charging_contact, feelers_touching
from pluggybot.envs.dock_env import (
  CTRL_DT, DETECT_PERIOD, DROOP_COMP, PLUG_AXIS_Z0, PLUG_LATERAL, DockEnv,
)
from pluggybot.perception.outlet_spotter import latest_weights

_LIFECYCLE = Path(__file__).parent / "lifecycle.py"

# -- trial error model (the "2 cm standoff jitter" protocol, made explicit) --
STANDOFF = 0.6              # m: the hand-off distance GO_CHARGE targets
POSE_JITTER_XY = 0.02       # m: true parked pose vs ideal standoff
POSE_JITTER_YAW = math.radians(1.0)
LANDMARK_ERR_XY = 0.02      # m: believed outlet position vs truth
LANDMARK_ERR_Z = 0.015      # m: believed outlet height vs truth (the gap)
ODOM_DRIFT_XY = 0.005       # m: believed robot pose vs truth (measured small)
ODOM_DRIFT_YAW = math.radians(0.3)

TRIAL_TIMEOUT = 60.0        # sim seconds per attempt
OUTLETS = ("outlet_a", "outlet_b", "outlet_c")


def load_lifecycle():
  spec = importlib.util.spec_from_file_location("lifecycle", _LIFECYCLE)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def make_trial(rng, model) -> dict:
  """One trial's ground truth + error draws (shared by both controllers)."""
  name = OUTLETS[rng.integers(len(OUTLETS))]
  body = model.body(name)
  ox, oy, oz = (float(v) for v in body.pos)
  # outlet local +x is the outward normal; bodies are yaw-only rotations
  w, x, y, z = (float(v) for v in body.quat)
  facing = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  nx, ny = math.cos(facing), math.sin(facing)
  hd = math.atan2(-ny, -nx)                     # docking heading: into the wall
  sx = ox + STANDOFF * nx - PLUG_LATERAL * math.sin(hd)
  sy = oy + STANDOFF * ny + PLUG_LATERAL * math.cos(hd)
  u = rng.uniform
  return {
    "outlet": name, "oz": oz,
    "true_axle": (sx + u(-POSE_JITTER_XY, POSE_JITTER_XY),
                  sy + u(-POSE_JITTER_XY, POSE_JITTER_XY)),
    "true_yaw": hd + u(-POSE_JITTER_YAW, POSE_JITTER_YAW),
    "believed_outlet": (ox + u(-LANDMARK_ERR_XY, LANDMARK_ERR_XY),
                        oy + u(-LANDMARK_ERR_XY, LANDMARK_ERR_XY),
                        oz + u(-LANDMARK_ERR_Z, LANDMARK_ERR_Z)),
    "believed_heading": hd + u(-math.radians(1.0), math.radians(1.0)),
    "odom_err": (u(-ODOM_DRIFT_XY, ODOM_DRIFT_XY),
                 u(-ODOM_DRIFT_XY, ODOM_DRIFT_XY),
                 u(-ODOM_DRIFT_YAW, ODOM_DRIFT_YAW)),
  }


def setup_sim(lc, weights, trial):
  """A Lifecycle harness parked mid-DOCK: robot at the trial's true pose,
  odometry at truth+drift, one believed landmark as the target."""
  sim = lc.Lifecycle(headless=True, max_sim_time=1e9, weights=weights,
                     explore_budget=1e9)
  tx, ty = trial["true_axle"]
  yaw = trial["true_yaw"]
  sim.data.qpos[0:3] = [tx + 0.08 * math.cos(yaw), ty + 0.08 * math.sin(yaw), 0.045]
  sim.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(sim.model, sim.data)
  ex, ey, eyaw = trial["odom_err"]
  sim.reckoner.x, sim.reckoner.y = tx + ex, ty + ey
  sim.reckoner.theta = yaw + eyaw
  bx, by, bz = trial["believed_outlet"]
  sim.target = sim.landmarks.add_sighting(bx, by, bz, seen_from=(tx, ty))
  sim.state = "DOCK"
  sim.dock_stage = "align"
  sim.stage_t0 = sim.data.time
  sim.model.opt.timestep = lc.DOCK_TIMESTEP
  sim.model.actuator_forcerange[sim.model.actuator("arm").id] = \
      [-lc.DOCK_ARM_FORCE, lc.DOCK_ARM_FORCE]
  return sim


def perceive_lite(sim):
  """Odometry only -- the map plays no part in terminal docking, and the
  reflex is disabled in DOCK anyway; both runners share this."""
  d, m = sim.data, sim.model
  sim.reckoner.update(
    d.qpos[sim.left_adr], d.qpos[sim.right_adr],
    gyro_yaw_rate=d.sensordata[sim.gyro.adr[0] + 2], dt=m.opt.timestep)


def run_scripted(lc, weights, trial) -> dict:
  sim = setup_sim(lc, weights, trial)
  while sim.state == "DOCK" and sim.data.time < TRIAL_TIMEOUT:
    perceive_lite(sim)
    v, w = sim.dock()
    sim.actuate(v, w)
    mujoco.mj_step(sim.model, sim.data)
  return {"success": bool(sim.docked), "t": sim.data.time}


def yolo_detection(sim):
  """One real dock_eye look, in the synthetic detector's tuple format."""
  r = sim.dock_spotter.renderer
  r.update_scene(sim.data, camera="dock_eye")
  rgb = r.render()
  r.enable_depth_rendering()
  r.update_scene(sim.data, camera="dock_eye")
  depth = r.render()
  r.disable_depth_rendering()
  res = sim.dock_spotter.detector.predict(
    np.ascontiguousarray(rgb[:, :, ::-1]), conf=0.5, verbose=False)[0]
  if not len(res.boxes):
    return 0.0, 0.0, 0.0, 0.0, 0.0, False
  u, v, bw, bh = (float(x) for x in res.boxes.xywh[0])
  iu, iv = min(int(u), 639), min(int(v), 359)
  rng = float(np.median(depth[max(0, iv - 1):iv + 2, max(0, iu - 1):iu + 2]))
  return u, v, bw, bh, rng, True


def run_rl(lc, weights, trial, policy) -> dict:
  from pluggybot.control import slew, wheel_targets, wrap_angle
  sim = setup_sim(lc, weights, trial)
  m, d = sim.model, sim.data
  lift_qadr = m.joint("lift_joint").qposadr[0]
  arm_qadr = m.joint("arm_joint").qposadr[0]
  lift_act, arm_act = m.actuator("lift").id, m.actuator("arm").id
  lvad = m.joint("left_wheel_joint").dofadr[0]
  rvad = m.joint("right_wheel_joint").dofadr[0]

  # align-stage equivalent: lift preset to the BELIEVED height (as in training)
  bx, by, bz = trial["believed_outlet"]
  lift_cmd = float(np.clip(bz - PLUG_AXIS_Z0 + DROOP_COMP, 0.0, 0.31))
  d.qpos[lift_qadr] = lift_cmd
  d.ctrl[lift_act] = lift_cmd
  arm_cmd = 0.0
  mujoco.mj_forward(m, d)
  for _ in range(200):
    mujoco.mj_step(m, d)

  substeps = round(CTRL_DT / m.opt.timestep)
  detect_every = round(DETECT_PERIOD / CTRL_DT)
  det = yolo_detection(sim)
  det_age = 0
  last_action = np.zeros(4, dtype=np.float32)
  success = False
  step = 0
  while d.time < TRIAL_TIMEOUT and not success:
    # -- observation from believed state + real detector, exactly as trained
    px, py, ptheta = sim.reckoner.x, sim.reckoner.y, sim.reckoner.theta
    dxy = np.array([bx - px, by - py])
    fwd = np.array([math.cos(ptheta), math.sin(ptheta)])
    left = np.array([-math.sin(ptheta), math.cos(ptheta)])
    v_body = (d.qvel[lvad] + d.qvel[rvad]) / 2 * 0.045
    w_body = (d.qvel[rvad] - d.qvel[lvad]) * 0.045 / 0.21
    obs = DockEnv.compose_obs(
      det, det_age,
      float(dxy @ fwd), float(dxy @ left),
      wrap_angle(trial["believed_heading"] - ptheta),
      bz - (PLUG_AXIS_Z0 + float(d.qpos[lift_qadr])),
      float(d.qpos[lift_qadr]), float(d.qpos[arm_qadr]),
      feelers_touching(m, d), float(v_body), float(w_body), last_action)

    action, _ = policy.predict(obs, deterministic=True)
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    v, w, lift_cmd, arm_cmd = DockEnv.action_to_commands(a, lift_cmd, arm_cmd)
    d.ctrl[lift_act] = lift_cmd
    d.ctrl[arm_act] = arm_cmd
    tl, tr = wheel_targets(v, w)
    for i in range(substeps):
      d.ctrl[sim.left_act] = slew(d.ctrl[sim.left_act], tl, m.opt.timestep)
      d.ctrl[sim.right_act] = slew(d.ctrl[sim.right_act], tr, m.opt.timestep)
      mujoco.mj_step(m, d)
      perceive_lite(sim)
      if i % 5 == 4 and charging_contact(m, d):
        success = True
        break
    step += 1
    det_age += 1
    if step % detect_every == 0:
      det = yolo_detection(sim)
      det_age = 0
    last_action = a
  return {"success": success, "t": d.time}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trials", type=int, default=24)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--controller", choices=("both", "rl", "scripted"), default="both")
  parser.add_argument("--policy", default=None,
                      help="SAC checkpoint; default: newest runs/docking/*/best.zip")
  parser.add_argument("--weights", default=None, help="YOLO weights")
  args = parser.parse_args()

  weights = args.weights or latest_weights()
  assert weights, "no trained YOLO weights found under runs/"
  lc = load_lifecycle()

  policy = None
  if args.controller in ("both", "rl"):
    from stable_baselines3 import SAC
    path = args.policy or max(
      glob.glob("runs/docking/*/best.zip"), key=os.path.getmtime, default=None)
    assert path, "no docking policy found; train one with scripts/train_docking.py"
    policy = SAC.load(path, device="cpu")
    print(f"policy: {path}")

  probe = mujoco.MjModel.from_xml_path("models/room_1.xml")
  rng = np.random.default_rng(args.seed)
  trials = [make_trial(rng, probe) for _ in range(args.trials)]

  results = {"scripted": [], "rl": []}
  for i, trial in enumerate(trials):
    line = f"trial {i:2d} {trial['outlet']} z={trial['oz']:.2f}"
    if args.controller in ("both", "scripted"):
      r = run_scripted(lc, weights, trial)
      results["scripted"].append(r["success"])
      line += f"  scripted={'DOCKED' if r['success'] else 'fail':7s} t={r['t']:5.1f}s"
    if args.controller in ("both", "rl"):
      r = run_rl(lc, weights, trial, policy)
      results["rl"].append(r["success"])
      line += f"  rl={'DOCKED' if r['success'] else 'fail':7s} t={r['t']:5.1f}s"
    print(line, flush=True)

  print()
  for name, wins in results.items():
    if wins:
      k, n = sum(wins), len(wins)
      # Wilson 95% interval: honest about small-N uncertainty
      p = k / n
      z = 1.96
      mid = (p + z * z / (2 * n)) / (1 + z * z / n)
      half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
      print(f"{name:9s}: {k}/{n} = {p:.1%}   (95% CI {mid - half:.1%} – {mid + half:.1%})")


if __name__ == "__main__":
  main()
