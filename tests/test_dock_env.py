"""Guards for the RL docking environment (envs/dock_env.py)."""

import math

import mujoco
import numpy as np
import pytest

import pluggybot.envs.dock_env as dock_env_mod
from pluggybot.docking.contact import feelers_touching
from pluggybot.envs.dock_env import (
  CTRL_DT, FOCAL, LIFT_RATE, OBS_DIM, PLUG_LATERAL, SOCKET_X, V_MAX, W_MAX,
  DockEnv,
)


@pytest.fixture(scope="module")
def env():
  e = DockEnv()
  yield e
  e.close()


class CheatPilot:
  """Ground-truth pilot (never available to training): drive to a tight
  standoff, square up, creep to the feelers, insert with the lift servoed on
  TRUE plug-face height. If THIS cannot dock, the env's mechanics broke --
  and no reward function can teach what the physics forbids."""

  def __init__(self, env):
    self.env = env
    self.phase = "approach"

  def act(self):
    from pluggybot.behavior.navigation import drive_toward
    from pluggybot.control import wrap_angle
    env = self.env
    x, y, yaw = env._true_pose()
    sock = env._socket_pos()
    feelers = feelers_touching(env.model, env.data)
    plug_y = y - PLUG_LATERAL * math.cos(yaw)
    lat_r = -(sock[1] - plug_y)
    face_z = env.data.site_xpos[env._face_site][2]
    a_lift = np.clip(60.0 * (sock[2] - face_z), -1, 1)
    v = w = 0.0
    a_arm = -1.0
    if self.phase == "approach":
      wp = (0.30, sock[1] - PLUG_LATERAL)
      if math.hypot(wp[0] - x, wp[1] - y) < 0.03:
        self.phase = "face"
      else:
        v, w = drive_toward((x, y, yaw), wp)
        v, w = min(v, 0.08), np.clip(w, -W_MAX, W_MAX)
    if self.phase == "face":
      err = wrap_angle(math.pi - yaw)
      if abs(err) > math.radians(0.5):
        v, w = 0.0, np.clip(2.0 * err, -W_MAX, W_MAX)
      else:
        self.phase = "creep"
    if self.phase == "creep":
      if feelers == 2:
        self.phase = "insert"
      else:
        v = 0.05 if feelers == 0 else 0.013
        w = np.clip(4.0 * lat_r, -0.2, 0.2)
    if self.phase == "insert":
      v, w, a_arm = 0.0045, 0.0, 1.0
    return np.array([v / V_MAX if v >= 0 else v / 0.06,
                     w / W_MAX, a_lift, a_arm], dtype=np.float32)


def test_obs_and_reset_are_deterministic(env):
  obs1, _ = env.reset(seed=7)
  obs2, _ = env.reset(seed=7)
  assert obs1.shape == (OBS_DIM,)
  assert np.array_equal(obs1, obs2), "same seed must give the same episode"
  assert np.all(np.abs(obs1) <= 1.0)
  a = np.array([0.5, -0.2, 0.1, 0.0], dtype=np.float32)
  env.reset(seed=7)
  s1 = env.step(a)[0]
  env.reset(seed=7)
  s2 = env.step(a)[0]
  assert np.allclose(s1, s2), "stepping must be deterministic under a seed"


def test_cheat_pilot_docks(env):
  """Dockability floor: a ground-truth pilot must seat the plug and collect
  the success reward. Guards the whole env stack -- spawn geometry, socket
  mocap, actuation mapping, charging criterion."""
  for seed in (101, 105):
    obs, _ = env.reset(seed=seed)
    pilot = CheatPilot(env)
    total = 0.0
    done = trunc = False
    while not (done or trunc):
      obs, r, done, trunc, info = env.step(pilot.act())
      total += r
    assert info["success"], f"cheat pilot failed to dock (seed {seed})"
    assert total > 10.0, "success should dominate the return"


def test_synthetic_detector_reproduces_clipping_bias(env, monkeypatch):
  """At close range the socket slides out of the frame bottom and the box
  centre biases UP while the lateral centre stays honest -- the real
  detector's measured hand-eye signature (+23 mm vertical bias at 0.19 m,
  lateral honest to +1 mm). The synthetic detector must inherit this from
  clipping geometry, or training teaches a vertical servo the real robot
  cannot fly."""
  monkeypatch.setattr(dock_env_mod, "DETECT_DROPOUT", 0.0)
  monkeypatch.setattr(dock_env_mod, "PIXEL_NOISE", 0.0)
  monkeypatch.setattr(dock_env_mod, "RANGE_NOISE", 0.0)
  env.reset(seed=3)
  env.data.mocap_pos[0] = [SOCKET_X, 0.0, 0.30]
  # Robot square-on, camera ~0.2 m from the plate, plug axis at socket height.
  yaw = math.pi
  env.data.qpos[0] = 0.249 - 0.08
  env.data.qpos[1] = -PLUG_LATERAL
  env.data.qpos[2] = 0.045
  env.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  env.data.qpos[env._lift_qadr] = 0.30 - 0.145
  mujoco.mj_forward(env.model, env.data)

  u, v, bw, bh, rng, valid = env._detect()
  assert valid
  # Unclipped projection of the box centre: 0.06 m below the camera axis.
  v_unclipped = 180 + FOCAL * 0.06 / rng
  full_bh = 2 * FOCAL * dock_env_mod.PLATE_HALF * env._box_scale / rng
  assert bh < full_bh * 0.98, "box should be clipped here"
  assert v < v_unclipped - 10, "clipped box centre must bias UP (smaller v)"
  assert abs(u - 320) < 8, "lateral centre must stay honest while clipped"

  # And the arm's SELF-OCCLUSION must eat the box from below as it extends:
  # the real YOLO boxes shrank ~25 % with the arm out, which an un-occluded
  # synthetic detector never showed the policy (measured in the first
  # end-to-end eval -- the policy had learned to approach arm-first).
  env.data.qpos[env._arm_qadr] = 0.15
  mujoco.mj_forward(env.model, env.data)
  u2, v2, _, bh2, rng2, valid2 = env._detect()
  assert valid2
  assert bh2 < bh - 5, "extending the arm must shrink the box from below"
  assert v2 < v, "occlusion must bias the box centre further up"


def test_action_mapping_clamps_and_integrates():
  v, w, lift, arm = DockEnv.action_to_commands(
    np.array([1.0, -1.0, 1.0, 1.0]), 0.0, 0.0)
  assert v == pytest.approx(V_MAX)
  assert w == pytest.approx(-W_MAX)
  assert lift == pytest.approx(LIFT_RATE * CTRL_DT)
  for _ in range(1000):
    _, _, lift, arm = DockEnv.action_to_commands(
      np.array([0.0, 0.0, 1.0, 1.0]), lift, arm)
  assert lift <= 0.31 and arm <= 0.20, "rate integration must respect ranges"
