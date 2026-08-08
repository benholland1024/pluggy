"""Gymnasium environment for the terminal docking approach (milestone 6).

The scripted DOCK controller seats the plug deterministically from a perfect
standoff but managed 4/12 end-to-end under 2 cm of standoff jitter, with the
failures dominated by plug-height error the box-centre servo cannot measure
(SimNotes: "vision-z is the gap"). This env formulates exactly that terminal
approach as an RL task:

  start   the docking hand-off pose FACE_OUTLET delivers, with its real
          error statistics (pose jitter + a *believed* target pose that is
          wrong by the landmark error -- the two are independent draws)
  act     wheels (v, w) + rate-limited lift/arm target trims, the arm under
          the measured 2.5 N force cap
  sense   what the physical robot senses, nothing more: a dock_eye detector
          box + range, believed target pose relative to odometry, lift/arm
          encoders, feeler contacts, body velocity
  succeed the electrical charging criterion (docking/contact.py), shared
          verbatim with the scripted controller

The world is deliberately minimal -- floor, one wall, one socket -- so
physics runs fast enough to train on. The socket rides a MOCAP body: reset
repositions it (height 0.24-0.40 m, lateral) without recompiling the model.

The detector is NOT run during training. The observation's "YOLO box" is
synthesized by projecting the socket plate into dock_eye with the pinhole
model and CLIPPING to the frame -- clipping is the measured mechanism behind
the real detector's close-range vertical bias (+2 mm at 0.32 m growing to
+23 mm at 0.19 m, hand-eye calibration in lifecycle.py), so the synthetic
sensor inherits the real one's signature failure honestly. It also ticks at
the real 4 Hz, not the control rate, with the detection's age exposed in the
observation. The real-YOLO check happens at eval time (eval_docking.py runs
the trained policy against actual detections in room_1).
"""

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from pluggybot.control import slew, wheel_targets, wrap_angle
from pluggybot.docking.contact import charging_contact, feelers_touching
from pluggybot.docking.schuko import WELL_DEPTH, socket_geoms_xml
from pluggybot.odometry.dead_reckoning import DeadReckoner

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

# ---- geometry constants shared with the scripted controller ----------------
PLUG_AXIS_Z0 = 0.145      # plug-axis world height at lift qpos 0 (pluggybot.xml)
DROOP_COMP = 0.0078       # measured gravity droop of the plug axis (lifecycle)
PLUG_LATERAL = 0.05       # plug rides 5 cm right of the robot centerline
SOCKET_X = 0.037          # socket origin proud of the wall face at x=0
CAM_ABOVE_PLUG = 0.06     # dock_eye rides this far above the plug axis
PLATE_HALF = 0.047        # EFFECTIVE detector-box half-extent, not the 42 mm
                          # plate: measured against the real YOLO on an
                          # aligned robot, the box spans plate + housing at a
                          # steady 47 mm half-extent from 0.6 m to 0.22 m
BOX_SCALE_RANGE = (0.85, 1.02)  # per-episode box-size factor: the real
                          # detector's box shrinks with partial occlusion in
                          # ways worth being robust to
# Self-occlusion by the arm: the tube top rides ~0.048 m below the camera,
# and its nearest visible point sits ~0.09 m + extension ahead -- as the arm
# extends, that edge climbs the frame and eats the socket from below. The
# real failure trail: YOLO boxes shrank ~25 % with the arm out, which the
# un-occluded synthetic detector never showed the policy.
OCCLUDE_DROP = 0.048      # m below the camera axis of the occluding edge
OCCLUDE_AHEAD = 0.09      # m ahead of the camera at zero extension

# ---- camera model (dock_eye, matches lifecycle's servo constants) ----------
CAM_W, CAM_H = 640, 360
FOVY_DEG = 41.0
FOCAL = (CAM_H / 2) / math.tan(math.radians(FOVY_DEG) / 2)

# ---- control interface ------------------------------------------------------
CTRL_DT = 0.05            # s per env step (20 Hz control over 1 ms physics)
DETECT_PERIOD = 0.25      # s between synthetic detector ticks (real YOLO ~4 Hz)
V_MAX = 0.10              # m/s forward clamp (scripted approach used 0.06)
V_MIN = -0.06             # m/s reverse clamp (enough to retreat and retry)
W_MAX = 0.4               # rad/s steering clamp (same as the scripted servo)
LIFT_RATE = 0.03          # m/s max lift-target rate (igus lead-screw class)
ARM_RATE = 0.10           # m/s max arm-target rate
ARM_FORCE = 2.5           # N cap: the measured push budget (lifecycle DOCK)

# ---- episode ----------------------------------------------------------------
HORIZON = 600             # control steps (30 s sim)
SUCCESS_REWARD = 20.0
LOST_PENALTY = 2.0
STEP_COST = 0.01
SHAPING_GAIN = 10.0       # reward per meter of potential decrease
YAW_WEIGHT = 0.1          # meters-equivalent per radian of facing error

# ---- randomization: the hand-off pose FACE_OUTLET actually delivers --------
STANDOFF_RANGE = (0.50, 0.70)   # m from the wall
SPAWN_LAT = 0.03                # m: plug axis vs socket axis at spawn
SPAWN_YAW = math.radians(1.5)
BELIEF_XY = 0.02                # m: landmark position error ("2 cm jitter")
BELIEF_Z = 0.015                # m: landmark height error (the measured gap)
BELIEF_YAW = math.radians(1.0)  # wall-normal belief error
ODOM_INIT_XY = 0.005            # m: odometry error at hand-off (measured small)
ODOM_INIT_YAW = math.radians(0.3)
SOCKET_Z_RANGE = (0.24, 0.40)   # covers room_1's 0.26-0.38 with margin
SOCKET_Y_RANGE = (-0.30, 0.30)

DETECT_DROPOUT = 0.05     # chance a detector tick misses (frames do)
PIXEL_NOISE = 1.5         # px std on synthetic box coordinates
RANGE_NOISE = 0.02        # relative std on synthetic range
MIN_BOX_PX = 6.0          # clipped boxes smaller than this are not detections

OBS_DIM = 21
ACT_DIM = 4


def _dock_world_xml() -> str:
  socket = socket_geoms_xml(prefix="sock_", solref="0.005 1")
  return f"""
<mujoco model="dock_world">
  <option timestep="0.001" integrator="implicitfast"/>
  <include file="pluggybot.xml"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <light pos="1.5 -0.5 1.5" dir="-0.7 0.25 -0.7"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall" type="box" size="0.01 1.5 0.6" pos="-0.01 0 0.6"
          rgba="0.75 0.72 0.55 1"/>
    <body name="socket" mocap="true" pos="{SOCKET_X} 0 0.30">
      {socket}
    </body>
    <camera name="watch" pos="0.55 -0.55 0.5" xyaxes="0.707 0.707 0 -0.35 0.35 0.87"/>
  </worldbody>
</mujoco>"""


class DockEnv(gym.Env):
  """Terminal docking as a Gymnasium task. See module docstring."""

  metadata = {"render_modes": ["rgb_array"]}

  def __init__(self, render_mode: str | None = None) -> None:
    assets = {"pluggybot.xml": (_MODELS_DIR / "pluggybot.xml").read_bytes()}
    self.model = mujoco.MjModel.from_xml_string(_dock_world_xml(), assets)
    self.data = mujoco.MjData(self.model)
    self.render_mode = render_mode
    self._renderer = None

    m = self.model
    self._arm_act = m.actuator("arm").id
    self._lift_act = m.actuator("lift").id
    self._left_act = m.actuator("left_motor").id
    self._right_act = m.actuator("right_motor").id
    self._lift_qadr = m.joint("lift_joint").qposadr[0]
    self._arm_qadr = m.joint("arm_joint").qposadr[0]
    self._left_qadr = m.joint("left_wheel_joint").qposadr[0]
    self._right_qadr = m.joint("right_wheel_joint").qposadr[0]
    self._gyro_adr = m.sensor("imu_gyro").adr[0]
    self._left_vadr = m.joint("left_wheel_joint").dofadr[0]
    self._right_vadr = m.joint("right_wheel_joint").dofadr[0]
    self._face_site = m.site("plug_face").id
    self._cam_id = m.camera("dock_eye").id
    # The arm works under the measured push budget, as in the DOCK state.
    m.actuator_forcerange[self._arm_act] = [-ARM_FORCE, ARM_FORCE]

    self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
    self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    self._substeps = round(CTRL_DT / m.opt.timestep)
    self._detect_every = round(DETECT_PERIOD / CTRL_DT)

  # ---- kinematic bookkeeping ----------------------------------------------

  def _true_pose(self) -> tuple[float, float, float]:
    """Ground-truth (x, y, yaw) of the AXLE midpoint (qpos tracks the body
    origin 8 cm ahead -- the standard frame correction, SimNotes)."""
    x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
    w, qx, qy, qz = (float(v) for v in self.data.qpos[3:7])
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x - 0.08 * math.cos(yaw), y - 0.08 * math.sin(yaw), yaw

  def _plug_axis_z(self) -> float:
    """Where the robot BELIEVES its plug axis is: lift encoder + geometry.
    (The true plug sags below this -- that is the droop the policy must
    learn to feed forward, exactly as the scripted controller calibrates.)"""
    return PLUG_AXIS_Z0 + float(self.data.qpos[self._lift_qadr])

  def _socket_pos(self) -> np.ndarray:
    return self.data.mocap_pos[0].copy()

  def _seat_point(self) -> np.ndarray:
    """World point where plug_face sits when fully seated."""
    p = self._socket_pos()
    return np.array([p[0] - WELL_DEPTH, p[1], p[2]])

  # ---- the synthetic detector ---------------------------------------------

  def _detect(self) -> tuple[float, float, float, float, float, bool]:
    """(u, v, bw, bh, range, valid): the socket's clipped, occluded image box.

    Projects the effective detector extent through the pinhole model, then
    clips to the frame AND to the arm's self-occlusion line. Both clips bias
    a partially-visible box's centre toward the surviving sliver -- the real
    detector's measured close-range behavior (frame clipping) and its
    measured arm-out behavior (boxes shrank ~25 % from below), reproduced by
    geometry rather than imitation.
    """
    cam_p = self.data.cam_xpos[self._cam_id]
    cam_r = self.data.cam_xmat[self._cam_id].reshape(3, 3)  # columns = axes
    sp = self._socket_pos()
    half = PLATE_HALF * self._box_scale
    # Box corners in world frame (socket axes are world-aligned: facing +x).
    corners = np.array([[SOCKET_X - sp[0] + 0.002, dy, dz]
                        for dy in (-half, half)
                        for dz in (-half, half)]) + sp
    rel = (corners - cam_p) @ cam_r          # into camera frame
    depths = -rel[:, 2]                      # camera looks down its own -z
    if np.any(depths < 0.08):
      return 0.0, 0.0, 0.0, 0.0, 0.0, False
    us = CAM_W / 2 + FOCAL * rel[:, 0] / depths
    vs = CAM_H / 2 - FOCAL * rel[:, 1] / depths
    # The extended arm occludes everything below its top edge in the image.
    ext = float(self.data.qpos[self._arm_qadr])
    v_occlude = CAM_H / 2 + FOCAL * OCCLUDE_DROP / (OCCLUDE_AHEAD + ext)
    x0, x1 = max(0.0, us.min()), min(float(CAM_W), us.max())
    y0, y1 = max(0.0, vs.min()), min(float(CAM_H), min(v_occlude, vs.max()))
    if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
      return 0.0, 0.0, 0.0, 0.0, 0.0, False
    # The real detector's confidence sinks as the box loses area to clipping
    # and occlusion -- in room_1 a mostly-hidden socket goes UNDETECTED (conf
    # < 0.5) long before the box shrinks to nothing. A flat dropout taught
    # the policy that heavy occlusion was survivable; visibility-scaled
    # dropout teaches it to keep the socket in view.
    full_area = (us.max() - us.min()) * (vs.max() - vs.min())
    visible = (x1 - x0) * (y1 - y0) / max(full_area, 1e-9)
    p_miss = min(0.9, DETECT_DROPOUT + 0.7 * max(0.0, 1.0 - visible))
    if self.np_random.random() < p_miss:
      return 0.0, 0.0, 0.0, 0.0, 0.0, False
    n = self.np_random.normal
    u = (x0 + x1) / 2 + n(0, PIXEL_NOISE)
    v = (y0 + y1) / 2 + n(0, PIXEL_NOISE)
    rng = float(np.mean(depths)) * (1 + n(0, RANGE_NOISE))
    return u, v, (x1 - x0), (y1 - y0), rng, True

  # ---- observation ---------------------------------------------------------

  @staticmethod
  def compose_obs(det, age_steps, along, lat, dyaw, dz,
                  lift_q, arm_q, feelers, v_body, w_body, last_action):
    """Pack the observation vector. Static so eval_docking.py can build the
    IDENTICAL vector from real YOLO detections + odometry in room_1."""
    u, v, bw, bh, rng, valid = det
    clip = np.clip
    return clip(np.array([
      1.0 if valid else 0.0,
      (u - CAM_W / 2) / (CAM_W / 2) if valid else 0.0,
      (v - CAM_H / 2) / (CAM_H / 2) if valid else 0.0,
      bw / CAM_W if valid else 0.0,
      bh / CAM_H if valid else 0.0,
      clip(rng, 0.0, 1.2) / 1.2 if valid else 0.0,
      min(age_steps, 10) / 10.0,
      clip(along, 0.0, 0.8) / 0.8,
      clip(lat, -0.2, 0.2) / 0.2,
      clip(dyaw, -0.5, 0.5) / 0.5,
      clip(dz, -0.1, 0.1) / 0.1,
      lift_q / 0.31,
      arm_q / 0.20,
      1.0 if feelers >= 1 else 0.0,
      1.0 if feelers >= 2 else 0.0,
      clip(v_body, -0.15, 0.15) / 0.15,
      clip(w_body, -1.0, 1.0),
      *last_action,
    ], dtype=np.float32), -1.0, 1.0)

  def _observe(self) -> np.ndarray:
    # The believed pose comes from DEAD RECKONING, not truth: wheels that
    # slip while grinding the wall feed the reckoner phantom distance, and
    # the policy must live with that -- the first policy trained on
    # true-pose observations learned to press at 0.1 m/s, and in room_1 its
    # odometry collapsed within seconds of wall contact (the milestone-4
    # slip lesson, relearned in a new suit).
    x, y, yaw = self._reckoner.x, self._reckoner.y, self._reckoner.theta
    bx, by = self._believed_target[:2]
    dxy = np.array([bx - x, by - y])
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    along = float(dxy @ fwd)
    lat = float(dxy @ left)
    dyaw = wrap_angle(self._believed_heading - yaw)
    dz = self._believed_target[2] - self._plug_axis_z()

    v_body = (self.data.qvel[self._left_vadr]
              + self.data.qvel[self._right_vadr]) / 2 * 0.045
    w_body = (self.data.qvel[self._right_vadr]
              - self.data.qvel[self._left_vadr]) * 0.045 / 0.21

    return self.compose_obs(
      self._last_det, self._det_age, along, lat, dyaw, dz,
      float(self.data.qpos[self._lift_qadr]),
      float(self.data.qpos[self._arm_qadr]),
      feelers_touching(self.model, self.data),
      float(v_body), float(w_body), self._last_action)

  # ---- potential for reward shaping ---------------------------------------

  def _potential(self) -> float:
    face = self.data.site_xpos[self._face_site]
    dist = float(np.linalg.norm(face - self._seat_point()))
    _, _, yaw = self._true_pose()
    return dist + YAW_WEIGHT * abs(wrap_angle(yaw - math.pi))

  # ---- gym API -------------------------------------------------------------

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    r = self.np_random
    mujoco.mj_resetData(self.model, self.data)

    # -- the world: socket somewhere on the wall
    z_s = r.uniform(*SOCKET_Z_RANGE)
    y_s = r.uniform(*SOCKET_Y_RANGE)
    self.data.mocap_pos[0] = [SOCKET_X, y_s, z_s]

    # -- the robot: at a jittered hand-off pose, facing the wall (-x)
    d = r.uniform(*STANDOFF_RANGE)
    yaw = math.pi + r.uniform(-SPAWN_YAW, SPAWN_YAW)
    axle_x = d
    axle_y = (y_s - PLUG_LATERAL) + r.uniform(-SPAWN_LAT, SPAWN_LAT)
    self.data.qpos[0] = axle_x + 0.08 * math.cos(yaw)
    self.data.qpos[1] = axle_y + 0.08 * math.sin(yaw)
    self.data.qpos[2] = 0.045
    self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    # -- the belief: where the landmark map THINKS the socket is
    self._believed_target = np.array([
      SOCKET_X + r.uniform(-BELIEF_XY, BELIEF_XY),
      y_s + r.uniform(-BELIEF_XY, BELIEF_XY),
      z_s + r.uniform(-BELIEF_Z, BELIEF_Z)])
    self._believed_heading = math.pi + r.uniform(-BELIEF_YAW, BELIEF_YAW)
    self._box_scale = r.uniform(*BOX_SCALE_RANGE)

    # -- the lift starts where the align stage would put it: at the BELIEVED
    # height with droop compensation. Teleported + settled rather than
    # servoed up, to keep resets cheap.
    lift0 = float(np.clip(self._believed_target[2] - PLUG_AXIS_Z0 + DROOP_COMP,
                          0.0, 0.31))
    self.data.qpos[self._lift_qadr] = lift0
    self._lift_cmd = lift0
    self._arm_cmd = 0.0
    self.data.ctrl[self._lift_act] = lift0
    self.data.ctrl[self._arm_act] = 0.0
    self.data.ctrl[self._left_act] = 0.0
    self.data.ctrl[self._right_act] = 0.0
    mujoco.mj_forward(self.model, self.data)
    for _ in range(200):                      # settle suspension + RCC droop
      mujoco.mj_step(self.model, self.data)

    # Odometry starts at truth plus the (small, measured) hand-off drift.
    tx, ty, tyaw = self._true_pose()
    self._reckoner = DeadReckoner(wheel_radius=0.045, track_width=0.21)
    self._reckoner.x = tx + r.uniform(-ODOM_INIT_XY, ODOM_INIT_XY)
    self._reckoner.y = ty + r.uniform(-ODOM_INIT_XY, ODOM_INIT_XY)
    self._reckoner.theta = tyaw + r.uniform(-ODOM_INIT_YAW, ODOM_INIT_YAW)
    self._reckoner.update(float(self.data.qpos[self._left_qadr]),
                          float(self.data.qpos[self._right_qadr]))

    self._last_action = np.zeros(ACT_DIM, dtype=np.float32)
    self._last_det = (0.0, 0.0, 0.0, 0.0, 0.0, False)
    self._det_age = 10
    self._step_count = 0
    self._spawn_x = axle_x
    self._last_det = self._detect()
    self._det_age = 0
    self._phi = self._potential()
    return self._observe(), {}

  @staticmethod
  def action_to_commands(a, lift_cmd: float, arm_cmd: float,
                         ) -> tuple[float, float, float, float]:
    """Action vector -> (v, w, lift_cmd, arm_cmd). One mapping, shared with
    eval_docking.py's room_1 runner so train and deploy cannot drift."""
    v = float(a[0]) * (V_MAX if a[0] >= 0 else -V_MIN)
    w = float(a[1]) * W_MAX
    lift_cmd = float(np.clip(lift_cmd + float(a[2]) * LIFT_RATE * CTRL_DT, 0.0, 0.31))
    arm_cmd = float(np.clip(arm_cmd + float(a[3]) * ARM_RATE * CTRL_DT, 0.0, 0.20))
    return v, w, lift_cmd, arm_cmd

  def step(self, action):
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    v, w, self._lift_cmd, self._arm_cmd = self.action_to_commands(
      a, self._lift_cmd, self._arm_cmd)
    self.data.ctrl[self._lift_act] = self._lift_cmd
    self.data.ctrl[self._arm_act] = self._arm_cmd

    tl, tr = wheel_targets(v, w)
    ts = self.model.opt.timestep
    success = False
    for i in range(self._substeps):
      self.data.ctrl[self._left_act] = slew(self.data.ctrl[self._left_act], tl, ts)
      self.data.ctrl[self._right_act] = slew(self.data.ctrl[self._right_act], tr, ts)
      mujoco.mj_step(self.model, self.data)
      self._reckoner.update(
        float(self.data.qpos[self._left_qadr]),
        float(self.data.qpos[self._right_qadr]),
        gyro_yaw_rate=float(self.data.sensordata[self._gyro_adr + 2]),
        dt=ts)
      if i % 5 == 4 and charging_contact(self.model, self.data):
        success = True
        break

    self._step_count += 1
    self._det_age += 1
    if self._step_count % self._detect_every == 0:
      self._last_det = self._detect()
      self._det_age = 0
    self._last_action = a

    phi = self._potential()
    reward = SHAPING_GAIN * (self._phi - phi) - STEP_COST
    self._phi = phi

    terminated = False
    x, y, yaw = self._true_pose()
    lost = (x > self._spawn_x + 0.15
            or abs(y - (self._socket_pos()[1] - PLUG_LATERAL)) > 0.35
            or abs(wrap_angle(yaw - math.pi)) > math.radians(60))
    if success:
      reward += SUCCESS_REWARD
      terminated = True
    elif lost:
      reward -= LOST_PENALTY
      terminated = True
    truncated = self._step_count >= HORIZON and not terminated

    info = {"success": success, "distance": phi}
    return self._observe(), float(reward), terminated, truncated, info

  def render(self):
    if self._renderer is None:
      self._renderer = mujoco.Renderer(self.model, 360, 480)
    self._renderer.update_scene(self.data, camera="watch")
    return self._renderer.render()

  def close(self):
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None
