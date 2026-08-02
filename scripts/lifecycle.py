"""Full lifecycle: PluggyBot explores, remembers outlets, then goes to charge.

A mission-level state machine on top of the milestone-4 navigation stack:

  EXPLORE ---- battery low, or map finished ----> GO_CHARGE ----> FACE_OUTLET
     |  frontier-drive the map while the           |  A* to the     |  pivot to
     |  outlet detector logs landmarks             |  standoff pose |  the outlet

FACE_OUTLET ends parked ~0.6 m out, squared up to the socket: exactly the
start state milestone 6's docking controller takes over from.

Usage:
  MUJOCO_GL=egl uv run python scripts/lifecycle.py --headless
  uv run python scripts/lifecycle.py                       # watch live
  uv run python scripts/lifecycle.py --headless --explore-budget 40
"""

import argparse
import math
import time
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from pluggybot.behavior.navigation import (
  BACKOFF_TIME,
  FRONT_STOP_RANGE,
  MAP_SAVE_PERIOD,
  REPLAN_PERIOD,
  SCAN_EVERY,
  STRIKES_TO_FINISH,
  W_SPIN,
  WAYPOINT_RADIUS,
  drive_toward,
  path_to_waypoints,
  plan,
  render_map,
)
from pluggybot.control import wheel_targets, slew
from pluggybot.mapping.landmarks import LandmarkStore
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.odometry.dead_reckoning import DeadReckoner
from pluggybot.perception.outlet_spotter import OutletSpotter, latest_weights
from pluggybot.perception.scanner import Scanner

State = Literal["EXPLORE", "GO_CHARGE", "FACE_OUTLET", "DONE"]

SPOT_PERIOD = 0.5        # sim seconds between YOLO looks (outlets don't move)
EXPLORE_BUDGET = 60.0    # sim seconds before the battery stand-in calls for a charge
BACKOFF_SPEED = -0.15    # m/s straight reverse when the safety reflex trips


class Lifecycle:
  """Mission state machine over the navigation stack.

  Two levels of state, deliberately kept apart:

    self.state -- the MISSION phase (EXPLORE / GO_CHARGE / FACE_OUTLET / DONE).
                  One method per phase. A phase method returns the (v, w) drive
                  command for this step, and hands off by assigning self.state.
    self.mode  -- the MANEUVER within a phase ("spin" / "drive"). "Turn in
                  place until done" is not a mission phase, it's how a phase
                  gets its job done.

  Why a class rather than nested functions: the phases share a dozen mutable
  values (waypoints, timers, strike counts, the map). A nested function can
  READ an enclosing local but cannot REBIND one without declaring `nonlocal`
  for every single name -- and `mode = "spin"` inside a nested function
  silently creates a new local instead, which is the bug that bites everyone
  once. Passing them all as arguments and returning them as tuples works but
  reads terribly. Instance attributes are the same sharing, spelled clearly.
  """

  def __init__(self, headless: bool, max_sim_time: float,
               weights: str | None, explore_budget: float) -> None:
    self.model = mujoco.MjModel.from_xml_path("models/room_1.xml")
    self.data = mujoco.MjData(self.model)
    self.headless = headless
    self.max_sim_time = max_sim_time
    self.explore_budget = explore_budget

    # -- subsystems
    self.reckoner = DeadReckoner(wheel_radius=0.045, track_width=0.21)
    self.scanner = Scanner(self.model)
    self.grid = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
    self.landmarks = LandmarkStore()
    self.spotter = OutletSpotter(self.model, weights) if weights else None

    # -- model handles, looked up once
    m = self.model
    self.left_act = m.actuator("left_motor").id
    self.right_act = m.actuator("right_motor").id
    self.left_adr = m.joint("left_wheel_joint").qposadr[0]
    self.right_adr = m.joint("right_wheel_joint").qposadr[0]
    self.gyro = m.sensor("imu_gyro")
    self.chassis_gid = m.geom("chassis").id

    # -- mission state
    self.state: State = "EXPLORE"
    self.target = None                 # the Landmark being driven to
    self.explore_done_reason = "running"

    # -- shared navigation state
    self.waypoints: list[tuple[float, float]] = []
    self.blacklist: set[tuple[int, int]] = set()
    self.backoff_until = 0.0

    # -- EXPLORE-only state
    self.mode = "spin"                 # open with a 360 look-around to seed the map
    self.spin_remaining = 2 * math.pi
    self.just_spun = False
    self.next_replan = 0.0
    self.strikes = 0

    # -- bookkeeping
    self.next_spot = 0.0
    self.step_count = 0
    self.collision_steps = 0
    self.last_save = 0.0
    self.last_report = 0.0

  # ---- shared plumbing -----------------------------------------------------

  @property
  def pose(self) -> tuple[float, float, float]:
    """Dead-reckoned (x, y, theta) of the axle midpoint."""
    return (self.reckoner.x, self.reckoner.y, self.reckoner.theta)

  def perceive(self) -> None:
    """Senses, on their own cadences -- shared by every phase, because what
    the robot can see doesn't depend on what it's trying to do."""
    d, m = self.data, self.model
    self.reckoner.update(
      d.qpos[self.left_adr], d.qpos[self.right_adr],
      gyro_yaw_rate=d.sensordata[self.gyro.adr[0] + 2], dt=m.opt.timestep,
    )

    if self.step_count % SCAN_EVERY == 0:
      angles, ranges = self.scanner.scan(d)
      self.grid.update(self.pose, angles, ranges, self.scanner.max_range)
      # Safety reflex: anything dead ahead that planning didn't expect.
      # Fires in EVERY phase and maneuver, including spins -- explore.py only
      # armed it while following waypoints, and measurement showed its
      # look-around spins passing within 0.257 m of the L-box, 7 mm outside
      # this threshold. It never collided by luck, not by design.
      # Not re-armed mid-backoff, so a backoff is a bounded 0.8 s pulse
      # rather than an open-ended reverse into whatever is behind.
      if d.time >= self.backoff_until:
        front = ranges[np.abs(angles) < 0.35]
        if front.min() < FRONT_STOP_RANGE:
          self.backoff_until = d.time + BACKOFF_TIME
          self.waypoints = []          # forces the phase to replan afterwards

    if self.spotter is not None and d.time >= self.next_spot:
      self.next_spot = d.time + SPOT_PERIOD
      px, py, _ = self.pose
      for x, y, z, _conf in self.spotter.spot(d, self.pose):
        self.landmarks.add_sighting(x, y, z, seen_from=(px, py))

  def actuate(self, v: float, w: float) -> None:
    """(v, w) body command -> slew-limited wheel velocity targets."""
    tl, tr = wheel_targets(v, w)
    ts = self.model.opt.timestep
    self.data.ctrl[self.left_act] = slew(self.data.ctrl[self.left_act], tl, ts)
    self.data.ctrl[self.right_act] = slew(self.data.ctrl[self.right_act], tr, ts)

  def follow_waypoints(self) -> tuple[float, float]:
    """Drive at the next waypoint, dropping any already reached. Returns
    (0, 0) once the list empties -- refilling it is the phase's job."""
    while self.waypoints:
      wx, wy = self.waypoints[0]
      if math.hypot(wx - self.pose[0], wy - self.pose[1]) < WAYPOINT_RADIUS:
        self.waypoints.pop(0)
        continue
      return drive_toward(self.pose, (wx, wy))
    return 0.0, 0.0

  def start_spin(self) -> None:
    self.mode = "spin"
    self.spin_remaining = 2 * math.pi

  # ---- mission phases ------------------------------------------------------
  # Each returns the (v, w) command for this step and may reassign self.state.

  def explore(self) -> tuple[float, float]:
    """Frontier-drive the map while the detector logs outlet landmarks."""
    d = self.data
    if d.time >= self.explore_budget:
      return self.leave_explore("battery-low")

    # -- decide: replan when the plan is stale or spent (a spin runs to completion)
    if self.mode == "drive" and (d.time >= self.next_replan or not self.waypoints):
      self.next_replan = d.time + REPLAN_PERIOD
      path, status = plan(self.grid, self.pose, self.blacklist)
      if status == "ok":
        self.waypoints = path_to_waypoints(self.grid, path)
        self.strikes = 0
        self.just_spun = False
      else:
        self.waypoints = []
        if not self.just_spun:
          self.start_spin()            # look around before concluding anything
          self.just_spun = True
        else:
          # spinning here didn't help: either truly done, or count a strike
          self.strikes += 1
          if status == "no-frontiers" or self.strikes >= STRIKES_TO_FINISH:
            return self.leave_explore(status)

    # -- act
    if self.mode == "spin":
      self.spin_remaining -= W_SPIN * self.model.opt.timestep
      if self.spin_remaining > 0:
        return 0.0, W_SPIN
      self.mode = "drive"
      self.waypoints = []
      self.next_replan = d.time        # replan immediately on the fresh map
      return 0.0, 0.0
    return self.follow_waypoints()

  def leave_explore(self, reason: str) -> tuple[float, float]:
    """Exit EXPLORE: head for the nearest known outlet, or stop if none."""
    self.explore_done_reason = reason
    self.target = self.landmarks.nearest_confirmed(self.pose[0], self.pose[1])
    self.state = "GO_CHARGE" if self.target is not None else "DONE"
    self.waypoints = []
    print(f"t={self.data.time:6.1f}s  EXPLORE -> {self.state}  ({reason}; "
          f"{len(self.landmarks.confirmed())} outlet(s) remembered)")
    return 0.0, 0.0

  def go_charge(self) -> tuple[float, float]:
    """TODO(Ben): drive to self.target.standoff().

    Sketch -- it is mostly the same shape as explore()'s decide/act split:
      * goal = self.target.standoff()  -> (x, y, heading); plan to the (x, y).
      * Replan on the REPLAN_PERIOD cadence like explore does, because the
        map (and the reflex) can invalidate the path underneath you.
      * To plan: traversable_mask(self.grid.grid) -> astar(trav, robot_cell,
        goal_cell) -> path_to_waypoints(self.grid, path). Both imports come
        from pluggybot.mapping; grid.world_to_cell does the conversion.
      * Then `return self.follow_waypoints()`.
      * Arrival: waypoints empty AND within ~WAYPOINT_RADIUS of the standoff
        point -> self.state = "FACE_OUTLET".
      * If astar returns None, just wait for the next replan tick -- the map
        keeps improving. Give up after a few strikes so it can't hang.

    Watch out: the standoff point may land inside the wall's inflation ring
    if the landmark estimate is a few cm off. If A* keeps failing, retry at a
    slightly larger standoff distance before declaring it unreachable.
    """
    self.state = "DONE"
    return 0.0, 0.0

  def face_outlet(self) -> tuple[float, float]:
    """TODO(Ben): pivot in place until squared up with the outlet, then stop.

    Sketch:
      * _, _, want = self.target.standoff()
      * err = wrap_angle(want - self.pose[2])   (from pluggybot.control)
      * while abs(err) > ~2 degrees: return (0.0, clamped K_HEADING * err)
      * then self.state = "DONE" and return (0.0, 0.0).

    The spike measured docking tolerance at +/-3 degrees of yaw, so this is
    the step whose accuracy actually matters -- worth printing the final
    heading error to see what the odometry gives you.
    """
    self.state = "DONE"
    return 0.0, 0.0

  # ---- the loop ------------------------------------------------------------

  def run(self) -> None:
    phases = {"EXPLORE": self.explore, "GO_CHARGE": self.go_charge,
              "FACE_OUTLET": self.face_outlet}
    viewer = None if self.headless else mujoco.viewer.launch_passive(self.model, self.data)
    try:
      while ((viewer is None or viewer.is_running())
             and self.data.time < self.max_sim_time
             and self.state != "DONE"):
        wall_start = time.time()

        self.perceive()
        if self.data.time < self.backoff_until:
          v, w = BACKOFF_SPEED, 0.0    # reflex pre-empts whatever the mission wanted
        else:
          v, w = phases[self.state]()
        self.actuate(v, w)

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        for i in range(self.data.ncon):
          c = self.data.contact[i]
          if self.chassis_gid in (c.geom1, c.geom2):
            self.collision_steps += 1
            break

        if viewer is not None:
          viewer.sync()
          leftover = self.model.opt.timestep - (time.time() - wall_start)
          if leftover > 0:
            time.sleep(leftover)
        self.report()
    finally:
      if viewer is not None:
        viewer.close()
    self.summarize()

  def report(self) -> None:
    """Periodic map dump and telemetry line."""
    t = self.data.time
    if t - self.last_save >= MAP_SAVE_PERIOD:
      self.last_save = t
      Image.fromarray(render_map(self.grid, self.pose, self.waypoints)).save("map.png")
    if self.headless and t - self.last_report >= 30.0:
      self.last_report = t
      known = int(np.count_nonzero(np.abs(self.grid.grid) > 0.5))
      print(f"t={t:6.1f}s  {self.state:<11s} mode={self.mode:<5s} "
            f"known cells={known}  outlets={len(self.landmarks.confirmed())}")

  def summarize(self) -> None:
    Image.fromarray(render_map(self.grid, self.pose, [])).save("map.png")
    known = int(np.count_nonzero(np.abs(self.grid.grid) > 0.5))
    print(f"\nended after {self.data.time:.1f} sim-seconds in state {self.state}"
          f" (explore: {self.explore_done_reason})")
    print(f"known cells: {known}   frontiers blacklisted: {len(self.blacklist)}")
    print(f"chassis-contact steps (should be 0): {self.collision_steps}")
    for lm in self.landmarks.confirmed():
      sx, sy, sh = lm.standoff()
      print(f"  outlet ({lm.x:+.2f}, {lm.y:+.2f}, z={lm.z:.2f}) seen x{lm.n_sightings}"
            f"  standoff ({sx:+.2f}, {sy:+.2f}) hdg {math.degrees(sh):+.0f} deg")
    print("final map saved to map.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--headless", action="store_true", help="no viewer, run at full speed")
  parser.add_argument("--max-sim-time", type=float, default=600.0, help="sim-seconds budget")
  parser.add_argument("--explore-budget", type=float, default=EXPLORE_BUDGET,
                      help="sim-seconds of exploring before the battery stand-in trips")
  parser.add_argument("--weights", default=None, help="YOLO weights; default: newest under runs/")
  args = parser.parse_args()
  Lifecycle(
    headless=args.headless, max_sim_time=args.max_sim_time,
    weights=args.weights or latest_weights(), explore_budget=args.explore_budget,
  ).run()
