"""Autonomous frontier exploration: PluggyBot maps its rooms on its own.

The loop: scan -> update the occupancy grid -> (every couple of seconds)
detect frontiers, pick the nearest reachable one, A* a path to it through
inflated free space -> follow the path with a proportional controller.
Exploration ends when no reachable frontiers remain.

Usage:
  uv run python scripts/explore.py                       # watch live in the viewer
  uv run python scripts/explore.py --headless            # full speed, no window
  uv run python scripts/explore.py --headless --max-sim-time 300
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from pluggybot.control import wheel_targets, slew, wrap_angle
from pluggybot.mapping.astar import astar
from pluggybot.mapping.frontier import find_frontiers, traversable_mask
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.odometry.dead_reckoning import DeadReckoner
from pluggybot.perception.scanner import Scanner

SCAN_EVERY = 20          # physics steps between scans (500 Hz sim -> 25 Hz scanning)
REPLAN_PERIOD = 2.0      # sim seconds between replans (the map changes under us)
MAP_SAVE_PERIOD = 0.5    # sim seconds between map.png saves
V_MAX = 0.4              # m/s cruise speed
W_MAX = 1.5              # rad/s turn-rate clamp
K_HEADING = 2.5          # P gain: heading error -> turn rate
WAYPOINT_RADIUS = 0.08   # m: close enough to advance (small: big radii cut corners
                         # through the inflation ring and clip obstacles)
FRONT_STOP_RANGE = 0.25  # m: reflex threshold — camera-measured range dead ahead
BACKOFF_TIME = 0.8       # s of straight reverse after the reflex trips
MAX_PLAN_ATTEMPTS = 20   # frontiers tried per replan before giving up this round
STRIKES_TO_FINISH = 3    # post-spin pathless replans before declaring done
MIN_FRONTIER_CELLS = 6   # ignore frontiers closer than this (0.3 m): a forward camera
                         # can't observe the cells beside its own wheels -- chasing
                         # them deadlocks; they dissolve while driving to real goals
W_SPIN = 1.0             # rad/s during a look-around spin


def plan(grid, pose, blacklist):
  """Pick the nearest reachable frontier and plan a path to it.

  Returns (path, status): path is a cell list or None; status is "ok",
  "no-frontiers" (map fully explored) or "no-reachable" (frontiers exist
  but none could be pathed to this round).
  """
  trav = traversable_mask(grid.grid)
  frontiers = find_frontiers(grid.grid, traversable=trav)
  if len(frontiers) == 0:
    return None, "no-frontiers"

  rows, cols = grid.grid.shape
  rix, riy = grid.world_to_cell(pose[0], pose[1])
  rix, riy = min(max(rix, 0), cols - 1), min(max(riy, 0), rows - 1)

  dist = np.hypot(frontiers[:, 0] - rix, frontiers[:, 1] - riy)
  eligible = dist >= MIN_FRONTIER_CELLS
  if not eligible.any():
    return None, "only-near"             # a look-around spin will resolve these

  order = np.argsort(np.where(eligible, dist, np.inf))
  attempts = 0
  for idx in order:
    if not eligible[idx]:
      break                              # only inf-distance (ineligible) cells remain
    cell = (int(frontiers[idx, 0]), int(frontiers[idx, 1]))
    if cell in blacklist:
      continue
    path = astar(trav, (rix, riy), cell)
    if path is not None:
      return path, "ok"
    blacklist.add(cell)                  # unreachable this round; skip it in future
    attempts += 1
    if attempts >= MAX_PLAN_ATTEMPTS:
      break
  return None, "no-reachable"


def path_to_waypoints(grid, path):
  """Cell path -> sparse world waypoints (every 3rd cell keeps driving smooth)."""
  cells = path[3::3]
  if not cells or cells[-1] != path[-1]:
    cells.append(path[-1])
  return [grid.cell_to_world(ix, iy) for ix, iy in cells]


def drive_toward(pose, waypoint):
  """Proportional controller: (v, w) command toward a world waypoint."""
  px, py, theta = pose
  heading_err = wrap_angle(math.atan2(waypoint[1] - py, waypoint[0] - px) - theta)
  w = max(-W_MAX, min(W_MAX, K_HEADING * heading_err))
  v = V_MAX * max(0.0, math.cos(heading_err))   # pivot first, drive when aligned
  return v, w


def render_map(grid, pose, waypoints):
  """Map image with overlays: blue planned path, red robot marker."""
  img = np.stack([grid.to_image()] * 3, axis=-1)
  for wx, wy in waypoints:
    ix, iy = grid.world_to_cell(wx, wy)
    img[iy, ix] = (60, 90, 220)
  rix, riy = grid.world_to_cell(pose[0], pose[1])
  img[max(0, riy - 1):riy + 2, max(0, rix - 1):rix + 2] = (220, 50, 50)
  return np.flipud(img)


def run(headless: bool, max_sim_time: float) -> None:
  model = mujoco.MjModel.from_xml_path("models/room_1.xml")
  data = mujoco.MjData(model)

  reckoner = DeadReckoner(wheel_radius=0.045, track_width=0.21)
  scanner = Scanner(model)
  grid = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)

  left = model.actuator("left_motor").id
  right = model.actuator("right_motor").id
  left_adr = model.joint("left_wheel_joint").qposadr[0]
  right_adr = model.joint("right_wheel_joint").qposadr[0]
  gyro = model.sensor("imu_gyro")
  chassis_gid = model.geom("chassis").id

  blacklist: set[tuple[int, int]] = set()
  waypoints: list[tuple[float, float]] = []
  mode = "spin"                          # start with a 360 look-around to seed the map
  spin_remaining = 2 * math.pi
  backoff_until = 0.0
  just_spun = False
  next_replan = 0.0
  last_save = 0.0
  last_report = 0.0
  strikes = 0
  collision_steps = 0
  step_count = 0
  done_reason = "time-limit"

  viewer = None if headless else mujoco.viewer.launch_passive(model, data)
  try:
    while (viewer is None or viewer.is_running()) and data.time < max_sim_time:
      start = time.time()
      pose = (reckoner.x, reckoner.y, reckoner.theta)

      # -- perceive
      reckoner.update(
        data.qpos[left_adr], data.qpos[right_adr],
        gyro_yaw_rate=data.sensordata[gyro.adr[0] + 2], dt=model.opt.timestep,
      )
      if step_count % SCAN_EVERY == 0:
        angles, ranges = scanner.scan(data)
        grid.update(pose, angles, ranges, scanner.max_range)
        # safety reflex: something dead ahead that planning didn't expect
        if mode == "drive" and waypoints:
          front = ranges[np.abs(angles) < 0.35]
          if front.min() < FRONT_STOP_RANGE:
            mode = "backoff"
            backoff_until = data.time + BACKOFF_TIME
            waypoints = []

      # -- decide (only while driving; a spin runs to completion)
      if mode == "drive" and (data.time >= next_replan or not waypoints):
        next_replan = data.time + REPLAN_PERIOD
        path, status = plan(grid, pose, blacklist)
        if status == "ok":
          waypoints = path_to_waypoints(grid, path)
          strikes = 0
          just_spun = False
        else:
          waypoints = []
          if just_spun:
            # spinning here didn't help: either truly done, or count a strike
            strikes += 1
            if status == "no-frontiers" or strikes >= STRIKES_TO_FINISH:
              done_reason = status
              break
          else:
            mode = "spin"                # look around before concluding anything
            spin_remaining = 2 * math.pi
            just_spun = True

      # -- act
      if mode == "backoff":
        v, w = -0.15, 0.0                # reverse straight, then replan
        if data.time >= backoff_until:
          mode = "drive"
          waypoints = []
          next_replan = data.time
      elif mode == "spin":
        v, w = 0.0, W_SPIN
        spin_remaining -= W_SPIN * model.opt.timestep
        if spin_remaining <= 0:
          mode = "drive"
          waypoints = []
          next_replan = data.time        # replan immediately on the fresh map
      elif waypoints:
        wx, wy = waypoints[0]
        if math.hypot(wx - pose[0], wy - pose[1]) < WAYPOINT_RADIUS:
          waypoints.pop(0)
        v, w = drive_toward(pose, waypoints[0]) if waypoints else (0.0, 0.0)
      else:
        v, w = 0.0, 0.0
      tl, tr = wheel_targets(v, w)
      data.ctrl[left] = slew(data.ctrl[left], tl, model.opt.timestep)
      data.ctrl[right] = slew(data.ctrl[right], tr, model.opt.timestep)

      # -- step
      mujoco.mj_step(model, data)
      step_count += 1
      for i in range(data.ncon):
        c = data.contact[i]
        if chassis_gid in (c.geom1, c.geom2):
          collision_steps += 1
          break

      if viewer is not None:
        viewer.sync()
        leftover = model.opt.timestep - (time.time() - start)
        if leftover > 0:
          time.sleep(leftover)

      if data.time - last_save >= MAP_SAVE_PERIOD:
        last_save = data.time
        Image.fromarray(render_map(grid, pose, waypoints)).save("map.png")
      if headless and data.time - last_report >= 30.0:
        last_report = data.time
        known = int(np.count_nonzero(np.abs(grid.grid) > 0.5))
        print(f"t={data.time:5.0f}s  mode={mode:5s}  known cells={known}")
  finally:
    if viewer is not None:
      viewer.close()

  Image.fromarray(render_map(grid, (reckoner.x, reckoner.y, reckoner.theta), [])).save("map.png")
  known = int(np.count_nonzero(np.abs(grid.grid) > 0.5))
  print(f"exploration ended after {data.time:.1f} sim-seconds: {done_reason}")
  print(f"known cells: {known}   frontiers blacklisted: {len(blacklist)}")
  print(f"chassis-contact steps (should be 0): {collision_steps}")
  print("final map saved to map.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--headless", action="store_true", help="no viewer, run at full speed")
  parser.add_argument("--max-sim-time", type=float, default=600.0, help="sim-seconds budget")
  args = parser.parse_args()
  run(headless=args.headless, max_sim_time=args.max_sim_time)
