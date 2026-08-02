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
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.odometry.dead_reckoning import DeadReckoner
from pluggybot.perception.scanner import Scanner


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
