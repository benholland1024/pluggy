import time
import numpy as np
from PIL import Image

import mujoco
import mujoco.viewer

from pluggybot.odometry.dead_reckoning import DeadReckoner
from pluggybot.perception.scanner import Scanner
from pluggybot.mapping.occupancy_grid import OccupancyGrid

WHEEL_RADIUS = 0.045
TRACK_WIDTH = 0.21
SPEED_STEP = 0.3      # m/s per keypress
TURN_STEP = 1.0       # rad/s per keypress
MAX_WHEEL_ACCEL = 30.0   # rad/s^2 at the wheel


model = mujoco.MjModel.from_xml_path("models/room_1.xml")
data = mujoco.MjData(model)

command = {
  "vel": 0.0,     # latched speed (up to increase, down to decrease)
  "ang_vel": 0.0  # angulary velocity of the robot (not the wheels)
}
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_SPACE = 262, 263, 264, 265, 32

def key_callback(keycode):
  if keycode == KEY_UP:
    command["vel"] += SPEED_STEP
  elif keycode == KEY_DOWN:
    command["vel"] -= SPEED_STEP
  elif keycode == KEY_LEFT:
    command["ang_vel"] += TURN_STEP
  elif keycode == KEY_RIGHT:
    command["ang_vel"] -= TURN_STEP
  elif keycode == KEY_SPACE:
    command["vel"] = command["ang_vel"] = 0.0

left = model.actuator("left_motor").id
right = model.actuator("right_motor").id

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
  step_count = 0
  last_save = 0.0

  # reckoner setup
  reckoner = DeadReckoner(WHEEL_RADIUS, TRACK_WIDTH)
  left_adr = model.joint("left_wheel_joint").qposadr[0]
  right_adr = model.joint("right_wheel_joint").qposadr[0]

  # scanner + grid setup
  scanner = Scanner(model)
  grid = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)

  while viewer.is_running():
    start = time.time()

    # Set the angular velocity of each wheel. Ramped, to avoid slipping
    target_l = (command["vel"] - command["ang_vel"] * TRACK_WIDTH / 2) / WHEEL_RADIUS
    target_r = (command["vel"] + command["ang_vel"] * TRACK_WIDTH / 2) / WHEEL_RADIUS
    max_step = MAX_WHEEL_ACCEL * model.opt.timestep
    data.ctrl[left]  += np.clip(target_l - data.ctrl[left],  -max_step, max_step)
    data.ctrl[right] += np.clip(target_r - data.ctrl[right], -max_step, max_step)

    # Update the tracked position using the reckoner
    reckoner.update(data.qpos[left_adr], data.qpos[right_adr])

    # Update the occupancy map every 10th step
    if step_count % 10 == 0:
      angles, ranges = scanner.scan(data)
      grid.update((reckoner.x, reckoner.y, reckoner.theta), angles, ranges, scanner.max_range)

    step_count += 1
    mujoco.mj_step(model, data)
    viewer.sync()
    leftover = model.opt.timestep - (time.time() - start)
    if leftover > 0:
      time.sleep(leftover)

    if data.time - last_save >= 0.5:
      last_save = data.time
      img = grid.to_image()
      cell_x, cell_y = grid.world_to_cell(reckoner.x, reckoner.y)
      img[max(0, cell_y - 1):cell_y + 2, max(0, cell_x - 1):cell_x + 2] = 64  # 3×3 dark block
      Image.fromarray(np.flipud(img)).save("map.png")


