"""Drive PluggyBot around room_1 with the arrow keys.

Usage:
  uv run python scripts/teleop.py            # viewer only
  uv run python scripts/teleop.py --views    # also save views.png (issue #1)
"""

import argparse
import time

import mujoco
import mujoco.viewer

from pluggybot.viz import ViewDashboard

WHEEL_RADIUS = 0.045
TRACK_WIDTH = 0.21
SPEED_STEP = 0.3      # m/s per keypress
TURN_STEP = 1.0       # rad/s per keypress
VIEWS_PERIOD = 0.5    # sim seconds between views.png saves (matches map.png)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--views", action="store_true",
                    help="save views.png: stereo pair + dock camera "
                         "(no map here -- teleop.py doesn't build one)")
args = parser.parse_args()

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
dashboard = ViewDashboard(model) if args.views else None
last_views = 0.0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
  while viewer.is_running():
    start = time.time()

    # Set the angular velocity of each wheel
    data.ctrl[left] = (command["vel"] - command["ang_vel"] * TRACK_WIDTH / 2) / WHEEL_RADIUS
    data.ctrl[right] = (command["vel"] + command["ang_vel"] * TRACK_WIDTH / 2) / WHEEL_RADIUS

    mujoco.mj_step(model, data)
    viewer.sync()

    if dashboard is not None and data.time - last_views >= VIEWS_PERIOD:
      last_views = data.time
      dashboard.save(data)

    leftover = model.opt.timestep - (time.time() - start)
    if leftover > 0:
      time.sleep(leftover)
