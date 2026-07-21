import time

import mujoco
import mujoco.viewer

WHEEL_RADIUS = 0.045
TRACK_WIDTH = 0.21
SPEED_STEP = 0.3      # m/s per keypress
TURN_STEP = 1.0       # rad/s per keypress

model = mujoco.MjModel.from_xml_path("models/playground.xml")
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
  while viewer.is_running():
    start = time.time()

    # Set the angular velocity of each wheel
    data.ctrl[left] = (command["vel"] - command["ang_vel"] * TRACK_WIDTH / 2) / WHEEL_RADIUS
    data.ctrl[right] = (command["vel"] + command["ang_vel"] * TRACK_WIDTH / 2) / WHEEL_RADIUS
    
    mujoco.mj_step(model, data)
    viewer.sync()
    leftover = model.opt.timestep - (time.time() - start)
    if leftover > 0:
      time.sleep(leftover)

