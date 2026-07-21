import math
import mujoco

from pluggybot.odometry.dead_reckoning import dead_reckoner

WHEEL_RADIUS = 0.045
TRACK_WIDTH = 0.21

pluggy_reckoner = dead_reckoner(WHEEL_RADIUS, TRACK_WIDTH)



def run_sim(model, data, seconds=2.0):
  left_ang_vel = model.joint("left_wheel_joint").qposadr
  right_ang_vel = model.joint("right_wheel_joint").qposadr
  for _ in range(int(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
    pluggy_reckoner.update(left_ang_vel, right_ang_vel, model.opt.timestep)


def test_dead_reckoning_straight(world_model, world_data):

  # Settle the world before anything
  run_sim(world_model, world_data)
  print("Theta before moving:")
  print(pluggy_reckoner.theta)

  # Drive straight
  world_data.ctrl[:] = 21.0
  run_sim(world_model, world_data, seconds=5.0)
  assert world_data.qpos[0] > 2.0           # actually went somewhere
  print("Position at end of straight line: ")
  print(world_data.qpos)
  print(pluggy_reckoner.x)
  print(pluggy_reckoner.y)
  print(pluggy_reckoner.theta)

def test_dead_reckoning_turn_in_place(world_model, world_data):

  # Settle before turn in place test
  run_sim(world_model, world_data)

  # Turn in place
  world_data.ctrl[:] = [-10.0, 10.0]
  run_sim(world_model, world_data, seconds=3.0)
  print(world_data.qpos)
  print(pluggy_reckoner.x)
  print(pluggy_reckoner.y)
  print(pluggy_reckoner.theta)
