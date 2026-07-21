import math
import mujoco
import pytest 

from pluggybot.odometry.dead_reckoning import dead_reckoner

WHEEL_RADIUS = 0.045
TRACK_WIDTH = 0.21

# Get the yaw of the model (angle rotated around z axis)
def yaw_of(data):
  w, x, y, z = data.qpos[3:7]
  return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

# Get the axle position, 8cm behind the body (qpos)
def axle_pos(data):
  psi = yaw_of(data)
  return (data.qpos[0] - 0.08*math.cos(psi), data.qpos[1] - 0.08*math.sin(psi))


def run_sim(reckoner, model, data, seconds=2.0):
  left_adr = model.joint("left_wheel_joint").qposadr[0]
  right_adr = model.joint("right_wheel_joint").qposadr[0]
  net_theta = 0.0
  traveled_theta = 0.0
  prev_theta = yaw_of(data)
  cur_theta = 0.0
  for _ in range(int(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
    reckoner.update(data.qpos[left_adr], data.qpos[right_adr])

    # Accumulate theta
    cur_theta = yaw_of(data)
    d_theta = math.atan2(math.sin(cur_theta - prev_theta), math.cos(cur_theta - prev_theta))
    net_theta += d_theta
    traveled_theta += abs(d_theta)
    prev_theta = cur_theta
  return net_theta, traveled_theta


def test_dead_reckoning_straight(world_model, world_data):
  reckoner = dead_reckoner(WHEEL_RADIUS, TRACK_WIDTH)

  # Settle the world before anything
  run_sim(reckoner, world_model, world_data)

  # Drive straight
  world_data.ctrl[:] = 10.0  # Run at less than full power to reduce slip inaccuracies
  run_sim(reckoner, world_model, world_data, seconds=5.0)
  assert world_data.qpos[0] > 1.0           # actually went somewhere

  x_error = abs(reckoner.x - world_data.qpos[0])
  y_error = abs(reckoner.y - world_data.qpos[1])
  total_distance = math.sqrt(pow(world_data.qpos[0], 2) + pow(world_data.qpos[1], 2))
  assert (x_error/total_distance < 0.05)
  assert (y_error/total_distance < 0.05)

def test_dead_reckoning_turn_in_place(world_model, world_data):
  reckoner = dead_reckoner(WHEEL_RADIUS, TRACK_WIDTH)

  # Track the true theta here, to accumulate it (instead of bounding between -pi and pi)
  true_theta = 0.0

  # Settle before turn in place test
  added_net_theta, _ = run_sim(reckoner, world_model, world_data)
  true_theta += added_net_theta

  # Turn in place
  world_data.ctrl[:] = [-10.0, 10.0]
  added_net_theta, _ = run_sim(reckoner, world_model, world_data, seconds=3.0)
  true_theta += added_net_theta

  # Did we track yaw correctly?
  theta_error = abs(reckoner.theta - true_theta) / abs(true_theta)
  assert(theta_error < 0.02)

  # Did we track distance correctly? (should barely move)
  total_distance = math.sqrt(pow(world_data.qpos[0], 2) + pow(world_data.qpos[1], 2))
  assert(total_distance < 0.2)


def test_dead_reckoning_arc(world_model, world_data):
  reckoner = dead_reckoner(WHEEL_RADIUS, TRACK_WIDTH)

  # Track the true theta here, to accumulate it (instead of bounding between -pi and pi)
  true_theta = 0.0

  # Get the initial relationship between the body + axel
  ax0, ay0 = axle_pos(world_data)

  # Settle before turn in place test
  added_net_theta, _ = run_sim(reckoner, world_model, world_data)
  true_theta += added_net_theta

  # Move while turning
  world_data.ctrl[:] = [6.0, 10.0]
  added_net_theta, _ = run_sim(reckoner, world_model, world_data, seconds=3.0)
  true_theta += added_net_theta

  ax1, ay1 = axle_pos(world_data)

  # Did we track yaw correctly?
  theta_error = abs(reckoner.theta - true_theta) / abs(true_theta)
  assert(theta_error < 0.02)

  # Did we track distance correctly? 
  distance_measurement_error = math.hypot(reckoner.x - (ax1-ax0), reckoner.y - (ay1-ay0))
  total_distance = math.hypot(ax1-ax0, ay1-ay0)
  assert(distance_measurement_error / total_distance < 0.05)

def test_dead_reckoning_s_curve(world_model, world_data):
  reckoner = dead_reckoner(WHEEL_RADIUS, TRACK_WIDTH)

  # Track the true theta here, to accumulate it (instead of bounding between -pi and pi)
  true_theta = 0.0
  traveled_theta = 0.0

  # Get the initial relationship between the body + axel
  ax0, ay0 = axle_pos(world_data)

  # Settle before turn in place test
  added_net_theta, added_traveled_theta = run_sim(reckoner, world_model, world_data)
  true_theta += added_net_theta
  traveled_theta += added_traveled_theta

  # Move while turning, one way then the other
  world_data.ctrl[:] = [6.0, 10.0]
  added_net_theta, added_traveled_theta = run_sim(reckoner, world_model, world_data, seconds=3.0)
  true_theta += added_net_theta
  traveled_theta += added_traveled_theta

  world_data.ctrl[:] = [10.0, 6.0]
  added_net_theta, added_traveled_theta = run_sim(reckoner, world_model, world_data, seconds=3.0)
  true_theta += added_net_theta
  traveled_theta += added_traveled_theta

  ax1, ay1 = axle_pos(world_data)

  # Did we track yaw correctly?
  theta_error = abs(reckoner.theta - true_theta) / abs(traveled_theta)
  assert(theta_error < 0.02)

  # Did we track distance correctly? 
  distance_measurement_error = math.hypot(reckoner.x - (ax1-ax0), reckoner.y - (ay1-ay0))
  total_distance = math.hypot(ax1-ax0, ay1-ay0)
  assert(distance_measurement_error / total_distance < 0.05)
