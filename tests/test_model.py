import math

import mujoco

def settle(model, data, seconds=2.0):
  for _ in range(int(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)


def pitch_deg(data):
  w, x, y, z = data.qpos[3:7]
  return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))

def test_model_compiles(world_model):
  assert world_model.nu == 2 # exactly 2 motors

def test_rests_level(world_model, world_data):
  settle(world_model, world_data)
  assert abs(pitch_deg(world_data)) < 1.0

def test_drives_straight(world_model, world_data):
  settle(world_model, world_data)
  world_data.ctrl[:] = 35.0
  settle(world_model, world_data, seconds=5.0)
  assert world_data.qpos[0] > 2.0           # actually went somewhere
  assert abs(world_data.qpos[1]) < 0.05     # without veering
  assert abs(pitch_deg(world_data)) < 10.0  # without wheelie-ing

def test_turns_in_place(world_model, world_data):
  settle(world_model, world_data)
  world_data.ctrl[:] = [-10.0, 10.0]
  settle(world_model, world_data, seconds=3.0)
  assert math.hypot(world_data.qpos[0], world_data.qpos[1]) < 0.1  # stayed put while spinning
   

