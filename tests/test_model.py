import math

import mujoco

def settle(model, data, seconds=2.0):
  for _ in range(int(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)


def pitch_deg(data):
  w, x, y, z = data.qpos[3:7]
  return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))

def test_model_compiles(model):
  assert model.nu == 2 # exactly 2 motors

def test_rests_level(model, data):
  settle(model, data)
  assert abs(pitch_deg(data)) < 1.0

def test_drives_straight(model, data):
  settle(model, data)
  data.ctrl[:] = 35.0
  settle(model, data, seconds=5.0)
  assert data.qpos[0] > 2.0           # actually went somewhere
  assert abs(data.qpos[1]) < 0.05     # without veering
  assert abs(pitch_deg(data)) < 10.0  # without wheelie-ing

def test_turns_in_place(model, data):
  settle(model, data)
  data.ctrl[:] = [-10.0, 10.0]
  settle(model, data, seconds=3.0)
  assert math.hypot(data.qpos[0], data.qpos[1]) < 0.1  # stayed put while spinning
   

