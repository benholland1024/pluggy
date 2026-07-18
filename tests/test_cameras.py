import mujoco
import numpy as np

def test_stereo_pair_sees_parallax(model, data):
  for _ in range(500):
    mujoco.mj_step(model, data)
  renderer = mujoco.Renderer(model, height=240, width=320)
  try:
    renderer.update_scene(data, camera="left_eye")
    left = renderer.render()
    renderer.update_scene(data, camera="right_eye")
    right = renderer.render()
  finally:
    renderer.close()  # frees the OpenGL context
  
  assert left.shape == (240, 320, 3)
  assert left.std() > 0                   # not a blank frame
  assert not np.array_equal(left, right)  # eyes see different things