"""Scanner: Detects depth of an object in front of a camera"""

import numpy as np
import mujoco

class Scanner:
  def __init__(self, model, width=320, height=180, camera_name="left_eye", max_range=5.0):
    self.renderer = mujoco.Renderer(model, height, width)
    self.renderer.enable_depth_rendering()
    self.camera_name = camera_name
    self.max_range = max_range
  def scan(self, data) -> tuple[np.ndarray, np.ndarray]:  # (angles, ranges)
    self.renderer.update_scene(data, camera=self.camera_name)
    depth_image = self.renderer.render()
    return

