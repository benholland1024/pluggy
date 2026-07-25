"""Scanner: Detects depth of an object in front of a camera"""

import numpy as np
import mujoco
import math

class Scanner:
  def __init__(self, model, width=320, height=180, camera_name="left_eye", max_range=5.0):
    self.renderer = mujoco.Renderer(model, height, width)
    self.renderer.enable_depth_rendering()
    self.camera_name = camera_name
    self.max_range = max_range

    # Get vertical FOV (radians from top of camera to bottom)
    fovy = math.radians(model.camera(camera_name).fovy[0])
    tan_half_hfov = math.tan(fovy / 2) * width / height

    # Focal length measured in pixels (dist. from camera pinhole to screen)
    f = (width / 2) / tan_half_hfov

    # Get an array of all pixels in the center, from (-width/2) to (width/2)
    #   Add 0.5 to get the center of each pixel
    u = np.arange(width) - width / 2 + 0.5 

    self.angles = -np.arctan(u/f) # Finally, the derived *horizontal* FOV

  def scan(self, data) -> tuple[np.ndarray, np.ndarray]:  # (angles, ranges)
    self.renderer.update_scene(data, camera=self.camera_name)
    depth_image = self.renderer.render()
    row = depth_image[depth_image.shape[0] // 2]
    ranges = np.minimum(row / np.cos(self.angles), self.max_range)
    return self.angles, ranges

