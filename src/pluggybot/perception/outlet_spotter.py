"""Outlet spotting: YOLO detection + depth -> world-frame outlet positions.

The projection chain, one hop at a time:

  YOLO box center (u, v)          which pixel matters
    + depth image at (u, v)       how far along the camera axis it is
    -> camera-frame point         pinhole model: offsets scale with depth
    -> world-frame point          rotate by heading, add camera position
    -> LandmarkStore.add_sighting rest of the pipeline (mapping/landmarks.py)

The Scanner (perception/scanner.py) is the 1-D special case of the same math:
it only ever reads the center row, so it gets bearing but no elevation. Here
the pixel can be anywhere in the frame, so the vertical axis buys the outlet's
height — which is all the 3-D the mission needs (no voxel map required).
"""

import math

import mujoco
import numpy as np

# Camera pose relative to the dead-reckoned pose (= axle midpoint, heading).
# From models/pluggybot.xml: body origin sits 8 cm ahead of the axle; the head
# is at body-local (-0.08, 0, 0.135) and left_eye at head-local (0.03, 0.03, 0)
# -> camera = axle + 0.03 forward, 0.03 left, at 0.045 + 0.135 = 0.18 m height.
# (occupancy_grid.update encodes the same 0.03/0.03 offsets for its scan origin.)
CAM_FORWARD = 0.03
CAM_LEFT = 0.03
CAM_HEIGHT = 0.18

CAM_W, CAM_H = 640, 360    # matches the detector's training image size/aspect
MAX_RANGE = 6.0            # m: deeper "detections" are floor/far-plane junk


def pixel_to_world(
  u: float, v: float, depth: float,
  pose: tuple[float, float, float],
  *, width: int = CAM_W, height: int = CAM_H, fovy_deg: float = 41.0,
) -> tuple[float, float, float]:
  """Project one pixel with known depth into world coordinates.

  u, v: pixel column/row (origin top-left). depth: MuJoCo depth value at that
  pixel — the PERPENDICULAR distance along the camera axis, not range along
  the ray (that convention is why Scanner divides by cos, and why here the
  forward component is simply `depth`). pose: dead-reckoned (x, y, theta) of
  the axle midpoint.
  """
  px, py, theta = pose

  # Pinhole camera: focal length in pixels, from the vertical FOV.
  f = (height / 2) / math.tan(math.radians(fovy_deg) / 2)

  # Pixel offsets from the optical axis (+0.5 = pixel centers, as in Scanner).
  u_c = u - width / 2 + 0.5
  v_c = v - height / 2 + 0.5

  # Camera-frame offsets, similar triangles: sideways/vertical distance is
  # (pixel offset / focal length) x depth. Signs: +u is image-right = robot
  # right (so negate for left), +v is image-down = world down (negate for up).
  forward = depth
  left = -u_c * depth / f
  up = -v_c * depth / f

  # Camera position in the world, then rotate (forward, left) by heading.
  sin_t, cos_t = math.sin(theta), math.cos(theta)
  cam_x = px + CAM_FORWARD * cos_t - CAM_LEFT * sin_t
  cam_y = py + CAM_FORWARD * sin_t + CAM_LEFT * cos_t
  wx = cam_x + forward * cos_t - left * sin_t
  wy = cam_y + forward * sin_t + left * cos_t
  wz = CAM_HEIGHT + up
  return wx, wy, wz


class OutletSpotter:
  """Runs the trained detector on the robot's camera and returns world points.

  Owns its own renderer (like Scanner) at the detector's native resolution.
  YOLO weights load once at construction — reuse the instance.
  """

  def __init__(self, model, weights: str, camera_name: str = "left_eye",
               conf: float = 0.5) -> None:
    from ultralytics import YOLO   # deferred: slow import, torn off for tests
    self.detector = YOLO(weights)
    self.renderer = mujoco.Renderer(model, CAM_H, CAM_W)
    self.camera_name = camera_name
    self.conf = conf
    self.fovy_deg = float(model.camera(camera_name).fovy[0])

  def spot(self, data, pose: tuple[float, float, float],
           ) -> list[tuple[float, float, float, float]]:
    """One look through the camera: returns [(x, y, z, confidence), ...]
    in world frame, empty when nothing is detected."""
    r = self.renderer
    r.update_scene(data, camera=self.camera_name)
    rgb = r.render()
    r.enable_depth_rendering()
    r.update_scene(data, camera=self.camera_name)
    depth = r.render()
    r.disable_depth_rendering()

    # Ultralytics' numpy path expects BGR (OpenCV convention); flip channels.
    result = self.detector.predict(
      np.ascontiguousarray(rgb[:, :, ::-1]), conf=self.conf, verbose=False,
    )[0]

    sightings = []
    for box, conf in zip(result.boxes.xywh, result.boxes.conf):
      u, v = float(box[0]), float(box[1])
      iu, iv = min(int(u), CAM_W - 1), min(int(v), CAM_H - 1)
      # Median of a 3x3 patch: robust to landing on a pin-hole or plate edge.
      d = float(np.median(depth[max(0, iv - 1):iv + 2, max(0, iu - 1):iu + 2]))
      if d > MAX_RANGE:
        continue
      x, y, z = pixel_to_world(u, v, d, pose, fovy_deg=self.fovy_deg)
      sightings.append((x, y, z, float(conf)))
    return sightings
