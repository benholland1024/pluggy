import glob
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.perception.outlet_spotter import (
  CAM_H,
  CAM_HEIGHT,
  CAM_W,
  OutletSpotter,
  pixel_to_world,
)

ROOT = Path(__file__).parent.parent
ROOM_1_PATH = ROOT / "models" / "room_1.xml"

# Pixel indices of the exact optical axis (u_c = v_c = 0 after the +0.5 shift).
U_CENTER, V_CENTER = CAM_W / 2 - 0.5, CAM_H / 2 - 0.5


@pytest.fixture(scope="module")
def room_model():
  return mujoco.MjModel.from_xml_path(str(ROOM_1_PATH))


def teleport(model, data, x, y, yaw):
  """Place the robot at a pose and settle the kinematics (no dynamics)."""
  mujoco.mj_resetData(model, data)
  data.qpos[0:3] = [x, y, 0.045]
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(model, data)


def axle_pose(x, y, yaw):
  """qpos tracks the body origin 8 cm ahead of the axle midpoint the
  dead reckoner (and therefore pixel_to_world) speaks. Transform."""
  return (x - 0.08 * math.cos(yaw), y - 0.08 * math.sin(yaw), yaw)


# -- pure projection math ----------------------------------------------------

def test_center_pixel_projects_straight_ahead():
  x, y, z = pixel_to_world(U_CENTER, V_CENTER, 2.0, (0.0, 0.0, 0.0))
  assert math.isclose(x, 2.0 + 0.03, abs_tol=1e-9)   # depth + camera forward
  assert math.isclose(y, 0.03, abs_tol=1e-9)          # camera's left offset
  assert math.isclose(z, CAM_HEIGHT, abs_tol=1e-9)    # on the optical axis


def test_pixel_left_of_center_lands_left():
  f = (CAM_H / 2) / math.tan(math.radians(41.0) / 2)
  # A pixel one focal length left of center sits 45 degrees off-axis:
  # sideways offset must equal the depth exactly.
  x, y, _ = pixel_to_world(U_CENTER - f, V_CENTER, 2.0, (0.0, 0.0, 0.0))
  assert math.isclose(y, 0.03 + 2.0, abs_tol=1e-6)
  assert math.isclose(x, 0.03 + 2.0, abs_tol=1e-6)


def test_pixel_above_center_lands_higher():
  _, _, z = pixel_to_world(U_CENTER, V_CENTER - 50, 2.0, (0.0, 0.0, 0.0))
  assert z > CAM_HEIGHT


def test_heading_rotates_the_projection():
  # Facing +y, straight-ahead depth should extend along +y, and the camera's
  # left offset should point along -x.
  x, y, _ = pixel_to_world(U_CENTER, V_CENTER, 2.0, (0.0, 0.0, math.pi / 2))
  assert math.isclose(y, 2.0 + 0.03, abs_tol=1e-9)
  assert math.isclose(x, -0.03, abs_tol=1e-9)


# -- projection vs MuJoCo ground truth (no YOLO: segmentation finds the pixel)

def _outlet_pixel_and_depth(model, data, outlet_body):
  """Center pixel of an outlet's segmentation mask, plus depth there.
  Median-of-mask is robust to MSAA stray pixels at the render edges."""
  r = mujoco.Renderer(model, CAM_H, CAM_W)
  try:
    r.enable_segmentation_rendering()
    r.update_scene(data, camera="left_eye")
    seg = r.render()
    r.disable_segmentation_rendering()
    r.enable_depth_rendering()
    r.update_scene(data, camera="left_eye")
    depth = r.render()
  finally:
    r.close()
  ids = [g for g in range(model.ngeom)
         if (model.geom(g).name or "").startswith(outlet_body + "_")]
  mask = np.isin(seg[:, :, 0], ids) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
  assert mask.sum() > 20, f"{outlet_body} not visible from this pose"
  ys, xs = np.nonzero(mask)
  u, v = float(np.median(xs)), float(np.median(ys))
  d = float(np.median(depth[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2]))
  return u, v, d


@pytest.mark.parametrize("x, y, yaw, outlet", [
  (-1.05, 1.0, math.pi, "outlet_a"),            # straight-on
  (0.5, -0.9, -math.pi / 2, "outlet_b"),        # straight-on, other wall
  (-0.7, 0.2, math.radians(130), "outlet_a"),   # oblique, outlet off-center
])
def test_projection_hits_ground_truth(room_model, x, y, yaw, outlet):
  data = mujoco.MjData(room_model)
  teleport(room_model, data, x, y, yaw)
  u, v, d = _outlet_pixel_and_depth(room_model, data, outlet)
  wx, wy, wz = pixel_to_world(u, v, d, axle_pose(x, y, yaw))
  truth = room_model.body(outlet).pos
  err = math.sqrt((wx - truth[0]) ** 2 + (wy - truth[1]) ** 2 + (wz - truth[2]) ** 2)
  assert err < 0.10, f"projected {outlet} {err:.3f} m from truth"


# -- end to end with the trained detector (skips when no weights exist) ------

WEIGHTS = sorted(glob.glob(str(ROOT / "runs/detect/*/weights/best.pt")),
                 key=os.path.getmtime)


@pytest.mark.skipif(not WEIGHTS, reason="no trained YOLO weights under runs/")
def test_spotter_end_to_end(room_model):
  data = mujoco.MjData(room_model)
  x, y, yaw = -1.05, 1.0, math.pi
  teleport(room_model, data, x, y, yaw)
  spotter = OutletSpotter(room_model, WEIGHTS[-1])
  sightings = spotter.spot(data, axle_pose(x, y, yaw))
  truth = room_model.body("outlet_a").pos
  assert any(
    math.hypot(sx - truth[0], sy - truth[1]) < 0.2 and abs(sz - truth[2]) < 0.15
    for sx, sy, sz, _ in sightings
  ), f"no sighting near outlet_a; got {sightings}"
