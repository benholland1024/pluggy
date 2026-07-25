"""Testing the scanner: the robot's visual depth perception"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.perception.scanner import Scanner

ROOM_PATH = Path(__file__).parent.parent / "models" / "room_1.xml"

WALL_FACE_X = 1.99          # divider-bottom at x=2.0, minus 0.01 half-thickness
ON_WALL = math.radians(25)  # beyond ~26.7 deg right, rays hit the black corner-box

@pytest.fixture(scope="module")
def room_model():
  return mujoco.MjModel.from_xml_path(str(ROOM_PATH))

@pytest.fixture
def room_data(room_model):
  data = mujoco.MjData(room_model)
  for _ in range(int(2.0 / room_model.opt.timestep)):
    mujoco.mj_step(room_model, data)
  return data

@pytest.fixture(scope="module")
def scanner(room_model):
  s = Scanner(room_model)
  yield s                      # tests run while we're "paused" here
  s.renderer.close()           # teardown (free the OpenGL context)


def test_center_ray_reads_wall_distance(scanner, room_data, room_model):
  angles, ranges = scanner.scan(room_data)
  cam_id = room_model.camera("left_eye").id
  camera_x = room_data.cam_xpos[cam_id][0]   # world x, kinematics already applied
  expected = WALL_FACE_X - camera_x
  mid = len(ranges) // 2
  center = (ranges[mid - 1] + ranges[mid]) / 2  # even width (no exact center column)
  assert abs(center - expected < 0.02)


def test_flat_wall_invariant(scanner, room_data):
  angles, ranges = scanner.scan(room_data)
  on_wall = np.abs(angles) < ON_WALL
  planar = ranges[on_wall] * np.cos(angles[on_wall])
  assert on_wall.sum() > 200       # Sanity: most rays are on the wall, not the black box
  assert planar.std() < 0.01       # all on-wall columns agree on the wall depth


def test_scan_shape_and_angles(scanner, room_data):
  angles, ranges = scanner.scan(room_data)
  assert angles.shape == ranges.shape == (320,)
  assert np.all(np.diff(angles) < 0)             # decreasing: +33.5° → −33.5°
  span = angles[0] - angles[-1]
  assert abs(span - math.radians(67)) < math.radians(2) # hfov from fovy=41 at 16:9
  assert np.all(ranges <= scanner.max_range + 1e-9)
