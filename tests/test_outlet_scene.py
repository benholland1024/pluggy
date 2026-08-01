import mujoco
import numpy as np
import pytest

from pluggybot.perception.outlet_scene import make_labeled_sample, random_scene_xml


@pytest.fixture
def rng():
  return np.random.default_rng(3)


def test_random_scene_compiles(rng):
  for _ in range(5):
    xml, info = random_scene_xml(rng, with_outlet=True)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.ncam == 1
    assert info["with_outlet"]


def test_positive_sample_has_valid_bbox(rng):
  rgb, bbox = make_labeled_sample(rng, with_outlet=True)
  assert rgb is not None and bbox is not None
  cx, cy, w, h = bbox
  assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0
  assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0
  # a Schuko plate is wider than tall never: it's square -- but perspective
  # can stretch either way; just require it isn't degenerate or image-sized
  assert w * h < 0.5


def test_negative_sample_has_no_bbox(rng):
  rgb, bbox = make_labeled_sample(rng, with_outlet=False)
  assert rgb is not None
  assert bbox is None
  assert rgb.std() > 0        # an actual rendered scene, not a blank frame
