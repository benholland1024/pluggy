import mujoco
import numpy as np
import pytest
from scipy import ndimage

from pluggybot.perception.outlet_scene import (
  IMG_H,
  IMG_W,
  MIN_LABEL_PIXELS,
  OUTLET_PREFIX,
  make_labeled_sample,
  random_scene_xml,
)


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


def _outlet_mask(xml):
  """The raw segmentation mask a label would be derived from, unfiltered."""
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
  try:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="cam")
    seg = renderer.render()
  finally:
    renderer.close()
  ids = [g for g in range(model.ngeom)
         if (model.geom(g).name or "").startswith(OUTLET_PREFIX)]
  return np.isin(seg[:, :, 0], ids) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))


def test_segmentation_mask_has_no_stray_blobs(rng):
  """Guards <quality offsamples="0"/> in random_scene_xml.

  With MSAA on, MuJoCo blends geom IDs at object edges and the segmentation
  buffer reports outlet IDs on pixels far from the outlet. Since a label box
  spans min/max of the mask, one stray pixel stretches it across the frame:
  12% of positives grew a second blob and 5% came out grossly elongated.
  An outlet renders as exactly one contiguous blob.
  """
  checked = 0
  for _ in range(30):
    mask = _outlet_mask(random_scene_xml(rng, with_outlet=True)[0])
    if mask.sum() < MIN_LABEL_PIXELS:
      continue                  # outlet occluded or off-frame; not this test's job
    checked += 1
    assert ndimage.label(mask)[1] == 1, "stray segmentation blob — is offsamples=0 still set?"
  assert checked >= 10, f"only {checked} usable scenes; the probe is not exercising anything"


def test_label_boxes_stay_plausibly_square(rng):
  """A Schuko plate is square (80x80 mm), so perspective can compress a box
  but never stretch it into a bar. Elongated boxes meant a corrupt label."""
  ratios = []
  for _ in range(20):
    _, bbox = make_labeled_sample(rng, with_outlet=True)
    if bbox is None:
      continue
    _, _, w, h = bbox
    ratios.append(w / h)
  assert len(ratios) >= 10
  assert max(ratios) < 2.0, f"elongated label box (w/h={max(ratios):.2f}): stray mask pixels?"
