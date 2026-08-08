"""What PluggyBot sees, in one PNG (issue #1).

A 2x2 dashboard image, refreshed on the same cadence as map.png:

    +-----------+-----------+
    | left_eye  | right_eye |     top row: the stereo pair (mapping eyes)
    +-----------+-----------+
    | occupancy | dock_eye  |     bottom: the map + the docking camera
    +-----------+-----------+

One offscreen Renderer serves all three cameras -- update_scene() just
switches which camera it looks through, so the cost is three renders plus a
map rescale per save. At the map.png cadence (2 Hz sim time) that is well
under the price the outlet spotter already pays for its own renders, which
is why the feature is flag-gated rather than always-on: scripts that never
ask for it never create the renderer at all.
"""

import mujoco
import numpy as np
from PIL import Image, ImageDraw

TILE_W, TILE_H = 320, 180        # per camera tile; the composite is 640x360
BACKGROUND = 40                  # letterbox gray behind the map tile
SEAM = 2                         # px gap between tiles, drawn in BACKGROUND

CAMERAS = ("left_eye", "right_eye", "dock_eye")


class ViewDashboard:
  """Renders the composite view image. Reuse one instance per run.

  map_img is optional so camera-only scripts (teleop.py) can still use the
  dashboard: without it the map tile is a labeled blank.
  """

  def __init__(self, model, tile_w: int = TILE_W, tile_h: int = TILE_H) -> None:
    self.tile_w, self.tile_h = tile_w, tile_h
    self.renderer = mujoco.Renderer(model, tile_h, tile_w)

  def _camera_tile(self, data, name: str) -> np.ndarray:
    self.renderer.update_scene(data, camera=name)
    return self.renderer.render()

  def _map_tile(self, map_img: np.ndarray | None) -> np.ndarray:
    tile = np.full((self.tile_h, self.tile_w, 3), BACKGROUND, dtype=np.uint8)
    if map_img is None:
      return tile
    if map_img.ndim == 2:                       # grayscale -> RGB
      map_img = np.stack([map_img] * 3, axis=-1)
    # Fit preserving aspect; NEAREST keeps grid cells crisp instead of smearing
    # occupancy values into in-between grays (the renderer-as-measurement
    # lesson: this image is data, not scenery).
    scale = min(self.tile_w / map_img.shape[1], self.tile_h / map_img.shape[0])
    w, h = round(map_img.shape[1] * scale), round(map_img.shape[0] * scale)
    scaled = np.asarray(
      Image.fromarray(map_img).resize((w, h), Image.NEAREST))
    x0, y0 = (self.tile_w - w) // 2, (self.tile_h - h) // 2
    tile[y0:y0 + h, x0:x0 + w] = scaled
    return tile

  def render(self, data, map_img: np.ndarray | None = None) -> np.ndarray:
    """The composite frame as an (2*tile_h+SEAM, 2*tile_w+SEAM, 3) uint8 array."""
    left, right, dock = (self._camera_tile(data, c) for c in CAMERAS)
    tiles = ((left, "left eye"), (right, "right eye"),
             (self._map_tile(map_img), "map"), (dock, "dock eye"))

    th, tw = self.tile_h, self.tile_w
    out = np.full((2 * th + SEAM, 2 * tw + SEAM, 3), BACKGROUND, dtype=np.uint8)
    for i, (tile, label) in enumerate(tiles):
      y0, x0 = (i // 2) * (th + SEAM), (i % 2) * (tw + SEAM)
      out[y0:y0 + th, x0:x0 + tw] = tile
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    for i, (_, label) in enumerate(tiles):
      y0, x0 = (i // 2) * (th + SEAM), (i % 2) * (tw + SEAM)
      draw.text((x0 + 5, y0 + 3), label, fill=(255, 220, 60))
    return np.asarray(img)

  def save(self, data, map_img: np.ndarray | None = None,
           path: str = "views.png") -> None:
    Image.fromarray(self.render(data, map_img)).save(path)

  def close(self) -> None:
    self.renderer.close()
