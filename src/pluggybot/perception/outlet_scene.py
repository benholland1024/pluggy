"""Procedural MJCF scenes for outlet-detector training data.

Everything here emits MJCF XML strings that get compiled per-image with
mujoco.MjModel.from_xml_string. Labels come for free from segmentation
rendering: every outlet geom is named with OUTLET_PREFIX, so its pixels can
be identified in the segmentation image and boxed. No human labeling.

Domain randomization: room size, wall/floor colors and textures, lighting,
outlet wall/position/height, camera pose, floor clutter, and wall-mounted
decoys (blank plates, light switches, pictures) that punish lazy features.
"""

import math

import mujoco
import numpy as np
from scipy import ndimage

OUTLET_PREFIX = "outlet_"
MIN_LABEL_PIXELS = 60       # an outlet smaller than this is unlearnable; reroll
IMG_W, IMG_H = 640, 360     # 16:9 to match the robot camera's aspect (fovy 41)


def outlet_xml(x, y, z, facing_deg):
  """A German Schuko socket from primitives. facing_deg: 0 faces +x, 90 faces +y.

  No real holes exist in MJCF primitives, so the recess is faked with layered
  discs: plate -> darker well ring -> lighter well floor -> near-black pin
  holes (19 mm apart) -> metal earthing clips top and bottom. Reads correctly
  from ~0.3 m and beyond, which is all the detector will ever see.
  """
  return f"""
    <body name="outlet" pos="{x:.4f} {y:.4f} {z:.4f}" euler="0 0 {facing_deg}">
      <geom name="{OUTLET_PREFIX}plate" type="box" size="0.004 0.04 0.04"
            rgba="0.93 0.92 0.88 1" contype="0" conaffinity="0"/>
      <geom name="{OUTLET_PREFIX}well" type="cylinder" size="0.028 0.0015" zaxis="1 0 0"
            pos="0.0045 0 0" rgba="0.72 0.71 0.67 1" contype="0" conaffinity="0"/>
      <geom name="{OUTLET_PREFIX}wellfloor" type="cylinder" size="0.024 0.0015" zaxis="1 0 0"
            pos="0.0058 0 0" rgba="0.86 0.85 0.81 1" contype="0" conaffinity="0"/>
      <geom name="{OUTLET_PREFIX}hole_l" type="cylinder" size="0.0024 0.0015" zaxis="1 0 0"
            pos="0.0068 0.0095 0" rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>
      <geom name="{OUTLET_PREFIX}hole_r" type="cylinder" size="0.0024 0.0015" zaxis="1 0 0"
            pos="0.0068 -0.0095 0" rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>
      <geom name="{OUTLET_PREFIX}clip_top" type="box" size="0.0018 0.005 0.0015"
            pos="0.0062 0 0.024" rgba="0.62 0.63 0.65 1" contype="0" conaffinity="0"/>
      <geom name="{OUTLET_PREFIX}clip_bot" type="box" size="0.0018 0.005 0.0015"
            pos="0.0062 0 -0.024" rgba="0.62 0.63 0.65 1" contype="0" conaffinity="0"/>
    </body>"""


def _blank_plate_xml(i, x, y, z, facing_deg, rgba):
  return f"""
    <body pos="{x:.4f} {y:.4f} {z:.4f}" euler="0 0 {facing_deg}">
      <geom name="decoy_plate_{i}" type="box" size="0.004 0.04 0.04"
            rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>"""


def _switch_xml(i, x, y, z, facing_deg, rgba):
  return f"""
    <body pos="{x:.4f} {y:.4f} {z:.4f}" euler="0 0 {facing_deg}">
      <geom name="decoy_swplate_{i}" type="box" size="0.004 0.04 0.04"
            rgba="{rgba}" contype="0" conaffinity="0"/>
      <geom name="decoy_rocker_{i}" type="box" size="0.004 0.018 0.026"
            pos="0.004 0 0" rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>"""


def _picture_xml(i, x, y, z, facing_deg, w, h, rgba):
  return f"""
    <body pos="{x:.4f} {y:.4f} {z:.4f}" euler="0 0 {facing_deg}">
      <geom name="decoy_pic_{i}" type="box" size="0.002 {w:.3f} {h:.3f}"
            rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>"""


def _rgba(rng, lo=0.15, hi=0.95):
  r, g, b = rng.uniform(lo, hi, 3)
  return f"{r:.3f} {g:.3f} {b:.3f} 1"


def _camera_xyaxes(cam_pos, look_at):
  """MuJoCo camera frame from a position and a look-at point."""
  fwd = np.asarray(look_at, float) - np.asarray(cam_pos, float)
  fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, [0.0, 0.0, 1.0])
  n = np.linalg.norm(right)
  right = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else right / n
  up = np.cross(right, fwd)
  return " ".join(f"{v:.5f}" for v in (*right, *up))


# Walls are indexed 0..3: south (y=0), east (x=W), north (y=D), west (x=0).
# Each entry: inward normal angle in degrees, and a function mapping a
# position `along` the wall + mount offset -> world (x, y).
def _wall_frames(W, D):
  return [
    (90.0, lambda a, off: (a, off)),          # south wall, faces +y
    (180.0, lambda a, off: (W - off, a)),     # east wall, faces -x
    (270.0, lambda a, off: (a, D - off)),     # north wall, faces -y
    (0.0, lambda a, off: (off, a)),           # west wall, faces +x
  ]


def random_scene_xml(rng, with_outlet=True):
  """One randomized room as MJCF. Returns (xml_string, info dict)."""
  W, D = rng.uniform(3.0, 7.0), rng.uniform(3.0, 7.0)
  wall_h = rng.uniform(0.5, 1.5)
  frames = _wall_frames(W, D)
  body_parts = []

  # -- outlet on a random wall
  info = {"with_outlet": with_outlet}
  if with_outlet:
    wall = int(rng.integers(4))
    length = W if wall in (0, 2) else D
    along = rng.uniform(0.3, length - 0.3)
    height = rng.uniform(0.22, 0.45)
    facing, to_world = frames[wall]
    ox, oy = to_world(along, 0.02 + 0.004)      # wall thickness + plate half
    body_parts.append(outlet_xml(ox, oy, height, facing))
    info.update(outlet_pos=(ox, oy, height), wall=wall)

  # -- wall decoys (hard negatives) and pictures
  for i in range(int(rng.integers(0, 4))):
    wall = int(rng.integers(4))
    length = W if wall in (0, 2) else D
    along = rng.uniform(0.3, length - 0.3)
    if with_outlet and wall == info.get("wall") and abs(along - info["outlet_pos"][0 if wall in (0, 2) else 1]) < 0.25:
      continue                                   # don't overlap the outlet
    facing, to_world = frames[wall]
    dx, dy = to_world(along, 0.025)
    mount_z = rng.uniform(0.2, wall_h - 0.12)   # keep decoys on the wall, whatever its height
    kind = rng.random()
    if kind < 0.35:
      body_parts.append(_blank_plate_xml(i, dx, dy, mount_z, facing, _rgba(rng, 0.7, 0.95)))
    elif kind < 0.7:
      body_parts.append(_switch_xml(i, dx, dy, mount_z, facing, _rgba(rng, 0.7, 0.95)))
    else:
      body_parts.append(_picture_xml(i, dx, dy, mount_z, facing,
                                     rng.uniform(0.08, 0.3), rng.uniform(0.05, 0.12), _rgba(rng)))

  # -- floor clutter
  for i in range(int(rng.integers(0, 5))):
    hx, hy, hz = rng.uniform(0.05, 0.4, 3)
    bx, by = rng.uniform(0.5, W - 0.5), rng.uniform(0.5, D - 0.5)
    body_parts.append(f"""
    <geom name="clutter_{i}" type="box" size="{hx:.3f} {hy:.3f} {hz:.3f}"
          pos="{bx:.3f} {by:.3f} {hz:.3f}" rgba="{_rgba(rng)}" contype="0" conaffinity="0"/>""")

  # -- camera
  cam_z = rng.uniform(0.10, 0.35)
  if with_outlet:
    ox, oy, oz = info["outlet_pos"]
    normal = math.radians(frames[info["wall"]][0])
    for _ in range(50):
      ang = normal + math.radians(rng.uniform(-60, 60))
      dist = rng.uniform(0.35, 2.5)   # beyond ~2.5 m a Schuko plate is <20 px: unlearnable
      cx, cy = ox + dist * math.cos(ang), oy + dist * math.sin(ang)
      if 0.15 < cx < W - 0.15 and 0.15 < cy < D - 0.15:
        break
    look = (ox + rng.uniform(-0.1, 0.1), oy + rng.uniform(-0.1, 0.1), oz + rng.uniform(-0.08, 0.08))
    cam_pos = (cx, cy, cam_z)
  else:
    cam_pos = (rng.uniform(0.3, W - 0.3), rng.uniform(0.3, D - 0.3), cam_z)
    yaw = rng.uniform(0, 2 * math.pi)
    look = (cam_pos[0] + math.cos(yaw), cam_pos[1] + math.sin(yaw), cam_z + rng.uniform(-0.15, 0.15))
  xyaxes = _camera_xyaxes(cam_pos, look)

  # -- floor: plain color or checker texture
  assets, floor_extra = "", f'rgba="{_rgba(rng)}"'
  if rng.random() < 0.4:
    assets = f"""
  <asset>
    <texture name="floortex" type="2d" builtin="checker" width="256" height="256"
             rgb1="{_rgba(rng)[:-2]}" rgb2="{_rgba(rng)[:-2]}"/>
    <material name="floormat" texture="floortex" texrepeat="8 8"/>
  </asset>"""
    floor_extra = 'material="floormat"'

  # -- walls (sometimes one shared color, sometimes per-wall)
  shared = _rgba(rng) if rng.random() < 0.5 else None
  wall_geoms = []
  for wall, (cx, cy, sx, sy) in enumerate([
    (W / 2, 0.0, W / 2 + 0.02, 0.02),
    (W, D / 2, 0.02, D / 2 + 0.02),
    (W / 2, D, W / 2 + 0.02, 0.02),
    (0.0, D / 2, 0.02, D / 2 + 0.02),
  ]):
    wall_geoms.append(f"""
    <geom name="wall_{wall}" type="box" size="{sx:.3f} {sy:.3f} {wall_h / 2:.3f}"
          pos="{cx:.3f} {cy:.3f} {wall_h / 2:.3f}" rgba="{shared or _rgba(rng)}"/>""")

  light = f"""
    <light directional="true" dir="{rng.uniform(-0.4, 0.4):.2f} {rng.uniform(-0.4, 0.4):.2f} -1"
           diffuse="{rng.uniform(0.4, 0.9):.2f} {rng.uniform(0.4, 0.9):.2f} {rng.uniform(0.4, 0.9):.2f}"/>
    <light pos="{rng.uniform(0, W):.2f} {rng.uniform(0, D):.2f} {rng.uniform(1.5, 3):.2f}"
           diffuse="{rng.uniform(0.2, 0.7):.2f} {rng.uniform(0.2, 0.7):.2f} {rng.uniform(0.2, 0.7):.2f}"/>"""

  xml = f"""
<mujoco>
  <visual><global offwidth="{IMG_W}" offheight="{IMG_H}"/>
    <!-- offsamples="0" disables multisample antialiasing. MSAA blends geom IDs
         at object edges, so the segmentation buffer invents pixels carrying IDs
         no geom owns there — and when one collides with an outlet geom's ID, the
         bbox stretches from the outlet to that speck. Measured: 12% of positive
         labels had stray blobs, 5% were grossly elongated; 0% with this set. -->
    <quality offsamples="0"/>
    <headlight ambient="{rng.uniform(0.15, 0.45):.2f} {rng.uniform(0.15, 0.45):.2f} {rng.uniform(0.15, 0.45):.2f}"/>
  </visual>{assets}
  <worldbody>{light}
    <geom name="floor" type="plane" size="{W:.2f} {D:.2f} 0.1" pos="{W / 2:.2f} {D / 2:.2f} 0" {floor_extra}/>
    {"".join(wall_geoms)}
    {"".join(body_parts)}
    <camera name="cam" pos="{cam_pos[0]:.4f} {cam_pos[1]:.4f} {cam_pos[2]:.4f}" xyaxes="{xyaxes}" fovy="41"/>
  </worldbody>
</mujoco>"""
  return xml, info


def make_labeled_sample(rng, with_outlet, max_attempts=8):
  """Render one (rgb, bbox) training sample; bbox is normalized YOLO
  (cx, cy, w, h) or None for a negative. Rerolls scenes where the outlet
  is present but occluded/too tiny to be honest supervision."""
  for _ in range(max_attempts):
    xml, _ = random_scene_xml(rng, with_outlet)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    try:
      renderer.update_scene(data, camera="cam")
      rgb = renderer.render()
      if not with_outlet:
        return rgb, None
      renderer.enable_segmentation_rendering()
      renderer.update_scene(data, camera="cam")
      seg = renderer.render()          # (H, W, 2): [geom id, object type]
    finally:
      renderer.close()

    outlet_ids = [g for g in range(model.ngeom)
                  if (model.geom(g).name or "").startswith(OUTLET_PREFIX)]
    mask = np.isin(seg[:, :, 0], outlet_ids) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))

    # Belt and braces: an outlet is one contiguous blob, so keep only the
    # largest. The box spans min/max of the mask, so a single stray pixel
    # elsewhere in the frame stretches it across the image. `offsamples="0"`
    # (see random_scene_xml) removes the known source of strays; this catches
    # any that survive rather than silently poisoning a label.
    labels, n_blobs = ndimage.label(mask)
    if n_blobs > 1:
      counts = np.bincount(labels.ravel())
      counts[0] = 0                    # index 0 is background, never the winner
      mask = labels == counts.argmax()

    if mask.sum() < MIN_LABEL_PIXELS:
      continue                         # occluded or too far away: reroll the scene
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bbox = ((x0 + x1) / 2 / IMG_W, (y0 + y1) / 2 / IMG_H,
            (x1 - x0) / IMG_W, (y1 - y0) / IMG_H)
    return rgb, bbox
  return None, None                    # give up on this sample entirely
