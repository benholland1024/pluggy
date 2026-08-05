"""Score the outlet detector on the REAL target distribution: room_1.xml.

The dataset's own val split is drawn from the same generator as training, so
it can only measure what the generator already thought to vary — it reported
mAP50-95 ~0.99 while the detector was calling a light switch an outlet. This
harness instead samples collision-free robot poses throughout room_1, renders
what the robot's camera would actually see, and scores detections against
segmentation ground truth.

Because it samples hundreds of poses rather than a handful, it is stable
enough to compare two trained models — an 8-pose spot check is not: a single
detection flipping is within run-to-run training variance.

Usage:
  MUJOCO_GL=egl uv run python scripts/eval_detector.py
  MUJOCO_GL=egl uv run python scripts/eval_detector.py --weights runs/detect/train-4/weights/best.pt
  MUJOCO_GL=egl uv run python scripts/eval_detector.py --poses 400 --conf 0.5
"""

import argparse
import math

import mujoco
import numpy as np

from pluggybot.perception.outlet_spotter import CAM_H, CAM_W, latest_weights

OUTLETS = ("outlet_a", "outlet_b", "outlet_c")
MIN_GT_PIXELS = 150      # below this an outlet is a few pixels; not a fair ask
IOU_MATCH = 0.3          # detection counts as finding a ground-truth outlet


def geom_ids(model, prefix):
  return [g for g in range(model.ngeom)
          if (model.geom(g).name or "").startswith(prefix)]


def mask_box(seg, ids):
  """(pixels, xyxy box) of a geom group in a segmentation image."""
  m = np.isin(seg[:, :, 0], ids) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
  n = int(m.sum())
  if n == 0:
    return 0, None
  ys, xs = np.nonzero(m)
  return n, (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def iou(a, b):
  ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
  ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
  if ix1 <= ix0 or iy1 <= iy0:
    return 0.0
  inter = (ix1 - ix0) * (iy1 - iy0)
  area_a = (a[2] - a[0]) * (a[3] - a[1])
  area_b = (b[2] - b[0]) * (b[3] - b[1])
  return inter / (area_a + area_b - inter)


def free_poses(model, data, rng, n_poses):
  """Collision-free (x, y, yaw) robot poses inside the rooms."""
  poses = []
  chassis = model.geom("chassis").id
  while len(poses) < n_poses:
    x, y = rng.uniform(-1.8, 5.8), rng.uniform(-1.8, 5.8)
    yaw = rng.uniform(-math.pi, math.pi)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [x, y, 0.045]
    data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    if not any(chassis in (data.contact[i].geom1, data.contact[i].geom2)
               for i in range(data.ncon)):
      poses.append((x, y, yaw))
  return poses


def main(weights, n_poses, conf, seed):
  from ultralytics import YOLO
  model = mujoco.MjModel.from_xml_path("models/room_1.xml")
  data = mujoco.MjData(model)
  detector = YOLO(weights)
  renderer = mujoco.Renderer(model, CAM_H, CAM_W)
  rng = np.random.default_rng(seed)

  outlet_ids = {o: geom_ids(model, o + "_") for o in OUTLETS}
  decoy_ids = geom_ids(model, "decoy_")

  n_gt = n_hit = n_fp = n_fp_on_decoy = 0
  frames_with_fp = 0
  missed_px = []

  for x, y, yaw in free_poses(model, data, rng, n_poses):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [x, y, 0.045]
    data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera="left_eye")
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="left_eye")
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    truths = []
    for o in OUTLETS:
      px, box = mask_box(seg, outlet_ids[o])
      if px >= MIN_GT_PIXELS:
        truths.append((o, px, box))

    res = detector.predict(np.ascontiguousarray(rgb[:, :, ::-1]),
                           conf=conf, verbose=False)[0]
    dets = [tuple(float(v) for v in b) for b in res.boxes.xyxy]

    used = set()
    for _, px, tbox in truths:
      n_gt += 1
      best, best_iou = None, IOU_MATCH
      for i, d in enumerate(dets):
        if i in used:
          continue
        s = iou(tbox, d)
        if s >= best_iou:
          best, best_iou = i, s
      if best is None:
        missed_px.append(px)
      else:
        used.add(best)
        n_hit += 1

    frame_fp = 0
    for i, d in enumerate(dets):
      if i in used:
        continue
      n_fp += 1
      frame_fp += 1
      # What did it fire on? Sample the segmentation under the box centre.
      cx, cy = int((d[0] + d[2]) / 2), int((d[1] + d[3]) / 2)
      cx, cy = min(max(cx, 0), CAM_W - 1), min(max(cy, 0), CAM_H - 1)
      if int(seg[cy, cx, 0]) in decoy_ids:
        n_fp_on_decoy += 1
    if frame_fp:
      frames_with_fp += 1

  recall = n_hit / n_gt if n_gt else float("nan")
  print(f"\nweights : {weights}")
  print(f"poses   : {n_poses}   conf threshold: {conf}")
  print(f"visible outlets (>= {MIN_GT_PIXELS} px): {n_gt}")
  print(f"  detected            : {n_hit}  (recall {recall:.3f})")
  print(f"  missed              : {n_gt - n_hit}"
        + (f"   median size of miss: {int(np.median(missed_px))} px" if missed_px else ""))
  print(f"false positives       : {n_fp}  in {frames_with_fp}/{n_poses} frames "
        f"({n_fp / n_poses:.3f} per frame)")
  print(f"  ...landing on a decoy: {n_fp_on_decoy}")
  renderer.close()


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--weights", default=None, help="default: newest under runs/")
  p.add_argument("--poses", type=int, default=300)
  p.add_argument("--conf", type=float, default=0.5)
  p.add_argument("--seed", type=int, default=5)
  a = p.parse_args()
  main(a.weights or latest_weights(), a.poses, a.conf, a.seed)
