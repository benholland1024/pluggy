"""Generate a YOLO-format outlet-detection dataset from randomized MJCF scenes.

Labels are free: outlets are located per-pixel via segmentation rendering,
so every bounding box is exact and no human ever labels an image.

Usage:
  MUJOCO_GL=egl uv run python scripts/generate_outlet_dataset.py --count 1200
  # then train (after `uv add ultralytics`):
  #   yolo detect train data=datasets/outlets/dataset.yaml model=yolo11n.pt epochs=50 imgsz=640
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pluggybot.perception.outlet_scene import IMG_H, IMG_W, make_labeled_sample

NEGATIVE_FRACTION = 0.25   # images with no outlet: the detector must learn silence
VAL_FRACTION = 0.1
CONTACT_SHEET_N = 24


def draw_bbox(img, bbox):
  d = ImageDraw.Draw(img)
  cx, cy, w, h = bbox
  x0, y0 = (cx - w / 2) * IMG_W, (cy - h / 2) * IMG_H
  x1, y1 = (cx + w / 2) * IMG_W, (cy + h / 2) * IMG_H
  d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=2)
  return img


def clean_output_dirs(out: Path):
  """Delete images/ and labels/ wholesale, then recreate the split dirs.

  Regenerating into a dirty directory silently contaminates the dataset: the
  train/val split is a random draw per image, so a file whose split changes
  between runs leaves its old copy alive in the other split. One run left 195
  stale image/label pairs from a buggy predecessor mixed into training.
  """
  for sub in ("images", "labels"):
    shutil.rmtree(out / sub, ignore_errors=True)
  for split in ("train", "val"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)


def main(count, out, seed):
  rng = np.random.default_rng(seed)
  out = Path(out)
  clean_output_dirs(out)

  sheet_tiles, n_pos, n_neg, box_areas = [], 0, 0, []
  saved = 0
  while saved < count:
    with_outlet = rng.random() > NEGATIVE_FRACTION
    rgb, bbox = make_labeled_sample(rng, with_outlet)
    if rgb is None:
      continue                             # scene rerolls exhausted; try a new draw
    split = "val" if rng.random() < VAL_FRACTION else "train"
    name = f"img_{saved:05d}"
    Image.fromarray(rgb).save(out / "images" / split / f"{name}.png")
    label = "" if bbox is None else f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
    (out / "labels" / split / f"{name}.txt").write_text(label)

    if bbox is None:
      n_neg += 1
    else:
      n_pos += 1
      box_areas.append(bbox[2] * bbox[3])
    if len(sheet_tiles) < CONTACT_SHEET_N:
      tile = Image.fromarray(rgb)
      if bbox is not None:
        tile = draw_bbox(tile, bbox)
      sheet_tiles.append(tile.resize((IMG_W // 2, IMG_H // 2)))
    saved += 1
    if saved % 100 == 0:
      print(f"{saved}/{count} images ({n_pos} positives, {n_neg} negatives)")

  (out / "dataset.yaml").write_text(
    f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: outlet\n"
  )

  cols, rows = 6, (len(sheet_tiles) + 5) // 6
  sheet = Image.new("RGB", (cols * IMG_W // 2, rows * IMG_H // 2))
  for i, tile in enumerate(sheet_tiles):
    sheet.paste(tile, ((i % cols) * IMG_W // 2, (i // cols) * IMG_H // 2))
  sheet.save(out / "contact_sheet.png")

  areas = np.array(box_areas)
  print(f"\ndone: {n_pos} positives, {n_neg} negatives -> {out}")
  print(f"bbox area (fraction of image): median {np.median(areas):.4f}, "
        f"min {areas.min():.4f}, max {areas.max():.4f}")
  print(f"inspect {out / 'contact_sheet.png'} before training!")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--count", type=int, default=1200)
  p.add_argument("--out", default="datasets/outlets")
  p.add_argument("--seed", type=int, default=7)
  args = p.parse_args()
  main(args.count, args.out, args.seed)
