"""Schuko contact spike: sweep insertion misalignment, report tolerance.

The de-risk experiment from the plan's parallel track (see docs/PluggyPlan.md):
no robot, just a plug on a compliant force-limited carrier rammed at a socket
built from convex primitives (see pluggybot/docking/schuko.py for why and how).
Answers, empirically, before milestone 6 depends on them:

  1. does the recess funnel a misaligned plug home?     (watch the filmstrip)
  2. how much lateral / angular error is forgiven?      (read the table)
  3. do the contacts stay stable at mm scale?           (max force, unstable)

Usage:
  MUJOCO_GL=egl uv run python scripts/schuko_spike.py
  MUJOCO_GL=egl uv run python scripts/schuko_spike.py --dump-xml /tmp/schuko.xml
  # then inspect interactively:  uv run python -m mujoco.viewer --mjcf=/tmp/schuko.xml
"""

import argparse

import numpy as np
from PIL import Image

from pluggybot.docking.schuko import run_trial, scene_xml

LATERAL_MM = [0, 1, 2, 3, 4, 5, 6]     # y offsets, honest 2 mm rim chamfer
VERTICAL_MM = [1, 2, 3, 4]
YAW_DEG = [1, 2, 3, 4, 6]
CHAMFER_STUDY_MM = [2, 4, 8]           # what a dished face plate would buy
FILMSTRIP = "schuko_spike.png"               # repo root, like map.png


def report(label: str, results: list[tuple[float, dict]], unit: str) -> float:
  """Print one sweep table; return the largest offset that still succeeded."""
  print(f"\n-- {label} --")
  last_good = 0.0
  for off, res in results:
    verdict = "OK  " if res["success"] else "JAM "
    if res["unstable"]:
      verdict = "UNSTABLE"
    print(f"  {off:5.1f} {unit}:  {verdict}  gap={res['gap_mm']:6.2f} mm  "
          f"max force={res['max_force_n']:5.1f} N")
    if res["success"]:
      last_good = max(last_good, off)
  return last_good


def main(dump_xml: str | None) -> None:
  if dump_xml:
    with open(dump_xml, "w") as f:
      f.write(scene_xml())
    print(f"wrote {dump_xml}")

  lat = [(o, run_trial(y_off=o / 1000)[0]) for o in LATERAL_MM]
  vert = [(o, run_trial(z_off=o / 1000)[0]) for o in VERTICAL_MM]
  yaw = [(o, run_trial(yaw_deg=o)[0]) for o in YAW_DEG]

  lat_tol = report("lateral offset (y)", lat, "mm")
  vert_tol = report("vertical offset (z)", vert, "mm")
  yaw_tol = report("angular offset (yaw)", yaw, "deg")

  # How much would a beveled/dished face plate buy? (A hardware decision:
  # the physical charging outlet could be chosen/mounted for a bigger bevel.)
  print("\n-- lateral tolerance vs entry chamfer size --")
  for ch in CHAMFER_STUDY_MM:
    tol = 0.0
    for off in [2, 3, 4, 6, 8, 10, 14, 18]:
      res, _ = run_trial(y_off=off / 1000, chamfer_len=ch / 1000)
      if res["success"]:
        tol = off
    print(f"  {ch} mm chamfer: inserts up to +/-{tol:.0f} mm lateral")

  print(f"\ntolerance summary (honest rim): lateral +/-{lat_tol:.0f} mm, "
        f"vertical +/-{vert_tol:.0f} mm, yaw +/-{yaw_tol:.0f} deg")
  print("(outlet landmarks are good to ~9 cm; the terminal visual servo / RL")
  print(" policy must close that gap down to these numbers before contact")
  print(" takes over -- or the physical outlet gets a dished plate, see above.)")

  # Filmstrip: one comfortable success and the edge of the envelope.
  strips = []
  for off in (max(1.0, lat_tol // 2), lat_tol):
    _, frames = run_trial(y_off=off / 1000, n_frames=8)
    strips.append(np.concatenate(frames, axis=1))
  Image.fromarray(np.concatenate(strips, axis=0)).save(FILMSTRIP)
  print(f"\nfilmstrip (top: mid-tolerance, bottom: edge of tolerance) -> {FILMSTRIP}")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--dump-xml", default=None,
                 help="also write the (centered) MJCF for viewer inspection")
  args = p.parse_args()
  main(args.dump_xml)
