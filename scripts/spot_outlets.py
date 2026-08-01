"""Demo: the full outlet-memory chain, driven by real (drifting) odometry.

The robot drives a short scripted route in room_1 — spin 360, drive forward,
spin 360 — while the OutletSpotter looks through its camera every half sim
second. Each detection is projected to world coordinates using the DEAD
RECKONED pose (not ground truth: this is the honest test) and fed to the
LandmarkStore, which merges repeat sightings across the route. At the end,
remembered outlets are compared against the true outlet positions in the MJCF.

Usage:
  MUJOCO_GL=egl uv run python scripts/spot_outlets.py
  MUJOCO_GL=egl uv run python scripts/spot_outlets.py --weights runs/detect/train-2/weights/best.pt
"""

import argparse
import glob
import math
import os

import mujoco

from pluggybot.control import wheel_targets, slew
from pluggybot.mapping.landmarks import LandmarkStore
from pluggybot.odometry.dead_reckoning import DeadReckoner
from pluggybot.perception.outlet_spotter import OutletSpotter

SPOT_PERIOD = 0.5    # sim seconds between YOLO looks
W_SPIN = 1.0         # rad/s during look-around spins
V_DRIVE = 0.3        # m/s during the straight leg

# (mode, amount): spin amounts are radians, drive amounts are seconds.
ROUTE = [("spin", 2 * math.pi), ("drive", 3.0), ("spin", 2 * math.pi)]


def latest_weights():
  hits = sorted(glob.glob("runs/detect/*/weights/best.pt"), key=os.path.getmtime)
  if not hits:
    raise SystemExit("no trained weights under runs/detect/ -- train first "
                     "(see README) or pass --weights")
  return hits[-1]


def main(weights: str) -> None:
  model = mujoco.MjModel.from_xml_path("models/room_1.xml")
  data = mujoco.MjData(model)
  reckoner = DeadReckoner(wheel_radius=0.045, track_width=0.21)
  spotter = OutletSpotter(model, weights)
  store = LandmarkStore()

  left = model.actuator("left_motor").id
  right = model.actuator("right_motor").id
  left_adr = model.joint("left_wheel_joint").qposadr[0]
  right_adr = model.joint("right_wheel_joint").qposadr[0]
  gyro = model.sensor("imu_gyro")

  next_spot = 0.0
  n_sightings = 0
  for mode, amount in ROUTE:
    remaining = amount
    while remaining > 0:
      reckoner.update(
        data.qpos[left_adr], data.qpos[right_adr],
        gyro_yaw_rate=data.sensordata[gyro.adr[0] + 2], dt=model.opt.timestep,
      )
      pose = (reckoner.x, reckoner.y, reckoner.theta)

      if data.time >= next_spot:
        next_spot = data.time + SPOT_PERIOD
        for x, y, z, conf in spotter.spot(data, pose):
          store.add_sighting(x, y, z)
          n_sightings += 1
          print(f"t={data.time:5.1f}s  sighting at ({x:+.2f}, {y:+.2f}, "
                f"z={z:.2f})  conf={conf:.2f}")

      if mode == "spin":
        v, w = 0.0, W_SPIN
        remaining -= W_SPIN * model.opt.timestep
      else:
        v, w = V_DRIVE, 0.0
        remaining -= model.opt.timestep
      tl, tr = wheel_targets(v, w)
      data.ctrl[left] = slew(data.ctrl[left], tl, model.opt.timestep)
      data.ctrl[right] = slew(data.ctrl[right], tr, model.opt.timestep)
      mujoco.mj_step(model, data)

  truths = {name: model.body(name).pos for name in ("outlet_a", "outlet_b", "outlet_c")}
  confirmed = store.confirmed(min_sightings=3)
  print(f"\nroute done at t={data.time:.1f}s: {n_sightings} sightings -> "
        f"{len(store.landmarks)} landmarks ({len(confirmed)} confirmed)")
  for lm in store.landmarks:
    name, truth = min(truths.items(),
                      key=lambda kv: math.hypot(kv[1][0] - lm.x, kv[1][1] - lm.y))
    err = math.sqrt((lm.x - truth[0]) ** 2 + (lm.y - truth[1]) ** 2
                    + (lm.z - truth[2]) ** 2)
    tag = "CONFIRMED" if lm in confirmed else f"seen x{lm.n_sightings}"
    print(f"  ({lm.x:+.2f}, {lm.y:+.2f}, z={lm.z:.2f})  {tag:<10s} "
          f"nearest truth: {name} err={err * 100:.1f} cm")
  print("\n(outlet_c is in room 2 -- the route never sees it; that's expected)")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--weights", default=None, help="YOLO weights (.pt); default: newest under runs/")
  args = p.parse_args()
  main(args.weights or latest_weights())
