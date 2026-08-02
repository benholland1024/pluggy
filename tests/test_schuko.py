"""Regression guards for the Schuko contact spike (docking/schuko.py).

These pin down the contact-modeling results milestone 6 will build on: if a
future MuJoCo upgrade or geometry tweak changes insertion behavior, this is
where it shows up first.
"""

from pluggybot.docking.schuko import PIN_LEN, run_trial


def test_centered_insertion_seats_fully():
  res, _ = run_trial()
  assert res["success"], res
  assert not res["unstable"]
  assert res["max_force_n"] < 5.0     # straight in: no fight with the funnel


def test_honest_rim_recovers_small_lateral_offset():
  """With the honest 2 mm rim chamfer, capture = clearance (0.75 mm) +
  chamfer: a 3 mm miss must still funnel home. This is the real tolerance
  the terminal approach has to hit -- do not widen the chamfer to make a
  failing controller pass; that's tuning the world instead of the robot."""
  res, _ = run_trial(y_off=0.003)
  assert res["success"], res
  assert not res["unstable"]


def test_dished_plate_widens_capture():
  """The parameterization itself: an 8 mm chamfer (a dished face plate, a
  possible hardware choice for the physical charging outlet) must recover
  a miss the honest rim cannot."""
  assert not run_trial(y_off=0.008)[0]["success"]
  assert run_trial(y_off=0.008, chamfer_len=0.008)[0]["success"]


def test_slight_yaw_jams_at_pin_length():
  """Yaw just past tolerance: the body enters the well, but the pins bottom
  on the floor beside their holes before the shallow well can square the
  body up -- the jam stops one pin length short. This signature is the
  diagnosis that made facing accuracy the docking controller's priority."""
  res, _ = run_trial(yaw_deg=6.0)
  assert not res["success"]
  assert not res["unstable"]
  assert abs(res["gap_mm"] - PIN_LEN * 1000) < 3.0, res


def test_large_yaw_jams_cleanly():
  """Way past tolerance (16 deg), the plug catches on the rim and never
  enters at all. What matters: the jam is CLEAN -- stalled against the
  force limit, no solver explosion."""
  res, _ = run_trial(yaw_deg=16.0)
  assert not res["success"]
  assert not res["unstable"]
  assert res["gap_mm"] > 15.0, res    # nowhere near seated
