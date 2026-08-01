"""Differential-drive command helpers shared by teleop and autonomous driving."""

import math

WHEEL_RADIUS = 0.045    # m — Pololu 90x10 wheel (see docs/Parts.md)
TRACK_WIDTH = 0.21      # m — distance between wheel centers
MAX_WHEEL_ACCEL = 30.0  # rad/s^2 — slew limit, like a real motor controller's ramp


def wheel_targets(vel: float, ang_vel: float) -> tuple[float, float]:
  """Body command (v m/s, w rad/s) -> (left, right) wheel angular velocities."""
  left = (vel - ang_vel * TRACK_WIDTH / 2) / WHEEL_RADIUS
  right = (vel + ang_vel * TRACK_WIDTH / 2) / WHEEL_RADIUS
  return left, right


def slew(current: float, target: float, dt: float, max_accel: float = MAX_WHEEL_ACCEL) -> float:
  """Move current toward target, changing by at most max_accel * dt."""
  step = max_accel * dt
  return current + max(-step, min(step, target - current))


def wrap_angle(a: float) -> float:
  """Wrap an angle to (-pi, pi]."""
  return math.atan2(math.sin(a), math.cos(a))
