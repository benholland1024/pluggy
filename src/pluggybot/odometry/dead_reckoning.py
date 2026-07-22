# This script provides dead reckoning using change in wheel angle over time.

import math


class DeadReckoner:
  def __init__(self, wheel_radius: float, track_width: float) -> None:
    self.wheel_radius = wheel_radius
    self.track_width = track_width
    self.x = 0.0
    self.y = 0.0
    self.theta = 0.0
    self._prev_left: float | None = None
    self._prev_right: float | None = None
  
  def update(self, left_angle: float, right_angle: float):
    """Advance the pose from new absolute wheel angles (radians)."""

    if self._prev_left is None or self._prev_right is None:
      self._prev_left, self._prev_right = left_angle, right_angle
      return
    
    d_left = left_angle - self._prev_left
    d_right = right_angle - self._prev_right
    self._prev_left, self._prev_right = left_angle, right_angle

    # Velocity is the average of wheel surface speeds
    d_dist = self.wheel_radius * (d_left + d_right) / 2

    # Angular velocity (of the bot) is difference of wheel speeds, divided by track width
    d_theta = self.wheel_radius * (d_right - d_left) / self.track_width

    # Update position + angle info
    heading = self.theta + d_theta / 2  # midpoint heading: cheap accuracy win
    self.x += d_dist * math.cos(heading)
    self.y += d_dist * math.sin(heading)
    self.theta += d_theta

