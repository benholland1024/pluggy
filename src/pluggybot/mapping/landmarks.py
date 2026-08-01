"""Landmark memory: world positions of recognized outlets.

Landmarks are the sparse complement to the occupancy grid: a handful of
continuous (x, y, z) points rather than dense cells. Repeat sightings of the
same outlet (inevitable, since odometry drifts between visits) are merged by
nearest-neighbor gating, and each merge refines the stored position with a
running average. Sighting counts double as a confidence filter: a spurious
detection is seen once, a real outlet every time the robot looks at it.
"""

import math

GATE_RADIUS = 0.4   # m: sightings closer than this (in 2D) are the same outlet.
                    # Odometry drift is <2% of path, outlets sit >1 m apart;
                    # z is excluded because it is the noisiest estimate and two
                    # outlets never share a wall spot at different heights.


class Landmark:
  def __init__(self, x: float, y: float, z: float) -> None:
    self.x = x
    self.y = y
    self.z = z
    self.n_sightings = 1     # a Landmark only exists because something was seen

  def merge(self, x: float, y: float, z: float) -> None:
    """Fold one new sighting into the position estimate.

    Running average: after n sightings the stored point is the mean of all n,
    so each new observation moves it by 1/n of the residual — early (noisy,
    far-away) sightings get corrected, later ones only fine-tune.
    """
    self.n_sightings += 1
    w = 1.0 / self.n_sightings
    self.x += (x - self.x) * w
    self.y += (y - self.y) * w
    self.z += (z - self.z) * w


class LandmarkStore:
  def __init__(self, gate_radius: float = GATE_RADIUS) -> None:
    self.gate_radius = gate_radius
    self.landmarks: list[Landmark] = []

  def add_sighting(self, x: float, y: float, z: float) -> Landmark:
    """Record one detection: merge into the nearest landmark within the gate
    (2D distance), or start a new landmark if none is close enough."""
    nearest = None
    min_dist = self.gate_radius
    for landmark in self.landmarks:
      dist = math.hypot(landmark.x - x, landmark.y - y)
      if dist < min_dist:
        min_dist = dist
        nearest = landmark

    if nearest is None:
      nearest = Landmark(x, y, z)
      self.landmarks.append(nearest)
    else:
      nearest.merge(x, y, z)
    return nearest

  def confirmed(self, min_sightings: int = 3) -> list[Landmark]:
    """Landmarks seen at least min_sightings times: the trustworthy ones."""
    return [lm for lm in self.landmarks if lm.n_sightings >= min_sightings]

  def nearest_confirmed(self, x: float, y: float,
                        min_sightings: int = 3) -> Landmark | None:
    """The confirmed landmark closest (2D) to a world point — recharge mode's
    'which outlet do I go to'. None if nothing is confirmed yet."""
    candidates = self.confirmed(min_sightings)
    if not candidates:
      return None
    return min(candidates, key=lambda lm: math.hypot(lm.x - x, lm.y - y))
