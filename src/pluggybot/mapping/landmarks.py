"""Landmark memory: world positions of recognized outlets.

Landmarks are the sparse complement to the occupancy grid: a handful of
continuous (x, y, z) points rather than dense cells. Repeat sightings of the
same outlet (inevitable, since odometry drifts between visits) are merged by
nearest-neighbor gating, and each merge refines the stored position with a
running average. Sighting counts double as a confidence filter: a spurious
detection is seen once, a real outlet every time the robot looks at it.

Each landmark also averages WHERE the robot stood when it saw the outlet.
The mean seen-from point is necessarily out in free space on the outlet's
open side, so it hands the recharge behavior an approach direction without
any wall-normal geometry: see Landmark.standoff().
"""

import math

GATE_RADIUS = 0.4   # m: sightings closer than this (in 2D) are the same outlet.
                    # Odometry drift is <2% of path, outlets sit >1 m apart;
                    # z is excluded because it is the noisiest estimate and two
                    # outlets never share a wall spot at different heights.


class Landmark:
  def __init__(self, x: float, y: float, z: float,
               seen_from: tuple[float, float]) -> None:
    self.x = x
    self.y = y
    self.z = z
    self.n_sightings = 1     # a Landmark only exists because something was seen
    # Mean robot position across sightings (2D: the camera height is fixed,
    # so a seen-from z would be a constant, not information).
    self.seen_from_x, self.seen_from_y = seen_from

  def merge(self, x: float, y: float, z: float,
            seen_from: tuple[float, float]) -> None:
    """Fold one new sighting into the estimates.

    Running average: after n sightings each stored value is the mean of all
    n, so a new observation moves it by 1/n of the residual — early (noisy,
    far-away) sightings get corrected, later ones only fine-tune.
    """
    self.n_sightings += 1
    w = 1.0 / self.n_sightings
    self.x += (x - self.x) * w
    self.y += (y - self.y) * w
    self.z += (z - self.z) * w
    self.seen_from_x += (seen_from[0] - self.seen_from_x) * w
    self.seen_from_y += (seen_from[1] - self.seen_from_y) * w

  def standoff(self, distance: float = 0.6) -> tuple[float, float, float]:
    """The docking start pose: (x, y, heading) `distance` out from the
    outlet on its open side, facing it.

    Direction comes from the mean seen-from point: the robot only ever saw
    the outlet from in front of its wall, so outlet -> mean-seen-from points
    into free space along (roughly) the wall normal. Heading points back at
    the outlet — the pose the docking controller starts from.
    """
    dx = self.seen_from_x - self.x
    dy = self.seen_from_y - self.y
    norm = math.hypot(dx, dy) or 1.0    # degenerate only if seen from inside the wall
    sx = self.x + distance * dx / norm
    sy = self.y + distance * dy / norm
    return sx, sy, math.atan2(self.y - sy, self.x - sx)


class LandmarkStore:
  def __init__(self, gate_radius: float = GATE_RADIUS) -> None:
    self.gate_radius = gate_radius
    self.landmarks: list[Landmark] = []

  def add_sighting(self, x: float, y: float, z: float,
                   seen_from: tuple[float, float]) -> Landmark:
    """Record one detection: merge into the nearest landmark within the gate
    (2D distance), or start a new landmark if none is close enough.
    seen_from: the robot's (x, y) at detection time."""
    nearest = None
    min_dist = self.gate_radius
    for landmark in self.landmarks:
      dist = math.hypot(landmark.x - x, landmark.y - y)
      if dist < min_dist:
        min_dist = dist
        nearest = landmark

    if nearest is None:
      nearest = Landmark(x, y, z, seen_from)
      self.landmarks.append(nearest)
    else:
      nearest.merge(x, y, z, seen_from)
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
