import math

from pluggybot.mapping.landmarks import LandmarkStore


def test_single_sighting_is_not_confirmed():
  store = LandmarkStore()
  store.add_sighting(1.0, 2.0, 0.24)
  assert store.confirmed(min_sightings=3) == []
  assert len(store.landmarks) == 1        # remembered, just not trusted yet


def test_repeat_sightings_confirm_one_landmark():
  store = LandmarkStore()
  for _ in range(3):
    store.add_sighting(1.0, 2.0, 0.24)
  confirmed = store.confirmed(min_sightings=3)
  assert len(confirmed) == 1
  assert confirmed[0].n_sightings == 3


def test_drifted_resighting_merges_not_duplicates():
  """The original worry: PluggyBot sees an outlet, wanders, and re-sees it
  after dead reckoning has drifted a bit. Within the gate that must merge."""
  store = LandmarkStore()
  store.add_sighting(1.0, 2.0, 0.24)
  store.add_sighting(1.15, 2.1, 0.26)     # ~0.18 m of drift: same outlet
  assert len(store.landmarks) == 1
  assert store.landmarks[0].n_sightings == 2


def test_distant_sighting_creates_new_landmark():
  store = LandmarkStore()
  store.add_sighting(1.0, 2.0, 0.24)
  store.add_sighting(2.0, 2.0, 0.24)      # 1 m away: a different outlet
  assert len(store.landmarks) == 2


def test_merged_position_is_the_mean_of_sightings():
  """The running average must converge to the mean, so one bad first
  sighting (typically the farthest, noisiest view) gets corrected."""
  store = LandmarkStore()
  sightings = [(1.00, 2.00, 0.20), (1.20, 2.10, 0.26), (1.10, 1.90, 0.23)]
  for s in sightings:
    store.add_sighting(*s)
  lm = store.landmarks[0]
  assert math.isclose(lm.x, sum(s[0] for s in sightings) / 3, abs_tol=1e-9)
  assert math.isclose(lm.y, sum(s[1] for s in sightings) / 3, abs_tol=1e-9)
  assert math.isclose(lm.z, sum(s[2] for s in sightings) / 3, abs_tol=1e-9)


def test_gate_ignores_z():
  """z is the noisiest coordinate (pixel row + depth), so it must not be able
  to split one outlet into two landmarks. Gate distance is 2D by design."""
  store = LandmarkStore()
  store.add_sighting(1.0, 2.0, 0.10)
  store.add_sighting(1.0, 2.0, 0.45)      # wild z disagreement, same (x, y)
  assert len(store.landmarks) == 1
  assert math.isclose(store.landmarks[0].z, 0.275)   # averaged, not fought over


def test_nearest_confirmed_picks_closest_and_needs_confirmation():
  store = LandmarkStore()
  for _ in range(3):
    store.add_sighting(0.0, 0.0, 0.24)    # confirmed, far from query
  store.add_sighting(4.9, 5.0, 0.24)      # near the query but seen only once
  lm = store.nearest_confirmed(5.0, 5.0)
  assert lm is not None
  assert (lm.x, lm.y) == (0.0, 0.0)       # trust beats proximity


def test_nearest_confirmed_empty_store():
  assert LandmarkStore().nearest_confirmed(0.0, 0.0) is None
