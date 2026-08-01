import numpy as np

from pluggybot.mapping.astar import astar


def open_grid(rows=10, cols=10):
  return np.ones((rows, cols), dtype=bool)


def test_straight_line_is_optimal():
  t = open_grid(5, 10)
  path = astar(t, (0, 2), (9, 2))
  assert path is not None
  assert path[0] == (0, 2) and path[-1] == (9, 2)
  assert len(path) == 10                 # 9 steps: Manhattan-optimal, no wandering


def test_path_goes_through_the_door():
  t = open_grid()
  t[:, 5] = False                        # wall across column ix=5...
  t[4, 5] = True                         # ...with a door at iy=4
  path = astar(t, (1, 1), (8, 8))
  assert path is not None
  assert (5, 4) in path                  # the only way through


def test_no_path_returns_none():
  t = open_grid()
  t[:, 5] = False                        # sealed wall
  assert astar(t, (1, 1), (8, 8)) is None


def test_untraversable_goal_returns_none():
  t = open_grid(5, 5)
  t[2, 2] = False
  assert astar(t, (0, 0), (2, 2)) is None


def test_start_equals_goal():
  t = open_grid(5, 5)
  assert astar(t, (3, 3), (3, 3)) == [(3, 3)]


def test_start_cell_is_always_traversable():
  # The robot may sit inside an obstacle's inflation ring; it must be able to leave.
  t = open_grid(5, 5)
  t[0, 0] = False
  path = astar(t, (0, 0), (4, 0))
  assert path is not None and path[0] == (0, 0)
