"""A* shortest path over a boolean traversable grid (4-connected).

A* is Dijkstra's cheapest-first expansion plus a sense of direction: nodes
are ranked by g + h, where g is the cost already paid from the start and h
is an optimistic estimate of the cost remaining (here Manhattan distance,
which never overestimates on a 4-connected grid — that "admissibility" is
what preserves optimality).
"""

import heapq


def astar(
  traversable,
  start: tuple[int, int],
  goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
  """Shortest 4-connected path from start to goal, or None if unreachable.

  traversable: 2-D bool array indexed [iy, ix]. start/goal: (ix, iy) cells.
  Returns the path as [(ix, iy), ...] including both endpoints.

  The start cell is always treated as traversable: the robot is physically
  standing on it, whatever the inflated map claims (it may sit inside an
  obstacle's inflation ring after a tight approach and must be able to leave).
  """
  rows, cols = traversable.shape
  gx, gy = goal
  if not (0 <= gx < cols and 0 <= gy < rows) or not traversable[gy, gx]:
    return None
  if start == goal:
    return [start]

  def h(cell):
    return abs(cell[0] - gx) + abs(cell[1] - gy)

  open_heap = [(h(start), 0, start)]     # (f = g + h, g, cell)
  came_from: dict[tuple[int, int], tuple[int, int]] = {}
  best_g = {start: 0}

  while open_heap:
    _, g, cell = heapq.heappop(open_heap)
    if cell == goal:
      path = [cell]
      while cell in came_from:
        cell = came_from[cell]
        path.append(cell)
      return path[::-1]
    if g > best_g.get(cell, g):
      continue                           # stale heap entry; a cheaper route won
    x, y = cell
    for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
      nx, ny = nxt
      if not (0 <= nx < cols and 0 <= ny < rows) or not traversable[ny, nx]:
        continue
      ng = g + 1
      if ng < best_g.get(nxt, float("inf")):
        best_g[nxt] = ng
        came_from[nxt] = cell
        heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))

  return None                            # open set exhausted: no route exists
