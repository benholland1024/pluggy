import pytest 

from pluggybot.mapping.occupancy_grid import OccupancyGrid

@pytest.fixture
def grid():
  return OccupancyGrid(x_min=-2, y_min=-2, x_max=6, y_max=6, resolution=0.05)


def test_grid_coordinate_mapping(grid):
  """Ensure that coordinates in the world map correctly to the occupancy map, and vice versa"""

  # Basic test
  test_coords_1 = grid.cell_to_world(*grid.world_to_cell(1.02, 0.99))
  assert abs(test_coords_1[0] - 1.02) < (grid.resolution/2)
  assert abs(test_coords_1[1] - 0.99) < (grid.resolution/2)

  # Test corner pin
  corner_pin_coords = grid.world_to_cell(-2,-2)
  assert corner_pin_coords[0] == 0 and corner_pin_coords[1] == 0

  # Asymmetric transpose
  test_coord_3 = grid.world_to_cell(5.0, -1.5)
  assert test_coord_3[0] == 140 and test_coord_3[1] == 10

# def test_empty_grid(grid):
  
# def test_one_ray(grid):

# def test_max_range_ray_no_flag(grid):

# def test_frames_compose(grid):

# def test_evidence_saturation(grid):

