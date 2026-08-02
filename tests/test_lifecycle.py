"""Guards for the mission state machine in scripts/lifecycle.py.

Loaded by path, like tests/test_dataset_generation.py, since scripts/ is not
part of the installed package.
"""

import importlib.util
import math
from pathlib import Path

import mujoco
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "lifecycle.py"

# Nose-on to L-box-north, whose near face is x=3.5 (body at x=4, half-len 0.5).
# The camera sits 0.05 m behind the body origin, so it reads 3.55 - x metres of
# range; x=3.34 gives 0.21 m (inside the 0.25 m reflex threshold) while the
# chassis front edge stops at 3.46, clear of the box.
NOSE_ON_TO_LBOX = (3.34, 1.0, 0.0)


def _load():
  spec = importlib.util.spec_from_file_location("lifecycle", _SCRIPT)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


@pytest.fixture(scope="module")
def lifecycle_module():
  return _load()


@pytest.fixture
def sim(lifecycle_module):
  # weights=None -> no detector; these tests are about navigation safety.
  return lifecycle_module.Lifecycle(
    headless=True, max_sim_time=10.0, weights=None, explore_budget=10.0)


def _place(sim, x, y, yaw):
  mujoco.mj_resetData(sim.model, sim.data)
  sim.data.qpos[0:3] = [x, y, 0.045]
  sim.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(sim.model, sim.data)


def test_reflex_arms_while_spinning(sim):
  """The reflex must fire in every maneuver, not just while following
  waypoints. explore.py armed it only when waypoints existed; measured, its
  look-around spins passed within 0.257 m of the L-box -- 7 mm outside the
  0.25 m threshold. A slightly different trajectory ground the chassis for
  503 steps. Spinning is not a reason to stop watching where you are."""
  _place(sim, *NOSE_ON_TO_LBOX)
  sim.mode = "spin"
  sim.waypoints = []                  # the condition that used to disarm it
  sim.perceive()
  assert sim.backoff_until > 0.0, "reflex did not arm during a spin"


def test_reflex_ignores_open_space(sim):
  """...and must not fire when nothing is close, or the robot never moves."""
  _place(sim, 1.0, 3.0, math.pi / 2)  # facing north up an empty corridor
  sim.mode = "spin"
  sim.waypoints = []
  sim.perceive()
  assert sim.backoff_until == 0.0


def test_backoff_is_a_bounded_pulse(sim):
  """Backoff must not re-arm while already backing off: an open-ended
  reverse would just drive into whatever is behind the robot."""
  _place(sim, *NOSE_ON_TO_LBOX)
  sim.waypoints = []
  sim.perceive()
  first = sim.backoff_until
  assert first > 0.0
  sim.step_count = 0                  # force another scan on the next perceive
  sim.perceive()
  assert sim.backoff_until == first, "reflex re-armed mid-backoff"


def test_leaving_explore_without_outlets_finishes(sim):
  """No confirmed outlet means there is nowhere to charge: the mission ends
  rather than entering GO_CHARGE with target=None (which would crash)."""
  sim.explore()                       # nothing seen yet
  sim.leave_explore("battery-low")
  assert sim.state == "DONE"
  assert sim.target is None


def test_leaving_explore_with_an_outlet_goes_to_charge(sim):
  for _ in range(3):                  # three sightings = confirmed
    sim.landmarks.add_sighting(-1.99, 1.0, 0.24, seen_from=(-1.2, 1.0))
  sim.leave_explore("battery-low")
  assert sim.state == "GO_CHARGE"
  assert sim.target is not None
  sx, sy, _ = sim.target.standoff()
  assert sx > -1.99, "standoff must sit on the room side of the outlet"
