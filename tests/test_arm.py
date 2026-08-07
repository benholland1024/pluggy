"""Guards for the docking arm (mast / lift / telescoping arm / RCC plug).

Why the transit test checks GEOMETRY, not contacts: the head body has no
joint, so MuJoCo welds it into the chassis, and contact filtering treats a
welded group as one body — the carriage, as that group's child, is
parent-filtered against ALL of it. The original long prongs swept straight
through the head mount during lift transit and the contact pipeline reported
nothing (measured: overlapping AABBs, ncon contribution zero). On hardware
that is a collision; in MuJoCo it is silence. Adjacent-link clearance must be
asserted from geometry.
"""

import mujoco
import pytest

# (moving geom, static geom) pairs that MuJoCo's weld/parent filtering will
# never report contacts for, checked geometrically through the whole envelope.
# Deliberately absent: carriage_block x mast / lift_motor — the carriage
# wraps its rail like a sleeve, so co-occupancy there is design intent, not a
# crash. Everything listed here must genuinely stay clear.
CLEARANCE_PAIRS = [
  (mov, ref)
  for mov in ("prong_l", "prong_r", "arm_tube", "plug_body")
  for ref in ("head_mount", "mast", "lift_motor", "chassis", "battery")
] + [
  ("carriage_block", "head_mount"),
  ("carriage_block", "chassis"),
  ("carriage_block", "battery"),
]


@pytest.fixture
def arm_data(world_model):
  d = mujoco.MjData(world_model)
  for _ in range(1000):
    mujoco.mj_step(world_model, d)
  return d


def _self_contacts(model, data):
  hits = set()
  for i in range(data.ncon):
    c = data.contact[i]
    a = model.geom(c.geom1).name or "?"
    b = model.geom(c.geom2).name or "?"
    if "floor" not in (a, b):
      hits.add((a, b))
  return hits


def _z_half(model, name):
  """Vertical half-extent: boxes store it in size[2]; the plug cylinder lies
  along x, so its vertical extent is its radius (size[0])."""
  g = model.geom(name)
  return float(g.size[0] if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER else g.size[2])


def _aabb_overlap(model, data, a, b):
  """Axis-aligned overlap test — valid here because the robot sits level and
  none of these geoms are rotated. Compliance wobble is sub-mm at rest."""
  ga, gb = model.geom(a), model.geom(b)
  pa, pb = data.geom_xpos[ga.id], data.geom_xpos[gb.id]
  return all(abs(pa[k] - pb[k]) < float(ga.size[k] + gb.size[k]) for k in range(3))


def test_parked_arm_is_stowed(world_model, arm_data):
  """Parked, the plug must hide behind the bumper (+0.12) and the whole arm
  must sit below the scanner's z=0.18 center row, or mapping sees phantom
  obstacles bolted to the robot."""
  tip = arm_data.site_xpos[world_model.site("plug_tip").id]
  assert tip[0] - arm_data.qpos[0] < 0.12, "plug tip pokes past the bumper"
  for g in ("arm_tube", "plug_body", "prong_l", "prong_r", "carriage_block"):
    gid = world_model.geom(g).id
    top = arm_data.geom_xpos[gid][2] + _z_half(world_model, g)
    assert top < 0.18, f"{g} reaches the scan row while parked (top {top:.3f})"


def test_lift_reaches_every_room1_outlet(world_model, arm_data):
  """Outlets sit at 0.26 / 0.30 / 0.38 m; the lift must put the plug axis on
  each (plug axis = 0.145 + lift qpos)."""
  lift = world_model.actuator("lift").id
  face = world_model.site("plug_face").id
  for target_z in (0.26, 0.30, 0.38):
    arm_data.ctrl[lift] = target_z - 0.145
    for _ in range(1500):
      mujoco.mj_step(world_model, arm_data)
    err = abs(arm_data.site_xpos[face][2] - target_z)
    assert err < 0.008, f"plug axis missed {target_z} by {err * 1000:.1f} mm"


def test_full_envelope_transit_has_no_self_collision(world_model, arm_data):
  """Sweep the lift bottom-to-top, then extend the arm at the top, asserting
  geometric clearance at every pose. This catches what contacts cannot (see
  module docstring): the original long prongs passed through the head mount
  with the contact pipeline reporting nothing."""
  lift = world_model.actuator("lift").id
  arm = world_model.actuator("arm").id
  hits = set()
  for step in range(4000):
    arm_data.ctrl[lift] = 0.31 * step / 4000
    mujoco.mj_step(world_model, arm_data)
    if step % 10 == 0:
      hits |= {p for p in CLEARANCE_PAIRS if _aabb_overlap(world_model, arm_data, *p)}
    hits |= _self_contacts(world_model, arm_data)
  for step in range(2000):
    arm_data.ctrl[arm] = 0.20 * step / 2000
    mujoco.mj_step(world_model, arm_data)
    if step % 10 == 0:
      hits |= {p for p in CLEARANCE_PAIRS if _aabb_overlap(world_model, arm_data, *p)}
    hits |= _self_contacts(world_model, arm_data)
  assert not hits, f"self-interpenetration during transit: {sorted(hits)}"


def test_lift_holds_under_gravity(world_model, arm_data):
  """A lead screw holds position; the servo model must too (droop stays
  small), or every height calibration drifts with payload."""
  lift_j = world_model.joint("lift_joint").qposadr[0]
  arm_data.ctrl[world_model.actuator("lift").id] = 0.31
  for _ in range(2500):
    mujoco.mj_step(world_model, arm_data)
  droop = abs(arm_data.qpos[lift_j] - 0.31)
  assert droop < 0.006, f"lift droops {droop * 1000:.1f} mm"


def test_rcc_wrist_recenters(world_model, arm_data):
  """The compliant wrist must spring back to center when released — a wrist
  that takes a set would bias every subsequent insertion."""
  jid = world_model.joint("plug_lat_y").qposadr[0]
  arm_data.qpos[jid] = 0.015
  mujoco.mj_forward(world_model, arm_data)
  for _ in range(1500):
    mujoco.mj_step(world_model, arm_data)
  assert abs(arm_data.qpos[jid]) < 0.001


def test_arm_mass_is_counterbalanced(world_model, arm_data):
  """The arm hangs to the right of centerline; the battery offsets it. Un-
  balanced, the robot veered 0.26 m right over 4 m of open-loop driving —
  drives-straight in test_model.py guards the behavior, this guards the
  cause. Tolerance is 10 mm, not 0: the battery offset was tuned to null the
  MEASURED veer (+4 mm over 4.2 m), and that optimum leaves ~7 mm of static
  CoM offset because load-dependent wheel slip is not linear in CoM."""
  com_y = arm_data.subtree_com[world_model.body("pluggybot").id][1]
  assert abs(com_y) < 0.010, f"CoM {com_y * 1000:+.1f} mm off centerline"
