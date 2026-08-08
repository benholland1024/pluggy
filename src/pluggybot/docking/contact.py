"""Contact queries shared by the scripted docking controller and the RL env.

Extracted from scripts/lifecycle.py so that DockEnv (envs/dock_env.py) and the
DOCK state judge success by literally the same code path. The charging
criterion's history is worth keeping in one place: four cleverer seat
detectors (extension windows, base-release, odometry advance, floor-contact
by name) were each defeated by a real jam mode; what survived is the
ELECTRICAL criterion — a pin touching socket-floor geometry at least 19 mm
into the recess, i.e. inside a pin channel, which is unreachable except
through a hole. It is also the sensor the physical robot will actually have
(charging voltage), and the milestone-7 battery hook.
"""


def charging_contact(model, data) -> bool:
  """True when a plug pin touches a socket-floor geom >= 19 mm deep.

  Depth matters: a mis-aligned pin bottoming on the floor's FRONT face
  (17 mm) touches the same geoms, and an earlier name-only version accepted
  exactly that jam as "plugged in" with the plug 9 mm below the holes.
  Works for any socket whose geoms follow schuko.socket_geoms_xml naming
  ("<prefix>floor_*") — room_1's generated sockets and DockEnv's alike.
  """
  pins = {model.geom("plug_pin_l").id, model.geom("plug_pin_r").id}
  for i in range(data.ncon):
    c = data.contact[i]
    pin = pins & {c.geom1, c.geom2}
    if not pin:
      continue
    other = ({c.geom1, c.geom2} - pin).pop()
    if "floor_" not in (model.geom(other).name or ""):
      continue
    bid = model.geom(other).bodyid
    bid = bid[0] if hasattr(bid, "__len__") else bid
    sp, mat = data.xpos[bid], data.xmat[bid]
    nx, ny = float(mat[0]), float(mat[3])   # socket +x = outward normal
    depth = -((c.pos[0] - sp[0]) * nx + (c.pos[1] - sp[1]) * ny)
    if depth >= 0.019:
      return True
  return False


def feelers_touching(model, data) -> int:
  """How many of the two alignment prongs are in contact (0, 1 or 2)."""
  prongs = {model.geom("prong_l").id, model.geom("prong_r").id}
  return len({g for i in range(data.ncon)
              for g in (data.contact[i].geom1, data.contact[i].geom2)
              if g in prongs})
