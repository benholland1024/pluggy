# PluggyBot — notes for Claude

Simulated self-charging robot in MuJoCo. Before doing anything, read:
- `docs/PluggyPlan.md` — goals, milestone status, architecture
- `docs/SimNotes.md` — hard-won simulation lessons; read BEFORE touching `models/` or contact/actuator params
- `docs/Parts.md` — locked hardware decisions and the sim parameters they feed

## Working style

- Ben is learning robotics + Python (strong webdev background): default to
  walkthroughs and explanations and offer to let him write the code himself. 
  Due to time constraints, he will ask Claude to build something directly, 
  so build it and explain how it works afterward.
- Ben runs ML training himself — that's the learning experience he wants.
  Prepare data/plumbing; leave training runs, curve-reading, tuning to him.
- Verify physics claims empirically (headless probes, filmstrip renders via
  offscreen Renderer) rather than by reasoning alone; it has won every time.
- Every debugged failure becomes a pytest assertion.

## Commands

- Tests: `uv run pytest -q` (headless rendering needs `MUJOCO_GL=egl`)
- Demos: `scripts/teleop.py`, `scripts/map_teleop.py`, `scripts/explore.py [--headless]`
- Dataset (deterministic; `datasets/` is intentionally gitignored):
  `MUJOCO_GL=egl uv run python scripts/generate_outlet_dataset.py --count 1200`

## Conventions

- 2-space Python indent; type hints in `src/`, loose in tests/scripts.
- `models/world.xml` = bare world for physics tests; `playground.xml` /
  `room_1.xml` add scenery for humans and mapping. Never put scenery in the
  test world.
- Grid code: cells are `(ix, iy)` tuples at APIs; numpy arrays index `[iy, ix]`.
- Odometry tracks the axle midpoint; `qpos` tracks the body origin 8 cm ahead.
