# PluggyBot — notes for Claude

Simulated self-charging robot in MuJoCo. Before doing anything, read:
- `docs/PluggyPlan.md` — goals, milestone status, architecture
- `docs/SimNotes.md` — hard-won simulation lessons; read BEFORE touching `models/` or contact/actuator params
- `docs/Parts.md` — locked hardware decisions and the sim parameters they feed

## Working style

- **Claude builds by default, and explains afterward.** Ben is learning
  robotics + Python (strong webdev background), so the explanation is the
  deliverable as much as the code: what was built, why it is shaped that way,
  what the measurements showed, and what surprised us. Do not offer to hand
  work over — Ben will claim a piece himself when he has the time and focus,
  and until he says so the default is Claude doing it.
- Explain in prose, at the level of "a teammate catching up": name the
  concepts (running average, pinhole projection, convex decomposition) rather
  than assuming them, and say what a number means, not just what it is.
- Ben may still claim ML training runs — he enjoys them — but Claude runs
  them by default now like anything else.
- Verify physics claims empirically (headless probes, filmstrip renders via
  offscreen Renderer) rather than by reasoning alone; it has won every time.
  When a result looks good, try to break it before believing it — the MSAA
  label bug, the stale-dataset contamination, and the spin-collision gap were
  all found this way, and two of them were hiding behind green metrics.
- Every debugged failure becomes a pytest assertion, and the assertion must be
  shown to fail without the fix — a regression test that cannot fail is décor.

## Commands

- Tests: `MUJOCO_GL=egl uv run pytest -q`; lint: `uv run ruff check src/ scripts/ tests/`
- Demos: `scripts/teleop.py`, `scripts/map_teleop.py`, `scripts/explore.py [--headless]`
  (milestone-4 mapping demo — kept as the minimal repro; `lifecycle.py` is the
  full mission), `scripts/spot_outlets.py` (detector → landmarks),
  `scripts/lifecycle.py [--headless] [--explore-budget N]` (explore → charge),
  `scripts/schuko_spike.py` (docking tolerance sweep)
- Dataset (deterministic; `datasets/` is gitignored, and the generator wipes it
  first — regenerating into a dirty dir once contaminated 195 labels):
  `MUJOCO_GL=egl uv run python scripts/generate_outlet_dataset.py --count 1200`
- Train: `uv run yolo detect train data=datasets/outlets/dataset.yaml model=yolo11n.pt epochs=50 imgsz=640`
  (torch is pinned to the cu128 index in `pyproject.toml`: the driver here is
  CUDA 12.8, and PyPI's default cu130 build silently falls back to CPU)

## Conventions

- 2-space Python indent; type hints in `src/`, loose in tests/scripts.
- `models/world.xml` = bare world for physics tests; `playground.xml` /
  `room_1.xml` add scenery for humans and mapping. Never put scenery in the
  test world.
- Grid code: cells are `(ix, iy)` tuples at APIs; numpy arrays index `[iy, ix]`.
- Odometry tracks the axle midpoint; `qpos` tracks the body origin 8 cm ahead.
