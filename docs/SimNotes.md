# Simulation Notes

Hard-won MuJoCo lessons from building PluggyBot. Each entry exists because something broke without it. Keep this file growing — future-you (and the Onshape import) will thank present-you.

## Physics modeling rules

### Wheel joints need `armature`
A bare 50 g wheel has ~3×10⁻⁵ kg·m² of rotational inertia; a motor torque applied to that overshoots any velocity target within a single 2 ms timestep, and the servo then chatters at the timestep frequency — the robot "vibrates and bounces" instead of driving. `armature` models the motor rotor's inertia *reflected through the gearbox*, which scales with gear-ratio²:

> armature ≈ rotor inertia (~5×10⁻⁶ kg·m²) × ratio² → **0.005 at 30:1, 0.012 at 50:1**

### Wheel joints need `damping` (and it's physically real)
Gearboxes eat torque: Pololu lists ~65 % efficiency for the 37D 50:1, so ~30–35 % of torque is lost to friction. `damping="0.05"` on the wheel joints models this — and it is also *load-bearing for stability*: the velocity servo regulates **joint** velocity, which includes chassis pitch rate, creating positive feedback that pumps energy into chassis-pitch oscillation. Joint damping dissipates it. Don't exceed honest magnitudes (0.2+ would mean a gearbox that eats the entire stall torque — unphysical, and it makes the robot crawl).

### Use `integrator="implicitfast"`
MuJoCo's default explicit Euler integrator adds energy to velocity-dependent forces (velocity actuators, joint damping) — the exact ingredients of a wheeled robot. `<option integrator="implicitfast"/>` is standard practice for wheeled/actuated models (most Menagerie models use it).

### THE caster lesson: MuJoCo combines pair friction as the elementwise MAX
`friction="0.001"` on the caster never worked: when two geoms touch, the contact's friction is (at equal priority) the element-wise **maximum** of the two geoms' values — and the floor's default is 1.0. Our "frictionless" caster was a full-grip rubber ball for weeks, silently exerting ~4 N of drag and ~0.4 N·m of yaw braking. That single hidden brake caused, downstream: the cruise pitch resonance blamed on 90 mm wheels, the tail-flip, the 12.6 % straight-line odometry creep, the 2.7× in-place-turn overestimate ("turn walk"), and the apparent head-height ceiling. The correct frictionless caster (copied from Menagerie's Stretch):

```xml
<geom name="caster" type="sphere" size="0.02" ... condim="1" priority="1"/>
```

`condim="1"` = normal-force-only contact (no friction dimensions exist at all); `priority="1"` makes the caster's contact parameters win over the floor's (otherwise condim also combines as max and the floor's 3 wins). With this fixed, the earlier `solref` tire-softening became unnecessary and was removed, the head can sit at mast height (tested to z = 0.16, pitch 0.1°), and dead-reckoning odometry agrees with ground truth to ~0.2–1 % on straights, spins, and arcs.

Meta-lesson: **know the pair-combination rules** (friction → max, condim → max, solref/solimp → priority/solmix-weighted) before trusting any per-geom contact attribute.

### Mass goes over the drive axle
A 120 g head cantilevered 16 cm ahead of the axle destabilized launches (wheelie → riding the rear chassis corner at ~30°); the identical mass directly above the axle is benign. Established empirically by bisection: it's mass × position, not mass. Same principle as Roomba/TurtleBot battery placement — keep it. *Historical correction:* the follow-on "head height ceiling" (z = 0.135 unstable) was measured **before** the caster-friction bug was found; with the caster truly frictionless, head heights to at least z = 0.16 are stable at 0.1° pitch. The forward-cantilever result may also have been amplified by caster drag (it loaded the caster harder), but weight-over-drive-wheels remains sound design regardless — traction depends on it.

### Motor sizing: torque-to-weight matters
The original 30:1/1.4 N·m spec on a 1.1 kg robot demands ~40 N of thrust per wheel against ~3 N of available traction — permanent wheelspin at launch, and enough reaction torque to wheelie. Real robots this size are torque-limited by traction, not by motor. When behavior looks violent, check whether the actuator could physically exist in that weight class.

## Known artifacts (accepted for now)

- ~~In-place turns "walk"~~ — **resolved**: this was the caster friction bug (the caster pinned its end of the robot, shifting the turn center off the axle). With `condim="1" priority="1"` the turn center returns to the axle line and spin odometry matches truth to <1 %. The residual "drift" in the turn regression test is mostly geometry: the freejoint origin sits 8 cm ahead of the axle and orbits it during a spin.
- **Frame mismatch in odometry comparisons** — dead reckoning tracks the **axle midpoint**; `qpos[:2]` tracks the **body origin**, 8 cm ahead. On curved paths they trace different circles (~0.1 m apart after a half-turn). Compare truth at the axle: `(x − 0.08·cos ψ, y − 0.08·sin ψ)`.

## Test & world hygiene

- **`models/world.xml` is bare** (floor + light + robot): physics tests run here. **`models/playground.xml`** adds scenery via `<include file="world.xml"/>`: teleop and camera scripts run here. Scenery once parked a box in the drive-test lane; separately, copying the floor into the playground doubled every wheel contact (the include already provides it — a "repeated name" MJCF error means *delete* the duplicate, not rename it).
- Every debugged failure becomes a pytest assertion (rests level, drives straight, no wheelie, turns in place, stereo parallax exists). The suite has already caught two real regressions.
- **Relative-error metrics need denominators that can't vanish.** Two incidents: position-error ÷ distance blows up on an in-place spin (distance ≈ 0), and heading-error ÷ net-rotation blows up on an S-curve (segments cancel, net ≈ 0 — a 0.17° error read as "142 %"). Normalize by *path traveled* (distance rolled, rotation swept), or assert absolute error.
- Odometry comparisons: dead reckoning tracks the **axle midpoint**; `qpos` tracks the body origin 8 cm ahead. Transform truth to the axle before comparing (see tests/test_odometry.py `axle_pos`).

## Conventions & gotchas

- MJCF `size` values are **half**-extents; `pos` is relative to the parent body frame.
- Cameras look down their own **−z**, image-up is +y. Forward-looking camera on a +x-facing body: `xyaxes="0 -1 0 0 0 1"`.
- Pitch extraction from the freejoint quaternion `(w,x,y,z)`: `asin(2·(w·y − z·x))`, **positive = nose down** with x-forward/z-up.
- A body with no joint is welded to its parent; a `<geom>` with `contype="0" conaffinity="0"` is visual-only (wheel spokes, future pretty meshes).
- Velocity actuators: `ctrlrange` = ± no-load speed (rad/s), `forcerange` = ± stall torque (N·m), both straight off the motor datasheet; `kv` is a tuning gain, not a datasheet number.

## Exploration lessons (milestone 4)

- **The nearest-frontier deadlock.** A forward camera cannot observe the cells beside its own wheels, so the nearest frontier is always the unscanned sliver just outside the FOV — the robot "arrives" instantly, stops, and the frontier never dissolves. First verification run: 420 sim-seconds, zero movement. Fixes: ignore frontiers closer than ~0.3 m, and do a 360° look-around spin at startup and whenever no distant frontier is reachable.
- **Collisions corrupt the map, not just the paint job.** Grinding a wall slips the wheels → odometry counts phantom distance → the map frame slides → old walls repaint at new believed positions ("jail bar" artifacts, evidence outside the room). Prevention beats cure: obstacle inflation must exceed the chassis **half-diagonal** (0.15 m) plus margin — we use 5 cells / 0.25 m; sparse waypoints + generous arrival radii cut corners through the inflation ring (use ≤3-cell spacing, ≤0.08 m radius); and a scan-based reflex (stop + back off when anything is <0.25 m dead ahead) catches what planning misses.
- **The safety reflex must be armed in every maneuver, not just while driving.**
  `explore.py` gated it on `mode == "drive" and waypoints`, so look-around spins ran
  blind. Measured over a full run: its spins passed within **0.257 m** of the L-box —
  7 mm outside the 0.25 m threshold. Zero collisions was luck, not design. Refactoring
  into `lifecycle.py` shifted the trajectory by <1 mm/step (closed-loop driving among
  obstacles is chaotic; identical logic will not retrace an identical path), the final
  spin landed inside the margin, and the chassis ground for **503 steps**. Arming the
  reflex in all maneuvers cut that to 43, all of them *during the escape*. Caveat: a
  forward ±20° reflex fundamentally cannot protect a spin (the chassis corner sweeps
  through arcs the camera never sees) — driving it to zero needs 360° clearance
  memory, e.g. checking the robot's own cell against the inflated grid before spinning.
- **Don't re-arm a reflex that is already firing.** Re-triggering backoff on every scan
  while reversing turns a bounded 0.8 s pulse into an open-ended reverse into whatever
  is behind the robot.
- **Termination is "no *reachable* frontiers," not "no frontiers."** Unreachable slivers (pockets inside obstacles, hairline gaps) are blacklisted when A* fails; exploration ends after a look-around spin plus repeated pathless replans. Benchmark: both rooms of room_1.xml in ~80 sim-seconds, 0 chassis contacts.

## Landmark & docking-approach lessons (milestone 5→6)

### "Where I saw it from" is not "which way it faces"
The first standoff-pose estimate derived the outlet's outward normal from the mean
robot position across sightings. It sounds sound — the robot only ever sees an outlet
from the open side — but it records *where the robot happened to drive*, and a drive-by
biases it badly. Measured end to end: **31.2° off the true wall normal**, which put the
docking hand-off pose 33 cm sideways with the socket at the very edge of the camera's
66° horizontal FOV. The fix reads the normal off the occupancy grid instead
(`landmarks.wall_normal`): sum a unit vector toward every nearby known-free cell, and
since a wall blocks half the circle the sum points out of it — no line fitting, no
normal-direction ambiguity. **31.2° → 0.0°.** Seen-from survives only as the fallback
for outlets whose surroundings were never mapped.

Meta-lesson: **decompose an error before fixing it.** The 33° miss was first written off
as odometry drift. Splitting it into controller settle / odometry drift / direction
estimate showed drift was **0.02°** (the gyro fusion is excellent) and the estimator
owned essentially all of it. Guessing would have wasted the effort on the wrong subsystem.

### The camera's FOV, not the wall, bounds usable outlet height
With the eye fixed at z = 0.18 m and fovy 41°, the visible band is z ≤ 0.40 m at the
0.6 m standoff but only z ≤ 0.31 m at 0.35 m. Verified in `room_1.xml`: outlet C at
0.38 m is detected at 0.6 m (conf 0.93) and **disappears entirely at 0.35 m**. So the
approach can see a high outlet and then lose it exactly when it matters — the concrete
argument for the prismatic lift in milestone 6, and outlet C is deliberately left high
as the test case. Room walls are 1.20 m (was 0.30 m), which also puts them inside the
detector's trained wall-height range of 0.5–1.5 m.

### The generator's own val split cannot measure the detector
Training and validation images come from the same `outlet_scene.py` sampler, so the val
split only measures what the generator already thought to vary. It reported **mAP50-95
0.9938 while the detector was calling a light switch an outlet**. `scripts/eval_detector.py`
exists for this: it samples collision-free robot poses throughout `room_1.xml`, renders
what the camera would really see, and scores against segmentation ground truth. Every
real defect in this milestone was found there or by the distance sweep — none by mAP.

### A handful of poses is a smoke test, not a measurement
Three data recipes were compared on an 8-pose spot check, and a decoy false positive
moving 0.61 → 0.93 was read as a regression signal. At that sample size it is inside
run-to-run training variance. Re-run over 300 poses the ranking was unambiguous
(false positives per frame): original 0.140, +close-range 0.187, **+decoy-aimed negatives
0.127**, +distance-scaled aim jitter 0.213. Compare recipes on hundreds of samples or
don't compare them.

### One measured mistake, kept as a warning
Aiming negatives at decoys with a *distance-scaled* jitter was meant to teach the
detector about decoys cropped by the frame edge. It measured worse: the wide jitter
pushed the decoy out of shot in 22 % of negatives, diluting the hard negatives it was
meant to sharpen (decoy false positives tripled, 9 → 26). Reverted to a tight aim.
The lesson generalizes: **check what a generator change actually produced** — a contact
sheet or a pixel-coverage histogram — before spending 18 minutes training on it.

### Recall is recoverable; precision errors compound
Raising the detector threshold 0.5 → 0.7 costs 2 detections in 105 (recall 0.952 →
0.933) and removes 26 of 40 false positives that land on wall decoys. That trade is
right *for this system* because the two errors are not symmetric: the robot sees each
outlet 13+ times per run, so a miss is recovered on the next glance, but a *systematic*
false positive on a fixed decoy accumulates sightings in the same place and graduates
into a confirmed phantom landmark the robot will drive to. Tune the threshold against
what the downstream consumer does with the errors, not against an F1 score.

### The sighting threshold earns its keep
Drawing tentative landmarks on `map.png` in olive immediately exposed one: the detector
fires on `decoy_switch_w` (a light switch) about twice per run, at the right position
and height. `min_sightings=3` filters it, and it never reaches the confirmed list. Worth
remembering that the detector's precision on the val set (0.98) is not zero false
positives in the world — the confirmation count is what makes the landmark map clean.

### Budget the tolerance across the whole chain, not per-stage
`FACING_TOLERANCE` was 2° against the spike's ±3° docking budget — the settle criterion
alone spent two-thirds of the allowance before odometry or the docking controller got
any. At 0.5° the full pipeline now parks at **-0.49° yaw, 1.3 cm lateral, 60 cm out**.

## Rendering lessons (milestone 5)

### Segmentation rendering needs `offsamples="0"`
Free labels from segmentation rendering are only free if the buffer is honest. MuJoCo
applies multisample antialiasing to the segmentation image too, blending geom IDs at
object edges — so pixels appear carrying IDs of geoms that aren't there. Most land
harmlessly, but a label box spans the **min/max** of its mask, so one stray pixel 200 px
from the outlet stretches the box across half the frame. Measured over 236 positive
scenes: **12.3 % grew a second blob, 5.5 % came out grossly elongated** (a square 80 mm
Schuko plate labeled as a bar with w/h up to 7.7). The training run scored mAP50-95 0.940
*despite* ~3 % of its labels being garbage, and the val images showed the model drawing
two correct tight boxes where the truth was one absurd bar — the model was right.

```xml
<visual><quality offsamples="0"/></visual>
```

Drops stray blobs to 0.0 %. `make_labeled_sample` additionally keeps only the largest
connected blob, so a survivor gets dropped rather than silently poisoning a label.

Meta-lesson: **the renderer is not a measurement device by default.** Anything that
smooths pixels for human eyes — antialiasing, filtering, interpolation — corrupts a
buffer whose values are identifiers rather than colors.

### Dataset regeneration must clean its output directory
The train/val split is a random per-image draw, so a file whose split changes between
runs leaves its old copy alive in the other split — regenerating after the MSAA fix
still left **195 stale image/label pairs** from the buggy run mixed into training, and
the "residual" corrupt labels were all leftovers. The generator now deletes `images/`
and `labels/` before writing (guarded by a pytest). Symptom to remember: the same
basename appearing in both splits.

## Schuko contact spike (milestone-6 prep)

Standalone plug/socket rig (`docking/schuko.py`, `scripts/schuko_spike.py`) — no robot,
a compliant carrier pushes the plug with a 10 N force limit. Findings (guarded by
tests/test_schuko.py):

- **Collision geoms must be convex**, so the concave recess is *composed*: 12-box
  dodecagonal well wall, 12 tilted boxes as a 45° entry funnel, 5 floor slabs leaving
  two square pin holes, 4 boxes framing the face. Capsule pin tips double as their own
  entry chamfer.
- **Capture is set by the entry chamfer, not the recess.** First version used an 8 mm
  45° funnel and measured a flattering ±18 mm lateral / ±4° yaw. Ben's challenge —
  "real Schuko rims don't have that funnel" — was correct: with an honest 2 mm rim
  bevel the envelope is **±3 mm lateral / ±3 mm vertical / ±2° yaw**. Capture ≈ body
  clearance (0.75 mm) + chamfer; the deep recess only *guides after* capture. Don't
  widen the chamfer to make a failing controller pass — that tunes the world, not the
  robot. (Measured chamfer→tolerance: 2 mm→±3 mm, 4 mm→±6 mm, 8 mm→±18 mm; a dished
  face plate on the physical charging outlet is a legitimate *hardware* choice that
  buys margin honestly — a Parts.md decision, not a sim default.)
- **Yaw is the tight constraint** — with a diagnostic signature: every yaw/lateral jam
  stops ~19 mm short (= pin length): the pins bottom on the floor beside their holes
  before the shallow well can square the 40 mm body. Docking must get *facing* right.
- Jams are clean (stall at the force cap, no solver explosions); worst transient
  contact force ~32 N during an edge-of-envelope wedge, fine at `timestep 0.001` +
  `solref "0.005 1"`.
- **The velocity-servo chatter lesson generalizes.** A velocity actuator driving the
  90 g carrier chattered at the timestep frequency exactly like the bare wheels once
  did (kv needed for 10 N @ 2 cm/s is far too stiff for that mass). Fix: constant-force
  `<motor>` + heavy joint damping (`damping = F/v`) — implicitfast integrates damping
  implicitly, so it cannot chatter, and "10 N push, 2 cm/s free speed" is the honest
  robot semantics anyway.
- Caveats for milestone 6: carrier compliance was 150 N/m lateral / 1 N·m/rad angular
  (a guess at arm+base flex — revisit with the real arm), gravity off (the lift owns
  height), and tolerances scale with that compliance.

## Debugging workflow that worked

1. Reproduce headlessly with printed telemetry (pose, wheel ω, contact list, `ncon`) — vibes don't bisect.
2. **Render a filmstrip** (offscreen `Renderer`, tracking camera, 12 tiled frames) when numbers confuse — "it's standing on its tail" was invisible in scalars.
3. Bisect one variable at a time, and assert that programmatic XML patches actually applied (`assert count == 2`) — a silent no-op replacement once produced identical "before/after" results and nearly a wrong conclusion.
4. When symptom-fixes keep trading one failure for another, stop tuning and **measure the force balance directly**: `mj_contactForce` per contact, summed as torque contributions about the COM, named the caster as the yaw-brake in one run — after days of plausible-but-wrong theories (integrator energy, tire stiffness, wheel radius resonance, servo feedback).
5. **Consult reference models** (MuJoCo Menagerie — Stretch for diff-drive). The caster `condim="1" priority="1"` idiom was sitting in their XML all along; professionally-tuned models encode solved problems.
