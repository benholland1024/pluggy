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

## Debugging workflow that worked

1. Reproduce headlessly with printed telemetry (pose, wheel ω, contact list, `ncon`) — vibes don't bisect.
2. **Render a filmstrip** (offscreen `Renderer`, tracking camera, 12 tiled frames) when numbers confuse — "it's standing on its tail" was invisible in scalars.
3. Bisect one variable at a time, and assert that programmatic XML patches actually applied (`assert count == 2`) — a silent no-op replacement once produced identical "before/after" results and nearly a wrong conclusion.
4. When symptom-fixes keep trading one failure for another, stop tuning and **measure the force balance directly**: `mj_contactForce` per contact, summed as torque contributions about the COM, named the caster as the yaw-brake in one run — after days of plausible-but-wrong theories (integrator energy, tire stiffness, wheel radius resonance, servo feedback).
5. **Consult reference models** (MuJoCo Menagerie — Stretch for diff-drive). The caster `condim="1" priority="1"` idiom was sitting in their XML all along; professionally-tuned models encode solved problems.
