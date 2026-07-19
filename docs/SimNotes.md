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

### Tires need compliance: `solref="0.05 1"` on wheel geoms
With default contact stiffness the wheels are rigid hoops and the robot micro-hops at cruise (measurably airborne ~20 % of the time). Each landing delivers a friction impulse whose pitch moment scales with **wheel radius** — at r = 0.035 the kicks were sub-threshold; at r = 0.045 (the real 90 mm wheels) they resonated with the chassis pitch mode and flipped the robot onto its tail at cruise. Softer, damped tire contact (`solref="0.05 1"` ≈ rubber squish) breaks the resonance. Note: `"0.02 1"` is the *default* — setting that is a no-op (ask me how I know).

### Mass goes over the drive axle
A 120 g head cantilevered 16 cm ahead of the axle destabilized launches (wheelie → riding the rear chassis corner at ~30°); the identical mass directly above the axle is benign. Established empirically by bisection: it's mass × position, not mass. Same principle as Roomba/TurtleBot battery placement. Corollary discovered later: **height is also a lever** — the head at z = 0.135 destabilizes cruise even with all other fixes in place (and no amount of command ramping saves it); at z = 0.095 everything passes. A camera mast requires a base redesign (wider, heavier — the Hello Robot Stretch proportions), not just a taller neck. Deferred to the milestone-6 arm/lift work.

### Motor sizing: torque-to-weight matters
The original 30:1/1.4 N·m spec on a 1.1 kg robot demands ~40 N of thrust per wheel against ~3 N of available traction — permanent wheelspin at launch, and enough reaction torque to wheelie. Real robots this size are torque-limited by traction, not by motor. When behavior looks violent, check whether the actuator could physically exist in that weight class.

## Known artifacts (accepted for now)

- **In-place turns "walk"** — the turn center sits near the body origin rather than the axle midpoint (axle midpoint orbits ~10–17 cm during a spin). Cause: cylinder wheels contact the ground at their two rim edges, which scrub during yaw. Remedy when it matters (odometry milestone): ellipsoid collision geoms for the wheels (`type="ellipsoid" size="r w r"`), which cut the walk ~3×. The regression test intentionally asserts on body-origin drift, which stays small.
- **Micro-hopping** — brief airborne moments at cruise persist even in stable configs. Harmless since the tire-compliance fix decoupled it from pitch; revisit if odometry noise looks unphysical.

## Test & world hygiene

- **`models/world.xml` is bare** (floor + light + robot): physics tests run here. **`models/playground.xml`** adds scenery via `<include file="world.xml"/>`: teleop and camera scripts run here. Scenery once parked a box in the drive-test lane; separately, copying the floor into the playground doubled every wheel contact (the include already provides it — a "repeated name" MJCF error means *delete* the duplicate, not rename it).
- Every debugged failure becomes a pytest assertion (rests level, drives straight, no wheelie, turns in place, stereo parallax exists). The suite has already caught two real regressions.

## Conventions & gotchas

- MJCF `size` values are **half**-extents; `pos` is relative to the parent body frame.
- Cameras look down their own **−z**, image-up is +y. Forward-looking camera on a +x-facing body: `xyaxes="0 -1 0 0 0 1"`.
- Pitch extraction from the freejoint quaternion `(w,x,y,z)`: `asin(2·(w·y − z·x))`, **positive = nose down** with x-forward/z-up.
- A body with no joint is welded to its parent; a `<geom>` with `contype="0" conaffinity="0"` is visual-only (wheel spokes, future pretty meshes).
- Velocity actuators: `ctrlrange` = ± no-load speed (rad/s), `forcerange` = ± stall torque (N·m), both straight off the motor datasheet; `kv` is a tuning gain, not a datasheet number.

## Debugging workflow that worked

1. Reproduce headlessly with printed telemetry (pose, wheel ω, contact list, `ncon`) — vibes don't bisect.
2. **Render a filmstrip** (offscreen `Renderer`, tracking camera, 12 tiled frames) when numbers confuse — "it's standing on its tail" was invisible in scalars.
3. Bisect one variable at a time; the surprising answer (wheel *radius*) was found only after mass, torque, gain, damping, and timestep all alibied out.
4. Assert that programmatic XML patches actually applied (`assert count == 2`) — a silent no-op replacement once produced identical "before/after" results and nearly a wrong conclusion.
