# PluggyBot 🔌

**A simulated self-charging robot that explores, maps, and plugs itself in.**

PluggyBot is a personal robotics project built in physics simulation ([MuJoCo](https://mujoco.org/)), with the long-term goal of a design faithful enough to real hardware components that it could eventually be built physically.

The core idea: a small wheeled robot with stereo vision that explores its environment to build and maintain a spatial map, visually recognizes wall outlets, and - when its (simulated) battery runs low - navigates to an outlet and docks itself using a rigid, plug-tipped arm.

## Design Philosophy

- **Simulation-first, hardware-honest.** Everything runs in MuJoCo, but component parameters (motor torque curves, camera baseline and FOV, masses) are modeled on real, purchasable parts, keeping an eventual sim-to-real transfer plausible.
- **Rigid plug, not a cable.** Manipulating a deformable wire plug is one of the hardest problems in robotics. PluggyBot sidesteps it: the plug is fixed to the end of a rigid arm. Docking is still a genuinely hard contact-rich alignment task - but a tractable one.
- **Decompose, don't end-to-end.** Rather than one giant RL policy from pixels to behavior, each capability uses the cheapest adequate technique: supervised learning where labels are free (simulation gives them away), classical robotics where the problem is solved, and RL where it actually earns its keep.

## Core Goals

1. **Mobility** - a differential-drive wheeled base.
2. **Binocular vision** - two cameras providing stereo depth perception.
3. **Spatial memory & exploration** - an internal map of the environment, plus a drive to explore: validating what it remembers and expanding into the unknown (active SLAM / frontier exploration).
4. **Outlet recognition** - visually detecting wall outlets with a CNN trained on domain-randomized synthetic images from the simulator.
5. **Self-docking** - plugging into a detected outlet using the rigid arm.
6. **Battery-driven behavior** - a simulated battery that drains with motor use; when low, PluggyBot seeks an outlet and charges. This closes the loop that gives the project its name.

## Stretch Goals

- **Recognizing beings** - visually detecting and distinguishing humans (or other creatures), and a drive to do so.
- **Playing a physical game** - e.g. checkers. A major manipulation and perception undertaking in its own right; explicitly out of scope until the core loop works.

## Architecture

| Capability | Approach |
|---|---|
| Outlet detection | Supervised CNN, fine-tuned on synthetic labeled images rendered from the sim with domain randomization |
| Depth perception | Ground-truth sim depth first; swap in real stereo matching (e.g. OpenCV SGBM or a learned stereo net) later |
| Odometry | Classical dead reckoning from wheel encoders + IMU (EKF fusion); learned regression demoted to a later experiment |
| Mapping & exploration | Occupancy grid + frontier-based exploration (classical baseline); learned/curiosity-driven exploration as a later experiment |
| Docking | RL policy (or scripted visual servoing baseline) over relative outlet pose + arm state; contact-rich, dense-rewardable. Arm architecture inspired by Hello Robot Stretch: the base owns x/yaw, a prismatic lift owns z, the arm owns reach — three nearly-independent 1D alignment problems. Caveat from sim experience: a mast raises the center of mass and pitch inertia, so the lift design must come with a wider/heavier base (see SimNotes.md) |
| Behavior arbitration | Finite-state machine over the modules, driven by battery level and map state |

## Milestones

1. ✅ Teleoperable differential-drive base in MuJoCo *(July 2026)*
2. ✅ Stereo camera pair rendering from the robot *(July 2026)*
3. ✅ Classical odometry — dead reckoning from wheel angles, verified <2 % against ground truth on straights, spins, arcs, and S-curves *(July 2026; IMU/EKF fusion deferred until drift actually hurts)*
4. Occupancy mapping + frontier exploration
5. Outlet detector trained on synthetic data
6. Docking controller (scripted baseline → RL)
7. Battery model + state machine tying the full loop together: explore → detect → approach → dock → charge

Each milestone is independently runnable and demoable.

**Parallel de-risk track** (can start anytime): a standalone Schuko plug/socket contact prototype — no robot, just a scripted insertion in its own MJCF — to validate contact modeling before milestone 6 depends on it. The Schuko recess is a natural alignment funnel; find out early how much of one.

**Later experiments** (off the critical path): learned odometry (competing against the classical baseline), curiosity-driven exploration.

## Tooling

- **Simulation:** MuJoCo (MJCF models authored directly in XML during prototyping)
- **CAD (later phase):** Onshape, exported to URDF/MJCF via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) once the design stabilizes
- **Learning:** PyTorch, Gymnasium, Stable-Baselines3 (RL); Ultralytics/torchvision (detection)
- **Classical vision & robotics:** OpenCV, NumPy

## Repository Layout (planned)

```
pluggy/
├── models/            # MJCF: pluggybot.xml, world.xml (bare, used by tests), playground.xml (scenery)
├── src/pluggybot/     # perception/, odometry/, mapping/, docking/, behavior/
├── scripts/           # teleop.py, view.py, stereo_snapshot.py
├── tests/             # physics + camera regression tests (pytest)
└── docs/              # this plan, Parts.md, SimNotes.md
```

(`envs/` for Gymnasium wrappers gets added when RL work starts. The world/playground split is deliberate: tests run in the bare world, humans drive in the playground — scenery in the test lane once broke the drive test.)

## Status

🚧 Milestones 1–3 complete (July 2026): teleoperable diff-drive base with a physics regression suite, stereo pair rendering with a parallax test, and classical dead-reckoning odometry (`src/pluggybot/odometry/`) verified to <2 % of motion against simulator ground truth. Hardware is anchored to real EU-purchasable parts in [Parts.md](Parts.md) (50:1 gearmotors w/ encoders, 90 mm wheels, 2× Pi Camera Module 3); hard-won simulation lessons live in [SimNotes.md](SimNotes.md). Next: occupancy mapping + frontier exploration (milestone 4) and the plug/socket contact spike.