 

# Parts List

This document provides specs + model info for parts used in this project. 

> **Sourcing note:** Pololu (US) parts are stocked by German/EU distributors — mainly [Eckstein-shop.de](https://eckstein-shop.de/Pololu_EN), plus BerryBase, EXP-Tech, Welectron, Botland and TME.eu — so no US import is needed. All prices below are **approximate, incl. 19% VAT, as of July 2026** — re-check before ordering.

---

## Drive system

### Motors — runner-up: 30:1 (not selected)

**Pololu 30:1 Metal Gearmotor 37Dx68L mm 12V with 64 CPR Encoder (Helical Pinion)** — Pololu #4752

- Source: [Eckstein-shop.de](https://eckstein-shop.de/Pololu-301-Metal-Gearmotor-37Dx68L-mm-12V-with-64CPR-EncoderHelical-Pinion-EN) — **€84,43 each** (in stock, 7–9 working days). Datasheet: [pololu.com/product/4752](https://www.pololu.com/product/4752)

| Spec | Value | → Sim / MJCF parameter |
|---|---|---|
| Gear ratio | 30:1 | — |
| No-load speed @ 12 V | 330 rpm = **34.6 rad/s** | actuator `ctrlrange` / velocity limit (matches sim ~35 rad/s ✓) |
| Stall torque @ 12 V | 14 kg·cm = **1.37 N·m** | actuator `forcerange` (sim uses 1.4 — close enough, or update to 1.37) |
| Stall current @ 12 V | 5.5 A | motor driver sizing (not sim) |
| No-load current | 0.2 A | — |
| Encoder | 64 CPR motor shaft = **1920 CPR at output** (≈0.19°/count) | future odometry / sensor noise model |
| Mass | **200 g** | motor `geom` mass (2× = 400 g of the ~1.1 kg budget) |
| Output shaft | 16 mm long, **6 mm D-shaft** | wheel/hub compatibility (see below) |
| Dimensions | ⌀37 × 68 mm (excl. shaft) | motor geom size |

Matches the original sim numbers exactly, but passed over in favor of the 50:1's docking push force — speed is a low priority for this robot.

### Motors (2×) — ✅ CHOSEN: 50:1 (July 2026)

**Pololu 50:1 Metal Gearmotor 37Dx70L mm 12V with 64 CPR Encoder (Helical Pinion)** — Pololu #4753

- Source: [Eckstein-shop.de](https://eckstein-shop.de/Pololu-501-Metal-Gearmotor-37Dx70L-mm-12V-with-64CPR-EncoderHelical-Pinion-EN) — **€84,43 each**. Datasheet: [pololu.com/product/4753](https://www.pololu.com/product/4753)
- 50:1, **200 rpm** no-load (= 20.9 rad/s), **21 kg·cm = 2.06 N·m** stall, 5.5 A stall, 3200 CPR at output, 205 g, same 6 mm D-shaft.
- ✅ **Applied to sim (July 2026):** `ctrlrange` ±21 rad/s, `forcerange` ±2.06 N·m, wheel-joint `armature="0.012"` (reflected rotor inertia ∝ gear-ratio², up from 0.005 at 30:1) and `damping="0.05"` (the 50:1 gearbox's ~30–35 % torque losses, per Pololu's ~65 % efficiency figure). The damping term also turned out to be load-bearing for sim stability — see SimNotes.md.

### Wheels (2×)

⚠ **Finding:** Pololu's 60–70 mm wheels only fit 3 mm shafts. For the 37D's **6 mm D-shaft**, the verified Pololu path is a 90 mm (or 80 mm) wheel + universal mounting hub.

| Part | Source | Price | Notes |
|---|---|---|---|
| **Pololu Wheel 90×10 mm pair** (#1435–1439) | [Eckstein-shop.de](https://eckstein-shop.de/Pololu-Wheel-90x10mm-Pair-Red-for-Micro-Metal-Gearmotors-EN) | **€11,13 / pair** | Six M3/#4-40 mounting holes matching Pololu universal hubs. Mass: TBD (verify on datasheet) |
| **Pololu Universal Aluminum Mounting Hub, 6 mm shaft, M3 holes (2-pack)** (#1999) | [Eckstein-shop.de](https://eckstein-shop.de/PololuUniversalAluminumMountingHubfor6mmShaft2CM3Holes2-PackEN) | **€11,95 / 2-pack** | Set-screw hub for the 6 mm D-shaft; wheel bolts to hub |
| Alt: Pololu Multi-Hub Wheel 80×10 mm (2-pack) | [Eckstein-shop.de](https://eckstein-shop.de/Pololu-Multi-Hub-Wheel-w-Inserts-for-3mm-and-4mm-Shafts-8010mm-Black-2-pack-EN) | €14,20 / 2-pack | Inserts are 3/4 mm only — **TBD: verify it accepts the 6 mm universal hub** before buying |

✅ **Decided + applied to sim (July 2026):** 90×10 mm wheels accepted → wheel geom r = 0.045 m, half-width 0.005. With the chosen 50:1 motor, top speed is 20.9 rad/s × 0.045 ≈ **0.94 m/s** and stall push force ≈ 2.06/0.045 ≈ 46 N per wheel — ample. Note the larger radius destabilized the sim's pitch dynamics until tire compliance (`solref="0.05 1"`) and gearbox damping were modeled — see SimNotes.md.

---

## Chassis & mechanical

| Part | Source | Price | Specs → sim |
|---|---|---|---|
| **Pololu Ball Caster with 3/4″ metal ball** (#955) | [EXP-Tech](https://www.exp-tech.de/zubehoer/mechanische-bauteile/5551/pololu-ball-caster-with-3/4-metal-ball) (in stock); also [BerryBase](https://www.berrybase.at/pololu-ball-caster-0-75-zoll-metallkugel-abs-gehaeuse-hoehenverstellbar-fuer-kleine-roboter), [TME.eu](https://www.tme.eu/en/details/pololu-955/accessories-for-robotics-and-rc/pololu/ball-caster-with-3-4-metal-ball/) | **≈ €4,50** (€3,72 net) | Ball ⌀ 19 mm → caster `sphere` geom; height 0.83″–≈1″ (21–25 mm) adjustable via spacers → chassis ground clearance. Mass: TBD (verify on datasheet) |
| Pololu 37D Metal Gearmotor Bracket (pair) | [Eckstein Pololu mounts category](https://eckstein-shop.de/Pololu-Motor-Mounts-Wheel-EN) | TBD | Sets motor axle height above chassis plate |
| Chassis plate | TBD (laser-cut acrylic/alu, or Misumi/igus stock profiles) | TBD | Track width target 0.21 m — set by motor bracket spacing, verify wheel-to-wheel once brackets chosen |

---

## Vision

Two viable routes — see Open decisions.

### ✅ CHOSEN — Option A: 2× Raspberry Pi Camera Module 3 (DIY stereo, July 2026)

- Source: [Welectron](https://www.welectron.com/Official-Raspberry-Pi-Camera-Module-3) — **€25,50 each** (≈ €51 for the pair); also [BerryBase](https://www.berrybase.de/en/raspberry-pi-camera-module-3-12mp). Specs: [raspberrypi.com](https://www.raspberrypi.com/documentation/accessories/camera.html)

| Spec | Value | → Sim / MJCF parameter |
|---|---|---|
| Sensor / resolution | Sony IMX708, 11.9 MP, 4608 × 2592 | camera resolution in renders |
| Horizontal FOV | 66° | — |
| **Vertical FOV** | **41°** | - |
| Focus | motorized PDAF, ~10 cm–∞ | near-field plug alignment OK |
| Mass | **4 g** each | camera geom mass (negligible) |
| Dimensions | 25 × 24 × 11.5 mm | mount design |

- Mounting at **60 mm baseline**: modules are 25 mm wide, so 60 mm center-to-center leaves ~35 mm clear between them — fine. Needs a rigid printed/laser-cut bracket (stereo calibration dies if the baseline flexes) — no off-the-shelf 60 mm stereo mount; plan a custom bracket. Note: Pi 5 has two CSI ports, so both cameras connect natively, but frame capture is not hardware-synchronized (software sync is usually acceptable at robot speeds ≤1.5 m/s).

### Option B (not selected): Luxonis OAK-D Lite (integrated stereo + depth ASIC)

- Source: [MYBOTSHOP.DE](https://www.mybotshop.de/Luxonis-DepthAI-OAK-D-Lite_1) — **€192,95**; [Reichelt](https://www.reichelt.com/de/en/shop/product/luxonis_depthai_oak-d_lite_fixed-focus-324637) — €199,40 (fixed-focus **backordered until ~17 Aug 2026**); also [Botland](https://botland.store/modules-smart-cameras/20955-oak-d-lite.html). Specs: [docs.luxonis.com](https://docs.luxonis.com/hardware/products/OAK-D%20Lite)

| Spec | Value | → Sim / MJCF parameter |
|---|---|---|
| **Stereo baseline** | **75 mm** | ⚠ sim assumes 60 mm — **update stereo camera separation to 0.075 m** if chosen |
| Mono (depth) cameras | 2 × 640 × 480 | depth quality model |
| Depth range | MinZ ≈ 35 cm (≈20 cm extended mode), MaxZ ≈ 10 m ± 10 % | ⚠ 35 cm min depth is tight for terminal plug-docking approach |
| Mono camera FOV | TBD (verify on datasheet) | camera `fovy` |
| RGB camera | 12.3 MP, 4K/30 | — |
| Compute | on-board depth + 1.4 TOPS NN (RVC2), USB-C | offloads Pi 5 |
| Mass | TBD (verify on datasheet, ~60 g class) | camera geom mass |

---

## Electronics — later (low priority)

- **Motor driver:** [Cytron MDD10A](https://botland.store/drivers-for-dc-motors/15818-cytron-mdd10a-dual-channel-30v-10a-motor-controller-5904422350444.html) (Botland, price TBD ~€20-class) — dual channel, 10 A continuous / 30 A peak per channel at 5–30 V, comfortably covers the 5.5 A stall current per motor.
- **Compute:** Raspberry Pi 5 8 GB — [BerryBase](https://www.berrybase.de/en/raspberry-pi-5-8gb-ram); ⚠ cheapest Geizhals listing July 2026 is **€202,90 incl. active-cooler kit** ([Geizhals](https://geizhals.de/raspberry-pi-5-modell-b-a3096144.html)) — well above the historical ~€90 board price (RAM price surge); verify standalone-board price before budgeting.

---

## Open decisions

1. ~~Stereo camera~~ → **Decided (July 2026): 2× Camera Module 3** with a custom rigid 60 mm bracket. Rationale: 4× cheaper, exact baseline control, 41° fovy (applied to sim), and the OAK-D Lite's ~35 cm minimum depth is a bad fit for terminal docking approach.
2. ~~Wheel diameter~~ → **Decided (July 2026): 90×10 mm** (sim r = 0.045 applied).
3. ~~Gear ratio~~ → **Decided (July 2026): 50:1 (#4753)** — docking push force over top speed (sim updated; see Motors section).
4. Chassis material/supplier (Misumi/igus vs laser-cut) — blocked on motor-bracket choice.
