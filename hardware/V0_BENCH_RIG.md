# OmniMarble V0 Bench Rig — single-stage coilgun proof-of-concept

> **Current live hardware path (2026-07-06).** A breadboard-level rig wired from
> off-the-shelf modules to answer one question before we commit to a custom PCB.
> The integrated driver board and gate-rail sensor board are **vNext / on hold** —
> DRC-clean and preserved, to be revisited once V0 proves the concept out.
> See `fab/BOARD_OVERVIEW.md` and `fab/RAIL_OVERVIEW.md`.

## Why this exists

The integrated 55 V driver PCB is a lot of cost and time to commit for a concept
that is still **unproven**. Before spinning custom hardware, prove on the bench
that a **timed coil pulse boosts a steel marble the way the PINN predicts**.

The rig answers exactly one thing:

> **Does measured Δv = v_out − v_in match the sim across a voltage/timing sweep —
> and can we hit the firing timing to get it?**

Everything else is reused unchanged: the **Pico**, the **Waveshare sensor
modules**, and the **entire PINN / sim stack** (`scripts/`, `models/`,
`config/coil_params.json`). The only swap is *custom integrated driver* → *three
COTS modules and a hand-wound coil on the bench*, where every node is probeable.

## Architecture

Two paths — the high-current **power path** and the low-voltage **signal/control
path** — sharing only a star ground at the bank negative terminal.

```
 POWER PATH (high current, ≤60 V SELV)
 ┌──────────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────────────────┐
 │ Bench supply │──▶│ Cap bank  │──▶│ N-MOSFET     │──▶│ Coil  (TVS/zener     │
 │ 24–48 V,     │   │ 2.2–4.7mF │   │ trigger      │   │ clamp across it)     │
 │ I-limited    │   │ 63 V      │   │ module       │   │  ▓▓▓ non-mag tube ▓▓▓ │
 └──────────────┘   │ +bleeder  │   └──────▲───────┘   │   ● steel marble →    │
                    └───────────┘          │ gate      └──────────────────────┘
                                           │
 SIGNAL / CONTROL PATH (logic, 3V3/5V)     │
 ┌────────────────┐                        │
 │ Waveshare A    │─5 lines─┐              │
 │ (BEFORE coil)  │         │        ┌─────┴──────┐
 └────────────────┘         ├──GPIO─▶│ Raspberry  │
 ┌────────────────┐         │        │ Pi Pico    │  reads v_in / v_out,
 │ Waveshare B    │─5 lines─┘        │            │  arms timed OFF, fires gate
 │ (AFTER coil)   │                  └────────────┘
 └────────────────┘
```

- **BEFORE** module → entry velocity `v_in` (marble crossing its 5 close sensors →
  Δt) **and** the coil-ON trigger.
- **AFTER** module → exit velocity `v_out`. `Δv = v_out − v_in` is the coil's boost.
- The Pico closes the loop; the FET module just switches; the supply/bank just
  store energy.

## Bill of materials

Spec-level with example parts — pick from what's on hand / in stock; nothing here
is a pinned SKU.

### Launcher (power path)

| Item | Spec | Example / note |
|---|---|---|
| Bench supply | Adjustable **24–48 V**, current-limited, ≥3 A | The current limit charges the bank gently and sets shot energy. A lab supply is ideal for iterating. |
| Cap bank | **2200–4700 µF, 63 V** electrolytic (one or a few) | ~3–6 J at 50 V — a real pulse, still manageable. Start at the low end. |
| Bleeder | **1–5 kΩ, ≥2 W** across the bank | Safety discharge — the bank holds charge. |
| Switch | **High-power N-MOSFET trigger module**, ≥60–100 V, tens of A peak, signal-in + power terminals | Or bare **IRFB4110** (100 V/180 A) + a gate-driver breakout (TC4420 / UCC27517). |
| Turn-off clamp | **TVS or zener** across the coil, clamp above bank V | Forces fast current decay = clean cutoff. **See suck-back note below.** |
| Coil | Hand-wound **18–22 AWG** enamelled magnet wire on a **non-magnetic** tube | Turns/length per the sim (`config/coil_params.json`). Or repurpose a solenoid. |
| Barrel + projectile | Non-magnetic tube (brass/aluminium/plastic) + **steel marble** | The tube is both barrel and sensor mount. |

**Clamp vs freewheel diode (important):** a plain freewheel diode across the coil
lets current keep circulating after the FET opens — slow decay, the field lingers,
and the marble gets *pulled back* (suck-back). A **TVS/zener clamp** at a higher
voltage forces the current down fast, which is what clean OFF timing needs. Your
existing **MBR60100** works as a starting freewheel, but plan to A/B it against a
TVS clamp.

### Sensing + control (signal path)

| Item | Spec | Example / note |
|---|---|---|
| Sensors | **2× Waveshare ITR20001/T** 5-ch reflective tracker module | Onboard LM393 comparators → **digital** outputs. One before, one after the coil. |
| Controller | **Raspberry Pi Pico** (or Pico W) | Reads sensors, computes firing, drives the FET gate. |
| Interconnect | 10 signal lines (5 + 5) + 5 V + GND | Pico has ~26 GPIO — plenty. |

## Wiring

- **Waveshare:** `VCC → +5 V`, `GND → common GND`. Its 5 digital outputs (per
  board) → **10 Pico GPIO** total. Because the modules threshold on-board, they go
  **straight to GPIO** — no external comparator needed.
- **FET module:** signal input ← a Pico GPIO (the fire line); module GND → common
  GND. Coil in series with the FET; bank across the FET + coil; **clamp across the
  coil**.
- **Grounding:** the pulse loop (bank → FET → coil) dumps tens of amps. Keep that
  loop physically short and tie logic ground to it at a **single star point (bank
  negative)** so pulse noise isn't injected into the sensor logic.

## Control / firing

- **ON:** the BEFORE module detects the marble approaching → Pico fires the coil ON.
- **OFF (the interesting part):** no sensor can sit at the coil *center* (the coil's
  there), so the cutoff is **open-loop from `v_in`**: feed `v_in` to the PINN, which
  predicts the **accelerated** transit-to-center — *not* `distance / v_in`, because
  the marble speeds up under the pulse — and the Pico arms the OFF timer accordingly.
  This is the digital twin closing its own loop.
- **Timing budget:** place the BEFORE module so there are a few ms between measuring
  `v_in` and the marble reaching the coil — enough for the Pico to compute and arm
  OFF. Don't butt it against the coil.

## Staged test protocol

Run in order — each stage de-risks the next.

1. **Stage 0 — dry fire (no marble).** Charge the bank low, fire the FET from the
   Pico, and scope the coil current pulse (current probe or a low-value sense
   resistor). Confirm: the FET switches, the clamp behaves, the Pico timing is what
   you commanded, and **both** Waveshare modules read cleanly. *Goal: the
   electronics + timing work before any marble is involved.*
2. **Stage 1 — first boost (marble, low energy).** Insert the marble, low bank
   voltage. Measure `v_in`, `v_out`. Confirm a **measurable Δv > 0** and that OFF
   fires before center (no suck-back deceleration). *Goal: the launcher does
   something, measurably.*
3. **Stage 2 — sim fit (ramp).** Sweep bank voltage and ON/OFF timing; log `v_in`,
   `v_out`, pulse current, and timing each shot. Fit measured **Δv vs the PINN
   prediction** across the sweep. *Goal: does the sim track reality? If yes, the
   integrated driver is a justified investment.*

## Safety

- Stay **≤60 V (SELV)**; charge through the supply's current limit.
- **Bleeder across the bank** — it holds charge. Bleeder + a shorting-stick habit
  before touching anything.
- **Clamp/diode on the coil** — inductive kick on turn-off.
- Keep the pulse loop short; fuse it if you scale energy.
- One shot per several seconds (thermal).

## Relationship to the rest of the project

- **Sim / PINN:** unchanged. The rig emits the same `v_in` / `v_out` observables the
  twin is validated against (`scripts/`, `models/`, `config/coil_params.json`).
- **vNext (on hold, preserved):** the integrated driver PCB
  (`fab/BOARD_OVERVIEW.md`) and gate-rail sensor board (`fab/RAIL_OVERVIEW.md`) —
  both DRC-clean. Revisit once V0 proves the concept.
