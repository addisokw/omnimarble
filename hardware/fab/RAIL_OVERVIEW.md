# OmniMarble Gate-Rail (sensor board) — Overview

The **sense side** of the OmniMarble rig: an IR break-beam sensor rail that
detects the steel marble as it passes six stations along the coil track, so the
measured launch can be compared against the PINN simulation. It is the companion
to the driver board and plugs into the driver's J6 / J7 headers.

## What it is
- **224 × 15 mm, 2-layer, 1 oz** copper. Low-current board (~144 mA on +5 V from
  6 IR LEDs at ~24 mA; sub-mA signal lines), so 1 oz is ample — no 2 oz/4-layer
  like the driver. B.Cu is a solid GND pour; 2 vias (one +5 V, one SIG_SPARE→TP1).
- **6 sensor stations** at z = **-60 / -40 / -20 / +5 / +60 / +120 mm** from the
  coil-center fiducial. Exact x-spacing is asserted to 0.01 mm at board generation
  (`gen_rail_pcb.py`), silk-labelled `z=±NN` per station.

## One PCB, two population variants
Both circuits are drawn on the same board; assembly selects which parts are fitted
(each symbol carries a `Variant` field, and a silk `EMIT [ ] RECV [ ]` checkbox
marks the built board):
- **EMIT** — populate **D1–6** (IR333C-A 5 mm IR LEDs, C5130) + **R1–6** (150 Ω
  0603, C22808). Drive: `+5V → 150 Ω → LED → GND`.
- **RECV** — populate **Q1–6** (PT204-6B 3 mm phototransistors, C5133). Collector
  → `SIGx`, emitter → GND (pulled up on the driver).
- **COMMON** — **J1** (IDC-10, C5665). D and Q share each station footprint; only
  one is fitted per build.

A break-beam setup needs at least one EMIT board and one RECV board facing each
other across the track.

## Connection to the driver
`J1` (IDC-10) pinout:
```
1=+5V  2=GND  3..8=SIG1..6  9=SIG_SPARE  10=GND
```
This matches driver **J7 (RAIL-RECV) pin-for-pin** — J7 carries +5 V/GND *and* the
six SIG lines to the front-end. Driver **J6 (RAIL-EMIT)** is **power-only**
(alternating +5 V/GND on all 10 pins); the EMIT rail only uses J1.1/J1.2/J1.10,
and its SIG traces dead-end at the unpopulated Q pads, so J6 works for EMIT.
**Cables are not interchangeable: never plug a RECV board into J6** (its
phototransistor collectors would sit across +5 V/GND — non-damaging but no sense
data). Label the ribbons; the silk EMIT/RECV checkbox helps.

## How it is generated (fully scripted, deterministic, no autorouter)
1. `gen_gate_rail.py` → schematic (`gate-rail.kicad_sch`) + `.kicad_pro`
2. `kicad-cli sch export netlist` → `gate-rail.net`
3. `gen_rail_pcb.py` → board — places footprints (with parity metadata), hand-emits
   every track / via / GND pour, asserts station spacing, fills. Needs only KiCad's
   bundled Python + `pcbnew`.
4. `gen_fab.py rail` → per-variant fab package (below)

## Status
- ERC **0**; DRC `--schematic-parity` **0 parity / 0 electrical / 0 unconnected**
  (20 cosmetic silk warnings — dense station labels, same as any tight rail).
- Fab package (`hardware/fab/rail/`): 2-layer gerbers + drill, gbrjob 1 oz + rev
  `r1`, **two variant assemblies** — `cpl_rail_emit.csv` (D1-6 + R1-6 + J1 = 13)
  and `cpl_rail_recv.csv` (Q1-6 + J1 = 7) — plus `bom_rail_{emit,recv}.csv` and
  `ORDER_NOTES.md`.

## Ordering (JLC turnkey)
- Enable **THT assembly** (mixed board: the resistors are 0603 SMD; the IDC header,
  IR LEDs and phototransistors are through-hole).
- **One** bare-board gerber order; **two** assembly jobs (EMIT and RECV) — the two
  variants populate different parts on the same footprints, so a single CPL can't
  build both. See `hardware/fab/rail/ORDER_NOTES.md`.
</content>
