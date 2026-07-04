# OmniMarble Driver PCB — Layout Review Handoff

**Repo:** github.com/addisokw/omnimarble
**Board:** `hardware/omnimarble-driver/omnimarble-driver.kicad_pcb`
**Head commit:** `bb3ea38`
**Toolchain:** KiCad 10.0.4 (kicad-cli + pcbnew Python API)

## Context

This is the experimental-side driver PCB (55 V SELV coilgun pulse driver) for the
sim-to-real loop. Schematics passed two prior review rounds and are frozen. This
handoff covers the **PCB layout only**, which was just completed to a
pause-for-feedback gate.

## Board facts

220.1 × 150.1 mm, 4-layer 2 oz outer / 1 oz inner (In1/In2 = solid GND + power
planes). 178 footprints, 4,390 tracks, 696 vias, 20 zones. Pulse section (5×
snap-in cap bank → MBR60100DC D2PAK diodes → 3× SFT040N150C3 TOLL FETs → 0.2 mΩ
5930 shunt → coil barrier terminals) isolated on the top-left with
via-farm-stitched pours; boost charger by the input jack; INA240/ADS7042 Kelvin
section; Pico 2×20 sockets + 2×13 MCU-agnostic breakout + 2× IDC-10 gate-rail
headers on the right.

## How it was routed

freerouting (all 3 versions) and OrthoRoute (GPU) both proved unusable for this
board — freerouting's exporters are broken/GUI-blocked in this build; OrthoRoute's
file format can't carry existing zones/planes as obstacles, so its solutions short
into the pours. A purpose-built scripted router (`hardware/scripts/gen_pcb.py`,
parallel PathFinder negotiation, `route_worker.py`) was used instead. It ran 7
DRC-gated cycles.

## Verified DRC state (kicad-cli, current board)

| Rule | Count | Note |
|---|---|---|
| **shorting_items** | **0** | from 199 — hard gate met |
| **tracks_crossing** | **0** | from 25 |
| clearance | 59 | from 499 |
| **unconnected** | **118 airwires / 59 nets** | deliberate: router converts every would-be short into a visible airwire rather than hidden copper |
| silk_overlap | 199 | cosmetic, footprint body outlines (not text), deferred to fab-prep |
| silk_over_copper | 199 | cosmetic, deferred to fab-prep |
| track_dangling | 119 | finishing residue |
| hole_to_hole / co-located | 32 | finishing residue |
| via_dangling | 5 | finishing residue |

## Known compromises / open items for your scrutiny

1. **118 unconnected nets** — ~23 GND (via-stitch), ~31 power-rail plane taps
   (mechanical), remainder long cross-board signals (VTH_GATE, GATE3/5, BST_*,
   FIRE_GATED, ADC_SDO/SCK, MCU_FIRE, ARM_SENSE). Not yet closed.
2. **Silkscreen** deferred; a bulk API cleanup was attempted and reverted (it
   added text-height violations without fixing the outline-based overlaps).
3. **Autorouted geometry** — machine-optimal for connectivity, not hand-tuned for
   aesthetics or controlled impedance (acceptable for a SELV pulse board, but flag
   if you disagree).
4. **Order-time gates still open:** 6.8 k bleed resistor LCSC C-number
   (ORDER-REVIEW placeholder), relay NO/NC continuity verification.

## What I want your opinion on specifically

- Is a board with 118 honest airwires the right artifact to iterate from, or
  should connectivity hit zero before any fab-export work?
- Any electrical/layout concerns in the pulse loop, Kelvin sense routing, or the
  boost section that the DRC-clean copper might be masking?
- Whether the deferred silkscreen and autorouted track geometry are acceptable for
  this board class.

## Artifacts

- Renders: `hardware/fab/renders/driver_top.png`, `driver_bottom.png`
- Full scorecard dashboard: `hardware/fab/driver_pcb_review.html`
- DRC report: `hardware/fab/drc_driver.json`
