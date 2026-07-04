# OmniMarble Driver PCB — Layout Review Handoff (rev 2)

**Repo:** github.com/addisokw/omnimarble
**Board:** `hardware/omnimarble-driver/omnimarble-driver.kicad_pcb`
**Toolchain:** KiCad 10.0.4 (kicad-cli + pcbnew Python API)

> **rev 2 corrects rev 1's overclaim.** The board is **NOT DRC-clean and NOT
> fab-ready.** It has 0 shorts / 0 crossings, but 59 real clearance errors (some
> near-shorts on safety/gate nets) and 118 unconnected nets (34 of them
> safety/control/analog). This is an **iteration checkpoint**, not a fab gate.
> All three prior commits (`e64fb78`, `bb3ea38`, `ae83ebe`) contain a
> byte-identical board file (sha256 `ed02f3b…`); only docs/renders differ between
> them. Rev-1 numbers were run without `--schematic-parity`, which under-reported.

## Context

Experimental-side driver PCB (55 V SELV coilgun pulse driver) for the sim-to-real
loop. Schematics passed two prior review rounds. This handoff covers **PCB layout**.
It was routed with a purpose-built parallel PathFinder engine
(`hardware/scripts/gen_pcb.py` + `route_worker.py`) after freerouting and
OrthoRoute both proved unusable for this board.

## Board facts

220.1 × 150.1 mm, 4-layer 2 oz outer / 1 oz inner (In1/In2 = solid GND + power
planes). 178 footprints, 4,390 tracks, 696 vias, 20 zones.

## Verified DRC state (kicad-cli, `--schematic-parity`)

| Rule | Count | Honest disposition |
|---|---|---|
| shorting_items | **0** | copper has no dead shorts |
| tracks_crossing | **0** | no illegal overlaps |
| **schematic_parity** | **0** | FIXED this rev (C92–94 DNP, TP1/NT1/NT2 BOM-exclude) |
| **clearance** | **59** | **BLOCKER — real, not cosmetic.** Gaps down to 0.010 mm on FIRE_GATED/DRV_SPARE, +5V/GATE*, BOOST_EN_N/BSTINH_B, ADC_CS/ESTOP_SENSE. These are fail-safe/gate/E-stop nets. |
| **unconnected** | **118 / 59 nets** | **BLOCKER for fab.** 34 airwires are safety/control/analog (see below). |
| silk_overlap / silk_over_copper | 199 / 199 | cosmetic, deferred; still needs cleanup before fab |
| track_dangling / hole / via | 156 | finishing residue, clears as nets close |

## The 34 airwires that are NOT harmless plane taps

Still open on safety/control/analog/boost/Kelvin nets — these make the board **not
electrically reviewable yet**:

`FIRE_GATED`(2), `MCU_FIRE`(1), `BOOST_EN_N`(1), `ESTOP_SENSE`(1),
`ARM_SENSE`(1), `GATE1/3/4/5/6`, `DRV1/2/3`, `BST_DRV`, `BST_G`,
`BST_CS`(1), `BST_CS_HI`(2), `BST_FB`(2), `VTH_GATE`(5),
`ISNS_P`(1), `ISNS_N`(1), `ADC_SDO/SCK/CS`.

## What DRC does NOT prove (external reviewer, agreed)

DRC passing does not validate the things that actually matter for this board:
current sharing across the 3 paralleled FETs, pulse-loop inductance, FET
drain/source copper symmetry, true Kelvin pickup at the shunt pads, thermal-relief
correctness on pulse pads, or quiet/short boost current-sense + feedback routing.
**The pulse/gate/boost/Kelvin/safety nets need hand routing and manual review with
highlighted nets** (`VBOOST`, `COIL_HI`, `SW_DRAIN`, `SHUNT_HI`, power-ground
return, `ISNS_P/N`, `BST_CS/CS_HI/FB`), not autorouting.

## Agreed next gate (before ANY fab/JLC export)

1. **0 unconnected** — do not waive safety/control/analog nets.
2. **0 clearance.**
3. **0 schematic parity** — done this rev.
4. Then **manual power-loop / Kelvin / boost review** with net-highlighted
   screenshots proving symmetric FET copper, intentional (or absent) thermal
   reliefs on pulse pads, and true Kelvin shunt pickup.

## Planned approach

Critical nets (pulse loop, gate drive, boost sense/feedback, Kelvin, safety
interlock) will be **routed deterministically as reviewed fat/short traces and
locked** before the autorouter touches the remaining low-speed interconnect —
autorouting is only acceptable for noncritical nets.

## Artifacts

- Renders: `hardware/fab/renders/driver_top.png`, `driver_bottom.png`
- Scorecard dashboard: `hardware/fab/driver_pcb_review.html`
- DRC report (with parity): regenerate via
  `kicad-cli pcb drc --schematic-parity --format json`
