# OmniMarble vbench (SIM-validation rig) — Overview

The **single board that validates the PINN twin** against a real single-coil
launcher: fire a timed coil pulse from an **expandable capacitor bank** and expose
the observables the sim already emits (`TRAJ_COLUMNS`). It is the live V0 hardware
path — see **[`../V0_BENCH_RIG.md`](../V0_BENCH_RIG.md)** and the switch theory in
**[`../STAGE2_SWITCH.md`](../STAGE2_SWITCH.md)**. Not the parked integrated driver
(no boost / OVP / interlocks).

## What it is
- **150 × 136 mm, 4-layer** (2 oz outer / 1 oz inner), **bare board / hand-populated**
  (iron-solderable throughout: TO-247, SOIC-8, D2PAK, Shunt-5930, 0805, THT bank +
  headers). Stackup: **F.Cu + B.Cu = signals, In1 = solid GND plane, In2 = GND plane
  with a +3V3 pocket over the logic zone.** The inner GND plane is never cut by a
  signal, so the return plane can't fragment — the reason this board is 4-layer.
- **One board** — the originally-planned power + logic boards were merged: the
  Waveshare sensor outputs are digital/noise-immune and the fire pulse never
  coincides with a velocity-gate read, so the GND plane + physical separation of the
  logic zone from the pulse loop is discipline enough.

## Sections
- **Expandable bank** — 5× **18 mm radial 2200 µF/63 V** positions, populate 1→5 via
  the `dnp=(i>=n)` pattern (expanding is soldering in the next can). Charge terminal +
  **aux-bank** barrier terminal to bolt on more capacitance off-board; bleeder. This
  is where the boost becomes measurable at SELV — the range to run near the top of.
- **Switch (sized for peak current, not cutoff speed)** — 3× parallel **IRFP4668**
  (TO-247, 200 V/130 A) for the ~500 A top of the C-sweep, driven by a **UCC27524A**
  gate driver. Rg 10 Ω/FET, 10 k gate pulldown (default-off), FIRE 100 Ω series.
- **Freewheel to MATCH the sim** — **MBR60100** across the coil (the sim sets
  `has_flyback_diode=true`; a plain freewheel, *not* a fast TVS clamp). The pulse
  self-terminates, so cutoff timing is non-critical.
- **Current sense** — Kelvin **0.2 mΩ shunt** (Shunt-5930) in the source return to a
  scope header (~100 mV at 500 A) — resolves the ~0.24 ms pulse. 100 k/5.1 k divider
  → Pico ADC for bank voltage.
- **On-board Pico + 4 Waveshare ITR20001 connectors** — 20 sensor channels for
  multi-station tuning (the rig needs 2, before/after the coil; the extra 2 are
  headroom). FIRE + a V_bank ADC pin stay free, 4 GPIO spare. Waveshares powered at
  **3.3 V** — the RP2040 is *not* 5 V-tolerant.
- **Interfaces** — coil / charge / aux-bank / 12 V-in barrier terminals, ISENSE scope
  tap; star ground at the bank-negative node.

## How it is generated (scripted, plane-locked, freerouted)
1. `gen_vbench_pwr.py` → schematic (`vbench-pwr.kicad_sch`) + `.kicad_pro`
2. `kicad-cli sch export netlist` → `vbench-pwr.net`
3. `gen_vbench_pwr_pcb.py place` → footprints + parity/DNP metadata + inner GND
   planes (In1/In2) + the In2 +3V3 pocket
4. `gen_vbench_pwr_pcb.py dsn` → Specctra DSN, **plane-locked** (`plane_lock_dsn`
   marks In1/In2 `(type power)` so freerouting never routes signals on them); then
   **freerouting** (`-mp 30 -mt 1`) → SES
5. `gen_vbench_pwr_pcb.py import` → back-import routing, drop a GND via at each SMD
   GND pad (freerouting doesn't via them to the inner planes), fill
6. `gen_fab.py vbench` → 4-layer gerber package

**Why this is robust.** Signals route on the two outer layers (F.Cu + B.Cu = ample
room, freerouting completes in ~2 s); the GND return is a **dedicated inner plane
that a signal can never cut**, so it can't fragment. +3V3 (all-THT pins) connects
through the In2 +3V3 pocket by via, with no routing. There is **no post-hoc GND
stitching** — a full re-route from scratch reproduces a clean board every time
(verified across repeated runs). This is the driver board's dedicated-plane pattern,
applied here to fix the 2-layer GND-fragmentation the earlier revision hit.

## Status
- ERC **0**; DRC `--schematic-parity` **0 parity / 0 electrical / 0 unconnected**
  (1 cosmetic silk-over-copper). `validate.py` PASS (bare board: LCSC not required).
  Reproducible: repeated place→route→import runs give the same DRC-clean board.
- Fab package (`hardware/fab/vbench/`): 4-layer gerbers + drill (14 files),
  `vbench-gerbers.zip`, reference `bom_vbench.csv` / `cpl_vbench.csv`.

## Ordering (bare board)
Order the **bare 4-layer PCB** (2 oz outer / 1 oz inner) — gerber zip; hand-populate
everything. The
IRFP4668, 18 mm cans, headers, Pico socket and Waveshare connectors are user-sourced.
Bring up per **[`../STAGE2_SWITCH.md`](../STAGE2_SWITCH.md)** — 12 V + logic first,
bank at ~10 V, add coil + shunt, then populate caps upward for the C-sweep.
