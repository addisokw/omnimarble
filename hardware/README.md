# OmniMarble driver board (60V-class SELV coilgun pulse driver)

Experimental-side electronics for the OmniMarble digital twin: drives a
real coil with the same fire/cutoff logic as the simulation's Kit
extension and emits telemetry in the sim's trajectory-CSV format for
direct sim-to-real comparison.

**Status: schematic checkpoint (externally reviewed, re-audited clean).
PCB layout not yet started.**

- Operating point: bank charged **24.5–55 V** (63 V-rated capacitors,
  hardware OVP trips 59.1–62.0 V worst-case, asserted in
  `scripts/calcs.py`), one shot per several seconds.
- Bank: 5× snap-in positions, 2200 µF/63 V each (baseline 2 populated =
  4400 µF; max 11000 µF).
- Switch: 3× paralleled 150 V TOLL MOSFETs, designed to a **600 A / 2 ms**
  pulse point; MOSFET-cutoff + freewheel topology reproduces the sim's
  circuit model.
- All engineering numbers in `design-calcs.md` are generated and
  margin-asserted by `scripts/calcs.py` — do not hand-edit.

## Supported coils

The coil is off-board (J5 barrier terminals) and interchangeable. A coil
is **validated** iff it satisfies the envelope swept by `calcs.py` at the
worst configuration (11000 µF bank at 55 V):

> **L ≥ 1 µH AND R_total ≥ 90 mΩ** (total series resistance *including*
> leads and lugs — keep leads short and heavy)

Below the envelope the pulse can exceed the 600 A switch design point —
do not fire. The envelope numbers live in `coil-envelope.json` (generated
by calcs.py) and are printed on the flyback schematic sheet next to J5;
the full safe/unsafe sweep table is in `design-calcs.md`.

### Validated default: demo coil (the coil the simulation models)

| Parameter | Value |
|---|---|
| Turns / wire | 30 × 0.8 mm (AWG20), single layer |
| Geometry | inner r 12 mm, outer r 18 mm, length 30 mm |
| Inductance | 12.406 µH |
| R_dc / R_total | 80.2 mΩ / **110.2 mΩ** (incl. 30 mΩ ESR+wiring budget) |
| At 55 V / 11000 µF | ~420 A peak, ~1.1 ms, 16.6 J |

Using this coil keeps hardware runs directly comparable to Kit runs
(`config/coil_params.json` is the shared source of truth).

### Optimizer-recommended coil for THIS board (55 V / 600 A constrained)

`uv run python scripts/optimize_coil_design.py --max-voltage 55 --max-current 600`
(PINN + coupled-ODE screening, 2000 LHS candidates, run 2026-07-03):

| Parameter | Value |
|---|---|
| Turns / wire | **75 × AWG18 (1.024 mm), 3 layers** |
| Geometry | inner r 8.5 mm, outer r 11.8 mm, length 30.5 mm |
| Inductance / R | 49.4 µH / R_dc 99 mΩ → R_total ~129 mΩ |
| Drive point | ~51 V, ~3700 µF (2 bank cans ≈ 4400 µF is the closest fit) |
| Pulse | 228 A peak, 1.6 ms, 4.8 J, ΔT ≈ 0.3 °C |
| Predicted boost | **+1.0 m/s (0.5 → 1.5 m/s, ~3×)** |

Envelope check: 129 mΩ ≥ 90 mΩ and 49.4 µH ≥ 1 µH — inside. Note the
earlier optimizer result quoted in the repo (396 V / 247 J design) was
computed for an unconstrained rig and is **not** a recommendation for
this board.

Caveat: these predictions come from the PINN + analytical circuit model.
Validating them against reality is this board's entire purpose — treat
the boost figure as a hypothesis, not a spec.

## Safety summary (details in design-calcs.md)

- Hardware interlock: FIRE = MCU_FIRE AND ARM-keyswitch (74LVC1G08), and
  the gate drivers are only powered through the E-STOP (NC) + keyswitch
  loop. Explicit pulldowns on all fire nets.
- Boost charging is fail-safe inhibited: a dead/absent Pico cannot charge.
- Hardware OVP (CJ431 ±0.5 %) independent of the MCU.
- Permanent 6.8 k bleed (3τ ≈ 224 s) + commanded 200 Ω dump (<5 V in
  5.3 s) + hardwired live-bank LED.
- Relays never break DC under load (sequencing: precharge before close,
  boost-inhibit before open, dump closes onto 0.28 A).

## Order/production gates

- `uv run python hardware/scripts/validate.py` — schematic completeness
  (review mode; warns on ORDER-REVIEW placeholders).
- `uv run python hardware/scripts/validate.py --production` — fab-export
  gate; **fails** while any ORDER-REVIEW placeholder remains. Currently
  open: the 6.8 k/2 W bleed resistor C-number (Ever Ohms CRH2512J6K80E04Z
  — search "CRH2512 6K80" on LCSC at order time).
- BOMs: `bom/bom.csv` (engineering), `bom/bom_rail_emit.csv` /
  `bom_rail_recv.csv` (gate-rail variants incl. common parts). JLC
  assembly exports (bom_jlc/cpl_jlc) are generated in the layout phase.

## TODO (next phases)

- PCB layout (4-layer 2 oz, pcbnew-scripted, DRC-gated) + gate-rail board
  with the ±0.1 mm station-position check — awaiting go after schematic
  audit.
- JLC fab/assembly exports; DRAFT footprints (TOLL-8, 9.5 mm barrier,
  5930 shunt, HF3FF relay) get datasheet-verified dimensions here.
- Pico firmware (fire/cutoff state machine, gate timestamps, DMA current
  capture, TRAJ_COLUMNS-compatible telemetry) + `compare_sim_real.py`.
- Bring-up procedure (first power at 25 V floor, staged to 55 V).
