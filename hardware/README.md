# OmniMarble driver board (60V-class SELV coilgun pulse driver)

Experimental-side electronics for the OmniMarble digital twin: drives a
real coil with the same fire/cutoff logic as the simulation's Kit
extension and emits telemetry in the sim's trajectory-CSV format for
direct sim-to-real comparison.

> ⚠️ **vNext — on hold (2026-07-06).** This integrated driver board is not the
> current path. The launch concept is still unproven, so we stepped back to a
> bench rig built from off-the-shelf modules — see **[`V0_BENCH_RIG.md`](V0_BENCH_RIG.md)**.
> This board is DRC-clean and preserved; revisit once V0 proves the concept out.

**Status: layout final — fully routed, electrically DRC-clean (0 unconnected /
shorts / clearance / parity), all validators green, and the JLCPCB fab/assembly
package generated (`hardware/fab/jlc/`). Externally reviewed.**

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

### KNOWN MODEL INCONSISTENCY (affects how you build the coil)

The simulation is internally inconsistent about where the windings sit:

- The **field model** (`analytical_bfield.solenoid_field`, which the PINN
  was trained on and the launch force derives from) places all 30 turns
  at a **single 15.0 mm radius** — the mean of config's inner 12 / outer
  18 mm.
- The **circuit model** (Wheeler inductance + winding resistance in
  `coil_physics.py` / `rlc_circuit.py`) assumes a single layer of 0.8 mm
  wire on a 12 mm former (wire centers ~12.4 mm), which is where config's
  persisted `inductance_uH = 12.406` and `resistance_ohm = 0.0802` come
  from.

No physical coil satisfies both. **Resolution: build to the FIELD
geometry** (that is what the PINN/force model is being validated
against), then reconcile the circuit side by measurement — the config
accepts measured `inductance_uH` / `resistance_ohm` and the RLC model
uses whatever is there. Building at 12 mm instead would bias the field
test: closer windings give a measurably stronger on-axis field than the
model predicts.

### Coil build sheet — experiment #1 (demo coil, field-geometry build)

The field model uses **ideal loop centers**: 30 loops at radius 15.0 mm
with centers at `np.linspace(-15, +15, 30)`. The physical winding must
match **turn centers**, not wire edges. Machine-readable spec:
`hardware/scripts/parts.py DEMO_COIL_BUILD`, contract-tested against the
field model by `tests/test_coil_physics.py::
test_demo_coil_build_matches_field_model` (fails if either side drifts).

| Item | Spec |
|---|---|
| Target geometry | **Wire CENTER radius = 15.0 mm** (this is the spec; former OD follows from your wire) |
| Former | For 0.87 mm finished wire OD (0.8 mm copper + 2×0.035 mm enamel): **~29.13 mm OD**. Measure your wire's finished OD and set former OD = 2×(15.0 − OD/2). 32 mm winding window between flanges |
| Wire | 0.8 mm copper (AWG20) enameled magnet wire, ~3 m incl. leads |
| Turns | **30 turns, single layer, one direction** |
| Spacing | **30 turn CENTERS evenly spanning 30.0 mm** → center-to-center pitch **30/29 ≈ 1.034 mm** (not 1.000). First and last turn centers sit at ±15.0 mm from coil center. NOT close-wound (that compresses the span to ~26 mm and shifts the field profile) |
| Leads | Short and heavy (>=AWG16 or doubled), crimped ring lugs to J5 |

Winding tips: score/print **1.034 mm-pitch** guide grooves (or a printed
thread) so the groove *centers* span exactly 30.0 mm; secure the ends
(CA glue) before the winding relaxes; leave 100 mm tails.

**Verify after winding** (LCR meter ~1 kHz + milliohm measurement), then
**write the measured values into `config/coil_params.json`** so sim and
hardware run identical circuit numbers:

| Quantity | Expected (field-geometry build) | Config's current value |
|---|---|---|
| Inductance | **~17.3 uH** (Wheeler at 15 mm) | 12.406 uH (12.4 mm assumption) |
| R_dc | **~97 mOhm** (2.83 m of AWG20) | 80.2 mOhm |
| R_total w/ leads | ~125–130 mOhm | 110.2 mOhm |
| Envelope check | 127 mOhm >= 90 mOhm, 17 uH >= 1 uH — inside; worst-case pulse ~430 A vs the 600 A design point | — |

### Experiment #2 (after #1 validates the model)

Wind the optimizer-recommended coil below; its predicted **+1.0 m/s
boost is itself a falsifiable prediction** of the digital twin.

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
  clean: the R24 bank-bleed resistor is resolved to UNI-ROYAL
  25121WJ0682T4E / LCSC **C26073** (6.8 k, 1 W, 2512).
- BOMs: `bom/bom.csv` (engineering), `bom/bom_rail_emit.csv` /
  `bom_rail_recv.csv` (gate-rail variants incl. common parts). JLC
  assembly exports (`fab/jlc/bom_jlc.csv`, `cpl_jlc.csv`) are generated by
  `gen_fab.py`.

## Status & next phases

- **Done:** schematic; 4-layer 2 oz PCB layout (routed, electrically
  DRC-clean, datasheet-verified footprints for TOLL-8 / 9.5 mm barrier /
  5930 shunt / HF3FF relay); and the JLCPCB fab/assembly package
  (`hardware/fab/jlc/`, via `gen_fab.py`).
- **Next:** upload to JLC and clear its DFM (see `fab/jlc/ORDER_NOTES.md`);
  Pico firmware (fire/cutoff state machine, gate timestamps, DMA current
  capture, TRAJ_COLUMNS-compatible telemetry) + `compare_sim_real.py`;
  bring-up procedure — first power at 25 V floor, staged to 55 V, and
  **verify OVP trip + boost regulation before installing max bank
  capacitance or firing into the coil** (the 63 V-cap / 59.1–62.0 V OVP
  window is tight).
