# Sim corrections from bench measurement (2026-07-26)

Findings from characterising the physical demo coil and pulse loop on the
`omnimarble-vbench` board. **Nothing here has been changed yet** — this is the
change list, written while the measurements are fresh.

Measurement provenance: 4-wire LCR (FNIRSI LC1020E), a gate-on-time discharge
sweep on the real hardware, and DC current injection. See
`../omnimarble-vbench/docs/REV_B_NOTES.md` for the raw values and method.

---

## S-1 — `mean_radius` is computed from the wrong geometry — **the big one**

`CoilPhysics.recompute_derived()` builds **two different mean radii** and uses
the wrong one for the circuit model:

```python
self.R_mean      = (self.inner_radius + self.outer_radius) / 2   # = 15.000 mm  <- field model
self.mean_radius = self.inner_radius + winding_depth / 2         # = 12.435 mm  <- L and R_dc
```

`R_mean` (15 mm) is right and matches the build contract — the jig's grooved
former enforces a **15 mm loop-centre radius**, and `parts.py`'s
`DEMO_COIL_BUILD` says to match loop centres, not wire edges.

`mean_radius` (12.435 mm) is wrong. With a 0.87 mm wire pitch the winding is a
**single layer**, so `inner_radius + depth/2` lands at 12.435 mm — it treats the
config's `inner_radius_mm` as the true winding inner edge and ignores
`outer_radius_mm` entirely. The config's 12/18 mm pair was written to *bracket* a
15 mm mean.

Both `inductance_uH` and `R_dc` are derived from the wrong radius:

| | As coded (12.435 mm) | Corrected (15.0 mm) | **Measured** |
|---|---|---|---|
| L | 12.41 µH | **17.30 µH** | **17.90 µH** |
| R_dc | 0.0802 Ω | **0.0968 Ω** | **0.1068 Ω** |

Both residuals after correction are physically accounted for:

- **L, 3.4% low** — Wheeler's multilayer formula is only good to ~±5%, and this
  coil is a single layer, at the edge of its intended domain.
- **R_dc, 10 mΩ low** — the model has no leads; the LCR measurement included the
  coil's flying leads.

**This closes the open question in `omnimarble-vbench/docs/BRINGUP.md` §5**, which
noted that the Wheeler estimate (17.3 µH) and the sim config (12.4 µH) disagreed
and said the measured value would reconcile them. It does: **Wheeler was right,
the sim was wrong, and the cause is this line.**

### Fix

Use the loop-centre radius for the circuit model, not the wire-edge estimate.
Simplest correct change is to compute L and R_dc from `R_mean`. Note the wire
length loop a few lines below has the same origin error — it accumulates
`r_layer = inner_radius + (layer + 0.5) * wire_pitch`, giving 2344 mm of wire
where the real coil has 2827 mm.

### Same bug, second copy

`scripts/rlc_circuit.py` line ~78 repeats it verbatim:

```python
mean_radius = inner_r + winding_depth / 2
```

It *does* read `outer_radius_mm`, but only as `winding_depth_available` for a
fit check — never for the mean radius. The header of `tests/test_coil_physics.py`
already flags this duplication as tracked tech debt; **both copies must change
together or the cross-check test will start failing.**

---

## S-2 — Config values that are cached model outputs, not measurements

`config/coil_params.json` carries `resistance_ohm` and `inductance_uH`, and
`test_derived_values_match_config` asserts they equal what `CoilPhysics` computes
from geometry to `rel=1e-9`.

**Do not "update them to the measured values."** They are a cache of the model's
own output. Overwriting them would break that test and, worse, hide S-1 — the
whole point of the bench rig is to surface exactly this kind of disagreement.

Once S-1 is fixed, regenerate them: they should become **17.30 µH** and
**0.0968 Ω**.

---

## S-3 — Bench circuit parameters differ from the config

These are genuine inputs, not derived, and the config does not describe the rig
that was actually built:

| Field | Config | Bench (measured) | Note |
|---|---|---|---|
| `capacitance_uF` | 470.0 | **1909 per can**, 1–5 cans | see below |
| `esr_ohm` | 0.01 | **0.048 per can** | parallels as 1/n |
| `wiring_resistance_ohm` | 0.02 | **~0.009** | board + FETs + shunt |
| `charge_voltage_V` | 50.0 | ≤ 55 hard limit | board invariant |

**I have deliberately not changed `capacitance_uF`.** 470 µF may be an intended
design point rather than a stale value, and it is 4× off the bench bank — that is
a decision, not a correction. If the sim is meant to model the vbench rig, it
wants 1909 µF × cans.

Beware when changing these: `test_rlc_regime` asserts `zeta == 0.34 ± 0.02`.
Fixing S-1 alone keeps it passing (ζ: 0.339 → 0.330), but changing ESR and
capacitance as well will move ζ out of that window, and the assertion will need
revisiting on physics grounds rather than being widened to fit.

---

## S-4 — Bank ESR scales with can count, so R is not constant across a C-sweep

Bank ESR parallels as **1/n**, so across a 1→5 can sweep the loop resistance
moves **0.164 Ω → 0.126 Ω** while capacitance rises.

A model holding R fixed will attribute that resistance change to capacitance and
appear to validate for the wrong reason. Model it as:

```
R_loop(n) = R_coil_ac + R_fixed + ESR_per_can / n
```

---

## S-5 — Use AC resistance, not DC, for the pulse

The coil's resistance is frequency dependent, and the discharge is not DC:

| Frequency | Coil R |
|---|---|
| DC / 1 kHz | 0.107 Ω |
| 10 kHz | 0.140 Ω |

The measured loop total of **0.164 Ω** is consistent with the coil's AC
resistance in the pulse's band (~0.3–1 kHz) plus 0.048 Ω of ESR plus ~0.009 Ω of
copper. Using the DC figure alone under-predicts damping.

Independent confirmation: the discharge-sweep fit gives 0.164 Ω and DC injection
of 1 A gives 0.161 Ω — 2% apart.

---

## Measured reference (as-built rig)

| Parameter | Value | Method |
|---|---|---|
| Coil L | **17.9 µH** | 4-wire LCR, 10 kHz (Q ≈ 8) |
| Coil R | 0.107 Ω @1 kHz, 0.140 Ω @10 kHz | 4-wire LCR |
| Bank can | **1909 µF**, ESR 48 mΩ | LCR @100 Hz |
| Loop R | **0.164 Ω** | on-time sweep fit |
| Loop R | 0.161 Ω | DC injection @1 A |
| I_peak, 1 can @9.5 V | 40 A at 195 µs | fitted model |
| I_peak, 5 cans @55 V | 358 A at 361 µs | fitted model |

Measure L at **10 kHz, not 1 kHz** — at 1 kHz this coil's Q is ≈ 1 (reactance and
resistance nearly equal), which is where LCR inductance readings are least
reliable. The 1 kHz reading first suggested 17.8 µH but a discharge fit disagreed
until the 10 kHz measurement settled it.
