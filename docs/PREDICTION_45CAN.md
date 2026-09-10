# Preregistered predictions: 4- and 5-can bank (the holdout)

Committed **before can 4 is soldered and before any 4/5-can measurement
exists**. Generated from tag `model-freeze-3can` — every constant in the
model derives from 1–3 can data, so cans 4 and 5 are pure extrapolation.
Per `VALIDATION_PROTOCOL.md` rule 4, this is the test that distinguishes a
sim from a lookup table. Nothing below may be revised after data exists;
misses are reported as loudly as hits.

Three layers, in order of what they actually test:

---

## Layer A — the decisive commitment: field + tail predict dv from any waveform

The frozen model's real content is the **PINN field, the force law, the
kinematics, and the τ = 275 µs freewheel tail**. The circuit is disposable
(the scope proved no linear RLC describes this bank). So the primary
preregistered claim is conditional and sharp:

> **Given a 49 V blank-fire scope capture at 4 (or 5) cans, the
> injected-current prediction (`simulate_rig_shot --current-csv`, frozen
> field, frozen τ) will match the paired-sweep marble dv at the same gate to
> within ±5%, at every gate length tested.**

This inherits nothing from the fallback circuit and is the direct
extrapolation test of everything the injection campaign validated at 3 cans
(1.03 / 1.01 / 1.04). A miss here is a genuine physics miss: field
nonlinearity with current, force-law breakdown at higher I, or τ failing to
transfer — each distinguishable by which gates miss.

## Layer B — the frozen sim's own numbers (strict, known-weak circuit)

The profile has no measured pulse pair for 4/5 cans, so the frozen sim falls
back to C = n × 1640 µF and the ESR-scaled loop R. **Declared in advance:
this circuit is expected to under-predict.** The droop trend
(0.859/0.931/0.958 of small-signal at 1/2/3 cans) says the real pulse C will
exceed n × 1640 by ~8–10%, so measured dv should land **above** these rows
by roughly 15–30%, with the excess largest near the knee. If measured lands
*below* Layer B, something new and unmodelled is happening.

Marble sweeps, 49 V, offset +9 (x = −13.78), v_in ≈ 0.245:

| on_us | 4 cans dv (mm/s) | 5 cans dv (mm/s) |
|------:|:---:|:---:|
| 400   | 159.2 | 172.3 |
| 700   | 229.8 | 263.6 |
| 1000  | 259.8 | 310.7 |
| 1500  | 276.4 | 343.5 |

Blank fires, V(t_on)/V0 at 49 V (fallback circuit; expected to read LOW at
long gates by up to ~0.05, as the linear model did at 3 cans — the real
discharge is underdamped and retains less):

| on_us | 4 cans | 5 cans |
|------:|:---:|:---:|
| 100 | 0.966 | 0.973 |
| 200 | 0.891 | 0.912 |
| 300 | 0.801 | 0.837 |
| 500 | 0.620 | 0.684 |
| 700 | 0.468 | 0.548 |
| 1000 | 0.300 | 0.389 |
| 1500 | 0.142 | 0.218 |

10 V blanks: same normalized rows minus a gap growing with on-time
(measured 0.003→0.035 across the gates at 2 and 3 cans; same sign, similar
size, expected here).

## Layer C — measured-trend expectations (component-level)

- **Can 4 pre-solder LCR** (100 Hz, standalone, per the can-3 ritual):
  C = 1880–1920 µF, ESR = 45–60 mΩ. Outside that band: do not solder,
  investigate.
- **Combined bank at J2 after soldering**: 4 cans ≈ **7.36–7.47 mF**,
  ESR ≈ **15 ± 2 mΩ**; 5 cans ≈ **9.23–9.34 mF**, ESR ≈ **13 ± 2 mΩ**
  (per-can 48.8 mΩ paralleled, plus ~3 mΩ leads; the combined-vs-sum offset
  has run 98.6–98.7% twice).
- **Peak current** (√C-scaled from the measured 298 A at 3 cans, the
  nonlinearities left free): ≈ **345 A at 4 cans, ≈ 385 A at 5** — capture
  and check.
- **Fire position**: the impulse peak stays at **x = −13.5 ± 1 mm** at every
  can count; `fireoffset +9` remains correct.
- **Monotonicity**: dv strictly increasing with n at fixed gate and voltage.

## Safety check before the first 4/5-can shot (not a prediction — a gate)

- Q1–Q3: three IRFP4668 share ~385 A worst case → ~128 A each, inside the
  130 A/leg continuous rating with pulse margin. OK, but verify the sharing
  assumption has not changed (heatsink, gate resistors intact).
- D1 (MBR60100) sees the full cut current as a surge — up to ~350 A at a
  400 µs gate at 5 cans. Fine for single-shot surge ratings; **do not run
  `sustain` at 4–5 cans until the D1 duty case is worked**.
- Discharge stick: ~13 J (4 cans) / ~16 J (5) — proportionally longer bleeds.

## Declared meanings of misses

| observation | meaning |
|---|---|
| Layer A misses at all gates uniformly | field amplitude wrong at higher I — PINN nonlinearity or force law |
| Layer A misses only at short gates | τ does not transfer across can count — tail physics is C-dependent |
| Layer B beaten by MORE than ~30% | droop trend broke, or a new conduction path at higher current |
| measured BELOW Layer B | unmodelled loss mechanism — stop and find it before trusting anything |
| combined LCR outside Layer C band | a can or joint problem — the can-2 lesson, apply pre-solder data |
| impulse peak moved > 1 mm | field geometry changes with can count — should be impossible; suspect the bench first |

---

# Addendum (preregistered 2026-09-06, before measurement): 3-can closeout tests

Frozen-model predictions for the two held-out 3-can axes, committed before
the data. These close the 3-can chapter before can 4 makes it unrepeatable.

## Fire-position curve at 3 cans, 700 us, 49 V

Predicted dv RELATIVE to the +9 offset point (ratios cancel the circuit
exactly -- same I(t) at every offset -- so this is a PURE field-shape test,
immune to every circuit issue the scope found):

| offset | centre x (mm) | dv / dv(+9) |
|---:|---:|:---:|
| +3  | -19.78 | 0.643 |
| +6  | -16.78 | 0.885 |
| +9  | -13.78 | 1.000 |
| +12 | -10.78 | 0.910 |
| +15 |  -7.78 | 0.683 |

Peak stays at x = -13.5 +/- 1 mm. Band: +/-0.06 on each ratio (n=2 pairs
per point). A shifted or asymmetric-beyond-band curve means the interior
field shape changed with can count -- which should be impossible and would
indict the bench (coil moved?) before the model.

## 30 V marble point at 3 cans, 700 us, offset +9

Layer-A style (decisive): given a 30 V blank scope capture, the injected
prediction must match the paired-sweep dv within +/-6%. Frozen-sim number
for reference (known-weak circuit, expected low): dv = 66.9 mm/s at
--voltage 30 -- scale with (v_pre/30)^2 for the actual charge reached.

---

# Scoring (2026-09-09): 3-can fire-position curve -- a MISS on the entry side

Data: `logs/3can_firepos.csv`, 700 us, 49 V, 2 pairs per offset, v_in 0.232-0.255.

| offset | x (mm) | coil dv pair (mm/s) | mean | measured ratio | predicted | verdict |
|---:|---:|:---:|:---:|:---:|:---:|:---|
| +3  | -19.78 | 183, 187 | 185 | **1.016** | 0.643 | MISS (+0.37) |
| +6  | -16.78 | 203, 183 | 193 | **1.060** | 0.885 | MISS (+0.18) |
| +9  | -13.78 | 182, 182 | 182 | 1.000 | 1.000 | reference |
| +12 | -10.78 | 184, 136 | 160 | 0.879 | 0.910 | hit (band +/-0.06) |
| +15 |  -7.78 | 128, 118 | 123 | 0.676 | 0.683 | hit |

The exit side is exactly as predicted. The entry side is FLAT from +3 to +9
where the model falls to 0.64: the measured curve is roughly 1.5x wider than
the model's on the entry side, with its apparent peak near +6 (x ~ -16.8).
The +9 point itself (182 mm/s) is where the 3-can campaign always put it,
so nothing about the reference moved; the flank came UP to meet it.

Declared meaning (above): "impulse peak moved > 1 mm -- should be
impossible; suspect the bench first". Honoured: the bench is suspect first.
But two facts narrow it:

1. The frozen model's curve shape is independent of current and gate --
   verified by running it at 3 can/49 V/200 us (I_pk 254 A), 3 can/30 V/
   700 us (166 A) and 1 can/49 V/700 us (191 A): 0.64/0.88/1.00/0.91/0.68
   every time. Its saturation cap (B_sat 1.8 T, chi_eff 3) never engages:
   3*Bz at the fire point is 0.49 T at 272 A. Lowering B_sat to 0.35 T moves
   the shape toward the data (0.78/1.02/1/0.80/0.55) but cannot reproduce it.
2. The 1-can 49 V/200 us sweep (2026-08-25, same firmware sensing
   convention as today -- the centre-timing fix f340baf predates every
   sweep) DID show the model's entry-side steepness: offsets 0/+8 gave
   ~18/50 = 0.36 against the model's 0.40. So the entry side was steep at
   191 A and short pulses, and is flat at 272 A and long pulses.

## Preregistered discriminator (before the data): dv(+3)/dv(+9) at 3 cans

Two more pairs of points, ratio of coil dv at +3 to +9, 2-3 pairs each:

| run | condition | I_pk (model) | pulse | frozen model | if BENCH position error | if CURRENT-amplitude physics (saturation-like) | if PULSE-DURATION physics (eddy / magnetisation lag) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| A | 49 V, 200 us | 254 A | short | 0.64 +/- 0.06 | ~1.0 (unchanged) | intermediate, ~0.8-0.95 | drops toward 0.64 |
| B | 30 V, 700 us | 166 A | long | 0.64 +/- 0.06 | ~1.0 (unchanged) | drops toward 0.64-0.75 | stays ~1.0 |

Reference already in hand: 49 V / 700 us (272 A, long) = 1.02.
The frozen model has already lost this axis; the table is about WHICH
replacement is true. If both A and B stay at ~1.0, the bench is guilty and
the next step is an independent fire-position witness (station-B arrival
time relative to the gate edge gives x_fire to ~1 mm) before any physics
is touched. Run B also supplies the 30 V +9 point preregistered above
(within +/-6% of the injected prediction), unchanged.

Rule 2 note: nothing has been refitted. The 4/5-can Layer A/B/C numbers
above stand exactly as frozen; any new field/force term fitted to this
3-can curve gets its own 4/5-can predictions registered ALONGSIDE them,
before can 4 data, never replacing them.
