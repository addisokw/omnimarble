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

## 30 V injected prediction (committed 2026-09-09 before run B's marble data)

Scope capture `scope_3can_30v_700.csv`: V_bank 29.45 -> 12.08 V, I_pk 187 A
at +184 us (sqrt-free voltage scaling of the 49 V capture said ~182 A;
I_pk/V0 is 6.35 A/V here against 6.08 A/V at 49 V -- noted, not fitted).

Injected-current prediction (frozen field + tau = 275 us), v_in 0.245:

| offset | dv (mm/s) |
|---:|:---:|
| +9 | **69.6** -- must match the 3-pair mean within +/-6% (65.4-73.8) |
| +3 | 44.7 -- frozen shape, ratio 0.64; the discriminator says whether this holds |

Frozen-circuit reference at 29.45 V: 66.9 x (29.45/30)^2 = 64.5 mm/s.

## Scoring the discriminator (2026-09-09)

Coil dv below is v_in-corrected within each pair (baseline slope
-490 mm/s per m/s); raw values in the CSVs. The correction tightens the
30 V +3 triple from 38/63/56 to 53/55/44 and barely moves the 49 V points.

| run | condition | I_pk | dv(+3) | dv(+9) | ratio | frozen | bench | current-amp | pulse-duration |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ref | 49 V, 700 us | 298 A | 178 | 180 | **0.99** | 0.64 | ~1.0 | -- | -- |
| A | 49 V, 200 us | ~250 A | 47 | 49 | 0.84 raw / 0.96 corr (+/-0.15) | 0.64 | ~1.0 | 0.8-0.95 | 0.64 |
| B | 30 V, 700 us | 187 A | 50.4 | 73.3 | **0.69** (+/-0.08) | 0.64 | ~1.0 | 0.64-0.75 | ~1.0 |

**Run B decides it: the bench is exonerated.** A mis-timed shot would have
given ~1.0 at 30 V; it gave 0.69, inside the frozen band. Pulse-duration
physics is out (B is a long pulse and it came back to the model). The
anomaly is tied to the STRENGTH of the shot.

30 V +9 point (preregistered 69.6, band 65.4-73.8): measured 73.3 -- HIT,
at the top of the band. So at +9 the frozen model holds at both 187 A and
298 A (ratio 1.05 / 1.03); the +9 reference is not what moved.

The entry point's excess over the frozen model, by shot:

| condition | dv(+3) measured | frozen (injected where available) | excess |
|---|:---:|:---:|:---:|
| 30 V, 700 us (187 A, dv_9 = 73) | 50.4 | 44.7 | 1.13 |
| 49 V, 200 us (~250 A, dv_9 = 49) | 47 | 41 | 1.14 |
| 49 V, 700 us (298 A, dv_9 = 180) | 178 | 114 | 1.56 |

Rejected mechanisms (worked, not guessed):
- any local M(B) law (soft or hard saturation, initial-permeability rise):
  the internal field at +3/49 V (3*Bz = 0.32 T) equals that at +9/30 V
  (0.31 T), yet one shows a 56% excess and the other 5%. The excess is not
  a function of the field the ball sits in.
- freewheel tail longer than modelled: the 30 V capture shows I = 0 +/- 10 A
  from 0.9 ms on. No long tail exists to move the ball into a better spot.
- remanence (force linear in I): would be LARGER at low current; it is smaller.
- ball motion during the pulse: 0.3 mm, worth ~2% on the flank.
- fire-timing latency: current-independent, and run B is at the model.

Open: the 49 V/200 us point cannot separate "peak current" from "impulse
delivered" (it has the higher I_pk but the lower dv, and its ratio is
ambiguous at +/-0.15). The next two runs are built to separate them.

## Preregistered next runs (before data)

All at 3 cans, 49 V, offsets +3 and +9, 2 pairs each, v_in-corrected ratio:

| run | gate | I_pk | frozen | if PEAK-CURRENT driven | if IMPULSE/DURATION driven |
|---|:---:|:---:|:---:|:---:|:---:|
| C | 400 us | ~290 A (peak inside the gate) | 0.64 | ~1.0 (same as 700) | ~0.85 (dv_9 ~ 130, between B and ref) |
| D | 1500 us | 298 A | 0.64 | ~1.0 | > 1.05 (dv_9 ~ 235; +3 overtakes +9) |

Frozen absolute dv at +3/+9: 400 us 84/132, 1500 us 133/206 mm/s (fallback
circuit; the injected 49 V captures at 400/1500 exist and give ~1.03x).

Run E, entry extension at 700 us (offsets -6, -3, 0; 2 pairs each), to map
how far the plateau reaches: frozen ratios to +9 are 0.139 / 0.232 / 0.400.
A plateau that persists to the face (0 -> ~1.0) would mean the extra force
acts on a ball that is mostly OUTSIDE the winding, where the model's field
is weakest -- which would point at the coil's exterior field (leads, the
former end, anything ferromagnetic on the entry side) rather than the ball.

Unchanged: nothing refitted, 4/5-can predictions stand.

## Scoring runs C, D, E (2026-09-09; v_in-corrected)

| run | condition | dv(+3) | dv(+9) | ratio | frozen | peak-current | impulse |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| C | 49 V, 400 us | 111.7 | 137.3 | **0.81** | 0.64 | ~1.0 | ~0.85 |
| D | 49 V, 1500 us | 201.2 | 227.6 | **0.88** | 0.64 | ~1.0 | > 1.05 |

The IMPULSE hypothesis is falsified: D was declared to overtake (+3 > +9)
and it did not. Peak-current survives only loosely -- both runs sit below
the ~1.0 it called for, but far above the frozen 0.64. Taking every 49 V
gate together (200: 0.84-0.96, 400: 0.81, 700: 0.99, 1500: 0.88) the ratio
at 49 V is 0.89 +/- 0.04 with no clear gate trend; at 30 V it is 0.69.
**Bank voltage (peak current) is the variable; gate length is at most a
weak modifier.**

Entry extension (run E, 49 V, 700 us), ratios to dv(+9) = 180:

| offset | x | measured | frozen | verdict |
|---:|---:|:---:|:---:|:---|
| -6 | -28.78 | 0.15 | 0.139 | hit |
| -3 | -25.78 | 0.31 | 0.232 | high, ~1 sigma |
| 0 | -22.78 | 0.70 | 0.400 | MISS (+0.30) |

So the full 49 V / 700 us curve, measured vs frozen:

| offset | -6 | -3 | 0 | +3 | +6 | +9 | +12 | +15 |
|---|---|---|---|---|---|---|---|---|
| measured | 0.15 | 0.31 | 0.70 | 0.99 | 1.06 | 1.00 | 0.88 | 0.68 |
| frozen | 0.14 | 0.23 | 0.40 | 0.64 | 0.885 | 1.00 | 0.91 | 0.68 |

Far outside the coil (-6) and on the exit flank (+12, +15) the model is
right. Between the face and the peak the measured curve is flat-topped
where the model is peaked: half-maximum points are at about -1.6 and +17.6
(measured) against +1.2 and +17.4 (frozen) -- the ENTRY half-maximum moved
~3 mm outward, the exit one did not. At 30 V the +3/+9 ratio is the
model's, so this outward growth of the entry flank is a high-current
effect. Excess at +3 over the frozen model by run: 30 V 1.13, 49 V/400
1.33, 49 V/700 1.56, 49 V/1500 1.52; at +9 the same runs give 1.05, 1.04,
1.03, 1.11.

Standing candidates (none confirmed):
- a genuine nonlinear force term that acts when the ball STRADDLES the
  winding end (half the ball outside the face) and grows faster than I^2 --
  no local M(B) law does this (see above), so it would have to involve the
  non-uniform magnetisation of a sphere sitting across a steep gradient;
- something mechanical at the entry that scales with current: the radial
  pull on a ball resting on the former's entry lip;
- ball state (remanence / a magnetised ball) -- disfavoured by the current
  scaling but not yet excluded by a direct swap.

## Preregistered next runs (before data): voltage series, fresh ball, fast release

All at 3 cans, 700 us, offsets +3 and +9, 2 pairs each, v_in-corrected ratio.

| run | condition | frozen | if peak-current physics | if ball-state | if mechanical / speed-dependent |
|---|---|:---:|:---:|:---:|:---:|
| F | 40 V | 0.64 | ~0.85 (between 0.69 and 0.99, monotonic in V) | same as F-physics | same |
| G | 20 V | 0.64 | <= 0.69 (at or below the model) | same | same |
| H | 49 V, FRESH ball (same spec, never pulsed) | 0.64 | 0.99 (unchanged) | drops toward 0.64 | 0.99 |
| J | 49 V, fast release (v_in ~0.45-0.5, the sustain height) | 0.64 | 0.99 (a magnetic force does not care about v) | 0.99 | changes (either way) |

Absolute +9 checks ride along: 40 V frozen-circuit reference 66.9 x
(40/30)^2 = 118.9 mm/s, 20 V: 29.7 mm/s (scale by (v_pre/V)^2); a scope
capture at 40 V would give the injected number (preferred) -- optional.

F+G also pin the FORM: excess(+3) - 1 against I_pk at 187 / ~250 / 298 A
(and ~125 A) tells I^1 from I^2 from a threshold.

## Scoring runs F, G, J (2026-09-09; H skipped -- no spare ball on hand)

| run | condition | dv(+3) | dv(+9) | ratio | frozen | declared physics reading |
|---|---|:---:|:---:|:---:|:---:|:---|
| F | 40 V, 700 us | 101.9 | 133.2 | **0.77** (+/-0.08) | 0.64 | ~0.85 -- consistent, monotonic in V |
| G | 20 V, 700 us | 22.2 | 29.9 | 0.74 (+/-0.35) | 0.64 | <= 0.69 -- uninformative at this dv |
| J | 49 V, fast release (v_in 0.79!) | 158.7 | 159.7 (one pair) | **0.99** | 0.64 | 0.99 = velocity-independent |

The ratio against bank voltage / peak current, 700 us, all v_in-corrected:

| V | I_pk | dv(+3)/dv(+9) | excess at +3 over frozen shape |
|---|:---:|:---:|:---:|
| 20 | ~125 A | 0.74 +/- 0.35 | -- |
| 30 | 187 A | 0.69 +/- 0.08 | 1.08 |
| 40 | ~250 A | 0.77 +/- 0.08 | 1.20 |
| 49 | 298 A | 0.99 +/- 0.05 | 1.55 |

A steep onset between ~250 and 300 A, not a gentle power law; run J says it
does not care about the ball's speed (0.245 -> 0.79 m/s, same ratio).

Aside from J: dv(+9) at v_in 0.79 was 160 against 180 at 0.245 -- the rig's
known steep speed-dependent loss (baseline -113 mm/s at 0.79 vs -50 at
0.245), unrelated to the entry anomaly but a reminder that the pairing
only cancels loss at the RELEASE speed, not at the exit speed.

## Sim-side check: can any local M(B) law do this? (scratch, not committed code)

Volume-integrated (B.grad)B over the real 12.7 mm ball with several
magnetisation laws, ratios to +9, at 298 A:

| law | -6 | -3 | 0 | +3 | +6 | +9 | +12 | +15 | F(+9) vs linear |
|---|---|---|---|---|---|---|---|---|:---:|
| measured 49 V | 0.15 | 0.31 | 0.70 | 0.99 | 1.06 | 1.00 | 0.88 | 0.68 | -- |
| linear chi 3 | 0.14 | 0.23 | 0.39 | 0.64 | 0.88 | 1.00 | 0.92 | 0.69 | 1.00 |
| hard cap 0.3 T | 0.24 | 0.40 | 0.64 | 0.90 | 1.04 | 1.00 | 0.80 | 0.55 | 0.56 |
| soft tanh 0.4 T | 0.20 | 0.32 | 0.51 | 0.76 | 0.96 | 1.00 | 0.85 | 0.60 | 0.64 |
| rising chi 1->3 | 0.10 | 0.18 | 0.33 | 0.58 | 0.85 | 1.00 | 0.94 | 0.72 | 0.89 |

Saturation DOES move the entry side toward the data (by depressing the +9
reference, whose whole volume sits in strong field) -- but only by cutting
F(+9) to 0.56-0.64 of linear at 298 A against 0.81-0.87 at 187 A, i.e. the
+9 point would scale as I^1.1 between 30 V and 49 V. Measured: 73.3 -> 180
= 2.46x against I^2 = 2.54x. **The +9 point scales quadratically to
+/-5%, so any saturation big enough to flatten the entry is excluded.**
Saturation also drops the exit flank (0.80/0.55), which measured at the
linear values (0.88/0.68). Rising-permeability laws go the wrong way.
No monotone local M(B) law fits; the earlier point-field argument reached
the same conclusion for the wrong reason, this one is the real constraint.

## Where this stands

Established: a current-dependent excess with a steep onset above ~250 A,
confined to fire positions where the ball straddles the winding end
(offsets 0 to +6), invisible at +9 and on the exit flank, independent of
gate length and of ball speed, absent at 30 V. Excluded: fire timing,
freewheel tail, remanence, ball motion, every local magnetisation law.

Not yet excluded: ball magnetic history (run H, needs a never-pulsed
ball); something mechanical at the former's entry lip that scales with the
radial pull; an asymmetry of the hardware between the two coil ends
(leads, mounting) that only shows at high current.

Cheap test queued for the next dead-rig window (the resistor swap):
**swap the coil leads at J1** and repeat +3/+9 at 49 V, 2 pairs. Reversing
the field flips the sign of anything that depends on a fixed magnetisation
(ball or hardware) but leaves induced-force physics untouched.
Preregistered: ratio 0.99 unchanged = not remanence of anything; ratio
drops toward 0.64 on the first shots and recovers = ball/hardware remanence.

For the 4-can holdout this changes ONE preregistered line: "impulse peak
stays at -13.5 +/- 1 mm" is now expected to FAIL the same way at 4 cans
(345 A), with a wider plateau: dv(+3)/dv(+9) >= 1.0 and dv(0)/dv(+9) >= 0.7
at 49 V / 700 us. The +9 point, which every Layer A/B number rests on,
has held at every gate and voltage tested and those numbers stand.

## Scoring the polarity flip (2026-09-09, coil leads reversed at J1)

Scope null test first: reversed-coil 49 V / 700 us blank capture is the
baseline waveform (I_pk 296 vs 298 A, bank 49.28 -> 20.20 vs 49.15 -> 20.19,
integral of I^2 39.02 vs 38.98 A^2 s, 12 A rms sample-by-sample). The flip
did nothing electrical.

Marble, v_in-corrected: dv(+3) = 146.5 / 124.7 (mean 135.6), dv(+9) = 186.9 /
192.5 (mean 189.7). **Ratio 0.72** against 0.99 with the original
polarity and the frozen 0.64. The +9 point is unchanged (190 vs 180, within
the pair scatter). The entry excess is GONE with the field reversed, and it
did not return between the first and second +3 shot (147 then 125).

What this establishes: the excess depends on the SIGN of the coil field,
so it involves a persistent magnetisation that the coil did not flip --
induced-force physics is sign-blind (the null capture above is the
electrical half of that statement). Two carriers remain:

- **the ball's remanent state** (hundreds of same-polarity pulses). Against
  it: the ball rolls ~5 turns between stations and is handled by hand between
  shots, so a remanent moment fixed in the ball would point in a random
  direction at each shot and the excess would be random -- it was 183/187
  on consecutive +3 shots. Also against it: the second reversed shot did
  not recover, though two shots may be too few for a semi-hard ball on
  minor loops.
- **a magnetised ferrous part fixed in the lab frame on the entry side**
  (screw, bracket, sensor mount, former clamp). Consistent with the
  shot-to-shot repeatability and the entry-only footprint. Still
  unexplained by it alone: the steep current dependence (a fixed field
  gives a cross term linear in I, which should be relatively LARGER at
  30 V; measured smaller). A fixed field plus a nonlinear ball response
  near its coercive point could do both; not demonstrated.

## Preregistered next tests (before data)

| test | ball-state predicts | fixed-hardware predicts |
|---|---|---|
| K: 8 conditioning shots reversed at +9, THEN +3/+9 pairs (reversed) | ratio climbs back toward 0.99 as the ball re-magnetises | stays ~0.72 |
| L: flip leads back to original, IMMEDIATELY +3/+9 pairs | stays low (~0.7) until re-conditioned | 0.99 immediately |
| M: static field survey at the entry face, coil OFF (compass, or the SS49E/MLX90393 when they arrive -- the flip protocol's "current off" reading, read for its own sake) | nothing above Earth's ~0.5 G | a local field of gauss at the offending part |

K and L are mutually checking: they cannot both say "ball" or both say
"hardware" unless the carrier is what they name.

## Scoring K (reversed, after 8 conditioning shots at +9)

Conditioning +9 raw dv: 201 / 178 / 185 / 206. After: dv(+3) = 132 / 128,
dv(+9) = 199 / 178 (v_in-corrected). **Ratio 0.69** -- unchanged from the
first reversed pairs (0.72). Eight-plus reversed pulses with the ball in the
coil did not bring the excess back, so the ball's magnetic state is NOT the
carrier. The carrier is fixed in the lab frame on the entry side of the
coil. L (flip back, 0.99 expected immediately) closes it; M (coil-off field
survey at the entry face) locates it.

Consequence for the model: this is a RIG artefact -- a static field from a
magnetised part -- not a failure of the field/force physics. Frozen
predictions are unchanged; the 4-can "peak stays put" line reverts to a
real prediction once the part is demagnetised or removed and the +3/+9
ratio is re-measured at 49 V (expected 0.64 +/- 0.06 after the fix).

## Scoring L (leads back to original, immediate pairs)

dv(+3) = 169 / 143.5, dv(+9) = 199 / 194 (v_in-corrected). **Ratio 0.80
+/- 0.08** -- between the two declared outcomes (hardware 0.99, ball 0.7).
First +3 shot 0.86, second 0.73. Read together with K (ball excluded), the
carrier is fixed hardware that the ten reversed pulses PARTLY demagnetised
or reversed: sign restored by the flip-back, magnitude reduced. That places
it inside the coil's exterior field, i.e. within a few cm of the entry
face, and it is semi-hard (a reversed pulse train moves it, one does not).

Preregistered N: 8 conditioning shots at +9 with the ORIGINAL polarity,
then +3/+9 pairs. Re-magnetisable part: ratio climbs back toward 0.99.
Permanently weakened part: stays ~0.80 (the rig then simply has a smaller
artefact). Either way the fix is the same: locate (coil-off survey at the
entry face) and remove or demagnetise, then confirm 0.64 +/- 0.06.

## N, conditioning half only (original polarity, 4 pairs at +9)

dv(+9) v_in-corrected: 204 / 185 / 197 / 204, mean 197. No +3 pairs were
taken, so N's ratio is not scored. What the free half does show is the +9
reference drifting UP through the session at fixed 49 V / 700 us:

| time | dv(+9) | context |
|---|:---:|---|
| morning | 180 | fire-position sweep |
| midday | 190, 189 | reversed polarity, before/after conditioning |
| afternoon | 196 | flip-back |
| late | 197 | conditioning |

+9% monotonic in time, unaffected by polarity. Candidates: bank warming
(electrolytic C rises and ESR falls with temperature -> more current, and
dv goes as I^2), coil warming (opposes), track/ball. One blank scope
capture at the end of the session against the morning baseline (integral
of I^2 = 38.98 A^2 s) attributes it without a single release. Until then,
same-session ratios are trustworthy and cross-session absolute dv carries
a ~5-10% thermal band -- which is also the band the 4-can Layer A/B
absolute predictions should be read with.

End-of-session blank capture (49 V, 700 us) against the morning baseline:
integral of I^2 38.59 vs 38.98 A^2 s (-1%), I_pk 294 vs 298 A, bank
49.29 -> 20.21 vs 49.15 -> 20.19. **The circuit did not drift.** The +9%
rise in dv(+9) is downstream of the current: ball, track, release or the
sensing chain -- and the roll baselines did NOT drift (-44..-52 all day),
so it is specific to the shot. Open. First measurement of the next session:
one +9 pair set cold, before anything else. Back at ~180 = something warms
with use; still ~197 = something changed permanently today (the reversed
pulse train is the only candidate event).

Correction: the "morning baseline" above is the 2026-08-29 capture, eleven
days earlier -- not the same session. Restated: the pulse is unchanged
across eleven days (-1% in the I^2 integral), and within today the
mid-session reversed-coil capture (39.02) and the end-of-day capture
(38.59) bracket the afternoon, over which dv(+9) rose 190 -> 197 while the
pulse fell 1%. So the afternoon drift is not the circuit. The morning ->
midday step (180 -> 190) has no same-day scope reference and is not
attributed. The cold +9 pair set next session stands as the first test.

## Status 2026-09-13: entry-side anomaly DEFERRED

Decision: no further releases on the entry flank. Every 4/5-can number
above rests on the +9 point, which held at every gate, voltage and
polarity; the anomaly is confined to fire positions the rig does not use.
The entry rows in the scoring sections above stand as "artefact,
unresolved". It resumes when the ball-tip wand (LCR dL(x) survey) or the
Hall sensors exist, or if the 4-can +9 point misses its band. Next session
takes only the cold +9 triple and the sham-flip control folded into the
resistor swap: `omnimarble-vbench/docs/TEST_PLAN_3CAN_RETURN.md`.

## Scoring 2026-09-24: cold reference and the untouched-rig ratio

Rig untouched since 09-09 (original polarity, 100 ohm resistor, leads not
handled). First six releases of the day, +9 / 49 V / 700 us: coil dv 212 /
182 / 198 raw, mean **197**. Shots 7-12 (a second +9 triple) gave 194 raw
/ 203 v_in-corrected. So the rig does NOT wake up at 180: the 09-09
morning value was the odd session, and **197-203 is the reference** the
4/5-can Layer A/B rows are read against, with a +/-5% band and no
warm-up term. The 180 stays on file, unexplained.

Then, still untouched, +3 triple: 155 / 142 / 147 corrected (154 / 134 /
160 raw). Ratio dv(+3)/dv(+9) = **0.73 corrected, 0.77 raw**, against
0.90-0.99 on 09-09 before the polarity flips and 0.80 right after the
flip-back. The entry excess shrank with NOTHING done to the rig in
fifteen days. This retires the sham-flip test (its 0.9 baseline is gone)
and it is the strongest evidence yet that the excess is not physics of
the coil-ball system: it is consistent with a semi-hard magnetised part
that the 09-09 reversed pulse train partly demagnetised and that has
not recovered. Deferred as before; nothing refitted.

Marble-shot capture (first ever, +9, shots 7-9 window): I_pk 295 A, cut
at 705 us, integral 38.2 A^2 s shunt-visible -- the August pulse to within
noise after fifteen days. The ball's inductance signature (early slope
~4% low) is inside the blank-to-blank slope spread (1.43-1.71 A/us) and
not resolvable from one capture; I(100 us) = 148 A against 169-183 for
every blank is the right sign at ~2 sigma. Needs a same-day blank.

22 ohm resistor in (09-24). Blank: I_pk 299 A at 49.77 V, cut 705 us,
integral 38.8 A^2 s normalised to 49.15 V (August 39.0) -- inert to the
pulse. +9 triple after the swap: 183 / 209 / 204 raw (mean 199), 200 / 192
/ 184 v_in-corrected (mean 192); predicted 197 +/- 8 -- HIT both ways
(two pairs carry >0.03 m/s release mismatch, hence the raw/corrected gap). Marble-shot
vs same-day blank: early slope -4.4% (predicted -4%), I(100 us) -9%. The
3-can chapter is closed; can 4 next, read against 192-199.

## Layer C scoring, can 4 (2026-09-24)

| item | predicted | measured | verdict |
|---|:---:|:---:|---|
| can 4 standalone, 100 Hz | 1880-1920 uF / 45-60 mohm | 1930.7 uF / 44.9 mohm | miss on both by 0.6% / 0.1 mohm, healthy direction; soldered |
| combined at J2 | 7360-7470 uF / 15 +/- 2 mohm | 7392.0 uF / 15.5 mohm | HIT / HIT |

Combined/sum 0.983 (0.986-0.987 before). Reference for the marble rows:
dv(+9, 3 cans, 49 V, 700 us) = 192-199 measured today.

## 4-can captures (2026-09-24) and the Layer A numbers, committed before the marble data

Blanks at 49.8 V, model-free extraction (`scripts` method as at 3 cans):

| gate | I_pk | cut | I(cut) | bank after | C from charge balance | L (initial slope) | R at peak |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 400 us | 318 A @ 392 us | 408 us | 306 A | 36.8 V | 6735 uF | 26 uH | 105 mohm |
| 700 us | 316 A @ 395 us | 712 us | 230 A | 26.7 V | 7137 uF | -- | 106 mohm |
| 1500 us | 320 A @ 385 us | 1505 us | 92 A | 11.5 V | 7443 uF | -- | 106 mohm |

Layer C peak-current line: predicted ~345 A by sqrt(C) scaling from 298 A;
measured **318 A -- MISS by -8%**. The peak did not scale as sqrt(C): the
loop is resistance-limited enough that adding capacitance mostly lengthens
the pulse (peak at 390 us vs 305 us) rather than raising it. Declared
meaning: none was assigned to this line; recorded as a plain miss of the
scaling assumption, not of the model (the model never predicted 345 -- its
fallback circuit said ~310, and the injected path does not care).

Charge-balance C climbs 6735 -> 7443 uF with gate length, as at 3 cans
(5229 at 700 us against a 100 Hz 5586); the 4-can pulse C at long gates
is ~0.96-1.0 of the 100 Hz 7392 uF, continuing the sub-linear droop
trend (0.859 / 0.931 / 0.958 / ~0.97).

Layer B blanks (V(t_on)/V0 at 49 V): measured 0.960 / 0.895 / 0.813 /
0.661 / 0.534 / 0.390 / 0.242 against frozen 0.966 / 0.891 / 0.801 /
0.620 / 0.468 / 0.300 / 0.142. Miss in the declared direction (fallback
circuit under-retains) but twice the "up to 0.05" allowed at long gates.
The fallback's n x 1640 = 6560 uF against a real ~7100-7400 is the cause.

**Layer A, the decisive claim -- injected-current predictions, frozen field
and tau, v_in 0.245, from these captures:**

| gate | injected dv (mm/s) | pass band (+/-5%) | frozen-circuit row (scaled to 49.8 V) |
|---|:---:|:---:|:---:|
| 400 us | **146.2** | 138.9-153.5 | 164 |
| 700 us | **232.6** | 221.0-244.2 | 237 |
| 1000 us | (no capture) | -- | 268 |
| 1500 us | **301.4** | 286.3-316.5 | 285 |

Reference from 3 cans today: 192-199 at 700 us. The 700 us prediction is a
+17-21% step from 3 to 4 cans.

## Layer A and B SCORED, 4 cans, 2026-09-24 (raw coil dv; v_in not logged by the sweep printout)

The 1000 us capture arrived after the table above: injected prediction
**272.5 mm/s** (band 258.9-286.1; I_pk 319 A, C 7235 uF).

| gate | pairs (coil dv, mm/s) | mean | injected | ratio | Layer A (+/-5%) | frozen circuit | vs frozen |
|---|---|:---:|:---:|:---:|---|:---:|:---:|
| 400 | 172 / 133 / 147 | 150.7 | 146.2 | 1.03 | **HIT** | 164 | 0.92 |
| 700 | 229 / 225 / 222 | 225.3 | 232.6 | 0.97 | **HIT** | 237 | 0.95 |
| 1000 | 287 / 242 / (lost) + 292 / 273 / 279 | 274.6 | 272.5 | 1.01 | **HIT** | 268 | 1.02 |
| 1500 | 314 / 326 / 364 | 334.7 | 301.4 | 1.11 | **MISS** (+11%; 1.06 without the 364 pair) | 285 | 1.17 |

Layer A: three hits and one miss, at the LONGEST gate. The declared-miss
table only covered "all gates" and "short gates"; a long-gate-only miss
was not anticipated. At 3 cans the 1500 us ratio was 1.04, so this is
new with the fourth can. The 1500 us triple is also the noisiest (spread
21; the third pair's baseline was -63 against -46/-53) -- with the
outlier dropped the ratio is 1.06, one point past the band. Recorded as
a miss; not explained. Candidates to test, not fit: a v_in effect (the
sweep's v_in column is in the CSV -- 3 of 6 rows survive the overwrite),
the entry-side artefact now reaching +9 at the longer pulse, or the tail
after a 1500 us gate at 4 cans (I(cut) = 92 A -- small, unlikely).

Layer B: the frozen-circuit rows were declared to under-predict by
15-30%. Measured/frozen = 0.92 / 0.95 / 1.02 / 1.17 -- the fallback
circuit is CLOSER than declared at short gates and only reaches the
declared excess at 1500 us. Not a hit on the declaration either way;
the fallback circuit is retired in favour of the injected path.

Monotonicity (Layer C): dv strictly increasing with n at every gate
(3-can 400/700/1000/1500 ~ 141/197/214/239 -> 151/225/275/335). HIT.

Bookkeeping: the first sweep run printed "board confirms +0.00 mm" for
the 400 us block while the second printed "+9.00"; the 400 us dv (151
against 146 predicted at +9, ~100 expected at offset 0) says the shots
went at +9 and the echo was stale. The second run overwrote
`logs/4can_9.csv` (same --out); the first run's 11 pairs survive only in
the terminal transcript recorded here.

Re-run and v_in-corrected (the correction slope, -490 mm/s per m/s, was
measured at 3 cans / 700 us and is carried over unverified at 4 cans):

| gate | pairs raw | corrected | injected | ratio raw / corrected | verdict |
|---|---|:---:|:---:|:---:|---|
| 1000 | 292 / 273 / 279 | 291 / 285 / 288, mean 288 | 272.5 | 1.03 / 1.06 | edge |
| 1500 repeat | 311 / 337 / 287 | 342 / 329 / 296, mean 322 | 301.4 | 1.03 / 1.07 | edge/miss |
| 1500 all six raw | 314 326 364 311 337 287 | -- | 301.4 | 1.07 | miss |

Final Layer A tally at 4 cans: 400 and 700 us hits (1.03, 0.97); 1000
and 1500 us sit +3 to +7% high depending on the correction, straddling
the +5% edge. The long-gate excess is real but small and is not the
"tail does not transfer" signature (that was declared for SHORT gates).
Nothing fitted. Note for 5 cans: expect the same +5% at 1000/1500 us;
if it grows with n it is current-dependent physics (the entry-side
artefact reaching +9, or the ball's eddy response, both on the books);
if it stays at +5% it is a constant to look for in the timing or the
correction slope.

## D1 duty case at 4 cans (2026-09-24) -- the gate for `sustain`

From the 4-can captures, not the pre-solder guesses: the freewheel current
D1 (MBR60100, 60 A average, 100 V Schottky) must carry is I(cut) = 306 A
at a 400 us gate, 230 A at 700 us, 170 A at 1000 us, 92 A at 1500 us,
decaying with tau ~275 us. Per shot at 700 us: charge ~0.063 C, energy
in the diode ~V_f x Q ~ 0.05-0.08 J, pulse width ~0.3 ms. The
non-repetitive surge rating of this class of part is 400-600 A for
8.3 ms; a 230-306 A pulse of 0.3 ms is far inside it. Repetitive duty in
sustain is at most ~2 shots/s for 10 shots: average dissipation under
0.2 W, transient junction rise per pulse of order 10-15 K on a TO-247
die. Every single-shot sweep today (dozens of shots at ~1 per 3 s) was
already this duty. The FETs: all three (Q1-Q3) are fitted, as the
earlier safety line assumed (a sensor-study doc claimed one; corrected).
318 A shared three ways is ~106 A per leg for 0.7 ms, inside the
IRFP4668 continuous rating, let alone pulsed.

**Gate lifted for 4 cans**, with the same budget as at 3: 700 us
(I(cut) 230 A rather than 306 A at 400 us), 10 shots, 60 s. Bank energy
9.1 J at 49.6 V -- the discharge stick needs proportionally longer.
5 cans re-does this case from its own captures.

Prediction for the first 4-can sustain run, same release height as at
3 cans: forward kick coil dv ~0.23 m/s (vs 0.20), return kick at the +12
trim ~0.08 (vs 0.06); the far-ramp and entry-ramp losses do not change.
Expect 3 full cycles, maybe 4, before the marble parks -- one more than
at 3 cans, not indefinite. Indefinite needs the return kick fixed or the
losses cut, not more cans.

## Sustain at 4 cans (2026-09-24): the first steady state

`sustain 700 10 30`, offset +9, return trim +12: five cycles, v_in at A
0.802 / 0.245 / 0.161 / 0.163 / 0.165 -- a limit cycle at 0.16 m/s,
ended by the shot budget with the marble still going. Forward kicks
+0.19-0.20 raw, return kicks +0.07-0.08 (predicted 0.23 / 0.08 coil dv:
hit); "three cycles, maybe four" was wrong -- the losses fall faster
than the kicks as the marble slows, and it does not stop.

`sustain 1500 30 120`: **fifteen cycles in 46 s, v_in at A steady at
0.19-0.20 m/s from cycle 3 to 15**, ended by the 30-shot ceiling.
Forward kicks +0.24-0.28 raw at 45-48 V; return kicks +0.04-0.11 at
39-41 V (recharge-limited: the 1500 us gate drains the bank to 12.5 V
and the marble outranks the 0.96 threshold on every return leg). Far
ramp -0.24, entry ramp -0.10 per cycle. Predicted "above 0.25 m/s at
A": MISS -- 0.195. The return kick fired on ~64% of full bank energy,
and the far-ramp loss rose from 0.195 to 0.24 with the faster excursion.

The rig oscillates. What limits the limit cycle now is the return leg:
its kick is a third of the forward one (B-side slope + timing) and it
fires on a two-thirds bank at 1500 us. Cheapest next step: PSU current
limit 0.3 -> 1.0 A (still under the 1.5 A ceiling; fault case 22 W in a
50 W shell on metal) so the bank recovers ~27 -> 48 V in ~0.5 s and the
return kick gets a full bank. Prediction: return kicks ~+0.10-0.13,
limit cycle 0.21-0.24 m/s at A.

### Return-trim runs at 1500 us, 4 cans, 300 mA (09-24, one release each)

| trim | first return kick (fast, ~0.6-0.7 m/s) | slow return kicks | cycle at A |
|:---:|:---:|:---:|---|
| +12 fixed | -0.307 | +0.04 to +0.11 | steady 0.19-0.20, 15 cycles |
| 0 fixed | -0.134 | +0.09 decaying to +0.02 | falling 0.20 -> 0.15, rescue at the end |
| auto 2.1/v (+3.6 fast, +10.4..+12.2 slow) | -0.121 | +0.08 to +0.12, mean ~0.10 | steady 0.19-0.21, 15 cycles, ends 0.200 |

Prediction scored: "fast first kick no worse than -0.05": MISS (-0.12);
"cycle unchanged or slightly higher": HIT (0.200 vs 0.195, within noise).
The speed-dependent trim serves the slow ball as well as +12 and the fast
ball as well as 0 -- it is the right form -- but the fast return kick at
its best trim still reads -0.12 raw, the same as the 3-can sweep's best
(-0.09 at +4). Whether that is "no coil dv minus the return-leg baseline"
or "negative" is unknowable from raw dv: the return leg has no `roll`
baseline. The decisive test is cycle 2's v_in at A with the fast kick
WITHHELD (SUSTAIN_RET_MAX_V_MPS = 0.40): with the kick it has been
0.267-0.281 in three runs. Preregistered: > 0.29 without it means the
fast kick costs momentum and the skip stays on; < 0.25 means the kick
helps despite the raw number and the skip goes back off.

Withheld-kick test scored: cycle-2 v_in at A = **0.201** without the fast
return kick against 0.267-0.281 with it -> the kick HELPS; skip back off.
The log also gave the first return-leg baseline: B inbound 0.643 ->
A outbound 0.315 with no kick (-0.33 m/s), against -0.10 with the kick,
so the fast return kick delivers ~+0.22 m/s of coil dv -- larger than
the forward kick -- and the raw -0.12 was the B-side loss, not the coil.
The return leg's ramp/slope loss at 0.64 m/s is three times the forward
leg's. Limit cycle 0.19-0.20 at A in every 1500 us run regardless of the
first cycles: the losses grow steeply with speed (entry-ramp excursion
~100 mm/s at 0.2, ~260 at 0.54) and pin it. The levers left are the
bank voltage on the return kick (PSU current) and the ramps themselves.

### 1500 us return-trim sweep, 4 cans, 48 V, PSU 1.0 A (09-24 evening; one run per trim)

| fixed trim | slow return kicks (raw) | mean | first fast kick |
|:---:|---|:---:|:---:|
| 0 | 0.107 / 0.074 / 0.023 (fading as the ball slows) | 0.068 | -0.027 |
| +4 | 0.129 / 0.124 / 0.117 | **0.123** | -0.045 |
| +8 | 0.122 / 0.133 / 0.106 | **0.120** | -0.133 |
| +12 | 0.090 / 0.048 / 0.078 | 0.072 | -0.204 |
| +16 | 0.040 / 0.048 / 0.079 | 0.056 | -0.355 |
| +20 | (run died) | -- | -0.548 |

Timing IS part of it: the slow kick at +4..+8 is 0.12 against 0.07-0.09
at the +11..+12 the 2.1/v formula gave. Peak near +6 for the slow ball,
near 0..+2 for the fast one; the 700 us sweep had +12 / +4. Refit: trim
= 1.7 / v_local - 0.004 x (on_us - 700) mm, shipped (vbench). Prediction
for the 30-shot check at auto: slow kicks ~0.12, first fast kick ~-0.03,
limit cycle 0.21-0.23 m/s at A (the +4/+8 short runs sat at 0.21-0.23).
The energy-insensitivity of the slow kick (40 -> 48 V, no change) is
NOT explained by this and stays open: a track loss between the coil and
A that grows with speed is the standing reading.

30-shot check at the refit auto trim (1.7/v - 3.2 mm at 1500 us), 1.0 A:
slow return kicks 0.096-0.146, **mean 0.120** (predicted ~0.12: HIT);
first fast kick -0.072 at 0.75 m/s (predicted ~-0.03: edge); limit cycle
0.192-0.213, mean 0.205 (predicted 0.21-0.23: edge, just under). The
return kick gained 40% over the morning's formula and the cycle moved
+5%: the entry-ramp loss rose from ~100 to ~125 mm/s per excursion with
the extra speed and ate most of it. The kicks are now at their timing
optimum on both legs; the losses set the cycle. Next lever is the track
(B-side slope, ramp profiles), not the coil.

End of 2026-09-24. Sustain runs at 4 cans today: 12, all budget-limited,
none parked after the first 700 us runs. Steady state on record at
0.20-0.21 m/s.
