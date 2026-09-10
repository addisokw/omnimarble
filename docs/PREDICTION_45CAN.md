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
