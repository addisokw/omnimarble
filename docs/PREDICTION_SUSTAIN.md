# Sustain twin: calibration, held-out scoring, and preregistered predictions

Written 2026-09-24/25 from the closed-loop 1-D twin (`scripts/simulate_sustain.py`,
`scripts/sustain_model.py`, `scripts/impulse_map.py`, `scripts/fit_track_losses.py`).
Rules: `docs/VALIDATION_PROTOCOL.md`. Everything below is either labelled
FITTED (in-sample), HELD OUT (scored against data the fit never saw), or
PREDICTED (data does not exist yet). Misses are reported as misses.

## What the twin is made of

- **Impulse map**: the validated single-kick simulator (`simulate_rig_shot.py`,
  frozen PINN field, tau = 275 us tail) driven by the measured 49.8 V blank
  current at the gate in question, over absolute fire positions -30..+30 mm
  and release speeds 0.15/0.25/0.5/0.8 m/s. Mirrored for the return leg,
  rescaled by (V_fire/V_map)^2. Nothing fitted. Files:
  `config/impulse_maps/impulse_map_scope_{3can_49v_700,3can_30v_700,4can_49v_1500,5can_49v_1500}.json`.
  4-can 1500 us map at the +9 point: 0.301 m/s (the injected Layer A prediction).
- **Track losses** (`config/track_losses.json`), FITTED:
  - flat zone: dv/dt = -(a0 + k v^2), k = 0.742 /m, a0 = 0.010 m/s^2 -- the
    three `roll` baselines (0.206/-0.034, 0.245/-0.050, 0.79/-0.113 over 204 mm).
  - B-side excess for travel toward the coil: k = 2.74 /m, g = 0.11 m/s^2,
    acting from the far edge to **station B's inner channel (x = 57.78)**,
    from 45 return-pass station reads (fit-vs-local velocity ratio) in the
    three 1.0 A auto-trim runs (4 cans 21:25, 5 cans 21:50 and 21:53).
    Return-pass rms 0.0097 m/s.
  - far ramp: v_back^2 = 0.50 v_out^2 (beta 0); entry ramp: 0.35 v_out^2
    (beta 0); fitted on excursions with edge speed <= 0.7 m/s (42 each),
    the fast first cycles held out.
- **Bank**: CC at the PSU limit up to V_psu - I R, then RC through 22 ohm,
  the firmware's 0.96 / 0.60 release policy, drain per shot from the
  capture's V(cut)/V(on).
- **Timing**: the firmware's own rules -- forward linear on the station
  fit; return linear on the local velocity plus the trim (1.7/v - 3.2 mm at
  1500 us, as shipped on 09-24) or the new constant-acceleration predictor.

## The one in-sample choice: where the B-side excess ends

The return passes measure the excess only across station B (57.78-146 mm
from the coil centre). Extended to the coil face (22.78) the twin parks the
marble on cycle 3; ending it at 40 mm it stalls on cycle 4; ending it at
station B it reproduces the fit case. So `x_from = 57.78` was chosen ON the
fit case (in-sample), and it is the twin's statement that the slowing the
firmware sees through B does not continue over the last 35 mm to the coil.
The bench can check that directly: the predictor firmware logs the fitted
deceleration and the fire timing per return pass.

## Scoring

| case | status | bench | twin | verdict |
|---|---|:---:|:---:|---|
| 4 cans 1500 us 1.0 A auto: cycle at A | FITTED | 0.205 | 0.190 | in band (+/-0.02) |
| same: forward kick raw | FITTED | 0.25 | 0.20 | low by 0.05 |
| same: slow return kick raw | FITTED | 0.12 | 0.19 | high by 0.07 |
| 4 cans 1500 us 0.3 A auto: cycle | HELD OUT | 0.195-0.200 | 0.188 | HIT |
| same: return bank at fire | HELD OUT | (logged bank_pre 39-41, read early) | 49.3-49.6 | consistent with the bank_pre finding |
| fixed-trim sweep: slow-kick optimum trim | HELD OUT | +4..+8 | +8 | HIT (band +/-4 mm) |
| fixed-trim sweep: slow kick at 0/4/8/12/16 | HELD OUT | 0.068/0.123/0.120/0.072/0.056 | 0.063/0.138/0.189/0.180/0.161 | shape MISS above +8: the twin's curve falls off far slower |
| fixed-trim sweep: first fast kick at 0..16 | HELD OUT | -0.03/-0.05/-0.13/-0.20/-0.36 | +0.09/+0.14/+0.10/+0.02/-0.06 | MISS, twin high by 0.15-0.30 |
| no-kick return leg B 0.643 -> A | HELD OUT | 0.315 | 0.425 | MISS by 0.11 |
| 5 cans 1500 us 1.0 A auto: cycle | RETRODICTION (bench ran first) | 0.246-0.251 | 0.259 | in band |
| same: fwd / ret raw | RETRODICTION | 0.32-0.34 / 0.165-0.20 | 0.256 / 0.240 | same low/high pair |

Reading: the cycle comes out right at three settings because two errors
cancel -- the forward kick is ~20% low (the twin lands it 3-4 mm short of
-13.78 through flat-zone loss over the reach; the bench found +9 optimal
with the same rule, so the bench's ball arrives closer to the target than
the twin's does) and the return kick is ~50% high (the twin's fast ball
loses less on the B side than the bench's: the no-kick leg and the fast
first kicks both say the B-side loss at 0.5-0.7 m/s is larger than the
station-region fit gives). Both point the same way: the B-side loss is
under-modelled at speed and somewhere between B and the coil. Nothing has
been refitted to the held-out rows.

## PREDICTED (data does not exist yet)

1. **The predictor firmware (vbench ff5cd87) on the bench**, 5 cans,
   1500 us, 1.0 A, retoffset 0. Two brackets, both in the twin:
   - if the slowing through B does NOT continue to the coil (the twin's
     fitted world): the constant-acceleration predictor fires the return
     kick ~13 mm past the target, return raw dv <= 0, the cycle collapses
     within 3 cycles;
   - if it does continue (the plan's assumption): the predictor lands on
     target, slow return kicks >= 0.17 raw, cycle >= 0.245.
   The logged `late_us`, `accel_mps2` and "later than constant-v" lines
   settle which; `retoffset -6` is the manual fallback for the first case.
2. **Levelling the B side** (removing the excess): cycle 0.190 -> 0.240 at
   half the excess -> 0.281 with none (4 cans, 1500 us, 1.0 A). Largest
   lever found. Dissipative reading; the conservative-slope bound is lower.
3. **700 us / 0.3 A hold-out** (4 cans, fixed trim +12; bench: cycle 0.16
   over 5 cycles, fwd 0.19-0.20, ret 0.07-0.08): HELD OUT, **MISS**. The
   twin stalls the marble on cycle 3 (0.81 -> 0.242 -> 0.157 -> stall);
   forward kick 0.144 raw (bench 0.19-0.20, 25% low), return kick 0.135
   (bench 0.07-0.08, high). The same cancelling pair as at 1500 us, larger,
   and at the shorter gate it no longer cancels. **Gate sweep** (4 cans,
   auto trim): the twin has 700 us stalling at both 1.0 A and 0.3 A and
   1500 us at 0.190 / 0.188 -- the 700 us rows are wrong by the bench's
   own 0.16-m/s cycle, so the sweep is not a prediction until the forward
   kick is fixed.

   Both misses point at the forward kick's delivered position: the twin
   lands it 3-4 mm short of -13.78 (flat-zone loss over the 49 mm reach);
   the bench found +9 optimal with the same rule, so its ball arrives
   closer. One measurement settles it: a marble-shot scope capture of a
   forward kick (early current slope vs the same-gate blank gives x at the
   gate to ~1 mm; COIL_AS_SENSOR 3B). Requested for the predictor check run.

## Bracket result (2026-09-25): ONE

The predictor firmware fired the slow return kicks 16 mm late (clamped
from a modelled +31..+39 mm) and stopped the ball in both runs -- see
`PREDICTION_45CAN.md`, "Predictor check scored". The twin's fitted
world (excess ends at station B) is the bench's world. The measured
per-pass decelerations (0.6-1.5 m/s^2 through B at 0.3-0.8 m/s, and the
same through A) are the direct measurement of the station-region loss
the fit inferred from velocity ratios. The empirical trim is back as the
firmware's return rule; the twin's "linear + trim" timing option is
therefore the one to run. Still open on the twin's side: the forward
kick lands 3-4 mm short in the twin and not on the bench (the scope
capture of a forward kick is the measurement); the B-side loss at speed
(a/v^2 = 9.7 /m at 0.28 m/s, 2.1 at 0.72: not one k) needs a form.

## The forward-kick miss, diagnosed (2026-09-25)

Twin kick records at the 4-can limit cycle: v_in_fit 0.228 -> v at the
gate 0.209 (-8%), delivered x = -16.8 mm against the -13.78 target. The
fitted flat drag (k 0.74 /m) over the ~137 mm from station A's centre to
the fire point costs that 8%, and a constant-velocity rule lands 3 mm
short. The bench's station A kinematics on 09-25 read a = -0.02..-0.14
m/s^2 at the same speed, i.e. the same drag -- the REAL kick lands ~3 mm
short of the model's peak too. The difference is the curve: the frozen
impulse map is peaked at -14.0 and pays ~10% at -16.8; the bench's
measured 3-can curve (PREDICTION_45CAN.md, 09-09) is flat-topped from
-20 to -14 and pays nothing there. So the twin's 20% forward under-
prediction IS the deferred entry-side anomaly, seen through sustain.
Not fitted. PREDICTED for the pending forward-kick scope capture: ball
position at the gate ~ -16.8 mm (early-slope dL ~0.65 uH, -3% of the
22 uH loop, against ~0.93 uH / -4.2% if it were at -13.78).

Bench kinematics for the B-side term (09-25): a/v^2 = 5-9 /m at 0.24 m/s
through B, 2-3 /m at 0.7; through A: ~1 /m at 0.27, 2-3 /m at 0.5-0.8.
The B-side excess grows FASTER than v^2 at low speed -- not one k. Left
as data; a form is chosen only for a physical reason.

## Forward-kick capture scored (2026-09-25, 5 cans, 1500 us, first kick at 0.85 m/s)

`data/captures/scope_5can_marble_fwd_kick1_49v_1500.csv` against the
5-can blank: peak 312 A vs 329, integral 75.9 vs 78.6 A^2 s, early
current slope (2-40 us) 2.7% lower with the ball -> dL = +0.60 uH on the
22 uH loop -> ball at **x ~ -17.2 mm** at the gate (model dL(x); the
longer windows drift as the ball couples in, the earliest is the one to
read). Predicted -16.8. HIT within the method's ~+/-3 mm (blank-to-blank
slope scatter ~2%). First direct fire-position measurement on the rig;
the forward kick lands ~3 mm short of the frozen map's peak on the bench
as in the twin, so the twin's 20% forward under-prediction is the map's
shape (peaked) against the bench's (flat-topped, the entry-side anomaly).
Next measurement: the fire-position sweep at 5 cans / 1500 us.

## What would change the model

- The bench run in (1) lands in the first bracket: keep x_from at B and
  add the predictor's over-correction to the firmware's known behaviour
  (a shorter extrapolation, or a clamp on the modelled extra delay).
- It lands in the second: x_from moves toward the coil, the far-ramp fit
  must be redone with it (they are coupled through the edge-speed
  conversion), and the trim-sweep shape miss should shrink.
- Either way the B-side loss at speed needs a term the station fit cannot
  see; the no-kick leg and the fast-kick rows are the data for it, and
  they stay held out until a form is chosen for physical reasons.
