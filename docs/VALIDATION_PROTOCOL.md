# Validation protocol — how this sim avoids becoming a lookup table

The risk being managed: a model with enough adjustable constants can "agree"
with any finite dataset without containing any physics. This project has
fitted several constants to bench data (that is proper — the bench is the
arbiter), so the protection cannot be "never fit". It is these four rules.

## 1. Predictions are frozen before data exists

Every new configuration (can count, voltage regime, marble, geometry change)
gets predictions committed to git BEFORE the first shot, with per-point bands
and the meaning of each possible miss declared in advance. The commit hash is
the timestamp. Precedent: `docs/PREDICTION_3CAN.md` (67b6f37) — which caught
two real misses and is the reason the droop and knee physics were found at
all. A prediction written after the data is a description, not a test.

## 2. No constant is scored against the data that produced it

Every fitted constant must name its training data, and any agreement claim
made on that same data must be labelled IN-SAMPLE. Current ledger:

| constant | fitted to | independent corroboration | may be scored on |
|---|---|---|---|
| pulse C/R pairs (1/2/3 can) | blank-fire V(t) | LCR small-signal + droop trend | marble data, scope I(t) |
| freewheel tau 275 us | 3-can marble dv at 3 gates | implied R fits under measured loop R | 4/5-can data, other axes |
| PINN field factor (none applied) | — | marble dv via injected current | everything |

The "field within ~1.4%" figure is CONDITIONAL on the tau fit's form; the
unconditional bound from the injection campaign alone is ~2.5–5%. Say which
one is meant.

## 3. Every fitted quantity needs a second, mechanism-independent handle

tau has the loop-R consistency check; pulse C has the LCR + charge-balance
check; L and loop R have the waveform slope/peak extractions. The FIELD has
exactly one handle (marble dv) — closing that is the highest-value open item:

> **Hall-probe spot check**: small DC current through the coil, Hall sensor
> at surveyed positions, compare the PINN's absolute B directly. No marble,
> no circuit, no dynamics — an unconditional field measurement for ~$10.

Untested model choices to keep on the books: chi_eff = 3.0, B_sat, the
finite-size quadrature. Each should eventually get its own handle or an
explicit uncertainty band.

## 4. The C-sweep is a HOLDOUT, not a tuning set

Every constant in the model as of tag `model-freeze-3can` derives from
1–3 can data. Before can 4 is soldered:

1. Preregister 4-can and 5-can predictions (blanks + marble points) from the
   frozen model, per rule 1.
2. After the data: score, publicly, misses included. Fitting anything NEW to
   4-can data resets the holdout to 5 cans.

Held-out axes that no fit has consumed, usable at 3 cans any time:
- fire-position curve (the model says the peak stays at −13.5 ± 1 mm),
- a 30 V marble point,
- a different (weighed) marble,
- scope captures at 2 cans retrodicted with zero refits.

## The standing test

If the model only works where it was tuned, it will fail rule 4's first
encounter with can 4 — and that failure must be reported as loudly as the
successes. The preregistration record is only worth something because it can
contain misses; `PREDICTION_3CAN.md` already does.
