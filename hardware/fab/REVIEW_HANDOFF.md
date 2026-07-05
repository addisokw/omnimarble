> ⚠️ **SUPERSEDED (2026-07-05).** This describes the mid-project freerouting
> attempt (13 unconnected). The board is now **fully routed and DRC-clean** via
> DeepPCB + local finish — see **`BOARD_OVERVIEW.md`** for the current handoff.
> Kept only for history.

# OmniMarble Driver PCB — Review Handoff (rev 3: autorouting pivot)

**Repo:** github.com/addisokw/omnimarble
**Board:** `hardware/omnimarble-driver/omnimarble-driver.kicad_pcb`
**Tool:** KiCad 10.0.4 + freerouting v2.2.4 (run on JDK 26)

## What changed since rev 2

We retired the from-scratch PathFinder router for bulk signal routing and pivoted
to **freerouting v2.2.4**. The earlier "freerouting is unusable" verdict was wrong
— its failures were fixable pipeline defects, not board complexity. New
reproducible pipeline: `preroute → dsn_fixup → freerouting → import-ses → DRC`.

**Key mechanism** (`hardware/scripts/dsn_fixup.py`, new): KiCad exports one flat
0.2 mm net class, so we post-process the DSN to (a) inject real per-width net
classes, (b) give pulse/bank nets the 1.0 mm bank-voltage clearance, (c) lock all
critical + pour copper `(type route)→(type fix)` so the router can't rip it.
`max_passes=30` (the 9999 default caused the prior ~700 GB / non-termination).
The same prepared DSN also uploads to DeepPCB unchanged (parallel option).

## Results (kicad-cli, `--schematic-parity`)

| Metric | Scripted (rev 2) | freerouting (now) |
|---|---|---|
| Unconnected | 118 | **13** |
| Clearance | 59 (to 0.010 mm on FIRE_GATED) | **6** (all mild 0.15 mm signal) |
| Shorts / crossings / parity | 0 / 0 / 0 | **0 / 0 / 0** |
| Hole-to-hole / hole-clearance | — | **0 / 0** |

The clearance quality is the material improvement: no more
sub-manufacturing-tolerance near-shorts on safety nets; the 6 remaining are benign
0.15 mm gaps on ADC-SPI / 3V3 / IMON_F.

## The 13 unconnected (honest categorization)

- **3 critical, unrouted by design**: SHUNT_HI Kelvin (×2), BST_G boost gate.
  Deterministic critical-net authoring (plan Step 1) was deferred to validate the
  freerouting pipeline first; these are exactly what that step authors and locks —
  which also satisfies the rev-2 requirement that critical nets be hand-finished,
  not autorouted.
- **5 GND**: congested fine-pitch pads the full-board GND pour + via-in-pad
  stitching couldn't reach.
- **5 signal taps**: 3V3 (U10), +5V, +24V (U2), VBANK_DIV (U3) in dense areas.

## Two decisions needing scrutiny

1. **Design-rule relaxation**: board min-via lowered 0.5→0.4 mm, min-hole
   0.3→0.2 mm (in `.kicad_pro`) to allow the 0.45/0.25 GND via-in-pad stitching.
   JLC-manufacturable, but a real rule change — please confirm acceptable for this
   board class.
2. **New architecture**: full-board F.Cu/B.Cu GND pours added (`build_zones`), so
   GND pads connect to the pour directly rather than each needing a threaded via.
   Standard practice, but it substantially changes the copper — worth confirming no
   unintended coupling into the analog / Kelvin / boost-sense sections.

## Still not fab-ready (unchanged gate from rev 2)

0 unconnected / 0 clearance / 0 parity, **then** manual power-loop / Kelvin / boost
review with net-highlighted screenshots (symmetric FET copper, true Kelvin pickup
at the shunt, intentional thermal reliefs on pulse pads), then silkscreen cleanup +
fab exports. Order-time gates still open: 6.8 k bleed LCSC C-number, relay NO/NC
continuity.

## What I want your opinion on

- Is the freerouted track geometry acceptable for the noncritical nets, given
  critical nets will be separately authored and locked?
- The min-via/hole relaxation and the full-board GND pour — any objection?
- Anything the DRC-clean copper is masking that a highlighted-net review should
  target first?

## Artifacts

- Renders: `hardware/fab/renders/driver_top.png`, `driver_bottom.png`
- Scorecard dashboard: `hardware/fab/driver_pcb_review.html`
- DRC report (regenerate): `kicad-cli pcb drc --schematic-parity --format json
  -o hardware/fab/drc_driver.json hardware/omnimarble-driver/omnimarble-driver.kicad_pcb`
- Pipeline scripts: `hardware/scripts/dsn_fixup.py`, `gen_pcb.py` (modes:
  preroute / import-ses / repair), `placement.py` (CRITICAL_NETS / PLANE_NETS /
  netclass_rules)
