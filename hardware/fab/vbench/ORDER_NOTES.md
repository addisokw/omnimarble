# OmniMarble Driver — JLCPCB Order Notes

Assembly package for `omnimarble-driver.kicad_pcb`. **Scope: SMT + THT
assembly** (JLC places surface-mount *and* the common through-hole parts).

## Board
- 220 × 150 mm, **4-layer, 2 oz outer / 1 oz inner** copper, **ENIG** finish
  (F.Cu / In1.GND / In2.GND / B.Cu). The stackup + gerber job metadata state this.
- Min track/space 0.15 mm; min via 0.4 mm / 0.2 mm drill
- Electrical DRC: **0** (0 unconnected / shorts / clearance / parity). 16 cosmetic
  silk warnings only. Report: `drc_driver.json`.

## Files in this package
| File | Upload to |
|---|---|
| `omnimarble-driver-gerbers.zip` | JLC **fabrication** (gerbers + drill) |
| `bom_jlc.csv` | JLC **assembly** — BOM (12 line items) |
| `cpl_jlc.csv` | JLC **assembly** — pick-and-place (13 placements) |
| `drc_driver.json` | reference (DRC evidence) |

Summary: **13 machine-placed** parts, **9 hand-installed**,
4 board features, 8 do-not-populate.

## When ordering (read first)
1. **Set 4-layer, 2 oz OUTER copper + ENIG finish.** The thermal/current design
   assumes 2 oz outer; verify the order form matches (the gerber stackup states it,
   but the copper weight is a paid order-form selection). ENIG is recommended for
   the fine-pitch U10.
2. **Enable "Assemble top side" + turn ON through-hole assembly** — 3 of the
   placed parts are THT (below). Without THT assembly JLC will skip them.
3. **Verify component rotations in JLC's preview.** KiCad and JLC differ on some
   part rotations (polarised caps, some ICs/connectors). **Check K1/K2 especially**
   — both relays are at rotation 0 in the CPL and JLC's relay origin convention
   often differs from KiCad's; a wrong relay rotation swaps its terminals. Also
   confirm J1 (barrel jack) and the polarised caps. Fix any flagged in JLC's
   online CPL editor before paying.
4. **Let JLC's DFM be the final arbiter.** Inspect DFM warnings around **U10**
   (fine-pitch ADS7042 escape), **U2**, the **ISNS_P/N traces** (0.15 mm width —
   intentional, 2 oz min; note these are *not* a tight matched pair — see bring-up
   item 5), and **J1** (barrel jack is now an **edge-mount** — its body overhangs
   the bottom outline ~6 mm by design; pads are on-board).

## Machine-placed through-hole parts (need JLC THT assembly) — 3
| Ref | Value | LCSC (package) |
|---|---|---|
| J1 | COIL | C707836 (TerminalBarrier_1x02_P9.50mm) |
| J2 | CHARGE | C707836 (TerminalBarrier_1x02_P9.50mm) |
| J3 | AUX-BANK | C707836 (TerminalBarrier_1x02_P9.50mm) |

## Hand-installed by you (NOT on the CPL) — 9
Order these separately (LCSC # given where known) and solder after JLC assembly.
| Ref | Value | LCSC (package) |
|---|---|---|
| C1 | 2200u-63V | C47858 (CP_Radial_D18.0mm_P7.50mm) |
| J10 | WAVESHARE-B | — (PinHeader_1x07_P2.54mm_Vertical) |
| J11 | WAVESHARE-C | — (PinHeader_1x07_P2.54mm_Vertical) |
| J12 | WAVESHARE-D | — (PinHeader_1x07_P2.54mm_Vertical) |
| J4 | 12V-IN | — (PinHeader_1x02_P2.54mm_Vertical) |
| J6 | ISENSE | — (PinHeader_1x02_P2.54mm_Vertical) |
| J7 | PICO-L | C50984 (PinSocket_1x20_P2.54mm_Vertical) |
| J8 | PICO-R | C50984 (PinSocket_1x20_P2.54mm_Vertical) |
| J9 | WAVESHARE-A | — (PinHeader_1x07_P2.54mm_Vertical) |

## Board features (no placement) — 4
| Ref | Value |
|---|---|
| H1 | MountingHole_3.2mm_M3_Pad |
| H2 | MountingHole_3.2mm_M3_Pad |
| H3 | MountingHole_3.2mm_M3_Pad |
| H4 | MountingHole_3.2mm_M3_Pad |

## Do-not-populate — 8
| Ref | Value |
|---|---|
| C2 | 2200u-63V |
| C3 | 2200u-63V |
| C4 | 2200u-63V |
| C5 | 2200u-63V |
| Q2 | IRFP4668 |
| Q3 | IRFP4668 |
| RG2 | 10R |
| RG3 | 10R |

## Bring-up checklist (physical checks DRC/parity CANNOT catch)
1. **Ohmmeter each relay (K1, K2) de-energized: COM↔NC continuity.** The HF3FF
   footprint's NO/NC were transposed and fixed by a pad-number swap — parity is
   green *by construction*, so a meter is the ONLY confirmation the dump/charge
   logic isn't inverted. De-energized, COM must read to NC (not NO).
2. **OVP is a *throttle*, not a hard stop — the authoritative over-voltage cut
   is firmware.** U5's TL431 cathode sits on the UC3843 COMP pin and can only
   clamp it to ~2–2.5 V, which *reduces* the charge current but does not reach
   zero duty (that needs COMP < 1.4 V). So on a bench supply the observable is a
   **charge-rate knee near ~59–62 V with continued slow creep**, NOT a clean
   plateau — a tester expecting a hard stop could mis-judge a working part.
   **Firmware MUST treat VBANK_SENSE > threshold as a hard `BOOST_EN_N` inhibit**
   (drives Q1, which saturates COMP to ~0.1 V and has full authority). Verify
   *that* interlock, and the analog knee, before the first bank charge.
   *(Rev-2: give the TL431 full authority via a PNP off BST_VREF onto COMP/Q1.)*
3. **Fan-cool U2 (78L12, SOT-89) and watch its temp** during a sustained charge
   — it runs hot (~0.5–0.85 W); the DPAK thermal upgrade was deferred.
4. Stage first power at the **25 V floor**, then up to 55 V, verifying boost
   regulation before installing max bank capacitance or firing into the coil.
5. **Current-sense rise-phase caveat (ISNS).** The re-route left ISNS_P and
   ISNS_N on different corridors (not a tight pair), so during the fast dI/dt at
   pulse onset expect ~5–10 A of transient pickup error on the *rising* edge;
   **peak and energy readings are unaffected.** For first-spin sim-to-real
   comparison, trust peak/energy; treat the rise-phase waveform as indicative.
   (Rev-2 fix: pre-author the ISNS pair as locked critical copper.)

---
*Generated by `hardware/scripts/gen_fab.py` from the committed board.*
