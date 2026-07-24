# V0 Stage-2 Switch — low-side coil driver

The full-energy switch for the [V0 bench rig](V0_BENCH_RIG.md): a **low-side
N-MOSFET with a dedicated gate driver**. Low-side is the whole trick — the FET
source sits at ground, so the gate driver references ground and there's no
bootstrap / high-side complication.

Use this for Stage 1–2. A cheap prebuilt module can stand in for the Stage-0
electronics bring-up, but not the real experiment — see *Prebuilt board?* below.

## Schematic

```
                       V+  ● bank rail (24–45 V, held by C1)
                           │
          bench(+) ────────┤
          24–45 V,         ├───────────┬───────────────┬──────────────┐
          I-limited        │           │               │              │
                         ══╪══ C1     R_bleed         coil L1      (fire path)
                           │ 4700µF   1 kΩ 5 W        hand-wound        │
                           │ 63 V      │               │                │
                           │           │            ┌──┴──┐             │
                           │           │            │ P1  │ ACS758      │  optional
                           │           │            └──┬──┘ VIOUT→ADC   │  current sense
                           │           │               ● DRAIN          │
                           │           │        D1 ────┤                │
                           │           │       TVS     │   ┌────────────┴┐
                           │           │      5KP48A    ├─D─┤ Q1 IRFB4110 │
                           │           │     (D→S)      │   │  100 V FET  │
              Rg 10 Ω      │           │               │   │             │
   ┌──[///]── G ●──────────┼───────────┼───────────────┼─G─┤  gate       │
   │          │            │           │               │   │             │
   │       Rgs 10k         │           │            S ──┼─S─┤  source     │
   │          │            │           │               │   └─────────────┘
 ┌─┴────────┐ │            │           │               │
 │ U1 TC4420│ │            │           │               │
 │  OUT ────┘ │            │           │               │
 │  VDD ◄───── +12 V (gate-driver rail, small wall-wart)
 │  IN  ◄───── FIRE  (Pico GPIO, 3.3 V — TTL input, drives fine)
 │  GND ──┐   │            │           │               │
 └────────┘   │            │           │               │
    GND ●━━━━━┷━━━━━━━━━━━━┷━━━━━━━━━━━┷━━━━━━━━━━━━━━━━┷━━━━●  ★ STAR GROUND (bank −)
              (logic gnd and power gnd meet here — and ONLY here)
```

## Net list (unambiguous — build from this)

| Node | Everything on it |
|---|---|
| **V+** | bench supply (+), C1 (+), R_bleed (a), coil L1 (a) |
| **DRAIN** | coil L1 (b) → *via P1 ACS758* → Q1 **drain**, D1 TVS **cathode** |
| **GATE** | Q1 **gate**, Rg (b), Rgs (a) |
| **DRV_OUT** | U1 TC4420 OUT, Rg (a) |
| **SOURCE / GND_P** | Q1 **source**, Rgs (b), D1 TVS **anode**, C1 (−), R_bleed (b), bench supply (−) → ★star |
| **VDD12** | U1 VDD, C2/C3 (+), 12 V supply (+) |
| **FIRE** | U1 IN, Pico GPIO |
| **GND_L** | U1 GND, C2/C3 (−), 12 V supply (−), Pico GND → ★star |
| **ISENSE** | P1 ACS758 VIOUT → Pico ADC (P1 VCC→5 V, P1 GND→GND_L) |

## BOM

| Ref | Part | Why |
|---|---|---|
| Q1 | **IRFB4110** (100 V/180 A, TO-220, heatsinked) | pulse-rated FET; use IRFP4668 (200 V) if bank > 45 V |
| U1 | **TC4420CPA** gate driver (DIP-8, 6 A, non-inverting) | breadboardable; TTL input works off the 3.3 V Pico |
| Rg | 10 Ω ½ W | limits gate current / controls dV/dt |
| Rgs | 10 kΩ ¼ W | holds Q1 **off** if the driver is unpowered |
| D1 | **5KP48A** TVS (standoff 48 V, clamp ~77 V) | fast field collapse — see clamp note |
| C1 | 2200–4700 µF / 63 V | the energy bank |
| R_bleed | 1 kΩ 5 W across C1 | safety discharge (~5 s τ) |
| C2, C3 | 1 µF + 100 nF at U1 VDD | driver decoupling — keep leads short |
| P1 | ACS758LCB-100U (optional) | logs the pulse current to the Pico ADC |

## The three things that make or break it

1. **Clamp vs FET voltage.** D1 sits across drain-source; when Q1 opens, the coil
   dumps its current through the TVS, clamping V_DS to ~77 V (< the 100 V FET) and
   collapsing the field in **~150 µs** instead of the ~1 ms a plain freewheel diode
   gives. That fast collapse is the anti-suck-back. Rule: **standoff > bank voltage,
   clamp < 90 V**, so with the 100 V FET **keep the bank ≤ 45 V** and use the 5KP48A.
   Want more voltage → IRFP4668 (200 V) and the margin is huge.
   *A/B test:* swap D1 for an **MBR60100 freewheel** (anode→DRAIN, cathode→V+) and
   you'll see the slower collapse + more suck-back — confirms the clamp earns its place.
2. **Star ground.** The pulse loop (C1 → coil → Q1 → back to C1−) carries hundreds of
   amps. Logic ground (Pico, driver) touches it at **one point only** — the bank
   negative — or pulse noise walks straight into your sensor readings.
3. **Gate default-off.** Rgs guarantees Q1 is off before the driver powers up.

## Prebuilt board? Stage-0 only

Worth knowing after surveying the market: no off-the-shelf board gives you **both**
adequate voltage **and** high pulse current at our bank voltage, because the cheap
opto-isolated MOSFET switch modules are single-FET boards with a fixed volt-amp
product:

| Prebuilt variant | Fits our ≤45 V bank? | Pulse current? |
|---|---|---|
| 5–36 V modules (F5305S, ANMBEST) | ✗ under-volted | — |
| **100 V / 9.4 A** isolated switch | ✓ | ✗ current-starved (9.4 A) |
| 80 V / 18 A | ✓ | ✗ weak |
| 40 V / 50 A · 30 V / 161 A | ✗ under-volted | ✓ (but too low V) |

They're also opto-isolated at PWM speeds (~µs turn-off), which smears the precise
cutoff this whole rig depends on. **Verdict:**
- A **100 V / 9.4 A opto-isolated module** (3.3 V/5 V trigger) is a fine, *isolated,
  zero-build* switch for **Stage 0** — dry-firing the logic and confirming Pico
  timing + sensor reads at low energy without exposing the Pico.
- It is **too current-starved and too timing-smeared for Stage 1–2**. The discrete
  switch above is the only thing that delivers both the current headroom and the
  deterministic ~ns cutoff the timing experiment measures.

Also avoid the "IGBT driver boards" in this market — they're **inverter / welder
back-ends** (continuous SPWM controllers, high-voltage bus, no single-shot trigger),
not triggerable pulse switches, and an IGBT is the wrong device below ~250 V anyway
(VCE(sat) floor + turn-off tail current).

## Safe first-light sequence

1. **No bank connected.** Power 12 V + Pico → pulse FIRE → scope confirms GATE swings
   0 ↔ 12 V.
2. **Bank at ~10 V**, current limit ~0.5 A → fire a 200 µs pulse → scope DRAIN: pulls
   to ~0 on, clamps clean on turn-off.
3. **Add coil + ACS758**, repeat at 10 V, capture the current pulse.
4. **Ramp** per the [V0 Stage 0→1→2 protocol](V0_BENCH_RIG.md). Bleeder + shorting
   stick between voltage changes.
