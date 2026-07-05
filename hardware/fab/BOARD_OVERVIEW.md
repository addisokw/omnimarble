# OmniMarble Driver PCB — Board Overview (handoff)

> Authoritative, current handoff for the driver board. Supersedes
> `REVIEW_HANDOFF.md` (which is from the earlier freerouting routing attempt and
> is out of date). Last updated 2026-07-05.

## What it is
A **coilgun-style pulse driver board** — the experimental-hardware side of the
OmniMarble project. It charges a capacitor bank to an elevated SELV voltage
(~55V), then dumps it through a coil via paralleled power MOSFETs, with
closed-loop current sensing and a safety interlock chain. Designed for JLCPCB
assembly; controlled by an external Raspberry Pi Pico (or compatible) via header.

**Repo:** `hardware/omnimarble-driver/omnimarble-driver.kicad_pcb` (KiCad 10 project)

## Key specs
| | |
|---|---|
| Board | 220 × 150 mm, 4-layer, 2oz copper |
| Layer stack | F.Cu (signal) / In1 (GND plane) / In2 (GND plane) / B.Cu (signal) |
| Components | 177 footprints, 108 nets |
| Bank voltage | ≤55V SELV (functional isolation) |
| Status | **Fully routed, electrically DRC-clean** (0 unconnected / shorts / clearance / courtyard / parity / annular / hole). Remaining DRC = 14 cosmetic silk warnings. |

## Architecture / subsystems

**Power input & rails**
- 24V barrel jack (J1) → fuse (F1) → reverse-polarity protection (Q90 AOD4185 P-FET) → `+24V`
- Buck U1 (AP63205) → `+5V`; 12V linear U2 (78L12) → `+12V`; `3V3` for logic/ADC

**Boost charger** (charges the cap bank)
- U4 (UC3843B current-mode PWM) + Q91 (AOD66923 N-FET) + L2 → `VBOOST` → cap bank `VBANK`
- Bank sense divider → U3 (MCP6002) buffers `VBANK_SENSE` / `V24_SENSE`; OVP reference U5 (CJ431)

**Cap bank & pulse stage** (the high-current path)
- 5 bulk caps (2 radial + 3 screw-terminal), `VBANK` pours
- Coil switch: **Q10/Q11/Q12 (SFT040N150C3, 150V) in parallel** — drains on `SW_DRAIN`, sources on `SHUNT_HI`
- Gate drivers U7/U8 (UCC27524A) → `QG1-3` / `DRV1-3`; snubbers `SNB1-3`
- Flyback clamp: MBR60100 diodes D9-D12 (`COIL_HI`); coil connects at J5

**Current sense** (Kelvin)
- Shunt RS1 (low-side, between `SHUNT_HI` and GND), tapped via net-ties NT1/NT2 → `ISNS_P` / `ISNS_N`
- U9 (INA240A1) current-sense amp → `IMON` → U10 (ADS7042 12-bit SAR ADC) → SPI to MCU

**Sensor front-end & logic**
- U11/U12 (LM339 quad comparators): IR-sensor thresholds — `CIN1-6`, `GATE1-6`, `SIG1-6`, `VTH_GATE`

**Safety interlock**
- E-stop (`ESTOP_SENSE` / `ESTOP_OK`), arm (`ARM_SENSE`), fire gating: `MCU_FIRE` AND'd through U6 (74LVC1G08) → `FIRE_GATED`
- Charge/dump relays K1/K2 (`RLY_CHARGE` / `RLY_DUMP`), dump resistors; buzzer BZ1 (Q4 driver)

**MCU interface**
- Pico sockets J9/J10; IDC connectors J6/J7/J8; aux/status LEDs

## How the board is generated (fully scripted, reproducible)
1. `hardware/scripts/gen_schematic.py` → schematic + netlist + `.kicad_pro` (DRC rules)
2. `hardware/scripts/placement.py` → placement, net widths/classes, critical-net taxonomy, authored-copper specs
3. `gen_pcb.py preroute` → places parts, pours planes, authors critical copper
   (Kelvin ties, U10 fine-pitch escape, SHUNT_HI B.Cu bus), exports Specctra DSN
4. `dsn_fixup.py` → net classes, name sanitizing, lock critical/pulse copper → DSN uploaded to
   **DeepPCB** (cloud router) for bulk signal routing (diff-pair mode OFF — see gotchas)
5. **`gen_pcb.py import-clean`** → imports routed SES + deterministic finishing
   (`local_finish`, `tidy_silk`, `fix_thin_annular`, F/B GND flood, fill) → final board

**Regenerate the exact current board:** `gen_pcb.py import-clean` on
`hardware/omnimarble-driver/omnimarble-driver-3-rev-43.ses` (the frozen routed input).
Run with KiCad 10's bundled Python: `"C:\Program Files\KiCad\10.0\bin\python.exe"`.

## Design decisions / gotchas (READ before editing)
- **Critical authored copper is locked and intentional** — a router must not touch it:
  the ISNS **Kelvin ties** (RS1→NT1/NT2); the **U10 dogbone escape** (fine-pitch ADS7042,
  0.2mm stubs to open-space vias); the **SHUNT_HI B.Cu bus** (ties the 3 FET sources to the
  shunt — the F.Cu pour fragments and can't be trusted for that current path).
- **ISNS_P/N are 0.1mm traces on purpose** — the Kelvin pair was coupled too tight by the
  router; narrowing (not re-routing) grew the clearance while preserving matched length.
  `min_track_width` is 0.1mm for this reason (within JLC's 0.089mm capability).
- **Never enable a differential-pair router on ISNS** — it hard-couples at ~0.02mm
  (unmanufacturable) every time. Route sense pairs as normal matched nets.
- **Reference designators live on the F.Fab layer**, not silkscreen (dense board); LCSC part
  numbers are hidden property fields (still in the data for the BOM). Physical silk keeps only
  outlines + annotations (K=tab, gate G, polarity, board labels).
- **Voltage domains:** `VBANK` / `VBOOST` / `AUX_BANK` are the elevated rails (VBANK/AUX_BANK
  carry 1.0mm creepage clearance). `SHUNT_HI` is high-*current* but low-*voltage* (low-side
  shunt, near GND) — do **not** give it high-voltage clearance.
- **KiCad 10 API traps:** `PCB_VIA.GetWidth()` needs a layer arg (else blocking-assert hang);
  `board.Remove()` / a second `LoadBoard()` corrupt SWIG footprint wrappers (reload fresh).

## State & next step
- Committed (tip `ef45801` at time of writing). Renders: `hardware/fab/renders/driver_{top,bottom}.png`.
  DRC report: `hardware/fab/drc_driver.json`.
- Remaining: 14 cosmetic silk warnings (edge/outline; JLC auto-trims).
- **Not yet generated: the JLCPCB fab package** — Gerbers, drill files, BOM (LCSC), pick-and-place
  (CPL). That is the next action to actually order the board. LCSC part numbers are preserved in
  the footprint property fields for automatic BOM mapping.
