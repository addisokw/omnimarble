"""Generate the vbench POWER board schematic (hardware/vbench-pwr/).

The high-current half of the SIM-validation rig: an expandable snap-in
capacitor bank dumped through a hand-wound coil by a low-side MOSFET switch,
with a freewheel diode (matches the sim's has_flyback_diode topology), a
Kelvin shunt for scope-captured I(t), and a bank-voltage divider. Fires on a
position trigger from the logic board (FIRE); cutoff timing is non-critical
(the pulse self-terminates when the cap empties).

Topology (star ground at the bank negative):
  VBANK --[coil]--> SW --[Q1..3 drain]  gate=GATE  source=SRC
  SRC  --[shunt 0.2m]--> GND            (scope reads V across the shunt)
  freewheel D1: SW(anode) -> VBANK(cathode)   (RL freewheel, matches sim)
  gate driver U1 (UCC27524A): FIRE -> OUT -> Rg -> gates; Rgs holds off
  bank C1..C5 (populate 1->5); bleeder; charge + aux-bank barrier terminals

Run: "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" hardware/scripts/gen_vbench_pwr.py
"""

import sys
from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))

from kicad_sch import Schematic
from parts import PARTS

PROJ_DIR = HW_DIR / "vbench-pwr"
PROJECT = "vbench-pwr"

FP = {
    "CP_RAD18": "Capacitor_THT:CP_Radial_D18.0mm_P7.50mm",
    "TO247": "Package_TO_SOT_THT:TO-247-3_Vertical",
    "SOIC8": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
    "TO263": "omnimarble:D2PAK_MBR60100DC",
    "SHUNT5930": "omnimarble:Shunt_5930",
    "SOD123": "Diode_SMD:D_SOD-123",
    "R0805": "Resistor_SMD:R_0805_2012Metric",
    "R2512": "Resistor_SMD:R_2512_6332Metric",
    "C0805": "Capacitor_SMD:C_0805_2012Metric",
    "TERM2": "omnimarble:TerminalBarrier_1x02_P9.50mm",
    "HDR1X2": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
    "HDR1X3": "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
    "HDR1X7": "Connector_PinHeader_2.54mm:PinHeader_1x07_P2.54mm_Vertical",
    "PICO": "Connector_PinSocket_2.54mm:PinSocket_1x20_P2.54mm_Vertical",
}

# Known-good LCSC for the generic passives (bare board -> reference only)
RLCSC = {"10R": "C22859", "10k": "C25804", "100k": "C25803", "5.1k": "C23186"}


def main():
    PROJ_DIR.mkdir(parents=True, exist_ok=True)
    s = Schematic("vbench-pwr: expandable-bank single-coil pulse switch (SIM validation)",
                  project=PROJECT)
    s.text("POWER BOARD (2-layer 2oz). Bank + low-side FET switch + freewheel + shunt + V-sense.\n"
           "Charge <=55V from bench supply; coil is the load; FIRE from the logic board;\n"
           "ISENSE (raw shunt) to a scope. Star ground at the bank negative.", (30, 18), 2.2)

    def R(ref, value, at, n1, n2, fp="R0805", dnp=False, lcsc=None):
        s.add_symbol("Device:R", ref, value, at, footprint=FP[fp],
                     lcsc=RLCSC.get(value, "") if lcsc is None else lcsc,
                     nets={"1": n1, "2": n2}, dnp=dnp)

    def C(ref, value, at, n1, n2):
        s.add_symbol("Device:C", ref, value, at, footprint=FP["C0805"],
                     nets={"1": n1, "2": n2})

    # --- expandable cap bank: 5x snap-in, baseline 1 populated ----------------
    for i in range(5):
        s.add_symbol("Device:C_Polarized", f"C{i+1}", "2200u-63V",
                     (40 + i * 20, 55), footprint=FP["CP_RAD18"],
                     lcsc="C47858",   # Chengx 18mm radial 2200uF/63V (compact bank)
                     nets={"1": "VBANK", "2": "GND"}, dnp=(i >= 1))

    # --- switch: 3x parallel TO-247 IRFP4668 (baseline 1), per-FET gate R -----
    for i in range(3):
        s.add_symbol("Transistor_FET:Q_NMOS_GDS", f"Q{i+1}", "IRFP4668",
                     (45 + i * 40, 120), footprint=FP["TO247"],
                     nets={"1": f"QG{i+1}", "2": "SW", "3": "SRC"}, dnp=(i >= 1))
        R(f"RG{i+1}", "10R", (60 + i * 40, 105), "DRV_OUT", f"QG{i+1}", dnp=(i >= 1))

    # gate driver: use channel A only (5A drives the 3 gates fine — cutoff speed
    # is non-critical); channel B parked (INB->GND, OUTB NC, both EN high)
    s.add_symbol("Driver_FET:UCC27524D", "U1", "UCC27524A", (200, 120),
                 footprint=FP["SOIC8"], lcsc=PARTS["gate_drv"]["lcsc"],
                 nets={"1": "+12V", "2": "FIRE", "3": "GND", "4": "GND",
                       "6": "+12V", "7": "DRV_OUT", "8": "+12V"},
                 no_connect=("5",))
    R("RGS", "10k", (180, 140), "DRV_OUT", "SRC")           # gate pulldown
    R("RF", "100R", (170, 105), "FIRE_IN", "FIRE", lcsc="")  # FIRE series
    R("RFP", "10k", (185, 105), "FIRE", "GND")               # FIRE pulldown
    s.add_symbol("Device:D_Zener", "DZ1", "BZT52C15", (215, 145),
                 footprint=FP["SOD123"], lcsc=PARTS["gate_clamp"]["lcsc"],
                 nets={"1": "DRV_OUT", "2": "SRC"})           # gate clamp (K=1,A=2)
    C("C10", "1u", (230, 100), "+12V", "GND")                # VDD decoupling
    C("C11", "100n", (245, 100), "+12V", "GND")

    # --- freewheel diode across the coil (SW->VBANK), matches the sim ---------
    s.add_symbol("Device:D_Schottky", "D1", "MBR60100DC", (110, 90),
                 footprint=FP["TO263"], lcsc=PARTS["diode_pulse"]["lcsc"],
                 nets={"1": "VBANK", "2": "SW"})              # K=1 (VBANK), A=2 (SW)

    # --- current sense: Kelvin shunt in the source return --------------------
    s.add_symbol("Device:R", "RS1", "0.2m", (90, 170), footprint=FP["SHUNT5930"],
                 lcsc=PARTS["shunt"]["lcsc"], nets={"1": "SRC", "2": "GND"})

    # --- bank-voltage divider -> VBANK_SENSE (~2.67V at 55V) ------------------
    R("RD1", "100k", (270, 55), "VBANK", "VBANK_SENSE")
    R("RD2", "5.1k", (270, 75), "VBANK_SENSE", "GND")
    C("C12", "100n", (285, 65), "VBANK_SENSE", "GND")

    # --- bleeder -------------------------------------------------------------
    R("RB1", "6.8k", (300, 60), "VBANK", "GND", fp="R2512",
      lcsc=PARTS["r_bleed"]["lcsc"])

    # --- terminals + headers -------------------------------------------------
    s.add_symbol("Connector:Screw_Terminal_01x02", "J1", "COIL", (60, 85),
                 footprint=FP["TERM2"], lcsc=PARTS["term_coil"]["lcsc"],
                 nets={"1": "VBANK", "2": "SW"})
    s.add_symbol("Connector:Screw_Terminal_01x02", "J2", "CHARGE", (60, 200),
                 footprint=FP["TERM2"], lcsc=PARTS["term_coil"]["lcsc"],
                 nets={"1": "VBANK", "2": "GND"})
    s.add_symbol("Connector:Screw_Terminal_01x02", "J3", "AUX-BANK", (90, 200),
                 footprint=FP["TERM2"], lcsc=PARTS["term_coil"]["lcsc"],
                 nets={"1": "VBANK", "2": "GND"})
    s.add_symbol("Connector_Generic:Conn_01x02", "J4", "12V-IN", (200, 200),
                 footprint=FP["HDR1X2"], lcsc="",
                 nets={"1": "+12V", "2": "GND"})
    s.add_symbol("Connector_Generic:Conn_01x02", "J6", "ISENSE", (140, 200),
                 footprint=FP["HDR1X2"], lcsc="",
                 nets={"1": "SRC", "2": "GND"})

    # --- on-board Pico + 2 Waveshare sensor connectors (logic zone) ----------
    # Pico socket = two 1x20 rows. Left = physical pins 1..20; Right symbol pins
    # 1..20 = physical 21..40. FIRE_IN drives the gate-driver front-end above;
    # VBANK_SENSE reads the divider on an ADC pin. Waveshares at 3.3V (RP2040
    # is NOT 5V-tolerant). Unused Pico pins are no-connect.
    # Pico-L (GP0..15): WA/WB/WC = GP0-14 (15 lines), FIRE_IN = GP15. GNDs at 3/8/13/18.
    s.add_symbol("Connector_Generic:Conn_01x20", "J7", "PICO-L", (300, 60),
                 footprint=FP["PICO"], lcsc=PARTS["pico_socket"]["lcsc"],
                 nets={"1": "WA_D0", "2": "WA_D1", "3": "GND", "4": "WA_D2",
                       "5": "WA_D3", "6": "WA_D4", "7": "WB_D0", "8": "GND",
                       "9": "WB_D1", "10": "WB_D2", "11": "WB_D3", "12": "WB_D4",
                       "13": "GND", "14": "WC_D0", "15": "WC_D1", "16": "WC_D2",
                       "17": "WC_D3", "18": "GND", "19": "WC_D4", "20": "FIRE_IN"})
    # Pico-R (GP16..): WD = GP16-20, VBANK_SENSE = GP26/ADC0 (pin11), +3V3 (pin16).
    s.add_symbol("Connector_Generic:Conn_01x20", "J8", "PICO-R", (340, 60),
                 footprint=FP["PICO"], lcsc=PARTS["pico_socket"]["lcsc"],
                 nets={"1": "WD_D0", "2": "WD_D1", "3": "GND", "4": "WD_D2",
                       "5": "WD_D3", "6": "WD_D4", "8": "GND", "11": "VBANK_SENSE",
                       "13": "GND", "16": "+3V3", "18": "GND"},
                 no_connect=("7", "9", "10", "12", "14", "15", "17", "19", "20"))
    for j, tag, d in [("J9", "A", "WA"), ("J10", "B", "WB"),
                      ("J11", "C", "WC"), ("J12", "D", "WD")]:
        s.add_symbol("Connector_Generic:Conn_01x07", j, f"WAVESHARE-{tag}",
                     (280 + (int(j[1:]) - 9) * 25, 130), footprint=FP["HDR1X7"],
                     lcsc="", nets={"1": "+3V3", "2": "GND", "3": f"{d}_D0",
                                    "4": f"{d}_D1", "5": f"{d}_D2", "6": f"{d}_D3",
                                    "7": f"{d}_D4"})

    # --- ERC power sources ---------------------------------------------------
    s.power_flag("VBANK", (55, 205))
    s.power_flag("GND", (65, 210))
    s.power_flag("+12V", (200, 215))
    s.power_flag("+3V3", (300, 45))
    s.power_symbol("GND", (75, 210))

    s.write(PROJ_DIR / f"{PROJECT}.kicad_sch")
    (PROJ_DIR / f"{PROJECT}.kicad_pro").write_text(
        '{\n  "meta": { "filename": "%s.kicad_pro", "version": 3 },\n'
        '  "sheets": []\n}\n' % PROJECT, encoding="utf-8")
    print(f"wrote {PROJECT}.kicad_sch + .kicad_pro")


if __name__ == "__main__":
    main()
