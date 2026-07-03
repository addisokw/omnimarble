"""Generate the gate-rail sensor board schematic (hardware/gate-rail/).

One PCB design, two population variants (documented in hardware/README):
  EMIT: populate D1..D6 (IR333C-A) + R1..R6 series resistors
  RECV: populate Q1..Q6 (PT204-6B phototransistors)
Both circuits are drawn; assembly choice selects the variant. Six stations
sit at z = -60/-40/-20/+5/+60/+120 mm from the coil-center fiducial —
exact spacing is enforced in the PCB layout phase, checked by validate.py.

IDC-10 pinout (matches driver board J6/J7):
  1=+5V 2=GND 3..8=SIG1..6 9=SIG_SPARE 10=GND
(EMIT rails simply don't drive SIGx.)

Run: uv run python hardware/scripts/gen_gate_rail.py
"""

import sys
from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))

from kicad_sch import Schematic
from parts import PARTS

PROJ_DIR = HW_DIR / "gate-rail"
PROJECT = "gate-rail"

FP_IDC = "Connector_IDC:IDC-Header_2x05_P2.54mm_Vertical"
FP_R = "Resistor_SMD:R_0603_1608Metric"
FP_LED5 = "LED_THT:LED_D5.0mm"
FP_PT3 = "LED_THT:LED_D3.0mm"

# 5V through 150R -> ~22mA into an IR333C (Vf ~1.4V)
LED_R_VALUE = "150"
STATIONS_MM = [-60, -40, -20, 5, 60, 120]  # z from coil center (layout)


def main():
    PROJ_DIR.mkdir(parents=True, exist_ok=True)
    s = Schematic("gate-rail: IR break-beam sensor rail (EMIT/RECV variants)",
                  project=PROJECT)
    s.text("6 stations at z = -60/-40/-20/+5/+60/+120 mm from coil fiducial",
           (30, 20), 2.5)
    s.text("EMIT variant: populate Dx+Rx | RECV variant: populate Qx",
           (30, 28), 2.5)

    s.add_symbol("Connector_Generic:Conn_02x05_Odd_Even", "J1", "RAIL-IDC",
                 (40, 60), footprint=FP_IDC, lcsc=PARTS["idc10"]["lcsc"],
                 nets={"1": "+5V", "2": "GND",
                       "3": "SIG1", "4": "SIG2", "5": "SIG3",
                       "6": "SIG4", "7": "SIG5", "8": "SIG6",
                       "9": "SIG_SPARE", "10": "GND"})
    # SIG_SPARE terminates on a test pad (unused on the rail itself)
    s.add_symbol("Connector:TestPoint", "TP1", "SIG_SPARE", (80, 60),
                 footprint="TestPoint:TestPoint_Pad_1.5x1.5mm",
                 nets={"1": "SIG_SPARE"})

    for i in range(6):
        x = 40 + i * 35
        # EMIT: +5V -> R -> LED -> GND
        s.add_symbol("Device:R", f"R{i+1}", LED_R_VALUE, (x, 100),
                     footprint=FP_R, lcsc="",
                     nets={"1": "+5V", "2": f"LEDA{i+1}"})
        s.add_symbol("Device:LED", f"D{i+1}", "IR333C-A", (x, 130),
                     footprint=FP_LED5, lcsc=PARTS["ir_led"]["lcsc"],
                     nets={"2": f"LEDA{i+1}", "1": "GND"})
        # RECV: phototransistor C -> SIGx (pulled up on driver board), E -> GND
        s.add_symbol("Device:Q_Photo_NPN_CE", f"Q{i+1}", "PT204-6B", (x, 170),
                     footprint=FP_PT3, lcsc=PARTS["ir_pt"]["lcsc"],
                     nets={"1": f"SIG{i+1}", "2": "GND"})

    s.power_flag("+5V", (260, 60))
    s.power_flag("GND", (275, 60))
    s.power_symbol("GND", (260, 90))

    s.write(PROJ_DIR / f"{PROJECT}.kicad_sch")
    (PROJ_DIR / f"{PROJECT}.kicad_pro").write_text(
        '{\n  "meta": { "filename": "%s.kicad_pro", "version": 3 },\n'
        '  "sheets": []\n}\n' % PROJECT, encoding="utf-8")
    print(f"wrote {PROJECT}.kicad_sch + .kicad_pro")


if __name__ == "__main__":
    main()
