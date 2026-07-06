"""Generate the omnimarble-driver KiCad 10 schematic (root + 7 sheets).

Connectivity is global-label-per-pin (see kicad_sch.py). Placement is a
coarse readability grid only. Every sheet is ERC-gated by run_erc.py /
validate.py; this script only authors the files.

Run: uv run python hardware/scripts/gen_schematic.py
"""

import sys
from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))

from kicad_sch import Schematic, Sym, dump, uid
from parts import PARTS

PROJ_DIR = HW_DIR / "omnimarble-driver"
PROJECT = "omnimarble-driver"

# ---------------------------------------------------------------------------
# Footprints (KiCad bundled libs; PCB phase validates they exist)
# ---------------------------------------------------------------------------

FP = {
    "R0603": "Resistor_SMD:R_0603_1608Metric",
    "R2512": "Resistor_SMD:R_2512_6332Metric",
    "C0603": "Capacitor_SMD:C_0603_1608Metric",
    "C0805": "Capacitor_SMD:C_0805_2012Metric",
    "CP_SMD_D10": "Capacitor_SMD:CP_Elec_10x10.5",
    "CP_SNAP": "Capacitor_THT:CP_Radial_D30.0mm_P10.00mm_SnapIn",
    "SOD123": "Diode_SMD:D_SOD-123",
    "SMA": "Diode_SMD:D_SMA",
    "SMB": "Diode_SMD:D_SMB",
    "TO263": "omnimarble:D2PAK_MBR60100DC",
    "TOLL": "omnimarble:TOLL-8",
    "TO252": "Package_TO_SOT_SMD:TO-252-2",
    "SOT23": "Package_TO_SOT_SMD:SOT-23",
    "SOT89": "Package_TO_SOT_SMD:SOT-89-3",
    "TSOT23_6": "Package_TO_SOT_SMD:TSOT-23-6",
    "SC70_5": "Package_TO_SOT_SMD:SOT-353_SC-70-5",
    "SOIC8": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
    "SOIC14": "Package_SO:SOIC-14_3.9x8.7mm_P1.27mm",
    "TSSOP8": "Package_SO:TSSOP-8_3x3mm_P0.65mm",
    "VSSOP8": "Package_SO:VSSOP-8_2.3x2mm_P0.5mm",
    "IND1265": "Inductor_SMD:L_12x12mm_H8mm",
    "IND6045": "Inductor_SMD:L_Changjiang_FNR6045S",
    "CEMENT5W": "Resistor_THT:R_Box_L26.0mm_W5.0mm_P20.00mm",
    "CEMENT10W": "Resistor_THT:R_Radial_Power_L11.0mm_W7.0mm_P5.00mm",
    "CP_SMD_D63": "Capacitor_SMD:CP_Elec_6.3x7.7",
    "SHUNT5930": "omnimarble:Shunt_5930",
    "RELAY": "omnimarble:Relay_HF3FF_T73",
    "BARREL": "Connector_BarrelJack:BarrelJack_Horizontal",
    "TERM2": "omnimarble:TerminalBarrier_1x02_P9.50mm",
    "IDC10": "Connector_IDC:IDC-Header_2x05_P2.54mm_Vertical",
    "JSTXH4": "Connector_JST:JST_XH_B4B-XH-A_1x04_P2.50mm_Vertical",
    "HDR1X20": "Connector_PinSocket_2.54mm:PinSocket_1x20_P2.54mm_Vertical",
    "HDR2X13": "Connector_PinHeader_2.54mm:PinHeader_2x13_P2.54mm_Vertical",
    "HDR1X2": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
    "HDR1X3": "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
    "FUSE_MINI": "Fuse:Fuseholder_Blade_Mini_Keystone_3568",
    "LED0603": "LED_SMD:LED_0603_1608Metric",
    "NETTIE": "NetTie:NetTie-2_SMD_Pad0.5mm",
    "BUZZER": "Buzzer_Beeper:Buzzer_12x9.5RM7.6",
    "IR5MM": "LED_THT:LED_D5.0mm",
    "IR3MM": "LED_THT:LED_D3.0mm",
}

# JLC Basic passives (LCSC C-numbers; blank = fill at order review)
RVALS = {
    "10R": "C22859", "1k": "C21190", "4.7k": "C23162", "5.1k": "C23186",
    "10k": "C25804", "22k": "C31850", "27k": "C22967", "47k": "C25819",
    "9.1k": "C23260", "4.3k": "C23159", "150": "C22808",
    "100k": "C25803", "220k": "C22961", "2k": "C22975", "51k": "C23196",
    "0.05R-2512": "C2994645", "0.15R-2512": "C2903485", "4.7R": "C146897",
    "47R-5W": "C5807995", "100R-10W": "C216413",
    # R24 bank-bleed: 6.8k in a 2512 land. The "2W" spec was inconsistent -- a
    # 2512 is a ~1W package and every stocked 6.8k 2512 on LCSC is 1W (the 2W
    # variants are out of stock). Actual dissipation is 55^2/6800 = 0.44W, so a
    # 1W 2512 carries it with 2.2x margin. Uses C26073 (UNI-ROYAL 25121WJ0682T4E,
    # 6.8k 2512 1W +/-5%) -- same UNI-ROYAL family as the board's other Rs,
    # in stock. JLC-placed; no longer hand-install.
    "6.8k": "C26073",
}
CVALS = {
    "100n": "C14663", "1u": "C15849", "1n": "C1588", "10n": "C57112",
    "2.2n": "C1604", "10u": "C15850", "100u-63V": "C28241", "100u-35V": "C176666",
}


class Counter:
    def __init__(self):
        self.n = {}

    def __call__(self, prefix):
        self.n[prefix] = self.n.get(prefix, 0) + 1
        return f"{prefix}{self.n[prefix]}"


def R(s, c, value, at, n1, n2, fp="R0603", ref=None):
    ref = ref or c("R")
    s.add_symbol("Device:R", ref, value, at, footprint=FP[fp],
                 lcsc=RVALS.get(value, ""), nets={"1": n1, "2": n2})
    return ref


def C_(s, c, value, at, n1, n2, fp="C0603"):
    ref = c("C")
    s.add_symbol("Device:C", ref, value, at, footprint=FP[fp],
                 lcsc=CVALS.get(value, ""), nets={"1": n1, "2": n2})
    return ref


def D_(s, c, lib, value, at, anode, cathode, fp, lcsc=""):
    ref = c("D")
    # Device:D* pin 1 = K (cathode), pin 2 = A (anode)
    s.add_symbol(lib, ref, value, at, footprint=FP[fp], lcsc=lcsc,
                 nets={"1": cathode, "2": anode})
    return ref


def LED_(s, c, color, at, anode, cathode, lcsc):
    ref = c("D")
    s.add_symbol("Device:LED", ref, f"LED_{color}", at, footprint=FP["LED0603"],
                 lcsc=lcsc, nets={"1": cathode, "2": anode})


def NPN(s, c, at, base, collector, emitter):
    ref = c("Q")
    # Transistor_BJT:MMBT3904: 1=B 2=E 3=C
    s.add_symbol("Transistor_BJT:MMBT3904", ref, "MMBT3904", at,
                 footprint=FP["SOT23"], lcsc="C20526",
                 nets={"1": base, "2": emitter, "3": collector})
    return ref


def GNDsym(s, c, at, net="GND"):
    s.power_symbol(net, at)


# ---------------------------------------------------------------------------
# Sheet 1: power input
# ---------------------------------------------------------------------------


GLOBAL_COUNTER = Counter()


def build_power_input(sch):
    s, c = sch, GLOBAL_COUNTER
    s.text("24V input, protection, 5V buck, 12V rail", (30, 20), 3)

    s.add_symbol("Connector:Barrel_Jack", "J1", "24V-IN", (40, 50),
                 footprint=FP["BARREL"], lcsc=PARTS["jack24"]["lcsc"],
                 nets={"1": "24V_JACK", "2": "GND"})
    s.add_symbol("Device:Fuse", "F1", "5A-MINI", (70, 45),
                 footprint=FP["FUSE_MINI"], lcsc=PARTS["fuse_holder"]["lcsc"],
                 nets={"1": "24V_JACK", "2": "24V_F"})
    # Reverse-polarity PFET: D=24V_F, S=+24V, G pulled to GND, zener G-S
    s.add_symbol("Transistor_FET:Q_PMOS_GDS", "Q90", "AOD4185", (100, 50),
                 footprint=FP["TO252"], lcsc=PARTS["pfet_rpp"]["lcsc"],
                 nets={"1": "RPP_G", "2": "24V_F", "3": "+24V"})
    R(s, c, "100k", (100, 80), "RPP_G", "GND")
    D_(s, c, "Device:D_Zener", "BZT52C15", (115, 80), "RPP_G", "+24V",
       "SOD123", PARTS["gate_clamp"]["lcsc"])
    s.add_symbol("Device:D_TVS", "D90", "SMBJ26CA", (135, 80), footprint=FP["SMB"],
                 lcsc=PARTS["tvs_in"]["lcsc"], nets={"1": "+24V", "2": "GND"})
    C_(s, c, "100u-35V", (150, 50), "+24V", "GND", fp="CP_SMD_D63")

    # 5V buck AP63205 (fixed 5V: FB senses VOUT)
    s.add_symbol("Regulator_Switching:AP63205WU", "U1", "AP63205", (60, 120),
                 footprint=FP["TSOT23_6"], lcsc=PARTS["buck5"]["lcsc"],
                 nets={"3": "+24V", "2": "+24V", "4": "GND",
                       "5": "BUCK_SW", "6": "BUCK_BST", "1": "+5V"})
    C_(s, c, "100n", (85, 110), "BUCK_SW", "BUCK_BST")
    s.add_symbol("Device:L", "L1", "10uH", (100, 115), footprint=FP["IND6045"],
                 lcsc=PARTS["ind_buck"]["lcsc"], nets={"1": "BUCK_SW", "2": "+5V"})
    C_(s, c, "10u", (120, 120), "+5V", "GND", fp="C0805")
    C_(s, c, "10u", (45, 120), "+24V", "GND", fp="C0805")

    # 12V linear rail (gate driver + relays, <60mA)
    # NOTE: 78L12 stays SOT-89 here. The thermal upgrade to a 78M12 in TO-252/
    # DPAK (Shikues C116325) is real and wanted, but the DPAK is larger and U2
    # sits in an auto-placement band -- swapping it reflows ~50 downstream parts
    # and desyncs the frozen route, so it needs a dedicated re-place/re-route
    # pass, not a local edit. Tracked as a follow-up.
    s.add_symbol("Regulator_Linear:L78L12_SOT89", "U2", "78L12", (60, 160),
                 footprint=FP["SOT89"], lcsc=PARTS["reg12"]["lcsc"],
                 nets={"3": "+24V", "1": "+12V", "2": "GND"})
    C_(s, c, "1u", (85, 160), "+12V", "GND")
    C_(s, c, "100n", (45, 160), "+24V", "GND")

    # 24V monitor divider + buffer (MCP6002 unit 2)
    R(s, c, "100k", (140, 120), "+24V", "V24_DIV")
    R(s, c, "10k", (140, 140), "V24_DIV", "GND")
    s.add_symbol("Amplifier_Operational:MCP6002-xSN", "U3", "MCP6002",
                 (170, 130), unit=2, footprint=FP["SOIC8"],
                 lcsc=PARTS["opamp"]["lcsc"],
                 nets={"5": "V24_DIV", "6": "V24_SENSE", "7": "V24_SENSE"})

    # Power flags: nets sourced by connectors/passives
    for i, net in enumerate(["+24V", "GND", "+5V", "3V3", "24V_JACK"]):
        s.power_flag(net, (40 + i * 25, 195))
        GNDsym(s, c, (40, 210)) if net == "GND" else None
    GNDsym(s, c, (170, 195))


# ---------------------------------------------------------------------------
# Sheet 2: boost charger (UC3843 boost, 24.5-55V operating, 63V-rated bank)
# ---------------------------------------------------------------------------


def build_boost_charger(sch):
    s, c = sch, GLOBAL_COUNTER
    s.text("Boost charger 24V -> 55V operating (63V-rated bank), OVP, relays, bleed", (30, 20), 3)

    u = "U4"
    s.add_symbol("Regulator_Controller:UC3843_SOIC8", u, "UC3843B", (60, 70),
                 footprint=FP["SOIC8"], lcsc=PARTS["boost_ctrl"]["lcsc"],
                 nets={"1": "BST_COMP", "2": "BST_FB", "3": "BST_CS",
                       "4": "BST_RC", "5": "GND", "6": "BST_DRV",
                       "7": "+12V", "8": "BST_VREF"})
    # Oscillator ~170kHz: RT 10k from VREF to RC, CT 1nF to GND
    R(s, c, "10k", (95, 55), "BST_VREF", "BST_RC")
    C_(s, c, "1n", (95, 75), "BST_RC", "GND")
    C_(s, c, "100n", (95, 95), "BST_VREF", "GND")
    C_(s, c, "100n", (40, 95), "+12V", "GND")
    # Compensation: COMP-FB RC
    R(s, c, "10k", (120, 55), "BST_COMP", "BST_COMP_Z")
    C_(s, c, "10n", (120, 75), "BST_COMP_Z", "BST_FB")

    # Power stage: +24V -> L2 -> SW node; FET low-side; SS510 -> VBOOST
    s.add_symbol("Device:L", "L2", "47uH", (40, 130), footprint=FP["IND1265"],
                 lcsc=PARTS["boost_ind"]["lcsc"], nets={"1": "+24V", "2": "BST_SW"})
    s.add_symbol("Transistor_FET:Q_NMOS_GDS", "Q91", "AOD66923", (70, 140),
                 footprint=FP["TO252"], lcsc=PARTS["boost_fet"]["lcsc"],
                 nets={"1": "BST_G", "2": "BST_SW", "3": "BST_CS_HI"})
    R(s, c, "10R", (55, 115), "BST_DRV", "BST_G")
    R(s, c, "0.15R-2512", (70, 170), "BST_CS_HI", "GND", fp="R2512")
    # CS filter
    R(s, c, "1k", (95, 150), "BST_CS_HI", "BST_CS")
    C_(s, c, "1n", (95, 170), "BST_CS", "GND")
    D_(s, c, "Device:D_Schottky", "SS510", (100, 130), "BST_SW", "VBOOST",
       "SMA", PARTS["boost_diode"]["lcsc"])
    C_(s, c, "100u-63V", (125, 140), "VBOOST", "GND", fp="CP_SMD_D10")
    C_(s, c, "100u-63V", (140, 140), "VBOOST", "GND", fp="CP_SMD_D10")
    R(s, c, "100k", (155, 140), "VBOOST", "GND")  # VBOOST local bleed

    # Feedback: FB(2.5V) node with Rt=100k from VBOOST, Rb=10k to GND, and
    # Rj=9.1k injection from the filtered MCU PWM. Transfer function (derived
    # and asserted in calcs.py):
    #   VBOOST = 2.5 + Rt*(2.5/Rb - (VSET-2.5)/Rj)
    #   VSET=0V   -> 55.0V  (= commanded MAX; PWM-stuck-low is a SAFE state,
    #                        below the worst-case OVP trip floor asserted in
    #                        calcs.py's protection-ladder checks)
    #   VSET=3.3V -> 18.7V  (below the ~24.5V boost floor -> converter idles)
    # NOTE the summing topology is inherently inverting (PWM duty UP ->
    # voltage DOWN); firmware maps duty = f(target) accordingly.
    R(s, c, "100k", (150, 60), "VBOOST", "BST_FB")
    R(s, c, "10k", (150, 80), "BST_FB", "GND")
    R(s, c, "9.1k", (175, 60), "VSET_FLT", "BST_FB")
    R(s, c, "10k", (175, 80), "VSET_PWM", "VSET_RC")
    C_(s, c, "1u", (175, 100), "VSET_RC", "GND")
    R(s, c, "10k", (190, 80), "VSET_RC", "VSET_FLT")
    C_(s, c, "1u", (190, 100), "VSET_FLT", "GND")

    # Hardware OVP: CJ431 (+/-0.5%) + 1% divider trips 60.5V nominal
    # (band + ladder ordering asserted in calcs.py against the same
    # constants: above the commanded max, below the cap rating); yanks COMP low,
    # independent of the MCU.
    R(s, c, "100k", (210, 60), "VBOOST", "OVP_REF")
    R(s, c, "4.3k", (210, 80), "OVP_REF", "GND")
    s.add_symbol("Reference_Voltage:TL431DBZ", "U5", "TL431B", (230, 70),
                 footprint=FP["SOT23"], lcsc=PARTS["tl431"]["lcsc"],
                 nets={"1": "BST_COMP", "2": "OVP_REF", "3": "GND"})

    # Boost enable is FAIL-SAFE INHIBITED: NPN base pulled up to +5V holds
    # COMP low (no switching) unless the MCU actively drives BOOST_EN_N low.
    # A dead, absent, or resetting Pico can never start a charge.
    NPN(s, c, (230, 110), "BSTINH_B", "BST_COMP", "GND")
    R(s, c, "10k", (245, 95), "+5V", "BSTINH_B")
    R(s, c, "1k", (210, 110), "BOOST_EN_N", "BSTINH_B")

    # Charge relay: VBOOST -> (NO contact) -> VBANK, 47R precharge across
    s.add_symbol("Relay:Relay_SPDT", "K1", "HF3FF-12V", (60, 210),
                 footprint=FP["RELAY"], lcsc=PARTS["relay"]["lcsc"],
                 nets={"A1": "+12V", "A2": "K1_COIL", "11": "VBOOST",
                       "14": "VBANK"}, no_connect=("12",))
    R(s, c, "47R-5W", (95, 200), "VBOOST", "VBANK", fp="CEMENT5W")
    NPN(s, c, (40, 240), "K1_B", "K1_COIL", "GND")
    R(s, c, "1k", (25, 230), "RLY_CHARGE", "K1_B")
    D_(s, c, "Device:D", "1N4148W", (60, 240), "K1_COIL", "+12V",
       "SOD123", "C81598")

    # Dump relay: VBANK -> (NO) -> 100R/25W -> GND
    s.add_symbol("Relay:Relay_SPDT", "K2", "HF3FF-12V", (140, 210),
                 footprint=FP["RELAY"], lcsc=PARTS["relay"]["lcsc"],
                 nets={"A1": "+12V", "A2": "K2_COIL", "11": "VBANK",
                       "14": "DUMP_R"}, no_connect=("12",))
    # Dump: 2x 100R/10W cement in series (=200R/20W; no single 20-25W part
    # stocked on LCSC). ~15W initial decaying pulse each, 5.3s to <5V.
    R(s, c, "100R-10W", (175, 205), "DUMP_R", "DUMP_MID", fp="CEMENT10W")
    R(s, c, "100R-10W", (175, 225), "DUMP_MID", "GND", fp="CEMENT10W")
    NPN(s, c, (120, 240), "K2_B", "K2_COIL", "GND")
    R(s, c, "1k", (105, 230), "RLY_DUMP", "K2_B")
    D_(s, c, "Device:D", "1N4148W", (140, 240), "K2_COIL", "+12V",
       "SOD123", "C81598")

    # Permanent bleed + live-bank LED (hardwired, works with logic dead)
    R(s, c, "6.8k", (200, 200), "VBANK", "GND", fp="R2512")  # bank bleed, 1W 2512
    R(s, c, "47k", (215, 200), "VBANK", "LIVE_LED")  # 47k: 0.064W @55V (10k->0.28W cooked 0603)
    LED_(s, c, "red", (215, 230), "LIVE_LED", "GND", "C2286")

    s.power_flag("VBOOST", (250, 200))
    s.power_flag("VBANK", (265, 200))
    GNDsym(s, c, (250, 240))


# ---------------------------------------------------------------------------
# Sheet 3: capacitor bank + V_bank sense
# ---------------------------------------------------------------------------


def build_cap_bank(sch):
    s, c = sch, GLOBAL_COUNTER
    s.text("Capacitor bank: 5x 2200uF/63V snap-in (populate per bank table)", (30, 20), 3)

    for i in range(5):
        s.add_symbol("Device:C_Polarized", f"C{90+i}", "2200u-63V",
                     (40 + i * 25, 60), footprint=FP["CP_SNAP"],
                     lcsc=PARTS["cap_bank"]["lcsc"],
                     nets={"1": "VBANK", "2": "GND"},
                     dnp=(i >= 2))  # baseline: 2 populated
    # Aux external bank: fuse + barrier terminal
    s.add_symbol("Connector:Screw_Terminal_01x02", "J2", "AUX-BANK", (180, 60),
                 footprint=FP["TERM2"], lcsc=PARTS["term_coil"]["lcsc"],
                 nets={"1": "AUX_BANK", "2": "GND"})
    s.add_symbol("Device:Fuse", "F2", "30A-MINI", (180, 90),
                 footprint=FP["FUSE_MINI"], lcsc=PARTS["fuse_holder"]["lcsc"],
                 nets={"1": "AUX_BANK", "2": "VBANK"})

    # V_bank divider + buffer (MCP6002 unit 1)
    R(s, c, "100k", (60, 120), "VBANK", "VBANK_DIV")
    R(s, c, "5.1k", (60, 140), "VBANK_DIV", "GND")
    C_(s, c, "100n", (75, 140), "VBANK_DIV", "GND")
    s.add_symbol("Amplifier_Operational:MCP6002-xSN", "U3", "MCP6002",
                 (100, 130), unit=1, footprint=FP["SOIC8"],
                 lcsc=PARTS["opamp"]["lcsc"],
                 nets={"3": "VBANK_DIV", "2": "VBANK_SENSE", "1": "VBANK_SENSE"})
    # MCP6002 power unit (shared with sheet 1 usage; declared once here)
    s.add_symbol("Amplifier_Operational:MCP6002-xSN", "U3", "MCP6002",
                 (140, 130), unit=3, footprint=FP["SOIC8"],
                 lcsc=PARTS["opamp"]["lcsc"],
                 nets={"8": "3V3", "4": "GND"})
    C_(s, c, "100n", (160, 130), "3V3", "GND")
    GNDsym(s, c, (60, 170))


# ---------------------------------------------------------------------------
# Sheet 4: pulse switch + ARM interlock
# ---------------------------------------------------------------------------


def build_pulse_switch(sch):
    s, c = sch, GLOBAL_COUNTER
    s.text("Pulse switch: 3x SFT040N150C3, UCC27524A drive, ARM interlock", (30, 20), 3)

    # ARM chain: +12V -> E-STOP (NC loop) -> keyswitch -> +12V_SW
    s.add_symbol("Connector_Generic:Conn_01x02", "J3", "E-STOP-NC", (40, 60),
                 footprint=FP["HDR1X2"], lcsc="",
                 nets={"1": "+12V", "2": "ESTOP_OK"})
    s.add_symbol("Connector_Generic:Conn_01x03", "J4", "ARM-KEY", (70, 60),
                 footprint=FP["HDR1X3"], lcsc="",
                 nets={"1": "ESTOP_OK", "2": "+12V_SW", "3": "GND"})
    # Sense dividers (12V -> 3.3V-safe); the divider midpoints ARE the
    # MCU sense nets, and ARM_SENSE also feeds the interlock AND gate
    R(s, c, "27k", (100, 50), "ESTOP_OK", "ESTOP_SENSE")
    R(s, c, "10k", (100, 70), "ESTOP_SENSE", "GND")
    R(s, c, "27k", (125, 50), "+12V_SW", "ARM_SENSE")
    R(s, c, "10k", (125, 70), "ARM_SENSE", "GND")
    C_(s, c, "10u", (150, 60), "+12V_SW", "GND", fp="C0805")
    C_(s, c, "100u-35V", (165, 60), "+12V_SW", "GND", fp="CP_SMD_D63")

    # Hardware interlock: FIRE_GATED = MCU_FIRE AND ARM_DIV(3.3V level)
    s.add_symbol("74xGxx:74LVC1G08", "U6", "74LVC1G08", (60, 110),
                 footprint=FP["SC70_5"], lcsc=PARTS["and_gate"]["lcsc"],
                 nets={"1": "MCU_FIRE", "2": "ARM_SENSE", "4": "FIRE_GATED",
                       "5": "3V3", "3": "GND"})
    C_(s, c, "100n", (85, 110), "3V3", "GND")
    # Fail-safe defaults: FIRE nets are pulled LOW so an absent/resetting
    # Pico or an unpowered AND gate cannot assert fire. (UCC27524A inputs
    # also have internal pulldowns per datasheet — this is belt-and-braces.)
    R(s, c, "10k", (35, 130), "MCU_FIRE", "GND")
    R(s, c, "10k", (85, 130), "FIRE_GATED", "GND")
    s.text("SAFETY NOTES:\n"
           "- E-STOP (J3) is a NORMALLY-CLOSED loop in series with the ARM key;\n"
           "  ship with jumper closed if no E-STOP fitted. Open = drivers unpowered.\n"
           "- ARM_SENSE divider bottom (10k) holds sense LOW if key/loop open.\n"
           "- Q10-Q12 MUST be same date/lot code; layout: mirror-symmetric\n"
           "  drain/source copper, equal-length gate loops (<25mm), driver GND\n"
           "  returns to the FET source rail at the shunt star point.",
           (170, 210), 2.0)

    # Gate drivers: 2x UCC27524A, VDD on the ARMed 12V rail.
    # DRVx = driver outputs to the pulse FETs (IR-gate signals are GATEx)
    for i, (ref, oa, ob) in enumerate([("U7", "DRV1", "DRV2"),
                                       ("U8", "DRV3", "DRV_SPARE")]):
        s.add_symbol("Driver_FET:UCC27524D", ref, "UCC27524A", (60 + i * 70, 150),
                     footprint=FP["SOIC8"], lcsc=PARTS["gate_drv"]["lcsc"],
                     nets={"1": "+12V_SW", "8": "+12V_SW",   # EN pins high
                           "2": "FIRE_GATED", "4": "FIRE_GATED",
                           "6": "+12V_SW", "3": "GND",
                           "7": oa, "5": ob})
        C_(s, c, "1u", (60 + i * 70 + 20, 145), "+12V_SW", "GND")
    # DRV_SPARE terminates on a test point so the net isn't single-ended
    s.add_symbol("Connector:TestPoint", "TP1", "DRV_SPARE", (215, 150),
                 footprint="TestPoint:TestPoint_Pad_1.5x1.5mm",
                 nets={"1": "DRV_SPARE"}, in_bom=False)

    # 3x pulse FET with per-gate R, pulldown, zener clamp
    for i in range(3):
        x = 50 + i * 55
        g_net, gd_net = f"QG{i+1}", f"DRV{i+1}"
        s.add_symbol("Transistor_FET:Q_NMOS_GDS", f"Q{10+i}", "SFT040N150C3",
                     (x, 210), footprint=FP["TOLL"],
                     lcsc=PARTS["fet_pulse"]["lcsc"],
                     nets={"1": g_net, "2": "SW_DRAIN", "3": "SHUNT_HI"})
        R(s, c, "10R", (x - 15, 195), gd_net, g_net)
        R(s, c, "10k", (x + 15, 225), g_net, "SHUNT_HI")
        D_(s, c, "Device:D_Zener", "BZT52C15", (x + 15, 245), "SHUNT_HI",
           g_net, "SOD123", PARTS["gate_clamp"]["lcsc"])
        # RC snubber D-S
        R(s, c, "4.7R", (x - 15, 235), "SW_DRAIN", f"SNB{i+1}", fp="R2512")
        C_(s, c, "10n", (x - 15, 255), f"SNB{i+1}", "SHUNT_HI")
    GNDsym(s, c, (200, 250))
    s.power_flag("+12V_SW", (220, 60))


# ---------------------------------------------------------------------------
# Sheet 5: pulse loop — blocking/freewheel/clamp diodes, coil, shunt
# ---------------------------------------------------------------------------


def build_flyback(sch):
    import json
    s, c = sch, GLOBAL_COUNTER
    s.text("Pulse loop: VBANK -> blocking -> COIL -> switch -> shunt -> GND", (30, 20), 3)

    # Coil requirements note, values sourced from calcs.py's swept envelope
    # (hardware/coil-envelope.json) so the schematic cannot drift from the
    # asserted numbers. Run calcs.py before gen_schematic.py.
    env = json.loads((HW_DIR / "coil-envelope.json").read_text(encoding="utf-8"))
    demo = env["demo_coil"]
    s.text("COIL REQUIREMENTS (J5) - validated envelope, swept in calcs.py at "
           f"{env['worst_bank_uF']}uF / {env['v_bank_max']:.0f}V:\n"
           f"  L >= {env['L_min_uH']} uH  AND  R_total >= "
           f"{env['R_total_min_ohm']*1000:.0f} mOhm\n"
           "  (R_total = coil DC resistance + leads/lugs; keep leads short)\n"
           f"  Validated default: 30T demo coil, {demo['L_uH']} uH, "
           f"{demo['R_total_ohm']*1000:.0f} mOhm total\n"
           f"  Below envelope the pulse can exceed the {env['design_i_pk_a']:.0f}A "
           "switch design point - DO NOT FIRE.",
           (30, 180), 2.2)

    # Series blocking diodes (2x parallel): reproduce sim's I>=0 clamp
    for i in range(2):
        D_(s, c, "Device:D_Schottky", "MBR60100DC", (50 + i * 30, 60),
           "VBANK", "COIL_HI", "TO263", PARTS["diode_pulse"]["lcsc"])
    # Coil terminal (32A barrier x2 positions bolted for pulse duty)
    s.add_symbol("Connector:Screw_Terminal_01x02", "J5", "COIL", (130, 60),
                 footprint=FP["TERM2"], lcsc=PARTS["term_coil"]["lcsc"],
                 nets={"1": "COIL_HI", "2": "SW_DRAIN"})
    # Freewheel diodes (2x): SW_DRAIN -> COIL_HI (conduct at FET turn-off)
    for i in range(2):
        D_(s, c, "Device:D_Schottky", "MBR60100DC", (50 + i * 30, 100),
           "SW_DRAIN", "COIL_HI", "TO263", PARTS["diode_pulse"]["lcsc"])
    # Bank reverse-clamp: GND -> VBANK (catches the -19V ring)
    D_(s, c, "Device:D_Schottky", "MBR60100DC", (130, 100),
       "GND", "VBANK", "TO263", PARTS["diode_pulse"]["lcsc"])

    # Shunt with Kelvin net ties: force nets join sense nets only at pads
    s.add_symbol("Device:R", "RS1", "0.2m", (60, 150),
                 footprint=FP["SHUNT5930"], lcsc=PARTS["shunt"]["lcsc"],
                 nets={"1": "SHUNT_HI", "2": "GND"})
    s.add_symbol("Device:NetTie_2", "NT1", "NetTie", (95, 140),
                 footprint=FP["NETTIE"], nets={"1": "SHUNT_HI", "2": "ISNS_P"},
                 in_bom=False)
    s.add_symbol("Device:NetTie_2", "NT2", "NetTie", (95, 165),
                 footprint=FP["NETTIE"], nets={"1": "GND", "2": "ISNS_N"},
                 in_bom=False)
    GNDsym(s, c, (170, 150))


# ---------------------------------------------------------------------------
# Sheet 6: current sense + ADC + IR gate comparators
# ---------------------------------------------------------------------------


def build_sense_gates(sch):
    s, c = sch, GLOBAL_COUNTER
    s.text("INA240 + ADS7042 current capture; 6x IR gate comparators", (30, 20), 3)

    s.add_symbol("Amplifier_Current:INA240A1PW", "U9", "INA240A1", (60, 60),
                 footprint=FP["TSSOP8"], lcsc=PARTS["ina240"]["lcsc"],
                 nets={"2": "ISNS_P", "3": "ISNS_N", "5": "3V3",
                       "1": "GND", "4": "GND", "6": "GND", "7": "GND",
                       "8": "IMON"})
    C_(s, c, "100n", (85, 55), "3V3", "GND")
    R(s, c, "10R", (100, 60), "IMON", "IMON_F")
    C_(s, c, "1n", (115, 75), "IMON_F", "GND")

    s.add_symbol("Analog_ADC:ADS7042xDCU", "U10", "ADS7042", (150, 60),
                 footprint=FP["VSSOP8"], lcsc=PARTS["adc"]["lcsc"],
                 nets={"6": "IMON_F", "5": "GND", "1": "3V3", "7": "3V3",
                       "8": "GND", "2": "ADC_SCK", "3": "ADC_SDO",
                       "4": "ADC_CS"})
    C_(s, c, "1u", (175, 55), "3V3", "GND")

    # IR gate rails: 2x IDC-10 (EMIT: power only; RECV: 6 signals)
    s.add_symbol("Connector_Generic:Conn_02x05_Odd_Even", "J6", "RAIL-EMIT",
                 (40, 120), footprint=FP["IDC10"], lcsc=PARTS["idc10"]["lcsc"],
                 nets={"1": "+5V", "2": "GND", "3": "+5V", "4": "GND",
                       "5": "+5V", "6": "GND", "7": "+5V", "8": "GND",
                       "9": "+5V", "10": "GND"})
    s.add_symbol("Connector_Generic:Conn_02x05_Odd_Even", "J7", "RAIL-RECV",
                 (80, 120), footprint=FP["IDC10"], lcsc=PARTS["idc10"]["lcsc"],
                 nets={"1": "+5V", "2": "GND",
                       "3": "SIG1", "4": "SIG2", "5": "SIG3",
                       "6": "SIG4", "7": "SIG5", "8": "SIG6",
                       "9": "SIG_SPARE", "10": "GND"})
    s.add_symbol("Connector_Generic:Conn_01x04", "J8", "SPARE-JST",
                 (120, 120), footprint=FP["JSTXH4"], lcsc=PARTS["jst_xh4"]["lcsc"],
                 nets={"1": "+5V", "2": "GND", "3": "SIG_SPARE", "4": "GND"})

    # Comparator threshold ~2.5V with hysteresis; SIGx pulled up to 5V,
    # phototransistor pulls low when beam present; beam broken -> high -> GATEx
    R(s, c, "10k", (160, 110), "+5V", "VTH_GATE")
    R(s, c, "10k", (160, 130), "VTH_GATE", "GND")
    C_(s, c, "100n", (175, 130), "VTH_GATE", "GND")

    s.text("IR GATE TRUTH TABLE:\n"
           "beam PRESENT -> phototransistor ON  -> SIGx ~0V  -> comparator OUT LOW\n"
           "beam BROKEN  -> phototransistor OFF -> SIGx ~5V  -> comparator OUT HIGH\n"
           "GATEx HIGH = marble in beam. Firmware timestamps the RISING edge.\n"
           "(open-collector outputs pulled to 3V3: Pico-safe levels)",
           (40, 285), 2.0)

    lm339_pins = {  # unit -> (out, minus, plus)
        1: ("2", "4", "5"), 2: ("1", "6", "7"),
        3: ("13", "10", "11"), 4: ("14", "8", "9"),
    }
    for i in range(6):
        pkg = "U11" if i < 4 else "U12"
        unit = (i % 4) + 1
        out, minus, plus = lm339_pins[unit]
        x = 40 + i * 35
        R(s, c, "10k", (x, 170), "+5V", f"SIG{i+1}")          # pull-up
        R(s, c, "1k", (x, 190), f"SIG{i+1}", f"CIN{i+1}")     # RC filter
        C_(s, c, "1n", (x + 12, 205), f"CIN{i+1}", "GND")
        s.add_symbol("Comparator:LM339", pkg, "LM339", (x, 225), unit=unit,
                     footprint=FP["SOIC14"], lcsc=PARTS["comparator"]["lcsc"],
                     nets={plus: f"CIN{i+1}", minus: "VTH_GATE",
                           out: f"GATE{i+1}"})
        R(s, c, "10k", (x, 250), "3V3", f"GATE{i+1}")         # OC pull-up
        R(s, c, "220k", (x + 12, 250), f"CIN{i+1}", f"GATE{i+1}")  # hysteresis

    # LM339 power units + unused unit tie-off (U12 units 3,4 unused)
    s.add_symbol("Comparator:LM339", "U11", "LM339", (255, 170), unit=5,
                 footprint=FP["SOIC14"], lcsc=PARTS["comparator"]["lcsc"],
                 nets={"3": "+5V", "12": "GND"})
    s.add_symbol("Comparator:LM339", "U12", "LM339", (255, 200), unit=5,
                 footprint=FP["SOIC14"], lcsc=PARTS["comparator"]["lcsc"],
                 nets={"3": "+5V", "12": "GND"})
    for unit in (3, 4):
        out, minus, plus = lm339_pins[unit]
        s.add_symbol("Comparator:LM339", "U12", "LM339", (255, 225 + (unit - 3) * 30),
                     unit=unit, footprint=FP["SOIC14"],
                     lcsc=PARTS["comparator"]["lcsc"],
                     nets={plus: "GND", minus: "3V3"}, no_connect=(out,))
    C_(s, c, "100n", (285, 170), "+5V", "GND")
    C_(s, c, "100n", (285, 200), "+5V", "GND")
    GNDsym(s, c, (285, 250))


# ---------------------------------------------------------------------------
# Sheet 7: Pico socket, breakout, status
# ---------------------------------------------------------------------------

PICO = {
    1: "GP0_NC", 2: "GP1_NC", 3: "GND", 4: "GATE1", 5: "GATE2", 6: "GATE3",
    7: "GATE4", 8: "GND", 9: "GATE5", 10: "GATE6", 11: "MCU_FIRE",
    12: "ARM_SENSE", 13: "GND", 14: "RLY_CHARGE", 15: "RLY_DUMP",
    16: "BOOST_EN_N", 17: "VSET_PWM", 18: "GND", 19: "BUZZ", 20: "ESTOP_SENSE",
    21: "ADC_SDO", 22: "ADC_CS", 23: "GND", 24: "ADC_SCK", 25: "GP19_NC",
    26: "LED_ARMED", 27: "LED_CHG", 28: "GND", 29: "GP22_NC", 30: "RUN_NC",
    31: "VBANK_SENSE", 32: "V24_SENSE", 33: "GND", 34: "GP28_NC",
    35: "ADCVREF_NC", 36: "3V3", 37: "3V3EN_NC", 38: "GND", 39: "+5V",
    40: "VBUS_NC",
}

BREAKOUT = [
    "GATE1", "GATE2", "GATE3", "GATE4", "GATE5", "GATE6",
    "MCU_FIRE", "ARM_SENSE", "RLY_CHARGE", "RLY_DUMP", "BOOST_EN_N",
    "VSET_PWM", "BUZZ", "ESTOP_SENSE", "ADC_SDO", "ADC_CS", "ADC_SCK",
    "VBANK_SENSE", "V24_SENSE", "LED_ARMED", "LED_CHG", "SIG_SPARE",
    "3V3", "+5V", "GND", "GND",
]


def build_mcu_status(sch):
    s, c = sch, GLOBAL_COUNTER
    s.text("Pico socket (2x 1x20), MCU-agnostic breakout, relay/status drive", (30, 20), 3)

    def pico_net(pin):
        net = PICO[pin]
        return None if net.endswith("_NC") else net

    for jref, pins, x in [("J9", range(1, 21), 50), ("J10", range(21, 41), 90)]:
        nets, ncs = {}, []
        for idx, pico_pin in enumerate(pins, start=1):
            net = pico_net(pico_pin)
            if net:
                nets[str(idx)] = net
            else:
                ncs.append(str(idx))
        s.add_symbol("Connector_Generic:Conn_01x20", jref, "PICO-SOCKET",
                     (x, 100), footprint=FP["HDR1X20"],
                     lcsc=PARTS["pico_socket"]["lcsc"],
                     nets=nets, no_connect=tuple(ncs))

    # 2x13 IO breakout (J11) REMOVED: it duplicated all 26 control/sense nets
    # into the congested MCU/sense area, blocking auto-routing. MCU-agnostic
    # access is preserved via the Pico socket pins themselves (a different
    # 3.3V micro connects on a Pico-form-factor carrier or via jumper leads to
    # J9/J10). BREAKOUT[] list kept above for documentation of the pin map.

    # Status LEDs (MCU-driven)
    R(s, c, "1k", (210, 60), "LED_ARMED", "LEDA_K")
    LED_(s, c, "red", (210, 80), "LEDA_K", "GND", "C2286")
    R(s, c, "1k", (230, 60), "LED_CHG", "LEDC_K")
    LED_(s, c, "green", (230, 80), "LEDC_K", "GND", "C72043")

    # Buzzer via NPN
    NPN(s, c, (210, 130), "BUZZ_B", "BUZZ_DRV", "GND")
    R(s, c, "1k", (195, 120), "BUZZ", "BUZZ_B")
    s.add_symbol("Device:Buzzer", "BZ1", "buzzer-5V", (230, 130),
                 footprint=FP["BUZZER"], lcsc=PARTS["buzzer"]["lcsc"],
                 nets={"1": "+5V", "2": "BUZZ_DRV"})  # +(pin1)->+5V, -(pin2)->NPN collector
    D_(s, c, "Device:D", "1N4148W", (250, 130), "BUZZ_DRV", "+5V",
       "SOD123", "C81598")
    GNDsym(s, c, (210, 160))


# ---------------------------------------------------------------------------
# Root sheet + project file
# ---------------------------------------------------------------------------

SHEETS = [
    ("power_input", build_power_input),
    ("boost_charger", build_boost_charger),
    ("cap_bank", build_cap_bank),
    ("pulse_switch", build_pulse_switch),
    ("flyback", build_flyback),
    ("sense_gates", build_sense_gates),
    ("mcu_status", build_mcu_status),
]


def build_root(sheet_plan, root_uuid):
    root = Schematic("OmniMarble 60V SELV coilgun driver", project=PROJECT)
    root.sheet_uuid = root_uuid
    doc = root.to_sexpr()
    for i, (name, sheet_sym_uuid) in enumerate(sheet_plan):
        x, y = 30 + (i % 4) * 65, 40 + (i // 4) * 50
        sheet = [Sym("sheet"),
                 [Sym("at"), x, y], [Sym("size"), 50, 30],
                 [Sym("exclude_from_sim"), Sym("no")],
                 [Sym("in_bom"), Sym("yes")], [Sym("on_board"), Sym("yes")],
                 [Sym("dnp"), Sym("no")],
                 [Sym("fields_autoplaced"), Sym("yes")],
                 [Sym("stroke"), [Sym("width"), 0.1524],
                  [Sym("type"), Sym("solid")]],
                 [Sym("fill"), [Sym("color"), 0, 0, 0, 0.0]],
                 [Sym("uuid"), sheet_sym_uuid],
                 [Sym("property"), "Sheetname", name,
                  [Sym("at"), x, y - 1, 0],
                  [Sym("effects"), [Sym("font"), [Sym("size"), 1.27, 1.27]],
                   [Sym("justify"), Sym("left"), Sym("bottom")]]],
                 [Sym("property"), "Sheetfile", f"{name}.kicad_sch",
                  [Sym("at"), x, y + 31, 0],
                  [Sym("effects"), [Sym("font"), [Sym("size"), 1.27, 1.27]],
                   [Sym("justify"), Sym("left"), Sym("top")]]],
                 [Sym("instances"),
                  [Sym("project"), PROJECT,
                   [Sym("path"), f"/{root_uuid}",
                    [Sym("page"), str(i + 2)]]]],
                 ]
        doc.insert(-1, sheet)  # before embedded_fonts
    (PROJ_DIR / f"{PROJECT}.kicad_sch").write_text(dump(doc) + "\n",
                                                   encoding="utf-8")


def write_project():
    # Include the board design rules so kicad-cli DRC uses them instead of
    # KiCad's stricter defaults (default min via 0.5 / hole 0.3 would flag the
    # 0.45/0.25 GND via-in-pad stitching). JLC-manufacturable at 0.4/0.2.
    import json as _json
    pro = {
        "board": {
            "design_settings": {
                "rules": {
                    "min_via_diameter": 0.4,
                    "min_through_hole_diameter": 0.2,
                    "min_hole_to_hole": 0.25,
                    # 0.15mm = JLC 2oz multilayer min trace width; matches the
                    # narrowed ISNS Kelvin pair (gen_pcb.local_finish).
                    "min_track_width": 0.15,
                }
            }
        },
        "meta": {"filename": f"{PROJECT}.kicad_pro", "version": 3},
        "sheets": [],
    }
    (PROJ_DIR / f"{PROJECT}.kicad_pro").write_text(
        _json.dumps(pro, indent=2) + "\n", encoding="utf-8")


def main():
    PROJ_DIR.mkdir(parents=True, exist_ok=True)
    # Fix the hierarchy uuids up front so child symbol instances carry the
    # correct "/<root>/<sheet-symbol>" paths
    root_uuid = uid()
    sheet_plan = [(name, uid()) for name, _ in SHEETS]
    for (name, builder), (_, sheet_sym_uuid) in zip(SHEETS, sheet_plan):
        sch = Schematic(f"omnimarble-driver / {name}", project=PROJECT,
                        instance_path=f"/{root_uuid}/{sheet_sym_uuid}")
        builder(sch)
        sch.write(PROJ_DIR / f"{name}.kicad_sch")
        print(f"wrote {name}.kicad_sch")
    build_root(sheet_plan, root_uuid)
    write_project()
    print(f"wrote {PROJECT}.kicad_sch (root) + {PROJECT}.kicad_pro")


if __name__ == "__main__":
    main()
