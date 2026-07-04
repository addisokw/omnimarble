"""Driver-board placement: manual pulse geometry + shelf-packed clusters.

Coordinates in mm, +y down, outline (0,0)-(220,140). rot deg CCW.
pcbnew rotation convention (empirically verified): positive rot maps a
local pad offset (x,y) -> (-y,x) in board coords. So for a two-pad part
with pad2 at local (+d,0): rot=90 puts pad2 NORTH, rot=270 puts it SOUTH.

Zone boundaries come from package pad windows (D2PAK tab 9.4x10.8, TOLL
tab 10.2x8.1) so straddling pads land fully inside their zones with
>= 1.0mm zone-zone gaps — machine-asserted by gen_pcb.py. Silicon band
sits at y>=42 to clear the D30 snap-in can rims (can centers y25,
r~15.3).

Vertical map:
  y12..22   GND strip (can negatives; bulk return via In1/In2 planes)
  y24..34   VBANK strip;  x8..25 y24..66 VBANK feeder column
  y42..68   COIL_HI island x27..48 | SW_DRAIN bus x50..126
  y66.5     TOLL FET row (tab north in SW, source leads south in SHUNT)
  y69.5..79 SHUNT_HI strip; shunt + Kelvin at x126..150
  y68..98   COIL/SW arms down to coil terminal J5 (left)
  y86..112  driver / sense / comparator shelf bands
  y96..124  boost | FB/OVP | buck shelf bands
  y113..140 bottom row: J1 J2 F2 K1 K2 R19 R21 R22 J6 J7 (+J8/J11 right)
"""

BOARD_W = 220.0
BOARD_H = 150.0

MOUNT_HOLES = [(6, 6), (214, 6), (6, 144), (214, 144)]

PULSE_ZONES = [
    ("GND", [(8, 12), (212, 12), (212, 22), (8, 22)]),
    ("GND", [(134, 66), (150, 66), (150, 86), (134, 86)]),   # shunt return
    ("VBANK", [(8, 24), (212, 24), (212, 34), (25, 34), (25, 66),
               (8, 66), (8, 34)]),
    # COIL_HI: main island + arm down to J5.1
    ("COIL_HI", [(27.1, 42), (48, 42), (48, 68), (27.1, 68)]),
    ("COIL_HI", [(27, 66), (38, 66), (38, 84), (27, 84)]),
    # SW_DRAIN: main bus + link column + low arm to J5.2
    ("SW_DRAIN", [(50.05, 42), (126, 42), (126, 68), (50.05, 68)]),
    ("SW_DRAIN", [(42, 70), (50.05, 70), (50.05, 66), (52, 66),
                  (52, 98), (42, 98)]),
    ("SW_DRAIN", [(27, 86), (50.05, 86), (50.05, 94), (27, 94)]),
    # SHUNT_HI strip (FET anchor y=66.5: leads 70.35..73.15)
    ("SHUNT_HI", [(66, 69.5), (131.5, 69.5), (131.5, 79), (66, 79)]),
]

P = {}

# --- pulse chain -----------------------------------------------------------
for i, ref in enumerate(["C90", "C91", "C92", "C93", "C94"]):
    P[ref] = (28 + i * 32, 30, 90)   # pad1(+)->VBANK y30, pad2(-)->GND y20

P["D9"] = (30.3, 47.5, 0)    # blocking: leads(A)->column, tab(K)->COIL
P["D10"] = (30.3, 60, 0)
P["D11"] = (44.75, 48, 180)  # freewheel: tab(K)->COIL, leads(A)->SW
P["D12"] = (44.75, 60, 180)
P["J5"] = (32, 80, 270)      # coil terminal: pin1(32,80) COIL arm,
                             #                pin2(32,89.5) SW low arm
P["Q10"] = (78, 66.5, 0)     # TOLL FETs: tab->SW(<=68), leads->SHUNT(69.5+)
P["Q11"] = (96, 66.5, 0)
P["Q12"] = (114, 66.5, 0)
P["RS1"] = (134, 74, 0)      # pad1->SHUNT_HI, pad2->GND patch
P["NT1"] = (126, 83, 0)      # Kelvin ties (explicit KELVIN_TRACKS)
P["NT2"] = (142, 83, 0)
P["D13"] = (14, 48, 90)      # bank clamp: tab->VBANK col, leads south->GND
P["R24"] = (18, 64, 270)     # 6.8k bleed: pad1 in col, pad2->GND cluster

GATE_RG = ["R34", "R37", "R40"]
GATE_RPD = ["R35", "R38", "R41"]
GATE_DZ = ["D6", "D7", "D8"]
SNUB_R = ["R36", "R39", "R42"]
SNUB_C = ["C23", "C24", "C25"]
for i in range(3):
    fx = 78 + i * 18
    P[GATE_RG[i]] = (fx - 4, 82.5, 90)    # 10R gate (QG north to gate pad)
    P[GATE_RPD[i]] = (fx + 4, 77.5, 90)   # 10k pulldown (inside SHUNT band)
    P[GATE_DZ[i]] = (fx + 7.5, 77.5, 90)  # zener clamp
    P[SNUB_R[i]] = (fx - 8, 69.5, 270)    # snubber R: pad1 north in SW
    P[SNUB_C[i]] = (fx - 8, 76.5, 270)    # snubber C: pad2 south in SHUNT

# --- boost power stage -----------------------------------------------------
P["L2"] = (58, 88, 0)
P["Q91"] = (46, 102, 270)
P["D2"] = (59, 74, 0)        # SS510 tab -> VBOOST
P["R7"] = (54, 104, 90)      # 0.05R CS
P["R6"] = (53, 116, 0)       # 10R boost gate (beside Q91)
P["C19"] = (148, 90, 0)      # +12V_SW reservoir
P["C1"] = (12, 88, 0)        # 24V input bulk

# --- connectors / big THT --------------------------------------------------
P["J1"] = (24, 142, 0)       # 24V barrel jack
P["F1"] = (34, 104, 270)     # input fuse holder (pads south)
P["J2"] = (54, 144, 180)     # aux-bank barrier (body up-left)
P["F2"] = (46, 122, 90)     # aux fuse (VBANK clips north, straggler)
P["K1"] = (68, 140, 0)      # charge relay
P["K2"] = (94, 140, 0)      # dump relay
P["R19"] = (166, 124, 0)    # 47R-5W precharge box
P["R21"] = (120, 140, 0)    # dump cement 1
P["R22"] = (136, 140, 0)    # dump cement 2
P["J6"] = (172, 145, 270)   # RAIL-EMIT IDC (rows run +x)
P["J7"] = (194, 145, 270)   # RAIL-RECV IDC (rows run +x)
P["J8"] = (207, 132, 90)    # spare JST
P["J9"] = (166, 60, 0)       # Pico socket L (pins run +y)
P["J10"] = (183.78, 60, 0)   # Pico socket R
P["J11"] = (206, 66, 0)     # 2x13 IO breakout (far east, wires exit right)
P["J3"] = (10, 120, 0)       # E-STOP header (left edge, bench switch)
P["J4"] = (10, 130, 0)       # ARM key header
P["TP1"] = (144, 80, 0)

# --- shelf-packed clusters (x0, y0, x1, y1) --------------------------------
AUTO_CLUSTERS = [
    # gate drivers + interlock (close to gate resistors)
    ((88, 86, 140, 96),
     ["U7", "C21", "U8", "C22", "U6", "C20", "C18", "R32", "R33"]),
    # band 1: VBOOST caps + sense + ADC + comparator ICs
    ((56, 98, 148, 111),
     ["C12", "C13", "U9", "C26", "R43", "C27", "U10", "C28",
      "U11", "C36", "U12", "C37"]),
    # band 2: buck + FB/OVP + comparator channel passives
    ((56, 113, 160, 132),
     ["U4", "R4", "C7", "C8", "C9", "R5", "C10", "R8", "C11",
      "U1", "C2", "L1", "C3", "C4", "U2", "C5", "C6", "R2", "R3",
      "R10", "R11", "R12", "R13", "C14", "R14", "C15",
      "R15", "R16", "U5", "Q1", "R17", "R18",
      "R44", "R45", "C29", "R9",
      "R46", "R47", "C30", "R48", "R49",
      "R50", "R51", "C31", "R52", "R53",
      "R54", "R55", "C32", "R56", "R57",
      "R58", "R59", "C33", "R60", "R61",
      "R62", "R63", "C34", "R64", "R65",
      "R66", "R67", "C35", "R68", "R69"]),
    # relay drivers (right end of band 1)
    ((150, 98, 162, 111),
     ["Q2", "R20", "D3", "Q3", "R23", "D4"]),
    # input protection (C1 placed manually at (12,88))
    ((8, 96, 22, 118),
     ["Q90", "R1", "D1", "D90"]),
    # V_bank + 24V sense buffers
    ((130, 46, 160, 58),
     ["U3", "C17", "R26", "R27", "C16"]),
    # status LEDs / buzzer / live-bank LED
    ((174, 36, 212, 57),
     ["BZ1", "Q4", "R72", "D16", "D14", "R70", "D15", "R71",
      "R25", "D5"]),
    # ARM / E-STOP sense dividers (beside J3/J4)
    ((16, 122, 34, 132),
     ["R28", "R29", "R30", "R31"]),
]

BRIDGE_TABLE = {}

KELVIN_TRACKS = [
    ("RS1", "1", "NT1", "1", 0.3),
    ("RS1", "2", "NT2", "1", 0.3),
]

# Manual fat tracks: (ref, pad, to_xy, width, layer)
MANUAL_TRACKS = [
    ("D13", "2", (14, 59), 3.0, 0),    # clamp anodes -> GND via cluster
    ("R24", "2", (18, 68.5), 0.8, 0),  # bleed GND end -> its own cluster
]

# Extra via clusters (net, cx, cy, nx, ny, pitch, drill, size)
VIA_CLUSTERS = [
    ("GND", 14, 60, 3, 2, 1.8, 0.6, 1.2),   # clamp surge return to planes
    ("GND", 18, 68.5, 2, 1, 1.8, 0.4, 0.8),  # bleed return
]

NET_WIDTHS = {
    "+24V": 1.5, "24V_F": 1.5, "24V_JACK": 1.5,
    "+12V": 0.8, "+12V_SW": 1.0, "+5V": 0.8, "3V3": 0.6,
    "VBOOST": 2.0, "BST_SW": 2.0, "BST_CS_HI": 2.0,
    "AUX_BANK": 2.5, "DUMP_R": 2.0, "DUMP_MID": 2.0,
    "VBANK": 2.5, "LIVE_LED": 0.4,
    "QG1": 0.5, "QG2": 0.5, "QG3": 0.5,
    "DRV1": 0.5, "DRV2": 0.5, "DRV3": 0.5,
    "ISNS_P": 0.3, "ISNS_N": 0.3,
}

PULSE_NETS = {"VBANK", "COIL_HI", "SW_DRAIN", "SHUNT_HI"}

# ---------------------------------------------------------------------------
# Autorouting strategy (freerouting/DeepPCB pivot) — net taxonomy.
#
# PLANE_NETS: exist only as copper pours/planes. Removed from the DSN routing
#   set entirely (their pins are already tied to the plane by preroute copper),
#   so the autorouter has no airwires to maze-route (this is what blew
#   freerouting up to ~700GB). Their (plane ...) polygons + stitching stay.
# CRITICAL_NETS: authored deterministically in preroute, then locked
#   (type route -> type fix) in the DSN so the autorouter treats them as
#   immovable and routes only around them.
# Everything else is noncritical and free for the autorouter.
# ---------------------------------------------------------------------------
PLANE_NETS = PULSE_NETS | {"GND"}

CRITICAL_NETS = {
    # gate drive (tight loop, source-referenced)
    "QG1", "QG2", "QG3", "DRV1", "DRV2", "DRV3",
    # Kelvin / shunt sense
    "ISNS_P", "ISNS_N",
    # boost sense / feedback / drive
    "BST_CS", "BST_CS_HI", "BST_FB", "BST_COMP", "BST_DRV", "BST_G", "BST_SW",
    # safety interlock
    "MCU_FIRE", "FIRE_GATED", "ARM_SENSE", "ESTOP_SENSE",
    "RLY_CHARGE", "RLY_DUMP",
    # fat rails (high current, keep short/wide)
    "AUX_BANK", "VBOOST", "DUMP_R", "DUMP_MID",
}


def netclass_rules():
    """Group NET_WIDTHS into DSN net classes (one width per class).

    Returns [(class_name, width_mm, clearance_mm, [nets_sorted])]. Nets not in
    NET_WIDTHS fall through to the DSN's default class (0.2/0.2). Wider rails
    get a touch more clearance. This is the single source that dsn_fixup.py
    turns into (class ...) blocks so the autorouter honours real widths.
    """
    from collections import defaultdict
    by_w = defaultdict(list)
    for net, w in NET_WIDTHS.items():
        by_w[w].append(net)
    rules = []
    for w in sorted(by_w, reverse=True):
        nets = sorted(by_w[w])
        name = "w" + str(w).replace(".", "p")
        # pulse/bank nets carry the 1.0mm bank-voltage clearance so the router
        # keeps signals clear of the high-current copper (else V24_SENSE etc
        # graze the VBANK pour); fat rails 0.3mm; small signals 0.2mm.
        if any(n in PLANE_NETS or n == "AUX_BANK" for n in nets):
            clr = 1.0
        elif w >= 1.0:
            clr = 0.3
        else:
            clr = 0.2
        rules.append((name, w, clr, nets))
    return rules
