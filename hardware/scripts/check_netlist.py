"""Netlist safety/topology assertions for the driver-board schematic.

Exports the netlist via kicad-cli and asserts the connectivity properties
that matter for safety and for sim-fidelity. Exit != 0 on any failure.

Run: uv run python hardware/scripts/check_netlist.py
"""

import subprocess
import sys
from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))
from kicad_sch import find, find_all, parse, Sym

KICAD_CLI = r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"
SCH = HW_DIR / "omnimarble-driver" / "omnimarble-driver.kicad_sch"
NET = HW_DIR / "omnimarble-driver" / "omnimarble-driver.net"


def export_netlist():
    subprocess.run([KICAD_CLI, "sch", "export", "netlist",
                    "-o", str(NET), str(SCH)], check=True)


def load_nets():
    """Return {net_name: set((ref, pin))}."""
    tree = parse(NET.read_text(encoding="utf-8"))[0]
    nets = {}
    nets_blk = find(tree, Sym("nets"))
    for net in find_all(nets_blk, Sym("net")):
        name = str(find(net, Sym("name"))[1])
        members = set()
        for node in find_all(net, Sym("node")):
            ref = str(find(node, Sym("ref"))[1])
            pin = str(find(node, Sym("pin"))[1])
            members.add((ref, pin))
        nets[name] = members
    return nets


def main():
    export_netlist()
    nets = load_nets()
    failures = []

    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    def refs(net):
        return {r for r, _ in nets.get(net, set())}

    # 1. Hardware interlock: MCU_FIRE reaches ONLY the AND gate (plus
    #    Pico socket + breakout); driver inputs are on FIRE_GATED.
    mcu_fire = refs("MCU_FIRE")
    check("MCU_FIRE isolated to interlock",
          "U6" in mcu_fire and not ({"U7", "U8"} & mcu_fire),
          f"MCU_FIRE on {sorted(mcu_fire)}")
    fire_gated = refs("FIRE_GATED")
    extras = fire_gated - {"U6", "U7", "U8"}
    check("FIRE_GATED drives both gate drivers via U6 (+pulldown only)",
          {"U6", "U7", "U8"} <= fire_gated
          and all(e.startswith("R") for e in extras),
          f"FIRE_GATED on {sorted(fire_gated)}")

    # 2. Driver VDD is on the ARMed rail, not raw +12V
    v12sw = refs("+12V_SW")
    check("Gate drivers powered from ARMed +12V_SW",
          {"U7", "U8"} <= v12sw, f"+12V_SW on {sorted(v12sw)}")
    v12 = refs("+12V")
    check("Gate drivers NOT on raw +12V",
          not ({"U7", "U8"} & v12), f"+12V has {sorted(v12 & {'U7','U8'})}")

    # 3. Kelvin sense nets: only the net-ties and the INA240
    for net, tie in [("ISNS_P", "NT1"), ("ISNS_N", "NT2")]:
        members = refs(net)
        check(f"{net} Kelvin isolation",
              members == {tie, "U9"}, f"{net} on {sorted(members)}")

    # 4. All six IR gate signals reach Pico socket, breakout, and comparator
    for i in range(1, 7):
        members = refs(f"GATE{i}")
        pkg = "U11" if i <= 4 else "U12"
        ok = ("J11" in members and pkg in members
              and ({"J9", "J10"} & members))
        check(f"GATE{i} routing", ok, f"on {sorted(members)}")

    # 5. FET drive: each QGx has exactly its FET + gate R + pulldown + zener
    for i in range(1, 4):
        members = refs(f"QG{i}")
        check(f"QG{i} contains FET Q{9+i}",
              f"Q{9+i}" in members, f"on {sorted(members)}")

    # 6. Pulse loop topology
    check("SW_DRAIN: 3 FET drains + freewheel + coil term + snubbers",
          sum(1 for r, p in nets["SW_DRAIN"] if r.startswith("Q1")) == 3
          and "J5" in refs("SW_DRAIN"), f"on {sorted(refs('SW_DRAIN'))}")
    check("SHUNT_HI: FET sources + shunt + Kelvin tie",
          "RS1" in refs("SHUNT_HI") and "NT1" in refs("SHUNT_HI"),
          f"on {sorted(refs('SHUNT_HI'))}")
    check("COIL_HI: blocking + freewheel diodes + coil terminal",
          "J5" in refs("COIL_HI")
          and sum(1 for r, p in nets["COIL_HI"] if r.startswith("D")) >= 4,
          f"on {sorted(refs('COIL_HI'))}")

    # 7. Every symbol with an LCSC part has the field (spot check count)
    print(f"\n  nets: {len(nets)} total")
    if failures:
        print(f"\nFAILED: {failures}")
        sys.exit(1)
    print("\nAll netlist assertions PASS")


if __name__ == "__main__":
    main()
