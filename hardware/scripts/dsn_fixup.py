"""Post-process a KiCad-exported Specctra DSN so an autorouter (freerouting
or DeepPCB) routes ONLY the noncritical signal nets, honours real per-net
widths, and never touches our critical or plane copper.

Two transforms applied by default (see hardware plan / placement.py taxonomy):

  1. NET CLASSES  — replace KiCad's single `(class kicad_default ...)` with one
     class per track width from placement.netclass_rules(), so the router uses
     VBANK 2.5mm, DRV 0.5mm, etc. instead of a flat 0.2mm. Pulse/bank nets also
     get the 1.0mm bank-voltage clearance.
  2. LOCK         — every wire/via on a CRITICAL_NET or PLANE_NET is rewritten
     `(type route)` -> `(type fix)`, so the router treats our authored/pour
     copper as immovable.

Plane nets are deliberately KEPT in the `(network)` (see fixup() default
drop_planes_from_network=False): freerouting connects GND/plane pads to the
`(plane ...)` polygons via short via-drops far more completely than our own
gnd_via_drops could, and the original ~700GB blow-up was max_passes=9999
(non-termination), NOT plane maze-routing. `exclude_planes_from_network()` is
retained as an opt-in escape hatch only.

Pure text / balanced-paren rewrite — no pcbnew dependency, unit-testable.
`new_board()` in gen_pcb.py builds on a fresh board that bypasses .kicad_pro,
so this DSN rewrite (not the pcbnew netclass API) is the authoritative vehicle.

Usage:  python hardware/scripts/dsn_fixup.py [in.dsn] [out.dsn]
Default in/out = hardware/omnimarble-driver/omnimarble-driver.dsn (in place).
"""

import re
import sys
from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR / "scripts"))
from placement import CRITICAL_NETS, PLANE_NETS, netclass_rules  # noqa: E402

DSN_DEFAULT = HW_DIR / "omnimarble-driver" / "omnimarble-driver.dsn"
UM_PER_MM = 1000  # DSN resolution is `um` (see header: (resolution um 10))


# --------------------------------------------------------------------------
# balanced-paren span finder that respects the DSN string quote (")
# --------------------------------------------------------------------------
def find_span(text, start):
    """Return (open_idx, close_idx_exclusive) of the paren group whose '('
    is at or after `start`. Quoted regions (") are skipped when counting."""
    i = text.index("(", start)
    depth = 0
    j = i
    in_q = False
    while j < len(text):
        c = text[j]
        if c == '"':
            in_q = not in_q
        elif not in_q:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i, j + 1
        j += 1
    raise ValueError("unbalanced parens from index %d" % start)


def net_of_line(line):
    """Extract NAME from a wire/via line's `(net NAME)` (quoted or bare)."""
    m = re.search(r'\(net\s+("([^"]+)"|(\S+?))\)', line)
    if not m:
        return None
    return m.group(2) if m.group(2) is not None else m.group(3)


# --------------------------------------------------------------------------
# transform 1 — net classes
# --------------------------------------------------------------------------
def rewrite_classes(text):
    """Replace the single kicad_default class with per-width classes + a
    default class, excluding PLANE_NETS from every list."""
    ci = text.index("(class ")
    open_i, close_i = find_span(text, ci)
    block = text[open_i:close_i]

    # nets listed in the original class (tokens between the class name and the
    # first sub-list `(circuit`). Handles bare + quoted names across lines.
    head = block[: block.index("(circuit")]
    # drop `(class` and the class name (`kicad_default`)
    toks = re.findall(r'"[^"]+"|[^\s()]+', head)
    # toks[0] == 'class'? no — we sliced from '(' so first tok is 'class'
    assert toks[0] == "class", toks[:3]
    all_nets = [t.strip('"') for t in toks[2:]]  # skip 'class', 'kicad_default'

    via_rule = '(circuit\n        (use_via "Via[0-3]_600:300_um")\n      )'
    rules = netclass_rules()
    classed = set()
    blocks = []
    for name, w_mm, clr_mm, nets in rules:
        present = [n for n in nets if n in all_nets]
        if not present:
            continue
        classed.update(present)
        w = int(round(w_mm * UM_PER_MM))
        c = int(round(clr_mm * UM_PER_MM))
        netlist = " ".join(present)
        blocks.append(
            "    (class %s %s\n      %s\n      (rule\n"
            "        (width %d)\n        (clearance %d)\n      )\n    )"
            % (name, netlist, via_rule, w, c)
        )

    # default class: everything not already width-classed (plane nets that
    # only need via-drops land here at 0.2/0.2 — fine, they're locked anyway)
    default_nets = [n for n in all_nets if n not in classed]
    netlist = " ".join(default_nets) if default_nets else ""
    blocks.append(
        "    (class kicad_default %s\n      %s\n      (rule\n"
        "        (width 200)\n        (clearance 200)\n      )\n    )"
        % (netlist, via_rule)
    )

    new_classes = "\n".join(blocks)
    return text[:open_i] + new_classes + text[close_i:]


# --------------------------------------------------------------------------
# transform 2 — lock critical + plane wires/vias (type route -> type fix)
# --------------------------------------------------------------------------
def lock_wires(text):
    lock = CRITICAL_NETS | PLANE_NETS
    out = []
    n_lock = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if (stripped.startswith("(wire ") or stripped.startswith("(via ")) \
                and "(type route)" in line:
            net = net_of_line(line)
            if net in lock:
                line = line.replace("(type route)", "(type fix)")
                n_lock += 1
        out.append(line)
    return "".join(out), n_lock


# --------------------------------------------------------------------------
# transform 3 — remove PLANE_NETS from (network) so no airwires to route
# --------------------------------------------------------------------------
def exclude_planes_from_network(text):
    ni = text.index("(network")
    net_open, net_close = find_span(text, ni)
    network = text[net_open:net_close]

    removed = 0
    pos = 0
    out_parts = []
    while True:
        m = re.search(r"\(net\s+(\S+)", network[pos:])
        if not m:
            out_parts.append(network[pos:])
            break
        s = pos + m.start()
        name = m.group(1).strip('"')
        o, c = find_span(network, s)
        out_parts.append(network[pos:o])
        if name in PLANE_NETS:
            removed += 1  # drop this (net ...) block
        else:
            out_parts.append(network[o:c])
        pos = c
    new_network = "".join(out_parts)
    return text[:net_open] + new_network + text[net_close:], removed


# --------------------------------------------------------------------------
def sanitize_net_names(text):
    """Strip '(' ')' from quoted net names. KiCad emits single-pad no-connects
    as `"unconnected-(J10-Pin_10-Pad10)"`; freerouting tolerates the parens but
    DeepPCB's Specctra parser rejects them ('no viable alternative at input').
    The rename is a pure text op so it stays consistent across class + network.
    """
    # quote-agnostic: the class-rewrite emits these names unquoted, and unquoted
    # parens are exactly what DeepPCB rejects. Strip the parens wherever the
    # `unconnected-(...)` token appears (class list, network, quoted or not).
    return re.sub(r'unconnected-\(([^)]*)\)', r'unconnected-\1', text)


def add_fet_strip_keepout(text):
    """Reserve the FET-source B.Cu corridor (KiCad x74-131mm, y69.5-74mm) for
    the SHUNT_HI bus. DeepPCB drops the locked bus wire and packs signals into
    that strip (islanding the FET sources); a keepout is a harder constraint the
    router respects. DSN units are um with y = -(mm)*1000. import-clean
    re-authors the bus into the cleared strip, so this keepout is DSN-only and
    never reaches the KiCad board (won't fight the bus in DRC)."""
    if "fet_shunt_bus" in text:
        return text
    ko = ('    (keepout "fet_shunt_bus" (polygon B.Cu 0  74000 -69500  '
          '131000 -69500  131000 -74000  74000 -74000  74000 -69500))\n')
    lines = text.splitlines(keepends=True)
    last = max(i for i, l in enumerate(lines) if "(keepout" in l)
    lines.insert(last + 1, ko)
    return "".join(lines)


def fixup(text, drop_planes_from_network=False):
    text = sanitize_net_names(text)
    """Prepare the DSN for the autorouter.

    We LOCK critical + plane copper (type fix) and inject real net classes.
    We deliberately KEEP plane nets in the (network): freerouting connects
    their pads to the (plane ...) polygons via short via-drops far more
    completely than our gnd_via_drops can. The original blow-up was
    max_passes=9999 (non-termination), not plane maze-routing, so there is no
    need to strip planes. `drop_planes_from_network` is retained as an escape
    hatch but defaults off.
    """
    text = rewrite_classes(text)
    text = add_fet_strip_keepout(text)
    text, n_lock = lock_wires(text)
    n_net = 0
    if drop_planes_from_network:
        text, n_net = exclude_planes_from_network(text)

    # assertion: no critical/plane net keeps a routable wire (all locked)
    for line in text.splitlines():
        st = line.lstrip()
        if (st.startswith("(wire ") or st.startswith("(via ")) \
                and "(type route)" in line:
            net = net_of_line(line)
            assert net not in (CRITICAL_NETS | PLANE_NETS), \
                "critical/plane net %s still (type route)" % net
    return text, n_lock, n_net


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DSN_DEFAULT
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    text = src.read_text(encoding="utf-8")
    out, n_lock, n_net = fixup(text)
    dst.write_text(out, encoding="utf-8")
    n_classes = out.count("(class ")
    print("dsn_fixup: %d classes, locked %d wires/vias, removed %d plane nets "
          "from network" % (n_classes, n_lock, n_net))
    print("wrote", dst)


if __name__ == "__main__":
    main()
