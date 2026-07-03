"""Minimal KiCad 10 schematic generation toolkit.

Approach: generated schematics use the global-label-per-pin style — every
symbol pin gets a short wire stub ending in a global label naming its net.
That makes the schematic electrically complete and ERC/netlist-correct
without solving general wire routing; readability comes from grouping and
placement order, not drawn nets.

Only what gen_schematic.py needs is implemented:
  - s-expression tokenizer/parser/writer
  - symbol extraction from KiCad's bundled .kicad_sym libraries
    (including `extends` resolution for derived symbols)
  - pin-position extraction so label stubs land exactly on pins
  - schematic document assembly (lib_symbols, symbol instances,
    wires, global labels, no-connects, text)

Coordinates: KiCad schematic Y grows DOWNWARD; pin offsets from the
library are in symbol space (Y up) and are negated here on placement.
"""

import math
import re
import uuid as _uuid
from pathlib import Path

KICAD_SHARE = Path(r"C:\Program Files\KiCad\10.0\share\kicad")
SYMBOL_DIR = KICAD_SHARE / "symbols"

SCH_VERSION = "20251024"
GENERATOR = "omnimarble_gen"

# ---------------------------------------------------------------------------
# S-expressions
# ---------------------------------------------------------------------------


class Sym(str):
    """Bare (unquoted) atom."""


def tokenize(text):
    # strings, parens, atoms
    token_re = re.compile(r'"(?:[^"\\]|\\.)*"|\(|\)|[^\s()"]+')
    return token_re.findall(text)


def parse(text):
    tokens = tokenize(text)
    pos = 0

    def read():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            lst = []
            while tokens[pos] != ")":
                lst.append(read())
            pos += 1
            return lst
        if tok.startswith('"'):
            return tok[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        return Sym(tok)

    out = []
    while pos < len(tokens):
        out.append(read())
    return out


def dump(node, indent=0):
    """Serialize parsed s-expr back to text (stable, KiCad-readable)."""
    pad = "\t" * indent
    if isinstance(node, list):
        if not node:
            return pad + "()"
        head = node[0]
        # Short lists with no sublists go on one line
        if all(not isinstance(x, list) for x in node):
            return pad + "(" + " ".join(_atom(x) for x in node) + ")"
        parts = [pad + "(" + _atom(head)]
        i = 1
        # leading atoms stay on the head line
        while i < len(node) and not isinstance(node[i], list):
            parts[0] += " " + _atom(node[i])
            i += 1
        for child in node[i:]:
            parts.append(dump(child, indent + 1))
        parts.append(pad + ")")
        return "\n".join(parts)
    return pad + _atom(node)


def _atom(x):
    if isinstance(x, Sym):
        return str(x)
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return _num(x)
    s = str(x)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _num(v):
    if isinstance(v, float):
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s not in ("-0", "") else "0"
    return str(v)


def find(node, tag):
    """First child list whose head is `tag`."""
    for child in node:
        if isinstance(child, list) and child and child[0] == tag:
            return child
    return None


def find_all(node, tag):
    return [c for c in node if isinstance(c, list) and c and c[0] == tag]


def uid():
    return str(_uuid.uuid4())


# ---------------------------------------------------------------------------
# Symbol library access
# ---------------------------------------------------------------------------

_lib_cache = {}


def _load_lib(lib_name):
    if lib_name not in _lib_cache:
        path = SYMBOL_DIR / f"{lib_name}.kicad_sym"
        tree = parse(path.read_text(encoding="utf-8"))[0]
        symbols = {}
        for child in find_all(tree, Sym("symbol")):
            symbols[child[1]] = child
        _lib_cache[lib_name] = symbols
    return _lib_cache[lib_name]


def get_symbol(lib_id):
    """Return (symbol_sexpr, parents) for `Lib:Name`, renamed to lib_id form.

    KiCad embeds library symbols in the schematic's lib_symbols block with
    the full `Lib:Name` id. Derived symbols (`extends`) need their parent
    embedded too (also renamed to `Lib:Parent`).
    """
    lib_name, sym_name = lib_id.split(":", 1)
    lib = _load_lib(lib_name)
    if sym_name not in lib:
        raise KeyError(f"symbol {sym_name!r} not in {lib_name}.kicad_sym")

    import copy
    sym = copy.deepcopy(lib[sym_name])
    sym[1] = lib_id
    parents = []
    ext = find(sym, Sym("extends"))
    if ext:
        parent_name = ext[1]
        parent = copy.deepcopy(lib[parent_name])
        parent[1] = f"{lib_name}:{parent_name}"
        ext[1] = f"{lib_name}:{parent_name}"
        parents.append(parent)
    return sym, parents


def symbol_pins(lib_id, unit=1):
    """Return {pin_number: (x, y, angle, name, etype)} in symbol coords.

    Pins come from sub-symbols named NAME_<unit>_<style>; unit 0 pins are
    common to all units (e.g. power pins of multi-unit parts share unit 0
    in some libs — KiCad convention is unit 0 = all units).
    """
    lib_name, sym_name = lib_id.split(":", 1)
    lib = _load_lib(lib_name)
    sym = lib[sym_name]
    ext = find(sym, Sym("extends"))
    if ext:
        sym = lib[ext[1]]
        sym_name = ext[1]

    pins = {}
    for sub in find_all(sym, Sym("symbol")):
        m = re.match(re.escape(sym_name) + r"_(\d+)_(\d+)$", sub[1])
        if not m:
            continue
        sub_unit = int(m.group(1))
        if sub_unit not in (0, unit):
            continue
        for pin in find_all(sub, Sym("pin")):
            at = find(pin, Sym("at"))
            name = find(pin, Sym("name"))
            number = find(pin, Sym("number"))
            etype = str(pin[1])
            pins[str(number[1])] = (
                float(at[1]), float(at[2]), float(at[3]),
                str(name[1]), etype,
            )
    return pins


# ---------------------------------------------------------------------------
# Schematic document
# ---------------------------------------------------------------------------

# Stub length from pin end to the global label
STUB = 2.54

GRID = 1.27  # KiCad connection grid (50 mil)


def snap(v):
    """Snap a coordinate to the 1.27mm connection grid."""
    return round(v / GRID) * GRID


class Schematic:
    def __init__(self, title, page="A3"):
        self.title = title
        self.page = page
        self.lib_symbols = {}
        self.items = []          # symbol/wire/label/... s-exprs
        self.sheet_uuid = uid()

    # -- symbols ------------------------------------------------------------

    def add_symbol(self, lib_id, ref, value, at, footprint="", lcsc="",
                   unit=1, nets=None, no_connect=(), dnp=False,
                   extra_fields=None):
        """Place a symbol and label its pins.

        nets: {pin_number: net_name} — each listed pin gets a wire stub +
              global label. Pins in no_connect get an NC marker. Pins in
              neither are left untouched (caller must cover them across
              units or ERC will flag them).
        at: (x, y) in mm, schematic coords (y down). Rotation fixed at 0.
        """
        sym, parents = get_symbol(lib_id)
        for p in parents:
            self.lib_symbols.setdefault(p[1], p)
        self.lib_symbols.setdefault(lib_id, sym)

        x0, y0 = snap(at[0]), snap(at[1])
        pins = symbol_pins(lib_id, unit=unit)

        inst = [Sym("symbol"),
                [Sym("lib_id"), lib_id],
                [Sym("at"), x0, y0, 0],
                [Sym("unit"), unit],
                [Sym("exclude_from_sim"), Sym("no")],
                [Sym("in_bom"), Sym("yes")],
                [Sym("on_board"), Sym("yes")],
                [Sym("dnp"), Sym("yes") if dnp else Sym("no")],
                [Sym("uuid"), uid()],
                _prop("Reference", ref, x0, y0 - 2.54),
                _prop("Value", value, x0, y0 + 2.54),
                _prop("Footprint", footprint, x0, y0, hide=True),
                _prop("Datasheet", "", x0, y0, hide=True),
                _prop("Description", "", x0, y0, hide=True),
                ]
        if lcsc:
            inst.append(_prop("LCSC", lcsc, x0, y0, hide=True))
        for k, v in (extra_fields or {}).items():
            inst.append(_prop(k, v, x0, y0, hide=True))
        for num in pins:
            inst.append([Sym("pin"), str(num), [Sym("uuid"), uid()]])
        inst.append([Sym("instances"),
                     [Sym("project"), "",
                      [Sym("path"), f"/{self.sheet_uuid}",
                       [Sym("reference"), ref],
                       [Sym("unit"), unit],
                       ]]])
        self.items.append(inst)

        # pin stubs + labels
        nets = nets or {}
        for num, net in nets.items():
            if num not in pins:
                raise KeyError(f"{lib_id} {ref}: pin {num!r} not found "
                               f"(unit {unit}; has {sorted(pins)})")
            px, py, pang, _, _ = pins[num]
            # pin position in schematic coords (y negated)
            sx, sy = x0 + px, y0 - py
            # stub direction: pins point INTO the body; the connection end
            # is at (sx, sy) and the label extends outward along pin angle
            dx, dy = _dir(pang)
            ex, ey = sx + dx * STUB, sy + dy * STUB
            self.wire((sx, sy), (ex, ey))
            self.global_label(net, (ex, ey), angle=_label_angle(pang))
        for num in no_connect:
            if num not in pins:
                raise KeyError(f"{lib_id} {ref}: NC pin {num!r} not found")
            px, py, _, _, _ = pins[num]
            self.items.append([Sym("no_connect"),
                               [Sym("at"), x0 + px, y0 - py],
                               [Sym("uuid"), uid()]])

    # -- primitives ----------------------------------------------------------

    def wire(self, a, b):
        self.items.append([Sym("wire"),
                           [Sym("pts"),
                            [Sym("xy"), a[0], a[1]],
                            [Sym("xy"), b[0], b[1]]],
                           [Sym("stroke"), [Sym("width"), 0],
                            [Sym("type"), Sym("default")]],
                           [Sym("uuid"), uid()]])

    def global_label(self, text, at, angle=0):
        self.items.append([Sym("global_label"), text,
                           [Sym("shape"), Sym("passive")],
                           [Sym("at"), at[0], at[1], angle],
                           [Sym("effects"),
                            [Sym("font"), [Sym("size"), 1.27, 1.27]],
                            [Sym("justify"),
                             Sym("right") if angle == 180 else Sym("left")]],
                           [Sym("uuid"), uid()]])

    def text(self, s, at, size=2.0):
        self.items.append([Sym("text"), s,
                           [Sym("exclude_from_sim"), Sym("no")],
                           [Sym("at"), at[0], at[1], 0],
                           [Sym("effects"),
                            [Sym("font"), [Sym("size"), size, size]]],
                           [Sym("uuid"), uid()]])

    # -- power helpers --------------------------------------------------------

    def power_flag(self, net, at):
        """PWR_FLAG marks a net as a legitimate power source for ERC."""
        self.add_symbol("power:PWR_FLAG", f"#FLG_{net}_{len(self.items)}",
                        "PWR_FLAG", at, nets={"1": net})

    # -- output ----------------------------------------------------------------

    def to_sexpr(self):
        doc = [Sym("kicad_sch"),
               [Sym("version"), Sym(SCH_VERSION)],
               [Sym("generator"), GENERATOR],
               [Sym("generator_version"), "10.0"],
               [Sym("uuid"), self.sheet_uuid],
               [Sym("paper"), self.page],
               [Sym("title_block"),
                [Sym("title"), self.title],
                [Sym("company"), "OmniMarble"],
                ],
               ]
        lib = [Sym("lib_symbols")]
        for k in sorted(self.lib_symbols):
            lib.append(self.lib_symbols[k])
        doc.append(lib)
        doc.extend(self.items)
        doc.append([Sym("embedded_fonts"), Sym("no")])
        return doc

    def write(self, path):
        Path(path).write_text(dump(self.to_sexpr()) + "\n", encoding="utf-8")


def _prop(name, value, x, y, hide=False):
    p = [Sym("property"), name, value,
         [Sym("at"), x, y, 0],
         [Sym("show_name"), Sym("no")],
         [Sym("do_not_autoplace"), Sym("no")],
         ]
    if hide:
        p.append([Sym("hide"), Sym("yes")])
    p.append([Sym("effects"), [Sym("font"), [Sym("size"), 1.27, 1.27]]])
    return p


def _dir(pin_angle):
    """Outward direction of a pin's connection point in schematic coords.

    A pin's `at` angle points from the connection end toward the body.
    The stub should extend the opposite way (outward). Schematic Y is down.
    """
    a = pin_angle % 360
    if a == 0:      # pin points right (into body on its right) -> stub left
        return (-1, 0)
    if a == 180:
        return (1, 0)
    if a == 90:     # symbol-space up -> schematic down is negated: stub down
        return (0, 1)
    if a == 270:
        return (0, -1)
    raise ValueError(f"unsupported pin angle {pin_angle}")


def _label_angle(pin_angle):
    a = pin_angle % 360
    return {0: 180, 180: 0, 90: 270, 270: 90}[a]
