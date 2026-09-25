"""Impulse map: coil dv against fire position and approach speed, per gate.

The single-kick model (scripts/simulate_rig_shot.py with the PINN field and a
measured scope current) is validated against the bench's launch-position
table; the closed-loop twin needs it as a lookup rather than a GPU run per
kick. This script builds that lookup once per (cans, gate) and stores it as
JSON with its provenance (capture path + sha256, bank voltage, peak current)
so a map can never be mistaken for a different pulse than the one it was
built from.

    uv run python scripts/impulse_map.py --cans 4 --gate 1500 \
        --current-csv data/captures/scope_4can_49v_1500.csv --voltage 49.8 \
        --commit-copy config/impulse_maps/impulse_map_4can_1500us.json

Grid: x = -30..+30 mm step 1 (absolute fire position of the ball's centre;
the coil entry face is -22.78, the measured optimum -13.78), v_in = 0.15,
0.25, 0.5, 0.8 m/s. `--max-time` defaults to 4 s because at 0.15 m/s the
default 2 s run truncates before station B.

The map stores the TRUE dv (v_out_true - v_in_true): the twin reads its own
stations, so it must not inherit the map's measurement bias twice.

Voltage rescale: force goes as I^2 and I as V, so dv(V) = dv(V_map) *
(V/V_map)^2. That is exact for a linear circuit and a linear field; the
saturation cap and the diode tail bend it. The 3-can 30/49 V pair is the
one direct test (`--check-rescale A B`).
"""

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source" / "extensions" / "omni.marble.coaster"
                       / "omni" / "marble" / "coaster"))

from rig_profile import load_profile  # noqa: E402
from simulate_rig_shot import (  # noqa: E402
    DEFAULT_DT_S,
    load_measured_current,
    simulate_shot,
)

SCHEMA = "impulse_map_v1"
DEFAULT_X_RANGE = (-30.0, 30.0, 1.0)
DEFAULT_V_IN = (0.15, 0.25, 0.5, 0.8)
VOLTAGE_EXPONENT = 2.0

# Layer A scoring of the injected predictions (docs/PREDICTION_45CAN.md):
# hits at 400/700 us, the long gates deliver MORE than the model says.
KNOWN_BIAS = {
    "1500": {"measured_over_model": [1.06, 1.27],
             "note": "Layer A long-gate excess: 4 cans 1.06-1.11, 5 cans 1.27 "
                     "at 1500 us; candidate magnetisation lag (TWIN_AUDIT S-7). "
                     "Nothing fitted; --kick-scale exposes it."},
    "1000": {"measured_over_model": [1.03, 1.12],
             "note": "4 cans 1.03-1.07, 5 cans 1.12 at 1000 us."},
    "700": {"measured_over_model": [0.99, 1.01], "note": "hit"},
    "400": {"measured_over_model": [0.96, 1.03], "note": "hit"},
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_map(profile, solver, cans, gate_us, current_fn, current_peak,
              x_grid_mm, v_in_list, dt=DEFAULT_DT_S, max_time_s=4.0,
              voltage=None, progress=None, **shot_kwargs):
    """Run simulate_shot over every (v_in, x) and return the map dict.

    `solver` is anything with field_with_grad(r, z, I) -- the PINN in
    production, a stub in tests. `current_fn`/`current_peak` from
    load_measured_current, or None to use the RLC model. fire_offset_mm in
    simulate_shot is an ABSOLUTE x, which is what the grid is.
    """
    if voltage is None:
        voltage = float(profile.circuit.get("charge_voltage_V", 50.0))
    x_grid = [float(x) for x in x_grid_mm]
    v_list = [float(v) for v in v_in_list]
    grid = []
    i_peak = 0.0
    incomplete = []
    for v_in in v_list:
        row = []
        for x in x_grid:
            shot = simulate_shot(profile, solver, cans, voltage, v_in,
                                 on_time_us=gate_us, fire_offset_mm=x, dt=dt,
                                 max_time_s=max_time_s, current_fn=current_fn,
                                 current_peak=current_peak, **shot_kwargs)
            if shot["aborted"]:
                raise RuntimeError(
                    f"shot aborted at x={x} v_in={v_in}: {shot['abort_reason']}")
            if shot["n_ch_out"] < 5:
                # A braking kick (fire point past the coil centre, slow ball)
                # can stop or reverse the ball before station B. dv_true is
                # the coil-only signed gain and is settled once the pulse is
                # over, so it is still the right map value; record the
                # point as incomplete rather than abandoning the grid.
                incomplete.append([float(v_in), float(x), int(shot["n_ch_out"])])
            row.append(shot["dv_true_mps"])
            i_peak = max(i_peak, shot["i_peak_model_A"])
            if progress:
                progress(v_in, x, shot)
        grid.append(row)
    return {
        "schema": SCHEMA,
        "incomplete_points": incomplete,
        "cans": int(cans),
        "gate_us": float(gate_us),
        "capture": None,
        "v_bank_V": float(voltage),
        "i_peak_A": float(current_peak) if current_peak else float(i_peak),
        "profile": profile.name,
        "tag": "",
        "x_mm": x_grid,
        "v_in_mps": v_list,
        "dv_mps": grid,
        "voltage_scaling": {"exponent": VOLTAGE_EXPONENT, "basis_V": float(voltage)},
        "known_bias": KNOWN_BIAS.get(str(int(gate_us)), {}),
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "dt_s": dt,
        "max_time_s": max_time_s,
    }


def save_map(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path


def load_map(path):
    return ImpulseMap(json.loads(Path(path).read_text(encoding="utf-8")))


class ImpulseMap:
    """Bilinear lookup over the (v_in, x) grid, clamped outside with a flag."""

    def __init__(self, data):
        if data.get("schema") != SCHEMA:
            raise ValueError(f"not an impulse map: schema {data.get('schema')!r}")
        self.data = data
        self.cans = int(data["cans"])
        self.gate_us = float(data["gate_us"])
        self.x = [float(x) for x in data["x_mm"]]
        self.v = [float(v) for v in data["v_in_mps"]]
        self.dv = [[float(d) for d in row] for row in data["dv_mps"]]
        self.v_bank_V = float(data["v_bank_V"])
        scaling = data.get("voltage_scaling", {})
        self.exponent = float(scaling.get("exponent", VOLTAGE_EXPONENT))
        self.basis_V = float(scaling.get("basis_V", self.v_bank_V))
        if len(self.dv) != len(self.v) or any(len(r) != len(self.x) for r in self.dv):
            raise ValueError("impulse map grid shape does not match its axes")
        if self.x != sorted(self.x) or self.v != sorted(self.v):
            raise ValueError("impulse map axes must be ascending")
        self.tag = data.get("tag", "")

    @staticmethod
    def _locate(axis, value):
        """(i0, i1, frac, clamped) bracketing value on an ascending axis."""
        n = len(axis)
        if n == 1:
            return 0, 0, 0.0, not (abs(value - axis[0]) < 1e-12)
        if value <= axis[0]:
            return 0, 0, 0.0, value < axis[0] - 1e-12
        if value >= axis[-1]:
            return n - 1, n - 1, 0.0, value > axis[-1] + 1e-12
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if axis[mid] <= value:
                lo = mid
            else:
                hi = mid
        return lo, hi, (value - axis[lo]) / (axis[hi] - axis[lo]), False

    def lookup(self, x_mm, v_in_mps):
        """(dv at the map's own voltage, clamped flag)."""
        i0, i1, fx, cx = self._locate(self.x, x_mm)
        j0, j1, fv, cv = self._locate(self.v, v_in_mps)
        d00 = self.dv[j0][i0]
        d01 = self.dv[j0][i1]
        d10 = self.dv[j1][i0]
        d11 = self.dv[j1][i1]
        top = d00 + fx * (d01 - d00)
        bot = d10 + fx * (d11 - d10)
        return top + fv * (bot - top), (cx or cv)

    def dv_at(self, x_mm, v_in_mps):
        return self.lookup(x_mm, v_in_mps)[0]

    def mirrored(self, x_mm, v_in_mps):
        """The return leg: the coil is symmetric, the geometry is x -> -x."""
        return self.dv_at(-x_mm, v_in_mps)

    def rescale_factor(self, v_bank):
        return (float(v_bank) / self.basis_V) ** self.exponent

    def v_in_spread(self, x_mm):
        """(max - min)/max of dv across the v_in axis at one x."""
        vals = [self.dv_at(x_mm, v) for v in self.v]
        top = max(abs(v) for v in vals)
        if top == 0.0:
            return 0.0
        return (max(vals) - min(vals)) / top

    def peak(self, v_in_mps=None):
        """(x, dv) of the largest dv along x, at one v_in or the grid mean."""
        if v_in_mps is None:
            v_in_mps = sum(self.v) / len(self.v)
        best = None
        for x in self.x:
            d = self.dv_at(x, v_in_mps)
            if best is None or d > best[1]:
                best = (x, d)
        return best


def check_rescale(map_a, map_b):
    """How well map_a rescaled to map_b's voltage reproduces map_b.

    Compares on the grid points the two maps share. Returns a dict with the
    ratio statistics (measured b / predicted-from-a): mean, rms deviation
    from 1, worst point, n. A ratio far from 1 is the voltage-dependent
    shape of the field/current the V^2 rule cannot express.
    """
    factor = map_a.rescale_factor(map_b.v_bank_V)
    ratios = []
    worst = None
    for j, v in enumerate(map_b.v):
        if v not in map_a.v:
            continue
        for i, x in enumerate(map_b.x):
            if x not in map_a.x:
                continue
            pred = map_a.dv_at(x, v) * factor
            meas = map_b.dv[j][i]
            if abs(pred) < 1e-4:
                continue
            r = meas / pred
            ratios.append(r)
            if worst is None or abs(r - 1.0) > abs(worst[2] - 1.0):
                worst = (x, v, r)
    if not ratios:
        return {"n": 0, "factor": factor}
    mean = sum(ratios) / len(ratios)
    rms = (sum((r - 1.0) ** 2 for r in ratios) / len(ratios)) ** 0.5
    return {"n": len(ratios), "factor": factor, "mean_ratio": mean,
            "rms_dev": rms, "worst": worst,
            "v_a": map_a.v_bank_V, "v_b": map_b.v_bank_V}


def _parse_range(spec):
    lo, hi, step = (float(s) for s in spec.split(":"))
    n = int(round((hi - lo) / step)) + 1
    return [lo + i * step for i in range(n)]


def default_out_path(cans, gate_us):
    return ROOT / "results" / f"impulse_map_{int(cans)}_{int(gate_us)}.json"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rig-profile", default="vbench_v0")
    parser.add_argument("--cans", type=int, required=False, default=4)
    parser.add_argument("--gate", type=float, default=1500.0,
                        help="requested on-time in us (the map's gate label)")
    parser.add_argument("--current-csv", type=Path, default=None,
                        help="scope capture to inject (t_s,i_A,...)")
    parser.add_argument("--voltage", type=float, default=None,
                        help="bank voltage the capture was taken at")
    parser.add_argument("--x-range", default="%g:%g:%g" % DEFAULT_X_RANGE,
                        help="MIN:MAX:STEP absolute fire x in mm")
    parser.add_argument("--v-in", default=",".join(str(v) for v in DEFAULT_V_IN),
                        help="comma-separated approach speeds, m/s")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, default=None,
                        help="default results/impulse_map_<cans>_<gate>.json")
    parser.add_argument("--commit-copy", type=Path, default=None,
                        help="also write the map here (config/impulse_maps/...)")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--max-time", type=float, default=4.0)
    parser.add_argument("--check-rescale", nargs=2, type=Path, metavar=("A", "B"),
                        help="compare two saved maps under the V^2 rule and exit")
    args = parser.parse_args(argv)

    if args.check_rescale:
        a, b = (load_map(p) for p in args.check_rescale)
        r = check_rescale(a, b)
        print(json.dumps(r, indent=2))
        return 0

    profile = load_profile(ROOT, args.rig_profile)
    voltage = args.voltage if args.voltage is not None \
        else float(profile.circuit["charge_voltage_V"])
    current_fn = current_peak = None
    capture = None
    if args.current_csv is not None:
        current_fn, current_peak = load_measured_current(args.current_csv)
        capture = {"path": str(args.current_csv).replace("\\", "/"),
                   "sha256": sha256_of(args.current_csv)}
        print(f"injected {args.current_csv} (I_pk {current_peak:.1f} A)")

    params = json.loads((ROOT / "config" / "coil_params.json").read_text())
    params["num_turns"] = profile.coil["num_turns"]
    params["length_mm"] = profile.coil["length_mm"]
    from warp_bfield_solver import WarpBFieldSolver
    solver = WarpBFieldSolver(params, chi_eff=3.0)

    x_grid = _parse_range(args.x_range)
    v_list = [float(v) for v in args.v_in.split(",")]
    print(f"building {args.cans}-can {args.gate:.0f} us map at {voltage:.1f} V: "
          f"{len(x_grid)} x {len(v_list)} shots")

    def progress(v_in, x, shot):
        print(f"  v_in {v_in:.2f} x {x:+6.1f}  dv {shot['dv_true_mps']*1000:7.1f} mm/s",
              flush=True)

    data = build_map(profile, solver, args.cans, args.gate, current_fn,
                     current_peak, x_grid, v_list, dt=args.dt,
                     max_time_s=args.max_time, voltage=voltage,
                     progress=progress)
    data["capture"] = capture
    data["tag"] = args.tag
    out = args.out or default_out_path(args.cans, args.gate)
    save_map(data, out)
    print(f"wrote {out}")
    if args.commit_copy:
        save_map(data, args.commit_copy)
        print(f"wrote {args.commit_copy}")
    imap = ImpulseMap(data)
    for v in imap.v:
        x_pk, dv_pk = imap.peak(v)
        print(f"  v_in {v:.2f}: peak dv {dv_pk*1000:.1f} mm/s at x = {x_pk:+.1f}; "
              f"dv(-13.78) = {imap.dv_at(-13.78, v)*1000:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
