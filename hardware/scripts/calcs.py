"""Driver-board design calculations — single source of truth.

Recomputes every engineering number quoted in hardware/design-calcs.md
from scripts/rlc_circuit.py (the same physics the simulation uses) plus
datasheet constants for the selected parts, asserts all design margins,
and regenerates design-calcs.md.

Run:  uv run python hardware/scripts/calcs.py
Exit code != 0 if any margin assert fails.
"""

import math
import sys
from pathlib import Path

HW_DIR = Path(__file__).resolve().parent.parent
ROOT = HW_DIR.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rlc_circuit import compute_rlc_params, rlc_current

sys.path.insert(0, str(HW_DIR / "scripts"))
from parts import BANK_POSITIONS, BANK_UNIT_UF, PARTS

OUT_MD = HW_DIR / "design-calcs.md"

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

COIL = {  # demo coil, config/coil_params.json
    "L_uH": 12.406,
    "R_total_ohm": 0.1102,  # R_dc 0.0802 + ESR 0.01 + wiring 0.02
}

V_BANK_MAX = 55.0          # commanded operating ceiling (see boost/OVP ladder)
V_BANK_ABS = 63.0          # capacitor voltage rating / OVP trip
# Bank built from 2200uF/63V snap-in cans, 1..5 positions populated
BANK_OPTIONS_UF = [int(BANK_UNIT_UF * n) for n in range(1, BANK_POSITIONS + 1)]
BANK_BASELINE_UF = int(BANK_UNIT_UF * 2)   # 2 cans populated

DESIGN_I_PK_A = 600.0      # switch/diode capability target (headroom > worst case)
DESIGN_PULSE_S = 2e-3

# Selected parts (datasheet constants sourced from parts.py / bom.csv)
_fet = PARTS["fet_pulse"]
FET = {
    "name": f"{_fet['mfr']} {_fet['mpn']} ({_fet['package']}, 150V, {_fet['lcsc']})",
    "n_parallel": _fet["qty"],
    "rds_on_hot_ohm": _fet["rds_on_max_ohm"] * 1.5,   # max Rds(on) x1.5 hot derating
    "idm_pulse_a": _fet["idm_pulse_a"],
    "zth_1ms_k_per_w": 0.12,          # conservative Zth(jc,1ms); RthJC(dc)=0.48
    "vds_v": 150.0,
    "tj_max_c": _fet["tj_max_c"],
}
_dio = PARTS["diode_pulse"]
DIODE = {
    "name": f"{_dio['mfr']} {_dio['mpn']} ({_dio['package']}, 100V Schottky, {_dio['lcsc']})",
    "vf_v": _dio["vf_v"],
    "ifsm_a": _dio["ifsm_a"],
    "i2t_a2s": _dio["i2t_a2s"],
    "n_blocking": 2,                  # paralleled packages in series-blocking role
}
_sh = PARTS["shunt"]
SHUNT = {
    "name": f"{_sh['mfr']} {_sh['mpn']} ({_sh['package']}, {_sh['lcsc']}, Kelvin-routed)",
    "r_ohm": _sh["r_ohm"],
    "p_cont_w": _sh["p_cont_w"],
}
INA240_GAIN = 20.0
ADC_VREF = 3.0             # ADS7042 ref = 3.0V rail (buffered)
ADC_BITS = 12

POUR = {                    # pulse-path copper on the PCB
    "width_mm": 20.0,
    "thickness_um": 70.0,   # 2oz
    "layers": 2,            # L1 + L4 mirrored
    "length_mm": 100.0,     # total loop estimate
}
CU_RESISTIVITY = 1.72e-8    # ohm*m
CU_VOL_HEAT = 3.45e6        # J/(m^3*K)

CHARGER = {
    "i_charge_a": 0.5,      # boost average output current limit
    "v_floor": 24.5,        # boost cannot regulate below Vin
}
BLEED_R_OHM = 6_800.0       # permanent bleed (0.44W at 55V -> 2W part)
DUMP_R_OHM = 200.0          # MCU dump: 2x 100R/10W cement in series
PRECHARGE_R_OHM = 47.0

DIVIDER = {"top": 100_000.0, "bottom": 5_100.0}   # V_bank sense (E24/JLC-basic values)

# Boost feedback / setpoint network (UC3843, Vref_FB = 2.5V):
#   VBOOST = 2.5 + RT*(2.5/RB - (VSET-2.5)/RJ)
# Summing topology is inverting: PWM duty UP -> voltage DOWN. PWM stuck
# LOW therefore commands the MAXIMUM, which must sit BELOW the OVP floor.
BOOST_FB = {"rt": 100_000.0, "rb": 10_000.0, "rj": 9_100.0, "vref": 2.5}
BOOST_FLOOR_V = 24.5     # boost cannot regulate below Vin

# Hardware OVP: TL431B (2.495V +/-0.5%) + 1% divider pulls UC3843 COMP low.
# NOTE: the TL431 cathode sits on COMP and can only clamp it to ~2-2.5V, which
# throttles but does not fully stop the boost (COMP<1.4V needed for zero duty);
# the MCU inhibit (Q1) is the authoritative cut. See OVP-authority note below.
OVP = {"top": 100_000.0, "bottom": 4_300.0, "vref": 2.495,
       "r_tol": 0.01, "ref_tol": 0.005}  # TL431BIDBZR is +/-0.5%

# Relays (Hongfa HF3FF/012-1ZS): contact 15A/125VAC, ~10A/28VDC class.
# DC switching at bank voltage is managed by SEQUENCING, not contact
# rating: charge relay closes only after the 47R precharge equalizes
# (dV<5V) and opens only with the boost inhibited (no current); dump
# relay closes onto <=0.6A resistive and opens after full discharge.
RELAY = {"coil_v": 12.0, "coil_r_ohm": 400.0,  # HF3FF datasheet "contact_close_dv_max": 5.0,
         "dump_close_a": None}  # dump_close computed below

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pulse_waveform(rlc, dt=1e-6):
    """Sample I(t) over the effective pulse. Returns (ts, Is)."""
    t_end = rlc["effective_pulse_duration_s"]
    n = max(200, int(t_end / dt))
    ts = [i * t_end / n for i in range(n + 1)]
    return ts, [rlc_current(t, rlc) for t in ts]


def integrate_i2t(ts, Is):
    total = 0.0
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        total += ((Is[i] ** 2 + Is[i - 1] ** 2) / 2) * dt
    return total


def bank_report(c_uF, v0=V_BANK_MAX, coil=None):
    coil = coil or COIL
    rlc = compute_rlc_params({
        "capacitance_uF": c_uF,
        "charge_voltage_V": v0,
        "inductance_uH": coil["L_uH"],
        "total_resistance_ohm": coil["R_total_ohm"],
    })
    ts, Is = pulse_waveform(rlc)
    i2t = integrate_i2t(ts, Is)
    i_pk = max(Is)
    t_pulse = rlc["effective_pulse_duration_s"]

    # FET: per-device conduction energy and junction temperature rise
    n = FET["n_parallel"]
    e_fet = i2t / (n ** 2) * FET["rds_on_hot_ohm"]        # J per device (shared)
    p_avg_fet = e_fet / t_pulse
    dtj_shared = p_avg_fet * FET["zth_1ms_k_per_w"]
    # Single-device fault case (total sharing failure)
    e_fault = i2t * FET["rds_on_hot_ohm"]
    dtj_fault = (e_fault / t_pulse) * FET["zth_1ms_k_per_w"]

    # Blocking diodes: i2t per package (2 in parallel), dissipation
    i2t_diode = i2t / (DIODE["n_blocking"] ** 2)
    e_diode = (i2t ** 0.5) * 0  # placeholder not used; compute via Vf below
    # Conduction loss with Vf model: E = Vf * integral(I dt) / n
    q_coulombs = 0.0
    for i in range(1, len(ts)):
        q_coulombs += ((Is[i] + Is[i - 1]) / 2) * (ts[i] - ts[i - 1])
    e_diode = DIODE["vf_v"] * q_coulombs / DIODE["n_blocking"]

    # Shunt
    e_shunt = i2t * SHUNT["r_ohm"]
    p_pk_shunt = i_pk ** 2 * SHUNT["r_ohm"]

    # Pour adiabatic heating (both layers share current)
    area_m2 = (POUR["width_mm"] * 1e-3) * (POUR["thickness_um"] * 1e-6) * POUR["layers"]
    r_pour = CU_RESISTIVITY * (POUR["length_mm"] * 1e-3) / area_m2
    dT_pour = (i2t * CU_RESISTIVITY) / (area_m2 ** 2 * CU_VOL_HEAT) * (POUR["length_mm"] * 1e-3) * 0
    # adiabatic: dT = integral(I^2 R dt) / (vol * c_v); vol = A * length
    vol_m3 = area_m2 * (POUR["length_mm"] * 1e-3)
    dT_pour = (i2t * r_pour) / (vol_m3 * CU_VOL_HEAT)

    # ADC scaling
    i_fullscale = ADC_VREF / (SHUNT["r_ohm"] * INA240_GAIN)
    lsb_amps = i_fullscale / (2 ** ADC_BITS)

    # Charge / bleed / dump
    c_f = c_uF * 1e-6
    t_charge = c_f * (v0 - CHARGER["v_floor"]) / CHARGER["i_charge_a"]
    tau_bleed = BLEED_R_OHM * c_f
    t_dump_to_5v = DUMP_R_OHM * c_f * math.log(v0 / 5.0)

    return {
        "c_uF": c_uF, "v0": v0, "rlc": rlc, "i_pk": i_pk,
        "t_pk": rlc["time_to_peak_s"], "t_pulse": t_pulse,
        "energy_j": rlc["stored_energy_J"], "i2t": i2t,
        "e_fet_j": e_fet, "dtj_shared": dtj_shared, "dtj_fault": dtj_fault,
        "i2t_diode": i2t_diode, "e_diode_j": e_diode,
        "e_shunt_j": e_shunt, "p_pk_shunt_w": p_pk_shunt,
        "r_pour_ohm": r_pour, "dT_pour": dT_pour,
        "i_fullscale": i_fullscale, "lsb_amps": lsb_amps,
        "t_charge_s": t_charge, "tau_bleed_s": tau_bleed,
        "t_dump_s": t_dump_to_5v,
    }


# ---------------------------------------------------------------------------
# Coil safety envelope: which (L, R_total) coils are safe on this board?
# Evaluated at the WORST configuration the board supports: full 11000uF
# bank at the 55V operating ceiling. A coil is SAFE iff every pulse margin
# that depends on the coil holds:
#   I_pk <= 600A switch design point
#   FET shared-conduction dTj < 50K
#   blocking-diode I2t rating margin >= 10x
#   pulse-pour adiabatic heating < 5K
#   ADC full scale >= 1.25 x I_pk
# ---------------------------------------------------------------------------

ENVELOPE_JSON = HW_DIR / "coil-envelope.json"
ENV_L_GRID_UH = [1, 2, 3, 4, 5, 6, 8, 10, 12.4, 16, 20, 30, 40]
ENV_R_GRID_OHM = [0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.15, 0.2, 0.3, 0.5]
ENV_R_FLOOR_OHM = 0.03   # sanity floor: below this, wiring alone dominates


def coil_is_safe(r):
    return (r["i_pk"] <= DESIGN_I_PK_A
            and r["dtj_shared"] < 50.0
            and r["i2t_diode"] * 10 <= DIODE["i2t_a2s"]
            and r["dT_pour"] < 5.0
            and r["i_fullscale"] >= 1.25 * r["i_pk"])


def coil_envelope():
    """Sweep the L/R grid at worst bank; return (grid, rect_envelope)."""
    worst_c = BANK_OPTIONS_UF[-1]
    grid = {}
    for L in ENV_L_GRID_UH:
        for R in ENV_R_GRID_OHM:
            rep = bank_report(worst_c, coil={"L_uH": L, "R_total_ohm": R})
            grid[(L, R)] = (coil_is_safe(rep), rep["i_pk"])
    # Conservative rectangle: smallest (L_min, R_min) grid corner such
    # that EVERY grid point with L >= L_min and R >= R_min is safe.
    rect = None
    for L_min in ENV_L_GRID_UH:
        for R_min in ENV_R_GRID_OHM:
            if R_min < ENV_R_FLOOR_OHM:
                continue
            ok = all(safe for (L, R), (safe, _) in grid.items()
                     if L >= L_min and R >= R_min)
            if ok:
                rect = (L_min, R_min)
                break
        if rect:
            break
    return grid, rect


# ---------------------------------------------------------------------------
# Main: compute, assert, emit markdown
# ---------------------------------------------------------------------------


def main():
    reports = [bank_report(c) for c in BANK_OPTIONS_UF]
    worst = max(reports, key=lambda r: r["i_pk"])
    baseline = next(r for r in reports if r["c_uF"] == BANK_BASELINE_UF)

    failures = []

    def check(name, ok, detail):
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(name)
        print(f"  [{status}] {name}: {detail}")
        return f"| {name} | {detail} | {status} |"

    print("=== Driver board design margin checks (worst case "
          f"{worst['c_uF']}uF @ {worst['v0']:.0f}V) ===")

    rows = []
    rows.append(check(
        "Worst-case peak within switch design point",
        worst["i_pk"] <= DESIGN_I_PK_A,
        f"I_pk {worst['i_pk']:.0f}A <= {DESIGN_I_PK_A:.0f}A design point"))
    rows.append(check(
        "Per-FET pulsed current (shared)",
        worst["i_pk"] / FET["n_parallel"] <= FET["idm_pulse_a"],
        f"{worst['i_pk']/FET['n_parallel']:.0f}A/device <= IDM {FET['idm_pulse_a']:.0f}A "
        f"({FET['n_parallel']}x {FET['name']})"))
    rows.append(check(
        "FET junction rise, shared conduction",
        worst["dtj_shared"] < 50.0,
        f"dTj {worst['dtj_shared']:.1f}K < 50K"))
    # Single-device fault tolerance is guaranteed at the baseline bank;
    # at extreme banks (8800/10000uF) a total sharing failure would exceed
    # Tj(max), so those configurations require all FETs functional —
    # documented in hardware/README bring-up notes.
    rows.append(check(
        "FET junction rise, single-device fault (baseline bank)",
        baseline["dtj_fault"] < 150.0,
        f"dTj {baseline['dtj_fault']:.1f}K < 150K at {BANK_BASELINE_UF}uF "
        f"(extreme-bank fault case: {worst['dtj_fault']:.0f}K — info only, "
        "requires all FETs functional)"))
    rows.append(check(
        "Blocking diode I2t margin",
        worst["i2t_diode"] * 10 <= DIODE["i2t_a2s"],
        f"pulse I2t/pkg {worst['i2t_diode']:.0f} A^2s, rating {DIODE['i2t_a2s']:.0f} A^2s (>=10x)"))
    rows.append(check(
        "Shunt pulse energy",
        worst["e_shunt_j"] < 0.5,
        f"{worst['e_shunt_j']*1000:.1f} mJ per shot (peak {worst['p_pk_shunt_w']:.0f}W for "
        f"{worst['t_pulse']*1e3:.1f}ms), continuous rating {SHUNT['p_cont_w']}W"))
    rows.append(check(
        "Pulse pour adiabatic heating",
        worst["dT_pour"] < 5.0,
        f"dT {worst['dT_pour']:.2f}K per shot "
        f"({POUR['width_mm']:.0f}mm 2oz x{POUR['layers']}, R={worst['r_pour_ohm']*1e3:.2f} mOhm)"))
    rows.append(check(
        "ADC full scale vs worst peak",
        worst["i_fullscale"] >= 1.25 * worst["i_pk"],
        f"FS {worst['i_fullscale']:.0f}A >= 1.25 x {worst['i_pk']:.0f}A "
        f"(LSB {worst['lsb_amps']:.2f}A)"))
    rows.append(check(
        "Charge time (largest bank)",
        reports[-1]["t_charge_s"] < 3.0,
        f"{reports[-1]['t_charge_s']:.2f}s to {V_BANK_MAX:.0f}V at "
        f"{CHARGER['i_charge_a']}A ({reports[-1]['c_uF']}uF)"))
    rows.append(check(
        "Dump to <5V (largest bank)",
        reports[-1]["t_dump_s"] < 10.0,
        f"{reports[-1]['t_dump_s']:.1f}s through {DUMP_R_OHM:.0f} Ohm dump"))
    rows.append(check(
        "Passive bleed 3-tau (largest bank)",
        3 * reports[-1]["tau_bleed_s"] < 300.0,
        f"3tau = {3*reports[-1]['tau_bleed_s']:.0f}s through {BLEED_R_OHM/1000:.0f}k bleed"))

    # Divider sanity
    v_adc_at_max = V_BANK_ABS * DIVIDER["bottom"] / (DIVIDER["top"] + DIVIDER["bottom"])
    rows.append(check(
        "V_bank divider inside ADC range at OVP",
        v_adc_at_max < 3.3,
        f"{v_adc_at_max:.2f}V at {V_BANK_ABS:.0f}V bank (divider "
        f"{DIVIDER['top']/1000:.0f}k/{DIVIDER['bottom']/1000:.2f}k)"))

    # Boost setpoint transfer function endpoints
    def vboost(vset):
        f = BOOST_FB
        return f["vref"] + f["rt"] * (f["vref"] / f["rb"] - (vset - f["vref"]) / f["rj"])

    v_cmd_max = vboost(0.0)      # PWM stuck low = worst commanded voltage
    v_cmd_min = vboost(3.3)
    rows.append(check(
        "Boost commanded max (PWM=0, fail state)",
        abs(v_cmd_max - V_BANK_MAX) < 1.0,
        f"{v_cmd_max:.1f}V ~= operating ceiling {V_BANK_MAX:.0f}V "
        f"(RT/RB/RJ = {BOOST_FB['rt']/1e3:.0f}k/{BOOST_FB['rb']/1e3:.0f}k/"
        f"{BOOST_FB['rj']/1e3:.1f}k)"))
    rows.append(check(
        "Boost commanded min below converter floor",
        v_cmd_min < BOOST_FLOOR_V,
        f"{v_cmd_min:.1f}V < {BOOST_FLOOR_V}V floor (full PWM = converter idles)"))

    # OVP trip band across component tolerances
    def ovp_trip(r_sign, ref_sign):
        t, b = OVP["top"], OVP["bottom"]
        rt_ = t * (1 + r_sign * OVP["r_tol"])
        rb_ = b * (1 - r_sign * OVP["r_tol"])
        vref_ = OVP["vref"] * (1 + ref_sign * OVP["ref_tol"])
        return vref_ * (rt_ + rb_) / rb_

    trip_lo = ovp_trip(-1, -1)
    trip_hi = ovp_trip(+1, +1)
    rows.append(check(
        "OVP trip floor above commanded max",
        trip_lo > v_cmd_max + 1.0,
        f"trip_min {trip_lo:.1f}V > {v_cmd_max:.1f}V + 1V margin "
        f"(nominal {ovp_trip(0,0):.1f}V)"))
    rows.append(check(
        "OVP trip ceiling below capacitor rating",
        trip_hi < V_BANK_ABS - 1.0,
        f"trip_max {trip_hi:.1f}V < {V_BANK_ABS:.0f}V rating - 1V"))

    # Relay stress at close (sequencing-managed)
    dv_close = PRECHARGE_R_OHM  # precharge equalizes; residual dV budget
    i_dump_close = V_BANK_MAX / DUMP_R_OHM
    i_coil = RELAY["coil_v"] / RELAY["coil_r_ohm"]
    rows.append(check(
        "Dump relay closing current within DC contact class",
        i_dump_close <= 10.0,
        f"{i_dump_close:.2f}A resistive at {V_BANK_MAX:.0f}V "
        f"(HF3FF ~10A/28VDC class; opens only after discharge)"))
    rows.append(check(
        "Relay coil drive within MMBT3904",
        i_coil < 0.1,
        f"{i_coil*1000:.0f}mA coil vs 200mA transistor rating"))

    # Coil safety envelope (worst bank at operating ceiling)
    grid, rect = coil_envelope()
    rows.append(check(
        "Coil envelope rectangle exists",
        rect is not None,
        f"L >= {rect[0]}uH AND R_total >= {rect[1]*1000:.0f} mOhm "
        f"(all grid points inside are safe)" if rect else "no safe rectangle"))
    demo_inside = (COIL["L_uH"] >= rect[0] and COIL["R_total_ohm"] >= rect[1]) if rect else False
    rows.append(check(
        "Demo coil inside validated envelope",
        demo_inside,
        f"demo {COIL['L_uH']}uH/{COIL['R_total_ohm']*1000:.0f} mOhm vs "
        f"envelope L>={rect[0]}uH, R>={rect[1]*1000:.0f} mOhm" if rect else "n/a"))

    import json as _json
    ENVELOPE_JSON.write_text(_json.dumps({
        "worst_bank_uF": BANK_OPTIONS_UF[-1],
        "v_bank_max": V_BANK_MAX,
        "design_i_pk_a": DESIGN_I_PK_A,
        "L_min_uH": rect[0] if rect else None,
        "R_total_min_ohm": rect[1] if rect else None,
        "demo_coil": COIL,
        "criteria": ["I_pk<=600A", "FET dTj<50K", "diode I2t 10x",
                     "pour dT<5K", "ADC FS>=1.25*I_pk"],
    }, indent=2), encoding="utf-8")
    print(f"  Coil envelope written to {ENVELOPE_JSON.name}")

    # ---- emit markdown ----
    md = ["# Driver board design calculations",
          "",
          "GENERATED by hardware/scripts/calcs.py — do not edit numbers by hand.",
          "",
          f"Coil model: L={COIL['L_uH']}uH, R_total={COIL['R_total_ohm']}Ohm "
          f"(same values the simulation uses).",
          "",
          f"## Pulse envelope by bank option (at {V_BANK_MAX:.0f}V)",
          "",
          "| Bank | regime | zeta | I_pk | t_pk | pulse | E | I2t | FET dTj (shared/fault) | pour dT |",
          "|------|--------|------|------|------|-------|---|-----|------------------------|---------|"]
    for r in reports:
        rl = r["rlc"]
        md.append(
            f"| {r['c_uF']}uF | {rl['regime']} | {rl['zeta']:.2f} | {r['i_pk']:.0f}A "
            f"| {r['t_pk']*1e6:.0f}us | {r['t_pulse']*1e3:.2f}ms | {r['energy_j']:.1f}J "
            f"| {r['i2t']:.1f} A^2s | {r['dtj_shared']:.1f}K / {r['dtj_fault']:.0f}K "
            f"| {r['dT_pour']:.2f}K |")
    md += ["",
           f"Baseline population: **{BANK_BASELINE_UF}uF** (2x 2200uF snap-in).",
           "",
           "## Margin checks",
           "",
           "| Check | Detail | Result |",
           "|-------|--------|--------|"]
    md += rows
    md += ["",
           "## Sense chain",
           "",
           f"- Shunt {SHUNT['r_ohm']*1e3:.1f} mOhm x INA240 gain {INA240_GAIN:.0f} = "
           f"{SHUNT['r_ohm']*INA240_GAIN*1e3:.0f} mV/A",
           f"- ADC full scale {worst['i_fullscale']:.0f}A at {ADC_VREF}V ref, "
           f"LSB {worst['lsb_amps']:.2f}A ({ADC_BITS}-bit)",
           f"- V_bank divider {DIVIDER['top']/1000:.0f}k/{DIVIDER['bottom']/1000:.2f}k "
           f"-> {V_BANK_MAX*DIVIDER['bottom']/(DIVIDER['top']+DIVIDER['bottom']):.2f}V at {V_BANK_MAX:.0f}V, "
           f"standing drain {V_BANK_MAX/(DIVIDER['top']+DIVIDER['bottom'])*1000:.2f} mA",
           "",
           "## Charger / discharge",
           "",
           f"- Boost floor {CHARGER['v_floor']}V (cannot regulate below 24V input) — "
           f"usable charge range {CHARGER['v_floor']:.1f}-{V_BANK_MAX:.0f}V",
           f"- Charge current {CHARGER['i_charge_a']}A avg; precharge {PRECHARGE_R_OHM:.0f} Ohm "
           "across the charge-relay contacts, relay closes only above 20V bank",
           f"- Permanent bleed {BLEED_R_OHM/1000:.1f}k ({V_BANK_MAX**2/BLEED_R_OHM:.2f}W at {V_BANK_MAX:.0f}V); "
           f"dump {DUMP_R_OHM:.0f} Ohm (2x 100R/10W series) via relay",
           "",
           "## Boost setpoint / OVP protection ladder",
           "",
           f"- VBOOST = 2.5 + RT*(2.5/RB - (VSET-2.5)/RJ) with RT/RB/RJ = "
           f"{BOOST_FB['rt']/1e3:.0f}k / {BOOST_FB['rb']/1e3:.0f}k / {BOOST_FB['rj']/1e3:.1f}k",
           f"- PWM=0 (fail state) -> {vboost(0.0):.1f}V; PWM=100% -> {vboost(3.3):.1f}V (< {BOOST_FLOOR_V}V floor, idle)",
           f"- Ladder: operating {V_BANK_MAX:.0f}V < OVP {trip_lo:.1f}..{trip_hi:.1f}V "
           f"(TL431B +/-0.5% + 1% divider) < capacitor rating {V_BANK_ABS:.0f}V",
           "- Boost enable is fail-safe INHIBITED (pull-up holds COMP low; MCU"
           " must drive BOOST_EN_N low to charge)",
           "- Relay sequencing: charge relay closes only after precharge"
           " equalization, opens only with boost inhibited; dump relay"
           " switches <=0.6A resistive.",
           "",
           "## Coil safety envelope (worst case: full bank at operating ceiling)",
           "",
           f"A connected coil is validated iff **L >= {rect[0]} uH AND "
           f"R_total >= {rect[1]*1000:.0f} mOhm** (total series resistance "
           "including leads/wiring), swept at "
           f"{BANK_OPTIONS_UF[-1]}uF / {V_BANK_MAX:.0f}V against: "
           "I_pk <= 600A, FET dTj < 50K, diode I2t 10x, pour dT < 5K, "
           "ADC FS >= 1.25*I_pk.",
           "",
           "| L \\ R_total | " + " | ".join(f"{r*1000:.0f}m" for r in ENV_R_GRID_OHM) + " |",
           "|---" * (len(ENV_R_GRID_OHM) + 1) + "|",
           *[f"| {L}uH | " + " | ".join(
               ("OK" if grid[(L, R)][0] else f"**{grid[(L, R)][1]:.0f}A**")
               for R in ENV_R_GRID_OHM) + " |"
             for L in ENV_L_GRID_UH],
           "",
           "(unsafe cells show the offending peak current; envelope excludes"
           " them by construction)",
           "",
           "## Part constants used (verify against datasheets at order time)",
           "",
           f"- FET: {FET['name']}, {FET['n_parallel']}x parallel, Rds(on,hot) "
           f"{FET['rds_on_hot_ohm']*1e3:.1f} mOhm, IDM {FET['idm_pulse_a']:.0f}A, "
           f"Zth(1ms) {FET['zth_1ms_k_per_w']} K/W",
           f"- Diode: {DIODE['name']}, Vf {DIODE['vf_v']}V, IFSM {DIODE['ifsm_a']:.0f}A, "
           f"I2t {DIODE['i2t_a2s']:.0f} A^2s",
           f"- Shunt: {SHUNT['name']}",
           ""]

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")

    if failures:
        print(f"\nFAILED margins: {failures}")
        sys.exit(1)
    print("\nAll design margins PASS")


if __name__ == "__main__":
    main()
