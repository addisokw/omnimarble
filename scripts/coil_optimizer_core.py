"""Core coil optimization logic — shared by CLI and Gradio UI.

Extracted from optimize_coil_design.py to allow multiple consumers
(CLI script, web UI) without code duplication.
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from analytical_bfield import (
    MU_0_MM,
    ferromagnetic_force,
    solenoid_field,
    solenoid_field_gradient,
)
from rlc_circuit import (
    COPPER_SPECIFIC_HEAT,
    compute_dc_resistance,
    compute_multilayer_inductance,
    compute_rlc_params,
    compute_winding_geometry,
    coupled_rlc_step_substep,
    eddy_braking_force,
    rlc_current,
    rlc_current_with_cutoff,
    saturated_force,
    wire_temperature_rise,
)

# ============================================================
# Constants
# ============================================================

MARBLE_RADIUS_MM = 5.0
MARBLE_MASS_KG = 0.004  # ~4g steel marble
CHI_EFF = 3.0
V_MARBLE_MM3 = (4 / 3) * math.pi * MARBLE_RADIUS_MM ** 3

# Physics constants for enhanced simulation
CUTOFF_GATE_OFFSET_MM = 5.0    # MOSFET cutoff gate (from config sensor_cutoff_offset_mm)
MARBLE_SATURATION_T = 1.8      # steel marble saturation flux density
MARBLE_CONDUCTIVITY = 6e6      # S/m, for eddy current calculation

# AWG wire diameters (mm)
AWG_DIAMETERS = {
    18: 1.024,
    20: 0.812,
    22: 0.644,
    24: 0.511,
    26: 0.405,
}

# Scoring weights
W_VEL = 1.0
W_DANGER = 0.15
W_COST = 0.10
W_BUILD = 0.08
W_THERMAL = 0.20
V_REF = 5.0  # m/s, reference boost for log-linear normalization

# PINN field profile settings
N_FIELD_POINTS = 200
Z_FIELD_RANGE_MM = 80.0  # -80 to +80 mm from coil center

# Simulation settings
SIM_DT = 5e-6  # 5 us timestep
ENTRY_GATE_OFFSET_MM = -20.0  # entry gate position relative to coil center (from config)
APPROACH_VELOCITY_MM_S = 500.0  # conservative gravity-fed approach (~13mm drop)
SIM_Z_END = 80.0  # mm, simulation ends well after coil


# ============================================================
# User constraints
# ============================================================

@dataclass
class UserConstraints:
    max_voltage_V: float = 450.0
    max_current_A: float = 4000.0
    thinnest_wire_awg: int = 26  # higher AWG = thinner; 26 means all gauges allowed
    max_turns: int = 80
    max_temp_rise_C: float = 200.0
    n_samples: int = 2000
    seed: int = 42
    target_boost_ms: float | None = None  # None = maximize boost (default)


@dataclass
class OptimizationResult:
    scored: list  # all scored candidates, sorted by score descending
    coupled_top: list  # top N reranked with coupled ODE
    n_samples: int
    n_valid: int
    n_rejected: int
    eval_time_s: float
    rerank_time_s: float
    constraints: UserConstraints = field(default_factory=UserConstraints)


# ============================================================
# Model loading
# ============================================================

def load_pinn_designspace(device):
    """Load the v7 designspace PINN checkpoint with validation checks."""
    import torch
    from train_pinn import BFieldPINN

    ckpt_path = ROOT / "models" / "pinn_checkpoint" / "pinn_designspace.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = BFieldPINN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Validate: current_normalized must be True
    current_normalized = bool(model.current_normalized.item())
    if not current_normalized:
        raise RuntimeError("pinn_designspace.pt does not have current_normalized=True")

    # Validate: step >= 200000 if present
    step = checkpoint.get("step", None)
    if step is not None:
        if step < 200000:
            raise RuntimeError(f"checkpoint step={step} < 200000 (not fully trained)")

    return model, device


# ============================================================
# PINN field profile (batch precompute)
# ============================================================

def precompute_field_profile(model, device, N, R_mean, L):
    """Precompute Bz and dBz/dz along the axis at I=1A.

    Returns z_arr, Bz_per_A, dBz_dz_per_A as numpy arrays.
    Since B ~ I for a current-normalized PINN, force ~ I^2.
    """
    import torch

    z_arr = np.linspace(-Z_FIELD_RANGE_MM, Z_FIELD_RANGE_MM, N_FIELD_POINTS).astype(np.float32)
    r_arr = np.full(N_FIELD_POINTS, 0.1, dtype=np.float32)  # near axis
    I_norm = 1.0  # current-normalized

    n = len(z_arr)
    inputs = np.column_stack([
        r_arr, z_arr,
        np.full(n, I_norm, dtype=np.float32),
        np.full(n, float(N), dtype=np.float32),
        np.full(n, float(R_mean), dtype=np.float32),
        np.full(n, float(L), dtype=np.float32),
    ]).astype(np.float32)

    inp_t = torch.tensor(inputs, device=device, requires_grad=True)
    out = model(inp_t)
    B_z = out[:, 2]  # Bz component

    # Autograd for dBz/dz
    grad_Bz = torch.autograd.grad(
        B_z, inp_t, grad_outputs=torch.ones_like(B_z),
        create_graph=False, retain_graph=False,
    )[0]

    # Scale by I (current_normalized=True means output is B/I)
    Bz_per_A = B_z.detach().cpu().numpy() * I_norm  # already at I=1
    dBz_dz_per_A = grad_Bz[:, 1].detach().cpu().numpy() * I_norm

    return z_arr, Bz_per_A, dBz_dz_per_A


# ============================================================
# Latin Hypercube Sampling
# ============================================================

def latin_hypercube_sample(n_samples, n_dims, rng):
    """Generate LHS samples in [0, 1]^n_dims."""
    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        samples[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
    return samples


def generate_candidates(n_samples, rng, constraints=None):
    """Generate candidate designs via Latin Hypercube Sampling.

    Returns list of dicts with design parameters.
    """
    if constraints is None:
        constraints = UserConstraints()

    # Filter AWG choices based on constraints
    all_awg = [18, 20, 22, 24, 26]
    awg_choices = [g for g in all_awg if g <= constraints.thinnest_wire_awg]
    if not awg_choices:
        awg_choices = [all_awg[0]]  # fallback to thickest wire

    max_V0 = min(constraints.max_voltage_V, 450.0)
    max_N = min(constraints.max_turns, 80)

    # 6 dimensions: N, inner_radius, L, wire_gauge_idx, V0, C_uF
    lhs = latin_hypercube_sample(n_samples, 6, rng)

    candidates = []
    for i in range(n_samples):
        N = int(round(10 + lhs[i, 0] * (max_N - 10)))  # 10 to max_N
        inner_radius = 6.0 + lhs[i, 1] * 12.0  # 6-18 mm
        L = 15.0 + lhs[i, 2] * 45.0  # 15-60 mm
        awg_idx = int(lhs[i, 3] * len(awg_choices)) % len(awg_choices)
        awg = awg_choices[awg_idx]
        V0 = 20.0 + lhs[i, 4] * (max_V0 - 20.0)  # 20 to max_V0
        C_uF = 50.0 + lhs[i, 5] * 4650.0  # 50-4700 uF

        candidates.append({
            "N": N,
            "inner_radius_mm": inner_radius,
            "length_mm": L,
            "wire_gauge_awg": awg,
            "wire_diameter_mm": AWG_DIAMETERS[awg],
            "V0": V0,
            "C_uF": C_uF,
        })
    return candidates


# ============================================================
# Candidate evaluation
# ============================================================

def evaluate_candidate(cand, model, device, field_cache, constraints=None):
    """Evaluate a single candidate design.

    Returns result dict or None if hard-rejected.
    """
    if constraints is None:
        constraints = UserConstraints()

    N = cand["N"]
    inner_r = cand["inner_radius_mm"]
    L = cand["length_mm"]
    wire_d = cand["wire_diameter_mm"]
    awg = cand["wire_gauge_awg"]
    V0 = cand["V0"]
    C_uF = cand["C_uF"]

    # --- Step 1: Winding geometry ---
    insulation = 0.035
    wire_pitch = wire_d + 2 * insulation
    turns_per_layer = max(1, int(L / wire_pitch))
    num_layers = math.ceil(N / turns_per_layer)
    winding_depth = num_layers * wire_pitch
    outer_r = inner_r + winding_depth
    R_mean = inner_r + winding_depth / 2

    # Hard reject: inner radius too small for marble
    if inner_r < MARBLE_RADIUS_MM + wire_d / 2:
        return None

    # Hard reject: R_mean outside PINN range [8, 20]
    if R_mean < 8.0 or R_mean > 20.0:
        return None

    params = {
        "inner_radius_mm": inner_r,
        "outer_radius_mm": outer_r,
        "length_mm": L,
        "num_turns": N,
        "wire_diameter_mm": wire_d,
        "insulation_thickness_mm": insulation,
    }
    geom = compute_winding_geometry(params)

    # --- Step 2: Electrical params ---
    R_dc = compute_dc_resistance(geom["wire_length_mm"], geom["wire_cross_section_mm2"])
    # Add ESR + wiring resistance estimate
    R_total = R_dc + 0.03  # ~30 mohm for ESR + wiring

    L_uH = compute_multilayer_inductance(N, R_mean, L, geom["winding_depth_mm"])

    rlc_params = {
        "capacitance_uF": C_uF,
        "charge_voltage_V": V0,
        "inductance_uH": L_uH,
        "total_resistance_ohm": R_total,
    }
    rlc = compute_rlc_params(rlc_params)

    # Hard reject: peak current exceeds constraint
    if rlc["peak_current_A"] > constraints.max_current_A:
        return None

    # --- Step 3: Precompute field profile ---
    cache_key = (N, R_mean, L)
    if cache_key in field_cache:
        z_field, Bz_per_A, dBz_dz_per_A = field_cache[cache_key]
    else:
        z_field, Bz_per_A, dBz_dz_per_A = precompute_field_profile(
            model, device, N, R_mean, L
        )
        field_cache[cache_key] = (z_field, Bz_per_A, dBz_dz_per_A)

    # --- Step 4: 1D launch simulation ---
    entry_vel, exit_vel, boost = simulate_launch(z_field, Bz_per_A, dBz_dz_per_A, rlc)

    # --- Step 5: Thermal rise ---
    delta_T = compute_thermal_rise(rlc, R_dc, geom["wire_mass_g"])

    # Hard reject: temperature rise exceeds constraint
    if delta_T > constraints.max_temp_rise_C:
        return None

    # --- Step 6: Scoring ---
    result = {
        **cand,
        "outer_radius_mm": outer_r,
        "R_mean_mm": R_mean,
        "num_layers": geom["num_layers"],
        "wire_length_mm": geom["wire_length_mm"],
        "wire_mass_g": geom["wire_mass_g"],
        "R_dc_ohm": R_dc,
        "R_total_ohm": R_total,
        "L_uH": L_uH,
        "regime": rlc["regime"],
        "peak_current_A": rlc["peak_current_A"],
        "pulse_duration_s": rlc["effective_pulse_duration_s"],
        "stored_energy_J": rlc["stored_energy_J"],
        "entry_velocity_ms": entry_vel,
        "exit_velocity_ms": exit_vel,
        "boost_ms": boost,
        "delta_T_C": delta_T,
    }
    return result


def simulate_launch(z_field, Bz_per_A, dBz_dz_per_A, rlc):
    """Euler integration of marble through precomputed field profile.

    Models the coaster entry scenario: marble arrives at the entry gate
    (z = ENTRY_GATE_OFFSET_MM) with APPROACH_VELOCITY_MM_S, coil fires
    immediately. Returns (entry_velocity_ms, exit_velocity_ms, boost_ms).

    Physics effects applied:
    - MOSFET cutoff: current cut when marble passes z = +CUTOFF_GATE_OFFSET_MM
    - Magnetic saturation: force capped when internal B > MARBLE_SATURATION_T
    - Eddy braking: opposes motion proportional to (dB/dt)^2
    """
    marble_params = {
        "chi_eff": CHI_EFF,
        "volume_mm3": V_MARBLE_MM3,
        "saturation_T": MARBLE_SATURATION_T,
    }
    marble_eddy_params = {
        "conductivity_S_per_m": MARBLE_CONDUCTIVITY,
        "radius_mm": MARBLE_RADIUS_MM,
        "volume_mm3": V_MARBLE_MM3,
    }

    pulse_dur = rlc["effective_pulse_duration_s"]
    dt = SIM_DT
    z = ENTRY_GATE_OFFSET_MM  # start at entry gate
    v = APPROACH_VELOCITY_MM_S  # mm/s, moving toward coil center
    v_entry = v

    t = 0.0
    t_end = max(pulse_dur * 3, 0.03)  # simulate 3x pulse or 30ms

    # MOSFET cutoff tracking
    cutoff_triggered = False
    t_cutoff = float('inf')

    # Eddy braking: track previous Bz for dB/dt
    Bz_prev = 0.0

    while t < t_end and z < SIM_Z_END:
        # MOSFET cutoff: trigger when marble passes the cutoff gate
        if not cutoff_triggered and z >= CUTOFF_GATE_OFFSET_MM:
            cutoff_triggered = True
            t_cutoff = t

        I = rlc_current_with_cutoff(t, t_cutoff, rlc)

        # Interpolate field at marble position (scaled by current)
        Bz = np.interp(z, z_field, Bz_per_A) * I
        dBz_dz = np.interp(z, z_field, dBz_dz_per_A) * I

        # Saturated force (replaces simple prefactor * Bz * dBz_dz)
        F_mN = saturated_force(Bz, dBz_dz, marble_params)

        # Eddy braking force
        dBdt = (Bz - Bz_prev) / dt  # T/s
        Bz_prev = Bz
        F_eddy = eddy_braking_force(dBdt, v, marble_eddy_params)
        F_total = F_mN + F_eddy

        # 1 mN = 1e-3 N; a = F/m in m/s^2, then *1000 for mm/s^2
        a = F_total / MARBLE_MASS_KG  # mm/s^2

        v += a * dt
        z += v * dt
        t += dt

    entry_ms = v_entry / 1000.0
    exit_ms = v / 1000.0
    boost_ms = exit_ms - entry_ms
    return entry_ms, exit_ms, boost_ms


def compute_thermal_rise(rlc, R_dc, wire_mass_g):
    """Numerically integrate temperature rise over the pulse."""
    pulse_dur = rlc["effective_pulse_duration_s"]
    n_steps = 100
    dt = pulse_dur / n_steps
    delta_T = 0.0

    for i in range(n_steps):
        t = i * dt
        I = rlc_current(t, rlc)
        delta_T += wire_temperature_rise(I, R_dc, dt, wire_mass_g)

    return delta_T


# ============================================================
# Scoring functions
# ============================================================

def voltage_danger(V0):
    """Voltage danger penalty (0-1 scale)."""
    if V0 <= 50:
        return 0.0
    elif V0 <= 120:
        # Quadratic ramp to 0.2
        frac = (V0 - 50) / 70
        return 0.2 * frac ** 2
    elif V0 <= 400:
        # Steep ramp from 0.2 to 0.7
        frac = (V0 - 120) / 280
        return 0.2 + 0.5 * frac
    else:
        # Lethal territory: 0.7+
        frac = (V0 - 400) / 50
        return 0.7 + 0.3 * min(frac, 1.0)


def capacitor_cost(V0, C_uF):
    """Capacitor cost proxy (0-1 scale). Reference: 470uF @ 50V."""
    ref = 470 * 50 ** 1.5
    cost = C_uF * V0 ** 1.5
    return min(cost / ref, 3.0) / 3.0  # Normalize so ref=0.33, max=1.0


def build_difficulty(N, R_mean, L, awg, num_layers, inner_r):
    """Construction difficulty penalty (0-1 scale)."""
    penalty = 0.0

    # Many turns
    if N > 30:
        penalty += 0.2 * min((N - 30) / 50, 1.0)

    # Tight bore (< 2mm clearance over marble)
    clearance = inner_r - MARBLE_RADIUS_MM
    if clearance < 2.0:
        penalty += 0.3 * (1 - clearance / 2.0)

    # Long coil
    if L > 50:
        penalty += 0.1 * min((L - 50) / 10, 1.0)

    # Multi-layer winding
    if num_layers > 1:
        penalty += 0.15 * min((num_layers - 1) / 3, 1.0)

    # Non-standard wire gauge (20 AWG = easiest)
    gauge_penalty = {18: 0.05, 20: 0.0, 22: 0.03, 24: 0.08, 26: 0.12}
    penalty += gauge_penalty.get(awg, 0.1)

    return min(penalty, 1.0)


def thermal_penalty(delta_T):
    """Thermal penalty (0-1 scale)."""
    if delta_T <= 50:
        return 0.0
    elif delta_T <= 100:
        frac = (delta_T - 50) / 50
        return 0.3 * frac ** 2
    elif delta_T <= 200:
        frac = (delta_T - 100) / 100
        return 0.3 + 0.4 * frac
    else:
        return 1.0


def compute_score(result):
    """Compute multi-objective composite score."""
    # Log-linear boost normalization: 5 m/s -> 1.0, unsaturated above
    boost = max(result["boost_ms"], 0)
    velocity_norm = math.log1p(boost) / math.log1p(V_REF)

    vd = voltage_danger(result["V0"])
    cc = capacitor_cost(result["V0"], result["C_uF"])
    bd = build_difficulty(
        result["N"], result["R_mean_mm"], result["length_mm"],
        result["wire_gauge_awg"], result["num_layers"], result["inner_radius_mm"],
    )
    tp = thermal_penalty(result["delta_T_C"])

    combined_penalty = (W_DANGER * vd + W_COST * cc + W_BUILD * bd + W_THERMAL * tp)

    score = W_VEL * velocity_norm - combined_penalty

    result["velocity_norm"] = velocity_norm
    result["penalty_voltage"] = vd
    result["penalty_cost"] = cc
    result["penalty_build"] = bd
    result["penalty_thermal"] = tp
    result["combined_penalty"] = combined_penalty
    result["score"] = score

    return result


# ============================================================
# Analytical velocity validation
# ============================================================

def analytical_boost(cand, rlc_params_for_cand):
    """Compute boost using analytical Biot-Savart field for validation.

    Returns (entry_velocity_ms, exit_velocity_ms, boost_ms).
    """
    N = cand["N"]
    inner_r = cand["inner_radius_mm"]
    outer_r = cand["outer_radius_mm"]
    L = cand["length_mm"]

    rlc = compute_rlc_params(rlc_params_for_cand)

    # Precompute analytical field profile at I=1A
    z_arr = np.linspace(-Z_FIELD_RANGE_MM, Z_FIELD_RANGE_MM, N_FIELD_POINTS)
    coil_params = {
        "current_A": 1.0,
        "num_turns": N,
        "inner_radius_mm": inner_r,
        "outer_radius_mm": outer_r,
        "length_mm": L,
    }

    Bz_per_A = np.zeros(N_FIELD_POINTS)
    dBz_dz_per_A = np.zeros(N_FIELD_POINTS)
    for i, z in enumerate(z_arr):
        _, bz = solenoid_field(0.1, float(z), coil_params)
        Bz_per_A[i] = bz
        _, _, _, dBz_dz_val = solenoid_field_gradient(0.1, float(z), coil_params)
        dBz_dz_per_A[i] = dBz_dz_val

    return simulate_launch(z_arr.astype(np.float32), Bz_per_A.astype(np.float32),
                           dBz_dz_per_A.astype(np.float32), rlc)


# ============================================================
# Coupled ODE reranking (top 50)
# ============================================================

def rerank_with_coupled_ode(top_candidates, model, device, field_cache,
                            progress_callback=None, target_boost_ms=None):
    """Rerun top candidates with coupled electromechanical ODE for back-EMF.

    Uses coupled_rlc_step_substep for current (position-dependent inductance)
    instead of closed-form. Also applies cutoff, saturation, and eddy braking.

    Returns list of candidates with added boost_coupled_ms and rank_coupled fields.
    """
    marble_params = {
        "chi_eff": CHI_EFF,
        "volume_mm3": V_MARBLE_MM3,
        "saturation_T": MARBLE_SATURATION_T,
    }
    marble_eddy_params = {
        "conductivity_S_per_m": MARBLE_CONDUCTIVITY,
        "radius_mm": MARBLE_RADIUS_MM,
        "volume_mm3": V_MARBLE_MM3,
    }

    for idx, cand in enumerate(top_candidates):
        N = cand["N"]
        inner_r = cand["inner_radius_mm"]
        L_coil = cand["length_mm"]
        R_mean = cand["R_mean_mm"]

        # Build RLC params
        rlc_p = {
            "capacitance_uF": cand["C_uF"],
            "charge_voltage_V": cand["V0"],
            "inductance_uH": cand["L_uH"],
            "total_resistance_ohm": cand["R_total_ohm"],
        }
        rlc = compute_rlc_params(rlc_p)

        # Coil params for coupled ODE
        coil_params = {
            "inner_radius_mm": inner_r,
            "outer_radius_mm": cand["outer_radius_mm"],
            "length_mm": L_coil,
            "marble_radius_mm": MARBLE_RADIUS_MM,
            "chi_eff": CHI_EFF,
            "has_flyback_diode": True,
        }

        # Get field profile
        cache_key = (N, R_mean, L_coil)
        if cache_key in field_cache:
            z_field, Bz_per_A, dBz_dz_per_A = field_cache[cache_key]
        else:
            z_field, Bz_per_A, dBz_dz_per_A = precompute_field_profile(
                model, device, N, R_mean, L_coil
            )
            field_cache[cache_key] = (z_field, Bz_per_A, dBz_dz_per_A)

        # Initialize coupled ODE state
        C_F = rlc["capacitance_F"]
        V0 = rlc["charge_voltage_V"]
        state = {"I": 0.0, "Q_cap": C_F * V0}

        pulse_dur = rlc["effective_pulse_duration_s"]
        dt = SIM_DT
        z = ENTRY_GATE_OFFSET_MM
        v = APPROACH_VELOCITY_MM_S
        v_entry = v
        t = 0.0
        t_end = max(pulse_dur * 3, 0.03)

        cutoff_triggered = False
        I_at_cutoff = 0.0
        t_cutoff = float('inf')
        Bz_prev = 0.0

        R = rlc["total_resistance_ohm"]
        L_H = rlc["inductance_H"]

        while t < t_end and z < SIM_Z_END:
            # MOSFET cutoff check
            if not cutoff_triggered and z >= CUTOFF_GATE_OFFSET_MM:
                cutoff_triggered = True
                t_cutoff = t
                I_at_cutoff = state["I"]

            if not cutoff_triggered:
                # Coupled ODE step (position-dependent inductance, back-EMF)
                state = coupled_rlc_step_substep(
                    state, dt, coil_params, rlc, z, v
                )
                I = state.get("_I_rms", abs(state["I"]))
            else:
                # Approximate: RL decay after MOSFET opens (ignores residual cap energy)
                I = I_at_cutoff * math.exp(-(R / L_H) * (t - t_cutoff))
                I = max(I, 0.0)

            # Interpolate field at marble position (scaled by current)
            Bz = np.interp(z, z_field, Bz_per_A) * I
            dBz_dz = np.interp(z, z_field, dBz_dz_per_A) * I

            # Saturated force + eddy braking
            F_mN = saturated_force(Bz, dBz_dz, marble_params)
            dBdt = (Bz - Bz_prev) / dt
            Bz_prev = Bz
            F_eddy = eddy_braking_force(dBdt, v, marble_eddy_params)
            F_total = F_mN + F_eddy

            a = F_total / MARBLE_MASS_KG  # mm/s^2
            v += a * dt
            z += v * dt
            t += dt

        exit_ms = v / 1000.0
        entry_ms = v_entry / 1000.0
        cand["boost_coupled_ms"] = exit_ms - entry_ms

        if progress_callback and (idx + 1) % 10 == 0:
            progress_callback(idx + 1, len(top_candidates), "reranking")

    # Assign coupled rank
    if target_boost_ms is not None:
        # Band filter: ±20% of target (or ±0.1 m/s minimum)
        margin = max(target_boost_ms * 0.20, 0.1)
        in_band = []
        near_miss = []
        out_of_band = []
        for c in top_candidates:
            bc = c["boost_coupled_ms"]
            in_coupled_band = (target_boost_ms - margin <= bc
                               <= target_boost_ms + margin)
            c["meets_target_coupled"] = in_coupled_band
            if in_coupled_band:
                in_band.append(c)
            elif (target_boost_ms - margin <= c["boost_ms"]
                  <= target_boost_ms + margin):
                # Passed screening band but drifted in coupled — near-miss
                near_miss.append(c)
            else:
                out_of_band.append(c)
        in_band.sort(key=lambda r: r["combined_penalty"])
        near_miss.sort(key=lambda r: abs(r["boost_coupled_ms"]
                                         - target_boost_ms))
        out_of_band.sort(key=lambda r: r["combined_penalty"])
        coupled_sorted = in_band + near_miss + out_of_band
    else:
        for c in top_candidates:
            c["meets_target_coupled"] = True
        coupled_sorted = sorted(top_candidates, key=lambda r: r["boost_coupled_ms"], reverse=True)

    for rank, c in enumerate(coupled_sorted):
        c["rank_coupled"] = rank + 1

    return coupled_sorted


# ============================================================
# Plotting
# ============================================================

def plot_pareto(scored, save_path=None, recommended=None):
    """Plot Pareto front: boost vs combined penalty.

    Args:
        scored: All scored candidates (screening pass), sorted by score.
        save_path: If given, saves figure to disk and closes it.
        recommended: The final recommended design (e.g. coupled winner).
            If None, scored[0] is highlighted as "Screening #1".

    Always returns the figure object.
    """
    boosts = [r["boost_ms"] for r in scored]
    pens = [r["combined_penalty"] for r in scored]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(pens, boosts, s=8, alpha=0.3, c="steelblue", label="All candidates")

    # Highlight screening top 10
    for i, r in enumerate(scored[:10]):
        ax.scatter(r["combined_penalty"], r["boost_ms"],
                   s=80, c="red", zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(f"#{i+1}", (r["combined_penalty"], r["boost_ms"]),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    # Highlight the recommended design (coupled winner) or screening #1
    star = recommended if recommended is not None else scored[0]
    star_label = "Recommended (coupled)" if recommended is not None else "Screening #1"
    ax.scatter(star["combined_penalty"], star["boost_ms"],
               s=150, c="gold", zorder=6, edgecolors="black", linewidth=1.5,
               marker="*", label=star_label)

    ax.set_xlabel("Combined Penalty", fontsize=12)
    ax.set_ylabel("Boost (m/s)", fontsize=12)
    ax.set_title("Coil Design Pareto Front: Boost vs Penalty", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    return fig


# ============================================================
# Top-level optimization pipeline
# ============================================================

def run_optimization(constraints, model, device, progress_callback=None):
    """Run the full optimization pipeline.

    Args:
        constraints: UserConstraints instance.
        model: Loaded PINN model.
        device: torch device.
        progress_callback: Optional callable(current, total, phase_name).

    Returns:
        OptimizationResult with scored candidates and coupled reranking.
    """
    rng = np.random.default_rng(constraints.seed)
    n_samples = constraints.n_samples

    # Generate candidates
    candidates = generate_candidates(n_samples, rng, constraints)

    # Evaluate all candidates
    t0 = time.time()
    field_cache = {}
    results = []
    n_rejected = 0

    for i, cand in enumerate(candidates):
        result = evaluate_candidate(cand, model, device, field_cache, constraints)
        if result is None:
            n_rejected += 1
            continue
        result = compute_score(result)
        results.append(result)

        if progress_callback and (i + 1) % 200 == 0:
            progress_callback(i + 1, n_samples, "screening")

    eval_time = time.time() - t0

    if not results:
        return OptimizationResult(
            scored=[], coupled_top=[], n_samples=n_samples,
            n_valid=0, n_rejected=n_rejected,
            eval_time_s=eval_time, rerank_time_s=0.0,
            constraints=constraints,
        )

    target = constraints.target_boost_ms

    if target is not None:
        # Target-boost mode: band filter — keep designs within ±20% of
        # target (or ±0.1 m/s minimum for small targets), then sort by
        # lowest combined_penalty to surface the cheapest/safest option.
        margin = max(target * 0.20, 0.1)
        band = [r for r in results
                if target - margin <= r["boost_ms"] <= target + margin]
        if band:
            scored = sorted(band, key=lambda r: r["combined_penalty"])
        else:
            # No candidates in band — fall back to all, sorted by proximity
            scored = sorted(results,
                            key=lambda r: abs(r["boost_ms"] - target))
    else:
        # Default mode: sort by score descending
        scored = sorted(results, key=lambda r: r["score"], reverse=True)

    # Coupled ODE reranking for top 50
    n_rerank = min(50, len(scored))
    t1 = time.time()
    coupled_top = rerank_with_coupled_ode(
        scored[:n_rerank], model, device, field_cache, progress_callback,
        target_boost_ms=target,
    )
    rerank_time = time.time() - t1

    return OptimizationResult(
        scored=scored, coupled_top=coupled_top,
        n_samples=n_samples, n_valid=len(results), n_rejected=n_rejected,
        eval_time_s=eval_time, rerank_time_s=rerank_time,
        constraints=constraints,
    )
