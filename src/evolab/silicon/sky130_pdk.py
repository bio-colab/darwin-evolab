"""SkyWater 130nm Open-Source PDK specification and device physics models.

Provides standard parameters, design rules, PVT corners (TT/SS/FF), and MOS
transconductance/conductance calculators for SkyWater 130nm (sky130_fd_pr).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Sky130Corner(str, Enum):
    """Process-Voltage-Temperature (PVT) manufacturing corners."""
    TT = "TT"  # Typical-Typical: 27°C, 1.80V
    SS = "SS"  # Slow-Slow: 125°C, 1.62V (-10% VDD)
    FF = "FF"  # Fast-Fast: -40°C, 1.98V (+10% VDD)
    SF = "SF"  # Slow NMOS, Fast PMOS: 27°C, 1.80V
    FS = "FS"  # Fast NMOS, Slow PMOS: 27°C, 1.80V


@dataclass(frozen=True)
class CornerSpec:
    """Operating conditions and physical factor variations per corner."""
    name: Sky130Corner
    temp_c: float
    vdd: float
    vth_n_shift: float  # Absolute threshold shift (V)
    vth_p_shift: float
    mobility_scale: float  # Scaling factor for carrier mobility


CORNER_SPECS: dict[Sky130Corner, CornerSpec] = {
    Sky130Corner.TT: CornerSpec(
        name=Sky130Corner.TT,
        temp_c=27.0,
        vdd=1.80,
        vth_n_shift=0.0,
        vth_p_shift=0.0,
        mobility_scale=1.0,
    ),
    Sky130Corner.SS: CornerSpec(
        name=Sky130Corner.SS,
        temp_c=125.0,
        vdd=1.62,
        vth_n_shift=0.065,   # Higher Vth -> slower
        vth_p_shift=-0.065,  # More negative for PMOS
        mobility_scale=0.78, # Reduced mobility at high temp
    ),
    Sky130Corner.FF: CornerSpec(
        name=Sky130Corner.FF,
        temp_c=-40.0,
        vdd=1.98,
        vth_n_shift=-0.065,  # Lower Vth -> faster
        vth_p_shift=0.065,
        mobility_scale=1.24, # Enhanced mobility at low temp
    ),
    Sky130Corner.SF: CornerSpec(
        name=Sky130Corner.SF,
        temp_c=27.0,
        vdd=1.80,
        vth_n_shift=0.05,
        vth_p_shift=0.05,
        mobility_scale=0.92,
    ),
    Sky130Corner.FS: CornerSpec(
        name=Sky130Corner.FS,
        temp_c=27.0,
        vdd=1.80,
        vth_n_shift=-0.05,
        vth_p_shift=-0.05,
        mobility_scale=1.08,
    ),
}


@dataclass(frozen=True)
class Sky130DeviceParams:
    """Standard technological constants for SkyWater 130nm 1.8V transistors."""
    # Physical dimensions limits (in micrometers)
    l_min_um: float = 0.15
    l_max_um: float = 10.0
    w_min_um: float = 0.42
    w_max_um: float = 100.0
    grid_um: float = 0.005

    # Nominal supply voltage (V)
    vdd_nominal: float = 1.80

    # Gate oxide capacitance per unit area (F/m^2)
    # tox approx 4.1nm -> Cox = eps_ox / tox = 3.9 * 8.854e-12 / 4.1e-9 approx 8.42e-3
    cox: float = 8.42e-3

    # Nominal low-field mobilities (m^2 / V*s)
    mu_n0: float = 450e-4
    mu_p0: float = 140e-4

    # Nominal zero-bias threshold voltages (V)
    vth0_n: float = 0.48
    vth0_p: float = -0.52

    # Channel length modulation parameter (1/V)
    lambda_n: float = 0.06
    lambda_p: float = 0.09


SKY130_PARAMS = Sky130DeviceParams()


@dataclass
class TransistorSmallSignal:
    """Operating point and small-signal parameters for a sized transistor."""
    w_um: float
    l_um: float
    is_pmos: bool
    id_a: float
    vgs_v: float
    vds_v: float
    vth_v: float
    gm_s: float   # Transconductance (Siemens / A/V)
    gds_s: float  # Output conductance (Siemens / 1/ohm)
    ro_ohm: float # Small-signal output resistance (ohms)
    v_overdrive: float


def compute_transistor_operating_point(
    w_um: float,
    l_um: float,
    id_target_a: float,
    vds_v: float,
    is_pmos: bool = False,
    corner: Sky130Corner = Sky130Corner.TT,
) -> TransistorSmallSignal:
    """Computes small-signal parameters (gm, ro) given geometry and bias current in Sky130."""
    p = SKY130_PARAMS
    cs = CORNER_SPECS[corner]

    # Enforce physical design rule limits
    w = max(p.w_min_um, min(w_um, p.w_max_um))
    l = max(p.l_min_um, min(l_um, p.l_max_um))

    # Convert to meters
    w_m = w * 1e-6
    l_m = l * 1e-6

    # Apply corner scaling
    if not is_pmos:
        mu = p.mu_n0 * cs.mobility_scale
        vth = p.vth0_n + cs.vth_n_shift
        lam = p.lambda_n / (l / p.l_min_um)  # Channel-length dependent Early effect
    else:
        mu = p.mu_p0 * cs.mobility_scale
        vth = p.vth0_p + cs.vth_p_shift
        lam = p.lambda_p / (l / p.l_min_um)

    # Beta = mu * Cox * (W / L)
    beta = mu * p.cox * (w_m / l_m)

    # In saturation: Id = 0.5 * beta * (Vgs - Vth)^2 * (1 + lambda * Vds)
    # Vov = sqrt(2 * Id / (beta * (1 + lambda * Vds)))
    id_eff = max(id_target_a, 1e-12)
    lam_factor = 1.0 + lam * abs(vds_v)
    vov = math.sqrt(max(2.0 * id_eff / (beta * lam_factor), 1e-12))
    vgs = (vth + vov) if not is_pmos else (vth - vov)

    # Transconductance gm = 2 * Id / Vov = sqrt(2 * beta * Id)
    gm = (2.0 * id_eff) / max(vov, 1e-4)

    # Output conductance gds = lambda * Id / (1 + lambda * Vds)
    gds = (lam * id_eff) / max(lam_factor, 0.1)
    ro = 1.0 / max(gds, 1e-12)

    return TransistorSmallSignal(
        w_um=round(w, 3),
        l_um=round(l, 3),
        is_pmos=is_pmos,
        id_a=id_eff,
        vgs_v=round(vgs, 4),
        vds_v=round(vds_v, 4),
        vth_v=round(vth, 4),
        gm_s=gm,
        gds_s=gds,
        ro_ohm=ro,
        v_overdrive=round(vov, 4),
    )


def generate_sky130_spice_header(corner: Sky130Corner = Sky130Corner.TT) -> str:
    """Generates pure SPICE model cards representing SkyWater 130nm devices for ngspice."""
    cs = CORNER_SPECS[corner]
    p = SKY130_PARAMS

    vth_n = p.vth0_n + cs.vth_n_shift
    vth_p = abs(p.vth0_p + cs.vth_p_shift)
    kp_n = p.mu_n0 * p.cox * cs.mobility_scale
    kp_p = p.mu_p0 * p.cox * cs.mobility_scale

    return f"""* SkyWater 130nm Open-Source PDK Models (Corner: {corner.value})
* Temperature: {cs.temp_c}°C, VDD: {cs.vdd}V
.param sky130_vdd={cs.vdd}
.param sky130_temp={cs.temp_c}

* 1.8V Standard NMOS Device (sky130_fd_pr__nfet_01v8)
.model sky130_fd_pr__nfet_01v8 NMOS (
+ LEVEL=1
+ VTO={vth_n:.3f}
+ KP={kp_n:.4e}
+ GAMMA=0.45
+ PHI=0.70
+ LAMBDA={p.lambda_n:.3f}
+ TOX=4.1e-9
+ CJ=1.0e-3
+ CJSW=3.0e-10
)

* 1.8V Standard PMOS Device (sky130_fd_pr__pfet_01v8)
.model sky130_fd_pr__pfet_01v8 PMOS (
+ LEVEL=1
+ VTO=-{vth_p:.3f}
+ KP={kp_p:.4e}
+ GAMMA=0.40
+ PHI=0.65
+ LAMBDA={p.lambda_p:.3f}
+ TOX=4.1e-9
+ CJ=1.2e-3
+ CJSW=3.5e-10
)
"""
