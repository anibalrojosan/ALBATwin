"""
ALBA aqueous chemistry: reference dissociation constants and van't Hoff scaling.

Reference K_a and K_w in mol/L for ka_at_T are at T_REF_K = 298.15 K: default_dissociation_constants_ref_molar
builds them from docs/MATH_MODEL.md §1.2.7 (pK_a at 25 °C) and K_w ~ 1e-14 at 25 °C.

Table SI.6.1 (docs/supporting_informations/SI.6) lists some K_A at ~293 K; those paper values can
differ slightly from the §1.2.7 pack. The implementation uses §1.2.7 + 298.15 K as SSOT for K_ref.

default_dissociation_enthalpy_j_per_mol supplies Delta H° (infinite dilution, 298.15 K, 1 bar) consistent
with docs/notes/aqueous-acid-base-reaction-enthalpies.md, matching the same T_ref as K_ref.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# SI.6.4: gas constant [J K^-1 mol^-1]
R_GAS = 8.314462618

# Reference T for K_a,ref and K_w,ref in ka_at_T / scale_dissociation_constants_at_t [K]
T_REF_K = 298.15

# Molar masses for ALBA state totals [g mol^-1] (SI.6 uses 12, 14, 31)
M_C = 12.0
M_N = 14.0
M_P = 31.0

# --- Stage 1: foundations and unit conventions ---


@dataclass(frozen=True, slots=True)
class AlbaDissociationConstantsRef:
    """Dissociation constants at T_REF_K (298.15 K by default), in mol/L. Field names match SI.6.1."""

    ka_co2: float
    ka_hco3: float
    ka_nh4: float
    ka_hno2: float
    ka_hno3: float
    ka_h3po4: float
    ka_h2po4: float
    ka_hpo4: float
    kw: float


@dataclass(frozen=True, slots=True)
class AlbaDissociationEnthalpy:
    """
    Standard reaction enthalpy Delta H in J/mol per equilibrium for SI.6.4.

    Same keys as AlbaDissociationConstantsRef; dh_kw is for water ionization.
    """

    dh_co2: float
    dh_hco3: float
    dh_nh4: float
    dh_hno2: float
    dh_hno3: float
    dh_h3po4: float
    dh_h2po4: float
    dh_hpo4: float
    dh_kw: float


def default_dissociation_constants_ref_molar() -> AlbaDissociationConstantsRef:
    """
    Build K_a,ref from docs/MATH_MODEL.md section 1.2.7 (pK_a at 298.15 K).

    Returns all values in mol/L. kw is 1e-14 at 25 C (common convention).
    """
    return AlbaDissociationConstantsRef(
        ka_co2=10.0**-6.37,
        ka_hco3=10.0**-10.33,
        ka_nh4=10.0**-9.25,
        ka_hno2=10.0**-3.35,
        ka_hno3=10.0**1.64,
        ka_h3po4=10.0**-2.14,
        ka_h2po4=10.0**-7.21,
        ka_hpo4=10.0**-12.67,
        kw=1.0e-14,
    )


def default_dissociation_enthalpy_j_per_mol() -> AlbaDissociationEnthalpy:
    """
    Standard reaction enthalpy ΔH° in J/mol for van't Hoff (SI.6.4), K_a = [products]/[acid].

    Values follow the consolidated recommendations in the aqueous-equilibria thermodynamics
    note (CODATA / ATcT / NEA-TDB–based), infinite dilution, 298.15 K, 1 bar. dh_co2 is for
    the lumped CO2(aq)/H2CO3* first dissociation as in bioprocess models; dh_hno3 is 0 under
    the strong-acid convention (undissociated HNO3(aq) ≡ ions at infinite dilution).
    """
    return AlbaDissociationEnthalpy(
        dh_co2=9_155.0,
        dh_hco3=14_700.0,
        dh_nh4=52_201.0,
        dh_hno2=11_400.0,
        dh_hno3=0.0,
        dh_h3po4=-8_480.0,
        dh_h2po4=3_600.0,
        dh_hpo4=14_600.0,
        dh_kw=55_830.0,
    )


def ka_at_T(ka_ref_molar: float, delta_h_j_per_mol: float, t_celsius: float) -> float:
    """
    Scale equilibrium constant K from T_REF_K to t_celsius using SI.6.4.

    ka_ref_molar: K at T_REF_K in mol/L (for K_w use the same formula; units stay consistent with the reference).
    delta_h_j_per_mol: Delta H in J/mol for that reaction.
    t_celsius: temperature in degrees Celsius.

    Returns K at t_celsius in the same concentration units as ka_ref_molar.
    """
    t_k = t_celsius + 273.15
    return ka_ref_molar * math.exp(
        (delta_h_j_per_mol / R_GAS) * (1.0 / T_REF_K - 1.0 / t_k),
    )


def kw_at_T(kw_ref_molar: float, delta_h_kw_j_per_mol: float, t_celsius: float) -> float:
    """Water ionization constant at t_celsius; same math as ka_at_T. kw_ref_molar is often 1e-14 (mol/L)^2 at 25 C."""
    return ka_at_T(kw_ref_molar, delta_h_kw_j_per_mol, t_celsius)


def mol_c_per_m3_carbon_from_s_ic(s_ic_g_per_m3: float) -> float:
    """Moles of inorganic carbon per m3 from S_IC in g C/m3 (SI.6 row 7)."""
    return s_ic_g_per_m3 / M_C


def mol_n_per_m3_nitrogen_from_s_nh(s_nh_g_per_m3: float) -> float:
    """Moles of ammoniacal nitrogen per m3 from S_NH in g N/m3."""
    return s_nh_g_per_m3 / M_N


def mol_n_per_m3_nitrogen_from_s_no2(s_no2_g_per_m3: float) -> float:
    """Moles of nitrite nitrogen per m3 from S_NO2 in g N/m3."""
    return s_no2_g_per_m3 / M_N


def mol_n_per_m3_nitrogen_from_s_no3(s_no3_g_per_m3: float) -> float:
    """Moles of nitrate nitrogen per m3 from S_NO3 in g N/m3."""
    return s_no3_g_per_m3 / M_N


def mol_p_per_m3_phosphorus_from_s_po4(s_po4_g_per_m3: float) -> float:
    """Moles of inorganic phosphorus per m3 from S_PO4 in g P/m3."""
    return s_po4_g_per_m3 / M_P


def scale_dissociation_constants_at_t(
    ref: AlbaDissociationConstantsRef,
    dh: AlbaDissociationEnthalpy,
    t_celsius: float,
) -> AlbaDissociationConstantsRef:
    """Apply ka_at_T / kw_at_T to each field of ref using dh."""
    return AlbaDissociationConstantsRef(
        ka_co2=ka_at_T(ref.ka_co2, dh.dh_co2, t_celsius),
        ka_hco3=ka_at_T(ref.ka_hco3, dh.dh_hco3, t_celsius),
        ka_nh4=ka_at_T(ref.ka_nh4, dh.dh_nh4, t_celsius),
        ka_hno2=ka_at_T(ref.ka_hno2, dh.dh_hno2, t_celsius),
        ka_hno3=ka_at_T(ref.ka_hno3, dh.dh_hno3, t_celsius),
        ka_h3po4=ka_at_T(ref.ka_h3po4, dh.dh_h3po4, t_celsius),
        ka_h2po4=ka_at_T(ref.ka_h2po4, dh.dh_h2po4, t_celsius),
        ka_hpo4=ka_at_T(ref.ka_hpo4, dh.dh_hpo4, t_celsius),
        kw=kw_at_T(ref.kw, dh.dh_kw, t_celsius),
    )


# --- Stage 2: speciation (SI.6.1, rows 2–14); [H+] and all species in mol m^-3; K_a, K_w in mol/L ---


@dataclass(frozen=True, slots=True)
class SpeciationTotals:
    """Total molar concentrations per SI.6.1 mass-balance rows (mol m^-3)."""

    c_tot_nh3: float
    """Total ammonia nitrogen NH3 + NH4+ (mol N m^-3), SI row 1 (as S_NH3/14)."""

    c_tot_no2: float
    """Total nitrite nitrogen NO2- + HNO2 (mol N m^-3), SI row 3."""

    c_tot_no3: float
    """Total nitrate nitrogen NO3- + HNO3 (mol N m^-3), SI row 5."""

    c_tot_ic: float
    """Total inorganic carbon CO2 + HCO3- + CO3^2- (mol C m^-3), SI row 7."""

    c_tot_po4: float
    """Total orthophosphate (all protonation states) (mol P m^-3), SI row 10."""


@dataclass(frozen=True, slots=True)
class AqueousSpeciesMolar:
    """Aqueous species concentrations (mol m^-3) from SI.6.1 rows 2, 4, 6, 8–9, 11–13, 14."""

    nh3: float
    nh4: float
    no2: float
    hno2: float
    no3: float
    hno3: float
    co2: float
    hco3: float
    co3: float
    h3po4: float
    h2po4: float
    hpo4: float
    po4: float
    oh: float


def speciate_ammonia(h_plus_mol_per_m3: float, ka_nh4_molar: float, c_tot_nh_mol_per_m3: float) -> tuple[float, float]:
    """
    SI.6.1 row 2; K_a in mol/L. Returns (NH3, NH4+) in mol m^-3.

    The SI table labels the quotient as NH3, but the same algebraic form is [NH4+] =
    C_T * h / (h + K_a * 10^3) in this unit convention; NH3 is the complement for mass balance.
    """
    denom = 1.0 + ka_nh4_molar * 1.0e3 / h_plus_mol_per_m3
    nh4 = c_tot_nh_mol_per_m3 / denom
    nh3 = c_tot_nh_mol_per_m3 - nh4
    return nh3, nh4


def speciate_nitrite(
    h_plus_mol_per_m3: float,
    ka_hno2_molar: float,
    c_tot_no2_mol_per_m3: float,
) -> tuple[float, float]:
    """SI.6.1 row 4. Returns (NO2-, HNO2) in mol m^-3."""
    denom = 1.0 + ka_hno2_molar * 1.0e3 / h_plus_mol_per_m3
    hno2 = c_tot_no2_mol_per_m3 / denom
    no2 = c_tot_no2_mol_per_m3 - hno2
    return no2, hno2


def speciate_nitrate(
    h_plus_mol_per_m3: float,
    ka_hno3_molar: float,
    c_tot_no3_mol_per_m3: float,
) -> tuple[float, float]:
    """SI.6.1 row 6. Returns (NO3-, HNO3) in mol m^-3."""
    denom = 1.0 + ka_hno3_molar * 1.0e3 / h_plus_mol_per_m3
    hno3 = c_tot_no3_mol_per_m3 / denom
    no3 = c_tot_no3_mol_per_m3 - hno3
    return no3, hno3


def speciate_carbonate(
    h_plus_mol_per_m3: float,
    ka_co2_molar: float,
    ka_hco3_molar: float,
    c_tot_ic_mol_per_m3: float,
) -> tuple[float, float, float]:
    """
    SI.6.1 rows 8–9; CO3^2- from row 7 closure. Returns (CO2, HCO3-, CO3^2-) in mol m^-3.
    """
    h = h_plus_mol_per_m3
    denom_co2 = 1.0 + ka_co2_molar * 1.0e3 / h + ka_co2_molar * ka_hco3_molar * 1.0e6 / (h * h)
    co2 = c_tot_ic_mol_per_m3 / denom_co2
    denom_hco3 = 1.0 + h / (ka_hco3_molar * 1.0e3) + ka_hco3_molar * 1.0e3 / h
    hco3 = c_tot_ic_mol_per_m3 / denom_hco3
    co3 = c_tot_ic_mol_per_m3 - co2 - hco3
    return co2, hco3, co3


def speciate_phosphate(
    h_plus_mol_per_m3: float,
    ka_h3po4_molar: float,
    ka_h2po4_molar: float,
    ka_hpo4_molar: float,
    c_tot_p_mol_per_m3: float,
) -> tuple[float, float, float, float]:
    """
    SI.6.1 rows 11–13; PO4^3- from row 10 closure.
    Returns (H3PO4, H2PO4-, HPO4^2-, PO4^3-) in mol m^-3.
    """
    h = h_plus_mol_per_m3
    denom_h3po4 = (
        1.0
        + ka_h3po4_molar * 1.0e3 / h
        + ka_h3po4_molar * ka_h2po4_molar * 1.0e6 / (h * h)
        + ka_h3po4_molar * ka_h2po4_molar * ka_hpo4_molar * 1.0e9 / (h * h * h)
    )
    h3po4 = c_tot_p_mol_per_m3 / denom_h3po4
    denom_h2po4 = (
        1.0
        + h / (ka_h2po4_molar * 1.0e3)
        + ka_h2po4_molar * 1.0e3 / h
        + ka_h2po4_molar * ka_hpo4_molar * 1.0e6 / (h * h)
    )
    h2po4 = c_tot_p_mol_per_m3 / denom_h2po4
    denom_hpo4 = (
        1.0
        + (h * h) / (ka_h2po4_molar * ka_hpo4_molar * 1.0e6)
        + h / (ka_hpo4_molar * 1.0e3)
        + ka_hpo4_molar * 1.0e3 / h
    )
    hpo4 = c_tot_p_mol_per_m3 / denom_hpo4
    po4 = c_tot_p_mol_per_m3 - h3po4 - h2po4 - hpo4
    return h3po4, h2po4, hpo4, po4


def speciate_water(h_plus_mol_per_m3: float, kw_molar: float) -> float:
    """SI.6.1 row 14. Returns OH- in mol m^-3."""
    return kw_molar * 1.0e3 / h_plus_mol_per_m3


def speciate_aqueous(
    h_plus_mol_per_m3: float,
    totals: SpeciationTotals,
    k: AlbaDissociationConstantsRef,
) -> AqueousSpeciesMolar:
    """
    Full aqueous speciation for SI.6.1 rows 2–14 (given [H+] and totals; no charge balance).
    """
    nh3, nh4 = speciate_ammonia(h_plus_mol_per_m3, k.ka_nh4, totals.c_tot_nh3)
    no2, hno2 = speciate_nitrite(h_plus_mol_per_m3, k.ka_hno2, totals.c_tot_no2)
    no3, hno3 = speciate_nitrate(h_plus_mol_per_m3, k.ka_hno3, totals.c_tot_no3)
    co2, hco3, co3 = speciate_carbonate(
        h_plus_mol_per_m3,
        k.ka_co2,
        k.ka_hco3,
        totals.c_tot_ic,
    )
    h3po4, h2po4, hpo4, po4 = speciate_phosphate(
        h_plus_mol_per_m3,
        k.ka_h3po4,
        k.ka_h2po4,
        k.ka_hpo4,
        totals.c_tot_po4,
    )
    oh = speciate_water(h_plus_mol_per_m3, k.kw)
    return AqueousSpeciesMolar(
        nh3=nh3,
        nh4=nh4,
        no2=no2,
        hno2=hno2,
        no3=no3,
        hno3=hno3,
        co2=co2,
        hco3=hco3,
        co3=co3,
        h3po4=h3po4,
        h2po4=h2po4,
        hpo4=hpo4,
        po4=po4,
        oh=oh,
    )


def speciate_from_alba_totals(
    h_plus_mol_per_m3: float,
    s_ic_g_per_m3: float,
    s_nh_g_per_m3: float,
    s_no2_g_per_m3: float,
    s_no3_g_per_m3: float,
    s_po4_g_per_m3: float,
    k: AlbaDissociationConstantsRef,
) -> AqueousSpeciesMolar:
    """
    Speciation from ALBA mass totals (g m^-3) via SI.6.1 /12, /14, /31 conversions.
    """
    totals = SpeciationTotals(
        c_tot_nh3=mol_n_per_m3_nitrogen_from_s_nh(s_nh_g_per_m3),
        c_tot_no2=mol_n_per_m3_nitrogen_from_s_no2(s_no2_g_per_m3),
        c_tot_no3=mol_n_per_m3_nitrogen_from_s_no3(s_no3_g_per_m3),
        c_tot_ic=mol_c_per_m3_carbon_from_s_ic(s_ic_g_per_m3),
        c_tot_po4=mol_p_per_m3_phosphorus_from_s_po4(s_po4_g_per_m3),
    )
    return speciate_aqueous(h_plus_mol_per_m3, totals, k)


# --- Stage 3: charge balance and pH solver (SI.6.1 row 15) ---


class PHSolveError(RuntimeError):
    """Raised when solve_pH cannot find a valid root in configured bounds."""


@dataclass(frozen=True, slots=True)
class ChargeBalanceInputs:
    """Inputs for SI.6.1 row 15 residual and pH solving."""

    totals: SpeciationTotals
    t_celsius: float
    delta_cat_an_mol_per_m3: float = 0.0
    k_ref: AlbaDissociationConstantsRef | None = None
    dh: AlbaDissociationEnthalpy | None = None


@dataclass(frozen=True, slots=True)
class PHSolverOptions:
    """Numerical settings for solve_pH in log10([H+]) space."""

    residual_tol: float = 1.0e-12
    x_step_tol: float = 1.0e-10
    max_iterations: int = 10
    ph_min: float = 0.0
    ph_max: float = 14.0
    derivative_dx: float = 1.0e-6
    fallback_iterations: int = 80


@dataclass(frozen=True, slots=True)
class PHSolveResult:
    """Result payload for solve_pH."""

    h_plus_mol_per_m3: float
    ph: float
    residual: float
    iterations: int
    converged: bool
    method_used: str


def ph_from_h_plus_mol_per_m3(h_plus_mol_per_m3: float) -> float:
    """Convert [H+] in mol m^-3 to pH on mol/L convention: pH = 3 - log10([H+]_m3)."""
    return 3.0 - math.log10(h_plus_mol_per_m3)


def h_plus_mol_per_m3_from_ph(ph: float) -> float:
    """Convert pH to [H+] in mol m^-3: [H+]_m3 = 10^(3 - pH)."""
    return 10.0 ** (3.0 - ph)


def charge_residual(h_plus_mol_per_m3: float, inputs: ChargeBalanceInputs) -> float:
    """
    SI.6.1 row 15 charge-balance residual in mol m^-3.

    Positive and negative ionic terms are assembled from Stage 2 speciation at current [H+].
    A root (residual = 0) corresponds to electroneutrality.
    """
    if h_plus_mol_per_m3 <= 0.0:
        raise ValueError("h_plus_mol_per_m3 must be > 0")
    if inputs.totals.c_tot_nh3 < 0.0:
        raise ValueError("total concentrations must be nonnegative")

    k_ref = inputs.k_ref or default_dissociation_constants_ref_molar()
    dh = inputs.dh or default_dissociation_enthalpy_j_per_mol()
    k_t = scale_dissociation_constants_at_t(k_ref, dh, inputs.t_celsius)
    sp = speciate_aqueous(h_plus_mol_per_m3, inputs.totals, k_t)
    return (
        h_plus_mol_per_m3
        + sp.nh4
        + inputs.delta_cat_an_mol_per_m3
        - sp.oh
        - sp.no2
        - sp.no3
        - sp.hco3
        - 2.0 * sp.co3
        - sp.h2po4
        - 2.0 * sp.hpo4
        - 3.0 * sp.po4
    )


def solve_pH(
    inputs: ChargeBalanceInputs,
    initial_ph: float = 7.0,
    options: PHSolverOptions | None = None,
) -> PHSolveResult:
    """
    Solve SI.6.1 row 15 for pH using Newton iterations in log10([H+]) with bounded fallback.
    """
    # 1) Load solver options (defaults if caller did not pass custom options)
    opt = options or PHSolverOptions()

    # 2) Validate configured pH bounds
    if not (opt.ph_min < opt.ph_max):
        raise ValueError("ph_min must be < ph_max")

    # 3) Work in x = log10([H+]_m3); convert pH bounds to x-bounds:
    #    pH = 3 - log10([H+]_m3)  =>  x = 3 - pH
    x_min = 3.0 - opt.ph_max
    x_max = 3.0 - opt.ph_min

    # 4) Initialize x from the user guess, clamped to the allowed bounds
    x = min(max(3.0 - initial_ph, x_min), x_max)

    # 5) Define residual in x-space: [H+] = 10^x, then evaluate charge balance
    def f_of_x(x_log10_h: float) -> float:
        return charge_residual(10.0**x_log10_h, inputs)

    # 6) Primary solver path: Newton iterations in log-space
    for i in range(1, opt.max_iterations + 1):
        fx = f_of_x(x)

        # 6a) Converged if residual is already below tolerance
        if abs(fx) <= opt.residual_tol:
            h_plus = 10.0**x
            return PHSolveResult(
                h_plus_mol_per_m3=h_plus,
                ph=ph_from_h_plus_mol_per_m3(h_plus),
                residual=fx,
                iterations=i,
                converged=True,
                method_used="newton_logh",
            )

        # 6b) Finite-difference derivative dF/dx around current x
        dx = opt.derivative_dx
        x_lo = max(x_min, x - dx)
        x_hi = min(x_max, x + dx)

        # If derivative window collapses, Newton cannot proceed
        if x_hi == x_lo:
            break

        dfdx = (f_of_x(x_hi) - f_of_x(x_lo)) / (x_hi - x_lo)

        # Guard against invalid derivative.
        if dfdx == 0.0 or not math.isfinite(dfdx):
            break

        # 6c) Newton update in x-space.
        step = fx / dfdx
        x_next = x - step

        # Keep Newton iterate inside physical pH bounds
        x_next = min(max(x_next, x_min), x_max)

        # 6d) If Newton step is tiny, accept only if residual is actually small
        if abs(x_next - x) <= opt.x_step_tol:
            residual_next = f_of_x(x_next)
            if abs(residual_next) <= opt.residual_tol:
                h_plus = 10.0**x_next
                return PHSolveResult(
                    h_plus_mol_per_m3=h_plus,
                    ph=ph_from_h_plus_mol_per_m3(h_plus),
                    residual=residual_next,
                    iterations=i,
                    converged=True,
                    method_used="newton_logh",
                )
            # Tiny step but not converged -> switch to bounded fallback
            break

        # 6e) Continue Newton from updated iterate
        x = x_next

    # 7) Fallback path: bracket check over full bounds
    f_min = f_of_x(x_min)
    f_max = f_of_x(x_max)

    # Exact root at lower/upper bound (rare but valid)
    if f_min == 0.0:
        h_plus = 10.0**x_min
        return PHSolveResult(h_plus, ph_from_h_plus_mol_per_m3(h_plus), 0.0, 0, True, "fallback_bracket")
    if f_max == 0.0:
        h_plus = 10.0**x_max
        return PHSolveResult(h_plus, ph_from_h_plus_mol_per_m3(h_plus), 0.0, 0, True, "fallback_bracket")

    # Without sign change in [x_min, x_max], bracket methods cannot guarantee a root
    if f_min * f_max > 0.0:
        raise PHSolveError("No sign change in pH bounds; cannot bracket charge residual root.")

    # 8) Bisection-like bounded solve inside the valid bracket
    left, right = x_min, x_max
    f_left = f_min
    for j in range(1, opt.fallback_iterations + 1):
        mid = 0.5 * (left + right)
        f_mid = f_of_x(mid)

        # Converged by residual or by interval width
        if abs(f_mid) <= opt.residual_tol or abs(right - left) <= opt.x_step_tol:
            h_plus = 10.0**mid
            return PHSolveResult(
                h_plus_mol_per_m3=h_plus,
                ph=ph_from_h_plus_mol_per_m3(h_plus),
                residual=f_mid,
                iterations=opt.max_iterations + j,
                converged=True,
                method_used="fallback_bracket",
            )

        # Keep the half-interval that preserves sign change
        if f_left * f_mid <= 0.0:
            right = mid
        else:
            left = mid
            f_left = f_mid

    # 9) If fallback ran out of iterations, raise explicit solver failure
    h_plus = 10.0 ** (0.5 * (left + right))
    residual = charge_residual(h_plus, inputs)
    raise PHSolveError(
        f"Fallback bracket did not converge (residual={residual:.3e}, "
        f"iterations={opt.max_iterations + opt.fallback_iterations}).",
    )
