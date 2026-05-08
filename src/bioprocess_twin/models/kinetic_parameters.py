"""
Default kinetic parameters for ALBA (Casagli et al., 2021).

Numeric values follow MATH_MODEL.md §1.2 (nominal / midpoint where SI.5 gives ±).
"""

from pydantic import BaseModel, ConfigDict, Field


class CardinalTemperature(BaseModel):
    """Cardinal temperatures for CTMI growth factor (§1.2.4), °C."""

    model_config = ConfigDict(frozen=True)

    t_min: float = Field(..., description="T_min [°C]")
    t_opt: float = Field(..., description="T_opt [°C]")
    t_max: float = Field(..., description="T_max [°C]")


class CardinalPH(BaseModel):
    """Cardinal pH for CPM growth factor (§1.2.5)."""

    model_config = ConfigDict(frozen=True)

    ph_min: float = Field(..., description="pH_min [-]")
    ph_opt: float = Field(..., description="pH_opt [-]")
    ph_max: float = Field(..., description="pH_max [-]")


class KineticParameters(BaseModel):
    """Kinetic and cardinal defaults for process rates (§1.2)."""

    model_config = ConfigDict(frozen=True)

    # --- §1.2.1 specific rates [d^-1] ---
    mu_max_alg: float = Field(..., description="μ_max,ALG")
    b_max_r_alg: float = Field(..., description="b_max,r,ALG aerobic respiration")
    b_max_d_alg: float = Field(..., description="b_max,d,ALG decay")
    mu_max_h: float = Field(..., description="μ_max,H")
    b_max_r_h: float = Field(..., description="b_max,r,H")
    b_max_d_h: float = Field(..., description="b_max,d,H")
    mu_hyd: float = Field(..., description="μ_Hyd hydrolysis X_S")
    mu_a: float = Field(..., description="μ_a urea / S_ND")
    mu_max_aob: float = Field(..., description="μ_max,AOB")
    b_max_r_aob: float = Field(..., description="b_max,r,AOB")
    b_max_d_aob: float = Field(..., description="b_max,d,AOB")
    mu_max_nob: float = Field(..., description="μ_max,NOB")
    b_max_r_nob: float = Field(..., description="b_max,r,NOB")
    b_max_d_nob: float = Field(..., description="b_max,d,NOB")

    # --- §1.2.2 half-saturation & light ---
    k_c_alg: float = Field(..., description="K_C,ALG [gC m^-3]")
    k_o_alg: float = Field(..., description="K_O,ALG [gO2 m^-3]")
    k_n_alg: float = Field(..., description="K_N,ALG [gN m^-3]")
    k_no3_alg: float = Field(..., description="K_NO3,ALG [gN m^-3]")
    k_p_alg: float = Field(..., description="K_P,ALG [gP m^-3]")
    ec50_o2: float = Field(..., description="EC_50,O2 Hill algae [gO2 m^-3]")
    hill_n_o2: float = Field(..., description="Hill exponent n for f_DO")
    i_opt: float = Field(..., description="I_opt Haldane [μmol m^-2 s^-1]")
    alpha_light: float = Field(..., description="α PI slope §3.3 [μmol^-1 m^2 s]")
    epsilon_light: float = Field(
        ...,
        description="ε light extinction [m^2 gCOD^-1]; Beer–Lambert, not in f_I scalar",
    )
    k_s_h: float = Field(..., description="K_S,H [gCOD m^-3]")
    k_o_h: float = Field(..., description="K_O,H [gO2 m^-3]")
    k_n_h: float = Field(..., description="K_N,H [gN m^-3]")
    k_no2_h: float = Field(..., description="K_NO2,H [gN m^-3]")
    k_no3_h: float = Field(..., description="K_NO3,H [gN m^-3]")
    k_p_h: float = Field(..., description="K_P,H [gP m^-3]")
    k_hyd: float = Field(..., description="K_HYD [-] ratio X_S/X_H")
    k_a: float = Field(..., description="K_a urea Monod S_ND [gN m^-3]")
    eta_anox: float = Field(..., description="η_ANOX [-]")
    k_c_aob: float = Field(..., description="K_C,AOB [gC m^-3]")
    k_o_aob: float = Field(..., description="K_O,AOB [gO2 m^-3]")
    k_n_aob: float = Field(..., description="K_N,AOB [gN m^-3]")
    k_p_aob: float = Field(..., description="K_P,AOB [gP m^-3]")
    k_c_nob: float = Field(..., description="K_C,NOB [gC m^-3]")
    k_o_nob: float = Field(..., description="K_O,NOB [gO2 m^-3]")
    k_no2_nob: float = Field(..., description="K_NO2,NOB [gN m^-3]")
    k_p_nob: float = Field(..., description="K_P,NOB [gP m^-3]")

    # --- §1.2.3 Arrhenius bases θ^(T-20) ---
    theta_kla: float = Field(..., description="θ gas transfer (SI.7); not used in ρ1–19 yet")
    theta_h: float = Field(..., description="θ_H decay X_H")
    theta_aob: float = Field(..., description="θ_AOB decay")
    theta_nob: float = Field(..., description="θ_NOB decay")
    theta_alg: float = Field(..., description="θ_ALG decay")
    theta_hyd: float = Field(..., description="θ_HYD hydrolysis")
    theta_amm: float = Field(..., description="θ_AMM ammonification")

    # --- Cardinals ---
    cardinal_temp_alg: CardinalTemperature
    cardinal_temp_h: CardinalTemperature
    cardinal_temp_aob: CardinalTemperature
    cardinal_temp_nob: CardinalTemperature
    cardinal_ph_alg: CardinalPH
    cardinal_ph_h: CardinalPH
    cardinal_ph_aob: CardinalPH
    cardinal_ph_nob: CardinalPH


def default_alba() -> KineticParameters:
    """
    Nominal parameters from MATH_MODEL.md §1.2 (central values; ignore ± in code).

    Cardinal T/pH use table centers only (e.g. −10, 20, 42 °C for algae).
    """
    return KineticParameters(
        mu_max_alg=2.5,
        b_max_r_alg=0.1,
        b_max_d_alg=0.03,
        mu_max_h=6.0,
        b_max_r_h=0.3,
        b_max_d_h=0.9,
        mu_hyd=3.0,
        mu_a=0.25,
        mu_max_aob=0.72,
        b_max_r_aob=0.05,
        b_max_d_aob=0.1,
        mu_max_nob=0.65,
        b_max_r_nob=0.03,
        b_max_d_nob=0.08,
        k_c_alg=0.004,
        k_o_alg=0.2,
        k_n_alg=0.1,
        k_no3_alg=0.3,
        k_p_alg=0.02,
        ec50_o2=20.0,
        hill_n_o2=15.0,
        i_opt=300.0,
        alpha_light=0.01,
        epsilon_light=0.067,
        k_s_h=4.0,
        k_o_h=0.2,
        k_n_h=0.05,
        k_no2_h=0.2,
        k_no3_h=0.5,
        k_p_h=0.01,
        k_hyd=1.0,
        k_a=1.0,
        eta_anox=0.6,
        k_c_aob=0.5,
        k_o_aob=0.8,
        k_n_aob=0.5,
        k_p_aob=0.01,
        k_c_nob=0.5,
        k_o_nob=2.2,
        k_no2_nob=0.5,
        k_p_nob=0.01,
        theta_kla=1.024,
        theta_h=1.07,
        theta_aob=1.1,
        theta_nob=1.04,
        theta_alg=1.04,
        theta_hyd=1.04,
        theta_amm=1.12,
        cardinal_temp_alg=CardinalTemperature(t_min=-10.0, t_opt=20.0, t_max=42.0),
        cardinal_temp_h=CardinalTemperature(t_min=-3.0, t_opt=25.0, t_max=42.0),
        cardinal_temp_aob=CardinalTemperature(t_min=-8.0, t_opt=24.5, t_max=40.0),
        cardinal_temp_nob=CardinalTemperature(t_min=-8.0, t_opt=20.0, t_max=38.5),
        cardinal_ph_alg=CardinalPH(ph_min=2.0, ph_opt=8.4, ph_max=12.0),
        cardinal_ph_h=CardinalPH(ph_min=2.0, ph_opt=7.0, ph_max=11.5),
        cardinal_ph_aob=CardinalPH(ph_min=5.8, ph_opt=8.1, ph_max=12.4),
        cardinal_ph_nob=CardinalPH(ph_min=5.0, ph_opt=7.9, ph_max=12.1),
    )
