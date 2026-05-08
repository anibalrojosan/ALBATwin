# ALBA Model Specification (Single Source of Truth)

This document serves as the technical specification for the ALBA (Algae-Bacteria) model implementation. It aggregates all parameters, equations, and logic required to build the Digital Twin.

**Source:** Casagli et al. (2021). *ALBA: A comprehensive growth model to optimize algae-bacteria wastewater treatment in raceway ponds*. Water Research, 190, 116734.

---

## Table of contents
- [1. Model Constants & Parameters](#1-model-constants--parameters)
  - [1.1 Stoichiometric Parameters](#11-stoichiometric-parameters)
  - [1.2 Kinetic & environmental parameters (SI.5)](#12-kinetic--environmental-parameters-si5)
    - [1.2.1 Specific rates](#121-specific-rates)
    - [1.2.2 Half-saturation, inhibition, light](#122-half-saturation-inhibition-light)
    - [1.2.3 Temperature coefficients](#123-temperature-coefficients)
    - [1.2.4 Cardinal temperatures (growth)](#124-cardinal-temperatures-growth)
    - [1.2.5 Cardinal pH (growth)](#125-cardinal-ph-growth)
    - [1.2.6 Gas–liquid exchange (atmosphere)](#126-gasliquid-exchange-atmosphere)
    - [1.2.7 pH sub-model reference (pK_a)](#127-ph-sub-model-reference-pka)
- [2. State Vector Definition](#2-state-vector-definition)
  - [2.1 Digital twin: state and Petersen layouts](#21-digital-twin-state-and-petersen-layouts)
- [3. Algebraic Sub-models](#3-algebraic-sub-models)
- [4. Process Rate Equations](#4-process-rate-equations)
- [5. Petersen matrix in this repository](#5-petersen-matrix-in-this-repository)
    - [5.1 Dimensions and ODE layout](#51-dimensions-and-ode-layout)
    - [5.2 Scope of `get_petersen_matrix()`](#52-scope-of-get_petersen_matrix)
    - [5.3 Where coefficients come from](#53-where-coefficients-come-from)
    - [5.4 Example: process 1 (ρ₁), phototrophic growth on ammonium](#54-example-process-1-ρ₁-phototrophic-growth-on-ammonium)
    - [5.5 “Oxygen vs yield”: when a short formula is enough](#55-oxygen-vs-yield-when-a-short-formula-is-enough)

## 1. Model Constants & Parameters

### 1.1 Stoichiometric Parameters
Values defining the elemental composition of biomass and substrates.

| Symbol | Value | Unit | Description |
| :--- | :--- | :--- | :--- |
| **Biomass Composition** | | | |
| $i_{C,BM,ALG}$ | 0.327 | $gC \cdot gCOD_{BM}^{-1}$ | Carbon fraction in Algae |
| $i_{N,BM,ALG}$ | 0.042 | $gN \cdot gCOD_{BM}^{-1}$ | Nitrogen fraction in Algae |
| $i_{P,BM,ALG}$ | 0.008 | $gP \cdot gCOD_{BM}^{-1}$ | Phosphorus fraction in Algae |
| $i_{O,BM,ALG}$ | 0.209 | $gO \cdot gCOD_{BM}^{-1}$ | Oxygen fraction in Algae |
| $i_{H,BM,ALG}$ | 0.050 | $gH \cdot gCOD_{BM}^{-1}$ | Hydrogen fraction in Algae |
| $i_{C,BM}$ | 0.36 | $gC \cdot gCOD_{BM}^{-1}$ | Carbon fraction in Bacteria (H, AOB, NOB) |
| $i_{N,BM}$ | 0.084 | $gN \cdot gCOD_{BM}^{-1}$ | Nitrogen fraction in Bacteria |
| $i_{P,BM}$ | 0.016 | $gP \cdot gCOD_{BM}^{-1}$ | Phosphorus fraction in Bacteria |
| $i_{O,BM}$ | 0.184 | $gO \cdot gCOD_{BM}^{-1}$ | Oxygen fraction in Bacteria |
| $i_{H,BM}$ | 0.043 | $gH \cdot gCOD_{BM}^{-1}$ | Hydrogen fraction in Bacteria |
| **Yields & Fractions** | | | |
| $Y_H$ | 0.63 | $gCOD_{BM} \cdot gCOD_{S}^{-1}$ | Heterotrophic yield (Aerobic) |
| $Y_{H,NO2}$ | 0.3 | $gCOD_{BM} \cdot gCOD_{S}^{-1}$ | Heterotrophic yield (Anoxic NO2) |
| $Y_{H,NO3}$ | 0.5 | $gCOD_{BM} \cdot gCOD_{S}^{-1}$ | Heterotrophic yield (Anoxic NO3) |
| $Y_{AOB}$ | 0.2 | $gCOD_{BM} \cdot gN^{-1}$ | AOB yield |
| $Y_{NOB}$ | 0.05 | $gCOD_{BM} \cdot gN^{-1}$ | NOB yield |
| $f_{Xi,ALG}$ | 0.062 | $gCOD_{Xi} \cdot gCOD_{BM}^{-1}$ | Inert fraction from Algae decay |
| $f_{XI}$ | 0.1 | $gCOD_{XI} \cdot gCOD_{BM}^{-1}$ | Inert fraction from Bacteria decay |
| $f_{SI}$ | 0.1 | $gCOD_{SI} \cdot gCOD_{BM}^{-1}$ | Soluble inert fraction from hydrolysis |

### 1.2 Kinetic & environmental parameters (SI.5)

Numeric **kinetic**, **cardinal** (temperature, pH), and **auxiliary** defaults for light extinction, gas–liquid transfer, and pH speciation are transcribed from **Table SI.5.1** and the companion tables in Casagli et al. (2021) supplementary **SI.5**. Symbols match §3–§4; where the paper uses different subscripts (e.g. `μ_max,g,ALG`), this document uses $\mu_{max,ALG}$; the SI symbol α for the initial slope of the PI curve is $\alpha_{light}$ here.

Uncertainty (±) is included when reported in SI.5. **Source** abbreviations follow the SI bibliography; “This study” refers to Casagli et al. (2021).

#### 1.2.1 Specific rates

| Symbol | Value | Unit | Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **Algae** | | | | |
| $\mu_{max,ALG}$ | 2.5 ± 0.055 | $d^{-1}$ | Maximum specific growth rate of $X_{ALG}$ | This study |
| $b_{max,r,ALG}$ | 0.1 | $d^{-1}$ | Specific respiration rate of $X_{ALG}$ | This study |
| $b_{max,d,ALG}$ | 0.03 | $d^{-1}$ | Specific decay rate of $X_{ALG}$ | Arashiro, 2017 |
| **Heterotrophs** | | | | |
| $\mu_{max,H}$ | 6 | $d^{-1}$ | Maximum specific growth rate of $X_H$ | Henze, 2000 |
| $b_{max,r,H}$ | 0.3 | $d^{-1}$ | Specific aerobic respiration rate of $X_H$ | Reichert, 2001 |
| $b_{max,d,H}$ | 0.9 | $d^{-1}$ | Specific decay rate of $X_H$ | This study |
| $\mu_{Hyd}$ | 3 | $d^{-1}$ | Hydrolysis rate of slowly biodegradable COD ($X_S$) | Arashiro, 2017 |
| $\mu_a$ | 0.25 | $d^{-1}$ | Hydrolysis / ammonification rate of urea ($S_{ND}$) | This study |
| **AOB** | | | | |
| $\mu_{max,AOB}$ | 0.72 ± 0.0005 | $d^{-1}$ | Maximum specific aerobic growth rate of $X_{AOB}$ | This study |
| $b_{max,r,AOB}$ | 0.05 | $d^{-1}$ | Specific aerobic respiration rate of $X_{AOB}$ | Arashiro, 2017 |
| $b_{max,d,AOB}$ | 0.1 | $d^{-1}$ | Specific decay rate of $X_{AOB}$ | Solimeno, 2017 |
| **NOB** | | | | |
| $\mu_{max,NOB}$ | 0.65 ± 0.023 | $d^{-1}$ | Maximum specific aerobic growth rate of $X_{NOB}$ | This study |
| $b_{max,r,NOB}$ | 0.03 | $d^{-1}$ | Specific aerobic respiration rate of $X_{NOB}$ | Reichert, 2001 |
| $b_{max,d,NOB}$ | 0.08 | $d^{-1}$ | Specific decay rate of $X_{NOB}$ | This study |

#### 1.2.2 Half-saturation, inhibition, light

| Symbol | Value | Unit | Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **Algae** | | | | |
| $K_{C,ALG}$ | 0.004 | $gC \cdot m^{-3}$ | Inorganic carbon half-saturation for $X_{ALG}$ | Solimeno, 2017 |
| $K_{O,ALG}$ | 0.2 | $gO_2 \cdot m^{-3}$ | Oxygen half-saturation for $X_{ALG}$ | Reichert, 2001 |
| $K_{N,ALG}$ | 0.1 | $gN \cdot m^{-3}$ | Ammoniacal N half-saturation for $X_{ALG}$ | Solimeno, 2017 |
| $K_{NO3,ALG}$ | 0.3 | $gN \cdot m^{-3}$ | Nitrate half-saturation for $X_{ALG}$ | Decostere, 2016 |
| $K_{P,ALG}$ | 0.02 | $gP \cdot m^{-3}$ | Phosphorus half-saturation for $X_{ALG}$ | Decostere, 2016 |
| $EC_{50,O2}$ | 20 | $gO_2 \cdot m^{-3}$ | Dissolved O₂ at 50 % algal growth reduction (Hill) | This study |
| $n$ | 15 | — | Hill shape parameter for O₂ inhibition / enhancement | This study |
| $I_{opt}$ | 300 ± 3.814 | $\mu mol \cdot m^{-2} \cdot s^{-1}$ | Optimal irradiance for $X_{ALG}$ | Martinez, 2018 |
| $\alpha_{light}$ | 0.01 ± 0.0003 | $\mu mol^{-1} \cdot m^2 \cdot s$ | Initial slope of irradiance response (§3.3) | This study |
| $\varepsilon$ | 0.067 | $m^2 \cdot gCOD^{-1}$ | Light extinction coefficient | This study |
| **Heterotrophs** | | | | |
| $K_{S,H}$ | 4 | $gCOD \cdot m^{-3}$ | Readily biodegradable substrate half-saturation | Jubany, 2007 |
| $K_{O,H}$ | 0.2 | $gO_2 \cdot m^{-3}$ | Oxygen half-saturation for $X_H$ | Henze, 2000 |
| $K_{N,H}$ | 0.05 | $gN \cdot m^{-3}$ | Ammonium half-saturation for $X_H$ | Henze, 2000 |
| $K_{NO2,H}$ | 0.2 | $gN \cdot m^{-3}$ | Nitrite half-saturation for anoxic $X_H$ | Reichert, 2001 |
| $K_{NO3,H}$ | 0.5 | $gN \cdot m^{-3}$ | Nitrate half-saturation for anoxic $X_H$ | Reichert, 2001 |
| $K_{P,H}$ | 0.01 | $gP \cdot m^{-3}$ | Phosphorus half-saturation for $X_H$ | Henze, 2000 |
| $K_{HYD}$ | 1 | $gCOD \cdot gCOD^{-1}$ | Half-saturation ratio for hydrolysis ($X_S/X_H$) | Reichert, 2001 |
| $K_a$ | 1.0 | $gN \cdot m^{-3}$ | Half-saturation for urea / $S_{ND}$ in $\rho_{12}$ (≈ 1 mg N L⁻¹) | Henze et al., 2000 (nominal; not in Casagli SI.5) |
| $\eta_{ANOX}$ | 0.6 | — | Anoxic growth reduction factor | De Kreuk, 2006 |
| **AOB** | | | | |
| $K_{C,AOB}$ | 0.5 | $gC \cdot m^{-3}$ | Inorganic carbon half-saturation for $X_{AOB}$ | Henze, 2000 |
| $K_{O,AOB}$ | 0.8 | $gO_2 \cdot m^{-3}$ | Oxygen half-saturation for $X_{AOB}$ | Henze, 2000 |
| $K_{N,AOB}$ | 0.5 | $gN \cdot m^{-3}$ | Ammonium half-saturation for $X_{AOB}$ | Reichert, 2001 |
| $K_{P,AOB}$ | 0.01 | $gP \cdot m^{-3}$ | Phosphorus half-saturation for $X_{AOB}$ | Henze, 2000 |
| **NOB** | | | | |
| $K_{C,NOB}$ | 0.5 | $gC \cdot m^{-3}$ | Inorganic carbon half-saturation for $X_{NOB}$ | Henze, 2000 |
| $K_{O,NOB}$ | 2.2 | $gO_2 \cdot m^{-3}$ | Oxygen half-saturation for $X_{NOB}$ | Wiesmann, 1994 |
| $K_{NO2,NOB}$ | 0.5 | $gN \cdot m^{-3}$ | Nitrite half-saturation for $X_{NOB}$ | Reichert, 2001 |
| $K_{P,NOB}$ | 0.01 | $gP \cdot m^{-3}$ | Phosphorus half-saturation for $X_{NOB}$ | Henze, 2000 |

**Simulator coupling — depth-averaged PAR (phase1-04e).** Kinetics consume a single scalar `irradiance_umol_m2_s` in `EnvConditions`. When `LiquidOdeRhsProblem.mixed_layer_depth_m` is set [m], the code replaces surface PAR $I_0(t)$ from forcing with the depth-averaged value $\bar{I} = I_0\,(1-e^{-\kappa h})/(\kappa h)$ over a well-mixed layer of thickness $h$, with $\kappa = \varepsilon\,X_{ALG}$ [m⁻¹] and $\varepsilon$ the table coefficient [m² gCOD⁻¹] (`epsilon_light`). For $\kappa \to 0$, $\bar{I} \to I_0$. SI 1.2 gives $\mathrm{TSS} \approx X_{ALG}/1.57$ [g m⁻³] as an algal suspension proxy; the implementation uses $\kappa = \varepsilon_{\mathrm{COD}} X_{ALG}$, equivalent to pairing $\varepsilon_{\mathrm{TSS}}$ with that TSS when $\varepsilon_{\mathrm{TSS}} = \varepsilon_{\mathrm{COD}}\cdot 1.57$.

#### 1.2.3 Temperature coefficients

Used in $\theta^{(T-20)}$-style corrections where §3–§4 apply (decay, hydrolysis, ammonification, gas-transfer scaling in SI.7). The first row ($\theta$ = 1.024) is the factor cited for $k_La$ temperature correction in SI.5 / SI.7.

| Symbol | Value | Unit | Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| $\theta$ | 1.024 | — | Base for $k_La$ (and related) temperature correction | Ginot et Hervé, 1994 |
| $\theta_H$ | 1.07 | — | Temperature coefficient for $X_H$ decay | Henze, 2000 |
| $\theta_{AOB}$ | 1.1 | — | Temperature coefficient for $X_{AOB}$ decay | Metcalf & Eddy |
| $\theta_{NOB}$ | 1.04 | — | Temperature coefficient for $X_{NOB}$ decay | Metcalf & Eddy |
| $\theta_{ALG}$ | 1.04 | — | Temperature coefficient for $X_{ALG}$ decay | Reichert, 2001 |
| $\theta_{HYD}$ | 1.04 ± 0.005 | — | Temperature coefficient for hydrolysis | This study |
| $\theta_{AMM}$ | 1.12 ± 0.002 | — | Temperature coefficient for ammonification | This study |

#### 1.2.4 Cardinal temperatures (growth)

Used in $f_T$ for growth (CTMI, §3.1). Uncertainties from SI.5 calibration.

| Group | $T_{min}$ | $T_{opt}$ | $T_{max}$ | Unit |
| :--- | :--- | :--- | :--- | :--- |
| $X_{ALG}$ | −10 ± 1.524 | 20 ± 0.148 | 42 ± 0.513 | °C |
| $X_{AOB}$ | −8 ± 0.741 | 24.5 ± 0.232 | 40 ± 0.817 | °C |
| $X_{NOB}$ | −8 ± 9.734 | 20 ± 0.940 | 38.5 ± 6.090 | °C |
| $X_H$ | −3 ± 0.335 | 25 ± 0.634 | 42 ± 1.919 | °C |

#### 1.2.5 Cardinal pH (growth)

Used in $f_{pH}$ (CPM, §3.2).

| Group | $pH_{min}$ | $pH_{opt}$ | $pH_{max}$ |
| :--- | :--- | :--- | :--- |
| $X_{ALG}$ | 2 ± 0.562 | 8.4 ± 0.066 | 12 ± 0.039 |
| $X_{AOB}$ | 5.8 ± 0.355 | 8.1 ± 0.078 | 12.4 ± 0.115 |
| $X_{NOB}$ | 5 ± 0.568 | 7.9 ± 0.320 | 12.1 ± 0.463 |
| $X_H$ | 2 ± 0.344 | 7 ± 0.066 | 11.5 ± 0.022 |

#### 1.2.6 Gas–liquid exchange (atmosphere)

Defaults for process rates $\rho_{20}$–$\rho_{22}$ (SI.7); not part of `get_petersen_matrix()` until the gas-transfer workstream lands. Henry’s laws: explicit formulas in [`SI.7 Gas-liquid mass transfer.md`](supporting_informations/SI.7%20Gas-liquid%20mass%20transfer.md) (SI.7.3–SI.7.5); SI.5 cites equation tags that refer to that section.

| Symbol | Value | Unit | Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| $k_La$ | 34 ± 0.1 | $d^{-1}$ | Volumetric mass-transfer coefficient (O₂ reference) | This study |
| $H_{O2}$ | see SI.7.3 | $gO_2 \cdot m^{-3} \cdot atm^{-1}$ | Henry constant for O₂ vs temperature | Sander, 2015 |
| $H_{CO2}$ | see SI.7.4 | $gC\text{-}CO_2 \cdot m^{-3} \cdot atm^{-1}$ | Henry constant for CO₂ vs temperature | Sander, 2015 |
| $H_{NH3}$ | see SI.7.5 | $gN\text{-}NH_3 \cdot m^{-3} \cdot atm^{-1}$ | Henry constant for NH₃ vs temperature | Sander, 2015 |
| $D_{O2}$ | 2.5×10⁻⁹ | $m^2 \cdot s^{-1}$ | Diffusivity of O₂ in water | Perry, 2007 |
| $D_{CO2}$ | 2.1×10⁻⁹ | $m^2 \cdot s^{-1}$ | Diffusivity of CO₂ in water | Perry, 2007 |
| $D_{NH3}$ | 2.4×10⁻⁹ | $m^2 \cdot s^{-1}$ | Diffusivity of NH₃ in water | Perry, 2007 |
| $p_{O2}$ | 0.21 | atm | Gas-phase partial pressure of O₂ | This study |
| $p_{CO2}$ | 0.0004 | atm | Gas-phase partial pressure of CO₂ | This study |
| $p_{NH3}$ | 1.5×10⁻⁶ | atm | Gas-phase partial pressure of NH₃ | This study |

#### 1.2.7 pH sub-model reference (pK_a)

Reference **pK_a** values at calibration conditions (SI.5); full speciation and **K_a(T)** are in [`SI.6 Explicit chemical equilibria….md`](supporting_informations/SI.6%20Explicit%20chemical%20equilibria%2C%20their%20dissociation%20constants%20with%20temperature%20dependence.md). Descriptions follow the acid–base pair (SI.5 table row labels in the PDF mix species names across rows).

| Symbol | Value | Unit | Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| $pK_{a,\mathrm{CO_2}}$ | 6.37 | — | CO₂(aq) / HCO₃⁻ (first dissociation) | Batstone, 2002 |
| $pK_{a,\mathrm{HCO_3}}$ | 10.33 | — | HCO₃⁻ / CO₃²⁻ | Batstone, 2002 |
| $pK_{a,\mathrm{NH_4}}$ | 9.25 | — | NH₄⁺ / NH₃ | Batstone, 2002 |
| $pK_{a,\mathrm{HNO_2}}$ | 3.35 | — | HNO₂ / NO₂⁻ | Batstone, 2002 |
| $pK_{a,\mathrm{HNO_3}}$ | −1.64 | — | HNO₃ / NO₃⁻ | Batstone, 2002 |
| $pK_{a,\mathrm{H_3PO_4}}$ | 2.14 | — | H₃PO₄ / H₂PO₄⁻ | Batstone, 2002 |
| $pK_{a,\mathrm{H_2PO_4}}$ | 7.21 | — | H₂PO₄⁻ / HPO₄²⁻ | Batstone, 2002 |
| $pK_{a,\mathrm{HPO_4}}$ | 12.67 | — | HPO₄²⁻ / PO₄³⁻ | Batstone, 2002 |

Calibration strategy and parameter uncertainty are documented in SI.8 and SI.9 of the supplementary material.

---

## 2. State Vector Definition
The model state vector $\mathbf{y}$ consists of 17 variables.

| Index | Symbol | Unit | Description |
| :--- | :--- | :--- | :--- |
| 1 | $X_{ALG}$ | $gCOD \cdot m^{-3}$ | Algal biomass |
| 2 | $X_{AOB}$ | $gCOD \cdot m^{-3}$ | Ammonia Oxidizing Bacteria |
| 3 | $X_{NOB}$ | $gCOD \cdot m^{-3}$ | Nitrite Oxidizing Bacteria |
| 4 | $X_H$ | $gCOD \cdot m^{-3}$ | Heterotrophic Bacteria |
| 5 | $X_S$ | $gCOD \cdot m^{-3}$ | Slowly biodegradable organic matter |
| 6 | $X_I$ | $gCOD \cdot m^{-3}$ | Inert particulate organic matter |
| 7 | $S_S$ | $gCOD \cdot m^{-3}$ | Readily biodegradable organic matter |
| 8 | $S_I$ | $gCOD \cdot m^{-3}$ | Inert soluble organic matter |
| 9 | $S_{IC}$ | $gC \cdot m^{-3}$ | Total Inorganic Carbon ($CO_2 + HCO_3^- + CO_3^{2-}$) |
| 10 | $S_{ND}$ | $gN \cdot m^{-3}$ | Soluble Organic Nitrogen (Urea) |
| 11 | $S_{NH}$ | $gN \cdot m^{-3}$ | Total Ammoniacal Nitrogen ($NH_3 + NH_4^+$) |
| 12 | $S_{NO2}$ | $gN \cdot m^{-3}$ | Total Nitrite Nitrogen ($HNO_2 + NO_2^-$) |
| 13 | $S_{NO3}$ | $gN \cdot m^{-3}$ | Total Nitrate Nitrogen ($HNO_3 + NO_3^-$) |
| 14 | $S_{N2}$ | $gN \cdot m^{-3}$ | Nitrogen Gas |
| 15 | $S_{PO4}$ | $gP \cdot m^{-3}$ | Total Inorganic Phosphorus |
| 16 | $S_{O2}$ | $gO_2 \cdot m^{-3}$ | Dissolved Oxygen |
| 17 | $S_{H2O}$ | $gH \cdot m^{-3}$ | Water (Balance term) |

### 2.1 Digital twin: state and Petersen layouts

The published ALBA state list above is the **17-component Casagli SI** basis. The reference implementation also supports **optional elemental closure** layers (see [ADR 007](adrs/007-elemental-mass-balance-oh-closure.md), [STOICHIOMETRY.md](STOICHIOMETRY.md), [mass_balances/README.md](mass_balances/README.md)):

| Layout | Code `closure_mode` / `StateVectorVariant` | Length of $\mathbf{C}$ | Shape of $\mathbf{S}$ |
| :--- | :--- | :---: | :--- |
| SI literal | `si` | 17 | 19 × 17 |
| O closed via $S_{\mathrm{H2O}}$ | `oxygen` | 17 | 19 × 17 |
| O + H (proton inventory) | `oxygen_and_protons` | 18 | 19 × 18 |

The **18th** component is $S_{H\_PROTON}$ (g H·m⁻³, free-proton pool), appended after $S_{\mathrm{H2O}}$ in the ODE array. Pair `get_petersen_matrix_for_simulation(closure_mode=...)` with `StateVector.to_array(variant=...)` so $\mathrm{d}\mathbf{C}/\mathrm{d}t$ and $\mathbf{S}$ share the same $n$. The process rate vector $\boldsymbol{\rho}$ remains **$\mathbb{R}^{19}$** in all layouts.

---

## 3. Algebraic Sub-models

For an extended, textbook-style treatment of **aqueous speciation**, **charge balance / alkalinity**, **temperature laws** (van’t Hoff, $\theta^{T-20}$), **gas–liquid transfer** (Henry, $k_La$), and how these layers fit the **DAE** strategy in code, see **[`HYDROCHEMISTRY.md`](HYDROCHEMISTRY.md)**. For the same material oriented toward **ODEs**, the **RHS** \(\mathbf{f}(\mathbf{C},t,\mathbf{u})\), the **Petersen assembly** \(\mathbf{S}^{\mathsf T}\boldsymbol{\rho}\), and **time integration**, see **[`SIMULATOR_MATH.md`](SIMULATOR_MATH.md)**. Those documents are pedagogical; this section and the SI excerpts remain the **normative** parameter and equation references.

### 3.1 Temperature Dependence ($f_T$)
*   **Growth:** Uses CTMI Model (Rosso et al., 1993).
    $$ f_T(T) = \frac{(T-T_{max})(T-T_{min})^2}{(T_{opt}-T_{min})[(T_{opt}-T_{min})(T-T_{opt}) - (T_{opt}-T_{max})(T_{opt}+T_{min}-2T)]} $$
*   **Decay:** Uses Arrhenius.
    $$ f_{T,decay}(T) = \theta^{(T-20)} $$

### 3.2 pH Dependence ($f_{pH}$)
Uses Cardinal pH Model (CPM).
$$ f_{pH}(pH) = \frac{(pH - pH_{min})(pH - pH_{max})}{(pH - pH_{min})(pH - pH_{max}) - (pH - pH_{opt})^2} $$

### 3.3 Light Dependence ($f_I$)
For Algae only. Uses Haldane model (Bernard & Rémond, 2012).
$$ f_I(I) = \frac{\mu_{max}}{1 + \frac{\mu_{max}}{\alpha} \left(\frac{I}{I_{opt}} - 1\right)^2} $$

**Light factor vs §4 (single $\mu_{max,ALG}$ scale).** The numerator contains $\mu_{max}$, so at optimal irradiance $f_I(I_{opt}) = \mu_{max}$ — this is **not** a dimensionless factor equal to $1$. Section §4 nevertheless writes $\rho_1$ and $\rho_2$ as $\mu_{max,ALG} \cdot f_I \cdot \cdots$. If one substitutes the **same** symbol $f_I$ from the equation above **without** adjustment, the leading specific-rate scale becomes $\mu_{max,ALG}^2$ at $I = I_{opt}$, i.e. $\mu_{max}$ is counted twice.

To keep **one** overall $\mu_{max,ALG}$ in the growth rate while preserving the Haldane shape in $I$, the reference implementation uses an equivalent **dimensionless** response (same algebra, factored form):

$$ f_{I,\mathrm{dim}}(I) = \frac{f_I(I)}{\mu_{max,ALG}} = \frac{1}{1 + \frac{\mu_{max,ALG}}{\alpha} \left(\frac{I}{I_{opt}} - 1\right)^2}, \qquad f_{I,\mathrm{dim}}(I_{opt}) = 1, $$

and evaluates $\rho_1$ and $\rho_2$ with **$\mu_{max,ALG} \, f_{I,\mathrm{dim}}(I)$** wherever §4 shows $\mu_{max,ALG} \, f_I$ and $f_I$ is identified with the §3.3 expression. The process rates $\boldsymbol{\rho}$ are unchanged in meaning; only the split between an overall maximum specific rate and the $I$-dependent multiplier is explicit. (In code: `f_i_haldane` returns $f_I$ as above; algae growth divides by `mu_max_alg` once.)

### 3.4 Oxygen Inhibition ($f_{DO}$)
For Algae.
*   **Growth Inhibition:** 
   $$f_{DO,g} = \frac{EC_{50,O2}^n}{S_{O2}^n + EC_{50,O2}^n}$$
*   **Decay Enhancement:** 
   $$f_{DO,d} = \frac{S_{O2}^n}{S_{O2}^n + EC_{50,O2}^n}$$

---

## 4. Process Rate Equations ($\rho$)
The system defines 19 biological processes using **Liebig's Law of the Minimum** for nutrient limitation. The expressions below depend only on concentrations and environmental factors; they do **not** change when switching SI vs oxygen vs proton **Petersen** closure modes (closure affects $\mathbf{S}$, not the definition of $\rho_1,\ldots,\rho_{19}$).

### Algae ($X_{ALG}$)
1.  **Growth on $NH_4$:**
    $$ \rho_1 = \mu_{max,ALG} \cdot f_I \cdot f_T \cdot f_{pH} \cdot f_{DO,g} \cdot \min\left(\frac{S_{IC}}{K_{C} + S_{IC}}, \frac{S_{NH}}{K_{N} + S_{NH}}, \frac{S_{PO4}}{K_{P} + S_{PO4}}\right) \cdot X_{ALG} $$
2.  **Growth on $NO_3$:**
    $$ \rho_2 = \mu_{max,ALG} \cdot f_I \cdot f_T \cdot f_{pH} \cdot f_{DO,g} \cdot \frac{K_{N}}{K_{N} + S_{NH}} \cdot \min\left(\frac{S_{IC}}{K_{C} + S_{IC}}, \frac{S_{NO3}}{K_{NO3} + S_{NO3}}, \frac{S_{PO4}}{K_{P} + S_{PO4}}\right) \cdot X_{ALG} $$
3.  **Respiration:**
    $$ \rho_3 = b_{max,r,ALG} \cdot f_T \cdot f_{pH} \cdot \frac{S_{O2}}{K_{O} + S_{O2}} \cdot X_{ALG} $$
4.  **Decay:**
    $$ \rho_4 = b_{max,d,ALG} \cdot \theta^{(T-20)} \cdot f_{pH} \cdot f_{DO,d} \cdot X_{ALG} $$

### Heterotrophs ($X_H$)
5.  **Aerobic Growth ($NH_4$):**
    $$ \rho_5 = \mu_{max,H} \cdot f_T \cdot f_{pH} \cdot \min\left(\frac{S_S}{K_S + S_S}, \frac{S_{O2}}{K_O + S_{O2}}, \frac{S_{NH}}{K_N + S_{NH}}, \frac{S_{PO4}}{K_P + S_{PO4}}\right) \cdot X_H $$
6.  **Aerobic Growth ($NO_3$):**
    $$ \rho_6 = \mu_{max,H} \cdot f_T \cdot f_{pH} \cdot \frac{K_N}{K_N + S_{NH}} \cdot \min\left(\frac{S_S}{K_S + S_S}, \frac{S_{O2}}{K_O + S_{O2}}, \frac{S_{NO3}}{K_{NO3} + S_{NO3}}, \frac{S_{PO4}}{K_P + S_{PO4}}\right) \cdot X_H $$
7.  **Aerobic Respiration:**
    $$ \rho_7 = b_{max,r,H} \cdot f_T \cdot f_{pH} \cdot \frac{S_{O2}}{K_O + S_{O2}} \cdot X_H $$
8.  **Anoxic Growth ($NO_2$):**
    $$ \rho_8 = \mu_{max,H} \cdot \eta_{ANOX} \cdot f_T \cdot f_{pH} \cdot \frac{K_O}{K_O + S_{O2}} \cdot \min\left(\frac{S_S}{K_S + S_S}, \frac{S_{NO2}}{K_{NO2} + S_{NO2}}, \frac{S_{PO4}}{K_P + S_{PO4}}\right) \cdot X_H $$
9.  **Anoxic Growth ($NO_3$):**
    $$ \rho_9 = \mu_{max,H} \cdot \eta_{ANOX} \cdot f_T \cdot f_{pH} \cdot \frac{K_O}{K_O + S_{O2}} \cdot \min\left(\frac{S_S}{K_S + S_S}, \frac{S_{NO3}}{K_{NO3} + S_{NO3}}, \frac{S_{PO4}}{K_P + S_{PO4}}\right) \cdot X_H $$
10. **Anoxic Respiration:**
    $$ \rho_{10} = b_{max,r,H} \cdot \eta_{ANOX} \cdot f_T \cdot f_{pH} \cdot \frac{K_O}{K_O + S_{O2}} \cdot \min\left(\frac{S_{NO2}}{K_{NO2} + S_{NO2}}, \frac{S_{NO3}}{K_{NO3} + S_{NO3}}\right) \cdot X_H $$

### Hydrolysis & Decay
11. **Hydrolysis ($X_S \to S_S$):**
    $$ \rho_{11} = \mu_{Hyd} \cdot \theta_{HYD}^{(T-20)} \cdot f_{pH} \cdot \frac{X_S/X_H}{K_{HYD} + (X_S/X_H)} \cdot X_H $$
12. **Ammonification (Urea $\to NH_4$):**
    $$ \rho_{12} = \mu_{a} \cdot \theta_{AMM}^{(T-20)} \cdot f_{pH} \cdot \frac{S_{ND}}{K_a + S_{ND}} \cdot X_H $$
13. **Decay ($X_H$):**
    $$ \rho_{13} = b_{max,d,H} \cdot \theta_H^{(T-20)} \cdot f_{pH} \cdot X_H $$

### Nitrifiers ($X_{AOB}, X_{NOB}$)
14. **AOB Growth:**
    $$ \rho_{14} = \mu_{max,AOB} \cdot f_T \cdot f_{pH} \cdot \min\left(\frac{S_{NH}}{K_N + S_{NH}}, \frac{S_{O2}}{K_O + S_{O2}}, \frac{S_{IC}}{K_C + S_{IC}}, \frac{S_{PO4}}{K_P + S_{PO4}}\right) \cdot X_{AOB} $$
15. **AOB Respiration:**
    $$ \rho_{15} = b_{max,r,AOB} \cdot f_T \cdot f_{pH} \cdot \frac{S_{O2}}{K_O + S_{O2}} \cdot X_{AOB} $$
16. **AOB Decay:**
    $$ \rho_{16} = b_{max,d,AOB} \cdot \theta_{AOB}^{(T-20)} \cdot f_{pH} \cdot X_{AOB} $$
17. **NOB Growth:**
    $$ \rho_{17} = \mu_{max,NOB} \cdot f_T \cdot f_{pH} \cdot \min\left(\frac{S_{NO2}}{K_{NO2} + S_{NO2}}, \frac{S_{O2}}{K_O + S_{O2}}, \frac{S_{IC}}{K_C + S_{IC}}, \frac{S_{PO4}}{K_P + S_{PO4}}\right) \cdot X_{NOB} $$
18. **NOB Respiration:**
    $$ \rho_{18} = b_{max,r,NOB} \cdot f_T \cdot f_{pH} \cdot \frac{S_{O2}}{K_O + S_{O2}} \cdot X_{NOB} $$
19. **NOB Decay:**
    $$ \rho_{19} = b_{max,d,NOB} \cdot \theta_{NOB}^{(T-20)} \cdot f_{pH} \cdot X_{NOB} $$

---

## 5. Petersen matrix in this repository

This section states **what** the implementation’s Petersen matrix is, **what is out of scope**, and how a **worked row** relates to the shorthand oxygen–yield formulas used in textbook ASM. Full numeric tables and closure policies are in [STOICHIOMETRY.md](STOICHIOMETRY.md) and in `get_petersen_matrix()` / `get_petersen_matrix_for_simulation()`.

### 5.1 Dimensions and ODE layout

The biological Petersen matrix **S** appears in

$$
\frac{\mathrm{d}\mathbf{C}}{\mathrm{d}t} = \mathbf{S}^\top \boldsymbol{\rho}.
$$

`get_petersen_matrix()` keeps the **19-row biological block** (rates ρ1..ρ19 in §4). Column count matches the state layout (§2.1): **17 columns** on the Casagli SI basis, or **18 columns** when the optional proton inventory **S_H_PROTON** is appended for elemental H auditing. Stage 6 liquid RHS extends this with SI rows 20-22 through `get_petersen_matrix_with_gas_transfer()` and a 22-entry process vector in the SI 17-state path.

### 5.2 Scope of `get_petersen_matrix()`

Some **full** ALBA diagrams add extra rows for **gas–liquid transfer or equilibrium** (e.g. dissolution of O₂, CO₂, NH₃). In code, those rows are now available through `get_gas_transfer_matrix()` and `get_petersen_matrix_with_gas_transfer()` for Stage 6 liquid RHS assembly. `get_petersen_matrix()` remains intentionally biological-only for backward compatibility and closure workflows.

### 5.3 Where coefficients come from

Each Petersen entry (row **i**, column **j**) follows the mass and COD conventions of Casagli et al. (2021), in particular **Table SI.3.3** (closed forms for many entries). The repo transcribes that table in code; **do not** infer a single universal formula for “the” oxygen column from a short mnemonic. Different processes use different normalizations (phototrophy vs heterotrophy, aerobic vs anoxic, etc.).

### 5.4 Example: process 1 (ρ₁), phototrophic growth on ammonium

Illustrative non-zero entries on species consumed or produced (signs and factors per SI.3.3; biomass fractions as in §1.1):

- **X_ALG:** +1 (production of algal biomass per unit process rate).
- **S_IC, S_NH, S_PO4:** consumption with stoichiometry $i_{C,BM,ALG}$, $i_{N,BM,ALG}$, and $i_{P,BM,ALG}$ (in code: **I_C_ALG**, **I_N_ALG**, **I_P_ALG**).
- **S_O2:** SI.3.3 gives a closed form from elemental composition and the model’s O₂/COD convention (implemented as `_alpha_alg_o2_growth_nh4()`):

$$
\alpha_{S_{O2}} = -i_{O,\mathrm{BM,ALG}} + \frac{32}{12}\, i_{C,\mathrm{BM,ALG}} - \frac{24}{14}\, i_{N,\mathrm{BM,ALG}} + \frac{40}{31}\, i_{P,\mathrm{BM,ALG}} + 8\, i_{H,\mathrm{BM,ALG}}.
$$

- **S_H2O:** fixed SI coefficient on the water balance column for this row (see `ALPHA_ALG_H2O_RHO1` in code).

### 5.5 “Oxygen vs yield”: when a short formula is enough

In **introductory ASM-style** aerobic heterotrophic growth, one often writes oxygen demand per **unit biomass COD formed** in terms of yield **Y** on a COD basis. That is a **different** normalization than phototrophic ρ₁ above. In **this** codebase, **aerobic growth of X_H on NH₄** (process 5, ρ₅) uses the familiar identity

$$
\alpha_{S_{O2}} = -\left(\frac{1}{Y_H} - 1\right),
$$

i.e. row 5, column **S_O2**, is **−**(1/**Y_H** − 1), which links oxygen consumption to heterotrophic yield for that row only. Other processes (algae, nitrifiers, anoxic routes) use their **own** SI.3.3 expressions. In particular, there is **no** universal shortcut such as

$$
\alpha_{O2} = 1 - Y_{\mathrm{biomass}}
$$

that reproduces every oxygen stoichiometry in ALBA; phototrophic and other rows need the full SI.3.3 construction (§5.4).

> For the complete Petersen matrix of the ALBA model, see [STOICHIOMETRY.md](STOICHIOMETRY.md).