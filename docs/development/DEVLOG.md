# DEVLOG

This document is a log of the development process of the project. It is used to track the progress of the project and to document the decisions made during the development process and the learned concepts.

## Index

- [2026-05-05 - Sprint phase1-04c (A–C): CSTR transport, Stage 6 RHS, schedule-linked dilution](#devlog-20260505-p104c-abc)
- [2026-05-02 - Sprint phase1-04c: Startup batch integration (SI 17 + evaporation / volume ODE)](#devlog-20260502-p104c-startup-batch)
- [2026-04-30 - Sprint phase1-04b: ODE RHS wrapper (`evaluate_liquid_ode_rhs`, no transport)](#devlog-20260430-p104b-ode-rhs)
- [2026-04-25 - Sprint phase1-04a prep: Fig. 1 daily forcing (T, PAR, evaporation)](#devlog-20260425-fig1-daily-forcing)
- [2026-04-25 - ALBA Casagli et al. (2021) Article Review](#devlog-20260425-p103-alba-paper-reactor)
- [2026-04-23 - Sprint phase1-03.5: Stage 6 liquid RHS, 22×17 assembly, and 'evaluate_liquid_rhs'](#devlog-20260423-p1035-stage6-liquid-rhs)
- [2026-04-21 - Sprint phase1-03: SIMULATOR_MATH: ODEs, RHS, 22-row Petersen (SI.3.1), nested pH](#devlog-20260421-simulator-math)
- [2026-04-20 - Sprint phase1-03: Stages 2-5: speciation, T_ref 298.15 K, pH solver, SI.7 gas transfer, simulator API](#devlog-20260420-p103-stages-2-5)
- [2026-04-19 - Sprint phase1-03: Stage 1 and Hydrochemistry gas–liquid driving forces](#devlog-20260419-p103-chemistry-stage1)
- [2026-04-17 - Sprint phase1-03: Hydrochemistry documentation and implementation plan](#devlog-20260417-p103-hydrochemistry-doc)
- [2026-04-17 - Sprint 2B: Nineteen process rates, test policy, and CI mass-balance audit](#devlog-20260417-sprint2b-rates-ci)
- [2026-04-16 - Sprint 2B: Kinetics API stub, SSOT parameters, and algebraic modifiers](#devlog-20260416-sprint2b-kinetics-stub)
- [2026-04-15 - Sprint 2.5A: Stoichiometric O and H closure layers and extended StateVector](#devlog-20260415-sprint25a-oh-closure)
- [2026-03-31 - Sprint 2: DeepResearch and Explicit O/H Balance (rho14, ALBA vs ASM)](#devlog-20260331-sprint2-deepresearch-oh)
- [2026-03-30 - Sprint 2: Mass Balance Cell Diagnostics and SI Alignment Checkpoint](#devlog-20260330-sprint2-mass-balance-diag)
- [2026-03-19 - Sprint 2: Petersen Matrix and Stoichiometry (Mass Balance Verification)](#devlog-20260319-sprint2-petersen)
- [2026-03-17 - Sprint 1: Implementation of State Vector](#devlog-20260317-sprint1-state-vector)
- [2026-03-16 - Sprint 1: Implementation of Reactor Configuration](#devlog-20260316-sprint1-reactor-config)
- [2026-03-13 - Sprint 0: CI/CD and Infrastructure Setup](#devlog-20260313-sprint0-cicd)
- [2026-03-12 - Phase 1: Technical Specification & Architecture Definition](#devlog-20260312-phase1-spec-arch)
- [2026-03-12 - Phase 1: ALBA Model Analysis & Data Digitization](#devlog-20260312-phase1-alba-digitization)
- [2026-03-10 - Phase 0: Project Initialization and Foundation](#devlog-20260310-phase0-init)

---

<a id="devlog-20260505-p104c-abc"></a>

## [2026-05-05] - Sprint phase1-04c (A–C): CSTR transport, Stage 6 RHS, schedule-linked dilution

### Context & Goals

Deliver the lumped **CSTR dilution** term for the **17-component SI** liquid state so it is **additive** to the Stage 6 kinetic RHS in **g m⁻³ d⁻¹**, split across **`phase1-04c-A`** (pure math), **`phase1-04c-B`** (ODE wiring), and **`phase1-04c-C`** (time-varying volumetric flow and optional influent composition on the **same daily clock** as `DielForcingSchedule` / `ForcingSample`). Deliberately **out of scope**: a single HRAP driver that couples **`startup_batch`** open-volume balance with continuous-flow dilution.

### Technical Implementation

- **`phase1-04c-A`:** Added `src/bioprocess_twin/simulator/cstr_transport.py` — NumPy-only helpers: dilution $(Q/V)(\mathbf{y}_\mathrm{in}-\mathbf{y})$, HRT $\tau=V/Q$, and **m³ h⁻¹ → m³ d⁻¹** via $\times 24$ (aligned with hour/day scaling elsewhere). Module narrative ties to `docs/theory/cstr_mass_balance_and_hrap_lumped_model.md`. Added `tests/unit/test_cstr_transport.py` with **no** import of `evaluate_liquid_rhs`.
- **`phase1-04c-B`:** Introduced `CstrContinuousConfig` (single bundle for $V$, **m³ d⁻¹** $Q$, and SI $\mathbf{y}_\mathrm{in}$ / `StateVector`) and optional `LiquidOdeRhsProblem.cstr` defaulting to **`None`** so behaviour matches **`phase1-04b`** when unset. `evaluate_liquid_ode_rhs` adds `cstr_dilution_rate_g_m3_d` after `evaluate_liquid_rhs`; `make_liquid_rhs` inherits this path unchanged.
- **`phase1-04c-C`:** Added `CstrScheduleFlowConfig` and alias `CstrTransportConfig = CstrContinuousConfig | CstrScheduleFlowConfig`. Flow rate follows **`ForcingSample.inflow_m3_h`** from the **same** `schedule.at(t_hours)` already used for kinetics (**one sample per RHS evaluation** for **T**, **PAR**, and **Q**). If `inflow_m3_h` is **`None`**, transport uses $Q=0$ (same semantics as constant config with zero flow). Optional **callable** influent $f(t_\mathrm{wrapped})$ with $t_\mathrm{wrapped}\in[0,24)$ for explicit **daily periodicity** under multi-day integration in absolute hours; CSV ingestion deferred.
- **Design choices:** Use **m³ d⁻¹** for $Q$ and **m³** for $V$ so $Q/V$ is **d⁻¹**, matching `dcdt_g_m3_d` without mixed units. Variable $Q(t)$ reuses `DielForcingSchedule` optional drivers (constant or callable) instead of duplicating parallel series.
- **Files touched:** 
  - `src/bioprocess_twin/simulator/liquid_ode_rhs.py` (`CstrContinuousConfig`, `CstrScheduleFlowConfig`, `q_m3_per_d_from_forcing_sample`, `effective_y_in_schedule`, branching in `evaluate_liquid_ode_rhs`); 
  - `src/bioprocess_twin/simulator/__init__.py` (public exports); 
  - `tests/unit/test_liquid_ode_rhs.py` and `tests/unit/test_cstr_transport.py` for parity vs constant $Q$, $Q=0$, schedule vs continuous flow, callable influent decomposition, HRT checks.

### 💡 Deep Dive: One forcing sample, two budgets

Kinetics and dilution both consume **`DielForcingSchedule.at(t_hours)`**. Reusing that **`ForcingSample`** for `inflow_m3_h` guarantees **T**, **irradiance**, and **Q** stay **consistent at the same wrapped clock time**—important when `inflow_m3_h` is a callable of hour-of-day. The dilution term remains **$(Q/V)(\mathbf{y}_\mathrm{in}-\mathbf{y})$** in **g m⁻³ d⁻¹** per component conventions; only the **scalar** $Q$ (and optionally $\mathbf{y}_\mathrm{in}$) varies with the schedule.

### Next Steps

- **`phase1-04d`:** stiff-friendly time integrator driver and documented tolerances.
- Optional later: unify **open-volume** `startup_batch` with **continuous-flow** dilution in one simulation product path if required.

---

<a id="devlog-20260502-p104c-startup-batch"></a>

## [2026-05-02] - Sprint phase1-04c: Startup batch integration (SI 17 + evaporation / volume ODE)

### Context & Goals
Provide a **batch startup** path for the open HRABP-style pond: **variable volume** from **rain minus evaporation** (Fig. 1 / `ForcingSample`) and the lumped concentration correction $-(C_i/V)dV/dt$ with **no CSTR inflow** yet (phase1-04c). Scope is **SI layout only (17 states)**; **proton closure / 18th component** is deferred.

### Technical Implementation
- **`src/bioprocess_twin/simulator/startup_batch.py`:** `StartupBatchProblem` / `StartupBatchResult`, `evaluate_startup_batch_rhs`, `run_startup_batch` (`solve_ivp` LSODA, hours as time; ALBA rates converted **g m⁻³ d⁻¹ → g m⁻³ h⁻¹** via `/24`); nonnegative **clip** on `y` before `evaluate_liquid_ode_rhs` to satisfy `StateVector` bounds under numerical integration; optional `evaporation_floor_m3_h`; terminal event if `V` hits `volume_minimum_m3`.
- **`default_reasonable_startup_y0`:** nominal rich-substrate + inoculum vector (not Casagli-calibrated).
- **`tests/unit/test_startup_batch.py`:** parity when dV/dt=0, concentration-term scaling, volume decrease under evaporation, rain mm→m³ helper.

### Next Steps
- **phase1-04c:** add continuous-flow dilution; optionally unify time-base conventions with a shared integrator module (phase1-04d).

---

<a id="devlog-20260430-p104b-ode-rhs"></a>

## [2026-04-30] - Sprint phase1-04b: ODE RHS wrapper (`evaluate_liquid_ode_rhs`, no transport)

### Context & Goals
Deliver **Sprint 4.2 / issue `phase1-04b`**: a callable **$\mathbf{f}(\mathbf{C}, t)$** in SciPy shape **`rhs(t, y)`** that pins the liquid ODE to **17** states via **`evaluate_liquid_rhs`**, with **temperature and irradiance** coming from **`DielForcingSchedule`** (phase1-04a). **No** CSTR dilution term yet (phase1-04c).

### Technical Implementation
- **`src/bioprocess_twin/simulator/liquid_ode_rhs.py`:** frozen **`LiquidOdeRhsProblem`** (schedule, `initial_ph` / `placeholder_ph_for_env` default 7, passthrough kwargs for gas and pH solver); **`evaluate_liquid_ode_rhs(t_hours, y, problem=...)`** and **`make_liquid_rhs`** for `solve_ivp`.
- **`src/bioprocess_twin/simulator/liquid_rhs.py`:** public **`state_vector_from_y`** (SI-length 17 `ndarray` to **`StateVector`**).
- **`tests/unit/test_liquid_ode_rhs.py`:** numeric parity vs direct **`evaluate_liquid_rhs`**, shape/finiteness, closure factory, day vs night forcing sanity, bad length error.
- **`docs/SIMULATOR_MATH.md`:** Appendix C map row + F.1 pointer to **`liquid_ode_rhs.py`**.

### Next Steps
- **phase1-04c:** add **$(Q/V)(\mathbf{C}_\mathrm{in}-\mathbf{C})$** on top of this RHS; **phase1-04d:** stiff integrator driver.

---

<a id="devlog-20260425-fig1-daily-forcing"></a>

## [2026-04-25] - Sprint phase1-04a prep: Fig. 1 daily forcing for simulator environmental drivers

### Context & Goals
Prepare **tabular and smooth diel series** consistent with **Casagli et al. (2021) Fig. 1** (typical daily patterns of **temperature**, **PAR / irradiance**, and **evaporation rate**) so they can feed **`EnvConditions`** and related hydrology inputs in **Sprint 4 / issue `phase1-04a`** (diel $T(t)$, surface irradiance, optional rain/evaporation schedules). The intent is to move from **paper graphics** to **versioned, reproducible forcing data** rather than ad hoc constants, supporting both **visual parity checks** and **simulator initial/boundary-style schedules** on a 24 h grid.

### Technical Implementation
- **Python module:** `src/bioprocess_twin/forcing/typical_daily_forcing_per_season.py` encodes **seasonal control points** (visual extraction from Fig. 1), evaluates **monotone** diel curves with **`PchipInterpolator`**, clips non-negative quantities, and sets **irradiance to zero for $t \geq 20\,\mathrm{h}$** (night) with **20 h knots at zero** to avoid non-physical overshoot between sparse night samples. Default CLI export:  
  `uv run python -m bioprocess_twin.forcing.typical_daily_forcing_per_season`
- **Exported CSV (hourly, four seasons):** `data/forcing/typical_daily_patterns_of_temperature_irradiance_and_evaporation_rates.csv`: columns aligned with °C, µmol m⁻² s⁻¹ (PAR), and m³ h⁻¹ evaporation as stated in the module docstring.
- **Notebook:** `notebooks/typical_daily_forcing_fig1.ipynb` rebuilds the three panels, compares smooth curves to control points, and saves PNGs under `figures/` at the repo root.
- **Schedule API (`phase1-04a`):** `src/bioprocess_twin/forcing/diel_forcing_schedule.py` exposes `DielForcingSchedule` with `at` / `at_many` for dense grids, optional **RH, wind, rain, evaporation override,** and **inflow Q** as `None`, constants, or callables; `to_env_conditions(sample, ph=...)` maps kinetic inputs into **`EnvConditions`** (pH remains caller-supplied). Package exports live in `src/bioprocess_twin/forcing/__init__.py`.

### Next Steps
- **phase1-04b** (done 2026-04-30): ODE RHS wrapper **`evaluate_liquid_ode_rhs`** / **`LiquidOdeRhsProblem`** — see dedicated DEVLOG entry above.
- **phase1-04c:** add **$(Q/V)(\mathbf{C}_\mathrm{in}-\mathbf{C})$** and hydraulics; optional hydrology fields on **`ForcingSample`** remain for that milestone.

---

<a id="devlog-20260425-p103-alba-paper-reactor"></a>

## [2026-04-25] - ALBA Casagli et al. (2021) Article Review

### Context & Goals
Re-read the **Casagli et al. (2021)** ALBA *Water Research* narrative (main text and SI cross-references) to align the **Python twin** not only with **tables and matrices** already encoded, but with **operating assumptions** the authors treat as normative: how the **raceway** is represented hydraulically, how **meteorology and water balance** enter, and which **observables** are meant to **constrain** the coupled biology–chemistry–gas system before finer state variables are trusted.

### Technical Implementation
- **Reactor idealization:** the modeled volume is a **fully mixed compartment** (AQUASIM-style CSTR); the paper cites **experimental validation of mixing** for **this** pilot (Hreiz et al., 2014), so the lumped control volume is an **evidence-backed** choice, not only a default template.
- **Forcing and hydrology:** **temperature, wind, and relative humidity** come from a **regional** weather station (~30 km), while **rain** is **local** (in situ); **evaporation** follows **Buckingham** (Béchet et al., 2011). Seasonal scenarios stress that **evaporation plus rain** materially change **hydraulic dilution**, concentrations, and (downstream) **light attenuation**—a coupling many prior lagoon models omit.
- **Numerical path in the paper:** the reference implementation is a **DAE** solved with **DASSL** inside AQUASIM. The stack as of now targets explicit **RHS** assembly (`evaluate_liquid_rhs`) plus a future **time integrator**—same reduced dynamics for the 17 liquid states if pH is solved inside each RHS call (BDF/LSODA family), not necessarily the same solver API as AQUASIM.
- **Kinetic bookkeeping:** **Liebig (minimum)** applies only to **limiting substrates** (nutrients, bioavailable inorganic carbon, etc.); **light**, **temperature**, and **pH** modifiers sit **outside** that minimum as a **product** with growth and inhibition terms. **Inorganic carbon** in Monod form uses **CO₂ + HCO₃⁻** as the relevant **S_IC** pool for uptake kinetics (not CO₃²⁻ in that term); fractions come from the **pH–speciation** block.
- **Biological scope choices** (for parity when we interpret fits): **autotrophic algae only** (no explicit mixotrophy), **predation folded into decay**, **hydrolysis** carried by **heterotrophs**, **two-step nitrification** (AOB/NOB) to allow **nitrite buildup**, and **algal respiration vs decay** split so **O₂** is tied to **respiration** while **decay** follows the paper’s RWQM1/BioAlgae2-style convention.
- **Chemical / charge layer:** **Δ_CAT_AN** tracks **non-explicit ions** in the charge balance; **no biological process** consumes or produces it—its dynamics are driven by **influent alkalinity / ions**, which matters when we wire **inflows** and compare to **pH** trajectories.
- **Gas–liquid:** **O₂, CO₂, and NH₃** exchange with the atmosphere (Fick-type formulation, **Henry** and **k_L a(T)** in SI); this is part of the same **closure** that makes **DO** and **pH** mutually informative.
- **Calibration policy (paper):** **online pH and dissolved oxygen** are treated as **primary** identification targets because they **tightly couple** speciation, growth, respiration, nitrification, and stripping, and because they are **high-frequency** probes compared with mostly **daily** grab-style variables; **NH₄⁺, NO₂⁻, NO₃⁻, COD, OD, TSS**, etc., refine the fit **after** that anchor. A large parameter count (~72 in the text) makes this **identifiability-first** ordering practically necessary.
- **Observation operators for validation:** **TSS**, **soluble COD**, and **algal biomass via optical density** are defined as **aggregates** of model states with **fixed conversion factors** from the paper/SI; any **plot-to-code** comparison must apply the same **lumped observables**, not raw `X_*` alone.

### 💡 Deep Dive: why pH and DO online sit “above” the rest
**pH** sets which **N** and **C** species actually participate in Monod terms and gas driving forces; **DO** closes the **oxygen balance** between **photosynthesis, heterotrophic and autotrophic respiration, nitrification, and gas transfer**. Together they **propagate** into almost every rate law and **algebraic** constraint, so matching them first **reduces non-unique** parameter sets that could otherwise agree on slower or sparser signals.

### Next Steps
- When implementing **Sprint 4** drivers (inflow, **HRT**, **diel** **T**/**light**, **evaporation**/**rain**), treat **hydraulic dilution** and **meteorology split** (regional vs local rain) as **first-class inputs**, not post-hoc scaling.
- Cross-check **influent** handling for **alkalinity / Δ_CAT_AN** consistency before trusting long **pH** runs.

**(optional)**

- **Solver surface (DAE vs reduced ODE):** keep the default plan—**stiff ODE integrator** (`solve_ivp` with **BDF** / **LSODA**) on `dC/dt = f(C, t)` with **pH nested inside** `evaluate_liquid_rhs`—mathematically aligned with index-1 reduction of the paper’s DAE. Revisit **explicit DAE** solvers (**IDA** / SUNDIALS, or legacy **DASSL** bindings) only if numerical evidence demands it (step limits, initialization pain, or a reformulation that exposes algebraics in the solver state vector).
- **Calibration / identification:** treat **forward simulation** (RHS + integrator + versioned **parameter vector**) as the **required** path for Sprint 4; defer **χ² minimization, Fisher bands, and automated simplex/secant loops** to a later workstream unless we need quantitative parity with Casagli figures or plant-specific fitting. When that phase starts, mirror the paper’s **identifiability-first** ordering: dominant weight on **online pH and DO**, then offline variables with the same **observation operators** ($TSS$, $COD_s$, $OD \rightarrow$ biomass).

---

<a id="devlog-20260423-p1035-stage6-liquid-rhs"></a>

## [2026-04-23] - Sprint phase1-03.5: Stage 6 liquid RHS, 22×17 assembly, and `evaluate_liquid_rhs`

### Context & Goals
Deliver **phase1-03.5 / Stage 6** as a single **liquid-phase RHS** that matches the **full SI.3.1** picture (19 biological processes plus **Equilibrium** rows 20–22) while keeping the existing **19×17** `get_petersen_matrix()` surface stable. The goal is a callable **orchestrator** that resolves **pH (SI.6)**, **gas–liquid transfer (SI.7)**, and **biological kinetics** in one pass, and returns a structured payload for diagnostics and later **time integrators (Sprint 4)**. Work landed on branch **`feat/hydrochemistry-ode-rhs`**.

### Technical Implementation
- **`src/bioprocess_twin/models/stoichiometry.py`:** add **`N_GAS_TRANSFER_PROCESSES`**, **`N_PROCESSES_WITH_GAS_TRANSFER`**, **`get_gas_transfer_matrix()`** (3×17, one-hots for **O₂, S_IC, S_NH** per Table SI.3.1), and **`get_petersen_matrix_with_gas_transfer()`** (22×17); preserve **`get_petersen_matrix()`** as the **19** biological rows only.
- **`src/bioprocess_twin/simulator/liquid_rhs.py`:** new **`evaluate_liquid_rhs`**, which calls **`hydrochemistry_step`**, then **`EnvConditions` with `pH` aligned to solved pH** for **`calculate_rates`**, stacks **ρ_bio (19)** and **ρ_gas (3)**, and forms **dC/dt = S_fullᵀ ρ_full**; return type **`AlbaLiquidRhsResult`** with **`dcdt_g_m3_d`**, rate vectors, **`PHSolveResult`**, **`GasTransferRates`**, and **`LiquidRhsDiagnostics`** (biomass snapshot, T/I/pH, **`AqueousSpeciesMolar`** for reporting).
- **`src/bioprocess_twin/simulator/__init__.py`:** re-exports the Stage 6 public types and **`evaluate_liquid_rhs`**.
- **`tests/unit/test_liquid_rhs_stage6.py`:** cover **S_full** shape, **bio + gas sparse** equivalence to the full 22-block product, **pH alignment** for kinetics vs a misleading input **pH**, O₂ **undersaturation vs supersaturation** sign, and a **pinned pH** regression.
- **Docs (Stage 6 SSOT):** update **`STOICHIOMETRY.md`**, **`MATH_MODEL.md` §5**, and **`SIMULATOR_MATH.md` Appendix C**; extend **Appendix F** with **F.1**: a **Mermaid** diagram of the **implementation** path in **`liquid_rhs.py`** (one RHS evaluation, not a time loop).

### 💡 Deep Dive: one RHS call versus a 6-month horizon
**`evaluate_liquid_rhs` implements the vector field** (rate of change of state) at a **single** tuple **(StateVector, `EnvConditions`, `GasTransferConditions` bundle)**. It is the building block
$$
\dot{\mathbf{C}} = f(\mathbf{C}, \mathbf{u})
$$
repeated in spirit at each **internal** time in an ODE stepper. A **6-month** run with **hourly** output requires an **orchestrated** loop (or `solve_ivp` / LSODA) in **Sprint 4** that: advances **t**, applies **T(t)**, **light(t)**, inflow/HRT, and **records** **C** and **diagnostics** on a schedule. Stage 6 **does not** own that engine yet; it only makes **f** well-defined and SI-consistent for the **17**-component Casagli layout (no proton-closure path in this slice).

### Next Steps
- **Sprint 4 (`phase1-04`):** wrap `evaluate_liquid_rhs` in a time-stepping driver, schedules for **diel** forcing, **CSTR/transport** terms, result export, and long-horizon tests.

---

<a id="devlog-20260421-simulator-math"></a>

## [2026-04-21] - Sprint phase1-03: SIMULATOR_MATH: ODEs, RHS, 22-row Petersen (SI.3.1), nested pH, and doc cross-links

### Context & Goals
Complement the existing **chemistry–centric** story in **`HYDROCHEMISTRY.md`** and the **SSOT** in **`MATH_MODEL.md`** with a **dynamics–centric** document that makes explicit how the digital twin is meant to be read as an **ODE initial-value problem** ($\mathrm{d}\mathbf{C}/\mathrm{d}t$), how **$\mathbf{S}^{\mathsf T}\boldsymbol{\rho}$** assembles the **RHS**, how the **19 biological** processes and **$\rho_{20}$–$\rho_{22}$** (Table SI.3.1 *Equilibrium phase* + SI.7) fit the **22-row** layout, and how **speciation and pH** act as a **nested algebraic map** inside each RHS evaluation. Prepare readers and future **`simulation`/integrator** work for the same language used in code issues and Stage 6 wiring.

### Technical Implementation
- **Added [`docs/SIMULATOR_MATH.md`](../SIMULATOR_MATH.md)** (*Mathematics of the ALBA simulator: dynamics, the RHS, and the path to time integration*): control volume and lumped state; RHS $\mathbf{f}$ and **$\mathbf{S}^{\mathsf T}\boldsymbol{\rho}$**; **Part D** contrasting biological kinetics and gas–liquid fluxes; **Part E** on **algebraic coupling** (pH, composition of maps); **Part F** on discretization and solvers; scope boundaries; **appendices** (notation, glossary, **concept to module** map, worked micro-examples, positivity/sensitivity remarks, **RHS evaluation flow** diagram with anchors for stable navigation.
- **`docs/HYDROCHEMISTRY.md` (intro):** one-line pointer to **`SIMULATOR_MATH.md`** for the **ODE / RHS / time-integration** viewpoint so chemistry readers know where the dynamics story lives.
- **`docs/MATH_MODEL.md` (§3 Algebraic sub-models):** pointer to **`SIMULATOR_MATH.md`** for **ODEs, RHS, Petersen assembly, and time integration**, keeping §3 as normative for parameters while steering “how it moves in time” to the new companion.


*Git trace (same day):* `6a97792`.

### Next Steps
- Stage 6: implement **22-process** RHS assembly (or **$\mathbf{S}$** extended per SI.3.1) and a **simulator module** that evaluates **pH** then **$\rho_{20}$–$\rho_{22}$** each RHS call, as anticipated in **`SIMULATOR_MATH.md`**; keep **MATH_MODEL** / **STOICHIOMETRY** in sync when code lands.

---

<a id="devlog-20260420-p103-stages-2-5"></a>

## [2026-04-20] - Sprint phase1-03: Stages 2-5: speciation, T_ref 298.15 K, pH solver, SI.7 gas transfer, simulator API

### Context & Goals
Complete the hydrochemistry path from Stage 2 (Speciation) to Stage 3 (pH solver), and align dissociation constants, reference temperature, and reaction enthalpies under a single thermodynamic convention to avoid 293 K / 298.15 K drift in van't Hoff scaling. 

Extend the same workstream with **Stage 4** numeric gas–liquid transfer (SI.7 / Table SI.7.1) using Stage-3 $[\mathrm{H}^+]$ and temperature-scaled $K_a$, without wiring $\rho_{20}$–$\rho_{22}$ into the ODE RHS yet. Add **Stage 5** simulator-facing facades over **`StateVector`** / **`EnvConditions`** so the future ODE RHS can call a thin API without duplicating SI.6 assembly.

### Technical Implementation

#### Stage 2 — speciation
- **`src/bioprocess_twin/models/chemistry.py`:** 
  - subsystem helpers for ammonia, nitrite/nitrate, carbonates, phosphate, and water; 
  - **`speciate_aqueous`** and **`speciate_from_alba_totals`** assemble SI.6.1 rows 2–14 from totals and $[\mathrm{H}^+]$.
- **`tests/unit/test_chemistry_stage2.py`:** 
  - mass closure per pool, acidic/basic limiting behaviour, one hand-checked carbonate case, and agreement between the ALBA-totals wrapper and the molar-totals path.
- **`docs/HYDROCHEMISTRY.md` and `docs/notes/aqueous-acid-base-reaction-enthalpies.md`:** 
  - stable table-of-contents anchors for long-form navigation; 
  - consolidated note for standard reaction enthalpies $\Delta H^\circ$ (conditions and lineage) linked from HydroChemistry Part E.
- **`src/bioprocess_twin/models/chemistry.py` (van’t Hoff baseline):** 
  - **`T_REF_K = 298.15 K`** so reference $K_a$ and $K_w$ from **MATH_MODEL.md** §1.2.7 share the same reference temperature as the tabulated $\Delta H^\circ$ bundle; 
  - docs clarify that SI.6.1’s ~293 K column is paper rounding, not the code SSOT for $K_{a,\mathrm{ref}}$.

#### Stage 3 — charge balance and pH
- **`src/bioprocess_twin/models/chemistry.py`:** 
  - **`charge_residual`** for SI.6.1 row 15; 
  - **`solve_pH`** with Newton in $\log_{10}[\mathrm{H^+}]$, physical pH bounds, and bounded bracket fallback; 
  - **`ChargeBalanceInputs`**, **`PHSolverOptions`**, **`PHSolveResult`**, and **`PHSolveError`** for a typed solver surface.
- **`tests/unit/test_chemistry_stage3.py`:** 
  - residual assembly vs manual row 15, convergence from different initial pH, fallback path, clear error when the bracket has no sign change, and a pinned regression case.
- **`src/bioprocess_twin/models/__init__.py`:** 
  - exports for the chemistry / pH-solver public API and, for Stage 4, gas-transfer types and helpers (**`HenryConstantsGas`**, **`GasTransferRates`**, **`calculate_gas_transfer`**, etc.).
- **`docs/HYDROCHEMISTRY.md` (Parts D, G, H):** 
  - maps the implemented Stage 3 API to the narrative (residual equation, nested-solver policy, internal-functions table).

#### Stage 4 — SI.7 gas transfer
- **`src/bioprocess_twin/models/gas_transfer.py`:** 
  - **`henry_o2_co2_nh3`** (SI.7.3–7.5), 
  - **`T_REF_CELSIUS_KLA`** and **`kinetic_kla_factor`** / **`effective_kla_o2_per_day`** for $\theta^{T-20}$ with **20 °C** as the $k_L a$ reference (distinct from the **298.15 K** anchor in Henry and **`scale_dissociation_constants_at_t`**); 
  - diffusivity ratios $(D_j/D_{\mathrm{O_2}})^{1/2}$ from **MATH_MODEL.md** §1.2.6; 
  - **`liquid_volatile_co2_gC_per_m3`** / **`liquid_volatile_nh3_gN_per_m3`** via **`speciate_carbonate`** / **`speciate_ammonia`** (SI.6–consistent volatile pools); 
  - **`calculate_gas_transfer`** returns $\rho_{20}$–$\rho_{22}$ with optional **`GasTransferConditions`** (defaults from §1.2.6); 
  - short line comments on the rate assembly in **`calculate_gas_transfer`** for readability.
- **`tests/unit/test_gas_transfer.py`:** 
  - O₂ row vs closed-form Table SI.7.1 at 20 °C; 
  - CO₂ / NH₃ driving terms vs [$\mathrm{H}^+$]; 
  - Henry $H(T)$ colder-vs-warmer check; 
  - end-to-end **`calculate_gas_transfer`** sensitivity of $\rho_{22}$ to [$\mathrm{H}^+$].
- **`docs/HYDROCHEMISTRY.md` (§9.1–9.2):** 
  - documents the implemented **`calculate_gas_transfer`** signature (explicit **`h_plus_mol_m3`** from **`solve_pH`**), 
  - renames the internal-function table to match code, and states the **20 °C** vs **298.15 K** reference split.
- **`src/bioprocess_twin/models/hydrochemistry_api.py`:** 
  - **`speciation_totals_from_state`** maps ALBA soluble totals to **`SpeciationTotals`**; 
  - **`solve_pH_from_state`** builds **`ChargeBalanceInputs`** from **`StateVector`** + **`env.temperature_C`** (ignores **`env.pH`** for SI.6); 
  - **`hydrochemistry_step`** returns **`HydrochemistryStepResult`** (**`PHSolveResult`** + **`GasTransferRates`**) by chaining **`solve_pH`** and **`calculate_gas_transfer`** with the solved **`h_plus_mol_m3`**.
- **`tests/unit/test_hydrochemistry_api.py`:** 
  - parity of **`solve_pH_from_state`** vs direct **`ChargeBalanceInputs`** (pinned stage-3 regression totals); 
  - gas bundle vs manual **`calculate_gas_transfer`**; 
  - **day vs night** **`EnvConditions`** snapshots (irradiance high vs zero, same $T$) with **`EnvConditions.pH`** aligned to solved pH before **`calculate_rates`**, asserting higher algal $\rho_1$ in daytime as preparation for **Sprint 4** time orchestration.
- **`docs/HYDROCHEMISTRY.md` (§9.1):** 
  - documents **`hydrochemistry_api.py`**, the split between **SI.6 solved pH** and **`EnvConditions.pH`** for cardinal kinetics until orchestration, and points to **Sprint 4** for time-varying $T$ / irradiance / RHS wiring.
- **`src/bioprocess_twin/models/__init__.py`:** 
  - exports **`solve_pH_from_state`**, **`hydrochemistry_step`**, **`HydrochemistryStepResult`**, **`speciation_totals_from_state`**.

*Git trace (same day, chronological):* `07987fb` → `fd4c503` → `b7a07fa` → `7389659` → `230e1bf` → `0ad017c` *(Stages 2–5).*

### 💡 Deep Dive: Why Newton + bounded fallback
Newton in log-space is fast in typical operating ranges, but can stall near flat derivatives or extreme compositions. The bounded fallback enforces physical pH limits and preserves a robust convergence path when Newton cannot guarantee progress.

### Next Steps
- Wire $\rho_{20}$–$\rho_{22}$ from **`calculate_gas_transfer`** into the assembled liquid-phase RHS (and optionally extend **`get_petersen_matrix()`** or a parallel rate vector) once the integrator path is ready (**Sprint 4**): also integrate **dynamic pH** into each RHS evaluation, **diel** profiles for $T$ and irradiance, and **inflow/outflow + HRT** transport terms not covered in phase1-03.
- Stage 5 **hydrochemistry_api** smoke path is covered by **`test_hydrochemistry_api.py`**; extend with integration tests when **`simulation.py`** exists.

---

<a id="devlog-20260419-p103-chemistry-stage1"></a>

## [2026-04-19] - Sprint phase1-03: Stage 1 and Hydrochemistry gas–liquid driving forces

### Context & Goals
Continue **phase1-03 (Hydrochemistry / pH–DAE path)** with runnable reference code for SI.6 constants and temperature scaling, and deepen the pedagogical doc toward SI.7 mass-transfer language used in the twin.

### Technical Implementation
- **`src/bioprocess_twin/models/chemistry.py`:** 
    - `R_GAS`, `T_REF_K`, reference $K_a$ from **MATH_MODEL.md** §1.2.7 (pK_a $\rightarrow$ mol/L), **`ka_at_T` / `kw_at_T`** per van’t Hoff **SI.6.4**, **`scale_dissociation_constants_at_t`**, and mol·m⁻³ helpers from ALBA totals $S_{\mathrm{IC}}$, $S_{\mathrm{NH}}$, nitrite/nitrate, $S_{\mathrm{PO4}}$. 
    - Default **`AlbaDissociationEnthalpy`** uses **zero** $\Delta H^\circ$ as of now. I need to research those missing parameters.
- **`tests/unit/test_chemistry_stage1.py`:** reference bundle, scaling API, conversion smoke checks.
- **`docs/HYDROCHEMISTRY.md`:** new/extended material on gas–liquid driving force $\Delta = C^* - C^{\mathrm{liq}}$, Henry intuition, and alignment with **SI.7**.

### Next Steps
- Source $\Delta H^\circ$ per reaction consistently with chosen $K_{a,\mathrm{ref}}$ and $T_\mathrm{ref}$; wire chemistry into speciation / charge balance when the pH solver lands.

---

<a id="devlog-20260417-p103-hydrochemistry-doc"></a>

## [2026-04-17] - Sprint phase1-03: Hydrochemistry documentation and implementation plan

### Context & Goals
Support **`phase1-03: Hydrochemistry Module and pH Solver (DAE)`** with a long-form, university-style guide that connects reactor biology, SI.6 aqueous equilibria, SI.7 gas transfer, temperature effects, and the nested DAE strategy (**ADR 003**) to an implementation map without duplicating full parameter tables in `MATH_MODEL.md`.

### Technical Implementation
- Added **[`docs/HYDROCHEMISTRY.md`](../HYDROCHEMISTRY.md)**: Parts A - I cover ODE state/totals, speciation and unit factors $10^3$/$10^6$, charge balance and alkalinity intuition, van’t Hoff vs $\theta^{T-20}$ vs Henry correlations, SI.7 driving forces, mermaid flow aligned with `ARCHITECTURE.md`, code module boundaries, testing checklist, and an appendix contrasting **`S_H_PROTON`** (ADR 007 audit) with SI.6 pH.
- Linked **`MATH_MODEL.md` §3** to **`HYDROCHEMISTRY.md`** as the pedagogical companion; SSOT tables remain in §1.2.6–1.2.7 and SI files.

### Implementation Plan
Due to the complexity of the Hydrochemistry module, I split the implementation into **six ordered stages**, each paired with **tests in the same stage** (ideally a failing test first, then code). Paths align with `src/bioprocess_twin/models/chemistry.py`, `gas_transfer.py`, and `tests/unit/test_chemistry_stage{1,2,3}.py`, `test_gas_transfer.py`. Brief scope per stage:
- **Stage 1 — Foundations and units:** 
     - Reference $pK_a$ / $K_a$ at $T_\mathrm{ref}$, $R$, $10^3$/$10^6$ factors, elemental weights (12, 14, 31); **`ka_at_T`** / **`kw_at_T`** (van’t Hoff, SI.6.4); 
     - mol-per-volume helpers from ALBA totals (e.g. $S_\mathrm{IC}/12$, $S_\mathrm{NH}/14$). 
     - *Tests:* $K_a(T)$ vs hand/SI reference at two temperatures; mass–mole conversion smoke checks.
- **Stage 2 — Speciation (SI.6.1 rows 2–14):** 
     - Given $[\mathrm{H}^+]$, $T$, and totals, return molar species concentrations; 
     - subsystem helpers + **`speciate_aqueous`** / **`speciate_from_alba_totals`**. 
     - *Tests:* mass closure per total pool; acidic/basic limiting fractions; one hand-checked multi-species case (e.g. carbonates).
- **Stage 3 — Charge balance and `solve_pH`:** 
     - Assemble SI.6.1 row 15 in **`charge_residual`**; 
     - **`solve_pH`** (Newton in $\log_{10}[\mathrm{H^+}]$ with bounded fallback); 
     - **`delta_cat_an`** (or agreed name). 
     - *Tests:* buffer/mixture vs expected pH; convergence from varied initial $[\mathrm{H}^+]$; robustness / clear errors for pathological cases.
- **Stage 4 — Gas–liquid transfer (SI.7):** 
     - **`calculate_gas_transfer`** and Table SI.7.1; 
     - **`henry_o2_co2_nh3`**, $\theta^{T-20}$ on $k_L a$ (20 °C reference), $(D_j/D_{\mathrm{O_2}})^{1/2}$; 
     - three rates using Stage-3 $[\mathrm{H}^+]$ for volatile $\mathrm{CO_2}$ / $\mathrm{NH_3}$ fractions. 
     - *Tests:* $O_2$ row closed form; $CO_2$/$NH_3$ driving force vs pH; $H(T)$ trend vs SI correlations.
- **Stage 5 — Thin simulator-facing API:** 
     - Stable facades (**`solve_pH`**, gas-transfer entry points) over **`StateVector`** / **`EnvConditions`** without forcing a full Petersen extension yet; 
     - optional smoke on a minimal synthetic state. 
     - *Tests:* light integration reusing Stage 3–4 numerics through project types.
- **Stage 6 — ODE hook-up:** 
     - Place $\rho_{20}$–$\rho_{22}$ in $\mathrm{d}\mathbf{C}/\mathrm{d}t$ when the $S$ / RHS design is ready; 
     - verify signs and mass sense on $S_\mathrm

### 💡 Deep Dive: Why a separate HydroChemistry narrative
The SSOT is optimized for **implementation lookup**; newcomers need a **single storyline** from control volume → $\mathbf{S}^\top\boldsymbol{\rho}$ → totals vs species → electroneutrality → nested Newton inside the ODE RHS. Centralizing that narrative reduces duplication in issues and onboarding without changing stoichiometric norms.

### Next Steps
- Implement **`src/bioprocess_twin/models/chemistry.py`**; extend with speciation, $\Delta H^\circ$ sources, and pH solver integration.
- Wire $\rho_{20}$–$\rho_{22}$ into the simulation RHS when the gas-transfer workstream extends `get_petersen_matrix()` or equivalent assembly.

---

<a id="devlog-20260417-sprint2b-rates-ci"></a>

## [2026-04-17] - Sprint 2B: Nineteen process rates, test policy, and CI mass-balance audit

### Context & Goals
Close **`phase1-02B: BioKinetics`** deliverables for the 19 biological rates $\rho_1\ldots\rho_{19}$ (`docs/MATH_MODEL.md` §4), keep CI green while retaining **visible, failing** strict SI elemental audits for O/H closure work, and document how to run tests locally.

### Technical Implementation
- **`src/bioprocess_twin/models/kinetics.py`:** 
    - Full `calculate_rates(state, env, kinetic_parameters=None)` with Liebig `min(...)`, NH₄/NO₃ switches, anoxic $K_O/(K_O+S_{O2})$, §3 modifiers, dimensionless light factor $f_{I,\mathrm{dim}}$ per `MATH_MODEL.md` §3.3 note; `rho[i]` = $\rho_{i+1}$ aligned with `get_petersen_matrix()` rows; nonnegative **clip** for raw `ndarray` state.
- **`docs/MATH_MODEL.md`:** 
    - Clarified §3.3 vs §4 composition for $f_I$ and $\mu_{\max,\mathrm{ALG}}$.
- **Tests:** 
    - Expanded `tests/unit/test_kinetics.py` (plateaus, cardinals, hand checks, `ndarray` clip); 
    - `pyproject.toml` marker **`strict_si_mass_balance`** on `test_mass_balance_conservation` and parametrized per-cell tests; 
    - default `pytest` uses **`-m "not strict_si_mass_balance"`** so coverage and the main CI job exclude those cases.
- **CI:** 
    - [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) — job **`mass-balance-si-audit`** runs `pytest -m strict_si_mass_balance -o addopts=''` with **`continue-on-error: true`**.
- **Docs:** 
    - [`docs/development/TESTING.md`](TESTING.md) — commands for default pytest, SI audit, subsets, and ruff.

### 💡 Deep Dive: Why two pytest modes
Elemental closure on the **literal** Casagli SI block is a **policy and modeling** topic (ADR 007, water/proton layers). Failing audits are **informative**, not regressions in $\boldsymbol{\rho}$. Splitting **gate** vs **audit** keeps velocity on kinetics while preserving a one-command view of residual cells.

### Next Steps
- Wire **ODE RHS** to `calculate_rates` and `Sᵀρ` with a chosen `closure_mode` for `S` and state length.
- Tighten or extend strict SI tests when closure rows and `I` are finalized; optionally raise audit job to hard-fail when residuals are within a new tolerance.

---

<a id="devlog-20260416-sprint2b-kinetics-stub"></a>

## [2026-04-16] - Sprint 2B: Kinetics API stub, SSOT parameters, and algebraic modifiers

### Context & Goals
Continue **`phase1-02B: BioKinetics`** on branch `feat/biokinetics-rates`: align documentation with ALBA supplementary material, expose a stable **`calculate_rates`** entry point (still returning zeros until full $\boldsymbol{\rho}$ wiring), central kinetic constants from `docs/MATH_MODEL.md` §1.2 and pure **modifier** functions for §3 (temperature, pH, light, DO).

### Technical Implementation
- **Mass-balance documentation layout (PR #13, `e8cc2df`):** Moved narratives under **`docs/mass_balances/guides/`** and case studies; generated **114-cell** audits under **`docs/mass_balances/artifacts/`**; **`scripts/generate_mass_balances_md.py`** writes to **artifacts**; hub **`docs/MASS_BALANCES.md`**; **`MATH_MODEL.md`** / **`STOICHIOMETRY.md`** updates for closure modes and SI rows 20–22; **ADR 007** and cross-links refreshed.
- **Documentation & SI layout:** Added/relocated ALBA supporting markdown and figures under `docs/supporting_informations/`; updated `docs/MATH_MODEL.md` (SSOT from SI.5, links to SI.6/SI.7, clarifications); stoichiometry docstrings now point SI.3 references at the new paths (`f8e80ab`).
- **`src/bioprocess_twin/models/kinetics.py`:** Introduced `EnvConditions` and `calculate_rates(state, env, n_processes=19)` returning a **19-vector of zeros** (contract for Paso 4); normalization for 17 vs 18 state components as in Paso 1 (`0a589bf`).
- **`src/bioprocess_twin/models/kinetic_parameters.py`:** Frozen Pydantic `KineticParameters` with nested `CardinalTemperature` / `CardinalPH`; factory `default_alba()` with nominal §1.2 values (midpoints where ranges use “±”); explicit **`k_a`** for urea / $S_{\mathrm{ND}}$ kinetics (`15450b0`).
- **`src/bioprocess_twin/models/kinetic_modifiers.py`:** Pure helpers—CTMI growth $f_T$, Arrhenius decay, CPM $f_{\mathrm{pH}}$, Haldane $f_I$, Hill $f_{\mathrm{DO}}$ (growth vs decay) per §3, with documented guards (zero outside cardinals, safe denominators).
- **`docs/MATH_MODEL.md`:** New §1.2.2 row for **`K_a`** (nominal 1.0 gN·m⁻³, Henze-style note where Casagli SI.5 does not list it).
- **Tests:** `tests/unit/test_kinetics.py`, `tests/unit/test_kinetic_parameters.py`, `tests/unit/test_kinetic_modifiers.py`; exports updated in `src/bioprocess_twin/models/__init__.py`.

### 💡 Deep Dive: SSOT parameters vs stoichiometry
`KineticParameters` holds **rates, half-saturation constants, $\theta$ factors, and cardinals** for use in $\rho_i$ expressions; **`stoichiometry.py`** remains the home for **Petersen `S`** and composition **`I`**. Keeping them separate avoids duplicating yields/composition (§1.1).

### Next Steps
- ~~Implement the **19 process rates** in `calculate_rates`~~ **Delivered** 2026-04-17 (`e2e362c` / `58f3dc4`); next: wire **ODE RHS** to `calculate_rates` and `S^\top\boldsymbol{\rho}` with chosen `closure_mode`.
- Optional: smoke test importing kinetics + modifiers from the integrator path.
- Note: strict **`S @ I.T`** mass-balance tests on literal SI rows still show large **O/H** residuals for bacterial processes (known from ADR 007 / closure work); closing the twin loop does not depend on relaxing that audit, but simulation should use **`get_petersen_matrix_for_simulation(closure_mode=...)`** when O/H-closed `S` is required.

---

<a id="devlog-20260415-sprint25a-oh-closure"></a>

## [2026-04-15] - Sprint 2.5A: Stoichiometric O and H closure layers and extended StateVector

### Context & Goals
Following **ADR 007** (policy **(a)**), the twin needs a **clear extension layer** on top of literal Casagli SI stoichiometry: first **elemental oxygen** via stoichiometric **water**, then **elemental hydrogen** via an explicit **free-proton** inventory column, without yet coupling **charge** or **SI.6 pH**. The state container must align with **17- vs 18-component** ODE layouts so future solvers can pick **SI**, **O-closure**, or **O+H-closure** consistently with the Petersen matrix shape.

### Technical Implementation
- **`src/bioprocess_twin/models/stoichiometry_closure.py`:
    - **Oxygen closure adjusts only** **`S_H2O`** using $ \alpha_i = -L_i / I_{\mathrm{O},S_{\mathrm{H2O}}} $ (`compute_stoichiometric_s_h2o_total_for_row`); 
    - proton step builds **19×18** `S` with **`S_H_PROTON`** via $ \beta_i = -R_i^{\mathrm{H}} $ (`compute_stoichiometric_s_h_proton_total_for_row`) after oxygen-closed rows. 
    - `get_petersen_matrix_for_simulation(closure_mode=...)` selects **SI / oxygen / oxygen_and_protons** (legacy `use_oh_closure` preserved). 
    - `get_composition_matrix_proton_closure()` returns **6×18** with $I_{\mathrm{H}}=1$ on the proton column only.
- **`scripts/generate_mass_balances_md.py`:
    - `--closure-of-oxygen`, `--closure-of-oxygen-and-protons`, `--closure` (alias for oxygen); 
    - Markdown generator supports **17 or 18** columns; 
    - generated audits under `docs/mass_balances/artifacts/` (`audit-si-114cell.md`, `audit-oxygen-closure-114cell.md`, `audit-oxygen-proton-closure-114cell.md`); index at `docs/MASS_BALANCES.md`.
- **Documentation:** 
    - `docs/mass_balances/guides/stoichiometric_water_rationale.md`, 
    - `docs/mass_balances/guides/proton_closure_rationale.md`, 
    - `docs/mass_balances/guides/OH_CLOSURE.md`, 
    - hub `docs/mass_balances/README.md`, 
    - and **ADR 007** implementation notes cross-link the three audit artifacts and CLI regeneration commands.
- **`src/bioprocess_twin/core/state.py`:** 
    - `StateVectorVariant` (`si`, `oxygen`, `oxygen_and_protons`) and `state_array_len`; 
    - new field **`S_H_PROTON`**; 
    - `to_array(variant=...)` and `from_array(..., variant=...)` for **17- or 18-length** arrays (inference from length when `variant` omitted). 
- **`src/bioprocess_twin/models/stoichiometry.py`:** 
    - Comment that extended layout appends **`S_H_PROTON`** at **j = 17** (aligned with closure column index).
- **Tests:** 
    - `tests/unit/test_stoichiometry_oh_closure.py`, 
    - `tests/unit/test_stoichiometry_proton_closure.py`, 
    - and expanded `tests/unit/test_state.py` for layout and round-trip behavior.

### 💡 Deep Dive: Closure as constrained stoichiometry, not ad hoc tuning
For each process row, once all **non-water** species fix $L_i$ (oxygen in explicit columns), **one** total coefficient on **`S_H2O`** is uniquely determined if missing O is booked as **water oxygen** in the existing $I$ convention. Likewise, after that step, $R_i^{\mathrm{H}}$ from the **17** columns fixes the **proton** coefficient if residual H is booked in a single column with $I_{\mathrm{H}}=1$. The numeric values match the old “delta to pass the audit” algebra; the **interpretation** is that these are the **only** coefficients consistent with the table plus the stated closure conventions, parallel to textbook balancing with explicit $\mathrm{H}_2\mathrm{O}$ and $\mathrm{H}^+$, but in the ALBA **mass/COD** state basis.

### Next Steps
- Wire the **ODE RHS** (and any integrator) to pass **`closure_mode`** through to both **`get_petersen_matrix_for_simulation`** and **`StateVector.to_array` / `from_array`** so $d\mathbf{C}/dt$ dimension always matches **`S`**.
- Later: **charge / alkalinity** and **SI.6** coupling; tighten **`MASS_BALANCE_ATOL`** on O/H once extensions are stable in simulation, not only in audit Markdown.

---

<a id="devlog-20260331-sprint2-deepresearch-oh"></a>

## [2026-03-31] - Sprint 2: DeepResearch and Explicit O/H Balance (rho14, ALBA vs ASM)

### Context & Goals
Mass balance tests still showed large residuals for **oxygen** and **hydrogen** even after careful transcription of Casagli et al. (2021) supplementary tables. I reviewed the literature and used **DeepResearch** (external literature-style synthesis) and a **term-by-term worksheet** in [**`Balance of oxygen for rho14`**](../mass_balances/case-studies/rho14-aob-oxygen-balance.md) for aerobic AOB growth (**rho14**) to separate “transcription error” from **structural** modeling conventions.

### Technical Implementation
- **Documentation:** [**`Balance of oxygen for rho14`**](../mass_balances/case-studies/rho14-aob-oxygen-balance.md) documents the algebraic check $B_{14,\mathrm{O}} = \sum_j S_{13,j} I_{\mathrm{O},j}$ with numeric breakdown; it matches the automated test `S @ I.T` and reproduces the residual $\approx -5.55$ with SI-aligned coefficients.
- **Research output:** Internal report  argues the residual is **not** a typo in $Y_{\mathrm{AOB}}$ or COD/N tracking, but missing **explicit water ($S_{\mathrm{H2O}}$)** and **acid–base carriers (e.g. $\mathrm{H}^+$)** on **ASM-style** bacterial rows. Algal processes in ALBA (rho1–rho3) already include $S_{\mathrm{H2O}}$ terms in SI.3.3; many bacterial processes do not, which breaks **strict elemental** O and H closure while leaving **COD/electron** balances (ASM tradition) largely intact.
- **Takeaway for the codebase:** Passing a strict $B \approx 0$ audit requires either extending Petersen rows with closure species (water/protons) and consistent `I`, or scoping the test to elements the SI was designed to conserve (documented policy decision).

### 💡 Deep Dive: Why Perfect SI Transcription Does Not Imply $B_{i,\mathrm{O}} = 0$
The test enforces $\sum_j S_{i,j} I_{k,j} \approx 0$ for every process $i$ and element $k$. ASM-class matrices are usually built so **COD** (electron balance) and **nutrients** close for design variables; **water** is treated as infinite and **pH** as buffered, so **atomic O and H** are often not closed in `S`. ALBA mixes **algal** stoichiometry (with $S_{\mathrm{H2O}}$) and **bacterial** blocks inherited from that tradition. For rho14, $I_{\mathrm{O}}$ for $\mathrm{NH_4^+}$ is zero (no O in the ion), while $\mathrm{NO_2^-}$ carries oxygen; without a compensating **water** (or equivalent) term in `S`, the same SI numbers that are correct for ASM-style energy bookkeeping can still fail a **full** O (and H) matrix audit—exactly what [**`Balance of oxygen for rho14`**](../mass_balances/case-studies/rho14-aob-oxygen-balance.md) quantifies.

### Next Steps
- **Policy (2026-04-14):** **(a)** adopted — extend the model for strict O/H closure; recorded in [**ADR 007**](../adrs/007-elemental-mass-balance-oh-closure.md) (also linked from `SPRINTS.md` and `docs/STOICHIOMETRY.md`).
- Derive and implement \(S_{\mathrm{H2O}}\) (and charge balance if needed) for rho14 and peer bacterial processes; re-tighten `MASS_BALANCE_ATOL` toward `1e-6`.
- Cross-check any proposed coefficients against Casagli SI and primary references before merging.

---

<a id="devlog-20260330-sprint2-mass-balance-diag"></a>

## [2026-03-30] - Sprint 2: Mass Balance Cell Diagnostics and SI Alignment Checkpoint

### Context & Goals
After aligning `stoichiometry.py` with corrected project docs and ALBA SI tables, **observable diagnostics** were needed for mass balance residuals (per element and per process×element cell) and a **git checkpoint** so further work on O/COD/H closure could proceed from a known state.

### Technical Implementation
- **`src/bioprocess_twin/models/stoichiometry.py`:** Petersen and composition matrices updated per ALBA SI and corrected documentation; continued use of explicit α expressions from SI.3.3 where applicable.
- **`tests/unit/stoichiometry_mass_balance_shared.py`:** Centralized `compute_mass_balance_matrix()` (`S @ comp.T`), `MASS_BALANCE_ATOL`, process labels, `format_mass_balance_by_element_summary` (worst row per element), and `format_mass_balance_all_cells` (114 residuals).
- **`tests/unit/test_stoichiometry_mass_balance_cells.py`:** Split 114 parametrized per-cell tests; optional full table print with `pytest -s`.
- **`tests/unit/test_stoichiometry.py`:** `test_mass_balance_conservation` prints element summary and full cell table; relaxed `atol` retained pending COD/O/H closure.
- **`pyproject.toml`:** `pythonpath = ["tests/unit"]` for clean imports of shared test helpers.
- **`docs/STOICHIOMETRY.md`**, **`docs/development/DEVLOG.md`:** Refreshed for current matrices and follow-up on mass balance investigation.

### 💡 Deep Dive: From Scalar `atol` to 114 Cells
A single `np.allclose` on `S @ I.T` hides which **process–element** pairs fail. Printing **max |residual|** per element (with `argmax` row) and **every** of the 19×6 cells turns mass balance debugging into a checklist: e.g. rho14 vs O points straight to AOB + inorganic nitrogen + $S_{\mathrm{O2}}$ interactions, which motivated the explicit rho14 worksheet in [**`Balance of oxygen for rho14`**](../mass_balances/case-studies/rho14-aob-oxygen-balance.md)).

### Next Steps
- Same as follow-up from mass balance investigation: resolve O/H (and remaining COD) closure policy.

---

<a id="devlog-20260319-sprint2-petersen"></a>

## [2026-03-19] - Sprint 2: Petersen Matrix and Stoichiometry (Mass Balance Verification)

### Context & Goals
The goal was to implement the **Stoichiometry** module for the ALBA model (`phase1-02`), providing the Petersen matrix (19×17) and composition matrix (6×17) required for mass balance validation. This work builds on the `StateVector` structure and establishes the biological stoichiometric backbone for the simulation engine.

### Technical Implementation
- **Stoichiometry Module (`src/bioprocess_twin/models/stoichiometry.py`):**
  - Implemented `get_composition_matrix()` returning a 6×17 array (rows: COD, O, C, N, P, H).
  - Implemented `get_petersen_matrix()` with 19 biological process rows.
  - Corrected composition matrix values to match supplementary material: `I_H_S_IC=0`, `I_H_S_ND=0.14`, `I_H_S_NH=0.22`, and `S_N2` N=1.
- **Documentation (`docs/STOICHIOMETRY.md`):**
  - Fixed typos in the H row of the composition matrix to align with SI.3.1.
- **Unit Tests (`tests/unit/test_stoichiometry.py`):**
  - Added dimension tests for both matrices.
  - Implemented mass balance conservation test for all 6 elements (COD, O, C, N, P, H) with per-species error reporting.
  - Spot-checked Petersen matrix structure (phototrophic growth, decay, urea hydrolysis).
- **Mass Balance Status:**
  - C, N, P balances close within numerical precision.
  - COD, O, and H show residuals (max ~5.56 for O in AOB process); `atol=6.0` used temporarily pending root cause analysis.

### 💡 Deep Dive: Mass Balance Verification via Matrix Multiplication
A fundamental physics law is the mass conservation law, that states that the mass of a system is conserved (There's no creation or destruction of mass). This law can be summarized as follows: 

The stoichiometric mass balance for each process $i$ and element $k$ is 
$$\sum_j S_{ij} \cdot I_{kj} \approx 0$$

Where $S_{ij}$ is the stoichiometric coefficient of the $i$-th process and the $j$-th element, and $I_{kj}$ is the composition of the $k$-th element in the $j$-th process.

In other words, the sum of the products of the stoichiometric coefficients and the composition matrix should be zero. This expression can be verified by computing `balance = S @ I.T`, which is a matrix multiplication between the Petersen matrix and the composition matrix, where a 19×6 matrix is obtained. Each entry of this matrix should be zero if mass is conserved. This algebraic check catches transcription errors and composition mismatches before they propagate into simulations.

### Next Steps
- **Investigate why mass balances do not close** for COD, O, and H across all processes; root cause may relate to ALBA composition conventions or matrix construction.
- Implement the BioKinetics module with the 19 process rate equations.
- Integrate the Petersen matrix into the ODE right-hand side for dynamic simulation.

---

<a id="devlog-20260317-sprint1-state-vector"></a>

## [2026-03-17] - Sprint 1: Implementation of State Vector

### Context & Goals
The goal was to implement the `StateVector` component, which serves as the central data structure for the ALBA model. This component must bridge the gap between human-readable biological variables and the numerical arrays required by the ODE solver, while enforcing physical constraints (non-negative concentrations). This completes the core data structures for `phase1-01`.

### Technical Implementation
- **State Vector Schema (`src/bioprocess_twin/core/state.py`):**
  - Implemented `StateVector` using `pydantic` to manage exactly 17 state variables.
  - Enforced non-negative constraints on all variables using `Field(..., ge=0.0)`.
  - Applied `frozen=True` to ensure state immutability at any given time step.
- **Numerical Integration Support:**
  - Added `to_array()` method to export states as NumPy arrays in the strict order required by the stoichiometric matrix.
  - Added `from_array()` class method to reconstruct the `StateVector` from solver outputs.
- **Testing & Quality:**
  - Developed `tests/unit/test_state.py` covering instantiation, array mapping, physical validation, and immutability.
  - Fixed Pydantic V2 deprecation warnings regarding `model_fields` access.
  - Verified 100% test pass rate and code formatting with `ruff`.
- **Documentation (`docs/STOICHIOMETRY.md`):**
  - Added the stoichiometric matrix and the composition matrix to the documentation. It serves as a quick reference for the stoichiometric coefficients and the composition matrix that will be used in the BioKinetics module.

### 💡 Deep Dive: Name-to-Index Mapping in Bioprocess Models
In large-scale models like ALBA (17 variables), managing array indices manually is a major source of bugs (e.g., swapping $S_{NH}$ with $S_{NO3}$). By using a Pydantic-based `StateVector`, the "Source of Truth" can be centralized for variable ordering. The `to_array` and `from_array` methods act as a translation layer, allowing the simulation logic to use descriptive names (e.g., `state.X_ALG`) while the numerical engine receives the raw vectors it expects.

### Next Steps
- Begin **Sprint 2** (`phase1-02`): Implementation of the `Stoichiometry` module.
- Define the dynamic construction of the Petersen Matrix using the parameters from `MATH_MODEL.md`.
- Implement mass balance verification tests for the stoichiometric coefficients.

---

<a id="devlog-20260316-sprint1-reactor-config"></a>

## [2026-03-16] - Sprint 1: Implementation of Reactor Configuration

### Context & Goals
The objective was to implement the physical and operational configuration layer for the ALBA model. This layer serves as the foundation for the Digital Twin, ensuring that all physical boundaries (geometry, volume, depth) are correctly validated and immutable during simulation. This work addresses the first part of issue `phase1-01`.

### Technical Implementation
- **Configuration Schema (`src/bioprocess_twin/core/reactor.py`):**
  - Implemented `ReactorConfig` using `pydantic` for strict data validation.
  - Enforced immutability using `ConfigDict(frozen=True)` to prevent accidental changes to the reactor's physical design.
  - Created nested models for `GeometryConfig`, `OperationalConfig`, and `LocationConfig`.
- **Process Logic & Engineering Methods:**
  - Added `calculate_hrt` to compute Hydraulic Retention Time dynamically.
  - Implemented `theoretical_volume` and `volume_discrepancy` properties to handle the difference between design specifications and ideal geometry ($Area \times Depth$).
- **Data Integration:**
  - Created `config/reactor_setup.yaml` with the exact specifications of the Narbonne HRABP reactor (56 m², 17 m³, 0.3 m depth).
  - Added a robust `from_yaml` loader with root-key validation.
- **Testing (TDD):**
  - Developed `tests/unit/test_reactor.py` covering 7 test cases: loading, validation, missing fields, HRT calculation, file-not-found handling, immutability, and volume logic.

### 💡 Deep Dive: Immutability with Pydantic Frozen Models
In a complex bioprocess simulation, maintaining the integrity of physical constants is critical. By using `frozen=True` in our Pydantic models, we ensure that once the `ReactorConfig` is loaded, it cannot be modified by any other part of the system. This prevents a whole class of "silent bugs" where a developer might accidentally overwrite a geometric parameter (like surface area) mid-simulation, which would invalidate all subsequent mass balance and light penetration calculations.

### Next Steps
- Complete **Sprint 1** (`phase1-01`): Implement the `StateVector` class.
- Define the 17 state variables in `src/bioprocess_twin/core/state.py`.
- Develop unit tests to verify name-to-index mapping and NumPy array conversion for the ODE solver.

---

<a id="devlog-20260313-sprint0-cicd"></a>

## [2026-03-13] - Sprint 0: CI/CD and Infrastructure Setup

### Context & Goals
To ensure the reliability and maintainability of the BioProcess-Twin Hub, I established a robust development environment before implementing complex biological logic. The goal was to configure a "Modular Monolith" architecture where backend and frontend coexist in a single installable package, backed by automated quality gates (linting and testing).

### Technical Implementation
- **Package Structure:** Created `src/bioprocess_twin` as the main package, containing `core`, `models`, and `ui` submodules to support the unified architecture.
- **Build System:** Configured `pyproject.toml` using `hatchling` as the build backend to support editable installs (`pip install -e .`).
- **Quality Tools:**
  - **Testing:** Configured `pytest` with `pytest-cov` to enforce code coverage. Added `tests/conftest.py` and a sanity check suite.
  - **Linting:** Configured `ruff` to enforce PEP 8 standards and import sorting.
- **CI/CD Pipeline:** Implemented `.github/workflows/ci.yml` using `uv` for ultra-fast dependency resolution and testing on every push.

### 💡 Deep Dive: Modular Monolith with Hatchling
A "Modular Monolith" approach was chosen over separating backend and frontend repositories. By using `hatchling` and defining `src/bioprocess_twin` as a single installable package, the Streamlit UI (`src/bioprocess_twin/ui/app.py`) can import core logic (`src/bioprocess_twin/core/`) directly as a library. This eliminates the need for complex REST APIs for internal communication and simplifies the deployment pipeline to a single artifact.

### Next Steps
- Begin **Sprint 1** (`phase1-01`): Implement `StateVector` and `ReactorConfig`.
- Translate the mathematical constants from `MATH_MODEL.md` into `src/config/constants.yaml`.
- Develop the initial unit tests for the reactor geometry and the initial state vector.

---

<a id="devlog-20260312-phase1-spec-arch"></a>

## [2026-03-12] - Phase 1: Technical Specification & Architecture Definition

### Context & Goals
The project transitioned from initial research to concrete technical specification. The primary goal was to establish a Single Source of Truth (SSOT) for the ALBA mathematical model to prevent ambiguity during implementation. Additionally, I needed to formalize architectural decisions regarding the simulation engine and data handling to ensure the system can handle the stiffness of bioprocess kinetics.

### Technical Implementation
- **Mathematical Specification (`docs/MATH_MODEL.md`):**
  - Consolidated all kinetic and stoichiometric parameters from supplementary materials into a unified configuration table.
  - Defined the state vector explicitly (17 variables).
  - Wrote out the 19 specific process rate equations ($\rho$) including Liebig's Law and inhibition terms.
- **Architecture Update (`docs/ARCHITECTURE.md`):**
  - Added the `Stoichiometry` module to decouple matrix construction from kinetic rate calculations.
  - Refined the data flow between Solver, BioKinetics, and HydroChemistry.
- **Architectural Decision Records (`docs/adrs/`):**
  - Created 6 ADRs to document critical engineering choices:
    1.  **Engine:** Python + SciPy + Numba (JIT) for performance.
    2.  **Stiffness:** Implicit Solvers (LSODA/Radau) for stability.
    3.  **DAE Strategy:** Nested Algebraic Solver for pH.
    4.  **Config:** YAML + Pydantic for validation.
    5.  **Data:** Apache Parquet for efficient I/O.
    6.  **Units:** Strict SI Base ($g, m^3, d$) standard.

### 💡 Deep Dive: Handling Stiff Bioprocess Systems
Bioprocess models are notoriously "stiff" because they couple phenomena occurring at vastly different time scales: chemical equilibria (milliseconds), gas transfer (minutes), and biomass growth (days). Using standard explicit solvers (like RK45) forces the time step to be infinitesimally small to maintain stability, making long-term simulations impractical. By adopting implicit solvers (LSODA) and decoupling the algebraic pH equations (Nested Solver approach), we ensure the Digital Twin can simulate weeks of operation in seconds.

### Next Steps
- Implement the configuration loader using Pydantic (`src/config/`).
- Create the `Stoichiometry` module to generate the Petersen Matrix dynamically.
- Implement the `BioKinetics` module with Numba-optimized rate equations.

---

<a id="devlog-20260312-phase1-alba-digitization"></a>

## [2026-03-12] - Phase 1: ALBA Model Analysis & Data Digitization

### Context & Goals
To ensure the fidelity of the **ALBATwinHub** simulation engine, it is essential to ground the mechanistic model in verified data. I performed a comprehensive review of the original ALBA model (Casagli et al., 2021), focusing on the mathematical backbone of the system. The goal was to digitize these complex kinetic expressions and stoichiometric data from static PDF formats into structured, machine-readable Markdown, laying the foundation for the first iteration of the simulation engine.

### Technical Implementation
- **Data Digitization:** Extracted and structured critical model parameters from the supplementary material, specifically:
    - **Growth Kinetics:** Digitized the mathematical expressions governing algal growth rates (SI.1.1).
    - **Stoichiometry:** Transcribed the stoichiometric matrix and coefficient values (SI.3) into Markdown tables.
- **Model Verification:** Reviewed the system's stoichiometry to ensure mass balance consistency before implementation.
- **Documentation:** Created structured Markdown files to serve as the "source of truth" for the Python implementation, facilitating easier review and version control compared to the original PDF documents.

### 💡 Deep Dive: Structured Data for Mechanistic Models
Transcribing mathematical models from academic papers to code is a critical step often prone to error. By first converting the raw PDF data into structured Markdown tables (rather than jumping straight to code), we create an intermediate verification layer. This allows for easier peer review of the constants and equations against the original paper, ensuring that the `ALBATwin` engine is built on accurate, traceable definitions of the biological system.

### Next Steps
- Update the `MATH_MODEL.md` file with the verified kinetic equations and stoichiometric matrix.
- Review how to translatate this mathematical expressions into Python code.

---

<a id="devlog-20260310-phase0-init"></a>

## 2026-03-10 - Phase 0: Project Initialization and Foundation

### Context & Goals
The project was kicked off with a focus on establishing a professional software engineering foundation before writing functional code. The primary goal was to translate high-level scientific research (Casagli et al., 2021) into a structured technical plan and initialize a reproducible, modern Python development environment.

### Technical Implementation
- **Documentation Suite:** Created core technical documents in the `docs/` directory:
    - `PRD.md`: Defined product vision, user stories, and functional requirements.
    - `ARCHITECTURE.md`: Designed a modular system separating physics, biology, and chemistry using Mermaid.js diagrams.
    - `MATH_MODEL.md`: Transcribed 19 biological processes and pH equilibria into LaTeX format.
    - `ROADMAP.md`: Established a 5-phase execution plan.
- **Environment Management:** Initialized the project using `uv`. Configured `pyproject.toml` with a comprehensive scientific stack: `numpy`, `scipy`, `pandas`, `pyyaml` for the mechanistic engine; `torch` for the PINN module; and `streamlit` for the visualization layer.
- **Git Configuration:** Implemented a robust `.gitignore` to handle large binary PDFs and generated datasets. Performed index cleanup (`git rm --cached`) to ensure reference papers are not tracked by version control while remaining available locally.

### 💡 Deep Dive: Environment Reproducibility with `uv`
I chose `uv` as the primary package manager to solve the common "research code" problem of broken dependencies. By using `uv.lock`, I ensure that the exact versions of `scipy` (for ODE solving) and `torch` (for neural networks) are locked. This is critical for the Hybrid Twin approach, as small changes in underlying library versions can lead to numerical instability in high-dimensional biological simulations.

### Next Steps
The phase 0 is complete. The project is now ready to start the Phase 1: Mechanistic Engine (ALBATwin).
