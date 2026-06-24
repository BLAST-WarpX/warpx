/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_HYBRIDPICMODEL_H_
#define WARPX_HYBRIDPICMODEL_H_

#include "HybridPICModel_fwd.H"

#include "Fields.H"

#include "EBJBoundary.H"
#include "ExternalVectorPotential.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <optional>
#include <utility>

/**
 * \brief This class contains the parameters needed to evaluate hybrid field
 * solutions (kinetic ions with fluid electrons).
 */
class HybridPICModel
{
public:
    HybridPICModel ();

    /** Read user-defined model parameters. Called in constructor. */
    void ReadParameters ();

    /** Allocate hybrid-PIC specific multifabs. Called in constructor. */
    void AllocateLevelMFs (
        ablastr::fields::MultiFabRegister & fields,
        int lev,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        int ncomps,
        const amrex::IntVect& ngJ,
        const amrex::IntVect& ngRho,
        const amrex::IntVect& ngEB,
        const amrex::IntVect& jx_nodal_flag,
        const amrex::IntVect& jy_nodal_flag,
        const amrex::IntVect& jz_nodal_flag,
        const amrex::IntVect& rho_nodal_flag,
        const amrex::IntVect& Ex_nodal_flag,
        const amrex::IntVect& Ey_nodal_flag,
        const amrex::IntVect& Ez_nodal_flag,
        const amrex::IntVect& Bx_nodal_flag,
        const amrex::IntVect& By_nodal_flag,
        const amrex::IntVect& Bz_nodal_flag
    ) const;

    void InitData (const ablastr::fields::MultiFabRegister& fields);

    /**
     * \brief Apply the collocated conformal magnetic-field wall treatment to the
     * INITIAL B field (call once after the initial grid B has been set). The
     * direct mirror fill sets the covered band to its Neumann value so the first
     * field-push substep does not see it jump from the staircase-masked zero
     * (which stalls the RKF45 substepping). A no-op when the wall fill is
     * disabled (conformal_b_off) or off the collocated embedded-boundary path.
     */
    void InitialBEBFill ();

    /**
     * \brief
     * Function to evaluate the external current expressions and populate the
     * external current multifab. Note the external current can be a function
     * of time and therefore this should be re-evaluated at every step.
     */
    void GetCurrentExternal ();

    /**
     * \brief
     * Function to calculate the total plasma current based on Ampere's law while
     * neglecting displacement current (J = curl x B). Any external current is
     * subtracted as well. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     *
     * \param[in] Bfield       Magnetic field from which the current is calculated.
     * \param[in] eb_update_E  Indicate in which cell J should be calculated (related to embedded boundaries).
     */
    void CalculatePlasmaCurrent (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E
    ) const;
    void CalculatePlasmaCurrent (
        ablastr::fields::VectorField const& Bfield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev
    ) const;

    /**
     * \brief
     * Function to update the E-field using Ohm's law (hybrid-PIC model).
     */
    void HybridPICSolveE (
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        bool solve_for_Faraday) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, bool solve_for_Faraday) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, PatchType patch_type, bool solve_for_Faraday) const;

    void BfieldEvolveRK (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolveRK (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolveRKF45 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt_half, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolveRKF45 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void FieldPush (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    /** Divergence target of the transitional Marder correction */
    enum class MarderTarget { Ohm, GradPeOnly, Zero };
    /** Cadence at which the transitional Marder correction is applied */
    enum class MarderLevel { AllSubsteps, HalfSteps, FullSteps };
    /** Where in the field advance a Marder application is requested */
    enum class MarderSite { Substep, HalfStep, FullStep };

    /**
     * \brief Cadence-gated, all-levels wrapper of the transitional Marder
     * correction: applies ApplyMarderCorrection on every level when
     * marder_alpha > 0 and `site` matches marder_correction_level. The
     * Substep site additionally honors marder_substep_interval through a
     * counter that is reset at the start of every hybrid field advance.
     *
     * \param[in,out] Efield      E field to correct
     * \param[in] Jfield          ion current density (same J the Ohm solve used)
     * \param[in] Bfield          magnetic field (same B the Ohm solve used)
     * \param[in] rhofield        nodal charge density (same rho the Ohm solve used)
     * \param[in] eb_update_E     embedded-boundary solution masks for E
     * \param[in] site            which call site this is
     */
    void MarderCorrectE (
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        MarderSite site);

    /**
     * \brief Transitional Marder divergence cleaning of the Ohm's-law E
     * field (matches the Phoenix explicit-branch correction). Iteratively
     * applies
     *
     *     E += alpha * min(dx)^2 * mask * grad( div(E) - div_target )
     *
     * restricted to the low-density transition band
     * (0 < rho <= n_floor * q_e); dense plasma, true vacuum, and
     * embedded-boundary masked edges are left unchanged. div_target is,
     * per marder_target: the divergence of the bare Hall/pressure
     * Ohm's-law field ((J - J_i) x B - grad Pe) / max(rho, n_floor * q_e)
     * ("ohm", with J the Ampere plasma current), of the pressure term only
     * -grad(Pe) over the same floored rho ("grad_pe_only"), or zero
     * ("zero"). The masked L2 norm of the divergence error controls
     * convergence: the loop exits when resid < atol or
     * resid < rtol * resid(first iteration). After every update the domain
     * and embedded-boundary E conditions are re-applied. Only supported in
     * 3D Cartesian and RZ geometry (aborts otherwise).
     *
     * \param[in,out] Efield   E field to correct (single level)
     * \param[in] Jfield       ion current density
     * \param[in] Bfield       magnetic field
     * \param[in] rhofield     nodal charge density
     * \param[in] eb_update_E  embedded-boundary solution masks for E
     * \param[in] lev          mesh refinement level
     * \param[in] alpha        override of marder_alpha (Python unit tests)
     * \param[in] max_iterations override of marder_max_iterations
     * \param[in] rtol         override of marder_rtol
     * \param[in] atol         override of marder_atol
     * \param[in] target       override of marder_target
     * \param[in] allow_target_cache  if false, bypass the per-level
     *            grad_pe_only target cache (Python unit tests modify Pe
     *            without bumping the field epoch)
     *
     * \return pair of (number of iterations executed, masked divergence
     * error L2 norm measured at the top of the last executed iteration)
     */
    std::pair<int, amrex::Real> ApplyMarderCorrection (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev,
        std::optional<amrex::Real> alpha = std::nullopt,
        std::optional<int> max_iterations = std::nullopt,
        std::optional<amrex::Real> rtol = std::nullopt,
        std::optional<amrex::Real> atol = std::nullopt,
        std::optional<MarderTarget> target = std::nullopt,
        bool allow_target_cache = true) const;

    /**
     * \brief Optional Marder-like diffusive divergence clean of a nodal vector
     * field. Iterates field += alpha_scaled * grad(div(field)) restricted to
     * a near-wall geometric band (|distance_to_eb| <= band_cells*h), re-applying
     * the embedded-boundary PEC fill (with the field's own parity) each sweep.
     * The correction is a pure gradient, so curl(field) -- and hence J=curl(B) /
     * the physics it feeds -- is unchanged to round-off; only the spurious
     * curved-wall divergence injected by the mirror fill is dissipated. Used to
     * remove the secular div(B) and div(J_total) growth at the conformal wall.
     * No-op when alpha<=0 or no EB. Collocated/nodal Cartesian + RZ/XZ.
     */
    void MarderCleanDivergence (
        ablastr::fields::VectorField const& field,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update,
        warpx::hybrid::EBFillStatus* status_cache,
        bool normal_odd,
        bool fill_covered_centers,
        amrex::Real alpha,
        int max_iters,
        amrex::Real clean_band_cells,
        amrex::Real fill_band_cells,
        int lev) const;

    /** Once-per-step driver for the div(B)/div(J_total) clean (the
     *  divb_clean_per_step cadence): cleans Bfield_fp (magnetic parity) and the
     *  total Ampere current hybrid_current_fp_plasma (electric parity) on every
     *  level via MarderCleanDivergence. No-op unless the respective alpha>0. */
    void MarderCleanFieldsPerStep () const;

    /**
     * \brief
     * Function to calculate the electron pressure using the simulation charge
     * density. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     */
    void CalculateElectronPressure () const;
    void CalculateElectronPressure (int lev) const;

    /**
     * \brief Fill the electron pressure multifab given the kinetic particle
     * charge density (and assumption of quasi-neutrality) using the user
     * specified electron equation of state.
     *
     * \param[out] Pe_field scalar electron pressure MultiFab at a given level
     * \param[in] rho_field scalar ion charge density Multifab at a given level
     */
    void FillElectronPressureMF (
        amrex::MultiFab& Pe_field,
        amrex::MultiFab const& rho_field ) const;

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active) */
    int m_substeps = 10;

    /** Relative tolerance for RKF45 adaptive substep error control */
    amrex::Real m_substep_rtol = 1.e-4;
    /** Absolute tolerance for RKF45 adaptive substep error control */
    amrex::Real m_substep_atol = 1.e-8;
    /** Safety factor for RKF45 step-size adjustment */
    amrex::Real m_substep_safety = 0.9;
    /** Maximum step-size growth factor per accepted step */
    amrex::Real m_substep_max_growth = 5.0;
    /** Maximum number of substep attempts (accepted + rejected) before abort */
    int m_max_substep_attempts = 250;
    /** If true, use the adaptive RKF45 integrator; if false, use fixed-step RK4 */
    bool m_use_rkf45 = false;

    bool m_holmstrom_vacuum_region = false;

    /** If true, in partially-covered EB cells (the cut-cell wall band) make the
     *  generalized Ohm's law resistive-only: drop the stiff 1/n Hall and
     *  electron-pressure terms (and the hyper-resistivity and corner-curl
     *  corrections, which read J in the covered/mirror region), leaving E = eta*J.
     *  J = curl(B)/mu0 there is well-posed because the wall-filled B feeds the
     *  curl. Requires a staggered (ECT) embedded boundary. Default off. */
    bool m_eb_resistive_only_partial = false;

    /** Electron temperature in eV */
    amrex::Real m_elec_temp;
    /** Reference electron density */
    amrex::Real m_n0_ref = 1.0;
    /** Electron pressure scaling exponent */
    amrex::Real m_gamma = 5.0/3.0;

    /** Plasma density floor - if n < n_floor it will be set to n_floor */
    amrex::Real m_n_floor = 1.0;

    /** Plasma resistivity */
    std::string m_eta_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_resistivity_parser;
    amrex::ParserExecutor<3> m_eta;
    bool m_resistivity_has_J_dependence = false;

    /** Plasma hyper-resisitivity */
    std::string m_eta_h_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_hyper_resistivity_parser;
    amrex::ParserExecutor<2> m_eta_h;
    bool m_include_hyper_resistivity_term = false;
    bool m_hyper_resistivity_has_B_dependence = false;
    /** If true (default), evaluate the hyper-resistivity Laplacian with the
     *  isotropic Mehrstellen (2D) / Patra-Karttunen (3D) stencils instead of
     *  the cross stencil, removing the cos(4*theta) anisotropy of its
     *  damping rate (fully isotropic on cubic cells; Cartesian only). */
    bool m_isotropic_hyper_resistivity = true;
    /** If true (default), add an isotropizing corner-curl correction to the
     *  resistive electric field so the emergent in-plane B diffusion uses the
     *  isotropic (Mehrstellen) Laplacian instead of the cross stencil. This
     *  removes the cos(4*theta) modulation of the resistive damping rate that
     *  drives a grid m=4 mode. Applied through E, so the standard Faraday
     *  curl is unchanged and div(B) stays exactly zero. Cartesian only (the
     *  RZ resistive operator is axisymmetric and has no such anisotropy). */
    bool m_isotropic_resistivity = true;

    /** External current */
    std::string m_Jx_ext_grid_function = "0.0";
    std::string m_Jy_ext_grid_function = "0.0";
    std::string m_Jz_ext_grid_function = "0.0";
    std::array< std::unique_ptr<amrex::Parser>, 3> m_J_external_parser;
    std::array< amrex::ParserExecutor<4>, 3> m_J_external;
    bool m_has_external_current = true;
    bool m_external_current_has_time_dependence = false;

    /** External E/B fields */
    bool m_add_external_fields = false;
    std::unique_ptr<ExternalVectorPotential> m_external_vector_potential;

    /** If true, use the conformal embedded-boundary Faraday update (enlarged
     *  cell technique, Xiao and Liu, IEEE MWCL 2004) for the B-field push
     *  instead of the stair-step approximation. Requires embedded boundaries
     *  and a 3D Cartesian staggered grid. */
    bool m_use_conformal_eb = false;
    /** If true, extend the magnetic field into the covered cells with a 2nd-order
     *  quadratic (even-reflection) gather BEFORE the Ampere curl curl(B)->J, so
     *  the near-wall plasma current (and the Ohm's-law E built from it) is 2nd
     *  order at a curved wall. Without it the curl reads the staircased covered B
     *  and the hybrid is only 1st order near a curved wall even though the
     *  conformal (ECT) B push is 2nd order. Staggered (Yee) grid only -- the
     *  nodal/collocated quadratic gather is not yet robust and is skipped. Opt-in
     *  (default off -> byte-identical); intended for use_conformal_eb runs. */
    bool m_conformal_b_curl_fill = false;
    /** If true, apply the along-edge curvature correction to the conformal-ECT
     *  Faraday circulation: in EvolveECTRho each cut edge's E is Taylor-shifted
     *  from the full-edge center to the centroid of its uncovered segment before
     *  forming the per-face EMF. Without it the circulation is a 1st-order
     *  midpoint quadrature over the curved contour, capping the conformal B push
     *  at ~1st order at a curved wall even when curl(B)->J is already 2nd order
     *  (conformal_b_curl_fill). Uses the edge_cent_offset geometry field. Staggered
     *  (Yee) grid only; opt-in (default off -> byte-identical, since the shift is
     *  zero on uncut/covered edges). */
    bool m_conformal_ect_curvature = false;
    /** If true, skip the collocated conformal magnetic-field wall treatment
     *  entirely (no EB B fill): the unmasked nodal Faraday curl is left as-is.
     *  This reproduces the pre-treatment baseline and isolates the wall fill as
     *  the source of the grid-locked m = 4 perturbation. Diagnostic only -- it
     *  leaves spurious fields at the wall, so it is not for production runs. */
    bool m_conformal_b_off = false;
    /** If true, apply the cylindrical (surface-of-revolution) radial metric
     *  correction to the embedded-boundary mirror fill of E, J and B: the
     *  radial and azimuthal parts of the reflected field are scaled by the
     *  radial Jacobian r_image/r_fill -- the leading O(h/R) curved-wall
     *  correction the planar mirror misses. Opt-in; requires use_conformal_eb
     *  and a 3D Cartesian grid; the wall must be a surface of revolution about
     *  m_eb_cyl_axis, centered on the transverse origin. */
    bool m_eb_cylindrical_correction = false;
    /** Symmetry axis (0=x, 1=y, 2=z) for m_eb_cylindrical_correction. */
    int m_eb_cyl_axis = 2;

    /** Relative residual tolerance of the Jacobi band relaxation of the
     *  embedded-boundary PEC field fills (only used when eb_bc_direct_fill
     *  is false) */
    amrex::Real m_eb_bc_rtol = 1.e-4;
    /** Sweep cap of the embedded-boundary PEC field fills: maximum Jacobi
     *  iterations of the band relaxation, and maximum cascade sweeps over
     *  ill-posed targets in the direct fill */
    int m_eb_bc_max_iters = 10;
    /** If true (default), the embedded-boundary PEC field fills use a
     *  single-pass mirrored interpolation over unmasked (solution-domain)
     *  values; if false, the iterative Jacobi band relaxation */
    bool m_eb_bc_direct_fill = true;
    /** Image parity of the embedded-boundary deposit fold: true (default)
     *  folds with the PEC image signs (charge and tangential current
     *  subtracted, normal current added); false uses the reflecting-wall
     *  parities (mass-conserving for a wall-supported column) */
    bool m_eb_fold_pec = true;
    /** Parity of the embedded-boundary rho mirror fill: true (default) is
     *  the odd reflection (Dirichlet 0 at the wall); false is the even
     *  reflection (Neumann: the wall supports the column) */
    bool m_eb_rho_dirichlet = true;

    /** Optional Marder-like diffusive clean of the small curved-wall div(B) and
     *  div(J_total) the pointwise mirror injects (see MarderCleanDivergence). alpha=0 (default)
     *  disables each; applied in a near-wall band of width divb_clean_band_cells
     *  for divb_clean_iters sweeps after the respective EB fill. The J clean acts
     *  on the TOTAL (Ampere) current hybrid_current_fp_plasma only, never the
     *  deposited ion-species current (only the total obeys div J = 0). */
    amrex::Real m_divb_clean_alpha = 0.0;
    amrex::Real m_divj_clean_alpha = 0.0;
    int m_divb_clean_iters = 5;
    amrex::Real m_divb_clean_band_cells = 4.0;
    /** Apply the div(B)/div(J) clean once per full step (cheap) instead of every
     *  RKF45 substage (default). The per-step clean removes the divergence
     *  accumulated over the step before the next step begins. */
    bool m_divb_clean_per_step = false;

    /** Dimensionless damping factor of the transitional Marder divergence
     *  cleaning; 0 (default) disables the correction. The discrete grad(div)
     *  operator scales as 1/min(dx)^2, so the applied factor is
     *  alpha * min(dx)^2 (CFL-like stability cap at 0.1). */
    amrex::Real m_marder_alpha = 0.0;
    /** Maximum number of fixed-point iterations per Marder application */
    int m_marder_max_iterations = 50;
    /** Relative tolerance on the masked divergence-error L2 norm */
    amrex::Real m_marder_rtol = 1.e-3;
    /** Absolute tolerance on the masked divergence-error L2 norm */
    amrex::Real m_marder_atol = 1.e-30;
    /** Apply the correction every N substep E evaluations (only at
     *  marder_correction_level = all_substeps) */
    int m_marder_substep_interval = 1;
    /** Divergence target of the Marder correction */
    MarderTarget m_marder_target = MarderTarget::Ohm;
    /** Cadence at which the Marder correction is applied */
    MarderLevel m_marder_level = MarderLevel::AllSubsteps;
    /** Substep E-evaluation counter for the interval cadence; reset at the
     *  start of every hybrid field advance */
    int m_marder_substep_counter = 0;

    /** Epoch counter for the fields entering the Marder divergence target
     *  (Pe and the rho the call sites pass); bumped by the hybrid field
     *  advance whenever either changes, so the per-stage target of the
     *  grad_pe_only mode can be cached between bumps */
    long m_marder_field_epoch = 0;
    /** Cached grad_pe_only divergence target, one per level (valid while
     *  the epoch stamp and source-rho identity match) */
    mutable amrex::Vector<amrex::MultiFab> m_marder_target_cache;
    mutable amrex::Vector<long> m_marder_target_cache_epoch;
    mutable amrex::Vector<const void*> m_marder_target_cache_rho;

    /** Cached embedded-boundary fill classification for the edge fields
     *  (current density and Ohm's-law E), one entry per level; reset on
     *  regrid. Mutable: the cache is (re)built inside const solver calls. */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_E;
    /** Cached embedded-boundary fill classification for the recovered
     *  magnetic field, one entry per level; reset on regrid */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_B;
    /** Cached EB fill classification for the Ampere plasma current, one entry
     *  per level; reset on regrid. Separate from m_eb_bc_status_E because the
     *  isotropic hyper-resistivity Laplacian reads the plasma current at its
     *  diagonal/corner neighbors, so this fill widens the mirror band
     *  (m_eb_fill_band_cells) -- a wider classification than the edge-field
     *  cache, which must stay at the one-cell band shared with the deposit
     *  fold. Only used when isotropic hyper-resistivity is enabled. */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_Jplasma;
    /** Mirror-fill band width (in cells) for the plasma-current EB fill, and --
     *  PINNED to it -- the electric-field (Efield_fp) EB fill. Value:
     *  sqrt(AMREX_SPACEDIM) when the isotropic hyper-resistivity Laplacian is
     *  enabled (so its diagonal/corner cut edges are mirror-filled, not zeroed),
     *  else 1. The E fill is tied to this same band because in the cut/covered
     *  region E = eta*J: filling E in cells BEYOND where J is filled leaves E != 0
     *  where J = 0, an inconsistent source that the curl(E)->B->curl(B)->J->E loop
     *  amplifies into a non-finite blow-up. So the E fill band must not exceed the
     *  J fill band; pinning them equal keeps E and J consistent at the wall. */
    amrex::Real m_eb_fill_band_cells = 1.0;
    /** Mirror-fill band width (in cells) for the magnetic-field (Bfield_fp) EB
     *  fill. The filled B couples back into the solution only through curl/B
     *  reads; with the isotropic corner-curl E correction on, that composite
     *  reach is 2 cells, so widening this band to >=3 pushes the level-set
     *  mirror's div(B) jump out of the solution-coupling zone (into the zeroed
     *  deep interior, where it is causally inert). Runtime knob
     *  hybrid_pic_model.eb_b_fill_band_cells; default 1 (legacy behavior). */
    amrex::Real m_eb_b_fill_band_cells = 1.0;

    /** Gpu Vector with index type of the Jx multifab */
    amrex::GpuArray<int, 3> Jx_IndexType;
    /** Gpu Vector with index type of the Jy multifab */
    amrex::GpuArray<int, 3> Jy_IndexType;
    /** Gpu Vector with index type of the Jz multifab */
    amrex::GpuArray<int, 3> Jz_IndexType;
    /** Gpu Vector with index type of the Bx multifab */
    amrex::GpuArray<int, 3> Bx_IndexType;
    /** Gpu Vector with index type of the By multifab */
    amrex::GpuArray<int, 3> By_IndexType;
    /** Gpu Vector with index type of the Bz multifab */
    amrex::GpuArray<int, 3> Bz_IndexType;
    /** Gpu Vector with index type of the Ex multifab */
    amrex::GpuArray<int, 3> Ex_IndexType;
    /** Gpu Vector with index type of the Ey multifab */
    amrex::GpuArray<int, 3> Ey_IndexType;
    /** Gpu Vector with index type of the Ez multifab */
    amrex::GpuArray<int, 3> Ez_IndexType;
};

/**
 * \brief
 * This struct contains only static functions to compute the electron pressure
 * using the particle density at a given point and the user provided reference
 * density and temperatures.
 */
struct ElectronPressure {

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real get_pressure (amrex::Real const n0,
                                     amrex::Real const T0,
                                     amrex::Real const gamma,
                                     amrex::Real const rho) {
        return n0 * T0 * std::pow((rho/PhysConst::q_e)/n0, gamma);
    }
};

#endif // WARPX_HYBRIDPICMODEL_H_
