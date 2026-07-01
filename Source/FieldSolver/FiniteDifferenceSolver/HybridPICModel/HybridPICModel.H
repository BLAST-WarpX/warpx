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
     * Take the once-per-half-step snapshot of the frozen covered/band B for
     * conformal_b_curl_fill_freeze: run the b-curl-fill on the supplied step-entry
     * B^n and snapshot the result into m_eb_b_curl_freeze_buf, so each later RKF45
     * substage can re-stamp the (static) covered values instead of re-evaluating
     * the nonsmooth fill from the live substage B. No-op unless the freeze flag is
     * on (and a staggered conformal-EB run). Called once per RKF45 half-step.
     */
    void SnapshotConformalBCurlFreeze (
        ablastr::fields::VectorField const& Bfield,
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

    /** conformal_ect_lsq Phase 1: OVERWRITE the near-wall (cut-band) Ampere-current
     *  edges with a noise-free weighted-least-squares CENTROID reconstruction of the
     *  standard Yee curl. For each uncovered wall-band edge (signed distance to EB in
     *  (~0.05h, ~2h)) of family x/y/z, fit a degree-2 polynomial to the just-computed
     *  curl over the surrounding genuine-fluid edges of the same family (radius 2,
     *  the validated default), weighted by an anisotropic Gaussian in the local
     *  wall-normal frame, and evaluate it at the edge's fluid centroid (the
     *  edge_cent_offset along-edge shift). This recovers a converging wall current
     *  where the bare masked curl is O(1)-wrong. Reads the standard curl from a
     *  snapshot (so the overwrite uses the original values), then matched cut-metric
     *  div-clean restores divergence-consistency. 3D Cartesian, staggered; no-op
     *  otherwise. */
    void ApplyConformalLSQOverwrite (
        ablastr::fields::VectorField const& Jfield,
        int lev) const;

    /** conformal_ect_lsq Phase 2b: matched cut-metric divergence clean of the Ampere
     *  current. Builds the node-dual open-facet area metric (cached, geometry-only),
     *  then enforces the cut-metric divergence flux-balance D J = 0 by a matched
     *  Helmholtz projection: solve the area-weighted nodal Laplacian K phi = D J (K =
     *  D . grad_node, symmetric SPD) by Jacobi-preconditioned CG and correct
     *  J -= grad_node(phi). The standard Yee curl is identically divergence-free in the
     *  full interior (D.curl = 0), so D J is nonzero only in the cut wall band and the
     *  correction is localized there. This restores divergence-consistency to the
     *  accurate Phase-1 wall current, clearing the near-wall RKF45 stiffness without
     *  degrading it. 3D Cartesian, staggered; no-op otherwise. */
    void ApplyConformalDivClean (
        ablastr::fields::VectorField const& Jfield,
        int lev) const;

    /** Build (once per level, lazily; invalidated on regrid) the node-dual open-facet
     *  area fractions used by the conformal_ect_lsq Phase-2b div-clean. Ai[c] is
     *  edge-staggered (same as J component c): the open fluid fraction of the dual
     *  facet perpendicular to edge c, obtained by sub-sampling the nodal distance_to_eb
     *  level set (the AMReX cell-EB metrics are the wrong staggering for a node-centred
     *  dual complex). Deep-fluid facets are 1, deep-covered 0, only the band is
     *  sub-sampled. */
    void BuildConformalDualAreas (int lev) const;

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
    /** If true, compute the covered-B curl fill (conformal_b_curl_fill) ONCE per
     *  RKF45 half-step from the step-entry B^n and HOLD those covered/band values
     *  fixed across all RKF45 substages, instead of re-evaluating the nonsmooth
     *  curved-wall extrapolation from the live substage B at every substage. The
     *  per-substage re-evaluation injects a near-wall radial-B feedback that grows
     *  with a building reversal field and collapses the adaptive substep, while a
     *  frozen (static) near-wall covered B integrates cleanly: this pulls the
     *  nonsmooth fill out of the stiff inner integrator. Requires
     *  conformal_b_curl_fill; opt-in (default off -> byte-identical, the fill is
     *  recomputed every substage as before). */
    bool m_conformal_b_curl_fill_freeze = false;
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
    /** If true, compute the Ampere current J = curl(B)/mu0 with the flux-weighted
     *  ("Form A") conformal-EB curl (CalculateCurrentAmpereCartesianECT): each B
     *  value in the Yee curl is scaled by its open cut-face-area fraction
     *  (face_areas / full face area), so J is a signed sum of open-face fluxes and
     *  is discretely divergence-consistent across the cut wall with no separate
     *  covered-B fill. Covered faces (face_areas = 0) drop out automatically, so the
     *  conformal_b_curl_fill mirror is skipped when this is on. Reduces byte-for-byte
     *  to the masked Yee curl on interior edges. Staggered (Yee) grid only; opt-in
     *  (default off -> byte-identical, the standard masked Yee curl is used). */
    bool m_conformal_ect_j = false;
    /** If true, compute the Ampere current J = curl(B)/mu0 with the accurate
     *  conformal-EB scheme ("Path 1"): the existing PEC covered-B fill (normal odd
     *  -> B_n=0, tangential even -> Neumann; already correct) feeds the standard Yee
     *  curl, then the near-wall (cut-band) J edges are OVERWRITTEN by a noise-free
     *  weighted-least-squares centroid reconstruction of that curl, evaluated at each
     *  edge's fluid centroid in the local wall-normal frame. This recovers an accurate
     *  (converging) wall current where the bare masked curl is O(1)-wrong, while the
     *  matched cut-metric div-clean (grad := -D^T of the centroid-aware cut divergence)
     *  enforces divergence-consistency. Unlike conformal_ect_j (Form A, div-free but
     *  O(1)-inaccurate at the wall), this is accurate AND div-consistent. Staggered
     *  (Yee) grid, 3D Cartesian only; opt-in (default off -> byte-identical). */
    bool m_conformal_ect_lsq = false;
    /** conformal_ect_lsq Phase-2b div-clean controls. Max Jacobi-PCG iterations for the
     *  matched cut-metric Laplacian solve. Default 0 = OFF: the matched clean zeros the
     *  cut-metric (flux-form) divergence, but the RKF45 stiffness responds to the plain
     *  Yee (point-J) divergence, which is kept small by the covered-J mirror fill, not
     *  this clean. Opt-in refinement only. */
    int m_conformal_divclean_iters = 0;
    /** Relative residual tolerance for the Phase-2b div-clean CG solve. */
    amrex::Real m_conformal_divclean_rtol = 1.e-6;
    /** Number of points per axis when sub-sampling distance_to_eb for the Phase-2b
     *  node-dual open-facet area fractions (M x M per facet). */
    int m_conformal_divclean_subsample = 16;
    /** conformal_ect_lsq sliver gate: at a wall-band edge whose stencil has fewer than
     *  this fraction of genuine-fluid taps, the full 3D least-squares fit is
     *  ill-conditioned (one-sided taps) and produces spurious wall-current spikes that
     *  seed an Ohm->Faraday growth. Such edges fall back to a robust wall-normal linear
     *  reconstruction (average + off-wall delta) instead. */
    amrex::Real m_conformal_lsq_sliver_frac = 0.4;
    /** If true (default), the hybrid Hall term's nodal J x B interpolation excludes
     *  covered edges/faces (EB-aware, mask-renormalized) so covered-cell field values do
     *  not pollute the near-wall J x B. Set false to recover the plain (polluting)
     *  averaging for A/B. Only affects conformal-EB staggered runs. */
    bool m_eb_hall_mask = true;
    /** Diagnostic/experimental: if nonzero, the Phase-2b clean targets the plain
     *  Cartesian (Yee) divergence (binary fluid-area operator, 1 in fluid / 0 covered)
     *  instead of the cut-metric flux-balance divergence. The Yee divergence is what the
     *  masked Marder clean reduces and what the RKF45 stiffness responds to; the
     *  cut-metric (flux-form) divergence is a different object. */
    int m_conformal_divclean_cartesian = 0;
    /** Cached node-dual open-facet area fractions per level, and a per-level built flag.
     *  A single NODAL MultiFab with 3 components: component c is the open fluid fraction
     *  of the node's +c-direction dual facet, which sits at the same index as the c-edge
     *  current J[c] (so the cut-metric divergence is an area-weighted node-from-edges
     *  stencil). Geometry-only; rebuilt on regrid. */
    mutable amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_eb_dual_area;
    mutable amrex::Vector<int> m_eb_dual_area_built;
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
    /** Frozen covered/band B-field snapshot for conformal_b_curl_fill_freeze,
     *  one (x,y,z) MultiFab triple per level. Filled once per RKF45 half-step
     *  from the step-entry B^n (the b-curl-fill output on B^n); on every later
     *  substage the cells the fill controls (status != S_SOLUTION) are re-stamped
     *  from this buffer instead of recomputing the fill from the live substage B.
     *  Mutable: (re)written inside const solver calls. */
    mutable amrex::Vector<std::array<amrex::MultiFab, 3>> m_eb_b_curl_freeze_buf;
    /** True while the frozen covered-B snapshot is valid and should be re-stamped
     *  on the current RKF45 substage's CalculatePlasmaCurrent (set at the half-step
     *  entry by SnapshotConformalBCurlFreeze, after the entry fill is taken). */
    mutable bool m_eb_b_curl_freeze_valid = false;
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

    /** Lower clamp (floor) on the wall-NORMAL reflection weight of the
     *  covered-B mirror fill (the b-curl-fill, normal-odd magnetic parity).
     *  The unclamped normal weight is g.s/d_im, which for the quadratic gather
     *  is -1: the masked covered B_normal (0) is replaced by -B_normal, a full
     *  sign reversal across one cell. curl(B) then sees a ~2x steeper near-wall
     *  gradient -> ~2x near-wall J wherever there is a near-wall RADIAL B (the
     *  liftoff's dense shell + reversal), which stalls the RKF45 Faraday solve.
     *  Clamping the normal weight to >= 0 (floor = 0) drives the covered
     *  B_normal toward 0 (the physical PEC condition B_normal -> 0 at the wall)
     *  instead of reversing it; the tangential even reflection (the desired
     *  m=4-killing smoothing) is untouched. Runtime knob
     *  hybrid_pic_model.eb_b_fill_normal_weight; default -1e30 = disabled /
     *  no clamp = byte-identical legacy behavior (g.s/d_im). */
    amrex::Real m_eb_b_fill_normal_weight = -1.e30;

    /** Near-wall stability blend for the covered-B b-curl-fill, applied only at
     *  CUT faces (B update mask != 0: faces with a positive area fraction for
     *  which the conformal ECT B push already computed a value). At a cut face
     *  the quadratic mirror DISCARDS that stabler ECT-solved B and writes a
     *  steeper near-wall extrapolation; on a dense plasma shell pinned against a
     *  smooth wall (the liftoff) this injects a near-wall radial-B feature that
     *  grows super-exponentially and collapses the RKF45 substep. With this knob
     *  in (0,1] the cut face is instead written as
     *  (1-blend)*B_mirror + blend*B_ECT (B_ECT = the value EvolveB already wrote
     *  at the fill point): blend=1 keeps the pure ECT cut-face B (the stable
     *  staircase-equivalent), blend=0 keeps the pure mirror. Fully-covered faces
     *  (mask == 0, no ECT value) always take the full mirror, so the gentle
     *  regime (eb_diffusion) keeps the full 2nd-order mirror there. Runtime knob
     *  hybrid_pic_model.conformal_b_curl_fill_blend; default 0 = full mirror =
     *  byte-identical. No effect without conformal_b_curl_fill. */
    amrex::Real m_conformal_b_curl_fill_blend = 0.0;

    /** Relative cap on the cut-face covered-B mirror's deviation from the
     *  conformal-ECT value. The quadratic fluid gather can overshoot the ECT
     *  value wildly on a stiff dense shell (oscillatory tap weights) while
     *  staying close to it in the gentle regime; this clamps the deviation to
     *  conformal_b_curl_fill_clamp * max(|B_ECT|, |B_image|) so the gentle
     *  2nd-order correction passes unclamped while the stiff overshoot is
     *  bounded. Runtime knob hybrid_pic_model.conformal_b_curl_fill_clamp;
     *  default 0 = no clamp = byte-identical. No effect without
     *  conformal_b_curl_fill. */
    amrex::Real m_conformal_b_curl_fill_clamp = 0.0;

    /** Option-2 concave re-entrant-corner skip for the covered-B mirror. The
     *  b-curl-fill mirror is benign on a smooth wall but, at a concave step-down
     *  corner ring, writes a sub-bias covered tangential B across the step plane
     *  while the fluid neighbour is 0; the one-cell cross-wall B jump the Ampere
     *  curl J = curl(B)/mu0 amplifies into a ~1e6 A/m^2 spike that crashes the
     *  RKF45 substepper. A magnitude clamp cannot bound a gradient, so instead a
     *  validated three-clause detector (a wall-normal bend > 30 deg, a
     *  load-bearing level-set radius jump to a fluid curl-difference neighbour
     *  across an opposite-sign phi, and the mirror image within reach) flags those
     *  measure-zero cells S_CORNER in the fill classification, and the fill SKIPS
     *  them so they keep their stable pre-fill / OFF value (reverting the Ampere
     *  curl there to the stable b-curl-fill-OFF behaviour). Runtime knob
     *  hybrid_pic_model.conformal_b_curl_fill_corner_skip; default false = no skip
     *  = byte-identical. No effect without conformal_b_curl_fill. */
    bool m_conformal_b_curl_fill_corner_skip = false;

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
