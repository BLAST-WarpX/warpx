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
#include <ablastr/utils/text/IntervalsParser.H>

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

    /** Constitutive-PEC zeroing of an edge-staggered field on the staggered
     *  conformal (ECT) path: zero every masked (fully covered) edge and every
     *  cut edge (open length < full). Used for the Ohm's-law E and the Ampere
     *  current when conformal_pec_zero_ej is set (see that knob's docs). */
    void ZeroConductorEdges (
        ablastr::fields::VectorField const& field,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update,
        int lev) const;

    /** Constitutive-PEC zeroing of a face-staggered field on the staggered
     *  conformal (ECT) path: zero every FULLY COVERED face (open area = 0).
     *  Cut faces keep their value (their open part carries real flux). Used
     *  for the external B (curl A) when conformal_pec_zero_ej is set: the
     *  conductor excludes the time-varying external flux, and leaving the
     *  through-wall ramp in the covered faces feeds a growing,
     *  staircase-patterned B reservoir to every open-side curl stencil. */
    void ZeroCoveredFaces (
        ablastr::fields::VectorField const& field,
        int lev) const;

    /**
     * \brief Optional Marder-like diffusive divergence clean of a staggered or
     * nodal vector field. Iterates field += alpha_scaled * grad(div(field))
     * restricted to a near-wall geometric band (distance_to_eb <=
     * band_cells*h), re-applying the embedded-boundary PEC fill (with the
     * field's own parity) each sweep. The correction is a pure gradient, so
     * curl(field) -- and hence J=curl(B) / the physics it feeds -- is
     * unchanged to round-off away from the band cutoffs; only the spurious
     * curved-wall divergence injected by the mirror fill is dissipated. With
     * the inner cutoffs at 0 (the wall-layer mode) the clean reaches the
     * first fluid layers, where a sharp re-entrant corner (e.g. a wall radius
     * step) otherwise pumps an exponentially growing div(B). No-op when
     * alpha<=0 or no EB. 3D/XZ Cartesian only (the gradient is the adjoint of
     * the Cartesian divergence; ReadParameters aborts if the knobs are set on
     * other geometries).
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
        int lev,
        bool cut_metric = false) const;

    /** Once-per-step driver for the div(B)/div(J_total) clean: cleans Bfield_fp
     *  (magnetic parity) and the total Ampere current hybrid_current_fp_plasma
     *  (electric parity) on every level via MarderCleanDivergence. Called from
     *  the hybrid field advance after the plasma-current update and before the
     *  external-field add-back, so it acts on the plasma field only. No-op
     *  unless the respective alpha>0. */
    void MarderCleanFieldsPerStep () const;

    /** Density-banded Marder clean of the Ohm's-law E at the n_floor seam
     *  (staggered grids; called per substage from FieldPush between the Ohm
     *  solve and the Faraday push). Iterates E += alpha_scaled * grad(div(E))
     *  restricted to edges whose endpoint-averaged density is at or below
     *  dive_seam_band * rho_floor, diffusing the vector-inconsistent
     *  (divergence-carrying) E content the per-component staggered sampling
     *  injects at the moving plasma/vacuum seam. The nodal div / edge grad pair
     *  is mutually adjoint, so the update is strictly dissipative of the
     *  divergence norm and preserves curl(E) to round-off away from the window
     *  edge. No-op unless dive_seam_alpha > 0. 3D/XZ Cartesian only. */
    void MarderCleanESeam (
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::IntVect ng, std::optional<bool> nodal_sync) const;

    void BfieldEvolve (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        int step, amrex::Real dt_half, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolve (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        int step, amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolveRK4 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        std::array<amrex::MultiFab, 3>& B_old,
        amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    amrex::Real BfieldEvolveRKF45 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        std::array<amrex::MultiFab, 3>& B_old,
        amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void FieldPush (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    /**
     * \brief
     * Function to calculate the electron pressure using the simulation charge
     * density. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     */
    void CalculateElectronPressure () const;
    void CalculateElectronPressure (int lev) const;

    /**
     * \brief Warn (once) if plasma sits within the particle shape order of the
     * embedded-boundary wall. Deposition reaches nox cells, so plasma that close
     * lays current onto wall-adjacent edges and drives near-wall current spikes
     * (and an unmodelled sub-grid sheath). Recommends scraping >= nox*dx off the
     * wall. Cheap: runs a single masked reduction on the first call, then latches.
     */
    void WarnIfPlasmaAgainstWall (int lev) const;

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

    /** Check if rkf45 should be used */
    [[nodiscard]] bool DoRKF45(int step) const { return m_rkf45_intervals.contains(step); }

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active) */
    int m_substeps = 10;

    /** Intervals to use RKF45 integrator and to update substeps parameter */
    ablastr::utils::text::IntervalsParser m_rkf45_intervals;
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

    bool m_holmstrom_vacuum_region = false;


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
    /** Evaluate the hyper-resistivity Laplacian of the plasma current with the
     *  isotropic Mehrstellen (2D) / Patra-Karttunen (3D) stencils instead of
     *  the cross stencil, removing the cos(4*theta) grid anisotropy of its
     *  damping rate (fully isotropic on cubic cells; Cartesian only). */
    bool m_isotropic_hyper_resistivity = false;
    /** If true (default), on the staggered conformal (ECT) path the
     *  isotropic stencils are DOWNGRADED to the standard compact ones on
     *  edges within a corner reach (sqrt(AMREX_SPACEDIM)+0.5 cells) of the
     *  level set: the isotropic stencils read the full 3^3 neighborhood, and
     *  near the wall those diagonal/corner reads land on EB-touching edges
     *  that the constitutive PEC zeroes (and that no mirror fill sets on
     *  this path), which would inject spurious wall-adjacent E. Engages the
     *  shared per-point fallback plumbing of the isotropic-operators PR
     *  (see m_isotropic_eb_compact_fallback). No effect off the staggered
     *  conformal path (the mirror-filled paths instead widen the fill bands
     *  to the corner reach; see m_eb_fill_band_cells). */
    bool m_isotropic_hyper_wall_compact = true;

    /** Isotropize the in-plane resistive diffusion of the out-of-plane B
     *  component (Bz in 3D, By in 2D XZ): a corner-curl correction added to
     *  the in-plane E components makes the Faraday curl of the resistive E
     *  emit the isotropic (Mehrstellen) Laplacian instead of the cross
     *  stencil. This preserves div(B) exactly since the correction enters
     *  only through E (Cartesian only). */
    bool m_isotropic_resistivity = false;

    /** Evaluate the electron-pressure gradient in Ohm's law with the
     *  transverse-smoothed (isotropized) staggered difference instead of the
     *  plain two-point stencil, removing the cos(4*theta) grid anisotropy of
     *  |grad Pe| (fully isotropic on cubic cells; Cartesian only). Only
     *  affects the E used for the particle push and diagnostics: the
     *  Faraday/B-integration path omits grad(Pe) entirely. */
    bool m_isotropic_gradient = false;

    /** Near an embedded boundary, fall back per point from the isotropic
     *  stencils to the standard compact ones within a corner reach of the
     *  level set, where the diagonal reads are not maintained by the EB
     *  update masks. Default true; a hybrid EB boundary layer that
     *  mirror-fills the wide-stencil bands (so the diagonal reads stay
     *  valid) may disable it. On this branch the hybrid EB boundary layer
     *  mirror-fills the J/B/Pe bands to the diagonal stencil reach, so the
     *  fallback defaults to OFF (opt back in for A/B isolation). */
    bool m_isotropic_eb_compact_fallback = false;

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

    /** If true, use the conformal embedded-boundary wall treatment for the
     *  B-field push instead of the stair-step approximation. On a staggered
     *  (Yee) grid this is the enlarged-cell technique (ECT) Faraday update
     *  (Xiao and Liu, IEEE MWCL 2004) on cut faces/edges; on a collocated grid
     *  it is the level-set masked nodal Faraday with the mirror B fill.
     *  Requires embedded boundaries and 3D or 2D (XZ) Cartesian geometry. */
    bool m_use_conformal_eb = false;
    /** If true, apply the along-edge curvature correction to the conformal-ECT
     *  Faraday circulation: in EvolveECTRho each cut edge's E is Taylor-shifted
     *  from the full-edge center to the centroid of its uncovered segment before
     *  forming the per-face EMF. Without it the circulation is a 1st-order
     *  midpoint quadrature over the curved contour, capping the conformal B push
     *  at ~1st order at a curved wall even when curl(B)->J is already 2nd order.
     *  Uses the edge_cent_offset geometry field. Requires use_conformal_eb on a
     *  staggered (Yee) grid; opt-in (default off -> byte-identical, since the
     *  shift is zero on uncut/covered edges). */
    bool m_conformal_ect_curvature = false;
    /** If true, compute the Ampere current J = curl(B)/mu0 with the flux-weighted
     *  ("Form A") conformal-EB curl (CalculateCurrentAmpereCartesianECT): each B
     *  value in the Yee curl is scaled by its open cut-face-area fraction
     *  (face_areas / full face area), so J is a signed sum of open-face fluxes and
     *  is discretely divergence-consistent across the cut wall with no separate
     *  covered-B fill. Covered faces (face_areas = 0) drop out automatically, so no
     *  covered-B mirror fill is needed. Reduces byte-for-byte
     *  to the masked Yee curl on interior edges. Requires use_conformal_eb on a
     *  staggered (Yee) grid; opt-in (default off -> byte-identical, the standard
     *  masked Yee curl is used). */
    bool m_conformal_ect_j = false;
    /** Experiment knob: keep the covered-J mirror fill when conformal_ect_j is
     *  on (the default skips it, since the mirror injects div(J) into Form A's
     *  div-consistent wall current). Isolates the flux-weighted READ from the
     *  loss of the mirror's smoothing in the liftoff wall-instability ladder. */
    bool m_conformal_ect_j_keep_mirror = false;
    /** RESEARCH KNOB (opt-in, default false -- enabling it is explicitly
     *  UNSTABLE today). If true, the staggered conformal (ECT) path imposes
     *  the wall on the Ohm's-law E geometrically, like the EM ECT solver: the
     *  E fill rewrites only the masked (fully covered, zero-open-length)
     *  edges, and LIVE cut edges keep their Ohm value for the circulation,
     *  whose open-length weights already exclude the covered portions.
     *
     *  Measured (vacuum ECT liftoff, res 72: pure eta*J + external ramp): the
     *  explicit composite map [l,S-circulation o eta*curl] has local gain > 1
     *  at small-S cut cells, so with this on the RKF45 collapses within STEP
     *  ONE from a round-off seed (~e-fold per substep) -- with either the raw
     *  masked Yee curl or the flux-weighted Form-A curl supplying J. The
     *  default pointwise mirror overwrite of covered-center cut edges is what
     *  damps that map enough to reach ~900 steps; it is load-bearing for the
     *  explicit scheme, not merely a BC choice. A stable geometric-PEC wall
     *  needs the wall-local implicit solve (or a genuinely higher-order
     *  reconstructed J), not just metric-consistent explicit reads. No effect
     *  off the staggered conformal path. */
    bool m_conformal_e_geometric_pec = false;
    /** On the staggered conformal (ECT) path, impose the PEC on the Ohm's-law
     *  E and on the Ampere current CONSTITUTIVELY: at a perfect conductor
     *  E = 0, and the hybrid cannot support wall (surface) currents, so with
     *  a standoff keeping the plasma off the metal there is no physics on
     *  EB-touching edges at all. Set E = 0 and J = 0 on every masked (fully
     *  covered) edge and on EVERY cut edge (open length < full length),
     *  instead of the pointwise level-set mirror extrapolations (which feed
     *  near-wall values into the 1/S-amplified circulation). Cut faces then
     *  evolve only through their fully-open edges: tangential E = 0 at the
     *  wall, imposed at the cut-edge level (a first-order wall, traded for
     *  unconditional boundedness). The density-based resistivity parser
     *  cannot express this (rho sits at the floor in the conductor AND the
     *  vacuum gap, so the conductor accidentally gets eta_vac -- the largest
     *  eta in the problem); this mask is the geometry-aware eta -> 0 it
     *  cannot write. Takes precedence over conformal_e_geometric_pec and the
     *  covered-J mirror. */
    bool m_conformal_pec_zero_ej = false;
    /** If true (default), the hybrid Hall term's nodal J x B interpolation excludes
     *  covered edges/faces (EB-aware, mask-renormalized) so covered-cell field values do
     *  not pollute the near-wall J x B. Set false to recover the plain (polluting)
     *  averaging for A/B. Only affects conformal-EB staggered runs. */
    bool m_eb_hall_mask = true;
    /** If true (opt-in), evaluate the resistivity and hyper-resistivity parsers
     *  ONCE per node -- from the nodal rho and the Hall-consistent nodal J/B --
     *  and interpolate the resulting eta fields to the staggered E locations
     *  (like the nodal enE/Hall term), instead of evaluating the parsers
     *  pointwise at each staggered component location. Pointwise evaluation
     *  samples a steep eta(rho) transition at three different positions per
     *  cell, giving the E components inconsistent resistivities across the
     *  plasma/vacuum (n_floor) seam -- a spurious-E source there. No-op on
     *  collocated grids and for rho/J-independent resistivities. Cartesian
     *  only. */
    bool m_eta_nodal_interp = false;
    /** If > 0 (opt-in), smooth the holmstrom vacuum switch with an
     *  above-floor power-law window: on rho in [rho_floor,
     *  holmstrom_blend_width*rho_floor] the Hall/pressure (divided-by-rho)
     *  content of the Ohm's-law E is weighted by
     *  w = ((rho-rho_floor)/((width-1)*rho_floor))^pow and blended from the
     *  vacuum value (E_ext or 0), so the stiff Hall physics fades in over the
     *  low-density band instead of switching on at full strength one cell
     *  above the floor. Below the floor the legacy vacuum branch holds
     *  exactly (purely diffusive E). <= 0 keeps the legacy binary switch.
     *  Only active with holmstrom_vacuum_region; Cartesian only. */
    amrex::Real m_holmstrom_blend_pow = 0.0;
    /** Upper edge of the blend window in units of the density floor: the
     *  Hall/pressure content ramps in over rho in [rho_floor,
     *  width*rho_floor]. Below the floor the legacy vacuum branch holds
     *  exactly (Hall fully off). Must be > 1 for the blend to be active. */
    amrex::Real m_holmstrom_blend_width = 2.0;
    /** Sampling mode of the density that decides the holmstrom vacuum switch
     *  and the blend weight (the physics division by rho and the resistivity
     *  evaluation are unchanged). 0 = "edge" (default): each staggered E
     *  component uses its own half-cell-shifted edge average -- the three
     *  components of a cell can take inconsistent vacuum/plasma branches along
     *  the moving plasma/vacuum seam (a grid-C4-patterned spurious-E source
     *  there). 1 = "node": the minimum of the nodal rho at the edge endpoints
     *  (vacuum-favoring: the stiff Hall branch runs only where every endpoint
     *  is above the floor; the plasma-favoring max variant was measured to
     *  destabilize -- full Hall on edges whose own density is sub-floor).
     *  2 = "cell": rho averaged to the CELL CENTER decides for all three E
     *  components of that index -- a single piecewise-constant-per-cell
     *  decision field with no per-component half-shifts. Cartesian only. */
    int m_holmstrom_switch_mode = 0;
    /** Density-banded Marder clean of the Ohm E at the n_floor seam
     *  (MarderCleanESeam): fraction of the explicit-diffusion cap applied per
     *  substage. 0 (default) = off. Staggered 3D/XZ Cartesian only. */
    amrex::Real m_dive_seam_alpha = 0.0;
    /** Sweeps of the seam E clean per substage. */
    int m_dive_seam_iters = 1;
    /** Upper edge of the seam clean window in units of the density floor:
     *  the correction applies on edges with rho <= dive_seam_band*rho_floor. */
    amrex::Real m_dive_seam_band = 4.0;
    /** If true, skip the collocated conformal magnetic-field wall treatment
     *  entirely (no EB B fill): the unmasked nodal Faraday curl is left as-is.
     *  This reproduces the pre-treatment baseline and isolates the wall fill as
     *  the source of the grid-locked m = 4 perturbation. Diagnostic only -- it
     *  leaves spurious fields at the wall, so it is not for production runs. */
    bool m_conformal_b_off = false;
    /** If true, apply the collocated-style direct level-set mirror wall fill to the
     *  STAGGERED (Yee) Bfield_fp after each Faraday push, in place of the ECT
     *  enlarged-cell face borrowing. Run with the standard masked Yee Faraday
     *  (use_conformal_eb off) + this straight mirror to get the collocated wall
     *  treatment on a staggered grid, avoiding the ECT face-extension / cross-box
     *  seam. Opt-in (default off -> byte-identical); intended for staggered runs
     *  with the plasma held off the wall (a standoff), which removes the dense
     *  near-wall radial B that made the pointwise Yee mirror spike. */
    bool m_eb_b_straight_mirror = false;

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
    /** Parity of the electron-pressure embedded-boundary fill at a PEC wall:
     *  true (default) is the odd reflection (Dirichlet 0 -> Pe vanishes at the
     *  wall, so grad(Pe) supports the allowable radial E a PEC sustains via
     *  surface charge); false is the even reflection (Neumann: zero normal
     *  gradient). A physical sheath/wall-function is not modelled here. */
    bool m_eb_pe_dirichlet = true;
    /** Latch so the near-wall plasma warning is emitted at most once. */
    mutable bool m_checked_plasma_at_wall = false;

    /** Cached embedded-boundary fill classification for the deposited edge
     *  fields (ion current fill + deposit fold), one entry per level; reset on
     *  regrid. Built at the one-cell band: the fold's covered gather set must
     *  equal this fill's write set, and both use the fixed one-cell geometry.
     *  Mutable: the cache is (re)built inside const solver calls. */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_E;
    /** Cached EB fill classification for the Ohm's-law electric field, one
     *  entry per level; reset on regrid. Separate from m_eb_bc_status_E
     *  because the E fill band is pinned to the (possibly widened) plasma
     *  current band m_eb_fill_band_cells, while the deposited-current cache
     *  stays at the one-cell band -- an EBFillStatus cache must never be
     *  shared between calls with different band widths (the first caller
     *  would pin the classification). */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_Eohm;
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
     *  hybrid_pic_model.eb_b_fill_band_cells; default 1 (legacy behavior),
     *  auto-widened to sqrt(2) on 3D/XZ when isotropic_resistivity is enabled
     *  so the corner-curl's in-plane diagonal B taps read mirror values rather
     *  than the zeroed deep interior (user query still overrides). */
    amrex::Real m_eb_b_fill_band_cells = 1.0;

    /** Marder-like diffusive div(B) clean coefficient (fraction of the explicit
     *  grad(div) CFL cap; stable below ~1/6 in 3D, PoC-validated sweet spot
     *  ~0.15). 0 (default) disables. Runtime knob hybrid_pic_model.divb_clean_alpha. */
    amrex::Real m_divb_clean_alpha = 0.0;
    /** Same for the total Ampere current hybrid_current_fp_plasma; dissipates the
     *  spurious near-wall div(J_total) the pointwise mirror injects (see
     *  MarderCleanDivergence). 0 (default) disables. */
    amrex::Real m_divj_clean_alpha = 0.0;
    /** Number of grad(div) sweeps per application of the divergence clean. */
    int m_divb_clean_iters = 5;
    /** Outer cutoff of the divergence-clean band, in cells from the wall. */
    amrex::Real m_divb_clean_band_cells = 4.0;
    /** If true (default), the div(B) clean on the staggered conformal (ECT)
     *  path uses the MATCHED CUT-METRIC operators: the divergence is the
     *  signed sum of open-face fluxes (face_areas-weighted, the invariant the
     *  ECT Faraday actually conserves) and the correction is its exact
     *  negative adjoint (frac-weighted gradient), so the sweep is dissipative
     *  in the flux norm and does not fight the ECT push the way the raw
     *  uniform-metric clean does (raw div at a cut face is O((1-frac)B/h) BY
     *  CONSTRUCTION for an ECT-evolved field). In this mode covered faces
     *  (frac = 0) are never read or written, live cut faces are gated on
     *  frac > 0 rather than the staircase update mask, and the per-sweep
     *  level-set mirror re-fill is skipped (the ECT arm imposes no mirror on
     *  B). Ignored (raw metric) off the staggered conformal path and in 2D. */
    bool m_divb_clean_cut_metric = true;
    /** Inner cutoff (in cells) below which the computed divergence is dropped.
     *  Default 1 trusts the divergence only where its own +/-1 stencil is fully
     *  in the fluid; 0 keeps it on every uncovered node (wall-layer mode). */
    amrex::Real m_divb_clean_inner_div_cells = 1.0;
    /** Inner cutoff (in cells) below which no correction is applied. Default 2
     *  keeps the full +/-2 grad(div) stencil in the fluid (order-preserving on a
     *  smooth wall); 0 corrects every uncovered band node (wall-layer mode, for
     *  a re-entrant-corner divergence instability). */
    amrex::Real m_divb_clean_inner_corr_cells = 2.0;

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
