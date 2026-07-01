/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "HybridPICModel.H"

#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include "EBJBoundary.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Python/callbacks.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "ExternalVectorPotential.H"
#include "WarpX.H"

using namespace amrex;
using warpx::fields::FieldType;

HybridPICModel::HybridPICModel ()
{
    ReadParameters();
}

void HybridPICModel::ReadParameters ()
{
    const ParmParse pp_hybrid("hybrid_pic_model");

    // The B-field update is subcycled to improve stability - the number
    // of sub steps can be specified by the user.
    utils::parser::queryWithParser(pp_hybrid, "substeps", m_substeps);
    if (m_substeps % 2 != 0) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.substeps must be divisible by 2. "
            "The value " + std::to_string(m_substeps) + " is not valid. "
            "Automatically adjusting to " + std::to_string(m_substeps + 1) + ".",
            ablastr::warn_manager::WarnPriority::medium);
        m_substeps += 1;
    }

    utils::parser::queryWithParser(pp_hybrid, "substep_rtol", m_substep_rtol);
    utils::parser::queryWithParser(pp_hybrid, "substep_atol", m_substep_atol);
    utils::parser::queryWithParser(pp_hybrid, "substep_safety", m_substep_safety);
    utils::parser::queryWithParser(pp_hybrid, "substep_max_growth", m_substep_max_growth);
    pp_hybrid.query("max_substep_attempts", m_max_substep_attempts);
    pp_hybrid.query("use_rkf45", m_use_rkf45);

    utils::parser::queryWithParser(pp_hybrid, "holmstrom_vacuum_region", m_holmstrom_vacuum_region);

    // The hybrid model requires an electron temperature, reference density
    // and exponent to be given. These values will be used to calculate the
    // electron pressure according to p = n0 * Te * (n/n0)^gamma
    utils::parser::queryWithParser(pp_hybrid, "gamma", m_gamma);
    if (!utils::parser::queryWithParser(pp_hybrid, "elec_temp", m_elec_temp)) {
        Abort("hybrid_pic_model.elec_temp must be specified when using the hybrid solver");
    }
    const bool n0_ref_given = utils::parser::queryWithParser(pp_hybrid, "n0_ref", m_n0_ref);
    if (m_gamma != 1.0 && !n0_ref_given) {
        Abort("hybrid_pic_model.n0_ref should be specified if hybrid_pic_model.gamma != 1");
    }

    pp_hybrid.query("plasma_resistivity(rho,J,t)", m_eta_expression);
    pp_hybrid.query("plasma_hyper_resistivity(rho,B)", m_eta_h_expression);

    utils::parser::queryWithParser(pp_hybrid, "n_floor", m_n_floor);

    // convert electron temperature from eV to J
    m_elec_temp *= PhysConst::q_e;

    // external currents
    pp_hybrid.query("Jx_external_grid_function(x,y,z,t)", m_Jx_ext_grid_function);
    pp_hybrid.query("Jy_external_grid_function(x,y,z,t)", m_Jy_ext_grid_function);
    pp_hybrid.query("Jz_external_grid_function(x,y,z,t)", m_Jz_ext_grid_function);

    // check if external currents are specified
    if ((m_Jx_ext_grid_function == "0.0") &&
        (m_Jy_ext_grid_function == "0.0") &&
        (m_Jz_ext_grid_function == "0.0"))
    {
        m_has_external_current = false;
    }

    // external fields
    pp_hybrid.query("add_external_fields", m_add_external_fields);

    if (m_add_external_fields) {
        m_external_vector_potential = std::make_unique<ExternalVectorPotential>();
    }

    // conformal (enlarged-cell technique) embedded-boundary Faraday update
    pp_hybrid.query("use_conformal_eb", m_use_conformal_eb);
    pp_hybrid.query("conformal_b_curl_fill", m_conformal_b_curl_fill);
    if (m_conformal_b_curl_fill
        && WarpX::grid_type == ablastr::utils::enums::GridType::Collocated) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.conformal_b_curl_fill is only implemented for a "
            "staggered (Yee) grid; it is ignored on a collocated grid (the nodal "
            "2nd-order covered-B gather is not yet robust).",
            ablastr::warn_manager::WarnPriority::medium);
    }
    if (m_conformal_b_curl_fill) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.conformal_b_curl_fill (the covered-B quadratic mirror "
            "gather run before the Ampere curl) is DEPRECATED and slated for removal: "
            "it is a pointwise, non-divergence-constrained extrapolation that injects "
            "div(B)/div(J) at the wall and has proven unstable. It will be replaced by "
            "the ECT edge-circulation current (ComputeJCartesianECT), which builds "
            "J = curl(B)/mu0 div-consistently from fluid-side B only.",
            ablastr::warn_manager::WarnPriority::medium);
    }
    // Freeze the covered-B curl fill across RKF45 substages: compute it once per
    // half-step from the step-entry B^n and hold it fixed, instead of re-evaluating
    // the nonsmooth curved-wall extrapolation from the live substage B each substage
    // (which injects a stiff near-wall radial-B feedback that collapses the adaptive
    // substep as a reversal field builds). Opt-in (default off -> byte-identical).
    pp_hybrid.query("conformal_b_curl_fill_freeze", m_conformal_b_curl_fill_freeze);
    if (m_conformal_b_curl_fill_freeze && !m_conformal_b_curl_fill) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.conformal_b_curl_fill_freeze has no effect without "
            "hybrid_pic_model.conformal_b_curl_fill; the covered-B curl fill is "
            "disabled, so there is nothing to freeze.",
            ablastr::warn_manager::WarnPriority::medium);
    }
    pp_hybrid.query("conformal_ect_curvature", m_conformal_ect_curvature);
    if (m_conformal_ect_curvature
        && WarpX::grid_type == ablastr::utils::enums::GridType::Collocated) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.conformal_ect_curvature applies the ECT Faraday "
            "circulation curvature correction, which is only defined on a staggered "
            "(Yee) grid; it is ignored on a collocated grid (which uses the masked "
            "nodal curl, not ECT circulations).",
            ablastr::warn_manager::WarnPriority::medium);
    }
    pp_hybrid.query("conformal_ect_j", m_conformal_ect_j);
    if (m_conformal_ect_j
        && WarpX::grid_type == ablastr::utils::enums::GridType::Collocated) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.conformal_ect_j computes the Ampere current with the "
            "flux-weighted (open-cut-face-area) ECT curl, which is only defined on a "
            "staggered (Yee) grid; it is ignored on a collocated grid (which uses the "
            "masked nodal curl, not open-face fluxes).",
            ablastr::warn_manager::WarnPriority::medium);
    }
    pp_hybrid.query("conformal_ect_lsq", m_conformal_ect_lsq);
    if (m_conformal_ect_lsq
        && WarpX::grid_type == ablastr::utils::enums::GridType::Collocated) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.conformal_ect_lsq computes the Ampere current with the "
            "accurate conformal-EB scheme (PEC covered-B fill + standard Yee curl + "
            "wall-band least-squares centroid overwrite + matched cut-metric div-clean), "
            "which is only defined on a staggered (Yee) grid; it is ignored on a "
            "collocated grid.",
            ablastr::warn_manager::WarnPriority::medium);
    }
    if (m_conformal_ect_lsq && m_conformal_ect_j) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.conformal_ect_lsq and conformal_ect_j are mutually "
            "exclusive Ampere-current schemes; conformal_ect_lsq takes precedence and "
            "conformal_ect_j is ignored.",
            ablastr::warn_manager::WarnPriority::medium);
    }
    // conformal_ect_lsq Phase-2b matched cut-metric divergence-clean controls.
    pp_hybrid.query("conformal_divclean_iters", m_conformal_divclean_iters);
    pp_hybrid.query("conformal_divclean_rtol", m_conformal_divclean_rtol);
    pp_hybrid.query("conformal_divclean_subsample", m_conformal_divclean_subsample);
    pp_hybrid.query("conformal_divclean_cartesian", m_conformal_divclean_cartesian);
    pp_hybrid.query("conformal_lsq_sliver_frac", m_conformal_lsq_sliver_frac);
    pp_hybrid.query("eb_hall_mask", m_eb_hall_mask);

    // Resistive-only generalized Ohm's law in partially-covered EB cells (Lever 2 /
    // GOL masking): drops the stiff 1/n Hall + electron-pressure terms in the cut-cell
    // wall band, leaving E = eta*J (J from the wall-filled curl B). The partial-cell
    // classification comes from the 3-state staggered (ECT) eb_update_E flag, so it is
    // a no-op on a collocated grid (no cut edges are marked partial there).
    pp_hybrid.query("eb_resistive_only_partial", m_eb_resistive_only_partial);
    if (m_eb_resistive_only_partial
        && WarpX::grid_type == ablastr::utils::enums::GridType::Collocated) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.eb_resistive_only_partial requires a staggered (Yee) "
            "embedded boundary; it is ignored on a collocated grid (no cut edges are "
            "marked partially covered there).",
            ablastr::warn_manager::WarnPriority::medium);
    }

    if (m_use_conformal_eb) {
#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ)
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.use_conformal_eb is only supported in 3D and 2D (XZ) "
            "Cartesian geometry");
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(EB::enabled(),
            "hybrid_pic_model.use_conformal_eb requires embedded boundaries to be enabled");
        // Both grid types are supported: a staggered grid uses the enlarged-cell (ECT)
        // Faraday update on Yee cut faces/edges, while a collocated grid uses the masked
        // nodal Faraday update with the level-set mirror boundary condition. The nodal
        // path is keyed off the signed level set rather than the (staggered) ECT geometry.
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                WarpX::field_boundary_lo[idim] != FieldBoundaryType::PML &&
                WarpX::field_boundary_hi[idim] != FieldBoundaryType::PML,
                "hybrid_pic_model.use_conformal_eb is not compatible with PML boundaries");
        }
    }

    // Optional cylindrical (surface-of-revolution) radial metric correction to
    // the embedded-boundary mirror fill (E, J, B). Opt-in refinement of the
    // conformal EB on a smoothly-curved wall: scales the radial/azimuthal parts
    // of the reflected field by the radial Jacobian r_image/r_fill. See
    // EBJBoundary.cpp (mirror_combine / cyl_lambda).
    pp_hybrid.query("eb_cylindrical_correction", m_eb_cylindrical_correction);
    if (m_eb_cylindrical_correction) {
#if !defined(WARPX_DIM_3D)
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.eb_cylindrical_correction is only supported in 3D "
            "Cartesian geometry");
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_use_conformal_eb,
            "hybrid_pic_model.eb_cylindrical_correction requires "
            "hybrid_pic_model.use_conformal_eb");
        std::string cyl_axis = "z";
        pp_hybrid.query("eb_cyl_axis", cyl_axis);
        if      (cyl_axis == "x") { m_eb_cyl_axis = 0; }
        else if (cyl_axis == "y") { m_eb_cyl_axis = 1; }
        else if (cyl_axis == "z") { m_eb_cyl_axis = 2; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.eb_cyl_axis must be 'x', 'y' or 'z'");
        }
    }

    // controls for the embedded-boundary PEC field boundary condition
    utils::parser::queryWithParser(pp_hybrid, "eb_bc_rtol", m_eb_bc_rtol);
    utils::parser::queryWithParser(pp_hybrid, "eb_bc_max_iters", m_eb_bc_max_iters);
    pp_hybrid.query("eb_bc_direct_fill", m_eb_bc_direct_fill);
    // Optionally disable the collocated conformal B wall treatment entirely (no
    // EB B fill) to recover the pre-treatment baseline (for A/B comparison).
    pp_hybrid.query("conformal_b_off", m_conformal_b_off);

    // Image parity of charge deposited beyond the embedded boundary: "pec" folds it back
    // with opposite sign (density vanishes at the wall), "reflect" folds it back with its
    // own sign (mass-conserving when the wall supports the plasma column).
    std::string fold = "pec";
    pp_hybrid.query("eb_deposit_fold", fold);
    if (fold == "pec") { m_eb_fold_pec = true; }
    else if (fold == "reflect") { m_eb_fold_pec = false; }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.eb_deposit_fold must be 'pec' or 'reflect'");
    }
    // Parity of the rho mirror fill across the wall: Dirichlet 0 (odd) by default,
    // Neumann (even) when the wall supports the column.
    pp_hybrid.query("eb_rho_dirichlet", m_eb_rho_dirichlet);

    // isotropized hyper-resistivity Laplacian (Cartesian geometries)
    pp_hybrid.query("isotropic_hyper_resistivity", m_isotropic_hyper_resistivity);

    // isotropized resistive diffusion via the corner-curl E correction
    // (Cartesian geometries; suppresses the grid m=4 from the resistive term)
    pp_hybrid.query("isotropic_resistivity", m_isotropic_resistivity);

    // The isotropic hyper-resistivity Laplacian reads the plasma current at its
    // diagonal/corner neighbors (sqrt(2)*h in plane, sqrt(3)*h at a 3D cube
    // corner). Widen the plasma-current EB mirror-fill band to that corner reach
    // so the diagonal edges near a curved wall are mirror-filled rather than left
    // in the zeroed deep interior (which would inject a spurious nabla^2 J there).
    //
    // With eb_resistive_only_partial on, HybridPICSolveECartesian disables the
    // iso-hyper Laplacian (and the iso-resistivity corner-curl) at every regular
    // edge whose 3x3x3 stencil reaches a non-regular edge near the wall. That
    // removes the diagonal-reach requirement on the mirror fill, so the band
    // collapses to 1: no near-wall edge ever reads diagonal J through the iso
    // stencil. The E and plasma-current (J) fills stay pinned to this same band,
    // now at 1.
    m_eb_fill_band_cells = m_eb_resistive_only_partial
        ? amrex::Real(1.0)
        : (m_isotropic_hyper_resistivity
            ? std::sqrt(static_cast<amrex::Real>(AMREX_SPACEDIM))
            : amrex::Real(1.0));

    // Mirror-fill band width for the Bfield_fp EB fill. The level-set mirror
    // injects a div(B) jump at the band/deep interface; the filled B couples
    // into the solution only via curl/B reads (effective reach 2 cells with the
    // isotropic corner-curl E correction on), so a band >= 3 pushes that jump
    // into the zeroed deep interior where it cannot reach a solution stencil.
    // Default 1 = legacy behavior.
    utils::parser::queryWithParser(pp_hybrid, "eb_b_fill_band_cells", m_eb_b_fill_band_cells);

    // Lower clamp on the wall-normal reflection weight of the covered-B mirror
    // fill (b-curl-fill). The unclamped normal-odd weight is g.s/d_im (= -1 for
    // the quadratic gather), which reverses the covered B_normal across one
    // cell, doubling the near-wall curl(B) -> J wherever there is a near-wall
    // radial B. Clamping to >= 0 (set to 0) drives the covered B_normal toward
    // 0 (the physical PEC B_normal -> 0) instead; the tangential even
    // reflection is untouched. Default -1e30 = disabled / byte-identical.
    pp_hybrid.query("eb_b_fill_normal_weight", m_eb_b_fill_normal_weight);

    // Near-wall stability blend of the covered-B b-curl-fill toward the
    // conformal-ECT cut-face B (see m_conformal_b_curl_fill_blend). 0 (default)
    // = full mirror = byte-identical; 1 = keep the ECT value at cut faces.
    utils::parser::queryWithParser(pp_hybrid, "conformal_b_curl_fill_blend",
                                   m_conformal_b_curl_fill_blend);
    // Relative cap on the cut-face mirror deviation from the ECT value (see
    // m_conformal_b_curl_fill_clamp). 0 (default) = no clamp = byte-identical.
    utils::parser::queryWithParser(pp_hybrid, "conformal_b_curl_fill_clamp",
                                   m_conformal_b_curl_fill_clamp);
    // Option-2 concave re-entrant-corner skip of the covered-B mirror (see
    // m_conformal_b_curl_fill_corner_skip). false (default) = no skip = byte-identical.
    pp_hybrid.query("conformal_b_curl_fill_corner_skip",
                    m_conformal_b_curl_fill_corner_skip);
    if (m_conformal_b_curl_fill_blend != 0.0_rt && !m_conformal_b_curl_fill) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.conformal_b_curl_fill_blend has no effect without "
            "hybrid_pic_model.conformal_b_curl_fill; the covered-B curl fill is "
            "disabled, so there is no mirror to blend.",
            ablastr::warn_manager::WarnPriority::medium);
    }

    // Optional Marder-like diffusive clean of the small curved-wall div(B) /
    // div(J_total) the pointwise mirror injects. Each alpha defaults to 0 (off).
    // The clean is a pure gradient correction, so it dissipates divergence
    // without touching curl/J.
    utils::parser::queryWithParser(pp_hybrid, "divb_clean_alpha", m_divb_clean_alpha);
    utils::parser::queryWithParser(pp_hybrid, "divj_clean_alpha", m_divj_clean_alpha);
    utils::parser::queryWithParser(pp_hybrid, "divb_clean_iters", m_divb_clean_iters);
    utils::parser::queryWithParser(pp_hybrid, "divb_clean_band_cells", m_divb_clean_band_cells);
    pp_hybrid.query("divb_clean_per_step", m_divb_clean_per_step);

    // Marder divergence cleaning of the Ohm's-law E field, applied only in the low-density
    // transition band (0 < rho <= n_floor*q_e). Disabled by default (marder_alpha = 0).
    utils::parser::queryWithParser(pp_hybrid, "marder_alpha", m_marder_alpha);
    utils::parser::queryWithParser(pp_hybrid, "marder_max_iterations", m_marder_max_iterations);
    utils::parser::queryWithParser(pp_hybrid, "marder_rtol", m_marder_rtol);
    utils::parser::queryWithParser(pp_hybrid, "marder_atol", m_marder_atol);
    utils::parser::queryWithParser(pp_hybrid, "marder_substep_interval", m_marder_substep_interval);

    std::string marder_target = "ohm";
    pp_hybrid.query("marder_target", marder_target);
    if (marder_target == "ohm") { m_marder_target = MarderTarget::Ohm; }
    else if (marder_target == "grad_pe_only") { m_marder_target = MarderTarget::GradPeOnly; }
    else if (marder_target == "zero") { m_marder_target = MarderTarget::Zero; }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.marder_target must be 'ohm', 'grad_pe_only' or 'zero'");
    }

    std::string marder_level = "all_substeps";
    pp_hybrid.query("marder_correction_level", marder_level);
    if (marder_level == "all_substeps") { m_marder_level = MarderLevel::AllSubsteps; }
    else if (marder_level == "half_steps") { m_marder_level = MarderLevel::HalfSteps; }
    else if (marder_level == "full_steps") { m_marder_level = MarderLevel::FullSteps; }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.marder_correction_level must be 'all_substeps', "
            "'half_steps' or 'full_steps'");
    }

    if (m_marder_alpha != 0.0_rt) {
#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_XZ)
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.marder_alpha > 0 is only supported in 3D Cartesian, "
            "2D Cartesian (XZ) and RZ geometry");
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_marder_alpha > 0.0_rt && m_marder_alpha <= 0.1_rt,
            "hybrid_pic_model.marder_alpha must be in (0, 0.1] (the explicit "
            "grad(div) update is CFL-limited)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_marder_max_iterations > 0,
            "hybrid_pic_model.marder_max_iterations must be positive");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_marder_substep_interval >= 1,
            "hybrid_pic_model.marder_substep_interval must be >= 1");
#if defined(WARPX_DIM_RZ)
        // The collocated Marder stencils (centered node-to-node grad/div) are
        // derived and validated for Cartesian geometry only; RZ keeps the
        // staggered requirement.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::grid_type != ablastr::utils::enums::GridType::Collocated,
            "hybrid_pic_model.marder_alpha requires a staggered grid in RZ geometry");
#endif
    }
}

void HybridPICModel::AllocateLevelMFs (
    ablastr::fields::MultiFabRegister & fields,
    int lev, const BoxArray& ba, const DistributionMapping& dm,
    const int ncomps,
    const IntVect& ngJ, const IntVect& ngRho,
    const IntVect& ngEB,
    const IntVect& jx_nodal_flag,
    const IntVect& jy_nodal_flag,
    const IntVect& jz_nodal_flag,
    const IntVect& rho_nodal_flag,
    const IntVect& Ex_nodal_flag,
    const IntVect& Ey_nodal_flag,
    const IntVect& Ez_nodal_flag,
    const IntVect& Bx_nodal_flag,
    const IntVect& By_nodal_flag,
    const IntVect& Bz_nodal_flag) const
{
    using ablastr::fields::Direction;

    // The "hybrid_electron_pressure_fp" multifab stores the electron pressure calculated
    // from the specified equation of state.
    fields.alloc_init(FieldType::hybrid_electron_pressure_fp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    // The "hybrid_rho_fp_temp" multifab is used to store the ion charge density
    // interpolated or extrapolated to appropriate timesteps.
    fields.alloc_init(FieldType::hybrid_rho_fp_temp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    // The "hybrid_current_fp_temp" multifab is used to store the ion current density
    // interpolated or extrapolated to appropriate timesteps.
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{0},
        lev, amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{1},
        lev, amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{2},
        lev, amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);

    // The "hybrid_current_fp_plasma" multifab stores the total plasma current calculated
    // as the curl of B minus any external current.
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{0},
        lev, amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{1},
        lev, amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{2},
        lev, amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);

    // the external current density multifab matches the current staggering and
    // one ghost cell is used since we interpolate the current to a nodal grid
    if (m_has_external_current) {
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{0},
            lev, amrex::convert(ba, jx_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{1},
            lev, amrex::convert(ba, jy_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{2},
            lev, amrex::convert(ba, jz_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
    }

    if (m_add_external_fields) {
        m_external_vector_potential->AllocateLevelMFs(
            fields,
            lev, ba, dm,
            ncomps, ngEB,
            Ex_nodal_flag, Ey_nodal_flag, Ez_nodal_flag,
            Bx_nodal_flag, By_nodal_flag, Bz_nodal_flag
        );
    }

#ifdef WARPX_DIM_RZ
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (ncomps == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");
#endif
}

void HybridPICModel::InitData (const ablastr::fields::MultiFabRegister& fields)
{
    m_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_expression, {"rho","J","t"}));
    m_eta = m_resistivity_parser->compile<3>();
    const std::set<std::string> resistivity_symbols = m_resistivity_parser->symbols();
    m_resistivity_has_J_dependence += resistivity_symbols.count("J");

    m_include_hyper_resistivity_term = (m_eta_h_expression != "0.0");
    m_hyper_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_h_expression, {"rho","B"}));
    m_eta_h = m_hyper_resistivity_parser->compile<2>();
    const std::set<std::string> hyper_resistivity_symbols = m_hyper_resistivity_parser->symbols();
    m_hyper_resistivity_has_B_dependence += hyper_resistivity_symbols.count("B");

    if (m_has_external_current) {
        m_J_external_parser[0] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jx_ext_grid_function,{"x","y","z","t"}));
        m_J_external_parser[1] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jy_ext_grid_function,{"x","y","z","t"}));
        m_J_external_parser[2] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jz_ext_grid_function,{"x","y","z","t"}));
        m_J_external[0] = m_J_external_parser[0]->compile<4>();
        m_J_external[1] = m_J_external_parser[1]->compile<4>();
        m_J_external[2] = m_J_external_parser[2]->compile<4>();

        // check if the external current parsers depend on time
        for (int i=0; i<3; i++) {
            const std::set<std::string> J_ext_symbols = m_J_external_parser[i]->symbols();
            m_external_current_has_time_dependence += J_ext_symbols.count("t");
        }
    }

    auto& warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    // Get the grid staggering of the fields involved in calculating E
    amrex::IntVect Jx_stag = fields.get(FieldType::current_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect Jy_stag = fields.get(FieldType::current_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Jz_stag = fields.get(FieldType::current_fp, Direction{2}, 0)->ixType().toIntVect();
    amrex::IntVect Bx_stag = fields.get(FieldType::Bfield_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect By_stag = fields.get(FieldType::Bfield_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Bz_stag = fields.get(FieldType::Bfield_fp, Direction{2}, 0)->ixType().toIntVect();
    amrex::IntVect Ex_stag = fields.get(FieldType::Efield_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect Ey_stag = fields.get(FieldType::Efield_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Ez_stag = fields.get(FieldType::Efield_fp, Direction{2}, 0)->ixType().toIntVect();

    // copy data to device
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Jx_IndexType[idim]    = Jx_stag[idim];
        Jy_IndexType[idim]    = Jy_stag[idim];
        Jz_IndexType[idim]    = Jz_stag[idim];
        Bx_IndexType[idim]    = Bx_stag[idim];
        By_IndexType[idim]    = By_stag[idim];
        Bz_IndexType[idim]    = Bz_stag[idim];
        Ex_IndexType[idim]    = Ex_stag[idim];
        Ey_IndexType[idim]    = Ey_stag[idim];
        Ez_IndexType[idim]    = Ez_stag[idim];
    }

    // Below we set all the unused dimensions to have nodal values for J, B & E
    // since these values will be interpolated onto a nodal grid - if this is
    // not done the Interp function returns nonsense values.
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z) || \
    defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    Jx_IndexType[2]    = 1;
    Jy_IndexType[2]    = 1;
    Jz_IndexType[2]    = 1;
    Bx_IndexType[2]    = 1;
    By_IndexType[2]    = 1;
    Bz_IndexType[2]    = 1;
    Ex_IndexType[2]    = 1;
    Ey_IndexType[2]    = 1;
    Ez_IndexType[2]    = 1;
#endif
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    Jx_IndexType[1]    = 1;
    Jy_IndexType[1]    = 1;
    Jz_IndexType[1]    = 1;
    Bx_IndexType[1]    = 1;
    By_IndexType[1]    = 1;
    Bz_IndexType[1]    = 1;
    Ex_IndexType[1]    = 1;
    Ey_IndexType[1]    = 1;
    Ez_IndexType[1]    = 1;
#endif

    if (m_has_external_current) {
        // Initialize external current - note that this approach skips the check
        // if the current is time dependent which is what needs to be done to
        // write time independent fields on the first step.
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            warpx.ComputeExternalFieldOnGridUsingParser(
                FieldType::hybrid_current_fp_external,
                m_J_external[0],
                m_J_external[1],
                m_J_external[2],
                lev, PatchType::fine,
                warpx.GetEBUpdateEFlag());
        }
    }

    if (m_add_external_fields) {
        m_external_vector_potential->InitData();
    }
}

void HybridPICModel::InitialBEBFill ()
{
#ifdef AMREX_USE_EB
    using ablastr::utils::enums::GridType;
    if (!EB::enabled() || !m_use_conformal_eb || m_conformal_b_off) { return; }
    if (WarpX::grid_type != GridType::Collocated) { return; }
    auto& warpx = WarpX::GetInstance();
    auto const& eb_update_B = warpx.GetEBUpdateBFlag();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        if (static_cast<int>(m_eb_bc_status_B.size()) <= lev) { m_eb_bc_status_B.resize(lev+1); }
        warpx::hybrid::ApplyPECBoundaryToField(
            warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev),
            eb_update_B[lev],
            *warpx.m_fields.get(FieldType::distance_to_eb, lev),
            warpx.Geom(lev),
            m_eb_bc_rtol, m_eb_bc_max_iters, m_eb_bc_direct_fill,
            /*normal_odd=*/true, /*fill_covered_centers=*/false,
            &m_eb_bc_status_B[lev], m_eb_b_fill_band_cells,
            m_eb_cylindrical_correction, m_eb_cyl_axis);
    }
#endif
}

void HybridPICModel::GetCurrentExternal ()
{
    if (!m_external_current_has_time_dependence) { return; }

    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        warpx.ComputeExternalFieldOnGridUsingParser(
            FieldType::hybrid_current_fp_external,
            m_J_external[0],
            m_J_external[1],
            m_J_external[2],
            lev, PatchType::fine,
            warpx.GetEBUpdateEFlag());
    }
}

void HybridPICModel::CalculatePlasmaCurrent (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculatePlasmaCurrent(Bfield[lev], eb_update_E[lev], lev);
    }
}

void HybridPICModel::CalculatePlasmaCurrent (
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev) const
{
    ABLASTR_PROFILE("HybridPICModel::CalculatePlasmaCurrent()");

    auto& warpx = WarpX::GetInstance();

    // Extend B into the covered cells with a 2nd-order quadratic (even-reflection)
    // gather BEFORE the Ampere curl, so curl(B) -> J reads a curved-wall-accurate
    // covered B. The conformal (ECT) B push is 2nd order, but the curl is plain-
    // masked and otherwise reads the staircased covered B, capping the near-wall
    // plasma current (and the Ohm's-law E built from it) at 1st order on a curved
    // wall. Staggered (Yee) grid only -- the nodal quadratic gather is not yet
    // robust (see m_conformal_b_curl_fill). Magnetic parity (normal odd /
    // tangential even), covered centers filled, on the B update mask.
    // The flux-weighted ECT curl (m_conformal_ect_j) drops covered B by its zero
    // open-face area, so it needs no covered-B mirror; skip the curl fill then.
    // The accurate conformal scheme (conformal_ect_lsq) feeds the standard Yee curl
    // from the same PEC covered-B fill (B_n=0 odd / tangential even), then overwrites
    // the wall-band J below; drive the fill for it too. Form A (conformal_ect_j) reads
    // open-face fluxes only and needs no covered-B fill, so it stays excluded.
    if (EB::enabled() && (m_conformal_b_curl_fill || m_conformal_ect_lsq) && !m_conformal_ect_j
        && WarpX::grid_type != ablastr::utils::enums::GridType::Collocated) {
        if (static_cast<int>(m_eb_bc_status_B.size()) <= lev) { m_eb_bc_status_B.resize(lev+1); }
        // Freeze path: a valid step-start snapshot exists (taken in
        // SnapshotConformalBCurlFreeze from B^n). Re-stamp the cells the fill
        // controls (status != S_SOLUTION: the band targets + zeroed deep interior +
        // cut/covered centers) from the frozen buffer, holding the nonsmooth
        // covered-B extrapolation static across the RKF45 substages instead of
        // re-evaluating it from the live substage B. The covered/band cells the
        // masked Faraday push touched are overwritten back to their B^n values.
        bool const use_freeze = m_conformal_b_curl_fill_freeze
            && m_eb_b_curl_freeze_valid
            && lev < static_cast<int>(m_eb_b_curl_freeze_buf.size())
            && !m_eb_bc_status_B[lev].empty();
        if (use_freeze) {
            auto const& st = m_eb_bc_status_B[lev];
            auto const& buf = m_eb_b_curl_freeze_buf[lev];
            for (int c = 0; c < 3; ++c) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(*Bfield[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    amrex::Box const tb = mfi.tilebox(Bfield[c]->ixType().toIntVect(),
                                                      Bfield[c]->nGrowVect());
                    auto const& Bc = Bfield[c]->array(mfi);
                    auto const& bufc = buf[c].const_array(mfi);
                    auto const& stat = st.status[c]->const_array(mfi);
                    amrex::ParallelFor(tb,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        // S_SOLUTION (= 0) cells are the live solution domain (never
                        // written by the fill); every other status is a covered/band
                        // cell the fill controls, so hold it at the frozen value.
                        if (stat(i, j, k) != 0) { Bc(i, j, k) = bufc(i, j, k); }
                    });
                }
            }
        } else {
        warpx::hybrid::ApplyPECBoundaryToField(
            Bfield, warpx.GetEBUpdateBFlag()[lev],
            *warpx.m_fields.get(FieldType::distance_to_eb, lev),
            warpx.Geom(lev),
            m_eb_bc_rtol, m_eb_bc_max_iters, m_eb_bc_direct_fill,
            /*normal_odd=*/true, /*fill_covered_centers=*/true,
            &m_eb_bc_status_B[lev], m_eb_b_fill_band_cells,
            m_eb_cylindrical_correction, m_eb_cyl_axis,
            /*quadratic_gather=*/true, m_eb_b_fill_normal_weight,
            m_conformal_b_curl_fill_blend, m_conformal_b_curl_fill_clamp,
            m_conformal_b_curl_fill_corner_skip);
        }
    }

    ablastr::fields::VectorField current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    if (m_conformal_ect_j && EB::enabled()
        && WarpX::grid_type != ablastr::utils::enums::GridType::Collocated) {
        // Flux-weighted ("Form A") conformal-EB curl: J = curl(B)/mu0 with each B
        // weighted by its open cut-face-area fraction. Divergence-consistent across
        // the wall, so it needs no covered-B mirror (skipped above).
        warpx.get_pointer_fdtd_solver_fp(lev)->CalculateCurrentAmpereECT(
            current_fp_plasma, Bfield,
            warpx.m_fields.get_alldirs(FieldType::face_areas, lev),
            eb_update_E, lev
        );
    } else {
        warpx.get_pointer_fdtd_solver_fp(lev)->CalculateCurrentAmpere(
            current_fp_plasma, Bfield, eb_update_E, lev
        );
    }

    if (m_has_external_current) {
        // Subtract external current from "Ampere" current calculated above. Note
        // we need to include 1 ghost cell since later we will interpolate the
        // plasma current to a nodal grid.
        ablastr::fields::VectorField current_fp_external = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_external, lev);
        for (int i=0; i<3; i++) {
            current_fp_plasma[i]->minus(*current_fp_external[i], 0, 1, 1);
        }
    }

    // Enforce the PEC current boundary condition at the embedded boundary:
    // tangential J vanishes at the surface, normal J has zero normal gradient
    // and the deep conductor interior carries no volume current. Cut edges
    // whose centers are on or inside the surface are filled too (the Ampere
    // current is evaluated at the covered centers and is not meaningful
    // there).
    //
    // The flux-weighted ("Form A") conformal curl is already divergence-
    // consistent at the wall (it reads only open-face fluxes, fully-covered
    // edges are zeroed in the kernel). The mirror fill below INJECTS div(J)
    // (see the MarderCleanDivergence comment) and would corrupt Form A's clean
    // wall current, so it is skipped when conformal_ect_j is on.
    if (EB::enabled() && !m_conformal_ect_j) {
        // The plasma current uses its own (wider-band) fill classification when
        // the isotropic hyper-resistivity Laplacian is enabled: that stencil
        // reads diagonal/corner edges, so the band is widened to m_eb_fill_band_cells
        // (see ReadParameters). Kept separate from m_eb_bc_status_E, whose
        // one-cell band is shared with the deposit fold and the Ohm's-law E fill.
        if (static_cast<int>(m_eb_bc_status_Jplasma.size()) <= lev) { m_eb_bc_status_Jplasma.resize(lev+1); }

        // Covered-cell mirror fill of the Ampere current. Needed by BOTH the masked and
        // the conformal_ect_lsq paths: the standard Yee curl zeroes fully-covered edges,
        // so without this fill there is a large fluid->covered current JUMP at the wall
        // (the unmasked wall div(J) runs ~10x the stable level, and the zeroed covered J
        // contaminates the nodal J x B) which stiffens the RKF45 B-push to a crawl.
        // conformal_ect_lsq overwrites the fluid wall band with the accurate LSQ current
        // afterwards; the covered edges keep this mirror fill.
        warpx::hybrid::ApplyPECBoundaryToField(
            current_fp_plasma, eb_update_E,
            *warpx.m_fields.get(FieldType::distance_to_eb, lev),
            warpx.Geom(lev),
            m_eb_bc_rtol, m_eb_bc_max_iters, m_eb_bc_direct_fill,
            /*normal_odd=*/false, /*fill_covered_centers=*/true,
            &m_eb_bc_status_Jplasma[lev], m_eb_fill_band_cells,
            m_eb_cylindrical_correction, m_eb_cyl_axis);

        if (m_conformal_ect_lsq
            && WarpX::grid_type != ablastr::utils::enums::GridType::Collocated) {
            // conformal_ect_lsq: OVERWRITE the fluid wall band with the accurate
            // LSQ-centroid reconstruction of the standard Yee curl. The covered edges
            // keep the mirror fill above (small fluid->covered jump). Phase 2b
            // (the matched cut-metric divergence clean) is an opt-in refinement
            // (off by default: conformal_divclean_iters == 0).
            ApplyConformalLSQOverwrite(current_fp_plasma, lev);
            ApplyConformalDivClean(current_fp_plasma, lev);
        } else if (m_divj_clean_alpha > 0.0_rt && !m_divb_clean_per_step) {
            // Masked path: optional diffusive clean dissipating the curved-wall div(J)
            // the mirror injects in the TOTAL (Ampere) current. div(J_total)=0 is
            // current continuity (no charge separation). This acts ONLY on the total
            // current; the deposited ion-species current is never continuity-
            // constrained. Skipped in the per-step cadence (done at the FullStep site).
            MarderCleanDivergence(
                current_fp_plasma, eb_update_E, &m_eb_bc_status_Jplasma[lev],
                /*normal_odd=*/false, /*fill_covered_centers=*/true,
                m_divj_clean_alpha, m_divb_clean_iters,
                m_divb_clean_band_cells, m_eb_fill_band_cells, lev);
        }
    }
}

void HybridPICModel::SnapshotConformalBCurlFreeze (
    ablastr::fields::VectorField const& Bfield,
    const int lev) const
{
    // Only active for a staggered conformal-EB b-curl-fill run with the freeze on.
    m_eb_b_curl_freeze_valid = false;
    if (!(EB::enabled() && m_conformal_b_curl_fill && m_conformal_b_curl_fill_freeze
          && WarpX::grid_type != ablastr::utils::enums::GridType::Collocated)) {
        return;
    }

    auto& warpx = WarpX::GetInstance();

    // Take the b-curl-fill ONCE on the step-entry B^n (this populates the status
    // classification m_eb_bc_status_B[lev] and writes the covered/band cells of
    // Bfield to their wall-consistent B^n values). The result is then snapshotted;
    // every later RKF45 substage re-stamps it (see CalculatePlasmaCurrent) instead
    // of recomputing the nonsmooth fill from the live substage B.
    if (static_cast<int>(m_eb_bc_status_B.size()) <= lev) { m_eb_bc_status_B.resize(lev+1); }
    warpx::hybrid::ApplyPECBoundaryToField(
        Bfield, warpx.GetEBUpdateBFlag()[lev],
        *warpx.m_fields.get(FieldType::distance_to_eb, lev),
        warpx.Geom(lev),
        m_eb_bc_rtol, m_eb_bc_max_iters, m_eb_bc_direct_fill,
        /*normal_odd=*/true, /*fill_covered_centers=*/true,
        &m_eb_bc_status_B[lev], m_eb_b_fill_band_cells,
        m_eb_cylindrical_correction, m_eb_cyl_axis,
        /*quadratic_gather=*/true, m_eb_b_fill_normal_weight,
        m_conformal_b_curl_fill_blend, m_conformal_b_curl_fill_clamp,
        m_conformal_b_curl_fill_corner_skip);

    // Snapshot the filled B^n (full field incl. ghosts) into the per-level buffer.
    if (static_cast<int>(m_eb_b_curl_freeze_buf.size()) <= lev) {
        m_eb_b_curl_freeze_buf.resize(lev+1);
    }
    auto& buf = m_eb_b_curl_freeze_buf[lev];
    for (int c = 0; c < 3; ++c) {
        amrex::IntVect const ng = Bfield[c]->nGrowVect();
        if (!buf[c].ok()
            || buf[c].boxArray() != Bfield[c]->boxArray()
            || buf[c].DistributionMap() != Bfield[c]->DistributionMap()
            || buf[c].nGrowVect() != ng) {
            buf[c] = amrex::MultiFab(Bfield[c]->boxArray(),
                                     Bfield[c]->DistributionMap(), 1, ng);
        }
        amrex::MultiFab::Copy(buf[c], *Bfield[c], 0, 0, 1, ng);
    }
    m_eb_b_curl_freeze_valid = true;
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    const bool solve_for_Faraday) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        HybridPICSolveE(
            Efield[lev], Jfield[lev], Bfield[lev], *rhofield[lev],
            eb_update_E[lev], lev, solve_for_Faraday
        );
    }
    // Allow execution of Python callback after E-field push
    ExecutePythonCallback("afterEpush");
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev, const bool solve_for_Faraday) const
{
    ABLASTR_PROFILE("WarpX::HybridPICSolveE()");

    HybridPICSolveE(
        Efield, Jfield, Bfield, rhofield, eb_update_E, lev,
        PatchType::fine, solve_for_Faraday
    );
    if (lev > 0)
    {
        amrex::Abort(Utils::TextMsg::Err(
        "HybridPICSolveE: Only one level implemented for hybrid-PIC solver."));
    }
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev, PatchType patch_type,
    const bool solve_for_Faraday) const
{
    auto& warpx = WarpX::GetInstance();

    ablastr::fields::VectorField current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    auto* const electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);

    // Solve E field in regular cells
    warpx.get_pointer_fdtd_solver_fp(lev)->HybridPICSolveE(
        Efield, current_fp_plasma, Jfield, Bfield, rhofield,
        *electron_pressure_fp, eb_update_E, lev, this, solve_for_Faraday
    );
    amrex::Real const time = warpx.gett_old(0) + warpx.getdt(0);
    warpx.ApplyEfieldBoundary(lev, patch_type, time);

    // The Ohm's-law E is algebraic rather than integrated from B, so the PEC condition
    // (tangential E zero at the surface, normal E with zero normal gradient, zero deep in
    // the conductor) must be imposed directly on the masked edges, including cut edges with
    // covered centers, where the floored interpolated density makes Ohm's law spurious.
    if (EB::enabled()) {
        if (static_cast<int>(m_eb_bc_status_E.size()) <= lev) { m_eb_bc_status_E.resize(lev+1); }
        warpx::hybrid::ApplyPECBoundaryToField(
            Efield, eb_update_E,
            *warpx.m_fields.get(FieldType::distance_to_eb, lev),
            warpx.Geom(lev),
            m_eb_bc_rtol, m_eb_bc_max_iters, m_eb_bc_direct_fill,
            /*normal_odd=*/false, /*fill_covered_centers=*/true,
            // E fill band is PINNED to the J fill band: E = eta*J in the cut/
            // covered region, so filling E beyond where J is filled leaves E != 0
            // where J = 0 (an inconsistent source that blows up through the
            // curl(E)->B->curl(B)->J->E loop). See m_eb_fill_band_cells.
            &m_eb_bc_status_E[lev], m_eb_fill_band_cells,
            m_eb_cylindrical_correction, m_eb_cyl_axis);
    }
}

void HybridPICModel::CalculateElectronPressure() const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateElectronPressure(lev);
    }
}

void HybridPICModel::CalculateElectronPressure(const int lev) const
{
    ABLASTR_PROFILE("WarpX::CalculateElectronPressure()");

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::ScalarField electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    ablastr::fields::ScalarField rho_fp = warpx.m_fields.get(FieldType::rho_fp, lev);

    // Calculate the electron pressure using rho^{n+1}.
    FillElectronPressureMF(
        *electron_pressure_fp,
        *rho_fp
    );
    // The conducting wall supports the plasma back-pressure, so reflect Pe evenly (zero
    // normal gradient) across the embedded boundary; grad(Pe) stencils straddling the wall
    // must not see the equation of state evaluated on the nonpositive mirrored density
    // inside the conductor.
    if (EB::enabled()) {
        warpx::hybrid::ApplyEBBoundaryToNodalScalar(
            *electron_pressure_fp,
            *warpx.m_fields.get(FieldType::distance_to_eb, lev),
            warpx.Geom(lev),
            /*odd=*/false);
    }
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
    ablastr::utils::communication::FillBoundary(
        *electron_pressure_fp,
        WarpX::do_single_precision_comms,
        warpx.Geom(lev).periodicity(),
        true);
}

void HybridPICModel::FillElectronPressureMF (
    amrex::MultiFab& Pe_field,
    amrex::MultiFab const& rho_field
) const
{
    const auto n0_ref = m_n0_ref;
    const auto elec_temp = m_elec_temp;
    const auto gamma = m_gamma;

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(Pe_field, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        // Extract field data for this grid/tile
        Array4<Real const> const& rho = rho_field.const_array(mfi);
        Array4<Real> const& Pe = Pe_field.array(mfi);

        // Extract tileboxes for which to loop
        const Box& tilebox  = mfi.tilebox();

        ParallelFor(tilebox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // The embedded-boundary Dirichlet condition mirrors the charge
            // density to negative values inside the conductor: clamp the
            // equation-of-state input to its physical domain (the covered
            // nodes are subsequently overwritten by the even EB reflection)
            Pe(i, j, k) = ElectronPressure::get_pressure(
                n0_ref, elec_temp, gamma, amrex::max(rho(i, j, k), 0._rt)
            );
        });
    }
}

void HybridPICModel::BfieldEvolveRK (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    amrex::Real dt, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        BfieldEvolveRK(
            Bfield, Efield, Jfield, rhofield, eb_update_E, dt, lev, subcycling_half,
            ng, nodal_sync
        );
    }
}

void HybridPICModel::BfieldEvolveRK (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    // Make copies of the B-field multifabs at t = n and create multifabs for
    // each direction to store the Runge-Kutta intermediate terms. Each
    // multifab has 2 components for the different terms that need to be stored.
    std::array< MultiFab, 3 > B_old;
    std::array< MultiFab, 3 > K;
    for (int ii = 0; ii < 3; ii++)
    {
        B_old[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 1,
            Bfield[lev][ii]->nGrowVect()
        );
        MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);

        K[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 2,
            Bfield[lev][ii]->nGrowVect()
        );
    }

    // The Runge-Kutta scheme begins here.
    // Step 1:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * [-curl x E(B_old)] = B_old + 0.5 * dt * K0.
    for (int ii = 0; ii < 3; ii++)
    {
        // Extract 0.5 * dt * K0 for each direction into index 0 of K.
        MultiFab::LinComb(
            K[ii], 1._rt, *Bfield[lev][ii], 0, -1._rt, B_old[ii], 0, 0, 1, ng
        );
    }

    // Step 2:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    //   B_new = B_old + 0.5 * dt * K0 + 0.5 * dt * [-curl x E(B_old + 0.5 * dt * K1)]
    //         = B_old + 0.5 * dt * K0 + 0.5 * dt * K1
    //
    // Subtract 0.5 * dt * K0 from the Bfield to get
    //   B_new = B_old + 0.5 * dt * K1.
    // Extract 0.5 * dt * K1 and write into index 1 of K.

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const &Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const &By = Bfield[lev][1]->array(mfi);
        Array4<Real> const &Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const &Kx = K[0].array(mfi);
        Array4<Real> const &Ky = K[1].array(mfi);
        Array4<Real> const &Kz = K[2].array(mfi);
        Array4<Real const> const &Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const &By_old = B_old[1].const_array(mfi);
        Array4<Real const> const &Bz_old = B_old[2].const_array(mfi);

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy  = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz  = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);

        amrex::ParallelFor(tjx, tjy, tjz,
            // x calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bx(i, j, k) -= Kx(i, j, k, 0);
                Kx(i, j, k, 1) = Bx(i, j, k) - Bx_old(i, j, k);
            },

            // y calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                By(i, j, k) -= Ky(i, j, k, 0);
                Ky(i, j, k, 1) = By(i, j, k) - By_old(i, j, k);
            },

            // z calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bz(i, j, k) -= Kz(i, j, k, 0);
                Kz(i, j, k, 1) = Bz(i, j, k) - Bz_old(i, j, k);
            }
        );
    }

    // Step 3:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * K1 + dt * [-curl  x E(B_old + 0.5 * dt * K1)]
    //       = B_old + 0.5 * dt * K1 + dt * K2
    for (int ii = 0; ii < 3; ii++)
    {
        // Subtract 0.5 * dt * K1 from the Bfield for each direction to get
        // B_new = B_old + dt * K2.
        MultiFab::Subtract(*Bfield[lev][ii], K[ii], 1, 0, 1, ng);
    }

    // Step 4:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    //   B_new = B_old + dt * K2 + 0.5 * dt * [-curl x E(B_old + dt * K2)]
    //         = B_old + dt * K2 + 0.5 * dt * K3
    // and
    //   index 0 of K = 0.5 * dt * K0
    //   index 1 of K = 0.5 * dt * K1
    //
    // We calculate:
    //   K = 0.5 * dt * K0 + dt * K1 + dt * K2 + 0.5 * dt * K3
    // then update B with the Runge-Kutta sum:
    //   B = B_old + 1/3 * K

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const &Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const &By = Bfield[lev][1]->array(mfi);
        Array4<Real> const &Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const &Kx = K[0].array(mfi);
        Array4<Real> const &Ky = K[1].array(mfi);
        Array4<Real> const &Kz = K[2].array(mfi);
        Array4<Real const> const &Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const &By_old = B_old[1].const_array(mfi);
        Array4<Real const> const &Bz_old = B_old[2].const_array(mfi);

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy  = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz  = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);

        amrex::ParallelFor(tjx, tjy, tjz,
            // Bx calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Kx(i, j, k, 0) += Bx(i, j, k) - Bx_old(i, j, k) + 2.0_rt * Kx(i, j, k, 1);
                Bx(i, j, k) = Bx_old(i, j, k) + Kx(i, j, k, 0) / 3.0_rt;
            },

            // By calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ky(i, j, k, 0) += By(i, j, k) - By_old(i, j, k) + 2.0_rt * Ky(i, j, k, 1);
                By(i, j, k) = By_old(i, j, k) + Ky(i, j, k, 0) / 3.0_rt;
            },

            // Bz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Kz(i, j, k, 0) += Bz(i, j, k) - Bz_old(i, j, k) + 2.0_rt * Kz(i, j, k, 1);
                Bz(i, j, k) = Bz_old(i, j, k) + Kz(i, j, k, 0) / 3.0_rt;
            }
        );
    }
}


void HybridPICModel::BfieldEvolveRKF45 (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    amrex::Real dt_half, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        BfieldEvolveRKF45(
            Bfield, Efield, Jfield, rhofield, eb_update_E, dt_half, lev, subcycling_half,
            ng, nodal_sync
        );
    }
}

void HybridPICModel::BfieldEvolveRKF45 (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    // Fehlberg RKF45 Butcher tableau coefficients
    constexpr amrex::Real a21 = 1._rt/4._rt;
    constexpr amrex::Real a31 = 3._rt/32._rt,      a32 = 9._rt/32._rt;
    constexpr amrex::Real a41 = 1932._rt/2197._rt,  a42 = -7200._rt/2197._rt, a43 = 7296._rt/2197._rt;
    constexpr amrex::Real a51 = 439._rt/216._rt,    a52 = -8._rt,
                          a53 = 3680._rt/513._rt,    a54 = -845._rt/4104._rt;
    constexpr amrex::Real a61 = -8._rt/27._rt,      a62 = 2._rt,
                          a63 = -3544._rt/2565._rt,  a64 = 1859._rt/4104._rt,  a65 = -11._rt/40._rt;
    // 4th-order solution weights (k2 and k6 terms are zero in Fehlberg's formula)
    constexpr amrex::Real b1 = 25._rt/216._rt,  b3 = 1408._rt/2565._rt,
                          b4 = 2197._rt/4104._rt, b5 = -1._rt/5._rt;
    // Error = B5 - B4 weights: h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6)
    constexpr amrex::Real e1 =  1._rt/360._rt,    e3 = -128._rt/4275._rt,
                          e4 = -2197._rt/75240._rt, e5 = 1._rt/50._rt, e6 = 2._rt/55._rt;

    // K: 5 components per field direction stored as:
    //   comp 0 = h*k1, comp 1 = h*k2 (overwritten with h*k6 after stage 6),
    //   comp 2 = h*k3, comp 3 = h*k4, comp 4 = h*k5
    std::array<MultiFab, 3> B_old;
    std::array<MultiFab, 3> K;
    std::array<MultiFab, 3> err_scratch;
    for (int ii = 0; ii < 3; ii++)
    {
        B_old[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 1,
            Bfield[lev][ii]->nGrowVect()
        );
        MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);

        K[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 5,
            Bfield[lev][ii]->nGrowVect()
        );
        err_scratch[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 1,
            amrex::IntVect(0)
        );
    }

    // conformal_b_curl_fill_freeze: take the once-per-half-step covered-B snapshot
    // from the step-entry B^n (B_old above is the pre-fill staircase, matching the
    // unfrozen baseline where stage 1 fills B from the staircase B_old). The fill
    // here writes B's covered/band cells to their wall-consistent B^n values and
    // records them; every RKF45 substage below re-stamps those frozen values rather
    // than re-evaluating the nonsmooth fill from the live substage B. No-op (and
    // byte-identical) unless the freeze flag is on.
    SnapshotConformalBCurlFreeze(Bfield[lev], lev);

    amrex::Real dt_sub = 2._rt * dt_half / m_substeps;
    amrex::Real t = 0._rt;
    int n_attempts = 0;
    int n_accepted = 0;

    while (t < dt_half)
    {
        if (t + dt_sub > dt_half) { dt_sub = dt_half - t; }

        // ---- Stage 1: B = B_old, FieldPush, K[comp0] = h*k1 fused with Stage 2 B-update ----
        FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                  dt_sub, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
            Array4<Real> const& By = Bfield[lev][1]->array(mfi);
            Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
            Array4<Real> const& Kx = K[0].array(mfi);
            Array4<Real> const& Ky = K[1].array(mfi);
            Array4<Real> const& Kz = K[2].array(mfi);
            Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
            Array4<Real const> const& By_old = B_old[1].const_array(mfi);
            Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
            Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
            Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
            Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
            amrex::ParallelFor(tjx, tjy, tjz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Bx(i, j, k) - Bx_old(i, j, k);
                    Kx(i, j, k, 0) = k1;
                    Bx(i, j, k) = Bx_old(i, j, k) + a21*k1;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = By(i, j, k) - By_old(i, j, k);
                    Ky(i, j, k, 0) = k1;
                    By(i, j, k) = By_old(i, j, k) + a21*k1;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Bz(i, j, k) - Bz_old(i, j, k);
                    Kz(i, j, k, 0) = k1;
                    Bz(i, j, k) = Bz_old(i, j, k) + a21*k1;
                }
            );
        }

        // ---- Stage 2: FieldPush, K[comp1] = h*k2 fused with Stage 3 B-update ----
        FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                  dt_sub, subcycling_half, ng, nodal_sync);
        // Stage 2 K[1]-readback fused with Stage 3 B-update.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
            Array4<Real> const& By = Bfield[lev][1]->array(mfi);
            Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
            Array4<Real> const& Kx = K[0].array(mfi);
            Array4<Real> const& Ky = K[1].array(mfi);
            Array4<Real> const& Kz = K[2].array(mfi);
            Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
            Array4<Real const> const& By_old = B_old[1].const_array(mfi);
            Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
            Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
            Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
            Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
            amrex::ParallelFor(tjx, tjy, tjz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kx(i, j, k, 0);
                    amrex::Real const k2 = Bx(i, j, k) - Bx_old(i, j, k) - a21*k1;
                    Kx(i, j, k, 1) = k2;
                    Bx(i, j, k) = Bx_old(i, j, k) + a31*k1 + a32*k2;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Ky(i, j, k, 0);
                    amrex::Real const k2 = By(i, j, k) - By_old(i, j, k) - a21*k1;
                    Ky(i, j, k, 1) = k2;
                    By(i, j, k) = By_old(i, j, k) + a31*k1 + a32*k2;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kz(i, j, k, 0);
                    amrex::Real const k2 = Bz(i, j, k) - Bz_old(i, j, k) - a21*k1;
                    Kz(i, j, k, 1) = k2;
                    Bz(i, j, k) = Bz_old(i, j, k) + a31*k1 + a32*k2;
                }
            );
        }

        // ---- Stage 3: FieldPush, then K[comp2] = h*k3 fused with Stage 4 B-update ----
        FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                  dt_sub, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
            Array4<Real> const& By = Bfield[lev][1]->array(mfi);
            Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
            Array4<Real> const& Kx = K[0].array(mfi);
            Array4<Real> const& Ky = K[1].array(mfi);
            Array4<Real> const& Kz = K[2].array(mfi);
            Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
            Array4<Real const> const& By_old = B_old[1].const_array(mfi);
            Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
            Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
            Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
            Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
            amrex::ParallelFor(tjx, tjy, tjz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kx(i, j, k, 0);
                    amrex::Real const k2 = Kx(i, j, k, 1);
                    amrex::Real const k3 = Bx(i, j, k) - Bx_old(i, j, k) - a31*k1 - a32*k2;
                    Kx(i, j, k, 2) = k3;
                    Bx(i, j, k) = Bx_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Ky(i, j, k, 0);
                    amrex::Real const k2 = Ky(i, j, k, 1);
                    amrex::Real const k3 = By(i, j, k) - By_old(i, j, k) - a31*k1 - a32*k2;
                    Ky(i, j, k, 2) = k3;
                    By(i, j, k) = By_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kz(i, j, k, 0);
                    amrex::Real const k2 = Kz(i, j, k, 1);
                    amrex::Real const k3 = Bz(i, j, k) - Bz_old(i, j, k) - a31*k1 - a32*k2;
                    Kz(i, j, k, 2) = k3;
                    Bz(i, j, k) = Bz_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
                }
            );
        }

        // ---- Stage 4: FieldPush, then K[comp3] = h*k4 fused with Stage 5 B-update ----
        FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                  dt_sub, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
            Array4<Real> const& By = Bfield[lev][1]->array(mfi);
            Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
            Array4<Real> const& Kx = K[0].array(mfi);
            Array4<Real> const& Ky = K[1].array(mfi);
            Array4<Real> const& Kz = K[2].array(mfi);
            Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
            Array4<Real const> const& By_old = B_old[1].const_array(mfi);
            Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
            Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
            Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
            Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
            amrex::ParallelFor(tjx, tjy, tjz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kx(i, j, k, 0);
                    amrex::Real const k2 = Kx(i, j, k, 1);
                    amrex::Real const k3 = Kx(i, j, k, 2);
                    amrex::Real const k4 = Bx(i, j, k) - Bx_old(i, j, k)
                                         - a41*k1 - a42*k2 - a43*k3;
                    Kx(i, j, k, 3) = k4;
                    Bx(i, j, k) = Bx_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Ky(i, j, k, 0);
                    amrex::Real const k2 = Ky(i, j, k, 1);
                    amrex::Real const k3 = Ky(i, j, k, 2);
                    amrex::Real const k4 = By(i, j, k) - By_old(i, j, k)
                                         - a41*k1 - a42*k2 - a43*k3;
                    Ky(i, j, k, 3) = k4;
                    By(i, j, k) = By_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kz(i, j, k, 0);
                    amrex::Real const k2 = Kz(i, j, k, 1);
                    amrex::Real const k3 = Kz(i, j, k, 2);
                    amrex::Real const k4 = Bz(i, j, k) - Bz_old(i, j, k)
                                         - a41*k1 - a42*k2 - a43*k3;
                    Kz(i, j, k, 3) = k4;
                    Bz(i, j, k) = Bz_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
                }
            );
        }

        // ---- Stage 5: FieldPush, then K[comp4] = h*k5 fused with Stage 6 B-update ----
        FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                  dt_sub, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
            Array4<Real> const& By = Bfield[lev][1]->array(mfi);
            Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
            Array4<Real> const& Kx = K[0].array(mfi);
            Array4<Real> const& Ky = K[1].array(mfi);
            Array4<Real> const& Kz = K[2].array(mfi);
            Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
            Array4<Real const> const& By_old = B_old[1].const_array(mfi);
            Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
            Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
            Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
            Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
            amrex::ParallelFor(tjx, tjy, tjz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kx(i, j, k, 0);
                    amrex::Real const k2 = Kx(i, j, k, 1);
                    amrex::Real const k3 = Kx(i, j, k, 2);
                    amrex::Real const k4 = Kx(i, j, k, 3);
                    amrex::Real const k5 = Bx(i, j, k) - Bx_old(i, j, k)
                                         - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                    Kx(i, j, k, 4) = k5;
                    Bx(i, j, k) = Bx_old(i, j, k)
                                + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Ky(i, j, k, 0);
                    amrex::Real const k2 = Ky(i, j, k, 1);
                    amrex::Real const k3 = Ky(i, j, k, 2);
                    amrex::Real const k4 = Ky(i, j, k, 3);
                    amrex::Real const k5 = By(i, j, k) - By_old(i, j, k)
                                         - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                    Ky(i, j, k, 4) = k5;
                    By(i, j, k) = By_old(i, j, k)
                                + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kz(i, j, k, 0);
                    amrex::Real const k2 = Kz(i, j, k, 1);
                    amrex::Real const k3 = Kz(i, j, k, 2);
                    amrex::Real const k4 = Kz(i, j, k, 3);
                    amrex::Real const k5 = Bz(i, j, k) - Bz_old(i, j, k)
                                         - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                    Kz(i, j, k, 4) = k5;
                    Bz(i, j, k) = Bz_old(i, j, k)
                                + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
                }
            );
        }

        // ---- Stage 6: FieldPush, then K[comp1] = h*k6 (overwrites h*k2) fused with B4 + error ----
        FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                  dt_sub, subcycling_half, ng, nodal_sync);
        // K[comp1] is overwritten here: reads h*k2 (old value) then writes h*k6 in each cell.
        // k6, B4 assembly (b2=0, so k2 is not needed for B4), and error assembly are fused into
        // one ParallelFor per direction. B4 is updated over ghost+valid cells; error is written
        // only for valid cells (err_scratch has no ghost), guarded by a box check.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
            Array4<Real> const& By = Bfield[lev][1]->array(mfi);
            Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
            Array4<Real> const& Kx = K[0].array(mfi);
            Array4<Real> const& Ky = K[1].array(mfi);
            Array4<Real> const& Kz = K[2].array(mfi);
            Array4<Real> const& error_x = err_scratch[0].array(mfi);
            Array4<Real> const& error_y = err_scratch[1].array(mfi);
            Array4<Real> const& error_z = err_scratch[2].array(mfi);
            Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
            Array4<Real const> const& By_old = B_old[1].const_array(mfi);
            Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
            Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect());
            Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect());
            Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect());
            Box const& tjx_ng = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
            Box const& tjy_ng = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
            Box const& tjz_ng = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
            amrex::ParallelFor(tjx_ng, tjy_ng, tjz_ng,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kx(i, j, k, 0);
                    amrex::Real const k2 = Kx(i, j, k, 1);
                    amrex::Real const k3 = Kx(i, j, k, 2);
                    amrex::Real const k4 = Kx(i, j, k, 3);
                    amrex::Real const k5 = Kx(i, j, k, 4);
                    amrex::Real const k6 = Bx(i, j, k) - Bx_old(i, j, k)
                                         - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                    Kx(i, j, k, 1) = k6;
                    Bx(i, j, k) = Bx_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                    if (tjx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                        error_x(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Ky(i, j, k, 0);
                    amrex::Real const k2 = Ky(i, j, k, 1);
                    amrex::Real const k3 = Ky(i, j, k, 2);
                    amrex::Real const k4 = Ky(i, j, k, 3);
                    amrex::Real const k5 = Ky(i, j, k, 4);
                    amrex::Real const k6 = By(i, j, k) - By_old(i, j, k)
                                         - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                    Ky(i, j, k, 1) = k6;
                    By(i, j, k) = By_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                    if (tjy.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                        error_y(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    amrex::Real const k1 = Kz(i, j, k, 0);
                    amrex::Real const k2 = Kz(i, j, k, 1);
                    amrex::Real const k3 = Kz(i, j, k, 2);
                    amrex::Real const k4 = Kz(i, j, k, 3);
                    amrex::Real const k5 = Kz(i, j, k, 4);
                    amrex::Real const k6 = Bz(i, j, k) - Bz_old(i, j, k)
                                         - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                    Kz(i, j, k, 1) = k6;
                    Bz(i, j, k) = Bz_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                    if (tjz.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                        error_z(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                    }
                }
            );
        }

        // ---- Error norm and adaptive step control ----
        // Compute local maxima first, then one combined AllReduce for both norms.
        amrex::Real err_norm = 0._rt;
        amrex::Real B4_norm  = 0._rt;
        for (int ii = 0; ii < 3; ii++) {
            err_norm = std::max(err_norm, err_scratch[ii].norm0(/*comp=*/0, /*nghost=*/0, /*local=*/true));
            B4_norm  = std::max(B4_norm,  Bfield[lev][ii]->norm0(/*comp=*/0, /*nghost=*/0, /*local=*/true));
        }
        amrex::ParallelDescriptor::ReduceRealMax({err_norm, B4_norm});
        const amrex::Real err_scalar = err_norm / (m_substep_atol + m_substep_rtol * B4_norm);
        const amrex::Real factor = m_substep_safety * std::pow(err_scalar + 1.e-10_rt, -0.2_rt);

        if (err_scalar <= 1._rt) {
            t += dt_sub;
            ++n_accepted;
            for (int ii = 0; ii < 3; ii++) {
                MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);
            }
            dt_sub *= std::min(m_substep_max_growth, factor);
        } else {
            for (int ii = 0; ii < 3; ii++) {
                MultiFab::Copy(*Bfield[lev][ii], B_old[ii], 0, 0, 1, ng);
            }
            dt_sub *= std::max(0.1_rt, factor);
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ++n_attempts <= m_max_substep_attempts,
            "BfieldEvolveRKF45: exceeded max substep attempts; "
            "consider relaxing hybrid_pic_model.substep_rtol/substep_atol."
        );
    }

    // Set the number of substeps such that dt_sub on the next step will be similar
    // to what was found to work in this step
    m_substeps = 2*n_accepted;

    if (WarpX::GetInstance().Verbose()) {
        amrex::Print() << "RKF45 "
            << (subcycling_half == SubcyclingHalf::FirstHalf ? "1st" : "2nd") << " half"
            << ": " << n_accepted << " accepted, "
            << (n_attempts - n_accepted) << " rejected substeps"
            << " (dt_sub_final/dt_half = " << dt_sub / dt_half << ")\n";
    }
}


void HybridPICModel::FieldPush (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    amrex::Real dt, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();

    amrex::Real const t_old = warpx.gett_old(0);

    // Calculate J = curl x B / mu0 - J_ext
    CalculatePlasmaCurrent(Bfield, eb_update_E);
    // Calculate the E-field from Ohm's law
    HybridPICSolveE(Efield, Jfield, Bfield, rhofield, eb_update_E, true);
    // Refresh the E ghosts before the Faraday push reads them. This is always
    // required on a collocated grid (the masked nodal curl reads ghost E). On a
    // staggered grid with the conformal embedded boundary, the ECT circulation
    // (EvolveECTRho below) — and its along-edge curvature correction when enabled —
    // also reads cross-box ghost E edges, which the 1-ghost ECTRhofield
    // FillBoundary only refreshes *after* each face's Rho is formed; fill the E
    // ghosts here first so the per-face circulation is seam-consistent.
    bool fill_E_pre_faraday = (Bz_IndexType[0] == Ez_IndexType[0]);
#ifdef AMREX_USE_EB
    fill_E_pre_faraday = fill_E_pre_faraday ||
        (m_use_conformal_eb &&
         WarpX::grid_type != ablastr::utils::enums::GridType::Collocated);
#endif
    if (fill_E_pre_faraday) {
        warpx.FillBoundaryE(ng, nodal_sync);
    }

    // Apply the Marder cleanup to the substep E before curl(E) so the Faraday push
    // integrates the corrected field. No-op unless marder_correction_level = all_substeps;
    // the cadence is set by marder_substep_interval.
    MarderCorrectE(Efield, Jfield, Bfield, rhofield, eb_update_E,
                   MarderSite::Substep);

#ifdef AMREX_USE_EB
    // With the conformal embedded-boundary update on a staggered grid, recompute the
    // face-centered electromotive-force density (ECTRhofield) from the new E-field so
    // that the following Faraday push uses circulations consistent with Ohm's law. The
    // collocated conformal path uses the masked nodal Faraday curl instead (no ECT
    // circulations), so this staggered-only precompute is skipped there.
    if (m_use_conformal_eb &&
        WarpX::grid_type != ablastr::utils::enums::GridType::Collocated) {
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            // Opt-in along-edge curvature correction of the circulation: shift each
            // cut edge's E to the uncovered-segment centroid (see EvolveECTRho).
            auto edge_cent_offset = warpx.m_fields.get_alldirs(FieldType::edge_cent_offset, lev);
            warpx.get_pointer_fdtd_solver_fp(lev)->EvolveECTRho(
                Efield[lev],
                warpx.m_fields.get_alldirs(FieldType::edge_lengths, lev),
                warpx.m_fields.get_alldirs(FieldType::face_areas, lev),
                warpx.m_fields.get_alldirs(FieldType::ECTRhofield, lev),
                lev,
                &edge_cent_offset,
                m_conformal_ect_curvature);
        }
    }
#endif

    // Push forward the B-field using Faraday's law
    warpx.EvolveB(dt, subcycling_half, t_old);
    warpx.FillBoundaryB(ng, nodal_sync);

#ifdef AMREX_USE_EB
    // The collocated push is the unmasked nodal curl, with no enlarged-cell
    // extension to impose the wall condition on B during the Faraday push.
    // Impose the wall condition on the freshly pushed B directly from the level
    // set so the masked covered nodes are wall-consistent for the next substep's
    // curl(B) plasma current. (The FillBoundaryB above gives the gather stencils
    // valid ghost values; a second exchange below propagates the band/covered
    // values.)
    if (m_use_conformal_eb && !m_conformal_b_off &&
        WarpX::grid_type == ablastr::utils::enums::GridType::Collocated) {
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            // The direct level-set mirror fill (magnetic parity, normal odd /
            // tangential even, flux-excluding PEC). It is self-consistent with
            // the Neumann-filled covered band set on the initial field
            // (InitialBEBFill), so the covered band stays at its wall value
            // through the substepping.
            auto const& eb_update_B = warpx.GetEBUpdateBFlag();
            if (static_cast<int>(m_eb_bc_status_B.size()) <= lev) { m_eb_bc_status_B.resize(lev+1); }
            warpx::hybrid::ApplyPECBoundaryToField(
                warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev),
                eb_update_B[lev],
                *warpx.m_fields.get(FieldType::distance_to_eb, lev),
                warpx.Geom(lev),
                m_eb_bc_rtol, m_eb_bc_max_iters, m_eb_bc_direct_fill,
                /*normal_odd=*/true, /*fill_covered_centers=*/false,
                &m_eb_bc_status_B[lev], m_eb_b_fill_band_cells,
                m_eb_cylindrical_correction, m_eb_cyl_axis);
            // Optional diffusive clean: dissipate the curved-wall div(B) the
            // mirror injects (a pure-gradient correction, so curl(B)=J is
            // untouched). Skipped in the per-step cadence (done once at the
            // FullStep site).
            if (m_divb_clean_alpha > 0.0_rt && !m_divb_clean_per_step) {
                MarderCleanDivergence(
                    warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev),
                    warpx.GetEBUpdateBFlag()[lev], &m_eb_bc_status_B[lev],
                    /*normal_odd=*/true, /*fill_covered_centers=*/false,
                    m_divb_clean_alpha, m_divb_clean_iters,
                    m_divb_clean_band_cells, m_eb_b_fill_band_cells, lev);
            }
        }
        warpx.FillBoundaryB(ng, nodal_sync);
    }
#endif
}
