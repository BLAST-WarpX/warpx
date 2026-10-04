/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "Fields.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <string>

using warpx::fields::FieldType;

/* The functions in this file define, in a single place, the sequence of
 * operations that is applied to the charge and current densities after the
 * particles have deposited them, and before they are used by the field
 * solvers or written by the diagnostics:
 *
 *  1. filtering (if used),
 *  2. summation of the guard cells across boxes, MPI ranks and
 *     mesh-refinement levels,
 *  3. application of the boundary conditions: the charge and current
 *     deposited in the guard cells beyond PEC, PMC and reflecting boundaries
 *     are folded back into the domain.
 *
 * In cylindrical and spherical geometry, the inverse volume scaling of the
 * deposited quantities (which also folds the deposition at negative radius
 * onto the cells above the axis) is applied by the deposition routines
 * themselves, right after the deposition and before these functions are
 * called.
 *
 * Every code path that deposits charge or current (main PIC loop,
 * electrostatic and magnetostatic solvers, hybrid-PIC, diagnostics, Python
 * interface) is expected to go through these functions, so that the order
 * of the operations above is the same everywhere.
 */

void
WarpX::FinalizeDepositedCharge (
    const ablastr::fields::MultiLevelScalarField& rho_fp,
    const ablastr::fields::MultiLevelScalarField& rho_cp,
    const ablastr::fields::MultiLevelScalarField& rho_buf)
{
    ABLASTR_PROFILE("WarpX::FinalizeDepositedCharge()");

    if (rho_fp.empty() || !rho_fp[0]) { return; }

    // Filter (if used), sum the guard cells across boxes and MPI ranks,
    // and interpolate across mesh-refinement levels
    SyncRho(rho_fp, rho_cp, rho_buf);

    // Apply the boundary conditions: fold the charge deposited in the guard
    // cells beyond PEC, PMC and reflecting boundaries back into the domain
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (rho_fp[lev]) {
            ApplyRhofieldBoundary(lev, rho_fp[lev], PatchType::fine);
        }
        if (lev > 0 && lev < static_cast<int>(rho_cp.size()) && rho_cp[lev]) {
            ApplyRhofieldBoundary(lev, rho_cp[lev], PatchType::coarse);
        }
    }
}

void
WarpX::FinalizeDepositedCharge ()
{
    bool const skip_lev0_coarse_patch = true;
    const ablastr::fields::MultiLevelScalarField rho_fp = m_fields.has(FieldType::rho_fp, 0) ?
        m_fields.get_mr_levels(FieldType::rho_fp, finest_level) :
        ablastr::fields::MultiLevelScalarField{static_cast<size_t>(finest_level+1)};
    const ablastr::fields::MultiLevelScalarField rho_cp = m_fields.has(FieldType::rho_cp, 1) ?
        m_fields.get_mr_levels(FieldType::rho_cp, finest_level, skip_lev0_coarse_patch) :
        ablastr::fields::MultiLevelScalarField{static_cast<size_t>(finest_level+1)};
    const ablastr::fields::MultiLevelScalarField rho_buf = m_fields.has(FieldType::rho_buf, 1) ?
        m_fields.get_mr_levels(FieldType::rho_buf, finest_level, skip_lev0_coarse_patch) :
        ablastr::fields::MultiLevelScalarField{static_cast<size_t>(finest_level+1)};

    FinalizeDepositedCharge(rho_fp, rho_cp, rho_buf);
}

void
WarpX::FinalizeDepositedCharge (amrex::MultiFab& rho, const int lev)
{
    ABLASTR_PROFILE("WarpX::FinalizeDepositedCharge(single level)");

    // Filter (if used) and sum the guard cells across boxes and MPI ranks
    ApplyFilterandSumBoundaryRho(lev, lev, rho, 0, rho.nComp());

    // Apply the boundary conditions: fold the charge deposited in the guard
    // cells beyond PEC, PMC and reflecting boundaries back into the domain
    ApplyRhofieldBoundary(lev, &rho, PatchType::fine);
}

void
WarpX::FinalizeDepositedCurrent (const std::string& current_fp_string)
{
    using ablastr::fields::Direction;

    ABLASTR_PROFILE("WarpX::FinalizeDepositedCurrent()");

    // Center the current from the nodal to the staggered grid (if current
    // centering is used), filter (if used), sum the guard cells across boxes
    // and MPI ranks, and interpolate across mesh-refinement levels
    SyncCurrent(current_fp_string);

    // With current centering, SyncCurrent stores the centered current in
    // current_fp: the boundary conditions are applied to that field.
    const std::string J_fp_string = (do_current_centering) ? std::string("current_fp") : current_fp_string;
    const ablastr::fields::MultiLevelVectorField J_fp =
        m_fields.get_mr_levels_alldirs(J_fp_string, finest_level);

    // Apply the boundary conditions: fold the current deposited in the guard
    // cells beyond PEC, PMC and reflecting boundaries back into the domain
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        ApplyJfieldBoundary(lev,
            J_fp[lev][Direction{0}], J_fp[lev][Direction{1}], J_fp[lev][Direction{2}],
            PatchType::fine);
        if (lev > 0 && m_fields.has_vector(FieldType::current_cp, lev)) {
            ApplyJfieldBoundary(lev,
                m_fields.get(FieldType::current_cp, Direction{0}, lev),
                m_fields.get(FieldType::current_cp, Direction{1}, lev),
                m_fields.get(FieldType::current_cp, Direction{2}, lev),
                PatchType::coarse);
        }
    }
}

void
WarpX::FinalizeDepositedCurrent (const ablastr::fields::VectorField& J, const int lev)
{
    ABLASTR_PROFILE("WarpX::FinalizeDepositedCurrent(single level)");

    // The filtering and guard-cell-summation routines expect a multi-level
    // container: wrap the single-level field into one.
    ablastr::fields::MultiLevelVectorField J_ml(lev+1);
    J_ml[lev] = J;

    // Filter (if used) and sum the guard cells across boxes and MPI ranks
    if (use_filter) {
        ApplyFilterJ(J_ml, lev);
    }
    SumBoundaryJ(J_ml, lev, Geom(lev).periodicity());

    // Apply the boundary conditions: fold the current deposited in the guard
    // cells beyond PEC, PMC and reflecting boundaries back into the domain
    ApplyJfieldBoundary(lev, J[0], J[1], J[2], PatchType::fine);
}
