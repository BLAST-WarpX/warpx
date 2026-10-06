/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (Realta Fusion)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "DarwinLinearFieldOperator.H"

#include "Fields.H"
#include "SemiImplicitDarwin.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include "NonlinearSolvers/DarwinMLMGPC.H"

#include <AMReX_MultiFab.H>

using warpx::fields::FieldType;

void DarwinLinearFieldOperator::define ( const WarpXSolverVec& a_U,
                                         SemiImplicitDarwin* a_ops,
                                         const PreconditionerType& a_pc_type )
{
    BL_PROFILE("DarwinLinearFieldOperator::define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_pc_type == PreconditionerType::none ||
        a_pc_type == PreconditionerType::pc_darwin_mlmg,
        "DarwinLinearFieldOperator::define(): the only preconditioner supported "
        "by the Darwin solver is pc_darwin_mlmg (or none for no preconditioning)");

    m_R.Define(a_U);
    m_ops = a_ops;

    m_pc_type = a_pc_type;
    if (m_pc_type == PreconditionerType::pc_darwin_mlmg) {
        m_preCond = std::make_unique<DarwinMLMGPC<WarpXSolverVec,SemiImplicitDarwin>>();
        m_preCond->Define(a_U, a_ops);
    }

    // Allocate the scratch space used by apply() once here (every iterate of
    // Z shares this same layout) rather than on every GMRES iteration. This
    // requires the mass matrices to have been initialized already, since the
    // ghost widths below are derived from the mass matrix stencil.
    const auto& Zvec = a_U.getArrayVec();
    const int lev = 0;

    // Guard-cell budget for evaluating
    //     bilaplacian(Z) + curl(chi curl(Z))
    // from a single exchange on Zscratch. Walking the chain backwards from the
    // output:
    //  - the final curl reads chi curl(Z) at i..i+1, so that intermediate is
    //    needed one cell beyond the valid region;
    //  - the mass matrix gather reaches half_width cells, and
    //    ApplyMassMatrices() silently clamps it to the guard region of its
    //    input, so curl(Z) must be valid that much further out again;
    //  - curl(Z) reads Z at i-1..i, so Z is needed one cell beyond that;
    //  - independently, the direct nabla^4 stencil reads Z at i-2..i+2.
    // This is the same trick the hybrid-PIC solver uses to avoid communicating
    // after its curl (see GuardCellManager).
    const amrex::IntVect mm_half_width = m_ops->MassMatricesStencilHalfWidth();
    const amrex::IntVect chidA_ng = amrex::IntVect(1);
    const amrex::IntVect dA_ng    = chidA_ng + mm_half_width;
    // (These are the scratch's own widths; Z's is always zero.)
    const amrex::IntVect Z_ng     = amrex::elemwiseMax(amrex::IntVect(2), dA_ng + 1);

    m_lapZ_x.define(Zvec[lev][0]->boxArray(), Zvec[lev][0]->DistributionMap(),
                    Zvec[lev][0]->nComp(), Zvec[lev][0]->nGrowVect());
    m_lapZ_y.define(Zvec[lev][1]->boxArray(), Zvec[lev][1]->DistributionMap(),
                    Zvec[lev][1]->nComp(), Zvec[lev][1]->nGrowVect());
    m_lapZ_z.define(Zvec[lev][2]->boxArray(), Zvec[lev][2]->DistributionMap(),
                    Zvec[lev][2]->nComp(), Zvec[lev][2]->nGrowVect());

    m_Zscratch_x.define(Zvec[lev][0]->boxArray(), Zvec[lev][0]->DistributionMap(),
                        Zvec[lev][0]->nComp(), Z_ng);
    m_Zscratch_y.define(Zvec[lev][1]->boxArray(), Zvec[lev][1]->DistributionMap(),
                        Zvec[lev][1]->nComp(), Z_ng);
    m_Zscratch_z.define(Zvec[lev][2]->boxArray(), Zvec[lev][2]->DistributionMap(),
                        Zvec[lev][2]->nComp(), Z_ng);

    // curl(Z) and chi curl(Z) live on the E/A/J staggering
    WarpX* const warpx_ptr = m_ops->GetWarpX();
    const ablastr::fields::VectorField Efield =
        warpx_ptr->m_fields.get_alldirs(FieldType::Efield_fp, lev);
    m_dA_x.define(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, dA_ng);
    m_dA_y.define(Efield[1]->boxArray(), Efield[1]->DistributionMap(), 1, dA_ng);
    m_dA_z.define(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, dA_ng);

    m_chidA_x.define(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, chidA_ng);
    m_chidA_y.define(Efield[1]->boxArray(), Efield[1]->DistributionMap(), 1, chidA_ng);
    m_chidA_z.define(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, chidA_ng);

    // ApplyMassMatrices() only fills as many guard cells of its output as the
    // mass matrices themselves hold valid data for (they are summed out into
    // their guard region by SumBoundaryJ), so those have to cover the one cell
    // that the final curl reads.
    const ablastr::fields::VectorField SX =
        warpx_ptr->m_fields.get_alldirs(FieldType::MassMatrices_X, lev);
    const ablastr::fields::VectorField SY =
        warpx_ptr->m_fields.get_alldirs(FieldType::MassMatrices_Y, lev);
    const ablastr::fields::VectorField SZ =
        warpx_ptr->m_fields.get_alldirs(FieldType::MassMatrices_Z, lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        SX[0]->nGrowVect().allGE(chidA_ng) &&
        SY[1]->nGrowVect().allGE(chidA_ng) &&
        SZ[2]->nGrowVect().allGE(chidA_ng),
        "The semi-implicit Darwin solver requires at least one guard cell on the "
        "mass matrices (i.e. on the current density).");

    m_is_defined = true;
}

auto DarwinLinearFieldOperator::makeVecRHS () const -> WarpXSolverVec
{
    BL_PROFILE("DarwinLinearFieldOperator::makeVecRHS()");
    WarpXSolverVec x;
    x.Define(m_R);
    return x;
}

auto DarwinLinearFieldOperator::makeVecLHS () const -> WarpXSolverVec
{
    BL_PROFILE("DarwinLinearFieldOperator::makeVecLHS()");
    WarpXSolverVec x;
    x.Define(m_R);
    return x;
}

void DarwinLinearFieldOperator::apply ( WarpXSolverVec& a_Ax, const WarpXSolverVec& a_x )
{
    BL_PROFILE("DarwinLinearFieldOperator::apply()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        isDefined(),
        "DarwinLinearFieldOperator::apply() called on undefined DarwinLinearFieldOperator");

    // Computes the action of the Darwin field operator on the given vector:
    //   a_Ax = bilaplacian(a_x) + curl(chi curl(a_x))
    // where chi is the mass matrix scaled by 2 * mu_0 / dt (see
    // SemiImplicitDarwin::ApplyScaledMassMatrices).
    //
    // Every stage of that expression is a short, local stencil, so the whole
    // operator is one banded stencil on Z. Rather than exchanging guard cells
    // between the stages, each intermediate is evaluated over enough of its own
    // guard region to feed the next one (widths are set in define()), and the
    // only communication is the single exchange on Zscratch below. Besides
    // being cheaper, this removes a class of bug: an intermediate exchange
    // reconciles the duplicated periodic-image cells of the nodal components
    // independently at each stage, which is exactly what made the two-pass
    // laplacian(laplacian(Z)) develop a parasitic sign-alternating mode.

    const int lev = 0;
    const int ncomps = 1;

    WarpX* const warpx_ptr = m_ops->GetWarpX();

    const auto& Zvec = a_x.getArrayVec();
    auto& rhs_vec = a_Ax.getArrayVec();

    // Scratch space, all allocated once in define().
    ablastr::fields::VectorField Zscratch = {&m_Zscratch_x, &m_Zscratch_y, &m_Zscratch_z};
    ablastr::fields::VectorField lapZ = {&m_lapZ_x, &m_lapZ_y, &m_lapZ_z};
    ablastr::fields::MultiLevelVectorField dA = {{&m_dA_x, &m_dA_y, &m_dA_z}};
    ablastr::fields::MultiLevelVectorField chidA = {{&m_chidA_x, &m_chidA_y, &m_chidA_z}};

    // WarpXSolverVec always allocates with zero guard cells (see its Define()),
    // so a_x has none at all - there is nothing to fill in place, and the
    // stencils below read beyond the valid region. Copy the candidate into a
    // B-staggered scratch (allocated once in define(), wide enough to feed the
    // whole stencil chain) and FillBoundary on that scratch, which derives its
    // guard cells from its own (just-copied) valid-region data via the periodic
    // halo exchange.
    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::Copy(*Zscratch[ii], *Zvec[lev][ii], 0, 0, ncomps, 0);
        // Plain FillBoundary only reconciles true ghost cells, not two
        // overlapping valid cells - use FillBoundaryAndSync instead (harmless
        // no-op for the transverse, cell-centered components, which have no
        // such duplication).
        Zscratch[ii]->FillBoundaryAndSync(warpx_ptr->Geom(lev).periodicity());
    }

    // Evaluation of the (single) 4th-order field equation:
    // bilaplacian(Z), discretized directly in a single pass over Z.
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeVectorBiLaplacian(
        rhs_vec[lev], Zscratch, warpx_ptr->GetEBUpdateBFlag()[lev], lev
    );

    // Calculate dA = curl(Z), out into dA's full guard region so that the mass
    // matrix gather below never runs off the end of it
    // (ComputeCurlB resets dA to zero internally).
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeCurlB(
        dA[lev], Zscratch, warpx_ptr->GetEBUpdateEFlag()[lev], lev, m_dA_x.nGrowVect()
    );

    // Calculate chi dA (the scaled mass matrices applied to dA) and write into
    // chidA. ApplyScaledMassMatrices() accumulates into its output and fills as
    // many of its guard cells as it has (one, here - just what the final curl
    // reads), so clear it first including guard cells.
    for (int ii = 0; ii < 3; ii++)
    {
        chidA[lev][ii]->setVal(0);
    }
    m_ops->ApplyScaledMassMatrices(chidA, dA);

    // Reuse lapZ as a temporary storage location for curl(chi curl(Z))
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
        lapZ, chidA[lev], warpx_ptr->GetEBUpdateBFlag()[lev], lev
    );

    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::Add(*rhs_vec[lev][ii], *lapZ[ii], 0, 0, ncomps, 0);
    }

    // No exchange is needed on rhs_vec, the operator's own output (B-staggered,
    // matching Z). GMRES's linComb/increment arithmetic is element-wise and has
    // no notion of the duplicated periodic-image cells of the nodal
    // component(s), so those two cells have to hold the same value - but they
    // do so here by construction rather than by being reconciled after the
    // fact: everything above is evaluated from Zscratch, which was synced, and
    // from mass matrices that SumBoundaryJ leaves equal at duplicated nodes, so
    // the two cells are computed by identical operations on identical inputs.
}
