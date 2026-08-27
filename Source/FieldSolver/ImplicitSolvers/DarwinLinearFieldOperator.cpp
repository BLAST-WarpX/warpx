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

#include <AMReX_MultiFab.H>

using warpx::fields::FieldType;

void DarwinLinearFieldOperator::define ( const WarpXSolverVec& a_U,
                                         SemiImplicitDarwin* a_ops,
                                         const PreconditionerType& a_pc_type )
{
    BL_PROFILE("DarwinLinearFieldOperator::define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_pc_type == PreconditionerType::none,
        "DarwinLinearFieldOperator::define(): preconditioners are not supported");

    m_R.Define(a_U);
    m_ops = a_ops;

    // Allocate the scratch space used by apply() once here (every iterate of
    // Z shares this same layout) rather than on every GMRES iteration.
    const auto& Zvec = a_U.getArrayVec();
    const int lev = 0;
    m_lapZ_x.define(Zvec[lev][0]->boxArray(), Zvec[lev][0]->DistributionMap(),
                    Zvec[lev][0]->nComp(), Zvec[lev][0]->nGrowVect());
    m_lapZ_y.define(Zvec[lev][1]->boxArray(), Zvec[lev][1]->DistributionMap(),
                    Zvec[lev][1]->nComp(), Zvec[lev][1]->nGrowVect());
    m_lapZ_z.define(Zvec[lev][2]->boxArray(), Zvec[lev][2]->DistributionMap(),
                    Zvec[lev][2]->nComp(), Zvec[lev][2]->nGrowVect());

    // 2 ghost cells for the nabla^4 stencil in apply(), which reads i-2..i+2.
    // This is the scratch's own width, unrelated to Z's (which is always zero).
    const amrex::IntVect biharmonic_ng = amrex::IntVect(2);
    m_Zscratch_x.define(Zvec[lev][0]->boxArray(), Zvec[lev][0]->DistributionMap(),
                        Zvec[lev][0]->nComp(), biharmonic_ng);
    m_Zscratch_y.define(Zvec[lev][1]->boxArray(), Zvec[lev][1]->DistributionMap(),
                        Zvec[lev][1]->nComp(), biharmonic_ng);
    m_Zscratch_z.define(Zvec[lev][2]->boxArray(), Zvec[lev][2]->DistributionMap(),
                        Zvec[lev][2]->nComp(), biharmonic_ng);

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

    const int lev = 0;
    const int ncomps = 1;

    WarpX* const warpx_ptr = m_ops->GetWarpX();

    const auto& Zvec = a_x.getArrayVec();
    auto& rhs_vec = a_Ax.getArrayVec();

    // The dA_fp and Efield_fp MultiFabs are used to store intermediate calculations.
    auto dA_fp = warpx_ptr->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, lev);
    auto E_temp = warpx_ptr->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, lev);

    // Scratch space (allocated once in define()), reused below to hold curl(chi curl(Z)).
    ablastr::fields::VectorField lapZ = {&m_lapZ_x, &m_lapZ_y, &m_lapZ_z};

    // WarpXSolverVec always allocates with zero guard cells (see its Define()),
    // so a_x has none at all - there is nothing to fill in place, and the
    // stencils below read beyond the valid region. Copy the candidate into a
    // B-staggered scratch (allocated once in define(), with 2 ghost cells for
    // the nabla^4 stencil below, which reads i-2..i+2) and FillBoundary on that
    // scratch, which derives its guard cells from its own (just-copied)
    // valid-region data via the periodic halo exchange.
    ablastr::fields::VectorField Zscratch = {&m_Zscratch_x, &m_Zscratch_y, &m_Zscratch_z};
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
    // bilaplacian(Z), discretized directly in a single pass.
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeVectorBiLaplacian(
        rhs_vec[lev], Zscratch, warpx_ptr->GetEBUpdateBFlag()[lev], lev
    );

    // Calculate dA = curl(Z)
    // Use Zscratch (guard cells already filled above) rather than Zvec directly.
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeCurlB(
        dA_fp[lev], Zscratch, warpx_ptr->GetEBUpdateEFlag()[lev], lev
    );

    // include guard cells. dA_fp is E-staggered: use FillBoundaryAndSync so
    // periodically wrapped cells agree before the mass matrices are applied to
    // dA_fp below.
    for (int ii = 0; ii < 3; ii++)
    {
        dA_fp[lev][ii]->FillBoundaryAndSync(warpx_ptr->Geom(lev).periodicity());
        // clear E_temp since ApplyScaledMassMatrices accumulates into its rhs argument
        E_temp[lev][ii]->setVal(0);
    }
    // Calculate chi dA (the scaled mass matrices applied to dA) and write into E_temp
    m_ops->ApplyScaledMassMatrices(E_temp, dA_fp);

    // E_temp (Efield_fp) shares dA_fp's staggering (nodal transverse
    // components) - sync it too before ComputeCurlA reads it with a stencil.
    for (int ii = 0; ii < 3; ii++)
    {
        E_temp[lev][ii]->FillBoundaryAndSync(warpx_ptr->Geom(lev).periodicity());
    }

    // Reuse lapZ as a temporary storage location for the curl(E)_temp = curl(chi curl(Z)_vec)
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
        lapZ, E_temp[lev], warpx_ptr->GetEBUpdateBFlag()[lev], lev
    );

    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::Add(*rhs_vec[lev][ii], *lapZ[ii], 0, 0, ncomps, 0);
    }

    // rhs_vec is the operator's own output (B-staggered, matching Z).
    // Nothing guarantees the stencil evaluations above produced identical
    // values at the duplicate periodic-image cells of the nodal
    // component(s), and GMRES's own linComb/increment arithmetic (used to
    // build every subsequent Krylov vector from this result) is
    // element-wise and has no notion of that duplication - so reconcile it
    // here before handing the result back.
    for (int ii = 0; ii < 3; ii++)
    {
        rhs_vec[lev][ii]->FillBoundaryAndSync(warpx_ptr->Geom(lev).periodicity());
    }
}
