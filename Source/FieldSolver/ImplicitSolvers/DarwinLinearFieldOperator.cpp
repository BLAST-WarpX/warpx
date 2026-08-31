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
using namespace amrex::literals;

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

    // Every stencil evaluated in apply() (two curls and one Laplacian) only
    // reads nearest neighbours, so a single ghost cell suffices for each
    // intermediate that is differentiated - their guard cells are refilled in
    // between. This is the scratch's own width, unrelated to Z's (which is
    // always zero).
    const amrex::IntVect stencil_ng = amrex::IntVect(1);
    m_Zscratch_x.define(Zvec[lev][0]->boxArray(), Zvec[lev][0]->DistributionMap(),
                        Zvec[lev][0]->nComp(), stencil_ng);
    m_Zscratch_y.define(Zvec[lev][1]->boxArray(), Zvec[lev][1]->DistributionMap(),
                        Zvec[lev][1]->nComp(), stencil_ng);
    m_Zscratch_z.define(Zvec[lev][2]->boxArray(), Zvec[lev][2]->DistributionMap(),
                        Zvec[lev][2]->nComp(), stencil_ng);

    m_curlcurlZ_x.define(Zvec[lev][0]->boxArray(), Zvec[lev][0]->DistributionMap(),
                         Zvec[lev][0]->nComp(), stencil_ng);
    m_curlcurlZ_y.define(Zvec[lev][1]->boxArray(), Zvec[lev][1]->DistributionMap(),
                         Zvec[lev][1]->nComp(), stencil_ng);
    m_curlcurlZ_z.define(Zvec[lev][2]->boxArray(), Zvec[lev][2]->DistributionMap(),
                         Zvec[lev][2]->nComp(), stencil_ng);

    // Nothing differentiates the Laplacian result, so it needs no guard cells.
    m_lap_curlcurlZ_x.define(Zvec[lev][0]->boxArray(), Zvec[lev][0]->DistributionMap(),
                             Zvec[lev][0]->nComp(), Zvec[lev][0]->nGrowVect());
    m_lap_curlcurlZ_y.define(Zvec[lev][1]->boxArray(), Zvec[lev][1]->DistributionMap(),
                             Zvec[lev][1]->nComp(), Zvec[lev][1]->nGrowVect());
    m_lap_curlcurlZ_z.define(Zvec[lev][2]->boxArray(), Zvec[lev][2]->DistributionMap(),
                             Zvec[lev][2]->nComp(), Zvec[lev][2]->nGrowVect());

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
    //   a_Ax = -laplacian(curl(curl(a_x))) + curl(chi curl(a_x))
    // where chi is the mass matrix scaled by 2 * mu_0 / dt (see
    // SemiImplicitDarwin::ApplyScaledMassMatrices). Both terms are built from
    // the same intermediate curl(Z), which is also the vector-potential
    // increment dA that the solve is ultimately after.

    const int lev = 0;
    const int ncomps = 1;

    WarpX* const warpx_ptr = m_ops->GetWarpX();

    const auto& Zvec = a_x.getArrayVec();
    auto& rhs_vec = a_Ax.getArrayVec();

    // The dA_fp and Efield_fp MultiFabs are used to store intermediate calculations.
    auto dA_fp = warpx_ptr->m_fields.get_mr_levels_alldirs(FieldType::dA_fp, lev);
    auto E_temp = warpx_ptr->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, lev);

    // Scratch space (allocated once in define()): curl(curl(Z)) and its Laplacian,
    // both on Z's (B) staggering.
    ablastr::fields::VectorField curlcurlZ = {&m_curlcurlZ_x, &m_curlcurlZ_y, &m_curlcurlZ_z};
    ablastr::fields::VectorField lap_curlcurlZ =
        {&m_lap_curlcurlZ_x, &m_lap_curlcurlZ_y, &m_lap_curlcurlZ_z};

    // WarpXSolverVec always allocates with zero guard cells (see its Define()),
    // so a_x has none at all - there is nothing to fill in place, and the
    // stencils below read beyond the valid region. Copy the candidate into a
    // B-staggered scratch (allocated once in define(), with one ghost cell for
    // the curl stencil below, which reads i-1) and FillBoundary on that
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

    // Calculate dA = curl(Z)
    // Use Zscratch (guard cells already filled above) rather than Zvec directly.
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeCurlB(
        dA_fp[lev], Zscratch, warpx_ptr->GetEBUpdateEFlag()[lev], lev
    );

    // include guard cells. dA_fp is E-staggered: use FillBoundaryAndSync so
    // periodically wrapped cells agree before it is differentiated (for the
    // curl-curl term) and before the mass matrices are applied to it below.
    for (int ii = 0; ii < 3; ii++)
    {
        dA_fp[lev][ii]->FillBoundaryAndSync(warpx_ptr->Geom(lev).periodicity());
        // clear E_temp since ApplyScaledMassMatrices accumulates into its rhs argument
        E_temp[lev][ii]->setVal(0);
    }

    // Fourth-order term: curl(curl(Z)) = curl(dA), back on Z's staggering,
    // followed by its (vector) Laplacian. Fill the intermediate's guard cells
    // in between, since the Laplacian stencil reads i-1..i+1.
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
        curlcurlZ, dA_fp[lev], warpx_ptr->GetEBUpdateBFlag()[lev], lev
    );
    for (int ii = 0; ii < 3; ii++)
    {
        curlcurlZ[ii]->FillBoundaryAndSync(warpx_ptr->Geom(lev).periodicity());
    }
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeVectorLaplacian(
        lap_curlcurlZ, curlcurlZ, warpx_ptr->GetEBUpdateBFlag()[lev], lev
    );

    // Calculate chi dA (the scaled mass matrices applied to dA) and write into E_temp
    m_ops->ApplyScaledMassMatrices(E_temp, dA_fp);

    // E_temp (Efield_fp) shares dA_fp's staggering (nodal transverse
    // components) - sync it too before ComputeCurlA reads it with a stencil.
    for (int ii = 0; ii < 3; ii++)
    {
        E_temp[lev][ii]->FillBoundaryAndSync(warpx_ptr->Geom(lev).periodicity());
    }

    // Plasma response term: curl(E_temp) = curl(chi curl(Z)), written straight
    // into the operator's output ...
    warpx_ptr->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
        rhs_vec[lev], E_temp[lev], warpx_ptr->GetEBUpdateBFlag()[lev], lev
    );

    // ... and then subtract the fourth-order term to complete
    // -laplacian(curl(curl(Z))) + curl(chi curl(Z)).
    for (int ii = 0; ii < 3; ii++)
    {
        amrex::MultiFab::Saxpy(
            *rhs_vec[lev][ii], -1.0_rt, *lap_curlcurlZ[ii], 0, 0, ncomps, 0
        );
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
