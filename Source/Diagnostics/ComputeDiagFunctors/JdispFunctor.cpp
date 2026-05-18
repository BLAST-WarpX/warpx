/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Avigdor Veksler (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "JdispFunctor.H"

#include "WarpX.H"
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"

#include <AMReX.H>
#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

using namespace amrex;
using warpx::fields::FieldType;

JdispFunctor::JdispFunctor (int dir, int lev,
        amrex::IntVect crse_ratio, bool convertRZmodes2cartesian, int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_dir(dir), m_lev(lev),
    m_convertRZmodes2cartesian(convertRZmodes2cartesian)
{ }

void
JdispFunctor::operator() (amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{
    using ablastr::fields::Direction;

    auto& warpx = WarpX::GetInstance();
    auto* hybrid_pic_model = warpx.get_pointer_HybridPICModel();

    /** pointer to total simulation current (J) multifab */
    amrex::MultiFab* mf_j = warpx.m_fields.get(FieldType::current_fp, Direction{m_dir}, m_lev);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(hybrid_pic_model,
        "Displacement current diagnostic is only implemented for the HybridPICModel.");
    AMREX_ASSUME(hybrid_pic_model != nullptr);

     /** pointer to current calculated from Ampere's Law (Jamp) multifab */
    amrex::MultiFab* mf_curlB = warpx.m_fields.get(FieldType::hybrid_current_fp_plasma, Direction{m_dir}, m_lev);

    //if (!hybrid_pic_model) {
        // To finish this implementation, we need to implement a method to
        // calculate (∇ x B).

        // Skeleton for future implementation for solvers other than HybridPIC.
        // Get curlB multifab

        // Divide curlB multifab by mu0 to get units of current
        // mf_curlB->mult(1.0/PhysConsts::mu0)
    //}

    // A Jdisp multifab is generated to hold displacement current.
    amrex::MultiFab Jdisp( mf_j->boxArray(), mf_j->DistributionMap(), 1, mf_j->nGrowVect() );
    Jdisp.setVal(0);

    // J_displacement = curl x B / mu0 - J
    amrex::MultiFab::LinComb(
        Jdisp, 1, *mf_curlB, 0,
        -1, *mf_j, 0, 0, 1, Jdisp.nGrowVect()
    );

    InterpolateMFForDiag(mf_dst, Jdisp, dcomp, warpx.DistributionMap(m_lev),
                         m_convertRZmodes2cartesian);
}
