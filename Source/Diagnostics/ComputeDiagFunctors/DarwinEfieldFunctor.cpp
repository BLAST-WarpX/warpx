/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (Realta Fusion)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "DarwinEfieldFunctor.H"

#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;

DarwinEfieldFunctor::DarwinEfieldFunctor (
    const amrex::MultiFab* mf_Efield,
    const amrex::MultiFab* mf_dA,
    const int lev,
    const amrex::IntVect crse_ratio,
    const int ncomp
)
    : ComputeDiagFunctor(ncomp, crse_ratio),
      m_mf_Efield(mf_Efield), m_mf_dA(mf_dA), m_lev(lev)
{}

void
DarwinEfieldFunctor::operator() (amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_mf_Efield != nullptr && m_mf_dA != nullptr,
        "m_mf_Efield/m_mf_dA can't be nullptr.");

    auto& warpx = WarpX::GetInstance();

    // SemiImplicitDarwin::OneStep() leaves Efield_fp (and therefore Efield_aux,
    // which is derived from it) holding only the electrostatic component plus
    // any external field at this point in the step; the inductive component
    // (E = -dA/dt) is recovered from dA_fp, which
    // SemiImplicitDarwin::ComputeInductiveEfromdA() computes once per step and
    // which nothing overwrites before this diagnostic runs.
    amrex::MultiFab total_E(m_mf_Efield->boxArray(), m_mf_Efield->DistributionMap(),
                            m_mf_Efield->nComp(), 0);
    const amrex::Real inv_dt = -1.0_rt / warpx.getdt(m_lev);
    amrex::MultiFab::LinComb(
        total_E, 1.0_rt, *m_mf_Efield, 0, inv_dt, *m_mf_dA, 0,
        0, m_mf_Efield->nComp(), 0);

    InterpolateMFForDiag(mf_dst, total_E, dcomp, warpx.DistributionMap(m_lev), false);
}
