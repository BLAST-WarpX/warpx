/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PhiFunctor.H"

#include <ablastr/constant.H>

#include "WarpX.H"
#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"


using warpx::fields::FieldType;

PhiFunctor::PhiFunctor (const int lev,
    const amrex::IntVect crse_ratio,
    bool convertRZmodes2cartesian, const int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio),
    m_lev(lev), m_convertRZmodes2cartesian(convertRZmodes2cartesian)
{ }

void
PhiFunctor::operator() (amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{
    auto& warpx = WarpX::GetInstance();

    // check if phi_fp exists in the multifab registry
    if (warpx.m_fields.has(FieldType::phi_fp, m_lev)) {
        // grab the global phi multifab
        amrex::MultiFab* global_phi = warpx.m_fields.get(FieldType::phi_fp, m_lev);

        InterpolateMFForDiag(
            mf_dst, *global_phi, dcomp, warpx.DistributionMap(m_lev),
            m_convertRZmodes2cartesian
        );
    }
    else
    {
        // allocate fields for charge and potential
        amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_vec(1);
        amrex::Vector<std::unique_ptr<amrex::MultiFab>> phi_vec(1);

        // calculate the total charge density as the divergence of the E-field
        const amrex::BoxArray& ba = warpx.boxArray(m_lev);
        rho_vec[0] = std::make_unique<amrex::MultiFab>(ba, warpx.DistributionMap(m_lev), 1, 1);
        rho_vec[0]->setVal(0.);
        warpx.ComputeDivE(*rho_vec[0], m_lev);
        // multiply divE by epsilon0 to get charge density
        rho_vec[0]->mult(ablastr::constant::SI::ep0);

        phi_vec[0] = std::make_unique<amrex::MultiFab>(ba, warpx.DistributionMap(m_lev), 1, 1);
        phi_vec[0]->setVal(0.);

        // grab the WarpX instance's electrostatic solver
        auto& es_solver = warpx.GetElectrostaticSolver();
        const std::array<amrex::Real, 3> beta = {0.0};
        es_solver.computePhi(
            amrex::GetVecOfPtrs(rho_vec), amrex::GetVecOfPtrs(phi_vec),
            beta, es_solver.self_fields_required_precision,
            es_solver.self_fields_absolute_tolerance,
            es_solver.self_fields_max_iters, es_solver.self_fields_verbosity,
            false
        );

        InterpolateMFForDiag(
            mf_dst, *phi_vec[0], dcomp, warpx.DistributionMap(m_lev),
            m_convertRZmodes2cartesian
        );
    }
}
