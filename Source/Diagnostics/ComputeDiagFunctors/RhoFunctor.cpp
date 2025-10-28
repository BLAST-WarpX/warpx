#include "RhoFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    #include "FieldSolver/SpectralSolver/SpectralFieldData.H"
    #include "FieldSolver/SpectralSolver/SpectralSolverRZ.H"
    #include "Utils/WarpXAlgorithmSelection.H"
#endif
#include "Particles/MultiParticleContainer.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <memory>

RhoFunctor::RhoFunctor (const int lev,
                        const amrex::IntVect crse_ratio,
                        bool apply_rz_psatd_filter,
                        const int species_index,
                        bool convertRZmodes2cartesian,
                        const int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio),
      m_lev(lev),
      m_apply_rz_psatd_filter(apply_rz_psatd_filter),
      m_species_index(species_index),
      m_convertRZmodes2cartesian(convertRZmodes2cartesian)
{}

void
RhoFunctor::operator() ( amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/ ) const
{
    auto& warpx = WarpX::GetInstance();
    std::unique_ptr<amrex::MultiFab> rho;

    // Deposit charge density
    // Call this with local=true since the parallel transfers will be handled
    // by ApplyFilterandSumBoundaryRho

    // Dump total rho
    if (m_species_index == -1) {
        auto& mypc = warpx.GetPartContainer();
        rho = mypc.GetChargeDensity(m_lev, true);
        if (warpx.DoFluidSpecies()) {
            auto& myfl = warpx.GetFluidContainer();
            myfl.DepositCharge(warpx.m_fields, *rho, m_lev);
        }
    }
    // Dump rho per species
    else {
        auto& mypc = warpx.GetPartContainer().GetParticleContainer(m_species_index);
        rho = mypc.GetChargeDensity(m_lev, true);
    }

    // Handle the parallel transfers of guard cells and
    // apply the filtering if requested.
    warpx.ApplyFilterandSumBoundaryRho(m_lev, m_lev, *rho, 0, rho->nComp());

#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    // Apply k-space filtering when using the PSATD solver
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
    {
        if (WarpX::use_kspace_filter && m_apply_rz_psatd_filter) {
            auto & solver = warpx.get_spectral_solver_fp(m_lev);
            const SpectralFieldIndex& Idx = solver.m_spectral_index;
            solver.ForwardTransform(m_lev, *rho, Idx.rho_new);
            solver.ApplyFilter(m_lev, Idx.rho_new);
            solver.BackwardTransform(m_lev, *rho, Idx.rho_new);
        }
    }
#else
    amrex::ignore_unused(m_apply_rz_psatd_filter);
#endif

    InterpolateMFForDiag(mf_dst, *rho, dcomp, warpx.DistributionMap(m_lev),
                         m_convertRZmodes2cartesian);
}
