#include "RhoFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    #include "FieldSolver/SpectralSolver/SpectralFieldData.H"
    #include "FieldSolver/SpectralSolver/SpectralSolverRZ.H"
    #include "Utils/WarpXAlgorithmSelection.H"
#endif
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EBJBoundary.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"
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

    // For the hybrid solver, fold the deposit collected by covered points
    // back across the embedded surface and enforce the embedded-boundary
    // Dirichlet condition on the freshly deposited charge density (the
    // density vanishes at a conducting wall), after the filter and
    // guard-cell sum so the mirrored values are not smeared into the
    // conductor, matching the solver's own rho treatment in
    // HybridPICDepositRhoAndJ
    if (EB::enabled() &&
        WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC)
    {
        auto const* hybrid = warpx.get_pointer_HybridPICModel();
        warpx::hybrid::FoldEBDepositToNodalScalar(
            *rho,
            *warpx.m_fields.get(warpx::fields::FieldType::distance_to_eb, m_lev),
            warpx.Geom(m_lev),
            hybrid->m_eb_fold_pec);
        warpx::hybrid::ApplyEBBoundaryToNodalScalar(
            *rho,
            *warpx.m_fields.get(warpx::fields::FieldType::distance_to_eb, m_lev),
            warpx.Geom(m_lev),
            /*odd=*/hybrid->m_eb_rho_dirichlet);
    }

    InterpolateMFForDiag(mf_dst, *rho, dcomp, warpx.DistributionMap(m_lev),
                         m_convertRZmodes2cartesian);
}
