#include "RhoFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    #include "FieldSolver/SpectralSolver/SpectralFieldData.H"
    #include "FieldSolver/SpectralSolver/SpectralSolverRZ.H"
    #include "Utils/WarpXAlgorithmSelection.H"
#endif
#include "Fluids/MultiFluidContainer.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <ablastr/coarsen/average.H>
#include <ablastr/coarsen/sample.H>

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <utility>

namespace
{
    std::unique_ptr<amrex::MultiFab>
    deposit_particles (WarpX& warpx, int lev, int species)
    {
        auto& particles = warpx.GetPartContainer();
        if (species < 0) {
            return particles.GetChargeDensityOnLevel(lev, true, false);
        }
        return particles.GetParticleContainer(species).GetChargeDensityOnLevel(lev, true, false);
    }

    // Each particle's shape and optional filter live on its own mesh. Include
    // cells outside fine patches so charge crossing a refinement face survives.
    amrex::MultiFab
    particle_density_on_cells (WarpX& warpx, int lev, int species)
    {
        auto rho = deposit_particles(warpx, lev, species);
        auto ng = warpx.get_ng_depos_rho();
        if (WarpX::use_filter) {
            ng += warpx.bilinear_filter.stencil_length_each_dir - 1;
        }
        if (lev > 0) {
            auto const ratio = warpx.refRatio(lev - 1);
            ng = ((ng + ratio - 1) / ratio) * ratio;
        }
        if (!rho->nGrowVect().allGE(ng)) {
            auto extended = std::make_unique<amrex::MultiFab>(
                rho->boxArray(), rho->DistributionMap(), rho->nComp(),
                amrex::max(ng, rho->nGrowVect()));
            extended->setVal(0);
            amrex::MultiFab::Copy(*extended, *rho, 0, 0, rho->nComp(), rho->nGrowVect());
            rho = std::move(extended);
        }
        warpx.ApplyFilterandSumBoundaryRho(lev, lev, *rho, 0, rho->nComp());
        warpx.FinalizeRhoOnLevel(lev, rho.get(), PatchType::fine);
        amrex::MultiFab cells(warpx.boxArray(lev), warpx.DistributionMap(lev), rho->nComp(), ng);
        ablastr::coarsen::sample::Coarsen(cells, *rho, 0, 0, rho->nComp(), ng, amrex::IntVect(1));
        return cells;
    }

    void
    add_parent_density (amrex::MultiFab const& parent, amrex::MultiFab& child,
                        amrex::IntVect const& ratio, amrex::Periodicity const& period)
    {
        auto const ng = (child.nGrowVect() + ratio - 1) / ratio;
        amrex::MultiFab local(amrex::coarsen(child.boxArray(), ratio),
                              child.DistributionMap(), child.nComp(), ng);
        local.setVal(0);
        local.ParallelCopy(parent, 0, 0, child.nComp(), parent.nGrowVect(), ng, period);
        auto const& dst = child.arrays();
        auto const& src = local.const_arrays();
        // A cell density is per unit volume. Each child inherits its parent's
        // contribution, including the part of a parent particle inside a fine patch.
        amrex::ParallelFor(child, child.nGrowVect(), child.nComp(),
            [=] AMREX_GPU_DEVICE (int b, int i, int j, int k, int n) noexcept {
                amrex::IntVect const cell(AMREX_D_DECL(i, j, k));
                dst[b](i, j, k, n) += src[b](amrex::coarsen(cell, ratio), n);
            });
        amrex::Gpu::streamSynchronize();
    }

    void
    add_child_density (amrex::MultiFab const& child, amrex::MultiFab& parent,
                       amrex::IntVect const& ratio, amrex::Geometry const& geometry)
    {
        amrex::MultiFab restricted(amrex::coarsen(child.boxArray(), ratio),
                                   child.DistributionMap(), child.nComp(), child.nGrowVect() / ratio);
        auto const* source = &child;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::MultiFab weighted;
        if (ratio[0] != 1) {
            // Child shells with equal dr have different physical volumes.
            // Weight the arithmetic average by the child/parent mean Jacobian.
            weighted.define(child.boxArray(), child.DistributionMap(),
                            child.nComp(), child.nGrowVect());
            auto const dr = geometry.CellSize(0);
            auto const rf = ratio[0];
            auto const df = dr / rf;
            auto const origin = geometry.ProbLo(0) - geometry.Domain().smallEnd(0) * dr;
            auto const& dst = weighted.arrays();
            auto const& src = child.const_arrays();
            amrex::ParallelFor(child, child.nGrowVect(), child.nComp(),
                [=] AMREX_GPU_DEVICE (int b, int i, int j, int k, int n) noexcept {
                    auto const r_fine = origin + (i + amrex::Real(0.5)) * df;
                    auto const r_parent = origin + (amrex::coarsen(i, rf) + amrex::Real(0.5)) * dr;
#if defined(WARPX_DIM_RSPHERE)
                    auto const jacobian = (r_fine * r_fine + df * df / 12)
                                         / (r_parent * r_parent + dr * dr / 12);
#else
                    auto const jacobian = amrex::Math::abs(r_fine / r_parent);
#endif
                    dst[b](i, j, k, n) = src[b](i, j, k, n) * jacobian;
                });
            amrex::Gpu::streamSynchronize();
            source = &weighted;
        }
#endif
        ablastr::coarsen::average::Coarsen(restricted, *source, ratio);
        // Synchronized field values agree in overlapping guards. Copy one
        // value at each location before adding it to the parent charge.
        amrex::MultiFab local(parent.boxArray(), parent.DistributionMap(),
                              parent.nComp(), parent.nGrowVect());
        local.setVal(0);
        local.ParallelCopy(restricted, 0, 0, parent.nComp(), restricted.nGrowVect(),
                           parent.nGrowVect(), geometry.periodicity());
        amrex::MultiFab::Add(parent, local, 0, 0, parent.nComp(), parent.nGrowVect());
    }

    amrex::MultiFab
    charge_from_other_levels (WarpX& warpx, int target, int species, int ncomp,
                              amrex::IntVect const& target_ng)
    {
        int const finest = warpx.finestLevel();
        amrex::Vector<amrex::MultiFab> cells(finest + 1);
        for (int lev = 0; lev <= finest; ++lev) {
            if (lev == target) {
                cells[lev].define(warpx.boxArray(lev), warpx.DistributionMap(lev),
                                  ncomp, target_ng);
                cells[lev].setVal(0);
            } else {
                cells[lev] = particle_density_on_cells(warpx, lev, species);
            }
        }
        for (int lev = 0; lev < target; ++lev) {
            add_parent_density(cells[lev], cells[lev + 1], warpx.refRatio(lev),
                               warpx.Geom(lev).periodicity());
        }
        for (int lev = finest; lev > target; --lev) {
            add_child_density(cells[lev], cells[lev - 1], warpx.refRatio(lev - 1), warpx.Geom(lev - 1));
        }
        return std::move(cells[target]);
    }
}

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
    auto rho = deposit_particles(warpx, m_lev, m_species_index);
    if (m_species_index == -1 && warpx.DoFluidSpecies()) {
        warpx.GetFluidContainer().DepositCharge(warpx.m_fields, *rho, m_lev);
    }

    // Handle the parallel transfers of guard cells and
    // apply the filtering if requested.
    warpx.ApplyFilterandSumBoundaryRho(m_lev, m_lev, *rho, 0, rho->nComp());
    warpx.FinalizeRhoOnLevel(m_lev, rho.get(), PatchType::fine);

    std::unique_ptr<amrex::MultiFab> extra;
    if (warpx.finestLevel() > 0) {
        // Fluid fields are already represented on each mesh. Only particles
        // residing on a different level have to be transferred here.
        auto const ng = amrex::max(rho->nGrowVect(), mf_dst.nGrowVect() * m_crse_ratio);
        extra = std::make_unique<amrex::MultiFab>(
            charge_from_other_levels(warpx, m_lev, m_species_index, rho->nComp(), ng));
        if (rho->ixType().cellCentered()) {
            // RZ PSATD filters the complete modal charge on the native cell centers.
            amrex::MultiFab::Add(*rho, *extra, 0, 0, rho->nComp(), rho->nGrowVect());
            extra.reset();
        }
    }

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
    if (extra) {
        amrex::MultiFab sampled(mf_dst.boxArray(), mf_dst.DistributionMap(), nComp(),
                                mf_dst.nGrowVect());
        sampled.setVal(0);
        InterpolateMFForDiag(sampled, *extra, 0, warpx.DistributionMap(m_lev),
                             m_convertRZmodes2cartesian);
        amrex::MultiFab::Add(mf_dst, sampled, 0, dcomp, nComp(), mf_dst.nGrowVect());
    }
}
