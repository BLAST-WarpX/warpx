/* Copyright 2019 Andrew Myers, Ann Almgren, Axel Huebl
 * David Grote, Maxence Thevenet, Michael Rowan
 * Remi Lehe, Weiqun Zhang, levinem, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "Diagnostics/MultiDiagnostics.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "EmbeddedBoundary/WarpXFaceInfoBox.H"
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/ExternalField.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabFactory.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_IndexType.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MakeType.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

using namespace amrex;

void
WarpX::CheckLoadBalance (int step)
{
    if (step > 0 && load_balance_intervals.contains(step+1))
    {
        LoadBalance();

        // Reset the costs to 0
        ResetCosts();
    }
    if (!costs.empty())
    {
        RescaleCosts(step);
    }
}

void
WarpX::LoadBalance ()
{
    WARPX_PROFILE_REGION("LoadBalance");
    WARPX_PROFILE("WarpX::LoadBalance()");

    AMREX_ALWAYS_ASSERT(!costs.empty());
    AMREX_ALWAYS_ASSERT(costs[0] != nullptr);

#ifdef AMREX_USE_MPI
    if (load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Heuristic)
    {
        // compute the costs on a per-rank basis
        ComputeCostsHeuristic(costs);
    }

    // By default, do not do a redistribute; this toggles to true if RemakeLevel
    // is called for any level
    int loadBalancedAnyLevel = false;

    const int nLevels = finestLevel();
    for (int lev = 0; lev <= nLevels; ++lev)
    {
        int doLoadBalance = false;

        // Compute the new distribution mapping
        DistributionMapping newdm;
        const amrex::Real nboxes = costs[lev]->size();
        const amrex::Real nprocs = ParallelContext::NProcsSub();
        const int nmax = static_cast<int>(std::ceil(nboxes/nprocs*load_balance_knapsack_factor));
        // These store efficiency (meaning, the  average 'cost' over all ranks,
        // normalized to max cost) for current and proposed distribution mappings
        amrex::Real currentEfficiency = 0.0;
        amrex::Real proposedEfficiency = 0.0;

        newdm = (load_balance_with_sfc)
            ? DistributionMapping::makeSFC(*costs[lev],
                                           currentEfficiency, proposedEfficiency,
                                           false,
                                           ParallelDescriptor::IOProcessorNumber())
            : DistributionMapping::makeKnapSack(*costs[lev],
                                                currentEfficiency, proposedEfficiency,
                                                nmax,
                                                false,
                                                ParallelDescriptor::IOProcessorNumber());
        // As specified in the above calls to makeSFC and makeKnapSack, the new
        // distribution mapping is NOT communicated to all ranks; the loadbalanced
        // dm is up-to-date only on root, and we can decide whether to broadcast
        if ((load_balance_efficiency_ratio_threshold > 0.0)
            && (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()))
        {
            doLoadBalance = (proposedEfficiency > load_balance_efficiency_ratio_threshold*currentEfficiency);
        }

        ParallelDescriptor::Bcast(&doLoadBalance, 1,
                                  ParallelDescriptor::IOProcessorNumber());

        if (doLoadBalance)
        {
            Vector<int> pmap;
            if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber())
            {
                pmap = newdm.ProcessorMap();
            } else
            {
                pmap.resize(static_cast<std::size_t>(nboxes));
            }
            ParallelDescriptor::Bcast(pmap.data(), pmap.size(), ParallelDescriptor::IOProcessorNumber());

            if (ParallelDescriptor::MyProc() != ParallelDescriptor::IOProcessorNumber())
            {
                newdm = DistributionMapping(pmap);
            }

            RemakeLevel(lev, t_new[lev], boxArray(lev), newdm);

            // Record the load balance efficiency
            setLoadBalanceEfficiency(lev, proposedEfficiency);
        }

        loadBalancedAnyLevel = loadBalancedAnyLevel || doLoadBalance;
    }
    if (loadBalancedAnyLevel)
    {
        mypc->Redistribute();
        mypc->defineAllParticleTiles();

        // redistribute particle boundary buffer
        m_particle_boundary_buffer->redistribute();

        // diagnostics & reduced diagnostics
        // not yet needed:
        //multi_diags->LoadBalance();
        reduced_diags->LoadBalance();
    }
#endif
}

void
WarpX::RemakeLevel (int lev, Real /*time*/, const BoxArray& ba, const DistributionMapping& dm)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const auto RemakeMultiFab = [&](auto& mf){
        if (mf == nullptr) { return; }
        const IntVect& ng = mf->nGrowVect();
        auto pmf = std::remove_reference_t<decltype(mf)>{};
        AllocInitMultiFab(pmf, mf->boxArray(), dm, mf->nComp(), ng, lev, mf->tags()[0]);
        mf = std::move(pmf);
    };

    bool const eb_enabled = EB::enabled();
    if (ba == boxArray(lev))
    {
        if (ParallelDescriptor::NProcs() == 1) { return; }

        m_fields.remake_level(lev, dm);

        // Fine patch
        ablastr::fields::MultiLevelVectorField const& Bfield_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level);
        for (int idim=0; idim < 3; ++idim)
        {
            if (eb_enabled) {
                RemakeMultiFab( m_eb_reduce_particle_shape[lev] );
                if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) {
                    RemakeMultiFab( m_eb_update_E[lev][idim] );
                    RemakeMultiFab( m_eb_update_B[lev][idim] );
                    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::ECT) {
                        m_borrowing[lev][idim] = std::make_unique<amrex::LayoutData<FaceInfoBox>>(amrex::convert(ba, Bfield_fp[lev][idim]->ixType().toIntVect()), dm);
                    }
                }
            }
        }

        if (eb_enabled) {
#ifdef AMREX_USE_EB
            int const max_guard = guard_cells.ng_FieldSolver.max();
            m_field_factory[lev] = amrex::makeEBFabFactory(Geom(lev), ba, dm,
                                                           {max_guard, max_guard, max_guard},
                                                           amrex::EBSupport::full);
#endif
            InitializeEBGridData(lev);
        } else {
            m_field_factory[lev] = std::make_unique<FArrayBoxFactory>();
        }

#ifdef WARPX_USE_FFT
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
            if (spectral_solver_fp[lev] != nullptr) {
                // Get the cell-centered box
                BoxArray realspace_ba = ba;   // Copy box
                realspace_ba.enclosedCells(); // Make it cell-centered
                auto ngEB = getngEB();
                auto dx = CellSize(lev);

#   ifdef WARPX_DIM_RZ
                if ( !fft_periodic_single_box ) {
                    realspace_ba.grow(1, ngEB[1]); // add guard cells only in z
                }
                AllocLevelSpectralSolverRZ(spectral_solver_fp,
                                           lev,
                                           realspace_ba,
                                           dm,
                                           dx);
#   else
                if ( !fft_periodic_single_box ) {
                    realspace_ba.grow(ngEB);   // add guard cells
                }
                bool const pml_flag_false = false;
                AllocLevelSpectralSolver(spectral_solver_fp,
                                         lev,
                                         realspace_ba,
                                         dm,
                                         dx,
                                         pml_flag_false);
#   endif
            }
        }
#endif

        // Coarse patch
        if (lev > 0) {

#ifdef WARPX_USE_FFT
            if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
                if (spectral_solver_cp[lev] != nullptr) {
                    BoxArray cba = ba;
                    cba.coarsen(refRatio(lev-1));
                    const std::array<Real,3> cdx = CellSize(lev-1);

                    // Get the cell-centered box
                    BoxArray c_realspace_ba = cba;  // Copy box
                    c_realspace_ba.enclosedCells(); // Make it cell-centered

                    auto ngEB = getngEB();

#   ifdef WARPX_DIM_RZ
                    c_realspace_ba.grow(1, ngEB[1]); // add guard cells only in z
                    AllocLevelSpectralSolverRZ(spectral_solver_cp,
                                               lev,
                                               c_realspace_ba,
                                               dm,
                                               cdx);
#   else
                    c_realspace_ba.grow(ngEB);
                    bool const pml_flag_false = false;
                    AllocLevelSpectralSolver(spectral_solver_cp,
                                             lev,
                                             c_realspace_ba,
                                             dm,
                                             cdx,
                                             pml_flag_false);
#   endif
                }
            }
#endif
        }

        if (lev > 0 && (n_field_gather_buffer > 0 || n_current_deposition_buffer > 0)) {
            if (current_buffer_masks[lev] || gather_buffer_masks[lev]) {
                BuildBufferMasks();
            }
        }

        // Re-initialize the lattice element finder with the new ba and dm.
        m_accelerator_lattice[lev]->InitElementFinder(lev, gamma_boost, ba, dm);

        if (costs[lev] != nullptr)
        {
            costs[lev] = std::make_unique<LayoutData<Real>>(ba, dm);
            const auto iarr = costs[lev]->IndexArray();
            for (const auto& i : iarr)
            {
                (*costs[lev])[i] = 0.0;
                setLoadBalanceEfficiency(lev, -1);
            }
        }

        SetDistributionMap(lev, dm);

    } else
    {
        WARPX_ABORT_WITH_MESSAGE("RemakeLevel: to be implemented");
    }

    // Re-initialize diagnostic functors that stores pointers to the user-requested fields at level, lev.
    multi_diags->InitializeFieldFunctors( lev );

    // Reduced diagnostics
    // not needed yet
}

void
WarpX::ComputeCostsHeuristic (amrex::Vector<std::unique_ptr<amrex::LayoutData<amrex::Real> > >& a_costs)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const auto & mypc_ref = GetPartContainer();
        const auto nSpecies = mypc_ref.nSpecies();

        // Species loop
        for (int i_s = 0; i_s < nSpecies; ++i_s)
        {
            auto & myspc = mypc_ref.GetParticleContainer(i_s);

            // Particle loop
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
            {
                (*a_costs[lev])[pti.index()] += costs_heuristic_particles_wt*pti.numParticles();
            }
        }

        // Cell loop
        MultiFab* Ex = m_fields.get(FieldType::Efield_fp, Direction{0}, lev);
        for (MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        {
            const Box& gbx = mfi.growntilebox();
            (*a_costs[lev])[mfi.index()] += costs_heuristic_cells_wt*gbx.numPts();
        }
    }
}

void
WarpX::ResetCosts ()
{
    AMREX_ALWAYS_ASSERT(!costs.empty());
    AMREX_ALWAYS_ASSERT(costs[0] != nullptr);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const auto iarr = costs[lev]->IndexArray();
        for (const auto& i : iarr)
        {
            // Reset costs
            (*costs[lev])[i] = 0.0;
        }
    }
}

void
WarpX::RescaleCosts (int step)
{
    // rescale is only used for timers
    if (WarpX::load_balance_costs_update_algo != LoadBalanceCostsUpdateAlgo::Timers)
    {
        return;
    }

    AMREX_ALWAYS_ASSERT(costs.size() == finest_level + 1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (costs[lev])
        {
            // Perform running average of the costs
            // (Giving more importance to most recent costs; only needed
            // for timers update, heuristic load balance considers the
            // instantaneous costs)
            for (const auto& i : costs[lev]->IndexArray())
            {
                (*costs[lev])[i] *= (1._rt - 2._rt/load_balance_intervals.localPeriod(step+1));
            }
        }
    }
}
