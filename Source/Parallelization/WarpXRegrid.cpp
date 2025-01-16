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
#include "LoadBalance/LoadBalance.H"
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
using namespace warpx::load_balance;
using warpx::fields::FieldType;

void
WarpX::CheckLoadBalance (int step)
{
    auto& load_balance = LoadBalance::get_instance();

    if (step > 0 && load_balance.get_intervals().contains(step+1))
    {
        LoadBalance();

        // Reset the costs to 0
        load_balance.reset_costs(finest_level);
    }
    load_balance.rescale_costs(finest_level, step);
}

void
WarpX::LoadBalance ()
{
    WARPX_PROFILE_REGION("LoadBalance");
    WARPX_PROFILE("WarpX::LoadBalance()");

#ifdef AMREX_USE_MPI

    // By default, do not do a redistribute; this toggles to true if RemakeLevel
    // is called for any level
    int loadBalancedAnyLevel = false;
    const int nLevels = finestLevel();

    auto& load_balance = LoadBalance::get_instance();
    load_balance.compute_costs_if_heuristic(
        nLevels, m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level), *mypc);

    for (int lev = 0; lev <= nLevels; ++lev)
    {
        int doLoadBalance = false;

        auto load_balance_result = load_balance.compute_new_distribution_mapping(lev);

        // As specified in the above calls to makeSFC and makeKnapSack, the new
        // distribution mapping is NOT communicated to all ranks; the loadbalanced
        // dm is up-to-date only on root, and we can decide whether to broadcast
        if ((load_balance.get_efficiency_ratio_threshold() > 0.0)
            && (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()))
        {
            doLoadBalance = (load_balance_result.proposedEfficiency > load_balance_result.currentEfficiency);
        }

        ParallelDescriptor::Bcast(&doLoadBalance, 1,
            ParallelDescriptor::IOProcessorNumber());

        if (doLoadBalance)
        {
            Vector<int> pmap;
            if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber())
            {
                pmap = load_balance_result.dm.ProcessorMap();
            } else
            {
                const auto& costs = load_balance.get_costs(lev);
                const Real nboxes = costs->size();
                pmap.resize(static_cast<std::size_t>(nboxes));
            }
            ParallelDescriptor::Bcast(pmap.data(), pmap.size(), ParallelDescriptor::IOProcessorNumber());

            if (ParallelDescriptor::MyProc() != ParallelDescriptor::IOProcessorNumber())
            {
                load_balance_result.dm = DistributionMapping(pmap);
            }

            RemakeLevel(lev, t_new[lev], boxArray(lev), load_balance_result.dm);

            // Record the load balance efficiency
            load_balance.set_efficiency(lev, load_balance_result.proposedEfficiency);
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
                if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) {
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

        auto& load_balance = LoadBalance::get_instance();

        load_balance.set_costs(lev, 0.0);
        load_balance.set_efficiency(lev, -1);

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
