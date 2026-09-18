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

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>

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
#include <numeric>
#include <utility>
#include <vector>

using namespace amrex;

std::pair<BoxArray, DistributionMapping>
WarpX::LoadBalanceMakeNewLayout (int lev, Real& efficiency)
{
    efficiency = -1;
#ifndef AMREX_USE_MPI
    amrex::ignore_unused(this, lev);
    return std::make_pair(amrex::BoxArray{},amrex::DistributionMapping{});
#else

    // The communicator type is implementation-dependent and need not be a pointer.
    auto comm = ParallelContext::CommunicatorSub(); // NOLINT
    int const myproc = ParallelContext::MyProcSub();
    int const nprocs = ParallelContext::NProcsSub();

    auto const& cst = *costs[lev];
    auto const& ba = cst.boxArray();
    auto const& dm = cst.DistributionMap();
    auto const& mgs = maxGridSize(lev);
    auto const split_multiple = (lev > 0) ? 2*refRatio(lev-1) : IntVect(4);

    BoxList newbl{};
    BoxArray newba{};
    DistributionMapping newdm{};

    Vector<Real> rcost(cst.size()); // global cost vector
    ParallelDescriptor::GatherLayoutDataToVector<Real>(cst, rcost, 0);

    int doLoadBalance = 0;
    if (myproc == 0) {
        bool split_high_density_boxes = false;
        ParmParse const pp0;
        pp0.query("warpx.split_high_density_boxes"
                  ,      split_high_density_boxes);
        // PSATD runtime load balancing only redistributes existing boxes.
        if (split_high_density_boxes &&
            electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD)
        {
            Real split_high_density_boxes_threshold = 1.1;
            int split_high_density_boxes_min_box_size = 8;
            pp0.query("warpx.split_high_density_boxes_threshold"
                      ,      split_high_density_boxes_threshold);
            pp0.query("warpx.split_high_density_boxes_min_box_size"
                      ,      split_high_density_boxes_min_box_size);

            // Preserve the bounds checked by CheckGuardCells(), in fine-level cell units.
            // Use allocated fields: their guard widths can differ from one another and by level.
            IntVect min_child_size(1);
            auto const require_guard_cells = [&min_child_size] (MultiFab const& mf) {
                min_child_size.max((mf.nGrowVect() + 1) * mf.boxArray().crseRatio());
            };
            using warpx::fields::FieldType;
            using ablastr::fields::Direction;
            for (auto const field : {FieldType::Efield_fp, FieldType::Bfield_fp,
                                     FieldType::current_fp, FieldType::Efield_avg_fp,
                                     FieldType::Bfield_avg_fp, FieldType::Efield_cp,
                                     FieldType::Bfield_cp, FieldType::current_cp,
                                     FieldType::Efield_avg_cp, FieldType::Bfield_avg_cp})
            {
                for (int dir = 0; dir < 3; ++dir) {
                    if (m_fields.has(field, Direction{dir}, lev)) {
                        require_guard_cells(*m_fields.get(field, Direction{dir}, lev));
                    }
                }
            }
            for (auto const field : {FieldType::rho_fp, FieldType::F_fp, FieldType::G_fp,
                                     FieldType::rho_cp, FieldType::F_cp, FieldType::G_cp})
            {
                if (m_fields.has(field, lev)) {
                    require_guard_cells(*m_fields.get(field, lev));
                }
            }

            Real const total_costs = std::accumulate(rcost.begin(), rcost.end(), Real(0));
            Real const target_cost = total_costs / Real(nprocs) * split_high_density_boxes_threshold;

            newbl = ba.boxList();
            bool any_changed = false;
            for (int it = 0; it < 8; ++it) {
                bool iteration_changed = false;
                BoxList bltmp;
                Vector<Real> coststmp;
                Vector<Box>& blv = newbl.data();
                auto nboxes = int(blv.size());
                for (int i = 0; i < nboxes; ++i) {
                    bool this_changed = false;
                    if (total_costs > Real(0) && rcost[i] >= target_cost) {
                        Box b = blv[i];
                        std::array<std::pair<int,int>,AMREX_SPACEDIM> dlpair;
                        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                            dlpair[idim].first = idim;
                            dlpair[idim].second = b.length(idim);
                        }
                        std::stable_sort(dlpair.begin(), dlpair.end(),
                                         [] (std::pair<int,int> const& x,
                                             std::pair<int,int> const& y) {
                                             return x.second < y.second;
                                         });
                        for (int idim = AMREX_SPACEDIM-1; idim >= 0; --idim) {
                            auto const [dir, len] = dlpair[idim];
                            // Keep both children aligned with the coarse patch.
                            if (len % split_multiple[dir] == 0 &&
                                len > split_high_density_boxes_min_box_size &&
                                len/2 >= min_child_size[dir])
                            {
                                Box const b2 = b.chop(dir, b.smallEnd(dir) + len/2);
                                bltmp.push_back(b);
                                bltmp.push_back(b2);
                                coststmp.push_back(rcost[i]*Real(0.5));
                                coststmp.push_back(rcost[i]*Real(0.5));
                                this_changed = true;
                                break;
                            }
                        }
                    } else if (i < nboxes-1) {
                        Real const merged_cost = rcost[i] + rcost[i+1];
                        if (merged_cost < Real(0.9)*target_cost) {
                            Box const merged_box = amrex::minBox(blv[i], blv[i+1]);
                            if ((merged_box.numPts() == blv[i].numPts() + blv[i+1].numPts()) &&
                                merged_box.length().allLE(mgs))
                            {
                                bltmp.push_back(merged_box);
                                coststmp.push_back(rcost[i] + rcost[i+1]);
                                this_changed = true;
                                ++i;
                            }
                        }
                    }
                    if (this_changed) {
                        iteration_changed = true;
                    } else {
                        bltmp.push_back(blv[i]);
                        coststmp.push_back(rcost[i]);
                    }
                }
                if (iteration_changed) {
                    any_changed = true;
                    std::swap(newbl, bltmp);
                    std::swap(rcost, coststmp);
                } else {
                    break;
                }
            } // end of for (int it = 0;
            if (any_changed) {
                doLoadBalance = -int(newbl.size());
                newba.define(newbl);
                Vector<Long> const lcost = DistributionMapping::ConvertCostRealToLong(rcost);
                Real eff = -1;
                if (load_balance_with_sfc) {
                    newdm.SFCProcessorMap(newba, lcost, nprocs, eff, false);
                } else {
                    auto nmax = static_cast<int>(std::ceil(Real(newba.size())/Real(nprocs)
                                                           *load_balance_knapsack_factor));
                    newdm.KnapSackProcessorMap(lcost, nprocs, &eff, true, nmax, false);
                }
                efficiency = eff;
                if (verbose) {
                    amrex::Print(0, comm) << "Load balance (level " << lev << "): boxes "
                        << ba.size() << " -> " << newba.size()
                        << ", estimated efficiency = " << eff << ", accepted\n";
                }
            }
        } // end of if (split_high_density_boxes)

        if (newba.empty() && (load_balance_efficiency_ratio_threshold > 0)) {
            // Load balance the existing BoxArray
            Vector<Long> const lcost = DistributionMapping::ConvertCostRealToLong(rcost);
            Real current_eff = -1;
            DistributionMapping::ComputeDistributionMappingEfficiency(dm, lcost, &current_eff);
            Real eff = -1;
            if (load_balance_with_sfc) {
                newdm.SFCProcessorMap(ba, lcost, nprocs, eff, false);
            } else {
                auto nmax = static_cast<int>(std::ceil(Real(ba.size())/Real(nprocs)
                                                       *load_balance_knapsack_factor));
                newdm.KnapSackProcessorMap(lcost, nprocs, &eff, true, nmax, false);
            }
            if (eff > load_balance_efficiency_ratio_threshold * current_eff) {
                doLoadBalance = 1;
                efficiency = eff;
            }
            if (verbose) {
                amrex::Print(0, comm) << "Load balance (level " << lev
                    << "): current efficiency = " << current_eff
                    << ", proposed efficiency = " << eff
                    << (doLoadBalance ? ", accepted\n" : ", skipped\n");
            }
        }
    }

    ParallelDescriptor::Bcast(&doLoadBalance, 1, 0, comm);

    if (doLoadBalance) {
        ParallelDescriptor::Bcast(&efficiency, 1, 0, comm);
        auto nboxes = int(ba.size());
        if (doLoadBalance < 0) { // Bcast BoxArray
            Vector<Box> boxes;
            Box* p = nullptr;
            if (myproc == 0) {
                p = newbl.data().data();
                nboxes = int(newbl.size());
            } else {
                nboxes = -doLoadBalance;
                boxes.resize(nboxes);
                p = boxes.data();
            }
            ParallelDescriptor::Bcast(p, nboxes, 0, comm);
            if (myproc != 0) {
                newba.define(BoxList(std::move(boxes)));
            }
        } else {
            newba = ba;
        }
        { // Bcast Distributionmapping
            Vector<int> pmap;
            int* p = nullptr;
            if (myproc == 0) {
                p = const_cast<int*>(newdm.ProcessorMap().data());
            } else {
                pmap.resize(nboxes);
                p = pmap.data();
            }
            ParallelDescriptor::Bcast(p, nboxes, 0, comm);
            if (myproc != 0) {
                newdm.define(std::move(pmap));
            }
        }
        return std::make_pair(newba, newdm);
    } else {
        return std::make_pair(amrex::BoxArray{},amrex::DistributionMapping{});
    }
#endif
}

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
    if (ParallelContext::NProcsSub() == 1) {
        for (int lev = 0; lev <= finestLevel(); ++lev) {
            setLoadBalanceEfficiency(lev, Real(1));
        }
        return;
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        evolve_scheme == EvolveScheme::Explicit,
        "Runtime load balancing requires an explicit evolution scheme. "
        "Set algo.load_balance_intervals = 0.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        electromagnetic_solver_id != ElectromagneticSolverAlgo::ECT,
        "Runtime load balancing is not supported with the ECT solver. "
        "Set algo.load_balance_intervals = 0.");

    ABLASTR_PROFILE_REGION("LoadBalance");
    ABLASTR_PROFILE("WarpX::LoadBalance()");

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
        Real efficiency = -1;
        auto const& [newba, newdm] = LoadBalanceMakeNewLayout(lev, efficiency);
        if (! newdm.empty()) {
            RemakeLevel(lev, t_new[lev], newba, newdm);
            setLoadBalanceEfficiency(lev, efficiency);
            loadBalancedAnyLevel = true;
        }
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
        AllocInitMultiFab(pmf, amrex::convert(ba,mf->ixType()), dm, mf->nComp(), ng, lev, mf->tags()[0]);
        *mf = std::move(*pmf);
    };

    bool const eb_enabled = EB::enabled();

    if (ParallelContext::NProcsSub() == 1) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ba == boxArray(lev),
            "RemakeLevel cannot change the BoxArray with a single MPI process.");
        return;
    }

    m_fields.remake_level(lev, ba, dm);

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
        auto const* eb_index_space = GetEBIndexSpace(lev);
        m_field_factory[lev] = amrex::makeEBFabFactory(eb_index_space, Geom(lev), ba, dm,
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
            if (field_boundary_hi[0] == FieldBoundaryType::PML && !do_pml_in_domain) {
                // Extend region that is solved for to include the guard cells
                // which is where the PML boundary is applied.
                realspace_ba.growHi(0, pml_ncell);
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
                if (field_boundary_hi[0] == FieldBoundaryType::PML && !do_pml_in_domain) {
                    // Extend region that is solved for to include the guard cells
                    // which is where the PML boundary is applied.
                    c_realspace_ba.growHi(0, pml_ncell);
                }
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

    // Re-initialize the lattice element finder with the new ba and dm.
    m_accelerator_lattice[lev]->InitElementFinder(lev, gamma_boost, gett_new(), ba, dm);

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

    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);

    if (lev > 0 && (n_field_gather_buffer > 0 || n_current_deposition_buffer > 0)) {
        if (current_buffer_masks[lev] || gather_buffer_masks[lev]) {
            if (current_buffer_masks[lev]) {
                RemakeMultiFab( current_buffer_masks[lev] );
            }
            if (gather_buffer_masks[lev]) {
                RemakeMultiFab( gather_buffer_masks[lev] );
            }
            BuildBufferMasks();
        }
    }

    // Re-initialize diagnostic functors that stores pointers to the user-requested fields at level, lev.
    multi_diags->InitializeFieldFunctors( lev );

    // LoadBalance refreshes reduced diagnostics after all levels have been remade.
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
