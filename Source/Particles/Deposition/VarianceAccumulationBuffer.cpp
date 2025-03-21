/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Particles/Deposition/VarianceAccumulationBuffer.H"
#include "Particles/Deposition/TemperatureDeposition.H"
#include "Parallelization/WarpXSumGuardCells.H"
#include "Fields.H"

#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;
using namespace warpx::particles::deposition;

VarianceAccumulationBuffer::VarianceAccumulationBuffer (ablastr::fields::MultiLevelVectorField const& T_vf, std::string const& species_name )
{
    using ablastr::fields::Direction;
    auto & warpx = WarpX::GetInstance();

    const int ncomps = 1;

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::BoxArray const& bax = T_vf[lev][Direction{0}]->boxArray();
        amrex::BoxArray const& bay = T_vf[lev][Direction{1}]->boxArray();
        amrex::BoxArray const& baz = T_vf[lev][Direction{2}]->boxArray();

        amrex::DistributionMapping const& dmx = T_vf[lev][Direction{0}]->DistributionMap();
        amrex::DistributionMapping const& dmy = T_vf[lev][Direction{1}]->DistributionMap();
        amrex::DistributionMapping const& dmz = T_vf[lev][Direction{2}]->DistributionMap();

        amrex::IntVect const& ngx = T_vf[lev][Direction{0}]->nGrowVect();
        amrex::IntVect const& ngy = T_vf[lev][Direction{1}]->nGrowVect();
        amrex::IntVect const& ngz = T_vf[lev][Direction{2}]->nGrowVect();

        warpx.m_fields.alloc_init("w_" + species_name, Direction{0}, lev, bax, dmx, ncomps, ngx, 0.0_rt);
        warpx.m_fields.alloc_init("w_" + species_name, Direction{1}, lev, bay, dmy, ncomps, ngy, 0.0_rt);
        warpx.m_fields.alloc_init("w_" + species_name, Direction{2}, lev, baz, dmz, ncomps, ngz, 0.0_rt);

        warpx.m_fields.alloc_init("w2_" + species_name, Direction{0}, lev, bax, dmx, ncomps, ngx, 0.0_rt);
        warpx.m_fields.alloc_init("w2_" + species_name, Direction{1}, lev, bay, dmy, ncomps, ngy, 0.0_rt);
        warpx.m_fields.alloc_init("w2_" + species_name, Direction{2}, lev, baz, dmz, ncomps, ngz, 0.0_rt);

        warpx.m_fields.alloc_init("vbar_" + species_name, Direction{0}, lev, bax, dmx, ncomps, ngx, 0.0_rt);
        warpx.m_fields.alloc_init("vbar_" + species_name, Direction{1}, lev, bay, dmy, ncomps, ngy, 0.0_rt);
        warpx.m_fields.alloc_init("vbar_" + species_name, Direction{2}, lev, baz, dmz, ncomps, ngz, 0.0_rt);
    }

    // Grab references to these arrays to store in this data structure
    this->w = warpx.m_fields.get_mr_levels_alldirs("w_" + species_name, warpx.finestLevel());
    this->w2 = warpx.m_fields.get_mr_levels_alldirs("w2_" + species_name, warpx.finestLevel());
    this->vbar = warpx.m_fields.get_mr_levels_alldirs("vbar_" + species_name, warpx.finestLevel());
}

void
VarianceAccumulationBuffer::setAllValues (amrex::Real val)
{
    using ablastr::fields::Direction;
    auto &warpx = WarpX::GetInstance();

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idir = 0; idir < 3; ++idir) {
            this->w[lev][Direction{idir}]->setVal(val);
            this->w2[lev][Direction{idir}]->setVal(val);
            this->vbar[lev][Direction{idir}]->setVal(val);
        }
    }
}

void
VarianceAccumulationBuffer::SynchronizeBoundaryAndNormalizeVariance (ablastr::fields::MultiLevelVectorField const& var_vf)
{
    using ablastr::fields::Direction;
    auto &warpx = WarpX::GetInstance();

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        auto const& periodicity = warpx.Geom(lev).periodicity();

        for (int idir = 0; idir < 3; ++idir) {
            // Grab references for the appropriate MFs
            amrex::MultiFab & wmf = *this->w[lev][Direction{idir}];
            amrex::MultiFab & w2mf = *this->w2[lev][Direction{idir}];
            amrex::MultiFab & vbarmf = *this->vbar[lev][Direction{idir}];
            amrex::MultiFab & variancemf = *var_vf[lev][Direction{idir}];

            amrex::IndexType var_type = variancemf.ixType();

            // Create Temporary MFs for ghost cell buffers
            amrex::MultiFab wmf_old{
                wmf.boxArray(),
                wmf.DistributionMap(),
                wmf.nComp(),
                wmf.nGrowVect()
            };

            amrex::MultiFab w2mf_old{
                w2mf.boxArray(),
                w2mf.DistributionMap(),
                w2mf.nComp(),
                w2mf.nGrowVect()
            };

            amrex::MultiFab vbarmf_old{
                vbarmf.boxArray(),
                vbarmf.DistributionMap(),
                vbarmf.nComp(),
                vbarmf.nGrowVect()
            };

            amrex::MultiFab variancemf_old{
                variancemf.boxArray(),
                variancemf.DistributionMap(),
                variancemf.nComp(),
                variancemf.nGrowVect()
            };

            amrex::IntVect ng_depos_J = warpx.get_ng_depos_J();
            ng_depos_J.min(variancemf.nGrowVect());

            const amrex::IntVect src_ngrow = ng_depos_J;
            const amrex::IntVect dst_ngrow = variancemf_old.nGrowVect();

            // Sum the overlapping cells together
            WarpXSumGuardCells(wmf_old, wmf, periodicity, wmf.nGrowVect(), 0, 1);
            WarpXSumGuardCells(w2mf_old, w2mf, periodicity, w2mf.nGrowVect(), 0, 1);
            WarpXSumGuardCells(vbarmf_old, vbarmf, periodicity, vbarmf.nGrowVect(), 0, 1);
            WarpXSumGuardCells(variancemf_old, variancemf, periodicity, variancemf.nGrowVect(), 0, 1);

            amrex::Gpu::streamSynchronize();

            // Since we need to do a weighted combination subtract off the old data
            amrex::MultiFab::Subtract(wmf_old, wmf, 0, 0, 1, wmf_old.nGrowVect());
            amrex::MultiFab::Subtract(w2mf_old, w2mf, 0, 0, 1, w2mf_old.nGrowVect());
            amrex::MultiFab::Subtract(vbarmf_old, vbarmf, 0, 0, 1, vbarmf_old.nGrowVect());
            amrex::MultiFab::Subtract(variancemf_old, variancemf, 0, 0, 1, variancemf_old.nGrowVect());

            amrex::Gpu::streamSynchronize();

            auto const &owner_mask = amrex::OwnerMask(variancemf, periodicity);

            // Kernel that updates remainder of ghost values after communicating weight sums
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for ( amrex::MFIter mfi(variancemf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
                // Extract Tileboxes for wmf and bbf
                const amrex::IntVect& ngv = variancemf.nGrowVect();

                auto const& tb = mfi.tilebox();

                const auto& w_old_arr = wmf_old.const_array(mfi);
                const auto& w2_old_arr = w2mf_old.const_array(mfi);
                const auto& vbar_old_arr = vbarmf_old.const_array(mfi);
                const auto& variance_old_arr = variancemf_old.const_array(mfi);

                auto const& w_arr = wmf.array(mfi);
                auto const& w2_arr = w2mf.array(mfi);
                auto const& vbar_arr = vbarmf.array(mfi);
                auto const& variance_arr = variancemf.array(mfi);

                auto const& mma = owner_mask->const_array(mfi);

                amrex::ParallelFor(tb,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        // Only update if this box is the owner
                        if (mma(i,j,k)) {
                            // Check if we are in a cell that requires combining variances
                            if (w_old_arr(i,j,k) > 0.0_rt || w_arr(i,j,k) > 0.0_rt) {
                                // Update weights and un-normalized variance
                                amrex::Real w_a = w_old_arr(i,j,k);
                                amrex::Real w_b = w_arr(i,j,k);
                                amrex::Real v_a = vbar_old_arr(i,j,k);
                                amrex::Real v_b = vbar_arr(i,j,k);
                                amrex::Real delta = v_b - v_a;

                                // Update accumulation arrays
                                w_arr(i,j,k) += w_a;
                                w2_arr(i,j,k) += w2_old_arr(i,j,k);

                                // Update Mean and Variance
                                vbar_arr(i,j,k) = (w_a*v_a + w_b*v_b)/w_arr(i,j,k);
                                variance_arr(i,j,k) +=  variance_old_arr(i,j,k) + delta*delta*w_a*w_b/w_arr(i,j,k);
                            }

                            // Apply Normalization in any owned cell
                            if (w_arr(i,j,k) > 0.0_rt) {
                                variance_arr(i,j,k) /= w_arr(i,j,k);
                            }
                        }
                });
            }
            amrex::Gpu::streamSynchronize();
        }
    }
}
