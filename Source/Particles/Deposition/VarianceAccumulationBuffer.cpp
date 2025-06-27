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

VarianceAccumulationBuffer::VarianceAccumulationBuffer (ablastr::fields::MultiLevelVectorField const& T_vf, std::string const& species_name ) :
    m_species_name(species_name)
{
    using ablastr::fields::Direction;
    auto & warpx = WarpX::GetInstance();

    const int ncomps = 1;

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idir = 0; idir < 3; ++idir) {
            amrex::BoxArray const& ba = T_vf[lev][Direction{idir}]->boxArray();
            amrex::DistributionMapping const& dm = T_vf[lev][Direction{idir}]->DistributionMap();
            amrex::IntVect const& ng = T_vf[lev][Direction{idir}]->nGrowVect();

            warpx.m_fields.alloc_init("variance_buffer_w_" + m_species_name, Direction{idir}, lev, ba, dm, ncomps, ng, 0.0_rt);
            warpx.m_fields.alloc_init("variance_buffer_w2_" + m_species_name, Direction{idir}, lev, ba, dm, ncomps, ng, 0.0_rt);
            warpx.m_fields.alloc_init("variance_buffer_vbar_" + m_species_name, Direction{idir}, lev, ba, dm, ncomps, ng, 0.0_rt);
        }
    }
}

void
VarianceAccumulationBuffer::setAllValues (amrex::Real val)
{
    using ablastr::fields::Direction;
    auto &warpx = WarpX::GetInstance();

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idir = 0; idir < 3; ++idir) {
            get("w", Direction{idir}, lev)->setVal(val);
            get("w2", Direction{idir}, lev)->setVal(val);
            get("vbar", Direction{idir}, lev)->setVal(val);
        }
    }
}

amrex::MultiFab*
VarianceAccumulationBuffer::get(std::string arr, ablastr::fields::Direction dir, int lev)
{
    auto &warpx = WarpX::GetInstance();
    return warpx.m_fields.get("variance_buffer_" + arr + "_" + m_species_name, dir, lev);
}

void
VarianceAccumulationBuffer::SynchronizeBoundaryAndNormalizeVariance (
    ablastr::fields::MultiLevelVectorField const& var_vf,
    amrex::Real normalization_factor,
    bool apply_filter)
{
    using ablastr::fields::Direction;
    auto &warpx = WarpX::GetInstance();

    // This update algorithm follows this reference:
    // Chan, Golub, LeVeque, http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        auto const& periodicity = warpx.Geom(lev).periodicity();

        for (int idir = 0; idir < 3; ++idir) {
            // Grab references for the appropriate MFs
            amrex::MultiFab & wmf = *get("w", Direction{idir}, lev);
            amrex::MultiFab & w2mf = *get("w2", Direction{idir}, lev);
            amrex::MultiFab & vbarmf = *get("vbar", Direction{idir}, lev);
            amrex::MultiFab & variancemf = *var_vf[lev][Direction{idir}];

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

            // Do a local copy of the weight sum to vbarmf_old
            amrex::MultiFab::Copy(vbarmf_old, wmf, 0, 0, 1, vbarmf_old.nGrowVect());
            amrex::Gpu::streamSynchronize();

            // Do a local multiply of the weight sum by vbar so we end up with w_a * vbar_a
            // This is the first part of the vbar update, which is being computed as
            // vbar_ab = (wsum_a * vbar_a + wsum_b * vbar_b) / wsum_ab,  where the owner cell
            // is a and b would be contributions from overlapping non-owner cells.
            // This can support tile corners where more than 2 boxes overlap as well since the
            // weights are combined prior to summation via the Guard Cell summation.
            amrex::MultiFab::Multiply(vbarmf_old, vbarmf, 0, 0, 1, vbarmf_old.nGrowVect());

            amrex::Gpu::streamSynchronize();

            // Combine guard cells so that the summation is over accumulated quantities
            // vbarmf_old currently holds the w_i * vbar_i for
            // WarpSumXGuardCells does an unweighted sum of all overlapping boxes
            // It contains contributions from 4 boxes at the corners of the tiles
            WarpXSumGuardCells(vbarmf_old, periodicity, vbarmf_old.nGrowVect(), 0, 1);

            // Sum the overlapping cells together and place in temporary MFs
            WarpXSumGuardCells(wmf_old, wmf, periodicity, wmf.nGrowVect(), 0, 1);
            WarpXSumGuardCells(w2mf_old, w2mf, periodicity, w2mf.nGrowVect(), 0, 1);
            WarpXSumGuardCells(variancemf_old, variancemf, periodicity, variancemf.nGrowVect(), 0, 1);

            amrex::Gpu::streamSynchronize();

            // Since we need to do a weighted combination subtract off the old data
            amrex::MultiFab::Subtract(wmf_old, wmf, 0, 0, 1, wmf_old.nGrowVect());
            amrex::MultiFab::Subtract(w2mf_old, w2mf, 0, 0, 1, w2mf_old.nGrowVect());
            amrex::MultiFab::Subtract(vbarmf_old, vbarmf, 0, 0, 1, vbarmf_old.nGrowVect());
            amrex::MultiFab::Subtract(variancemf_old, variancemf, 0, 0, 1, variancemf_old.nGrowVect());

            amrex::Gpu::streamSynchronize();

            // Now the old arrays should only have summation contributions from neighboring tiles.
            // The kernel below handles the weighted combination for the online variance update.
            // These gymnastics are required due to the current interface in AMReX for MF synchronizations.
            // Perhaps a weighted guard cell summation could be include in AMReX to avoid this way of
            // doing things.

            auto const &owner_mask = amrex::OwnerMask(variancemf, periodicity);

            // Kernel that updates remainder of ghost values after communicating weight sums
            // It only occurs in cells that are owned by the current tile.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for ( amrex::MFIter mfi(variancemf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
                // Extract Tileboxes for wmf and bbf
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
                        // Check if we are in a cell that requires combining variances
                        if (mma(i,j,k) && w_old_arr(i,j,k) > 0.0_rt) {
                            // Update weights and un-normalized variance
                            amrex::Real w_a = w_old_arr(i,j,k);
                            amrex::Real w_b = w_arr(i,j,k);
                            amrex::Real vsum_a = vbar_old_arr(i,j,k);
                            amrex::Real v_a = vsum_a/w_a;
                            amrex::Real v_b = vbar_arr(i,j,k);

                            // Delta squared is ill-behaved when v_b and v_a are close.
                            // Check this to avoid underflow when squaring a small number.
                            amrex::Real delta = v_b - v_a;
                            if (2.*delta/(v_a+v_b) < 10._rt * std::numeric_limits<amrex::Real>::epsilon()) {
                                delta = 0._rt;
                            }

                            const amrex::Real dwa = delta*w_a;
                            const amrex::Real dwb = delta*w_b;

                            // Update accumulation arrays
                            w_arr(i,j,k) += w_a;
                            w2_arr(i,j,k) += w2_old_arr(i,j,k);

                            // Update Mean and Variance
                            vbar_arr(i,j,k) = (vsum_a + w_b*v_b)/w_arr(i,j,k);
                            variance_arr(i,j,k) +=  variance_old_arr(i,j,k) + dwa*dwb/w_arr(i,j,k);
                        }

                        // Apply Normalization in any owned cell
                        if (w_arr(i,j,k) > 0.0_rt) {
                            amrex::Real denom = (w_arr(i,j,k) - w2_arr(i,j,k)/w_arr(i,j,k) );
                            if (denom > 10._rt * std::numeric_limits<amrex::Real>::epsilon()) {
                                variance_arr(i,j,k) /= denom;
                            } else {
                                // This occurs when only a single sample is within a bin
                                // In this case should be zero variance
                                variance_arr(i,j,k) = 0._rt;
                            }
                        }
                    });
            }
            amrex::Gpu::streamSynchronize();

            // Multiplies internal cells to convert variance to temperature
            var_vf[lev][Direction{idir}]->mult(normalization_factor, 0, 1);

            amrex::Gpu::streamSynchronize();

            // Synchronize Ghost cells after normalization.
            ablastr::utils::communication::FillBoundary(
                *var_vf[lev][Direction{idir}],
                WarpX::do_single_precision_comms,
                warpx.Geom(lev).periodicity(),
                true);

            // If filtering, apply filter
            if (apply_filter) {
                amrex::Gpu::streamSynchronize();

                warpx.ApplyFilterMF(var_vf, lev, idir);

                amrex::Gpu::streamSynchronize();

                // Re-synchronize MF after filtering
                ablastr::utils::communication::FillBoundary(
                    *var_vf[lev][Direction{idir}],
                    WarpX::do_single_precision_comms,
                    warpx.Geom(lev).periodicity(),
                    true);
            }
        }
    }
    amrex::Gpu::streamSynchronize();
}
