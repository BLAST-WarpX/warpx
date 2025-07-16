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

    m_nsamples.resize(warpx.finestLevel() + 1);

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idir = 0; idir < 3; ++idir) {
            amrex::BoxArray const& ba = T_vf[lev][Direction{idir}]->boxArray();
            amrex::DistributionMapping const& dm = T_vf[lev][Direction{idir}]->DistributionMap();
            amrex::IntVect const& ng = T_vf[lev][Direction{idir}]->nGrowVect();

            warpx.m_fields.alloc_init("variance_buffer_w_" + m_species_name, Direction{idir}, lev, ba, dm, ncomps, ng, 0.0_rt);
            warpx.m_fields.alloc_init("variance_buffer_w2_" + m_species_name, Direction{idir}, lev, ba, dm, ncomps, ng, 0.0_rt);
            warpx.m_fields.alloc_init("variance_buffer_vbar_" + m_species_name, Direction{idir}, lev, ba, dm, ncomps, ng, 0.0_rt);

            m_nsamples[lev][idir] = std::make_unique<amrex::iMultiFab>(ba, dm, ncomps, ng);
            m_nsamples[lev][idir]->setVal(0);
        }
    }
}

void
VarianceAccumulationBuffer::reset ()
{
    using ablastr::fields::Direction;
    auto &warpx = WarpX::GetInstance();

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idir = 0; idir < 3; ++idir) {
            warpx.m_fields.get("variance_buffer_w_" + m_species_name, Direction{idir}, lev)->setVal(0._rt);
            warpx.m_fields.get("variance_buffer_w2_" + m_species_name, Direction{idir}, lev)->setVal(0._rt);
            warpx.m_fields.get("variance_buffer_vbar_" + m_species_name, Direction{idir}, lev)->setVal(0._rt);
            m_nsamples[lev][idir]->setVal(0);
        }
    }
}

amrex::MultiFab*
VarianceAccumulationBuffer::get(std::string arr, ablastr::fields::Direction dir, int lev)
{
    auto &warpx = WarpX::GetInstance();
    return warpx.m_fields.get("variance_buffer_" + arr + "_" + m_species_name, dir, lev);
}

amrex::iMultiFab*
VarianceAccumulationBuffer::get_n(ablastr::fields::Direction dir, int lev)
{
    return m_nsamples[lev][dir].get();
}

void
VarianceAccumulationBuffer::ConvertVarianceToTemperatureAndFilter (
    ablastr::fields::MultiLevelVectorField const& var_vf,
    amrex::Real normalization_factor,
    bool apply_filter)
{
    using ablastr::fields::Direction;
    auto &warpx = WarpX::GetInstance();

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        auto const& periodicity = warpx.Geom(lev).periodicity();

        for (int idir = 0; idir < 3; ++idir) {
            // Multiplies internal cells to convert variance to temperature
            var_vf[lev][Direction{idir}]->mult(normalization_factor, 0, 1);

            amrex::Gpu::streamSynchronize();

            // Synchronize Ghost cells after normalization.
            ablastr::utils::communication::FillBoundary(
                *var_vf[lev][Direction{idir}],
                WarpX::do_single_precision_comms,
                periodicity,
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
                    periodicity,
                    true);
            }
        }
    }
    amrex::Gpu::streamSynchronize();
}
