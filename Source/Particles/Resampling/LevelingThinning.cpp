/* Copyright 2019-2020 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "LevelingThinning.H"

#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/TextMsg.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_DenseBins.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_Particles.H>
#include <AMReX_Random.H>
#include <AMReX_StructOfArrays.H>

#include <AMReX_BaseFwd.H>

LevelingThinning::LevelingThinning (const std::string& species_name)
{
    using namespace amrex::literals;

    const amrex::ParmParse pp_species_name(species_name);
    utils::parser::queryWithParser(
        pp_species_name, "resampling_algorithm_target_ratio", m_target_ratio);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( m_target_ratio > 0._rt,
                                    "Resampling target ratio should be strictly greater than 0");
    if (m_target_ratio <= 1._rt)
    {
        ablastr::warn_manager::WMRecordWarning("Species",
            "For species '" + species_name + "' " +
            "target ratio for leveling thinning is smaller or equal to one." +
            "It is possible that no particle will be removed during resampling");
    }

    utils::parser::queryWithParser(
        pp_species_name, "resampling_min_ppc", m_min_ppc);
    BackwardCompatibility(species_name);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_min_ppc >= 1,
                                     "Resampling min_ppc should be greater than or equal to 1");
}

void LevelingThinning::BackwardCompatibility (const std::string& species_name )
{
    const amrex::ParmParse pp_species_name(species_name);
    int backward_min_ppc;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_species_name.query("resampling_algorithm_min_ppc", backward_min_ppc),
        "<species>.resampling_algorithm_min_ppc is no longer a valid option. "
        "Please use the renamed option <species>.resampling_min_ppc instead."
    );
}

void LevelingThinning::operator() (WarpXParIter& pti, const int lev,
                                   WarpXParticleContainer * const pc) const
{
    using namespace amrex::literals;

    auto& ptile = pc->ParticlesAt(lev, pti);
    auto& soa = ptile.GetStructOfArrays();
    amrex::ParticleReal * const AMREX_RESTRICT w = soa.GetRealData(PIdx::w).data();
    auto * const AMREX_RESTRICT idcpu = soa.GetIdCPUData().data();

    // Using this function means that we must loop over the cells in the ParallelFor. In the case
    // of the leveling thinning algorithm, it would have possibly been more natural and more
    // efficient to directly loop over the particles. Nevertheless, this structure with a loop over
    // the cells is more general and can be readily used to implement almost any other resampling
    // algorithm.
    auto bins = ParticleUtils::findParticlesInEachCell(lev, pti, ptile);

    const auto n_cells = static_cast<int>(bins.numBins());
    auto *const indices = bins.permutationPtr();
    auto *const cell_offsets = bins.offsetsPtr();

    const amrex::Real target_ratio = m_target_ratio;
    const int min_ppc = m_min_ppc;

    // Loop over cells
    amrex::ParallelForRNG( n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
        {
            // The particles that are in the cell `i_cell` are
            // given by the `indices[cell_start:cell_stop]`
            const auto cell_start = static_cast<int>(cell_offsets[i_cell]);
            const auto cell_stop  = static_cast<int>(cell_offsets[i_cell+1]);
            const int cell_numparts = cell_stop - cell_start;

            // do nothing for cells with less particles than min_ppc
            // (this intentionally includes skipping empty cells, too)
            if (cell_numparts < min_ppc) {
                return;
            }
            amrex::Real average_weight = 0._rt;

            // First loop over cell particles to compute average particle weight in the cell
            for (int i = cell_start; i < cell_stop; ++i)
            {
                average_weight += w[indices[i]];
            }
            average_weight /= cell_numparts;

            const amrex::Real level_weight = average_weight*target_ratio;


            // Second loop over cell particles to perform the thinning
            for (int i = cell_start; i < cell_stop; ++i)
            {
                // Particles with weight greater than level_weight are left unchanged
                if (w[indices[i]] > level_weight) {continue;}

                amrex::Real const random_number = amrex::Random(engine);
                // Remove particle with probability 1 - particle_weight/level_weight
                if (random_number > w[indices[i]]/level_weight)
                {
                    idcpu[indices[i]] = amrex::ParticleIdCpus::Invalid;
                }
                // Set particle weight to level weight otherwise
                else
                {
                    w[indices[i]] = level_weight;
                }
            }
        }
    );
}
