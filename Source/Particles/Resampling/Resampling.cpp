/* Copyright 2019-2020 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Resampling.H"

#include "VelocityCoincidenceThinning.H"
#include "LevelingThinning.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

Resampling::Resampling (const std::string& species_name)
{
    const amrex::ParmParse pp_species_name(species_name);
    std::string resampling_algorithm_string = "leveling_thinning"; // default resampling algorithm
    pp_species_name.query("resampling_algorithm", resampling_algorithm_string);

    if (resampling_algorithm_string == "leveling_thinning")
    {
        m_resampling_algorithm = std::make_unique<LevelingThinning>(species_name);
    }
    else if (resampling_algorithm_string == "velocity_coincidence_thinning")
    {
        m_resampling_algorithm = std::make_unique<VelocityCoincidenceThinning>(species_name);
    }
    else
    { WARPX_ABORT_WITH_MESSAGE("Unknown resampling algorithm."); }

    m_resampling_trigger = ResamplingTrigger(species_name);
}

bool Resampling::triggered (const int timestep, const amrex::Real global_numparts) const
{
    return m_resampling_trigger.triggered(timestep, global_numparts);
}

void Resampling::operator() (WarpXParIter& pti, const int lev, WarpXParticleContainer * const pc) const
{
    (*m_resampling_algorithm)(pti, lev, pc);
}
