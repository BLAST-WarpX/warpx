/* Copyright 2019-2020 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RESAMPLING_TRIGGER_H_
#define WARPX_RESAMPLING_TRIGGER_H_

#include "Utils/Parser/IntervalsParser.H"

#include <AMReX_REAL.H>

#include <limits>
#include <string>

/**
 * \brief This class is used to determine if resampling should be done at a given timestep for
 * a given species. Specifically resampling is performed if the current timestep is included in
 * the IntervalsParser m_resampling_intervals or if the average number of particles per cell of
 * the considered species exceeds the threshold m_max_avg_ppc.
 */
class ResamplingTrigger
{
public:

    /**
     * \brief Default constructor of the ResamplingTrigger class.
     */
    ResamplingTrigger () = default;

    /**
     * \brief Constructor of the ResamplingTrigger class. Reads the resampling trigger parameters
     * from the input file.
     */
    ResamplingTrigger (const std::string& species_name);

    /**
     * \brief A method that returns true if resampling should be done for the considered species
     * at the considered timestep.
     *
     * @param[in] timestep the current timestep
     * @param[in] global_numparts the total number of particles of the considered species
     */
    bool triggered (int timestep, amrex::Real global_numparts) const;

    /**
     * \brief A method that initializes the member m_global_numcells. It is only called once (the
     * first time triggered() is called) and is needed because warpx.boxArray(lev) is not yet
     * initialized when the constructor of this class is called.
     */
    void initialize_global_numcells () const;

private:
    // Intervals that define predetermined timesteps at which resampling is performed for all
    // species.
    utils::parser::IntervalsParser m_resampling_intervals;

    // Average number of particles per cell above which resampling is performed for a given species
    amrex::Real m_max_avg_ppc = std::numeric_limits<amrex::Real>::max();

    //Total number of simulated cells, summed over all mesh refinement levels.
    mutable amrex::Real m_global_numcells = amrex::Real(0.0);

    mutable bool m_initialized = false;
};

#endif //WARPX_RESAMPLING_TRIGGER_H_
