/* Copyright 2019-2020 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_LEVELING_THINNING_H_
#define WARPX_LEVELING_THINNING_H_

#include "Resampling.H"

#include "Particles/WarpXParticleContainer_fwd.H"

#include <AMReX_REAL.H>

#include <string>

/**
 * \brief This class implements the leveling thinning algorithm as defined in Muraviev, A., et al.
 * arXiv:2006.08593 (2020).
 * The main steps of the algorithm are the following: for every cell we calculate a level weight,
 * defined by the average weight of the species particles in that cell multiplied by the target
 * ratio. Then, particles with a weight lower than the level weight are either removed, with a
 * probability 1 - particle_weight/level_weight, or have their weight set to the level weight.
 */
class LevelingThinning: public ResamplingAlgorithm {
public:

    /**
     * \brief Default constructor of the LevelingThinning class.
     */
    LevelingThinning () = default;

    /**
     * \brief Constructor of the LevelingThinning class
     *
     * @param[in] species_name the name of the resampled species
     */
    LevelingThinning (const std::string& species_name);

    /**
     * This function queries deprecated input parameters and aborts
     * the run if one of them is specified.
     */
    void BackwardCompatibility (const std::string& species_name );

    /**
     * \brief A method that performs leveling thinning for the considered species.
     *
     * @param[in] pti WarpX particle iterator of the particles to resample.
     * @param[in] lev the index of the refinement level.
     * @param[in] pc a pointer to the particle container.
     */
    void operator() (WarpXParIter& pti, int lev, WarpXParticleContainer* pc) const final;

private:
    amrex::Real m_target_ratio = amrex::Real(1.5);
    int m_min_ppc = 1;
};


#endif //WARPX_LEVELING_THINNING_H_
