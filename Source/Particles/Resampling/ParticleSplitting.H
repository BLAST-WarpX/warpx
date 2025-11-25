/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLE_SPLITTING_H_
#define WARPX_PARTICLE_SPLITTING_H_

#include "Particles/Algorithms/KineticEnergy.H"
#include "Resampling.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#include <string>


/**
 * \brief This class implements a particle splitting scheme.
 */
class ParticleSplitting: public ResamplingAlgorithm {
public:

    /**
     * \brief Default constructor of the ParticleSplitting class.
     */
    ParticleSplitting () = default;

    /**
     * \brief Constructor of the ParticleSplitting class
     *
     * @param[in] species_name the name of the resampled species
     */
    ParticleSplitting (const std::string& species_name);

    /**
     * \brief A method that performs merging for the considered species.
     *
     * @param[in] geom_lev the geometry of the current refinement level
     * @param[in] pti WarpX particle iterator of the particles to resample.
     * @param[in] lev the index of the refinement level.
     * @param[in] pc a pointer to the particle container.
     */
    void operator() (
        const amrex::Geometry& geom_lev, WarpXParIter& pti,
        int lev, WarpXParticleContainer* pc) const final;

private:

    int m_min_ppc = 1;
    bool m_resampling_random_splitting_angle = true;
    amrex::ParticleReal m_splitting_angle = 0.0_rt;
    std::string m_splitting_type = "position_axes_aligned_split";
};
#endif // WARPX_PARTICLE_SPLITTING_H_
