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
    /** skip cells with fewer than this number of macroparticles to obtain a sufficiently accurate average weight */
    int m_min_ppc = 8;
    /** split macroparticles with w > m_split_weight_koef * (cell mean weight) */
    amrex::Real m_split_weight_koef = 4.0;
    /** if m_split_max_ppc > 0, do not split in cells with more than this many macroparticles */
    int m_split_max_ppc = -1; // -1: no limit
    bool m_do_random_splitting_angle = false;
    amrex::Real m_splitting_angle = 0.0_rt;
    std::string m_splitting_type = "axis_aligned";
    int m_splitting_type_id = 1; // 0: trivial (nothing changes except of weight); 1: axis-aligned (default), 2: velocity-aligned splitting type
};
#endif // WARPX_PARTICLE_SPLITTING_H_
