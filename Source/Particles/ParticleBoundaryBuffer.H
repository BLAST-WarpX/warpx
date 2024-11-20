/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLEBOUNDARYBUFFER_H_
#define WARPX_PARTICLEBOUNDARYBUFFER_H_

#include "Particles/MultiParticleContainer_fwd.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/PinnedMemoryParticleContainer.H"
#include "Utils/export.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <vector>


/**
 *  This stores particles that have left / been absorbed by domain and embedded boundaries.
 */
class WARPX_EXPORT ParticleBoundaryBuffer
{
public:
    ParticleBoundaryBuffer ();

    ~ParticleBoundaryBuffer() = default;

    /** Copy constructor for ParticleBoundaryBuffer */
    ParticleBoundaryBuffer ( const ParticleBoundaryBuffer &) = delete;
    /** Copy operator for ParticleBoundaryBuffer */
    ParticleBoundaryBuffer& operator= ( const ParticleBoundaryBuffer & ) = delete;

    /** Move constructor for NamedComponentParticleContainer */
    ParticleBoundaryBuffer ( ParticleBoundaryBuffer && ) = default;
    /** Move operator for NamedComponentParticleContainer */
    ParticleBoundaryBuffer& operator= ( ParticleBoundaryBuffer && ) = default;

    int numSpecies() const { return static_cast<int>(getSpeciesNames().size()); }

    const std::vector<std::string>& getSpeciesNames() const;

    void gatherParticlesFromDomainBoundaries (MultiParticleContainer& mypc);
    void gatherParticlesFromEmbeddedBoundaries (
        MultiParticleContainer& mypc, ablastr::fields::MultiLevelScalarField const& distance_to_eb
    );

    void redistribute ();
    void clearParticles ();
    void clearParticles (int i);

    void printNumParticles () const;

    int getNumParticlesInContainer(const std::string& species_name, int boundary, bool local);

    PinnedMemoryParticleContainer& getParticleBuffer(const std::string& species_name, int boundary);

    PinnedMemoryParticleContainer* getParticleBufferPointer(const std::string& species_name, int boundary);

    static constexpr int numBoundaries () {
        return AMREX_SPACEDIM*2
#ifdef AMREX_USE_EB
            + 1
#endif
            ;
    }

    bool isDefinedForAnySpecies (int const ibuffer) {return (m_do_any_boundary[ibuffer] != 0);}

    std::string boundaryName (int const ibuffer) {return m_boundary_names[ibuffer];}

private:
    // over boundary, then number of species
    std::vector<std::vector<PinnedMemoryParticleContainer> > m_particle_containers;

    // over boundary, then number of species
    std::vector<std::vector<int> > m_do_boundary_buffer;

    // over boundary
    std::vector<int> m_do_any_boundary;
    std::vector<std::string> m_boundary_names;

    mutable std::vector<std::string> m_species_names;
};

#endif /*WARPX_PARTICLEBOUNDARYBUFFER_H_*/
