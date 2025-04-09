/* Copyright 2021 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_COMPUTEPARTICLEDIAGFUNCTOR_H_
#define WARPX_COMPUTEPARTICLEDIAGFUNCTOR_H_

#include "Particles/WarpXParticleContainer.H"
#include "Particles/PinnedMemoryParticleContainer.H"
#include <AMReX.H>
#include <AMReX_AmrParticles.H>

/**
 * \brief Functor to compute a diagnostic and store the result in existing ParticleContainer.
 */
class ComputeParticleDiagFunctor
{
public:

    ComputeParticleDiagFunctor( ) = default;
    /** Virtual Destructor to handle clean destruction of derived classes */
    virtual ~ComputeParticleDiagFunctor() = default;

    /** Default assignment and copy operations*/
    ComputeParticleDiagFunctor(const ComputeParticleDiagFunctor&) = default;
    ComputeParticleDiagFunctor& operator=(const ComputeParticleDiagFunctor&) = default;
    ComputeParticleDiagFunctor(ComputeParticleDiagFunctor&&) = default;
    ComputeParticleDiagFunctor& operator=(ComputeParticleDiagFunctor&&) = default;

    /** \brief Prepare data required to back-transform particle attributes for
     *         lab-frame snapshot, with index i_buffer.
     *         Note that this function has parameters that are specific to
     *         back-transformed diagnostics, that are unused for regular diagnostics.
     *
     * \param[in] i_buffer           index of the snapshot
     * \param[in] ZSliceInDomain     if the z-slice at current_z_boost is within the bounds of
     *            the boosted-frame and lab-frame domain. The particles are transformed
     *            only if this value is true.
     * \param[in] old_z_boost        old z co-ordinate of the slice selected in boosted-frame.
     * \param[in] current_z_boost    z co-ordinate of the slice selected in boosted-frame.
     * \param[in] t_lab              current time in lab-frame for snapshot, i_buffer.
     * \param[in] snapshot_full      if the current snapshot, with index, i_buffer, is
     *            full (1) or not (0). If it is full, then Lorentz-transform is not performed
     *            by setting m_perform_backtransform to 0 for the corresponding ith snapshot.
     */
    virtual void PrepareFunctorData ( int i_buffer, bool ZSliceInDomain,
                                      amrex::Real old_z_boost,
                                      amrex::Real current_z_boost, amrex::Real t_lab,
                                      int snapshot_full)
                                      {
                                          amrex::ignore_unused(i_buffer,
                                          ZSliceInDomain, old_z_boost,
                                          current_z_boost, t_lab, snapshot_full);
                                      }
    /** Compute particle attributes and store the result in pc_dst particle container.
     *  \param[out] pc_dst  output particle container where the result is stored.
     *  \param[out] totalParticlesInBuffer total number of particles in the container
     *  \param[in] i_buffer snapshot index for which the particle buffer is processed
     */
    virtual void operator () (PinnedMemoryParticleContainer& pc_dst, int &totalParticlesInBuffer, int i_buffer = 0) const = 0;
    virtual void InitData () {}
};

#endif // WARPX_COMPUTEPARTICLEDIAGFUNCTOR_H_
