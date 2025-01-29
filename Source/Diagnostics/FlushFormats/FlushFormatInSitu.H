#ifndef WARPX_FLUSHFORMATINSITU_H_
#define WARPX_FLUSHFORMATINSITU_H_

#include "FlushFormat.H"

#include "Diagnostics/ParticleDiag/ParticleDiag_fwd.H"

#if defined(AMREX_USE_CONDUIT) || defined(AMREX_USE_ASCENT)
#   include <AMReX_Conduit_Blueprint.H>
#endif
#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <string>

/**
 * \brief This class is a wrapper for the two single-backend in-situ formats, those
 * being Catalyst and Ascent. They both use the exact same code for writing particles
 * and this class aims to reduce redundancy by defining the method only once.
 */
class FlushFormatInSitu : public FlushFormat
{
public:
    // May need to include separate exporters for particles, meshes, etc etc. Would probably also be useful
    // to give some higher level debug functions (like in the plotfile flush) since it's useful sometimes.
    // Could probably also include AMREX_USE_ASCENT or AMREX_USE_CONDUIT in here
    #if defined(AMREX_USE_CONDUIT) || defined(AMREX_USE_ASCENT)
        /** \brief Do in-situ visualization for particle data.
         * \param[in] particle_diags Each element of this vector handles output of 1 species.
         * \param[out] a_bp_mesh blueprint mesh generated from the container
         * Only compile if AMREX_USE_CONDUIT or AMREX_USE_ASCENT because we need to pass a
         * conduit class (conduit is required for catalyst so it does not need to be checked)
         */
        void WriteParticles(const amrex::Vector<ParticleDiag>& particle_diags, conduit::Node& a_bp_mesh) const;
    #endif

    FlushFormatInSitu ()          = default;
    ~FlushFormatInSitu() override = default;

    FlushFormatInSitu ( FlushFormatInSitu const &)             = default;
    FlushFormatInSitu& operator= ( FlushFormatInSitu const & ) = default;
    FlushFormatInSitu ( FlushFormatInSitu&& )                  = default;
    FlushFormatInSitu& operator= ( FlushFormatInSitu&& )       = default;
};

#endif
