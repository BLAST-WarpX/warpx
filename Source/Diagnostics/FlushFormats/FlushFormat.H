#ifndef WARPX_FLUSHFORMAT_H_
#define WARPX_FLUSHFORMAT_H_

#include "FlushFormat_fwd.H"

#include "Diagnostics/ParticleDiag/ParticleDiag.H"
#include "Particles/MultiParticleContainer.H"

class FlushFormat
{
public:
    /** Flush fields and particles to file */
    virtual void WriteToFile (
        const amrex::Vector<std::string>& varnames,
        const amrex::Vector<amrex::MultiFab>& mf,
        amrex::Vector<amrex::Geometry>& geom,
        amrex::Vector<int> iteration, double time,
        const amrex::Vector<ParticleDiag>& particle_diags, int nlev,
        std::string prefix, int file_min_digits,
        bool plot_raw_fields,
        bool plot_raw_fields_guards,
        int verbose = 2,
        bool use_pinned_pc = false,
        bool isBTD = false, int snapshotID = -1,
        int bufferID = 1, int numBuffers = 1,
        const amrex::Geometry& full_BTD_snapshot = amrex::Geometry(),
        bool isLastBTDFlush = false) const = 0;

    FlushFormat () = default;
    virtual ~FlushFormat() = default;

    FlushFormat ( FlushFormat const &)             = default;
    FlushFormat& operator= ( FlushFormat const & ) = default;
    FlushFormat ( FlushFormat&& )                  = default;
    FlushFormat& operator= ( FlushFormat&& )       = default;
};

#endif // WARPX_FLUSHFORMAT_H_
