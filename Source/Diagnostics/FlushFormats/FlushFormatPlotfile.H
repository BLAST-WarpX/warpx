#ifndef WARPX_FLUSHFORMATPLOTFILE_H_
#define WARPX_FLUSHFORMATPLOTFILE_H_

#include "FlushFormat.H"

#include "Diagnostics/ParticleDiag/ParticleDiag_fwd.H"

#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <iosfwd>
#include <string>

/**
 * \brief This class aims at dumping diags data to disk using the AMReX Plotfile format.
 * In particular, function WriteToFile takes fields and particles as input arguments,
 * and writes data to file.
 */
class FlushFormatPlotfile : public FlushFormat
{
public:
    /** Flush fields and particles to plotfile */
    void WriteToFile (
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
        bool isLastBTDFlush = false) const override;

    /** Write general info of the run into the plotfile */
    void WriteJobInfo(const std::string& dir) const;
    /** Write WarpX-specific plotfile header */
    void WriteWarpXHeader(const std::string& name, amrex::Vector<amrex::Geometry>& geom) const;
    void WriteAllRawFields (bool plot_raw_fields, int nlevels,
                            const std::string& plotfilename,
                            bool plot_raw_fields_guards) const;
    /** \brief Write particles data to file.
     * \param[in] dir name of output directory
     * \param[in] particle_diags Each element of this vector handles output of 1 species.
     * \param[in] time the simulation time on the coarsest level
     * \param[in] isBTD whether this is a back-transformed diagnostic
     */
    void WriteParticles(const std::string& dir,
                        const amrex::Vector<ParticleDiag>& particle_diags,
                        amrex::Real time,
                        bool isBTD = false) const;

    FlushFormatPlotfile () = default;
    ~FlushFormatPlotfile() override = default;

    FlushFormatPlotfile ( FlushFormatPlotfile const &)             = default;
    FlushFormatPlotfile& operator= ( FlushFormatPlotfile const & ) = default;
    FlushFormatPlotfile ( FlushFormatPlotfile&& )                  = default;
    FlushFormatPlotfile& operator= ( FlushFormatPlotfile&& )       = default;
};

#endif // WARPX_FLUSHFORMATPLOTFILE_H_
