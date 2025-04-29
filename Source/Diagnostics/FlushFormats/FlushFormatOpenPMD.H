#ifndef WARPX_FLUSHFORMATOPENPMD_H_
#define WARPX_FLUSHFORMATOPENPMD_H_

#include "Diagnostics/WarpXOpenPMD.H"
#include "FlushFormat.H"

#include "Diagnostics/ParticleDiag/ParticleDiag_fwd.H"

#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <memory>
#include <string>

/**
 * \brief This class aims at dumping diags data to disk using the openPMD standard.
 * In particular, function WriteToFile takes fields and particles as input arguments,
 * and writes data to file.
 */
class FlushFormatOpenPMD final : public FlushFormat
{
public:

    /** Constructor takes name of diagnostics to set the output directory */
    explicit FlushFormatOpenPMD (const std::string& diag_name);

    /** Flush fields and particles to plotfile */
    void WriteToFile (
        const amrex::Vector<std::string>& varnames,
        const amrex::Vector<amrex::MultiFab>& mf,
        amrex::Vector<amrex::Geometry>& geom,
        amrex::Vector<int> iteration, double time,
        const amrex::Vector<ParticleDiag>& particle_diags, int output_levels,
        std::string prefix, int file_min_digits,
        bool plot_raw_fields,
        bool plot_raw_fields_guards,
        int verbose,
        bool use_pinned_pc = false,
        bool isBTD = false, int snapshotID = -1,
        int bufferID = 1, int numBuffers = 1,
        const amrex::Geometry& full_BTD_snapshot = amrex::Geometry(),
        bool isLastBTDFlush = false) const override;

    ~FlushFormatOpenPMD () override = default;

    FlushFormatOpenPMD ( FlushFormatOpenPMD const &)             = delete;
    FlushFormatOpenPMD& operator= ( FlushFormatOpenPMD const & ) = delete;
    FlushFormatOpenPMD ( FlushFormatOpenPMD&& )                  = default;
    FlushFormatOpenPMD& operator= ( FlushFormatOpenPMD&& )       = default;

private:
    /** This is responsible for dumping to file */
    std::unique_ptr< WarpXOpenPMDPlot > m_OpenPMDPlotWriter;
};

#endif // WARPX_FLUSHFORMATOPENPMD_H_
