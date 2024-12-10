#include "FlushFormatOpenPMD.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "Diagnostics/OpenPMDHelpFunction.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <map>
#include <memory>
#include <set>
#include <string>

using namespace amrex;


FlushFormatOpenPMD::FlushFormatOpenPMD (const std::string& diag_name)
{
    ParmParse pp_diag_name(diag_name);
    // Which backend to use (ADIOS, ADIOS2 or HDF5). Default depends on what is available
    std::string openpmd_backend {"default"};
    pp_diag_name.query("openpmd_backend", openpmd_backend);
    // pick first available backend if default is chosen
    if( openpmd_backend == "default" ) {
        openpmd_backend = WarpXOpenPMDFileType();
    }
    pp_diag_name.add("openpmd_backend", openpmd_backend);


    // one file per timestep (or one file for all steps)
    std::string  openpmd_encoding {"f"};
    const bool encodingDefined = pp_diag_name.query("openpmd_encoding", openpmd_encoding);

    openPMD::IterationEncoding encoding = openPMD::IterationEncoding::groupBased;
    if ( openpmd_encoding == "v" ) {
        encoding = openPMD::IterationEncoding::variableBased;
    } else if ( openpmd_encoding == "g" ) {
        encoding = openPMD::IterationEncoding::groupBased;
    } else if ( openpmd_encoding == "f" ) {
        encoding = openPMD::IterationEncoding::fileBased;
    }

    std::string diag_type_str;
    pp_diag_name.get("diag_type", diag_type_str);
    if (diag_type_str == "BackTransformed")
    {
        if ( ( openPMD::IterationEncoding::fileBased != encoding ) &&
            ( openPMD::IterationEncoding::groupBased != encoding ) )
        {
            const std::string warnMsg = diag_name+" Unable to support BTD with streaming. Using GroupBased ";
            ablastr::warn_manager::WMRecordWarning("Diagnostics", warnMsg);
            encoding = openPMD::IterationEncoding::groupBased;
        }
    }

    //
    // if no encoding is defined, then check to see if tspf is defined.
    // (backward compatibility)
    //
    if ( !encodingDefined )
    {
        bool openpmd_tspf = false;
        const bool tspfDefined = pp_diag_name.query("openpmd_tspf", openpmd_tspf);
        if ( tspfDefined && openpmd_tspf ) {
            encoding = openPMD::IterationEncoding::fileBased;
        }
    }

    // ADIOS2 operator type & parameters
    std::string operator_type;
    pp_diag_name.query("adios2_operator.type", operator_type);
    std::string const prefix = diag_name + ".adios2_operator.parameters";
    const ParmParse pp;
    auto entr = amrex::ParmParse::getEntries(prefix);

    std::map< std::string, std::string > operator_parameters;
    auto const prefix_len = prefix.size() + 1;
    for (std::string k : entr) {
        std::string v;
        pp.get(k.c_str(), v);
        k.erase(0, prefix_len);
        operator_parameters.insert({k, v});
    }

    // ADIOS2 engine type & parameters
    std::string engine_type;
    pp_diag_name.query("adios2_engine.type", engine_type);
    std::string const engine_prefix = diag_name + ".adios2_engine.parameters";
    const ParmParse ppe;
    auto eng_entr = amrex::ParmParse::getEntries(engine_prefix);

    std::map< std::string, std::string > engine_parameters;
    auto const prefixlen = engine_prefix.size() + 1;
    for (std::string k : eng_entr) {
        std::string v;
        ppe.get(k.c_str(), v);
        k.erase(0, prefixlen);
        engine_parameters.insert({k, v});
    }

    auto & warpx = WarpX::GetInstance();
    m_OpenPMDPlotWriter = std::make_unique<WarpXOpenPMDPlot>(
        encoding, openpmd_backend,
        operator_type, operator_parameters,
        engine_type, engine_parameters,
        warpx.getPMLdirections(),
        warpx.GetAuthors()
    );
}

void
FlushFormatOpenPMD::WriteToFile (
    const amrex::Vector<std::string>& varnames,
    const amrex::Vector<amrex::MultiFab>& mf,
    amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<int> iteration, const double time,
    const amrex::Vector<ParticleDiag>& particle_diags, int output_levels,
    const std::string prefix, int file_min_digits, bool plot_raw_fields,
    bool plot_raw_fields_guards,
    int verbose,
    const bool use_pinned_pc,
    bool isBTD, int snapshotID, int bufferID, int numBuffers,
    const amrex::Geometry& full_BTD_snapshot,
    bool isLastBTDFlush) const
{
    WARPX_PROFILE("FlushFormatOpenPMD::WriteToFile()");
    const std::string& filename = amrex::Concatenate(prefix, iteration[0], file_min_digits);
    if (verbose > 0) {
        if (!isBTD)
        {
            amrex::Print() << Utils::TextMsg::Info("Writing openPMD file " + filename);
        } else
        {
            amrex::Print() << Utils::TextMsg::Info("Writing buffer " + std::to_string(bufferID+1) + " of " + std::to_string(numBuffers)
                                + " to snapshot " + std::to_string(snapshotID) +  " to openPMD BTD " + prefix);
            if (isLastBTDFlush)
            {
                amrex::Print() << Utils::TextMsg::Info("Finished writing snapshot " + std::to_string(snapshotID) + " in openPMD BTD " + prefix);
            }
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !plot_raw_fields && !plot_raw_fields_guards,
        "Cannot plot raw data with OpenPMD output format. Use plotfile instead.");

    // we output at full steps of the coarsest level
    int output_iteration = iteration[0];
    // in backtransformed diagnostics (BTD), we dump into a series of labframe
    // snapshots
    if( isBTD ) {
        output_iteration = snapshotID;
    }

    // Set step and output directory name.
    m_OpenPMDPlotWriter->SetStep(output_iteration, prefix, file_min_digits, isBTD);

    // fields: only dumped for coarse level
    m_OpenPMDPlotWriter->WriteOpenPMDFieldsAll(
        varnames, mf, geom, output_levels, output_iteration, static_cast<amrex::Real>(time), isBTD, full_BTD_snapshot);

    // particles: all (reside only on locally finest level)
    m_OpenPMDPlotWriter->WriteOpenPMDParticles(
        particle_diags, static_cast<amrex::Real>(time), use_pinned_pc, isBTD, isLastBTDFlush);

    // signal that no further updates will be written to this iteration
    m_OpenPMDPlotWriter->CloseStep(isBTD, isLastBTDFlush);
}
