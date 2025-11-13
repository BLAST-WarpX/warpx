#include "FlushFormatAscent.H"

#include "WarpX.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <AMReX.H>
#include <AMReX_REAL.H>

using namespace amrex;

void
FlushFormatAscent::WriteToFile (
    const amrex::Vector<std::string>& varnames,
    const amrex::Vector<amrex::MultiFab>& mf,
    amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<int> iteration, const double time,
    const amrex::Vector<ParticleDiag>& particle_diags, int nlev,
    const std::string prefix, int file_min_digits, bool plot_raw_fields,
    bool plot_raw_fields_guards,
    int verbose,
    const bool /*use_pinned_pc*/,
    bool isBTD, int /*snapshotID*/, int /*bufferID*/, int /*numBuffers*/,
    const amrex::Geometry& /*full_BTD_snapshot*/,
    bool /*isLastBTDFlush*/) const
{
#ifdef AMREX_USE_ASCENT
    WARPX_PROFILE("FlushFormatAscent::WriteToFile()");
    auto & warpx = WarpX::GetInstance();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !isBTD,
        "In-situ visualization is not currently supported for back-transformed diagnostics.");

    const std::string& filename = amrex::Concatenate(prefix, iteration[0], file_min_digits);
    if (verbose > 0) {
        amrex::Print() << Utils::TextMsg::Info("Writing Ascent file " + filename);
    }

    // wrap mesh data
    WARPX_PROFILE_VAR("FlushFormatAscent::WriteToFile::MultiLevelToBlueprint", prof_ascent_mesh_blueprint);
    conduit::Node bp_mesh;
    amrex::MultiLevelToBlueprint(
        nlev, amrex::GetVecOfConstPtrs(mf), varnames, geom, time, iteration, warpx.refRatio(), bp_mesh);
    WARPX_PROFILE_VAR_STOP(prof_ascent_mesh_blueprint);

    WARPX_PROFILE_VAR("FlushFormatAscent::WriteToFile::WriteParticles", prof_ascent_particles);
    WriteParticles(particle_diags, bp_mesh);
    WARPX_PROFILE_VAR_STOP(prof_ascent_particles);

    // If you want to save blueprint HDF5 files w/o using an Ascent
    // extract, you can call the following AMReX helper:
    // const auto step = istep[0];
    // WriteBlueprintFiles(bp_mesh,"bp_export",step,"hdf5");

    WARPX_PROFILE_VAR("FlushFormatAscent::WriteToFile::publish", prof_ascent_publish);
    ascent::Ascent ascent;
    conduit::Node opts;
    opts["exceptions"] = "catch";
    opts["mpi_comm"] = MPI_Comm_c2f(ParallelDescriptor::Communicator());
    ascent.open(opts);
    ascent.publish(bp_mesh);
    WARPX_PROFILE_VAR_STOP(prof_ascent_publish);

    WARPX_PROFILE_VAR("FlushFormatAscent::WriteToFile::execute", prof_ascent_execute);
    conduit::Node actions;
    ascent.execute(actions);
    ascent.close();
    WARPX_PROFILE_VAR_STOP(prof_ascent_execute);

#else
    amrex::ignore_unused(varnames, mf, geom, iteration, time,
        particle_diags, nlev, file_min_digits, verbose, isBTD);
#endif // AMREX_USE_ASCENT
    amrex::ignore_unused(prefix, plot_raw_fields, plot_raw_fields_guards);
}
