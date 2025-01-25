#include "FlushFormatCatalyst.H"

#include "WarpX.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <string>
#include <algorithm>

#ifdef AMREX_USE_CATALYST
    #include "conduit_cpp_to_c.hpp"
#endif

#ifdef AMREX_USE_CATALYST
namespace internal {

void EmptyParticleData (
    const std::vector<std::string>& real_fields,
    conduit::Node& node)
{
    const std::string topology_name = "particles";

    // make a dummy node for catalyst
    const std::string& patch_name = amrex::Concatenate("domain_x", 0, 6);
    conduit::Node &patch = node[patch_name];

    patch["state/domain_id"] = 0;

    std::string coordset_name = topology_name + "+coords";
    conduit::Node &n_coords = patch["coordsets"][coordset_name];
    n_coords["type"] = "explicit";

    // create an explicit points topology
    conduit::Node &n_topo = patch["topologies"][topology_name];
    n_topo["coordset"] = coordset_name;
    n_topo["type"] = "unstructured";
    n_topo["elements/shape"] = "point";
    n_topo["elements/connectivity"].set(conduit::DataType::c_int(0));

    n_coords["values/x"].set(conduit::DataType::c_int(0));
    n_coords["values/y"].set(conduit::DataType::c_int(0));
    n_coords["values/z"].set(conduit::DataType::c_int(0));

    conduit::Node &n_fields = patch["fields"];

    // id field
    conduit::Node &n_f_id = n_fields[topology_name + "_id"];
    n_f_id["topology"] = topology_name;
    n_f_id["association"] = "element";
    n_f_id["values"].set(conduit::DataType::c_int(0));

    // cpu field
    conduit::Node &n_f_cpu = n_fields[topology_name + "_cpu"];
    n_f_cpu["topology"] = topology_name;
    n_f_cpu["association"] = "element";
    n_f_cpu["values"].set(conduit::DataType::c_int(0));

    for (auto& rfield : real_fields)
    {
        conduit::Node & n_f = n_fields[rfield];
        n_f["topology"] = topology_name;
        n_f["association"] = "element";
        n_f["values"].set(conduit::DataType::c_int(0));
    }
}

} // namespace internal
#endif

using namespace amrex;

FlushFormatCatalyst::FlushFormatCatalyst() {
    ParmParse const pp_catalyst("catalyst");
    std::string scriptPaths;
    std::string implementation {"paraview"};
    std::string searchPaths;
    pp_catalyst.query("script_paths", scriptPaths);
    pp_catalyst.query("implementation", implementation);
    pp_catalyst.query("implementation_search_paths", searchPaths);

#ifdef AMREX_USE_CATALYST
    conduit::Node node;

    // Loop over all given paths and load all the scripts. Delimiters are ';' and ':'
    size_t scriptNumber = 0;
    size_t pos = 0;
    std::string subpath;
    while (scriptPaths.find(':') != std::string::npos || scriptPaths.find(';') != std::string::npos) {
        pos = std::min(scriptPaths.find(':'), scriptPaths.find(';'));
        subpath = scriptPaths.substr(0, pos);

        node["catalyst/scripts/script" + std::to_string(scriptNumber)].set_string(subpath);

        scriptNumber++;
        scriptPaths.erase(0, pos + 1);
    }
    // Prevent empty end paths
    if (scriptPaths.length() != 0) {
        node["catalyst/scripts/script" + std::to_string(scriptNumber)].set_string(scriptPaths);
    }

    node["catalyst_load/implementation"].set_string(implementation);
    node["catalyst_load/search_paths/" + implementation].set_string(searchPaths);

    catalyst_status err = catalyst_initialize(conduit::c_node(&node));
    if (err != catalyst_status_ok)
    {
        std::string message = " Error: Failed to initialize Catalyst!\n";
        std::cerr << message << err << std::endl;
        amrex::Print() << message;
        amrex::Abort(message);
    }

#endif // AMREX_USE_CATALYST
}

void
FlushFormatCatalyst::WriteToFile (
    const amrex::Vector<std::string>& varnames,
    const amrex::Vector<amrex::MultiFab>& mf,
    amrex::Vector<amrex::Geometry>& geom,
    amrex::Vector<int> iteration, double time,
    const amrex::Vector<ParticleDiag>& particle_diags, int nlev,
    const std::string prefix, int file_min_digits, bool plot_raw_fields,
    bool plot_raw_fields_guards,
    int /*verbose*/,
    bool /*use_pinned_pc*/,
    bool isBTD, int /*snapshotID*/, int /*bufferID*/, int /*numBuffers*/,
    const amrex::Geometry& /*full_BTD_snapshot*/,
    bool /*isLastBTDFlush*/) const
{
#ifdef AMREX_USE_CATALYST
    amrex::Print() << Utils::TextMsg::Info("Running Catalyst pipeline scripts...");

    WARPX_PROFILE("FlushFormatCatalyst::WriteToFile()");
    auto & warpx = WarpX::GetInstance();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !isBTD,
        "In-situ visualization is not currently supported for back-transformed diagnostics.");

    // Mesh data
    WARPX_PROFILE_VAR("FlushFormatCatalyst::WriteToFile::MultiLevelToBlueprint", prof_catalyst_mesh_blueprint);
    conduit::Node node;
    auto & state = node["catalyst/state"];
    state["timestep"].set(iteration[0]);
    state["time"].set(time);

    auto& meshChannel = node["catalyst/channels/mesh"];
    meshChannel["type"].set_string("multimesh");
    auto& meshData = meshChannel["data"];

    amrex::MultiLevelToBlueprint(
        nlev, amrex::GetVecOfConstPtrs(mf), varnames, geom, time, iteration, warpx.refRatio(), meshData);
    WARPX_PROFILE_VAR_STOP(prof_catalyst_mesh_blueprint);

    // Particle data
    WARPX_PROFILE_VAR("FlushFormatCatalyst::WriteToFile::WriteParticles", prof_catalyst_particles);
    auto& particleChannel = node["catalyst/channels/particles"];
    particleChannel["type"].set_string("multimesh");
    auto& particleData = particleChannel["data"];

    conduit::Node amrexParticles;
    WriteParticles(particle_diags, amrexParticles);
    particleData.update(amrexParticles);
    if(!particleData.dtype().is_object())
    {
        internal::EmptyParticleData(varnames, particleData);
    }

    WARPX_PROFILE_VAR_STOP(prof_catalyst_particles);

    // Execution
    WARPX_PROFILE_VAR("FlushFormatCatalyst::WriteToFile::execute", prof_catalyst_execute);
    catalyst_status err = catalyst_execute(conduit::c_node(&node));
    if (err != catalyst_status_ok)
    {
        std::string message = " Error: Failed to execute Catalyst!\n";
        std::cerr << message << err << std::endl;
        amrex::Print() << message;
    }
    WARPX_PROFILE_VAR_STOP(prof_catalyst_execute);

#else
    amrex::ignore_unused(varnames, mf, geom, iteration, time,
        particle_diags, nlev, file_min_digits, isBTD);
#endif // AMREX_USE_CATALYST
    amrex::ignore_unused(prefix, plot_raw_fields, file_min_digits, plot_raw_fields_guards);
}

#ifdef AMREX_USE_CATALYST
FlushFormatCatalyst::~FlushFormatCatalyst() {

    conduit::Node node;
    catalyst_status err = catalyst_finalize(conduit::c_node(&node));
    if (err != catalyst_status_ok)
    {
        std::string message = " Error: Failed to finalize Catalyst!\n";
        std::cerr << message << err << std::endl;
        amrex::Print() << message;
        amrex::Abort(message);
    } else {
        // Temporary, remove for final
        std::cout << "Successfully finalized Catalyst\n";
    }

}
#endif // AMREX_USE_CATALYST
