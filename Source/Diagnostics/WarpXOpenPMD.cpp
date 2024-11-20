/* Copyright 2019-2021 Axel Huebl, Junmin Gu
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpXOpenPMD.H"

#include "Particles/ParticleIO.H"
#include "Diagnostics/ParticleDiag/ParticleDiag.H"
#include "FieldIO.H"
#include "Particles/Filter/FilterFunctors.H"
#include "Particles/NamedComponentParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/RelativeCellPosition.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"
#include "OpenPMDHelpFunction.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_DataAllocator.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Particle.H>
#include <AMReX_Particles.H>
#include <AMReX_Periodicity.H>
#include <AMReX_StructOfArrays.H>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace detail
{
#ifdef WARPX_USE_OPENPMD

    /** \brief Convert a snake_case string to a camelCase one.
     *
     *  WarpX uses snake_case internally for some component
     *  names, but OpenPMD assumes "_" indicates vector or
     *  tensor fields.
     *
     * @return camelCase version of input
     */
    inline std::string
    snakeToCamel (const std::string& snake_string)
    {
        std::string camelString = snake_string;
        const auto n = static_cast<int>(camelString.length());
        for (int x = 0; x < n; x++)
        {
            if (x == 0)
            {
                std::transform(camelString.begin(), camelString.begin()+1, camelString.begin(),
                               [](unsigned char c){ return std::tolower(c); });
            }
            if (camelString[x] == '_')
            {
                std::string tempString = camelString.substr(x + 1, 1);
                std::transform(tempString.begin(), tempString.end(), tempString.begin(),
                               [](unsigned char c){ return std::toupper(c); });
                camelString.erase(x, 2);
                camelString.insert(x, tempString);
            }
        }

        return camelString;
    }

    /** Create the option string
     *
     * @return JSON option string for openPMD::Series
     */
    inline std::string
    getSeriesOptions (std::string const & operator_type,
                      std::map< std::string, std::string > const & operator_parameters,
                      std::string const & engine_type,
                      std::map< std::string, std::string > const & engine_parameters)
    {
        if (operator_type.empty() && engine_type.empty()) {
            return "{}";
        }

        std::string options;
        std::string top_block;
        std::string end_block;
        std::string op_block;
        std::string en_block;

        std::string op_parameters;
        for (const auto& kv : operator_parameters) {
            if (!op_parameters.empty()) { op_parameters.append(",\n"); }
            op_parameters.append(std::string(12, ' '))         /* just pretty alignment */
                    .append("\"").append(kv.first).append("\": ")    /* key */
                    .append("\"").append(kv.second).append("\""); /* value (as string) */
        }

        std::string en_parameters;
        for (const auto& kv : engine_parameters) {
            if (!en_parameters.empty()) { en_parameters.append(",\n"); }
            en_parameters.append(std::string(12, ' '))         /* just pretty alignment */
                    .append("\"").append(kv.first).append("\": ")    /* key */
                    .append("\"").append(kv.second).append("\""); /* value (as string) */
        }

        // create the outer-level blocks
        top_block = R"END(
{
  "adios2": {)END";

        end_block = R"END(
  }
})END";

        // add the operator string block
        if (!operator_type.empty()) {
            op_block = R"END(
    "dataset": {
      "operators": [
        {
          "type": ")END";
            op_block += operator_type + "\"";

            if (!op_parameters.empty()) {
                op_block += R"END(,
          "parameters": {
)END";
            op_block += op_parameters +
                        "\n          }";
        }
            op_block += R"END(
        }
      ]
    })END";
            if (!engine_type.empty() || !en_parameters.empty()) {
                op_block += ",";
            }
        }  // end operator string block

        // add the engine string block
        if (!engine_type.empty() || !en_parameters.empty())
        {
            en_block = R"END(
    "engine": {)END";

            // non-default engine type
            if (!engine_type.empty()) {
                en_block += R"END(
      "type": ")END";
                en_block += engine_type + "\"";

                if(!en_parameters.empty()) {
                    en_block += ",";
                }
            }

            // non-default engine parameters
            if (!en_parameters.empty()) {
                en_block += R"END(
      "parameters": {
)END";
                en_block += en_parameters +
                            "\n      }";
            }

            en_block += R"END(
    })END";
        }  // end engine string block

        options = top_block + op_block + en_block + end_block;
        return options;
    }

    /** Unclutter a real_names to openPMD record
     *
     * @param fullName name as in real_names variable
     * @return pair of openPMD record and component name
     */
    inline std::pair< std::string, std::string >
    name2openPMD ( std::string const& fullName )
    {
        std::string record_name = fullName;
        std::string component_name = openPMD::RecordComponent::SCALAR;

        // we use "_" as separator in names to group vector records
        const std::size_t startComp = fullName.find_last_of('_');
        if( startComp != std::string::npos ) {  // non-scalar
            record_name = fullName.substr(0, startComp);
            component_name = fullName.substr(startComp + 1u);
        }
        return make_pair(record_name, component_name);
    }

    /** Return the user-selected components for particle positions
     *
     * @param[in] write_real_comp The real attribute ids, from WarpX
     * @param[in] real_comp_names The real attribute names, from WarpX
     */
    inline std::vector< std::string >
    getParticlePositionComponentLabels (
        amrex::Vector<int> const & write_real_comp,
        amrex::Vector<std::string> const & real_comp_names)
    {
        std::vector< std::string > positionComponents;

        int idx = 0;
        for (auto const & comp : real_comp_names ) {
            if (write_real_comp[idx]) {
                if (comp == "position_x" || comp == "position_y" || comp == "position_z") {
                    std::string const last_letter{comp.back()};
                    positionComponents.push_back(last_letter);
                }
            }
            idx++;
        }

        return positionComponents;
    }

    /** Return the axis (index) names of a mesh
     *
     * This will be returned in C order. This is inverse of the Fortran order
     * of the index labels for the AMReX FArrayBox.
     *
     * @param var_in_theta_mode indicate if this field will be output with theta
     *                          modes (instead of a reconstructed 2D slice)
     */
    inline std::vector< std::string >
    getFieldAxisLabels ([[maybe_unused]] bool const var_in_theta_mode)
    {
        using vs = std::vector< std::string >;

        // Fortran order of the index labels for the AMReX FArrayBox
#if defined(WARPX_DIM_1D_Z)
        vs const axisLabels{"z"};  // z varies fastest in memory
#elif defined(WARPX_DIM_XZ)
        vs const axisLabels{"x", "z"};  // x varies fastest in memory
#elif defined(WARPX_DIM_RZ)
        // when we write individual modes of a field (default)
        vs const circAxisLabels{"r", "z"};  // r varies fastest in memory
        // if we just write reconstructed 2D fields at theta=0
        vs const cartAxisLabels{"x", "z"};  // x varies fastest in memory

        vs const axisLabels = var_in_theta_mode ? circAxisLabels : cartAxisLabels;
#elif defined(WARPX_DIM_3D)
        vs const axisLabels{"x", "y", "z"};  // x varies fastest in memory
#else
#   error Unknown WarpX dimensionality.
#endif
        // revert to C order (fastest varying index last)
        return {axisLabels.rbegin(), axisLabels.rend()};
    }

    /** Return the component names of a mesh
     *
     * @param var_in_theta_mode indicate if this field will be output with theta
     *                          modes (instead of a reconstructed 2D slice)
     */
    inline std::vector< std::string >
    getFieldComponentLabels (bool const var_in_theta_mode)
    {
        using vs = std::vector< std::string >;
        if (var_in_theta_mode) {
            // if we write individual modes
            vs fieldComponents{"r", "t", "z"};
            return fieldComponents;
        } else {
            // if we just write reconstructed fields at theta=0 or are Cartesian
            // note: 1D3V and 2D3V simulations still have 3 components for the fields
            vs fieldComponents{"x", "y", "z"};
            return fieldComponents;
        }
    }

    /** Get the openPMD physical dimensionality of a record
     *
     * @param record_name name of the openPMD record
     * @return map with base quantities and power scaling
     */
    inline std::map< openPMD::UnitDimension, double >
    getUnitDimension ( std::string const & record_name )
    {

        if( (record_name == "position") || (record_name == "positionOffset") ) {
            return {{openPMD::UnitDimension::L,  1.}};
        } else if( record_name == "momentum" ) {
            return {{openPMD::UnitDimension::L,  1.},
                    {openPMD::UnitDimension::M,  1.},
                    {openPMD::UnitDimension::T, -1.}};
        } else if( record_name == "charge" ) {
            return {{openPMD::UnitDimension::T,  1.},
                    {openPMD::UnitDimension::I,  1.}};
        } else if( record_name == "mass" ) {
            return {{openPMD::UnitDimension::M, 1.}};
        } else if( record_name == "weighting" ) {  // NOLINT(bugprone-branch-clone)
#if defined(WARPX_DIM_1D_Z)
            return {{openPMD::UnitDimension::L, -2.}};
#elif defined(WARPX_DIM_XZ)
            return {{openPMD::UnitDimension::L, -1.}};
#else  // 3D and RZ
            return {};
#endif
        } else if( record_name == "E" ) {
            return {{openPMD::UnitDimension::L,  1.},
                    {openPMD::UnitDimension::M,  1.},
                    {openPMD::UnitDimension::T, -3.},
                    {openPMD::UnitDimension::I, -1.}};
        } else if( record_name == "B" ) {
            return {{openPMD::UnitDimension::M,  1.},
                    {openPMD::UnitDimension::I, -1.},
                    {openPMD::UnitDimension::T, -2.}};
        } else {  // NOLINT(bugprone-branch-clone)
            return {};
        }
    }

    /** \brief For a given field that is to be written to an openPMD file,
     * set the metadata that indicates the physical unit.
     */
    inline void
    setOpenPMDUnit ( openPMD::Mesh mesh, const std::string& field_name )
    {
        if (field_name[0] == 'E'){  // Electric field
            mesh.setUnitDimension({
                                          {openPMD::UnitDimension::L,  1},
                                          {openPMD::UnitDimension::M,  1},
                                          {openPMD::UnitDimension::T, -3},
                                          {openPMD::UnitDimension::I, -1},
                                  });
        } else if (field_name[0] == 'B'){ // Magnetic field
            mesh.setUnitDimension({
                                          {openPMD::UnitDimension::M,  1},
                                          {openPMD::UnitDimension::I, -1},
                                          {openPMD::UnitDimension::T, -2}
                                  });
        } else if (field_name[0] == 'j'){ // current
            mesh.setUnitDimension({
                                          {openPMD::UnitDimension::L, -2},
                                          {openPMD::UnitDimension::I,  1},
                                  });
        } else if (field_name.substr(0,3) == "rho"){ // charge density
            mesh.setUnitDimension({
                                          {openPMD::UnitDimension::L, -3},
                                          {openPMD::UnitDimension::I,  1},
                                          {openPMD::UnitDimension::T,  1},
                                  });
        }
    }
#endif // WARPX_USE_OPENPMD
} // namespace detail

#ifdef WARPX_USE_OPENPMD
WarpXOpenPMDPlot::WarpXOpenPMDPlot (
    openPMD::IterationEncoding ie,
    const std::string& openPMDFileType,
    const std::string& operator_type,
    const std::map< std::string, std::string >& operator_parameters,
    const std::string& engine_type,
    const std::map< std::string, std::string >& engine_parameters,
    const std::vector<bool>& fieldPMLdirections,
    const std::string& authors)
    : m_Series(nullptr),
      m_MPIRank{amrex::ParallelDescriptor::MyProc()},
      m_MPISize{amrex::ParallelDescriptor::NProcs()},
      m_Encoding(ie),
      m_OpenPMDFileType{openPMDFileType},
      m_fieldPMLdirections{fieldPMLdirections},
      m_authors{authors}
{
    m_OpenPMDoptions = detail::getSeriesOptions(operator_type, operator_parameters,
                                                engine_type, engine_parameters);
}

WarpXOpenPMDPlot::~WarpXOpenPMDPlot ()
{
  if( m_Series )
  {
    m_Series->flush();
    m_Series.reset( nullptr );
  }
}

std::string
WarpXOpenPMDPlot::GetFileName (std::string& filepath)
{
    filepath.append("/");
    // transform paths for Windows
    #ifdef _WIN32
    filepath = openPMD::auxiliary::replace_all(filepath, "/", "\\");
    #endif

    std::string filename = "openpmd";
    //
    // OpenPMD supports timestepped names
    //
    if (m_Encoding == openPMD::IterationEncoding::fileBased) {
        const std::string fileSuffix = std::string("_%0") + std::to_string(m_file_min_digits) + std::string("T");
        filename = filename.append(fileSuffix);
    }
    filename.append(".").append(m_OpenPMDFileType);
    filepath.append(filename);
    return filename;
}

void WarpXOpenPMDPlot::SetStep (int ts, const std::string& dirPrefix, int file_min_digits,
                                bool isBTD)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ts >= 0 , "openPMD iterations are unsigned");

    m_dirPrefix = dirPrefix;
    m_file_min_digits = file_min_digits;

    if( ! isBTD ) {
        if (m_CurrentStep >= ts) {
            // note m_Series is reset in Init(), so using m_Series->iterations.contains(ts) is only able to check the
            // last written step in m_Series's life time, but not other earlier written steps by other m_Series
            ablastr::warn_manager::WMRecordWarning("Diagnostics",
                    " Warning from openPMD writer: Already written iteration:"
                    + std::to_string(ts)
                );
        }
    }

    m_CurrentStep = ts;
    Init(openPMD::Access::CREATE, isBTD);
}

void WarpXOpenPMDPlot::CloseStep (bool isBTD, bool isLastBTDFlush)
{
    // default close is true
    bool callClose = true;
    // close BTD file only when isLastBTDFlush is true
    if (isBTD and !isLastBTDFlush) { callClose = false; }
    if (callClose) {
        if (m_Series) {
            GetIteration(m_CurrentStep, isBTD).close();
        }

        // create a little helper file for ParaView 5.9+
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            // see Init()
            std::string filepath = m_dirPrefix;
            std::string const filename = GetFileName(filepath);

            std::ofstream pv_helper_file(m_dirPrefix + "/paraview.pmd");
            pv_helper_file << filename << "\n";
            pv_helper_file.close();
        }
    }
}

void
WarpXOpenPMDPlot::Init (openPMD::Access access, bool isBTD)
{
    if( isBTD && m_Series != nullptr ) {
        return; // already open for this snapshot (aka timestep in lab frame)
    }

    // either for the next ts file,
    // or init a single file for all ts
    std::string filepath = m_dirPrefix;
    GetFileName(filepath);

    // close a previously open series before creating a new one
    // see ADIOS1 limitation: https://github.com/openPMD/openPMD-api/pull/686
    if ( m_Encoding == openPMD::IterationEncoding::fileBased ) {
        m_Series = nullptr;
    } else if ( m_Series != nullptr ) {
        return;
    }

    if (amrex::ParallelDescriptor::NProcs() > 1) {
#if defined(AMREX_USE_MPI)
        m_Series = std::make_unique<openPMD::Series>(
                filepath, access,
                amrex::ParallelDescriptor::Communicator(),
                m_OpenPMDoptions
        );
#else
        WARPX_ABORT_WITH_MESSAGE("openPMD-api not built with MPI support!");
#endif
    } else {
        m_Series = std::make_unique<openPMD::Series>(filepath, access, m_OpenPMDoptions);
    }

    m_Series->setIterationEncoding( m_Encoding );

    // input file / simulation setup author
    if( !m_authors.empty()) {
        m_Series->setAuthor( m_authors );
    }
    // more natural naming for PIC
    m_Series->setMeshesPath( "fields" );
    // conform to ED-PIC extension of openPMD
    uint32_t const openPMD_ED_PIC = 1u;
    m_Series->setOpenPMDextension( openPMD_ED_PIC );
    // meta info
    m_Series->setSoftware( "WarpX", WarpX::Version() );
}

void
WarpXOpenPMDPlot::WriteOpenPMDParticles (const amrex::Vector<ParticleDiag>& particle_diags,
                  const amrex::Real time,
                  const bool use_pinned_pc,
                  const bool isBTD,
                  const bool isLastBTDFlush
)
{
WARPX_PROFILE("WarpXOpenPMDPlot::WriteOpenPMDParticles()");

for (unsigned i = 0, n = particle_diags.size(); i < n; ++i) {

    WarpXParticleContainer* pc = particle_diags[i].getParticleContainer();
    PinnedMemoryParticleContainer* pinned_pc = particle_diags[i].getPinnedParticleContainer();
    if (isBTD || use_pinned_pc) {
        if (!pinned_pc->isDefined()) {
            continue;  // Skip to the next particle container
        }
    }

    PinnedMemoryParticleContainer tmp = (isBTD || use_pinned_pc) ?
        pinned_pc->make_alike<amrex::PinnedArenaAllocator>() :
        pc->make_alike<amrex::PinnedArenaAllocator>();

    const auto mass = pc->AmIA<PhysicalSpecies::photon>() ? PhysConst::m_e : pc->getMass();
    RandomFilter const random_filter(particle_diags[i].m_do_random_filter,
                                     particle_diags[i].m_random_fraction);
    UniformFilter const uniform_filter(particle_diags[i].m_do_uniform_filter,
                                       particle_diags[i].m_uniform_stride);
    ParserFilter parser_filter(particle_diags[i].m_do_parser_filter,
                               utils::parser::compileParser<ParticleDiag::m_nvars>
                                     (particle_diags[i].m_particle_filter_parser.get()),
                                 pc->getMass(), time);
    parser_filter.m_units = InputUnits::SI;
    GeometryFilter const geometry_filter(particle_diags[i].m_do_geom_filter,
                                           particle_diags[i].m_diag_domain);

    if (isBTD || use_pinned_pc) {
        particlesConvertUnits(ConvertDirection::WarpX_to_SI, pinned_pc, mass);
        using SrcData = WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType;
        tmp.copyParticles(*pinned_pc,
            [random_filter,uniform_filter,parser_filter,geometry_filter]
            AMREX_GPU_HOST_DEVICE
            (const SrcData& src, int ip, const amrex::RandomEngine& engine)
            {
                const SuperParticleType& p = src.getSuperParticle(ip);
                return random_filter(p, engine) * uniform_filter(p, engine)
                        * parser_filter(p, engine) * geometry_filter(p, engine);
            }, true);
        particlesConvertUnits(ConvertDirection::SI_to_WarpX, pinned_pc, mass);
    } else {
        particlesConvertUnits(ConvertDirection::WarpX_to_SI, pc, mass);
        using SrcData = WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType;
        tmp.copyParticles(*pc,
            [random_filter,uniform_filter,parser_filter,geometry_filter]
            AMREX_GPU_HOST_DEVICE
            (const SrcData& src, int ip, const amrex::RandomEngine& engine)
            {
                const SuperParticleType& p = src.getSuperParticle(ip);
                return random_filter(p, engine) * uniform_filter(p, engine)
                        * parser_filter(p, engine) * geometry_filter(p, engine);
            }, true);
        particlesConvertUnits(ConvertDirection::SI_to_WarpX, pc, mass);
    }

    // Gather the electrostatic potential (phi) on the macroparticles
    if ( particle_diags[i].m_plot_phi ) {
        storePhiOnParticles( tmp, WarpX::electrostatic_solver_id, !use_pinned_pc );
    }

    // names of amrex::Real and int particle attributes in SoA data
    amrex::Vector<std::string> real_names;
    amrex::Vector<std::string> int_names;
    amrex::Vector<int> int_flags;
    amrex::Vector<int> real_flags;
    // see openPMD ED-PIC extension for namings
    // note: an underscore separates the record name from its component
    //       for non-scalar records
    // note: in RZ, we reconstruct x,y,z positions from r,z,theta in WarpX
#if !defined (WARPX_DIM_1D_Z)
    real_names.push_back("position_x");
#endif
#if defined (WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
    real_names.push_back("position_y");
#endif
    real_names.push_back("position_z");
    real_names.push_back("weighting");
    real_names.push_back("momentum_x");
    real_names.push_back("momentum_y");
    real_names.push_back("momentum_z");
    // get the names of the real comps
    real_names.resize(tmp.NumRealComps());
    auto runtime_rnames = tmp.getParticleRuntimeComps();
    for (auto const& x : runtime_rnames)
    {
        real_names[x.second+PIdx::nattribs] = detail::snakeToCamel(x.first);
    }
    // plot any "extra" fields by default
    real_flags = particle_diags[i].m_plot_flags;
    real_flags.resize(tmp.NumRealComps(), 1);
    // and the names
    int_names.resize(tmp.NumIntComps());
    auto runtime_inames = tmp.getParticleRuntimeiComps();
    for (auto const& x : runtime_inames)
    {
        int_names[x.second+0] = detail::snakeToCamel(x.first);
    }
    // plot by default
    int_flags.resize(tmp.NumIntComps(), 1);

    // real_names contains a list of all real particle attributes.
    // real_flags is 1 or 0, whether quantity is dumped or not.
    DumpToFile(&tmp,
        particle_diags.at(i).getSpeciesName(),
        m_CurrentStep,
        real_flags,
        int_flags,
        real_names, int_names,
        pc->getCharge(), pc->getMass(),
        isBTD, isLastBTDFlush);
    }
}

void
WarpXOpenPMDPlot::DumpToFile (ParticleContainer* pc,
                    const std::string& name,
                    int iteration,
                    const amrex::Vector<int>& write_real_comp,
                    const amrex::Vector<int>& write_int_comp,
                    const amrex::Vector<std::string>& real_comp_names,
                    const amrex::Vector<std::string>&  int_comp_names,
                    amrex::ParticleReal const charge,
                    amrex::ParticleReal const mass,
                    const bool isBTD,
                    const bool isLastBTDFlush
)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_Series != nullptr, "openPMD: series must be initialized");

    AMREX_ALWAYS_ASSERT(write_real_comp.size() == pc->NumRealComps());
    AMREX_ALWAYS_ASSERT(write_int_comp.size() == pc->NumIntComps());
    AMREX_ALWAYS_ASSERT(real_comp_names.size() == pc->NumRealComps());
    AMREX_ALWAYS_ASSERT(int_comp_names.size() == pc->NumIntComps());

    WarpXParticleCounter counter(pc);
    auto const num_dump_particles = counter.GetTotalNumParticles();

    openPMD::Iteration currIteration = GetIteration(iteration, isBTD);
    openPMD::ParticleSpecies currSpecies = currIteration.particles[name];

    // only BTD writes multiple times into the same step, zero for other methods
    const unsigned long ParticleFlushOffset = isBTD ? num_already_flushed(currSpecies) : 0;

    // prepare data structures the first time BTD has non-zero particles
    //   we set some of them to zero extent, so we need to time that well
    bool const is_first_flush_with_particles = num_dump_particles > 0 && ParticleFlushOffset == 0;
    // BTD: we flush multiple times to the same lab step and thus need to resize
    //   our declared particle output sizes
    bool const is_resizing_flush = num_dump_particles > 0 && ParticleFlushOffset > 0;
    // write structure & declare particles in this (lab) step empty:
    //   if not BTD, then this is the only (and last) time we flush to this step
    //   if BTD, then we may do this multiple times until it is the last BTD flush
    bool const is_last_flush_to_step = !isBTD || (isBTD && isLastBTDFlush);
    // well, even in BTD we have to recognize that some lab stations may have no
    //   particles - so we mark them empty at the end of station reconstruction
    bool const is_last_flush_and_never_particles =
            is_last_flush_to_step && num_dump_particles == 0 && ParticleFlushOffset == 0;

    //
    // prepare structure and meta-data
    //

    // define positions & offset structure
    const unsigned long long NewParticleVectorSize = num_dump_particles + ParticleFlushOffset;
    // we will set up empty particles unless it's BTD, where we might add some in a following buffer dump
    //   during this setup, we mark some particle properties as constant and potentially zero-sized
    bool doParticleSetup = true;
    if (isBTD) {
        doParticleSetup = is_first_flush_with_particles || is_last_flush_and_never_particles;
    }

    auto const positionComponents = detail::getParticlePositionComponentLabels(write_real_comp, real_comp_names);

    // this setup stage also implicitly calls "makeEmpty" if needed (i.e., is_last_flush_and_never_particles)
    //   for BTD, we call this multiple times as we may resize in subsequent dumps if number of particles in the buffer > 0
    if (doParticleSetup || is_resizing_flush) {
        SetupPos(currSpecies, positionComponents, NewParticleVectorSize, isBTD);
        SetupRealProperties(pc, currSpecies, write_real_comp, real_comp_names, write_int_comp, int_comp_names,
                            NewParticleVectorSize, isBTD);
    }

    if (is_last_flush_to_step) {
        SetConstParticleRecordsEDPIC(currSpecies, positionComponents, NewParticleVectorSize, charge, mass);
    }

    // open files from all processors, in case some will not contribute below
    m_Series->flush();

    // dump individual particles
    bool contributed_particles = false;  // did the local MPI rank contribute particles?
    for (auto currentLevel = 0; currentLevel <= pc->finestLevel(); currentLevel++) {
        auto offset = static_cast<uint64_t>( counter.m_ParticleOffsetAtRank[currentLevel] );
        // For BTD, the offset include the number of particles already flushed
        if (isBTD) { offset += ParticleFlushOffset; }
        for (ParticleIter pti(*pc, currentLevel); pti.isValid(); ++pti) {
            auto const numParticleOnTile = pti.numParticles();
            auto const numParticleOnTile64 = static_cast<uint64_t>( numParticleOnTile );

            // Do not call storeChunk() with zero-sized particle tiles:
            //   https://github.com/openPMD/openPMD-api/issues/1147
            //   https://github.com/ECP-WarpX/WarpX/pull/1898#discussion_r745008290
            if (numParticleOnTile == 0) { continue; }

            contributed_particles = true;

            //  save particle properties
            SaveRealProperty(pti,
                             currSpecies,
                             offset,
                             write_real_comp, real_comp_names,
                             write_int_comp, int_comp_names);

            offset += numParticleOnTile64;
        } // pti
    } // currentLevel

    // work-around for BTD particle resize in ADIOS2
    //
    // This issues an empty ADIOS2 Put to make sure the new global shape
    // meta-data is committed for each variable.
    //
    // Refs.:
    //   https://github.com/ECP-WarpX/WarpX/issues/3389
    //   https://github.com/ornladios/ADIOS2/issues/3455
    //   BP4 (ADIOS 2.8): last MPI rank's `Put` meta-data wins
    //   BP5 (ADIOS 2.8): everyone has to write an empty block
    if (is_resizing_flush && !contributed_particles && isBTD && m_Series->backend() == "ADIOS2") {
        for( auto & [record_name, record] : currSpecies ) {
            for( auto & [comp_name, comp] : record ) {
                if (comp.constant()) { continue; }

                auto dtype = comp.getDatatype();
                switch (dtype) {
                    case openPMD::Datatype::FLOAT :
                        [[fallthrough]];
                    case openPMD::Datatype::DOUBLE : {
                        auto empty_data = std::make_shared<amrex::ParticleReal>();
                        comp.storeChunk(empty_data, {uint64_t(0)}, {uint64_t(0)});
                        break;
                    }
                    case openPMD::Datatype::UINT : {
                        auto empty_data = std::make_shared<unsigned int>();
                        comp.storeChunk(empty_data, {uint64_t(0)}, {uint64_t(0)});
                        break;
                    }
                    case openPMD::Datatype::ULONG : {
                        auto empty_data = std::make_shared<unsigned long>();
                        comp.storeChunk(empty_data, {uint64_t(0)}, {uint64_t(0)});
                        break;
                    }
                    case openPMD::Datatype::ULONGLONG : {
                        auto empty_data = std::make_shared<unsigned long long>();
                        comp.storeChunk(empty_data, {uint64_t(0)}, {uint64_t(0)});
                        break;
                    }
                    default : {
                        std::string msg = "WarpX openPMD ADIOS2 work-around has unknown dtype: ";
                        msg += datatypeToString(dtype);
                        WARPX_ABORT_WITH_MESSAGE(msg);
                        break;
                    }
                }
            }
        }
    }

    m_Series->flush();
}

void
WarpXOpenPMDPlot::SetupRealProperties (ParticleContainer const * pc,
                      openPMD::ParticleSpecies& currSpecies,
                      const amrex::Vector<int>& write_real_comp,
                      const amrex::Vector<std::string>& real_comp_names,
                      const amrex::Vector<int>& write_int_comp,
                      const amrex::Vector<std::string>& int_comp_names,
                      const unsigned long long np, bool const isBTD) const
{
    std::string options = "{}";
    if (isBTD) { options = "{ \"resizable\": true }"; }
    auto dtype_real = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), {np}, options);
    auto dtype_int  = openPMD::Dataset(openPMD::determineDatatype<int>(), {np}, options);
    //
    // the beam/input3d showed write_real_comp.size() = 16 while only 10 real comp names
    // so using the min to be safe.
    //
    auto const getComponentRecord = [&currSpecies](std::string const& comp_name) {
        // handle scalar and non-scalar records by name
        const auto [record_name, component_name] = detail::name2openPMD(comp_name);
        return currSpecies[record_name][component_name];
    };
    auto const real_counter = std::min(write_real_comp.size(), real_comp_names.size());
    for (int i = 0; i < real_counter; ++i) {
      if (write_real_comp[i]) {
          getComponentRecord(real_comp_names[i]).resetDataset(dtype_real);
      }
    }
    auto const int_counter = std::min(write_int_comp.size(), int_comp_names.size());
    for (int i = 0; i < int_counter; ++i) {
        if (write_int_comp[i]) {
            getComponentRecord(int_comp_names[i]).resetDataset(dtype_int);
        }
    }

    std::set< std::string > addedRecords; // add meta-data per record only once
    for (auto idx=0; idx<pc->NumRealComps(); idx++) {
        if (write_real_comp[idx]) {
            // handle scalar and non-scalar records by name
            const auto [record_name, component_name] = detail::name2openPMD(real_comp_names[idx]);
            auto currRecord = currSpecies[record_name];

            // meta data for ED-PIC extension
            [[maybe_unused]] const auto [_, newRecord] = addedRecords.insert(record_name);
            if( newRecord ) {
                currRecord.setUnitDimension( detail::getUnitDimension(record_name) );
                if( record_name == "weighting" ) {
                    currRecord.setAttribute( "macroWeighted", 1u );
                } else {
                    currRecord.setAttribute( "macroWeighted", 0u );
                }
                if( record_name == "momentum" || record_name == "weighting" ) {
                    currRecord.setAttribute( "weightingPower", 1.0 );
                } else {
                    currRecord.setAttribute( "weightingPower", 0.0 );
                }
            }
        }
    }
    for (auto idx=0; idx<int_counter; idx++) {
        if (write_int_comp[idx]) {
            // handle scalar and non-scalar records by name
            const auto [record_name, component_name] = detail::name2openPMD(int_comp_names[idx]);
            auto currRecord = currSpecies[record_name];

            // meta data for ED-PIC extension
            [[maybe_unused]] const auto [_, newRecord] = addedRecords.insert(record_name);
            if( newRecord ) {
                currRecord.setUnitDimension( detail::getUnitDimension(record_name) );
                currRecord.setAttribute( "macroWeighted", 0u );
                if( record_name == "momentum" || record_name == "weighting" ) {
                    currRecord.setAttribute( "weightingPower", 1.0 );
                } else {
                    currRecord.setAttribute( "weightingPower", 0.0 );
                }
            }
        }
    }
}

void
WarpXOpenPMDPlot::SaveRealProperty (ParticleIter& pti,
                       openPMD::ParticleSpecies& currSpecies,
                       unsigned long long const offset,
                       amrex::Vector<int> const& write_real_comp,
                       amrex::Vector<std::string> const& real_comp_names,
                       amrex::Vector<int> const& write_int_comp,
                       amrex::Vector<std::string> const& int_comp_names) const

{
    auto const numParticleOnTile = pti.numParticles();
    auto const numParticleOnTile64 = static_cast<uint64_t>(numParticleOnTile);
    auto const& soa = pti.GetStructOfArrays();

    auto const getComponentRecord = [&currSpecies](std::string const& comp_name) {
        // handle scalar and non-scalar records by name
        const auto [record_name, component_name] = detail::name2openPMD(comp_name);
        return currSpecies[record_name][component_name];
    };

    // here we the save the SoA properties (idcpu)
    {
        // todo: add support to not write the particle index
        getComponentRecord("id").storeChunkRaw(
            soa.GetIdCPUData().data(), {offset}, {numParticleOnTile64});
    }

    // here we the save the SoA properties (real)
    {
        auto const real_counter = std::min(write_real_comp.size(), real_comp_names.size());

#if defined(WARPX_DIM_RZ)
        // reconstruct Cartesian positions for RZ simulations
        // r,z,theta -> x,y,z
        // If each comp is being written, create a temporary array, otherwise create an empty array.
        std::shared_ptr<amrex::ParticleReal> const x(
            new amrex::ParticleReal[(write_real_comp[0] ? numParticleOnTile : 0)],
            [](amrex::ParticleReal const *p) { delete[] p; }
        );
        std::shared_ptr<amrex::ParticleReal> const y(
            new amrex::ParticleReal[(write_real_comp[1] ? numParticleOnTile : 0)],
            [](amrex::ParticleReal const *p) { delete[] p; }
        );

        const auto& tile = pti.GetParticleTile();
        const auto& ptd = tile.getConstParticleTileData();

        for (int i = 0; i < numParticleOnTile; ++i) {
            const auto& p = ptd.getSuperParticle(i);
            amrex::ParticleReal xp, yp, zp;
            get_particle_position(p, xp, yp, zp);
            if (write_real_comp[0]) { x.get()[i] = xp; }
            if (write_real_comp[1]) { y.get()[i] = yp; }
        }
        if (write_real_comp[0]) {
            getComponentRecord(real_comp_names[0]).storeChunk(x, {offset}, {numParticleOnTile64});
        }
        if (write_real_comp[1]) {
            getComponentRecord(real_comp_names[1]).storeChunk(y, {offset}, {numParticleOnTile64});
        }
#endif

        for (auto idx=0; idx<real_counter; idx++) {
#if defined(WARPX_DIM_RZ)
            // skip over x,y
            if (idx < 2) {
                continue;
            }
            // mak names and write flags to SoA real array number
            int const soa_r_idx = idx - 1 < PIdx::theta ?
                idx - 1 :  // z and momenta before theta (we added y)
                idx        // jump over theta (skipped)
            ;
#else
            int const soa_r_idx = idx;
#endif
            if (write_real_comp[idx]) {
                getComponentRecord(real_comp_names[idx]).storeChunkRaw(
                    soa.GetRealData(soa_r_idx).data(), {offset}, {numParticleOnTile64});
            }
        }
    }
    // and now SoA int properties
    {
        auto const int_counter = std::min(write_int_comp.size(), int_comp_names.size());
        for (auto idx=0; idx<int_counter; idx++) {
            if (write_int_comp[idx]) {
                getComponentRecord(int_comp_names[idx]).storeChunkRaw(
                    soa.GetIntData(idx).data(), {offset}, {numParticleOnTile64});
            }
        }
    }
}


void
WarpXOpenPMDPlot::SetupPos (
    openPMD::ParticleSpecies& currSpecies,
    std::vector<std::string> const & positionComponents,
    const unsigned long long& np,
    bool const isBTD)
{
    std::string options = "{}";
    if (isBTD) { options = "{ \"resizable\": true }"; }
    auto realType = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), {np}, options);
    auto idType = openPMD::Dataset(openPMD::determineDatatype< uint64_t >(), {np}, options);

    for( auto const& comp : positionComponents ) {
        currSpecies["position"][comp].resetDataset( realType );
    }

    const auto *const scalar = openPMD::RecordComponent::SCALAR;
    currSpecies["id"][scalar].resetDataset( idType );
}

void
WarpXOpenPMDPlot::SetConstParticleRecordsEDPIC (
        openPMD::ParticleSpecies& currSpecies,
        std::vector<std::string> const & positionComponents,
        const unsigned long long& np,
        amrex::ParticleReal const charge,
        amrex::ParticleReal const mass)
{
    auto realType = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), {np});
    const auto *const scalar = openPMD::RecordComponent::SCALAR;

    // define record shape to be number of particles
    for( auto const& comp : positionComponents ) {
        currSpecies["positionOffset"][comp].resetDataset( realType );
    }
    currSpecies["charge"][scalar].resetDataset( realType );
    currSpecies["mass"][scalar].resetDataset( realType );
#if defined(WARPX_DIM_1D_Z)
    currSpecies["position"]["x"].resetDataset( realType );
    currSpecies["position"]["y"].resetDataset( realType );
    currSpecies["positionOffset"]["x"].resetDataset( realType );
    currSpecies["positionOffset"]["y"].resetDataset( realType );
#endif
#if defined(WARPX_DIM_XZ)
    currSpecies["position"]["y"].resetDataset( realType );
    currSpecies["positionOffset"]["y"].resetDataset( realType );
#endif

    // make constant
    using namespace amrex::literals;
    for( auto const& comp : positionComponents ) {
        currSpecies["positionOffset"][comp].makeConstant( 0._prt );
    }
    currSpecies["charge"][scalar].makeConstant( charge );
    currSpecies["mass"][scalar].makeConstant( mass );
    //   convention: in 1D3V and 2D3V, omitted positions are set to zero
#if defined(WARPX_DIM_1D_Z)
    currSpecies["position"]["x"].makeConstant( 0._prt );
    currSpecies["position"]["y"].makeConstant( 0._prt );
    currSpecies["positionOffset"]["x"].makeConstant( 0._prt );
    currSpecies["positionOffset"]["y"].makeConstant( 0._prt );
#endif
#if defined(WARPX_DIM_XZ)
    currSpecies["position"]["y"].makeConstant( 0._prt );
    currSpecies["positionOffset"]["y"].makeConstant( 0._prt );
#endif

    // meta data
    if (!positionComponents.empty()) {
        currSpecies["position"].setUnitDimension( detail::getUnitDimension("position") );
        currSpecies["positionOffset"].setUnitDimension( detail::getUnitDimension("positionOffset") );
    }
    currSpecies["charge"].setUnitDimension( detail::getUnitDimension("charge") );
    currSpecies["mass"].setUnitDimension( detail::getUnitDimension("mass") );

    // meta data for ED-PIC extension
    if (!positionComponents.empty()) {
        currSpecies["position"].setAttribute( "macroWeighted", 0u );
        currSpecies["position"].setAttribute( "weightingPower", 0.0 );
        currSpecies["positionOffset"].setAttribute( "macroWeighted", 0u );
        currSpecies["positionOffset"].setAttribute( "weightingPower", 0.0 );
    }
    currSpecies["id"].setAttribute( "macroWeighted", 0u );
    currSpecies["id"].setAttribute( "weightingPower", 0.0 );
    currSpecies["charge"].setAttribute( "macroWeighted", 0u );
    currSpecies["charge"].setAttribute( "weightingPower", 1.0 );
    currSpecies["mass"].setAttribute( "macroWeighted", 0u );
    currSpecies["mass"].setAttribute( "weightingPower", 1.0 );

    // more ED-PIC attributes
    currSpecies.setAttribute("particleShape", double(WarpX::noz));
    // TODO allow this per direction in the openPMD standard, ED-PIC extension?
    currSpecies.setAttribute("particleShapes", []() {
        return std::vector<double>{
#if AMREX_SPACEDIM >= 2
                double(WarpX::nox),
#endif
#if defined(WARPX_DIM_3D)
                double(WarpX::noy),
#endif
                double(WarpX::noz)
        };
    }());
    currSpecies.setAttribute("particlePush", []() {
        switch (WarpX::particle_pusher_algo) {
            case ParticlePusherAlgo::Boris :
                return "Boris";
            case ParticlePusherAlgo::Vay :
                return "Vay";
            case ParticlePusherAlgo::HigueraCary :
                return "HigueraCary";
            default:
                return "other";
        }
    }());
    currSpecies.setAttribute("particleInterpolation", []() {
        switch (WarpX::field_gathering_algo) {
            case GatheringAlgo::EnergyConserving :
                return "energyConserving";
            case GatheringAlgo::MomentumConserving :
                return "momentumConserving";
            default:
                return "other";
        }
    }());
    currSpecies.setAttribute("particleSmoothing", "none");
    currSpecies.setAttribute("currentDeposition", []() {
        switch (WarpX::current_deposition_algo) {
            case CurrentDepositionAlgo::Esirkepov :
                return "Esirkepov";
            case CurrentDepositionAlgo::Vay :
                return "Vay";
            case CurrentDepositionAlgo::Villasenor :
                return "Villasenor";
            default:
                return "directMorseNielson";
        }
    }());
}


/*
 * Set up parameter for mesh container using the geometry (from level 0)
 *
 * @param [in] meshes: openPMD-api mesh container
 * @param [in] full_geom: field geometry
 *
 */
void
WarpXOpenPMDPlot::SetupFields ( openPMD::Container< openPMD::Mesh >& meshes,
                                amrex::Geometry& full_geom ) const
{
    // meta data for ED-PIC extension
    auto const period = full_geom.periodicity(); // TODO double-check: is this the proper global bound or of some level?
    std::vector<std::string> fieldBoundary(6, "reflecting");
    std::vector<std::string> particleBoundary(6, "absorbing");
    fieldBoundary.resize(AMREX_SPACEDIM * 2);
    particleBoundary.resize(AMREX_SPACEDIM * 2);

    const auto HalfFieldBoundarySize = static_cast<int>(fieldBoundary.size() / 2u);

    for (auto i = 0; i < HalfFieldBoundarySize; ++i) {
        if (m_fieldPMLdirections.at(i)) {
            fieldBoundary.at(i) = "open";
        }
    }

    for (int i = 0; i < HalfFieldBoundarySize; ++i) {
        if (period.isPeriodic(i)) {
            fieldBoundary.at(2u * i) = "periodic";
            fieldBoundary.at(2u * i + 1u) = "periodic";
            particleBoundary.at(2u * i) = "periodic";
            particleBoundary.at(2u * i + 1u) = "periodic";
        }
    }

    meshes.setAttribute("fieldSolver", []() {
        switch (WarpX::electromagnetic_solver_id) {
            case ElectromagneticSolverAlgo::Yee :
                return "Yee";
            case ElectromagneticSolverAlgo::CKC :
                return "CK";
            case ElectromagneticSolverAlgo::PSATD :
                return "PSATD";
            default:
                return "other";
        }
    }());
    meshes.setAttribute("fieldBoundary", fieldBoundary);
    meshes.setAttribute("particleBoundary", particleBoundary);
    meshes.setAttribute("currentSmoothing", []() {
        if (WarpX::use_filter) { return "Binomial"; }
        else { return "none"; }
    }());
    if (WarpX::use_filter) {
        meshes.setAttribute("currentSmoothingParameters", []() {
            std::stringstream ss;
            ss << "period=1;compensator=false";
#if (AMREX_SPACEDIM >= 2)
            ss << ";numPasses_x=" << WarpX::filter_npass_each_dir[0];
#endif
#if defined(WARPX_DIM_3D)
            ss << ";numPasses_y=" << WarpX::filter_npass_each_dir[1];
            ss << ";numPasses_z=" << WarpX::filter_npass_each_dir[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            ss << ";numPasses_z=" << WarpX::filter_npass_each_dir[1];
#elif defined(WARPX_DIM_1D_Z)
            ss << ";numPasses_z=" << WarpX::filter_npass_each_dir[0];
#endif
            std::string currentSmoothingParameters = ss.str();
            return currentSmoothingParameters;
        }());
    }
    meshes.setAttribute("chargeCorrection", []() {
        if (WarpX::do_dive_cleaning) { return "hyperbolic"; // TODO or "spectral" or something? double-check
        } else { return "none"; }
    }());
    if (WarpX::do_dive_cleaning) {
        meshes.setAttribute("chargeCorrectionParameters", "period=1");
    }
    // TODO set meta-data information for time-averaged quantities
    // but we need information of the specific diagnostic in here
}


/*
 * Setup component properties  for a field mesh
 * @param [in]: mesh          a mesh field
 * @param [in]: full_geom     geometry for the mesh
 * @param [in]: mesh_comp     a component for the mesh
 */
void
WarpXOpenPMDPlot::SetupMeshComp (openPMD::Mesh& mesh,
                                 amrex::Geometry const& full_geom,
                                 std::string const& comp_name,
                                 std::string const& field_name,
                                 amrex::MultiFab const& mf,
                                 bool var_in_theta_mode) const
{
    auto mesh_comp = mesh[comp_name];
    amrex::Box const & global_box = full_geom.Domain();
    auto global_size = getReversedVec(global_box.size() );
    // - Grid spacing
    std::vector<double> const grid_spacing = getReversedVec(full_geom.CellSize());
    // - Global offset
    std::vector<double> const global_offset = getReversedVec(full_geom.ProbLo());
#if defined(WARPX_DIM_RZ)
    if (var_in_theta_mode) {
            global_size.emplace(global_size.begin(), WarpX::ncomps);
    }
#endif
    // - AxisLabels
    const std::vector<std::string> axis_labels = detail::getFieldAxisLabels(var_in_theta_mode);

    // Prepare the type of dataset that will be written
    openPMD::Datatype const datatype = openPMD::determineDatatype<amrex::Real>();
    auto const dataset = openPMD::Dataset(datatype, global_size);
    mesh.setDataOrder(openPMD::Mesh::DataOrder::C);
    if (var_in_theta_mode) {
        mesh.setGeometry("thetaMode");
        mesh.setGeometryParameters("m=" + std::to_string(WarpX::n_rz_azimuthal_modes) + ";imag=+");
    }
    mesh.setAxisLabels(axis_labels);
    mesh.setGridSpacing(grid_spacing);
    mesh.setGridGlobalOffset(global_offset);
    mesh.setAttribute("fieldSmoothing", "none");
    mesh_comp.resetDataset(dataset);

    detail::setOpenPMDUnit( mesh, field_name );
    auto relative_cell_pos = utils::getRelativeCellPosition(mf);     // AMReX Fortran index order
    std::reverse( relative_cell_pos.begin(), relative_cell_pos.end() ); // now in C order
    mesh_comp.setPosition( relative_cell_pos );
}

void
WarpXOpenPMDPlot::GetMeshCompNames (int meshLevel,
                                    const std::string& varname,
                                    std::string& field_name,
                                    std::string& comp_name,
                                    bool var_in_theta_mode) const
{
    if (varname.size() >= 2u ) {
        std::string const varname_1st = varname.substr(0u, 1u); // 1st character
        std::string const varname_2nd = varname.substr(1u, 1u); // 2nd character

        // Check if this field is a vector. If so, then extract the field name
        std::vector< std::string > const vector_fields = {"E", "B", "j"};
        std::vector< std::string > const field_components = detail::getFieldComponentLabels(var_in_theta_mode);
        for( std::string const& vector_field : vector_fields ) {
            for( std::string const& component : field_components ) {
                if( vector_field == varname_1st &&
                    component == varname_2nd )
                {
                    field_name = varname_1st + varname.substr(2); // Strip component
                    comp_name = varname_2nd;
                }
            }
        }
    }

    if ( 0 == meshLevel ) {
        return;
    }

    field_name += std::string("_lvl").append(std::to_string(meshLevel));
}

/** Find fieldName in varname and get the theta mode index, if varname = fieldName_mode_realOrImag
 *
 * @param[in] varname name of the field variable being parsed
 * @returns varname fieldName if varname = fieldName_mode_realOrImag, otherwise, varname and
 * if varname = fieldName_modeNumber_realOrImag, returns 2 * mode - 1 + (realOrImag == 'imag'), otherwise, -1
 *
 * Examples :
 * rho -> rho, -1
 * rho_0_real -> rho, 1
 * Er_1_real -> Er, 2
 * rho_species_12_1_imag -> rho_species_12, 3
 */
std::tuple<std::string, int>
GetFieldNameModeInt (const std::string& varname)
{
    // mode_index = -1 if varname isn't of form fieldName_mode_realOrImag
    // mode_index = 2 * mode - 1 + (realOrImag == 'imag')
    // in either case, there is a -1 in mode_index
    int mode_index = -1;

    const std::regex e_real_imag("(.*)_([0-9]*)_(real|imag)");
    std::smatch sm;
    std::regex_match(varname, sm, e_real_imag, std::regex_constants::match_default);

    if (sm.size() != 4 ) {
        return std::make_tuple(varname, mode_index);
    } else {
        // sm = [varname, field_name, mode, real_imag]
        const int mode = std::stoi(sm[2]);
        if (mode == 0) {
            mode_index = 0;
        } else {
            if (sm[3] == "imag") {
                mode_index += 1;
            }
            mode_index += 2 * mode;
        }

        return std::make_tuple(std::string(sm[1]), mode_index);
    }
}

/** Write Field with all mesh levels
 *
 */
void
WarpXOpenPMDPlot::WriteOpenPMDFieldsAll ( //const std::string& filename,
                      const std::vector<std::string>& varnames,
                      const amrex::Vector<amrex::MultiFab>& mf,
                      amrex::Vector<amrex::Geometry>& geom,
                      int output_levels,
                      const int iteration,
                      const double time,
                      bool isBTD,
                      const amrex::Geometry& full_BTD_snapshot ) const
{
    //This is AMReX's tiny profiler. Possibly will apply it later
    WARPX_PROFILE("WarpXOpenPMDPlot::WriteOpenPMDFields()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_Series != nullptr, "openPMD series must be initialized");

    // is this either a regular write (true) or the first write in a
    // backtransformed diagnostic (BTD):
    bool const first_write_to_iteration = ! m_Series->iterations.contains( iteration );

    // meta data
    openPMD::Iteration series_iteration = GetIteration(m_CurrentStep, isBTD);

    // collective open
    series_iteration.open();

    auto meshes = series_iteration.meshes;
    if (first_write_to_iteration) {
        // lets see whether full_geom varies from geom[0]   xgeom[1]
        series_iteration.setTime( time );
    }

    // If there are no fields to be written, interrupt the function here
    if ( varnames.empty() ) { return; }

    // loop over levels up to output_levels
    //   note: this is usually the finestLevel, not the maxLevel
    for (int lev=0; lev < output_levels; lev++) {
        amrex::Geometry full_geom = geom[lev];
        if( isBTD ) {
            full_geom = full_BTD_snapshot;
        }

        // setup is called once. So it uses property "period" from first
        // geometry for <all> field levels.
        if ( (0 == lev) && first_write_to_iteration ) {
            SetupFields(meshes, full_geom);
        }

        amrex::Box const & global_box = full_geom.Domain();

        int const ncomp = mf[lev].nComp();
        for ( int icomp=0; icomp<ncomp; icomp++ ) {
            std::string const & varname = varnames[icomp];

            auto [varname_no_mode, mode_index] = GetFieldNameModeInt(varname);
            const bool var_in_theta_mode = mode_index != -1; // thetaMode or reconstructed Cartesian 2D slice
            std::string field_name = varname_no_mode;
            std::string comp_name = openPMD::MeshRecordComponent::SCALAR;
            // assume fields are scalar unless they match the following match of known vector fields
            GetMeshCompNames( lev, varname_no_mode, field_name, comp_name, var_in_theta_mode );
            if ( first_write_to_iteration )
            {
                if (comp_name == openPMD::MeshRecordComponent::SCALAR) {
                    if ( ! meshes.contains(field_name) ) {
                        auto mesh = meshes[field_name];
                        SetupMeshComp(  mesh,
                                        full_geom,
                                        comp_name,
                                        field_name,
                                        mf[lev],
                                        var_in_theta_mode );
                    }
                } else {
                    auto mesh = meshes[field_name];
                    if ( ! mesh.contains(comp_name) ) {
                        SetupMeshComp(  mesh,
                                        full_geom,
                                        comp_name,
                                        field_name,
                                        mf[lev],
                                        var_in_theta_mode );
                    }
                }
            }
        } // icomp setup loop

        for ( int icomp=0; icomp<ncomp; icomp++ ) {
            std::string const & varname = varnames[icomp];

            auto [varname_no_mode, mode_index] = GetFieldNameModeInt(varname);
            [[maybe_unused]] const bool var_in_theta_mode = mode_index != -1;

            std::string field_name(varname_no_mode);
            std::string comp_name = openPMD::MeshRecordComponent::SCALAR;
            // assume fields are scalar unless they match the following match of known vector fields
            GetMeshCompNames( lev, varname_no_mode, field_name, comp_name, var_in_theta_mode );

            auto mesh = meshes[field_name];
            auto mesh_comp = mesh[comp_name];

            // Loop through the multifab, and store each box as a chunk,
            // in the openPMD file.
            for( amrex::MFIter mfi(mf[lev]); mfi.isValid(); ++mfi )
            {
                amrex::FArrayBox const& fab = mf[lev][mfi];
                amrex::Box const& local_box = fab.box();

                // Determine the offset and size of this chunk
                amrex::IntVect const box_offset = local_box.smallEnd() - global_box.smallEnd();
                auto chunk_offset = getReversedVec( box_offset );
                auto chunk_size = getReversedVec( local_box.size() );

                if (var_in_theta_mode) {
                    chunk_offset.emplace(chunk_offset.begin(), mode_index);
                    chunk_size.emplace(chunk_size.begin(), 1);
                }

                // we avoid relying on managed memory by copying explicitly to host
                //   remove the copies and "streamSynchronize" if you like to pass
                //   GPU pointers to the I/O library
#ifdef AMREX_USE_GPU
                if (fab.arena()->isManaged() || fab.arena()->isDevice()) {
                    amrex::BaseFab<amrex::Real> foo(local_box, 1, amrex::The_Pinned_Arena());
                    std::shared_ptr<amrex::Real> data_pinned(foo.release());
                    amrex::Gpu::dtoh_memcpy_async(data_pinned.get(), fab.dataPtr(icomp), local_box.numPts()*sizeof(amrex::Real));
                    // intentionally delayed until before we .flush(): amrex::Gpu::streamSynchronize();
                    mesh_comp.storeChunk(data_pinned, chunk_offset, chunk_size);
                } else
#endif
                {
                    amrex::Real const *local_data = fab.dataPtr(icomp);
                    mesh_comp.storeChunkRaw(
                        local_data, chunk_offset, chunk_size);
                }
            }
        } // icomp store loop

#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif
        // Flush data to disk after looping over all components
        m_Series->flush();
    } // levels loop (i)
}
#endif // WARPX_USE_OPENPMD



//
//
//
WarpXParticleCounter::WarpXParticleCounter (ParticleContainer* pc):
    m_MPIRank{amrex::ParallelDescriptor::MyProc()},
    m_MPISize{amrex::ParallelDescriptor::NProcs()}
{
    m_ParticleCounterByLevel.resize(pc->finestLevel()+1);
    m_ParticleOffsetAtRank.resize(pc->finestLevel()+1);
    m_ParticleSizeAtRank.resize(pc->finestLevel()+1);

    for (auto currentLevel = 0; currentLevel <= pc->finestLevel(); currentLevel++)
    {
        long numParticles = 0; // numParticles in this processor

        for (ParticleIter pti(*pc, currentLevel); pti.isValid(); ++pti) {
            auto numParticleOnTile = pti.numParticles();
            numParticles += numParticleOnTile;
        }

        unsigned long long offset=0; // offset of this level
        unsigned long long sum=0; // numParticles in this level (sum from all processors)

        GetParticleOffsetOfProcessor(numParticles, offset,  sum);

        m_ParticleCounterByLevel[currentLevel] = sum;
        m_ParticleOffsetAtRank[currentLevel] = offset;
        m_ParticleSizeAtRank[currentLevel] = numParticles;

        // adjust offset, it should be numbered after particles from previous levels
        for (auto lv=0; lv<currentLevel; lv++) {
            m_ParticleOffsetAtRank[currentLevel] += m_ParticleCounterByLevel[lv];
        }

        m_Total += sum;
    }
}


// get the offset in the overall particle id collection
//
// note: this is a MPI-collective operation
//
// input: num of particles  of from each   processor
//
// output:
//     offset within <all> the particles in the comm
//     sum of all particles in the comm
//
void
WarpXParticleCounter::GetParticleOffsetOfProcessor (
    const long& numParticles,
    unsigned long long& offset,
    unsigned long long& sum
) const
{
    offset = 0;
#if defined(AMREX_USE_MPI)
    std::vector<long> result(m_MPISize, 0);
    amrex::ParallelGather::Gather (numParticles, result.data(), -1, amrex::ParallelDescriptor::Communicator());

    sum = 0;
    auto const num_results = static_cast<int>(result.size());
    for (int i=0; i<num_results; i++) {
        sum += result[i];
        if (i<m_MPIRank) {
            offset += result[i];
        }
    }
#else
    sum = numParticles;
#endif
}
