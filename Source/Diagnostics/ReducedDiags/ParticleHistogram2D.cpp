/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Juliette Pech, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleHistogram2D.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Diagnostics/OpenPMDHelpFunction.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_TableData.H>
#include <AMReX_FabConv.H>

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <ostream>
#include <vector>
#include <string>

#ifdef WARPX_USE_OPENPMD
namespace io = openPMD;
#endif

using namespace amrex;


// constructor
ParticleHistogram2D::ParticleHistogram2D (const std::string& rd_name)
        : ReducedDiags{rd_name}
{
    ParmParse pp_rd_name(rd_name);

    pp_rd_name.query("openpmd_backend", m_openpmd_backend);
    pp_rd_name.query("file_min_digits", m_file_min_digits);
    // pick first available backend if default is chosen
    if( m_openpmd_backend == "default" ) {
        m_openpmd_backend = WarpXOpenPMDFileType();
    }
    pp_rd_name.add("openpmd_backend", m_openpmd_backend);

    // read species
    std::string selected_species_name;
    pp_rd_name.get("species",selected_species_name);

    // read bin parameters
    utils::parser::getWithParser(pp_rd_name, "bin_number_abs",m_bin_num_abs);
    utils::parser::getWithParser(pp_rd_name, "bin_max_abs",   m_bin_max_abs);
    utils::parser::getWithParser(pp_rd_name, "bin_min_abs",   m_bin_min_abs);
    m_bin_size_abs = (m_bin_max_abs - m_bin_min_abs) / m_bin_num_abs;
    utils::parser::getWithParser(pp_rd_name, "bin_number_ord",m_bin_num_ord);
    utils::parser::getWithParser(pp_rd_name, "bin_max_ord",   m_bin_max_ord);
    utils::parser::getWithParser(pp_rd_name, "bin_min_ord",   m_bin_min_ord);
    m_bin_size_ord = (m_bin_max_ord - m_bin_min_ord) / m_bin_num_ord;

    // read histogram function
    utils::parser::Store_parserString(pp_rd_name,"histogram_function_abs(t,x,y,z,ux,uy,uz,w)",
                                      function_string_abs);
    m_parser_abs = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(function_string_abs,{"t","x","y","z","ux","uy","uz","w"}));
    utils::parser::Store_parserString(pp_rd_name,"histogram_function_ord(t,x,y,z,ux,uy,uz,w)",
                                      function_string_ord);
    m_parser_ord = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(function_string_ord,{"t","x","y","z","ux","uy","uz","w"}));

    // get MultiParticleContainer class object
    const auto & mypc = WarpX::GetInstance().GetPartContainer();
    // get species names (std::vector<std::string>)
    auto const species_names = mypc.GetSpeciesNames();
    // select species
    for ( int i = 0; i < mypc.nSpecies(); ++i )
    {
        if ( selected_species_name == species_names[i] ){
            m_selected_species_id = i;
        }
    }
    // if m_selected_species_id is not modified
    if ( m_selected_species_id == -1 ){
        WARPX_ABORT_WITH_MESSAGE("Unknown species for ParticleHistogram2D reduced diagnostic.");
    }

    // Read optional filter
    std::string buf;
    m_do_parser_filter = pp_rd_name.query("filter_function(t,x,y,z,ux,uy,uz,w)", buf);
    if (m_do_parser_filter) {
        utils::parser::Store_parserString(
                pp_rd_name,"filter_function(t,x,y,z,ux,uy,uz,w)", filter_string);
        m_parser_filter = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(filter_string,{"t","x","y","z","ux","uy","uz","w"}));
    }

    // Read optional value function
    m_do_parser_value = pp_rd_name.query("value_function(t,x,y,z,ux,uy,uz,w)", buf);
    if (m_do_parser_value) {
        utils::parser::Store_parserString(
                pp_rd_name,"value_function(t,x,y,z,ux,uy,uz,w)", value_string);
        m_parser_value = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(value_string, {"t", "x", "y", "z", "ux", "uy", "uz", "w"}));
    }
}
// end constructor

// function that computes the histogram
void ParticleHistogram2D::ComputeDiags (int step)
{
    // Judge if the diags should be done at this step
    if (!m_intervals.contains(step+1)) { return; }

    // resize data array
    Array<int,2> tlo{0,0}; // lower bounds
    Array<int,2> thi{m_bin_num_abs-1, m_bin_num_ord-1}; // inclusive upper bounds
    amrex::TableData<amrex::Real,2> d_data_2D(tlo, thi);
    m_h_data_2D.resize(tlo, thi, The_Pinned_Arena());
    auto const& h_table_data = m_h_data_2D.table();

    // Initialize data on the host
    for (int i = tlo[0]; i <= thi[0]; ++i) {
        for (int j = tlo[1]; j <= thi[1]; ++j) {
            h_table_data(i,j) = 0.0_rt;
        }
    }

    d_data_2D.copy(m_h_data_2D);
    auto d_table = d_data_2D.table();

    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // get time at level 0
    auto const t = warpx.gett_new(0);

    // get MultiParticleContainer class object
    const auto & mypc = warpx.GetPartContainer();

    // get WarpXParticleContainer class object
    auto & myspc = mypc.GetParticleContainer(m_selected_species_id);

    // get parser
    auto fun_partparser_abs =
            utils::parser::compileParser<m_nvars>(m_parser_abs.get());
    auto fun_partparser_ord =
            utils::parser::compileParser<m_nvars>(m_parser_ord.get());

    // get filter parser
    auto fun_filterparser =
            utils::parser::compileParser<m_nvars>(m_parser_filter.get());

    // get value parser
    auto fun_valueparser =
            utils::parser::compileParser<m_nvars>(m_parser_value.get());

    // declare local variables
    auto const num_bins_abs = m_bin_num_abs;
    auto const num_bins_ord = m_bin_num_ord;
    Real const bin_min_abs  = m_bin_min_abs;
    Real const bin_size_abs = m_bin_size_abs;
    Real const bin_min_ord  = m_bin_min_ord;
    Real const bin_size_ord = m_bin_size_ord;

    bool const do_parser_filter = m_do_parser_filter;

    int const nlevs = std::max(0, myspc.finestLevel()+1);
    for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        {
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
            {
                auto const GetPosition = GetParticlePosition<PIdx>(pti);

                auto & attribs = pti.GetAttribs();
                ParticleReal* const AMREX_RESTRICT d_w = attribs[PIdx::w].dataPtr();
                ParticleReal* const AMREX_RESTRICT d_ux = attribs[PIdx::ux].dataPtr();
                ParticleReal* const AMREX_RESTRICT d_uy = attribs[PIdx::uy].dataPtr();
                ParticleReal* const AMREX_RESTRICT d_uz = attribs[PIdx::uz].dataPtr();

                long const np = pti.numParticles();

                //Flag particles that need to be copied if they cross the z_slice
                amrex::ParallelFor(np,
                                   [=] AMREX_GPU_DEVICE(int i)
                                   {
                                       amrex::ParticleReal x, y, z;
                                       GetPosition(i, x, y, z);
                                       auto const w  = (amrex::Real)d_w[i];
                                       auto const ux = d_ux[i] / PhysConst::c;
                                       auto const uy = d_uy[i] / PhysConst::c;
                                       auto const uz = d_uz[i] / PhysConst::c;

                                       // don't count a particle if it is filtered out
                                       if (do_parser_filter) {
                                           if(!static_cast<bool>(fun_filterparser(t, x, y, z, ux, uy, uz, w))) {
                                               return;
                                           }
                                       }

                                       // continue function if particle is not filtered out
                                       auto const f_abs = fun_partparser_abs(t, x, y, z, ux, uy, uz, w);
                                       auto const f_ord = fun_partparser_ord(t, x, y, z, ux, uy, uz, w);
                                       auto const weight = fun_valueparser(t, x, y, z, ux, uy, uz, w);

                                       // determine particle bin
                                       int const bin_abs = int(Math::floor((f_abs-bin_min_abs)/bin_size_abs));
                                       if ( bin_abs<0 || bin_abs>=num_bins_abs ) { return; } // discard if out-of-range

                                       int const bin_ord = int(Math::floor((f_ord-bin_min_ord)/bin_size_ord));
                                       if ( bin_ord<0 || bin_ord>=num_bins_ord ) { return; } // discard if out-of-range

                                       amrex::Real &data = d_table(bin_abs, bin_ord);
                                       amrex::HostDevice::Atomic::Add(&data, weight);
                                   });
            }
        }
    }

    // Copy data from GPU memory
    m_h_data_2D.copy(d_data_2D);

    // reduced sum over mpi ranks
    const int size = static_cast<int> (d_data_2D.size());
    ParallelDescriptor::ReduceRealSum
            (h_table_data.p, size, ParallelDescriptor::IOProcessorNumber());

    // Return for all that are not IO processor
    if ( !ParallelDescriptor::IOProcessor() ) { return; }
}
// end void ParticleHistogram2D::ComputeDiags

void ParticleHistogram2D::WriteToFile (int step) const
{
#ifdef WARPX_USE_OPENPMD
    // only IO processor writes
    if ( !ParallelDescriptor::IOProcessor() ) { return; }

    // TODO: support different filename templates
    std::string filename = "openpmd";
    // TODO: support also group-based encoding
    const std::string fileSuffix = std::string("_%0") + std::to_string(m_file_min_digits) + std::string("T");
    filename = filename.append(fileSuffix).append(".").append(m_openpmd_backend);

    // transform paths for Windows
    #ifdef _WIN32
        const std::string filepath = openPMD::auxiliary::replace_all(
            m_path + m_rd_name + "/" + filename, "/", "\\");
    #else
        const std::string filepath = m_path + m_rd_name + "/" + filename;
    #endif

    // Create the OpenPMD series
    auto series = io::Series(
            filepath,
            io::Access::CREATE);
    auto i = series.iterations[step + 1];
    // record
    auto f_mesh = i.meshes["data"];
    f_mesh.setAttribute("function_abscissa", function_string_abs);
    f_mesh.setAttribute("function_ordinate", function_string_ord);
    f_mesh.setAttribute("filter", filter_string);

    // record components
    auto data = f_mesh[io::RecordComponent::SCALAR];

    // meta data
    f_mesh.setAxisLabels({function_string_ord, function_string_abs}); // ordinate, abscissa
    std::vector< double > const& gridGlobalOffset = {m_bin_min_ord, m_bin_min_abs};
    f_mesh.setGridGlobalOffset(gridGlobalOffset);
    f_mesh.setGridSpacing<amrex::Real>({m_bin_size_ord, m_bin_size_abs});

    data.setPosition<amrex::Real>({0.5, 0.5});

    auto dataset = io::Dataset(
            io::determineDatatype<double>(),
            {static_cast<unsigned long>(m_bin_num_ord), static_cast<unsigned long>(m_bin_num_abs)});
    data.resetDataset(dataset);

    // UNIT DIMENSION IS NOT SET ON THE VALUES

    // Get time at level 0
    auto & warpx = WarpX::GetInstance();
    auto const time = warpx.gett_new(0);
    i.setTime(time);

    auto const& h_table_data = m_h_data_2D.table();
    data.storeChunkRaw(
            h_table_data.p,
            {0, 0},
            {static_cast<unsigned long>(m_bin_num_ord), static_cast<unsigned long>(m_bin_num_abs)});

    series.flush();
    i.close();
    series.close();
#else
    amrex::ignore_unused(step);
    WARPX_ABORT_WITH_MESSAGE("ParticleHistogram2D: Needs openPMD-api compiled into WarpX, but was not found!");
#endif
}
