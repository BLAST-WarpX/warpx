/* Copyright 2023 Luca Fedeli
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ExternalField.H"

#include "Utils/TextMsg.H"
#include "Utils/Parser/ParserUtils.H"

#include <ablastr/warn_manager/WarnManager.H>

#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
#   include <openPMD/openPMD.hpp>
#endif

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace
{

    enum class EMFieldType{E, B};

    template <EMFieldType T>
    ExternalFieldType string_to_external_field_type(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);

        if constexpr (T == EMFieldType::E){
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s != "parse_b_ext_grid_function",
                "parse_B_ext_grid_function can be used only for B_ext_grid_init_style");
        }
        else{
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s != "parse_e_ext_grid_function",
                "parse_E_ext_grid_function can be used only for E_ext_grid_init_style");
        }

        if ( s.empty() || s == "default"){
            return ExternalFieldType::default_zero;
        }
        else if ( s == "constant"){
            return ExternalFieldType::constant;
        }
        else if ( s == "parse_b_ext_grid_function" || s == "parse_e_ext_grid_function"){
            return ExternalFieldType::parse_ext_grid_function;
        }
        else if ( s == "read_from_file"){
            return ExternalFieldType::read_from_file;
        }
        else if ( s == "load_from_python"){
            return ExternalFieldType::load_from_python;
        }
        else{
            WARPX_ABORT_WITH_MESSAGE(
                "'" + s + "' is an unknown external field type!");
        }

        return ExternalFieldType::default_zero;
    }
}

ExternalFieldParams::ExternalFieldParams(const amrex::ParmParse& pp_warpx)
{
    // default values of E_external_grid and B_external_grid
    // are used to set the E and B field when "constant" or
    // "parser" is not explicitly used in the input.
    std::string B_ext_grid_s;
    pp_warpx.query("B_ext_grid_init_style", B_ext_grid_s);
    B_ext_grid_type = string_to_external_field_type<EMFieldType::B>(B_ext_grid_s);

    std::string E_ext_grid_s;
    pp_warpx.query("E_ext_grid_init_style", E_ext_grid_s);
    E_ext_grid_type = string_to_external_field_type<EMFieldType::E>(E_ext_grid_s);

    //
    //  Constant external field
    //

    // if the input string is "constant", the values for the
    // external grid must be provided in the input.
    auto v_B = std::vector<amrex::Real>(3);
    if (B_ext_grid_type == ExternalFieldType::constant) {
        utils::parser::getArrWithParser(pp_warpx, "B_external_grid", v_B);
    }
    std::copy(v_B.begin(), v_B.end(), B_external_grid.begin());

    // if the input string is "constant", the values for the
    // external grid must be provided in the input.
    auto v_E = std::vector<amrex::Real>(3);
    if (E_ext_grid_type == ExternalFieldType::constant) {
        utils::parser::getArrWithParser(pp_warpx, "E_external_grid", v_E);
    }
    std::copy(v_E.begin(), v_E.end(), E_external_grid.begin());
    //___________________________________________________________________________


    //
    //  External E field with parser
    //

    // if the input string for the B-field is "parse_b_ext_grid_function",
    // then the analytical expression or function must be
    // provided in the input file.
    if (B_ext_grid_type == ExternalFieldType::parse_ext_grid_function) {

        //! Strings storing parser function to initialize the components of the magnetic field on the grid
        std::string str_Bx_ext_grid_function;
        std::string str_By_ext_grid_function;
        std::string str_Bz_ext_grid_function;

#if defined(WARPX_DIM_RZ)
        std::stringstream warnMsg;
        warnMsg << "Parser for external B (r and theta) fields does not work with cylindrical and spherical\n"
            << "The initial Br and Bt fields are currently hardcoded to 0.\n"
            << "The initial Bz field should only be a function of z.\n";
        ablastr::warn_manager::WMRecordWarning(
          "Inputs", warnMsg.str(), ablastr::warn_manager::WarnPriority::high);
        str_Bx_ext_grid_function = "0";
        str_By_ext_grid_function = "0";
#else
        utils::parser::Store_parserString(pp_warpx, "Bx_external_grid_function(x,y,z)",
          str_Bx_ext_grid_function);
        utils::parser::Store_parserString(pp_warpx, "By_external_grid_function(x,y,z)",
          str_By_ext_grid_function);
#endif
        utils::parser::Store_parserString(pp_warpx, "Bz_external_grid_function(x,y,z)",
            str_Bz_ext_grid_function);

        Bxfield_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_Bx_ext_grid_function,{"x","y","z","t"}));
        Byfield_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_By_ext_grid_function,{"x","y","z","t"}));
        Bzfield_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_Bz_ext_grid_function,{"x","y","z","t"}));
    }
    //___________________________________________________________________________


    //
    //  External B field with parser
    //

    // if the input string for the E-field is "parse_e_ext_grid_function",
    // then the analytical expression or function must be
    // provided in the input file.
    if (E_ext_grid_type == ExternalFieldType::parse_ext_grid_function) {

#ifdef WARPX_DIM_RZ
        WARPX_ABORT_WITH_MESSAGE(
            "E parser for external fields does not work with RZ -- TO DO");
#endif

        //! Strings storing parser function to initialize the components of the electric field on the grid
        std::string str_Ex_ext_grid_function;
        std::string str_Ey_ext_grid_function;
        std::string str_Ez_ext_grid_function;

        utils::parser::Store_parserString(pp_warpx, "Ex_external_grid_function(x,y,z)",
            str_Ex_ext_grid_function);
        utils::parser::Store_parserString(pp_warpx, "Ey_external_grid_function(x,y,z)",
           str_Ey_ext_grid_function);
        utils::parser::Store_parserString(pp_warpx, "Ez_external_grid_function(x,y,z)",
           str_Ez_ext_grid_function);

        Exfield_parser = std::make_unique<amrex::Parser>(
           utils::parser::makeParser(str_Ex_ext_grid_function,{"x","y","z","t"}));
        Eyfield_parser = std::make_unique<amrex::Parser>(
           utils::parser::makeParser(str_Ey_ext_grid_function,{"x","y","z","t"}));
        Ezfield_parser = std::make_unique<amrex::Parser>(
           utils::parser::makeParser(str_Ez_ext_grid_function,{"x","y","z","t"}));
    }
    //___________________________________________________________________________


    //
    //  External fields from file
    //
    if (E_ext_grid_type == ExternalFieldType::read_from_file ||
        B_ext_grid_type == ExternalFieldType::read_from_file){
            const std::string read_fields_from_path="./";
            pp_warpx.query("read_fields_from_path", external_fields_path);
    }
    //___________________________________________________________________________
}

ExternalFieldReader::ExternalFieldReader (std::string const& read_fields_from_path,
                                          std::string const& F_name,
                                          std::string const& F_component)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)

    auto series = openPMD::Series(read_fields_from_path, openPMD::Access::READ_ONLY);
    auto iseries = series.iterations.begin()->second;
    auto F = iseries.meshes[F_name];

#if (AMREX_SPACEDIM > 1)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(F.getAttribute("dataOrder").get<std::string>() == "C",
                                     "Reading from files with non-C dataOrder is not implemented");
#endif

    auto axisLabels = F.getAttribute("axisLabels").get<std::vector<std::string>>();
    auto fileGeom = F.getAttribute("geometry").get<std::string>();

#if defined(WARPX_DIM_3D)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "cartesian", "3D can only read from files with cartesian geometry");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels.at(0) == "x" && axisLabels.at(1) == "y" && axisLabels.at(2) == "z",
                                     "3D expects axisLabels {x, y, z}");
#elif defined(WARPX_DIM_XZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "cartesian", "XZ can only read from files with cartesian geometry");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels.at(0) == "x" && axisLabels.at(1) == "z",
                                     "XZ expects axisLabels {x, z}");
#elif defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "thetaMode", "RZ can only read from files with 'thetaMode'  geometry");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels.at(0) == "r" && axisLabels.at(1) == "z",
                                     "RZ expects axisLabels {r, z}");
#elif defined(WARPX_DIM_1D_Z)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "cartesian", "1D3V can only read from files with cartesian geometry");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels.at(0) == "z", "1D3V expects axisLabel {z}");
#endif

    const auto d = F.gridSpacing<long double>();
    AMREX_D_TERM(m_external_field_view.dx[0] = amrex::Real(d.at(0));,
                 m_external_field_view.dx[1] = amrex::Real(d.at(1));,
                 m_external_field_view.dx[2] = amrex::Real(d.at(2)));

    const auto offset = F.gridGlobalOffset();
    AMREX_D_TERM(m_external_field_view.offset[0] = amrex::Real(offset.at(0));,
                 m_external_field_view.offset[1] = amrex::Real(offset.at(1));,
                 m_external_field_view.offset[2] = amrex::Real(offset.at(2)));

    // Load the first component if F_component is empty
    auto FC = F_component.empty() ? F.begin()->second : F[F_component];
    const auto extent = FC.getExtent();

    // Determine the chunk data that will be loaded.
    // Now, the full range of data is loaded.
    // Loading chunk data can speed up the process.
    // Thus, `chunk_offset` and `chunk_extent` should be modified accordingly in another PR.
    const openPMD::Offset chunk_offset(extent.size(), 0);
    const openPMD::Extent chunk_extent = extent;

    m_FC_data_cpu = FC.loadChunk<double>(chunk_offset,chunk_extent);
    series.flush();

#if defined(AMREX_USE_GPU)
    auto *FC_data_host = m_FC_data_cpu.get();
    auto const total_extent = std::accumulate(chunk_extent.begin(), chunk_extent.end(),
                                              1, std::multiplies<std::size_t>());
    m_FC_data_gpu.resize(total_extent);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, FC_data_host, FC_data_host + total_extent, m_FC_data_gpu.begin());
    amrex::Gpu::streamSynchronize();
    m_FC_data_cpu.reset();
    auto *FC_data = m_FC_data_gpu.data();
#else
    auto *FC_data = m_FC_data_cpu.get();
#endif

#if defined(WARPX_DIM_RZ)
    // extent[0] is for theta
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(extent[0] == 1,
                                     "External field reading is not implemented for more than one RZ mode (see #3829)");
    const auto extent0 = static_cast<int>(extent.at(1));
    const auto extent1 = static_cast<int>(extent.at(2));
#else
    AMREX_D_TERM(const auto extent0 = static_cast<int>(extent.at(0));,
                 const auto extent1 = static_cast<int>(extent.at(1));,
                 const auto extent2 = static_cast<int>(extent.at(2)));
#endif

    m_external_field_view.table = decltype(m_external_field_view.table)
#if (AMREX_SPACEDIM == 1)
        (FC_data, 0, extent0);
#else
        (FC_data, {AMREX_D_DECL(0,0,0)}, {AMREX_D_DECL(extent0, extent1, extent2)});
#endif

#else
    amrex::ignore_unused(read_fields_from_path, F_name, F_component);
    WARPX_ABORT_WITH_MESSAGE("ExternalFieldReader requires openPMD and it is not supported for 1D RCYLINDER and RSPHERE");
#endif
}
