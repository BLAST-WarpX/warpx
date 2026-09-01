/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "TemperatureProperties.H"

#include "ExternalField.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>

#include <cmath>
#include <sstream>
#include <string>

namespace {
/** Parse shared ``temperature_in_eV`` for ``maxwellian`` / ``maxwell_juttner`` (constant, parser, or file). */
void
parse_temperature_in_eV (
    amrex::ParmParse const& pp,
    std::string const& source_name,
    [[maybe_unused]] amrex::Geometry const& geom,
    std::string const& dist_type_param,
    std::string const& mom_dist_s,
    TemperatureProperties& temp)
{
    amrex::Real mass = 0.0;
    std::string physical_species_s;
    if (pp.query("species_type", physical_species_s)) {
        const auto physical_species_from_string = species::from_string(physical_species_s);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(physical_species_from_string,
            physical_species_s + " does not exist!");
        mass = species::get_mass(physical_species_from_string.value());
    }
    utils::parser::queryWithParser(pp, "mass", mass);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mass > 0.0,
        "Need to specify species_type or mass > 0.0 for the " + mom_dist_s +
        " temperature in eV initialization");

    std::string temperature_in_eV_dist_s = "constant";
    utils::parser::query(pp, source_name, dist_type_param.c_str(), temperature_in_eV_dist_s);

    const amrex::Real q_e_over_mc2 =
        PhysConst::q_e / (mass * PhysConst::c * PhysConst::c);

    if (temperature_in_eV_dist_s == "constant") {
        amrex::Real temperature_in_eV = 0.0;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            utils::parser::queryWithParser(pp, source_name, "temperature_in_eV",
                                           temperature_in_eV),
            "Temperature parameter temperature_in_eV not specified");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(temperature_in_eV >= 0.0,
            "temperature_in_eV = " + std::to_string(temperature_in_eV) +
            " is less than zero, which is not allowed");
        temp.m_temperature = temperature_in_eV;
        temp.m_q_e_over_mc2 = q_e_over_mc2;
        temp.m_type = TempConstantValue;
    }
    else if (temperature_in_eV_dist_s == "parser") {
        std::string str_temperature_in_eV_function;
        utils::parser::Store_parserString(
            pp, source_name, "temperature_in_eV_function(x,y,z)",
            str_temperature_in_eV_function);
        temp.m_ptr_temperature_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_temperature_in_eV_function, {"x", "y", "z"}));
        temp.m_q_e_over_mc2 = q_e_over_mc2;
        temp.m_type = TempParserFunction;
    }
    else if (temperature_in_eV_dist_s == "read_from_file") {
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
        if (WarpX::gamma_boost > 1.0) {
            WARPX_ABORT_WITH_MESSAGE(
                dist_type_param + " = read_from_file is not "
                "supported in boosted-frame simulations yet.");
        }
        utils::parser::get(pp, source_name, "read_temperature_in_eV_from_path",
                           temp.m_read_temperature_in_eV_path);
        std::string field_name = "temperature_in_eV";
        utils::parser::query(pp, source_name, "temperature_in_eV_mesh_name", field_name);
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const problo =
            geom.ProbLoArray();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx =
            geom.CellSizeArray();
        amrex::Box const dombox = amrex::convert(geom.Domain(), amrex::IntVect(1));
        temp.m_temperature_in_eV_reader = std::make_unique<ExternalFieldReader>(
            temp.m_read_temperature_in_eV_path, field_name, "", problo, dx, dombox, false);
        amrex::BoxArray const grids;
        amrex::DistributionMapping const dmap;
        temp.m_temperature_in_eV_reader->prepare(grids, dmap, amrex::IntVect(0));
        temp.m_q_e_over_mc2 = q_e_over_mc2;
        temp.m_type = TempFromFileValue;
#else
        WARPX_ABORT_WITH_MESSAGE(
            dist_type_param + " = read_from_file requires "
            "WarpX built with openPMD support and is not supported in "
            "RZ/RCYLINDER/RSPHERE geometries.");
#endif
    }
    else {
        std::stringstream ss;
        ss << mom_dist_s << " temperature distribution type '"
           << temperature_in_eV_dist_s << "' not recognized.";
        WARPX_ABORT_WITH_MESSAGE(ss.str());
    }
}

} // namespace

/** Construct TemperatureProperties from the passed particle source parameters.
 *  Parse the momentum distribution type and initialize the corresponding
 *  temperature parameters: thermal spread `ux_std`, `uy_std`, `uz_std` or `temperature_in_eV` for the Maxwellian distributions;
 *  and `temperature_in_eV` for `maxwell_juttner` distributions.
 */
TemperatureProperties::TemperatureProperties (const amrex::ParmParse& pp, std::string const& source_name,
                                              amrex::Geometry const& geom)
{
    std::string mom_dist_s;
    utils::parser::query(pp, source_name, "momentum_distribution_type", mom_dist_s);

    if (mom_dist_s == "maxwell_juttner") {
        parse_temperature_in_eV(
            pp, source_name, geom,
            "maxwell_juttner_temperature_in_eV_distribution_type",
            mom_dist_s,
            *this);
    }
    else if (mom_dist_s == "maxwellian") {
        const bool use_temperature_in_eV =
            pp.contains("maxwellian_temperature_in_eV_distribution_type") ||
            (!source_name.empty() &&
             pp.contains(source_name + ".maxwellian_temperature_in_eV_distribution_type"));
        const bool use_u_std =
            pp.contains("maxwellian_u_std_distribution_type") ||
            (!source_name.empty() &&
             pp.contains(source_name + ".maxwellian_u_std_distribution_type"));

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(use_temperature_in_eV && use_u_std),
            "Cannot specify both maxwellian_u_std_distribution_type and "
            "maxwellian_temperature_in_eV_distribution_type.");

        if (use_temperature_in_eV) {
            parse_temperature_in_eV(
                pp, source_name, geom,
                "maxwellian_temperature_in_eV_distribution_type",
                mom_dist_s,
                *this);
        }
        else {
            // ``maxwellian`` distribution uses ``u_std_*``
            std::string u_std_dist_s = "constant";
            utils::parser::query(pp, source_name, "maxwellian_u_std_distribution_type", u_std_dist_s);

            if (u_std_dist_s == "constant") {
                utils::parser::queryWithParser(pp, source_name, "ux_std", m_ux_std);
                utils::parser::queryWithParser(pp, source_name, "uy_std", m_uy_std);
                utils::parser::queryWithParser(pp, source_name, "uz_std", m_uz_std);
                m_type = TempConstantVector;
            }
            else if (u_std_dist_s == "parser") {
                std::string sx, sy, sz;
                utils::parser::Store_parserString(pp, source_name, "ux_std_function(x,y,z)", sx);
                utils::parser::Store_parserString(pp, source_name, "uy_std_function(x,y,z)", sy);
                utils::parser::Store_parserString(pp, source_name, "uz_std_function(x,y,z)", sz);
                m_ptr_ux_std_parser =
                    std::make_unique<amrex::Parser>(utils::parser::makeParser(sx, {"x", "y", "z"}));
                m_ptr_uy_std_parser =
                    std::make_unique<amrex::Parser>(utils::parser::makeParser(sy, {"x", "y", "z"}));
                m_ptr_uz_std_parser =
                    std::make_unique<amrex::Parser>(utils::parser::makeParser(sz, {"x", "y", "z"}));
                m_type = TempParserFunctionVector;
            }
            else if (u_std_dist_s == "read_from_file") {
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
                if (WarpX::gamma_boost > 1.0) {
                    WARPX_ABORT_WITH_MESSAGE(
                        "maxwellian_u_std_distribution_type = read_from_file is not "
                        "supported in boosted-frame simulations yet.");
                }
                utils::parser::get(pp, source_name, "read_u_std_from_path", m_read_u_std_path);
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const problo =
                    geom.ProbLoArray();
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx =
                    geom.CellSizeArray();
                amrex::Box const dombox = amrex::convert(geom.Domain(), amrex::IntVect(1));
                m_u_std_x_reader = std::make_unique<ExternalFieldReader>(
                    m_read_u_std_path, "u_std", "x", problo, dx, dombox, false);
                m_u_std_y_reader = std::make_unique<ExternalFieldReader>(
                    m_read_u_std_path, "u_std", "y", problo, dx, dombox, false);
                m_u_std_z_reader = std::make_unique<ExternalFieldReader>(
                    m_read_u_std_path, "u_std", "z", problo, dx, dombox, false);
                amrex::BoxArray const grids;
                amrex::DistributionMapping const dmap;
                m_u_std_x_reader->prepare(grids, dmap, amrex::IntVect(0));
                m_u_std_y_reader->prepare(grids, dmap, amrex::IntVect(0));
                m_u_std_z_reader->prepare(grids, dmap, amrex::IntVect(0));
                m_type = TempFromFileVector;
#else
                WARPX_ABORT_WITH_MESSAGE(
                    "maxwellian_u_std_distribution_type = read_from_file requires "
                    "WarpX built with openPMD support and is not supported in "
                    "RZ/RCYLINDER/RSPHERE geometries.");
#endif
            }
            else {
                std::stringstream ss;
                ss << "Maxwellian velocity standard deviation distribution type '" << u_std_dist_s
                   << "' not recognized.";
                WARPX_ABORT_WITH_MESSAGE(ss.str());
            }
        }
    }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "TemperatureProperties: unexpected momentum_distribution_type '" + mom_dist_s +
            "' (expected 'maxwellian' or 'maxwell_juttner').");
    }
}
