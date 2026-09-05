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

/** Construct TemperatureProperties from the passed particle source parameters.
 *  Parse the momentum distribution type and initialize the corresponding
 *  temperature parameters: thermal spread `ux_std`, `uy_std`, `uz_std`
 *  for `maxwellian` distribution, and `theta` for `maxwell_juttner`.
 */
TemperatureProperties::TemperatureProperties (const amrex::ParmParse& pp, std::string const& source_name,
                                              amrex::Geometry const& geom)
{
    amrex::ignore_unused(geom);

    std::string mom_dist_s;
    utils::parser::query(pp, source_name, "momentum_distribution_type", mom_dist_s);

    if (mom_dist_s == "maxwell_juttner") {
        // Set defaults
        amrex::Real theta = 0; // quiet GCC warning maybe-uninitialized
        std::string temp_dist_s = "constant";
        utils::parser::query(pp, source_name, "theta_distribution_type", temp_dist_s);

        if (temp_dist_s == "constant") {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                utils::parser::queryWithParser(pp, source_name, "theta", theta),
                "Temperature parameter theta not specified");

            // Do validation on theta value.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(theta >= 0,
                "Temperature parameter theta = " + std::to_string(theta) +
                " is less than zero, which is not allowed");

            m_type = TempConstantValue;
            m_temperature = theta;
        }
        else if (temp_dist_s == "parser") {
            std::string str_theta_function;
            utils::parser::Store_parserString(pp, source_name, "theta_function(x,y,z)", str_theta_function);
            m_ptr_temperature_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_theta_function,{"x","y","z"}));
            m_type = TempParserFunction;
        }
        else {
            std::stringstream ss;
            ss << "Temperature distribution type '" << temp_dist_s << "' not recognized.";
            WARPX_ABORT_WITH_MESSAGE(ss.str());
        }
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
            amrex::Real mass = 0.0;
            std::string physical_species_s;
            const bool species_is_specified = pp.query("species_type", physical_species_s);
            if (species_is_specified) {
                const auto physical_species_from_string = species::from_string(physical_species_s);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(physical_species_from_string,
                    physical_species_s + " does not exist!");
                mass = species::get_mass(physical_species_from_string.value());
            }
            utils::parser::queryWithParser(pp, "mass", mass);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mass > 0.0,
                "Need to specify species_type or mass > 0.0 for the Maxwellian temperature in eV initialization");

            std::string temperature_in_eV_dist_s = "constant";
            utils::parser::query(pp, source_name, "maxwellian_temperature_in_eV_distribution_type",
                                 temperature_in_eV_dist_s);

            if (temperature_in_eV_dist_s == "constant") {
                // Store scalar temperature_in_eV; convert to isotropic u_std at evaluation.
                amrex::Real temperature_in_eV = 0.0;
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    utils::parser::queryWithParser(pp, source_name, "temperature_in_eV", temperature_in_eV),
                    "Temperature parameter temperature_in_eV not specified");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(temperature_in_eV >= 0.0,
                    "temperature_in_eV = " + std::to_string(temperature_in_eV) +
                    " is less than zero, which is not allowed");
                m_temperature = temperature_in_eV;
                m_q_e_over_mc2 =
                    PhysConst::q_e / (mass * PhysConst::c * PhysConst::c);
                m_type = TempConstantValue;
            }
            else if (temperature_in_eV_dist_s == "parser") {
                // Parse temperature_in_eV(x,y,z); convert to isotropic u_std at evaluation.
                std::string str_temperature_in_eV_function;
                utils::parser::Store_parserString(
                    pp, source_name, "temperature_in_eV_function(x,y,z)",
                    str_temperature_in_eV_function);
                m_ptr_temperature_parser = std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_temperature_in_eV_function, {"x", "y", "z"}));
                m_q_e_over_mc2 =
                    PhysConst::q_e / (mass * PhysConst::c * PhysConst::c);
                m_type = TempParserFunction;
            }
            else if (temperature_in_eV_dist_s == "read_from_file") {
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
                if (WarpX::gamma_boost > 1.0) {
                    WARPX_ABORT_WITH_MESSAGE(
                        "maxwellian_temperature_in_eV_distribution_type = read_from_file is not "
                        "supported in boosted-frame simulations yet.");
                }
                utils::parser::get(pp, source_name, "read_temperature_in_eV_from_path",
                                   m_read_temperature_in_eV_path);
                std::string field_name = "temperature_in_eV";
                utils::parser::query(pp, source_name, "temperature_in_eV_mesh_name", field_name);
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const problo =
                    geom.ProbLoArray();
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx =
                    geom.CellSizeArray();
                amrex::Box const dombox = amrex::convert(geom.Domain(), amrex::IntVect(1));
                m_temperature_in_eV_reader = std::make_unique<ExternalFieldReader>(
                    m_read_temperature_in_eV_path, field_name, "", problo, dx, dombox, false);
                amrex::BoxArray const grids;
                amrex::DistributionMapping const dmap;
                m_temperature_in_eV_reader->prepare(grids, dmap, amrex::IntVect(0));
                m_q_e_over_mc2 = PhysConst::q_e / (mass * PhysConst::c * PhysConst::c);
                m_type = TempFromFileValue;
#else
                WARPX_ABORT_WITH_MESSAGE(
                    "maxwellian_temperature_in_eV_distribution_type = read_from_file requires "
                    "WarpX built with openPMD support and is not supported in "
                    "RZ/RCYLINDER/RSPHERE geometries.");
#endif
            }
            else {
                std::stringstream ss;
                ss << "Maxwellian temperature distribution type '" << temperature_in_eV_dist_s
                   << "' not recognized.";
                WARPX_ABORT_WITH_MESSAGE(ss.str());
            }
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
