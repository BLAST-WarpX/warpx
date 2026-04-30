/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "VelocityProperties.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <cmath>

VelocityProperties::VelocityProperties (const amrex::ParmParse& pp, std::string const& source_name,
                                        amrex::Geometry const& geom)
    : m_geom(geom)
{

    std::string mom_dist_s;
    utils::parser::query(pp, source_name, "momentum_distribution_type", mom_dist_s);
    if (mom_dist_s != "maxwellian") {
        // Set defaults
        std::string vel_dist_s = "constant";
        std::string vel_dir_s = "x";

        utils::parser::query(pp, source_name, "bulk_vel_dir", vel_dir_s);

        if(vel_dir_s.empty()){
            WARPX_ABORT_WITH_MESSAGE("'<s_name>.bulk_vel_dir input ' can't be empty.");
        }

        m_sign_dir = (vel_dir_s[0] == '-') ? -1 : 1;

        const auto dir = std::tolower(vel_dir_s.back());

        if (dir == 'x'){
            m_dir = 0;
        }
        else if (dir == 'y'){
            m_dir = 1;
        }
        else if (dir == 'z'){
            m_dir = 2;
        }
        else{
            WARPX_ABORT_WITH_MESSAGE(
                "Cannot interpret <s_name>.bulk_vel_dir input '" + vel_dir_s +
                "'. Please enter +/- x, y, or z with no whitespace between the sign and"+
                " other character.");
        }

        utils::parser::query(pp, source_name, "beta_distribution_type", vel_dist_s);
        if (vel_dist_s == "constant") {
            utils::parser::queryWithParser(pp, source_name, "beta", m_velocity);
            m_type = VelConstantValue;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_velocity > -1 && m_velocity < 1,
                "Magnitude of velocity beta = " + std::to_string(m_velocity) +
                " is greater than or equal to 1"
            );
        }
        else if (vel_dist_s == "parser") {
            std::string str_beta_function;
            utils::parser::Store_parserString(pp, source_name, "beta_function(x,y,z)", str_beta_function);
            m_ptr_velocity_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_beta_function,{"x","y","z"}));
            m_type = VelParserFunction;
        }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "Velocity distribution type '" + vel_dist_s + "' not recognized.");
        }
    } else {
        std::string u_mean_dist_s = "constant";
        utils::parser::query(pp, source_name, "maxwellian_u_mean_distribution_type", u_mean_dist_s);
        if (u_mean_dist_s == "constant") {
            utils::parser::queryWithParser(pp, source_name, "ux_mean", m_ux_mean);
            utils::parser::queryWithParser(pp, source_name, "uy_mean", m_uy_mean);
            utils::parser::queryWithParser(pp, source_name, "uz_mean", m_uz_mean);

            amrex::Real velocity_magnitude = std::sqrt(m_ux_mean*m_ux_mean +
                                                  m_uy_mean*m_uy_mean +
                                                  m_uz_mean*m_uz_mean);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                velocity_magnitude < 1.0,
                "Magnitude of mean velocity abs(u_mean) = " + std::to_string(velocity_magnitude) +
                " is greater than or equal to 1"
            );
            m_type = VelConstantVector;
        } else if (u_mean_dist_s == "parser") {
            std::string str_ux_mean_function, str_uy_mean_function, str_uz_mean_function;
            utils::parser::Store_parserString(pp, source_name, "ux_mean_function(x,y,z)", str_ux_mean_function);
            utils::parser::Store_parserString(pp, source_name, "uy_mean_function(x,y,z)", str_uy_mean_function);
            utils::parser::Store_parserString(pp, source_name, "uz_mean_function(x,y,z)", str_uz_mean_function);
            m_ptr_ux_mean_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_ux_mean_function,{"x","y","z"}));
            m_ptr_uy_mean_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_uy_mean_function,{"x","y","z"}));
            m_ptr_uz_mean_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_uz_mean_function,{"x","y","z"}));
            m_type = VelParserFunctionVector;

        } else if (u_mean_dist_s == "read_from_file") {
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
            utils::parser::get(pp, source_name, "read_ux_mean_from_path", m_read_ux_mean_path);
            utils::parser::get(pp, source_name, "read_uy_mean_from_path", m_read_uy_mean_path);
            utils::parser::get(pp, source_name, "read_uz_mean_from_path", m_read_uz_mean_path);
            utils::parser::query(pp, source_name, "ux_mean_openpmd_mesh", m_ux_mean_openpmd_mesh);
            utils::parser::query(pp, source_name, "uy_mean_openpmd_mesh", m_uy_mean_openpmd_mesh);
            utils::parser::query(pp, source_name, "uz_mean_openpmd_mesh", m_uz_mean_openpmd_mesh);
            {
                std::string const key_with_src =
                    source_name.empty() ? std::string("read_u_mean_distributed")
                                        : source_name + ".read_u_mean_distributed";
                if (pp.contains(key_with_src.c_str())) {
                    pp.query(key_with_src.c_str(), m_read_u_mean_distributed);
                } else {
                    pp.query("read_u_mean_distributed", m_read_u_mean_distributed);
                }
            }
            m_type = VelFromFileVector;
#else
            WARPX_ABORT_WITH_MESSAGE(
                "maxwellian_u_mean_distribution_type = read_from_file requires "
                "WarpX built with openPMD support and is not supported in "
                "RCYLINDER/RSPHERE geometries.");
#endif
        }
        else {
            std::stringstream stringstream;
            std::string string;
            stringstream << "Maxwellian velocity mean distribution type '" << u_mean_dist_s << "' not recognized.";
            string = stringstream.str();
            WARPX_ABORT_WITH_MESSAGE(string);
        }
    }
}
