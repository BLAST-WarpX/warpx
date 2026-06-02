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

#include <AMReX_Math.H>

#include <cmath>
/**
* Construct VelocityProperties from the passed particle source parameters.
* Parse the momentum distribution type and initialize the corresponding
* velocity parameters: `ux_mean`, `uy_mean`, and `uz_mean` for
* `maxwellian` distributions, or `bulk_vel_dir` and `beta` for `maxwell_juttner`.
*/
VelocityProperties::VelocityProperties (const amrex::ParmParse& pp, std::string const& source_name)
{
    using namespace amrex;

    std::string mom_dist_s;
    utils::parser::query(pp, source_name, "momentum_distribution_type", mom_dist_s);
    if (mom_dist_s == "maxwell_juttner") {
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
    } else if (mom_dist_s == "maxwellian") {
        std::string u_mean_dist_s = "constant";
        utils::parser::query(pp, source_name, "maxwellian_u_mean_distribution_type", u_mean_dist_s);
        if (u_mean_dist_s == "constant") {
            utils::parser::queryWithParser(pp, source_name, "ux_mean", m_ux_mean);
            utils::parser::queryWithParser(pp, source_name, "uy_mean", m_uy_mean);
            utils::parser::queryWithParser(pp, source_name, "uz_mean", m_uz_mean);

            amrex::Real const u2 =
                Math::powi<2>(m_ux_mean) + Math::powi<2>(m_uy_mean) + Math::powi<2>(m_uz_mean);
            amrex::Real const velocity_magnitude = std::sqrt(u2/(1.0_rt+u2));

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                velocity_magnitude < 1.0,
                "Magnitude of mean velocity sqrt(u_mean^2/(1 + u_mean^2)) = " + std::to_string(velocity_magnitude) +
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
        }
        else {
            std::stringstream stringstream;
            std::string string;
            stringstream << "Maxwellian velocity mean distribution type '" << u_mean_dist_s << "' not recognized.";
            string = stringstream.str();
            WARPX_ABORT_WITH_MESSAGE(string);
        }
    }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "VelocityProperties: unexpected momentum_distribution_type '" + mom_dist_s +
            "' (expected 'maxwellian' or 'maxwell_juttner').");
    }
}
