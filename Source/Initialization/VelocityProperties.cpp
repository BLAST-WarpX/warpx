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

VelocityProperties::VelocityProperties (const amrex::ParmParse& pp, std::string const& source_name)
{
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
}
