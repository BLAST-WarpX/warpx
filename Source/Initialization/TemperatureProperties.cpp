/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "TemperatureProperties.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <algorithm>

/*
 * Construct TemperatureProperties based on the passed parameters.
 * If temperature is a constant, store value. If a parser, make and
 * store the parser function
 */
TemperatureProperties::TemperatureProperties (const amrex::ParmParse& pp, std::string const& source_name) {
    // Set defaults
    std::string mom_dist_s;
    utils::parser::query(pp, source_name, "momentum_distribution_type", mom_dist_s);
    std::transform(mom_dist_s.begin(),
                   mom_dist_s.end(),
                   mom_dist_s.begin(),
                   ::tolower);

    if (mom_dist_s != "maxwellian") {
        // Set defaults
        amrex::Real theta = 0; // quiet GCC warning maybe-uninitialized
        std::string temp_dist_s = "constant";
        utils::parser::query(pp, source_name, "theta_distribution_type", temp_dist_s);

        if (temp_dist_s == "constant") {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                utils::parser::queryWithParser(pp, source_name, "theta", theta),
                "Temperature parameter theta not specified");

            // Do validation on theta value

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(theta >= 0,
                "Temperature parameter theta = " + std::to_string(theta) +
                " is less than zero, which is not allowed");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                mom_dist_s != "maxwell_juttner" ||
                theta >= 0.1,
                "Temperature parameter theta = " +
                std::to_string(theta) +
                " is less than minimum 0.1 allowed for Maxwell-Juttner."
            );

            // if (mom_dist_s == "maxwellian" && theta > 0.01) {
            //     ablastr::warn_manager::WMRecordWarning(
            //         "Temperature",
            //         std::string{"Maxwell-Boltzmann distribution has errors greater than 1%"} +
            //         std::string{" for temperature parameter theta > 0.01. (theta = "} +
            //         std::to_string(theta) + " given)");
            // }

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
            std::stringstream stringstream;
            std::string string;
            stringstream << "Temperature distribution type '" << temp_dist_s << "' not recognized.";
            string = stringstream.str();
            WARPX_ABORT_WITH_MESSAGE(string);
        }
    }
    else {
        std::string u_std_dist_s = "constant";
        utils::parser::query(pp, source_name, "maxwellian_u_std_distribution_type", u_std_dist_s);
        if (u_std_dist_s == "constant") {

          // Query for all three components. Defaults are zeros if not provided.
            utils::parser::queryWithParser(pp, source_name, "ux_std", m_ux_std);
            utils::parser::queryWithParser(pp, source_name, "uy_std", m_uy_std);
            utils::parser::queryWithParser(pp, source_name, "uz_std", m_uz_std);

            if ( (m_ux_std*m_ux_std > 0.01) ||
                (m_uy_std*m_uy_std > 0.01) ||
                (m_uz_std*m_uz_std > 0.01) )
            {
                ablastr::warn_manager::WMRecordWarning(
                    "Temperature",
                    "Maxwellian distribution has errors greater than 1% "
                    "for temperature parameter(s) ux_std*ux_std, uy_std*uy_std, uz_std*uz_std > 0.01. "
                    "Values given: ux_std*ux_std = " + std::to_string(m_ux_std*m_ux_std) +
                    ", uy_std*uy_std = " + std::to_string(m_uy_std*m_uy_std) +
                    ", uz_std*uz_std = " + std::to_string(m_uz_std*m_uz_std)
                );
            }
            m_type = TempConstantVector;
        } else if (u_std_dist_s == "parser") {
            std::string str_ux_std_function, str_uy_std_function, str_uz_std_function;
            utils::parser::Store_parserString(pp, source_name, "ux_std_function(x,y,z)", str_ux_std_function);
            utils::parser::Store_parserString(pp, source_name, "uy_std_function(x,y,z)", str_uy_std_function);
            utils::parser::Store_parserString(pp, source_name, "uz_std_function(x,y,z)", str_uz_std_function);
            m_ptr_ux_std_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_ux_std_function,{"x","y","z"}));
            m_ptr_uy_std_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_uy_std_function,{"x","y","z"}));
            m_ptr_uz_std_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_uz_std_function,{"x","y","z"}));
            m_type = TempParserFunctionVector;

        }
        else {
            std::stringstream stringstream;
            std::string string;
            stringstream << "Maxwellian velocity standard deviation distribution type '" << u_std_dist_s << "' not recognized.";
            string = stringstream.str();
            WARPX_ABORT_WITH_MESSAGE(string);
        }
    }
}
