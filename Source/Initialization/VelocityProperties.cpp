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

namespace {
    /** Parse the bulk drift momentum vector (ux_mean, uy_mean, uz_mean) shared by the
     * `maxwellian` and `maxwell_juttner` momentum distributions.
     *
     * The bulk drift is the normalized momentum u_mean = gamma*v/c. Its components are
     * either constant, given by spatially-dependent parser functions, or read from
     * openPMD data, selected by the input parameter `<dist_type_param>` (`constant`
     * by default, `parser`, or `read_from_file`).
     */
    void ParseVelocityVector (const amrex::ParmParse& pp, std::string const& source_name,
                              std::string const& dist_type_param, VelocityProperties& vel)
    {
        std::string u_mean_dist_s = "constant";
        utils::parser::query(pp, source_name, dist_type_param.c_str(), u_mean_dist_s);
        if (u_mean_dist_s == "constant") {
            utils::parser::queryWithParser(pp, source_name, "ux_mean", vel.m_ux_mean);
            utils::parser::queryWithParser(pp, source_name, "uy_mean", vel.m_uy_mean);
            utils::parser::queryWithParser(pp, source_name, "uz_mean", vel.m_uz_mean);
            vel.m_type = VelConstantVector;
        } else if (u_mean_dist_s == "parser") {
            std::string str_ux_mean_function, str_uy_mean_function, str_uz_mean_function;
            utils::parser::Store_parserString(pp, source_name, "ux_mean_function(x,y,z)", str_ux_mean_function);
            utils::parser::Store_parserString(pp, source_name, "uy_mean_function(x,y,z)", str_uy_mean_function);
            utils::parser::Store_parserString(pp, source_name, "uz_mean_function(x,y,z)", str_uz_mean_function);
            vel.m_ptr_ux_mean_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_ux_mean_function,{"x","y","z"}));
            vel.m_ptr_uy_mean_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_uy_mean_function,{"x","y","z"}));
            vel.m_ptr_uz_mean_parser =
                std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(str_uz_mean_function,{"x","y","z"}));
            vel.m_type = VelParserFunctionVector;
        } else if (u_mean_dist_s == "read_from_file") {
            if (dist_type_param != "maxwellian_u_mean_distribution_type") {
                WARPX_ABORT_WITH_MESSAGE(
                    dist_type_param + " = read_from_file is only supported for "
                    "maxwellian momentum distributions.");
            }
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
            utils::parser::get(pp, source_name, "read_u_mean_from_path",
                               vel.m_read_u_mean_path);
            {
                std::string const key_with_src =
                    source_name.empty() ? std::string("read_u_mean_distributed")
                                        : source_name + ".read_u_mean_distributed";
                if (pp.contains(key_with_src)) {
                    pp.query(key_with_src.c_str(), vel.m_read_u_mean_distributed);
                } else {
                    pp.query("read_u_mean_distributed", vel.m_read_u_mean_distributed);
                }
            }
            vel.m_type = VelFromFileVector;
#else
            WARPX_ABORT_WITH_MESSAGE(
                dist_type_param + " = read_from_file requires WarpX built with "
                "openPMD support and is not supported in RZ/RCYLINDER/RSPHERE geometries.");
#endif
        }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "Mean velocity distribution type '" + u_mean_dist_s + "' not recognized.");
        }
    }
}

/**
* Construct VelocityProperties from the passed particle source parameters.
* Parse the momentum distribution type and initialize the corresponding
* velocity parameters: `ux_mean`, `uy_mean`, and `uz_mean` (or the parser
* functions `ux_mean_function`, `uy_mean_function`, `uz_mean_function`) for
* the `maxwellian` and `maxwell_juttner` distributions; or the parser functions
* `momentum_function_ux`, `momentum_function_uy`, `momentum_function_uz` for
* `parse_momentum_function`.
*/
VelocityProperties::VelocityProperties (const amrex::ParmParse& pp, std::string const& source_name,
                                        amrex::Geometry const& geom)
    : m_geom(geom)
{
    std::string mom_dist_s;
    utils::parser::query(pp, source_name, "momentum_distribution_type", mom_dist_s);
    if (mom_dist_s == "maxwell_juttner") {
        ParseVelocityVector(pp, source_name, "maxwell_juttner_u_mean_distribution_type", *this);
    } else if (mom_dist_s == "maxwellian") {
        ParseVelocityVector(pp, source_name, "maxwellian_u_mean_distribution_type", *this);
    }
    else if (mom_dist_s == "parse_momentum_function") {
        std::string str_ux_mean_function, str_uy_mean_function, str_uz_mean_function;
        utils::parser::Store_parserString(pp, source_name, "momentum_function_ux(x,y,z)", str_ux_mean_function);
        utils::parser::Store_parserString(pp, source_name, "momentum_function_uy(x,y,z)", str_uy_mean_function);
        utils::parser::Store_parserString(pp, source_name, "momentum_function_uz(x,y,z)", str_uz_mean_function);
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
        WARPX_ABORT_WITH_MESSAGE(
            "VelocityProperties: unexpected momentum_distribution_type '" + mom_dist_s +
            "' (expected 'maxwellian', 'maxwell_juttner', or 'parse_momentum_function').");
    }
}
