/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "VelocityProperties.H"

#include "ExternalField.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>

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
                              std::string const& dist_type_param, VelocityProperties& vel,
                              amrex::Geometry const& geom)
    {
        amrex::ignore_unused(geom);

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
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
            if (WarpX::gamma_boost > 1.0) {
                WARPX_ABORT_WITH_MESSAGE(
                    dist_type_param + " = read_from_file is not "
                    "supported in boosted-frame simulations yet.");
            }
            utils::parser::get(pp, source_name, "read_u_mean_from_path",
                               vel.m_read_u_mean_path);
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const problo =
                geom.ProbLoArray();
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx =
                geom.CellSizeArray();
            amrex::Box const dombox = amrex::convert(geom.Domain(), amrex::IntVect(1));
            vel.m_u_mean_x_reader = std::make_unique<ExternalFieldReader>(
                vel.m_read_u_mean_path, "u_mean", "x", problo, dx, dombox, false);
            vel.m_u_mean_y_reader = std::make_unique<ExternalFieldReader>(
                vel.m_read_u_mean_path, "u_mean", "y", problo, dx, dombox, false);
            vel.m_u_mean_z_reader = std::make_unique<ExternalFieldReader>(
                vel.m_read_u_mean_path, "u_mean", "z", problo, dx, dombox, false);
            amrex::BoxArray const grids;
            amrex::DistributionMapping const dmap;
            vel.m_u_mean_x_reader->prepare(grids, dmap, amrex::IntVect(0));
            vel.m_u_mean_y_reader->prepare(grids, dmap, amrex::IntVect(0));
            vel.m_u_mean_z_reader->prepare(grids, dmap, amrex::IntVect(0));
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
{
    std::string mom_dist_s;
    utils::parser::query(pp, source_name, "momentum_distribution_type", mom_dist_s);
    if (mom_dist_s == "maxwell_juttner") {
        ParseVelocityVector(pp, source_name, "maxwell_juttner_u_mean_distribution_type", *this,
                            geom);
    } else if (mom_dist_s == "maxwellian") {
        ParseVelocityVector(pp, source_name, "maxwellian_u_mean_distribution_type", *this, geom);
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
