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

#include <sstream>

namespace
{
    bool specified (const amrex::ParmParse& pp, const std::string& group, const char* name)
    {
        if (pp.contains(name)) {
            return true;
        }
        if (!group.empty()) {
            return pp.contains(group + '.' + name);
        }
        return false;
    }

    bool any_u_std_specified (const amrex::ParmParse& pp, const std::string& group)
    {
        static const char* keys[] = {
            "maxwellian_u_std_distribution_type",
            "ux_std", "uy_std", "uz_std",
            "ux_std_function(x,y,z)",
            "uy_std_function(x,y,z)",
            "uz_std_function(x,y,z)",
            "read_ux_std_from_path",
            "read_uy_std_from_path",
            "read_uz_std_from_path",
            "ux_std_openpmd_mesh",
            "uy_std_openpmd_mesh",
            "uz_std_openpmd_mesh",
            "read_u_std_distributed"
        };
        return std::ranges::any_of(keys, [&](const char* k) {
            return specified(pp, group, k);
        });
    }
}

TemperatureProperties::TemperatureProperties (const amrex::ParmParse& pp, std::string const& source_name,
                                                amrex::Geometry const& geom)
    : TemperatureProperties(pp, source_name, std::numeric_limits<amrex::Real>::quiet_NaN(), geom)
{
}
/** Construct TemperatureProperties from the passed particle source parameters.
 *  Parse the momentum distribution type and initialize the corresponding
 *  temperature parameters: thermal spread `ux_std`, `uy_std`, `uz_std`
 *  for `maxwellian` distribution, and `theta` for `maxwell_juttner`.
 */
TemperatureProperties::TemperatureProperties (const amrex::ParmParse& pp, std::string const& source_name,
                                             amrex::Real species_mass, amrex::Geometry const& geom)
{
    m_species_mass = species_mass;
    m_geom = geom;

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

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                theta >= 0.1,
                "Temperature parameter theta = " +
                std::to_string(theta) +
                " is less than minimum 0.1 allowed for Maxwell-Juttner."
            );

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
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
                utils::parser::get(pp, source_name, "read_ux_std_from_path", m_read_ux_std_path);
                utils::parser::get(pp, source_name, "read_uy_std_from_path", m_read_uy_std_path);
                utils::parser::get(pp, source_name, "read_uz_std_from_path", m_read_uz_std_path);
                utils::parser::query(pp, source_name, "ux_std_openpmd_mesh", m_ux_std_openpmd_mesh);
                utils::parser::query(pp, source_name, "uy_std_openpmd_mesh", m_uy_std_openpmd_mesh);
                utils::parser::query(pp, source_name, "uz_std_openpmd_mesh", m_uz_std_openpmd_mesh);
                {
                    std::string const key_with_src =
                        source_name.empty() ? std::string("read_u_std_distributed")
                                            : source_name + ".read_u_std_distributed";
                    if (pp.contains(key_with_src.c_str())) {
                        pp.query(key_with_src.c_str(), m_read_u_std_distributed);
                    } else {
                        pp.query("read_u_std_distributed", m_read_u_std_distributed);
                    }
                }
                m_type = TempFromFileVector;
#else
                WARPX_ABORT_WITH_MESSAGE(
                    "maxwellian_u_std_distribution_type = read_from_file requires "
                    "WarpX built with openPMD support and is not supported in "
                    "RCYLINDER/RSPHERE geometries.");
#endif
            }
        else {
                std::stringstream ss;
                ss << "Maxwellian velocity standard deviation distribution type '" << u_std_dist_s
                   << "' not recognized.";
                WARPX_ABORT_WITH_MESSAGE(ss.str());
            }
        else {
            std::stringstream ss;
            ss << "Maxwellian velocity standard deviation distribution type '" << u_std_dist_s
               << "' not recognized.";
            WARPX_ABORT_WITH_MESSAGE(ss.str());
        }
    }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "TemperatureProperties: unexpected momentum_distribution_type '" + mom_dist_s +
            "' (expected 'maxwellian' or 'maxwell_juttner').");
    }
}
