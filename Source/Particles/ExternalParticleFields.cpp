/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Diederichs (CERN)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ExternalParticleFields.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>
#include <ablastr/warn_manager/WarnManager.H>

using amrex::ParmParse;

void
ExternalParticleFields::ReadParameters () {
    amrex::ParmParse pp_particles("particles");

    m_E_field_metadata.clear();
    m_B_field_metadata.clear();

    std::string single_field_path;

    // helper function to read in all the meta data for an external field from
    // file
    auto read_named_spec = [&] (const std::string& fname,
                                bool isE) -> ParticleFieldMetaData {
        ParticleFieldMetaData md;

        // if the name is specified, so must be the path
        pp_particles.get((fname + ".read_fields_from_path").c_str(), md.path);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !md.path.empty(),
            "External particle field '" + fname + "' requires 'particles." +
                fname + ".read_fields_from_path = <file>' to be set.");

        // The mathematical expression for the time dependency
        // read_fields_E_dependency(t) can be provided in the input file. If not
        // provided, it defaults to '1.0'
        md.time_function = "1.0";
        const std::string tf_key =
            isE ? (fname + ".read_fields_E_dependency(t)")
                : (fname + ".read_fields_B_dependency(t)");
        pp_particles.query(tf_key.c_str(), md.time_function);

        md.time_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(md.time_function, {"t"}));
        md.time_executor = md.time_parser->compile<1>();

        return md;
    };

    std::string e_init;
    pp_particles.query("E_ext_particle_init_style", e_init);
    if (e_init == "read_from_file") {

        // Read in E field names
        std::vector<std::string> E_names;
        pp_particles.queryarr("E_ext_particle_fields", E_names);

        // Initialize E fields by name
        for (const auto& n : E_names) {
            m_E_field_metadata.emplace_back(read_named_spec(n, /*isE=*/true));
        }

        // No E field names provided, initialize for single E field without name
        if (m_E_field_metadata.empty()) {

            // no E field names provided but read_from_file requested, force to
            // read path
            pp_particles.get("read_fields_from_path", single_field_path);

            // initialize single E field meta data
            ParticleFieldMetaData md;
            md.path = single_field_path;

            // The mathematical expression for the time dependency
            // read_fields_E_dependency(t) can be provided in the input file. If
            // not provided, it defaults to '1.0'
            md.time_function = "1.0";
            pp_particles.query("read_fields_E_dependency(t)", md.time_function);

            md.time_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(md.time_function, {"t"}));
            md.time_executor = md.time_parser->compile<1>();

            m_E_field_metadata.emplace_back(std::move(md));

        } else {
            // Provide warning in case syntax for single field is used, although
            // it is initialized via name list
            if (pp_particles.query("read_fields_from_path",
                                   single_field_path)) {
                ablastr::warn_manager::WMRecordWarning(
                    "read_fields_from_path",
                    "particles.read_fields_from_path is ignored when "
                    "particles.B_ext_particle_fields is used.",
                    ablastr::warn_manager::WarnPriority::low);
            }
            if (pp_particles.query("read_fields_E_dependency(t)",
                                   single_field_path)) {
                ablastr::warn_manager::WMRecordWarning(
                    "read_fields_E_dependency(t)",
                    "particles.read_fields_E_dependency(t) is ignored when "
                    "particles.E_ext_particle_fields is used.",
                    ablastr::warn_manager::WarnPriority::low);
            }
        }
    }

    std::string b_init;
    pp_particles.query("B_ext_particle_init_style", b_init);
    if (b_init == "read_from_file") {

        // Read in B field names
        std::vector<std::string> B_names;
        pp_particles.queryarr("B_ext_particle_fields", B_names);

        // Initialize B fields by name
        for (const auto& n : B_names) {
            m_B_field_metadata.emplace_back(read_named_spec(n, /*isE=*/false));
        }

        // No E field names provided, initialize for single E field without name
        if (m_B_field_metadata.empty()) {

            // no B field names provided but read_from_file requested, force to
            // read path
            pp_particles.get("read_fields_from_path", single_field_path);

            // initialize single B field meta data

            ParticleFieldMetaData md;
            md.path = single_field_path;

            // The mathematical expression for the time dependency
            // read_fields_E_dependency(t) can be provided in the input file. If
            // not provided, it defaults to '1.0'
            md.time_function = "1.0";
            pp_particles.query("read_fields_B_dependency(t)", md.time_function);

            md.time_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(md.time_function, {"t"}));
            md.time_executor = md.time_parser->compile<1>();

            m_B_field_metadata.emplace_back(std::move(md));
        } else {
            // Provide warning in case syntax for single field is used, although
            // it is initialized via name list
            if (pp_particles.query("read_fields_from_path",
                                   single_field_path)) {
                ablastr::warn_manager::WMRecordWarning(
                    "read_fields_from_path",
                    "particles.read_fields_from_path is ignored when "
                    "particles.B_ext_particle_fields is used.",
                    ablastr::warn_manager::WarnPriority::low);
            }
            if (pp_particles.query("read_fields_B_dependency(t)",
                                   single_field_path)) {
                ablastr::warn_manager::WMRecordWarning(
                    "read_fields_B_dependency(t)",
                    "particles.read_fields_B_dependency(t) is ignored when "
                    "particles.B_ext_particle_fields is used.",
                    ablastr::warn_manager::WarnPriority::low);
            }
        }
    }

    // since we add both the multi-field read or the single field read, the size
    // can simply be used
    m_nEfields = static_cast<int>(m_E_field_metadata.size());
    m_nBfields = static_cast<int>(m_B_field_metadata.size());
}
