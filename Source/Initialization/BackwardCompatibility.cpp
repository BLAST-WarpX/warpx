/* Copyright 2025 Luca Fedeli, Arianna Formenti, Remi Lehe,  Olga Shapoval
 * Davide Terzani, Maxence Thevenet, Axel Huebl, Edoardo Zoni
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/TextMsg.H"

#include "ablastr/warn_manager/WarnManager.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <iostream>
#include <string>
#include <vector>

using namespace amrex;

namespace warpx::initialization
{

    void
    backward_compatibility ()
    {
        // Auxiliary variables
        int backward_int;
        bool backward_bool;
        std::string backward_str;
        amrex::Real backward_Real;

        const ParmParse pp_amr("amr");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_amr.query("plot_int", backward_int),
            "amr.plot_int is not supported anymore. Please use the new syntax for diagnostics:\n"
            "diagnostics.diags_names = my_diag\n"
            "my_diag.intervals = 10\n"
            "for output every 10 iterations. See documentation for more information"
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_amr.query("plot_file", backward_str),
            "amr.plot_file is not supported anymore. "
            "Please use the new syntax for diagnostics, see documentation."
        );

        const ParmParse pp_warpx("warpx");
        std::vector<std::string> backward_strings;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.queryarr("fields_to_plot", backward_strings),
            "warpx.fields_to_plot is not supported anymore. "
            "Please use the new syntax for diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("plot_finepatch", backward_int),
            "warpx.plot_finepatch is not supported anymore. "
            "Please use the new syntax for diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("plot_crsepatch", backward_int),
            "warpx.plot_crsepatch is not supported anymore. "
            "Please use the new syntax for diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.queryarr("load_balance_int", backward_strings),
            "warpx.load_balance_int is no longer a valid option. "
            "Please use the renamed option algo.load_balance_intervals instead."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.queryarr("load_balance_intervals", backward_strings),
            "warpx.load_balance_intervals is no longer a valid option. "
            "Please use the renamed option algo.load_balance_intervals instead."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("load_balance_efficiency_ratio_threshold", backward_Real),
            "warpx.load_balance_efficiency_ratio_threshold is not supported anymore. "
            "Please use the renamed option algo.load_balance_efficiency_ratio_threshold."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("load_balance_with_sfc", backward_int),
            "warpx.load_balance_with_sfc is not supported anymore. "
            "Please use the renamed option algo.load_balance_with_sfc."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("load_balance_knapsack_factor", backward_Real),
            "warpx.load_balance_knapsack_factor is not supported anymore. "
            "Please use the renamed option algo.load_balance_knapsack_factor."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.queryarr("override_sync_int", backward_strings),
            "warpx.override_sync_int is no longer a valid option. "
            "Please use the renamed option warpx.override_sync_intervals instead."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.queryarr("sort_int", backward_strings),
            "warpx.sort_int is no longer a valid option. "
            "Please use the renamed option warpx.sort_intervals instead."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("do_nodal", backward_int),
            "warpx.do_nodal is not supported anymore. "
            "Please use the flag warpx.grid_type instead."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("use_kspace_filter", backward_int),
            "warpx.use_kspace_filter is not supported anymore. "
            "Please use the flag use_filter, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("do_pml", backward_int),
            "do_pml is not supported anymore. Please use boundary.field_lo and boundary.field_hi"
            " to set the boundary conditions."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("serialize_ics", backward_bool),
            "warpx.serialize_ics is no longer a valid option. "
            "Please use the renamed option warpx.serialize_initial_conditions instead."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("do_back_transformed_diagnostics", backward_int),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("lab_data_directory", backward_str),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("num_snapshots_lab", backward_int),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("dt_snapshots_lab", backward_Real),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("dz_snapshots_lab", backward_Real),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("do_back_transformed_fields", backward_int),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("buffer_size", backward_int),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("do_multi_J", backward_bool),
            "warpx.do_multi_J is no longer used. Please use psatd.JRhom instead."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_warpx.query("do_multi_J_n_depositions", backward_int),
            "warpx.do_multi_J_n_depositions is no longer used. Please use psatd.JRhom instead."
        );

        const ParmParse pp_psatd("psatd");

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_psatd.query("J_in_time", backward_str),
            "psatd.J_in_time is no longer used. Please use psatd.JRhom instead."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_psatd.query("rho_in_time", backward_str),
            "psatd.rho_in_time is no longer used. Please use psatd.JRhom instead."
        );

        const ParmParse pp_slice("slice");

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_slice.query("num_slice_snapshots_lab", backward_int),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_slice.query("dt_slice_snapshots_lab", backward_Real),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_slice.query("particle_slice_width_lab", backward_Real),
            "Legacy back-transformed diagnostics are not supported anymore. "
            "Please use the new syntax for back-transformed diagnostics, see documentation."
        );

        const ParmParse pp_interpolation("interpolation");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_interpolation.query("nox", backward_int) &&
            !pp_interpolation.query("noy", backward_int) &&
            !pp_interpolation.query("noz", backward_int),
            "interpolation.nox (as well as .noy, .noz) are not supported anymore."
            " Please use the new syntax algo.particle_shape instead"
        );

        const ParmParse pp_algo("algo");
        int backward_mw_solver;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_algo.query("maxwell_fdtd_solver", backward_mw_solver),
            "algo.maxwell_fdtd_solver is not supported anymore. "
            "Please use the renamed option algo.maxwell_solver");

        const ParmParse pp_particles("particles");
        int nspecies;
        if (pp_particles.query("nspecies", nspecies)){
            ablastr::warn_manager::WMRecordWarning("Species",
                "particles.nspecies is ignored. Just use particles.species_names please.",
                ablastr::warn_manager::WarnPriority::low);
        }

        if (pp_particles.contains("photon_species")){
            ablastr::warn_manager::WMRecordWarning("Species",
                "particles.photon_species is deprecated and may be removed in the future. "
                "It is recommended to initialize photon particles by setting their "
                "'species_type' to 'photon', instead.",
                ablastr::warn_manager::WarnPriority::low);
        }

        std::vector<std::string> backward_sp_names;
        pp_particles.queryarr("species_names", backward_sp_names);
        for(const std::string& speciesiter : backward_sp_names){
            const ParmParse pp_species(speciesiter);
            std::vector<amrex::Real> backward_vel;
            std::stringstream ssspecies;

            ssspecies << "'" << speciesiter << ".multiple_particles_vel_<x,y,z>'";
            ssspecies << " are not supported anymore. ";
            ssspecies << "Please use the renamed variables ";
            ssspecies << "'" << speciesiter << ".multiple_particles_u<x,y,z>' .";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !pp_species.queryarr("multiple_particles_vel_x", backward_vel) &&
                !pp_species.queryarr("multiple_particles_vel_y", backward_vel) &&
                !pp_species.queryarr("multiple_particles_vel_z", backward_vel),
                ssspecies.str());

            ssspecies.str("");
            ssspecies.clear();
            ssspecies << "'" << speciesiter << ".single_particle_vel'";
            ssspecies << " is not supported anymore. ";
            ssspecies << "Please use the renamed variable ";
            ssspecies << "'" << speciesiter << ".single_particle_u' .";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !pp_species.queryarr("single_particle_vel", backward_vel),
                ssspecies.str());
        }

        const ParmParse pp_collisions("collisions");
        int ncollisions;
        if (pp_collisions.query("ncollisions", ncollisions)){
            ablastr::warn_manager::WMRecordWarning("Collisions",
                "collisions.ncollisions is ignored. Just use particles.collision_names please.",
                ablastr::warn_manager::WarnPriority::low);
        }

        std::vector<std::string> backward_coll_names;
        pp_collisions.queryarr("collision_names", backward_coll_names);
        for(const std::string& coll_name : backward_coll_names){
            const ParmParse pp_coll(coll_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !pp_coll.query("fusion_multiplier", backward_Real) &&
                !pp_coll.query("fusion_probability_threshold", backward_Real) &&
                !pp_coll.query("fusion_probability_target_value", backward_Real),
                "Inputs fusion_multiplier, fusion_probability_threshold & fusion_probability_target_value "
                "are deprecated. Please use event_multiplier, probability_threshold & probability_target_value."
            );
        }

        const ParmParse pp_lasers("lasers");
        int nlasers;
        if (pp_lasers.query("nlasers", nlasers)){
            ablastr::warn_manager::WMRecordWarning("Laser",
                "lasers.nlasers is ignored. Just use lasers.names please.",
                ablastr::warn_manager::WarnPriority::low);
        }
    }

}
