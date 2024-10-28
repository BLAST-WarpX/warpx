/* Copyright 2019-2020 Andrew Myers, Axel Huebl, Cameron Yang
 * David Grote, Luca Fedeli, Maxence Thevenet
 * Remi Lehe, Revathi Jambunathan, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PlasmaInjector.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Initialization/GetTemperature.H"
#include "Initialization/GetVelocity.H"
#include "Initialization/InjectorDensity.H"
#include "Initialization/InjectorMomentum.H"
#include "Initialization/InjectorPosition.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/SpeciesUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

using namespace amrex::literals;

PlasmaInjector::PlasmaInjector (int ispecies, const std::string& name,
    const amrex::Geometry& geom, const std::string& src_name):
    species_id{ispecies}, species_name{name}, source_name{src_name}
{

#ifdef AMREX_USE_GPU
    static_assert(std::is_trivially_copyable_v<InjectorPosition>,
                  "InjectorPosition must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<InjectorDensity>,
                  "InjectorDensity must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<InjectorMomentum>,
                  "InjectorMomentum must be trivially copyable");
#endif

    const amrex::ParmParse pp_species(species_name);

    utils::parser::queryWithParser(pp_species, source_name, "radially_weighted", radially_weighted);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(radially_weighted, "ERROR: Only radially_weighted=true is supported");

    // Unlimited boundaries
    xmin = std::numeric_limits<amrex::Real>::lowest();
    ymin = std::numeric_limits<amrex::Real>::lowest();
    zmin = std::numeric_limits<amrex::Real>::lowest();

    xmax = std::numeric_limits<amrex::Real>::max();
    ymax = std::numeric_limits<amrex::Real>::max();
    zmax = std::numeric_limits<amrex::Real>::max();

    // NOTE: When periodic boundaries are used, default injection range is set to mother grid dimensions.
    if( geom.isPeriodic(0) ) {
#       ifndef WARPX_DIM_1D_Z
        xmin = geom.ProbLo(0);
        xmax = geom.ProbHi(0);
#       else
        zmin = geom.ProbLo(0);
        zmax = geom.ProbHi(0);
#       endif
    }

#   ifndef WARPX_DIM_1D_Z
    if( geom.isPeriodic(1) ) {
#       ifndef WARPX_DIM_3D
        zmin = geom.ProbLo(1);
        zmax = geom.ProbHi(1);
#       else
        ymin = geom.ProbLo(1);
        ymax = geom.ProbHi(1);
#       endif
    }
#       endif

#   ifdef WARPX_DIM_3D
    if( geom.isPeriodic(2) ) {
        zmin = geom.ProbLo(2);
        zmax = geom.ProbHi(2);
    }
#   endif

    utils::parser::queryWithParser(pp_species, source_name, "xmin", xmin);
    utils::parser::queryWithParser(pp_species, source_name, "ymin", ymin);
    utils::parser::queryWithParser(pp_species, source_name, "zmin", zmin);
    utils::parser::queryWithParser(pp_species, source_name, "xmax", xmax);
    utils::parser::queryWithParser(pp_species, source_name, "ymax", ymax);
    utils::parser::queryWithParser(pp_species, source_name, "zmax", zmax);

    utils::parser::queryWithParser(pp_species, source_name, "density_min", density_min);
    utils::parser::queryWithParser(pp_species, source_name, "density_max", density_max);

    std::string injection_style = "none";
    utils::parser::query(pp_species, source_name, "injection_style", injection_style);
    std::transform(injection_style.begin(),
                   injection_style.end(),
                   injection_style.begin(),
                   ::tolower);

    num_particles_per_cell_each_dim.assign(3, 0);

    if (injection_style == "singleparticle") {
        setupSingleParticle(pp_species);
        return;
    } else if (injection_style == "multipleparticles") {
        setupMultipleParticles(pp_species);
        return;
    } else if (injection_style == "gaussian_beam") {
        setupGaussianBeam(pp_species);
    } else if (injection_style == "nrandompercell") {
        setupNRandomPerCell(pp_species);
    } else if (injection_style == "nfluxpercell") {
        setupNFluxPerCell(pp_species);
    } else if (injection_style == "nuniformpercell") {
        setupNuniformPerCell(pp_species);
    } else if (injection_style == "external_file") {
        setupExternalFile(pp_species);
    } else if (injection_style != "none") {
        SpeciesUtils::StringParseAbortMessage("Injection style", injection_style);
    }

    if (h_inj_rho) {
#ifdef AMREX_USE_GPU
        d_inj_rho = static_cast<InjectorDensity*>
            (amrex::The_Arena()->alloc(sizeof(InjectorDensity)));
        amrex::Gpu::htod_memcpy_async(d_inj_rho, h_inj_rho.get(), sizeof(InjectorDensity));
#else
        d_inj_rho = h_inj_rho.get();
#endif
    }
    if (h_inj_mom) {
#ifdef AMREX_USE_GPU
        d_inj_mom = static_cast<InjectorMomentum*>
            (amrex::The_Arena()->alloc(sizeof(InjectorMomentum)));
        amrex::Gpu::htod_memcpy_async(d_inj_mom, h_inj_mom.get(), sizeof(InjectorMomentum));
#else
        d_inj_mom = h_inj_mom.get();
#endif
    }
    amrex::Gpu::synchronize();
}

#ifdef AMREX_USE_GPU
PlasmaInjector::~PlasmaInjector ()
{
    if (d_inj_pos) {
        amrex::The_Arena()->free(d_inj_pos);
    }
    if (d_flux_pos) {
        amrex::The_Arena()->free(d_flux_pos);
    }
    if (d_inj_rho) {
        amrex::The_Arena()->free(d_inj_rho);
    }
    if (d_inj_mom) {
        amrex::The_Arena()->free(d_inj_mom);
    }
}
#else
PlasmaInjector::~PlasmaInjector () = default;
#endif

void PlasmaInjector::setupSingleParticle (amrex::ParmParse const& pp_species)
{
    utils::parser::getArrWithParser(pp_species, source_name, "single_particle_pos", single_particle_pos, 0, 3);
    utils::parser::getArrWithParser(pp_species, source_name, "single_particle_u", single_particle_u, 0, 3);
    for (auto& x : single_particle_u) {
        x *= PhysConst::c;
    }
    utils::parser::getWithParser(pp_species, source_name, "single_particle_weight", single_particle_weight);
    add_single_particle = true;
}

void PlasmaInjector::setupMultipleParticles (amrex::ParmParse const& pp_species)
{
    utils::parser::getArrWithParser(pp_species, source_name, "multiple_particles_pos_x", multiple_particles_pos_x);
    utils::parser::getArrWithParser(pp_species, source_name, "multiple_particles_pos_y", multiple_particles_pos_y);
    utils::parser::getArrWithParser(pp_species, source_name, "multiple_particles_pos_z", multiple_particles_pos_z);
    utils::parser::getArrWithParser(pp_species, source_name, "multiple_particles_ux", multiple_particles_ux);
    utils::parser::getArrWithParser(pp_species, source_name, "multiple_particles_uy", multiple_particles_uy);
    utils::parser::getArrWithParser(pp_species, source_name, "multiple_particles_uz", multiple_particles_uz);
    utils::parser::getArrWithParser(pp_species, source_name, "multiple_particles_weight", multiple_particles_weight);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        ((multiple_particles_pos_x.size() == multiple_particles_pos_y.size()) &&
         (multiple_particles_pos_x.size() == multiple_particles_pos_z.size()) &&
         (multiple_particles_pos_x.size() == multiple_particles_ux.size()) &&
         (multiple_particles_pos_x.size() == multiple_particles_uy.size()) &&
         (multiple_particles_pos_x.size() == multiple_particles_uz.size()) &&
         (multiple_particles_pos_x.size() == multiple_particles_weight.size())),
        "Error: The multiple particles source quantities must all have the same number of elements");
    for (auto& vx : multiple_particles_ux) { vx *= PhysConst::c; }
    for (auto& vy : multiple_particles_uy) { vy *= PhysConst::c; }
    for (auto& vz : multiple_particles_uz) { vz *= PhysConst::c; }
    add_multiple_particles = true;
}

void PlasmaInjector::setupGaussianBeam (amrex::ParmParse const& pp_species)
{
    utils::parser::getWithParser(pp_species, source_name, "x_m", x_m);
    utils::parser::getWithParser(pp_species, source_name, "y_m", y_m);
    utils::parser::getWithParser(pp_species, source_name, "z_m", z_m);
    utils::parser::getWithParser(pp_species, source_name, "x_rms", x_rms);
    utils::parser::getWithParser(pp_species, source_name, "y_rms", y_rms);
    utils::parser::getWithParser(pp_species, source_name, "z_rms", z_rms);
    utils::parser::queryWithParser(pp_species, source_name, "x_cut", x_cut);
    utils::parser::queryWithParser(pp_species, source_name, "y_cut", y_cut);
    utils::parser::queryWithParser(pp_species, source_name, "z_cut", z_cut);
    utils::parser::getWithParser(pp_species, source_name, "q_tot", q_tot);
    utils::parser::getWithParser(pp_species, source_name, "npart", npart);
    utils::parser::queryWithParser(pp_species, source_name, "do_symmetrize", do_symmetrize);
    utils::parser::queryWithParser(pp_species, source_name, "symmetrization_order", symmetrization_order);
    const bool focusing_is_specified = pp_species.contains("focal_distance");
    if(focusing_is_specified){
        do_focusing = true;
        utils::parser::queryWithParser(pp_species, source_name, "focal_distance", focal_distance);
    }
    const std::set<int> valid_symmetries = {4,8};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( valid_symmetries.count(symmetrization_order),
        "Error: Symmetrization only supported to orders 4 or 8 ");
    gaussian_beam = true;
    SpeciesUtils::parseMomentum(species_name, source_name, "gaussian_beam", h_inj_mom,
                                ux_parser, uy_parser, uz_parser,
                                ux_th_parser, uy_th_parser, uz_th_parser,
                                h_mom_temp, h_mom_vel);

#if defined(WARPX_DIM_XZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( y_rms > 0._rt,
        "Error: Gaussian beam y_rms must be strictly greater than 0 in 2D "
        "(it is used when computing the particles' weights from the total beam charge)");
#elif defined(WARPX_DIM_1D_Z)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( x_rms > 0._rt,
        "Error: Gaussian beam x_rms must be strictly greater than 0 in 1D "
        "(it is used when computing the particles' weights from the total beam charge)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( y_rms > 0._rt,
        "Error: Gaussian beam y_rms must be strictly greater than 0 in 1D "
        "(it is used when computing the particles' weights from the total beam charge)");
#endif
}

void PlasmaInjector::setupNRandomPerCell (amrex::ParmParse const& pp_species)
{
    utils::parser::getWithParser(pp_species, source_name, "num_particles_per_cell", num_particles_per_cell);
#if WARPX_DIM_RZ
    if (WarpX::n_rz_azimuthal_modes > 1) {
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_particles_per_cell>=2*WarpX::n_rz_azimuthal_modes,
        "Error: For accurate use of WarpX cylindrical geometry the number "
        "of particles should be at least two times n_rz_azimuthal_modes "
        "(Please visit PR#765 for more information.)");
    }
#endif
    // Construct InjectorPosition with InjectorPositionRandom.
    h_inj_pos = std::make_unique<InjectorPosition>(
        (InjectorPositionRandom*)nullptr,
        xmin, xmax, ymin, ymax, zmin, zmax);
#ifdef AMREX_USE_GPU
    d_inj_pos = static_cast<InjectorPosition*>
        (amrex::The_Arena()->alloc(sizeof(InjectorPosition)));
    amrex::Gpu::htod_memcpy_async(d_inj_pos, h_inj_pos.get(), sizeof(InjectorPosition));
#else
    d_inj_pos = h_inj_pos.get();
#endif

    SpeciesUtils::parseDensity(species_name, source_name, h_inj_rho, density_parser);
    SpeciesUtils::parseMomentum(species_name, source_name, "nrandompercell", h_inj_mom,
                                ux_parser, uy_parser, uz_parser,
                                ux_th_parser, uy_th_parser, uz_th_parser,
                                h_mom_temp, h_mom_vel);
}

void PlasmaInjector::setupNFluxPerCell (amrex::ParmParse const& pp_species)
{
    utils::parser::getWithParser(pp_species, source_name, "num_particles_per_cell", num_particles_per_cell_real);
#ifdef WARPX_DIM_RZ
    if (WarpX::n_rz_azimuthal_modes > 1) {
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_particles_per_cell_real>=2*WarpX::n_rz_azimuthal_modes,
        "Error: For accurate use of WarpX cylindrical geometry the number "
        "of particles should be at least two times n_rz_azimuthal_modes "
        "(Please visit PR#765 for more information.)");
    }
#endif

    // Check whether injection from the embedded boundary is requested
    utils::parser::queryWithParser(pp_species, source_name, "inject_from_embedded_boundary", m_inject_from_eb);
    if (m_inject_from_eb) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( EB::enabled(),
            "Error: Embedded boundary injection is only available when "
            "embedded boundaries are enabled.");
        flux_normal_axis = 2; // Interpret z as the normal direction to the EB
        flux_direction = 1;
    } else {
        // Injection is through a plane in this case.
        // Parse the parameters of the plane (position, normal direction, etc.)

        utils::parser::getWithParser(pp_species, source_name, "surface_flux_pos", surface_flux_pos);
        utils::parser::queryWithParser(pp_species, source_name, "flux_tmin", flux_tmin);
        utils::parser::queryWithParser(pp_species, source_name, "flux_tmax", flux_tmax);
        std::string flux_normal_axis_string;
        utils::parser::get(pp_species, source_name, "flux_normal_axis", flux_normal_axis_string);
        flux_normal_axis = -1;
#ifdef WARPX_DIM_RZ
        if      (flux_normal_axis_string == "r" || flux_normal_axis_string == "R") {
            flux_normal_axis = 0;
        }
        if      (flux_normal_axis_string == "t" || flux_normal_axis_string == "T") {
            flux_normal_axis = 1;
        }
#else
#    ifndef WARPX_DIM_1D_Z
        if      (flux_normal_axis_string == "x" || flux_normal_axis_string == "X") {
            flux_normal_axis = 0;
        }
#    endif
#endif
#ifdef WARPX_DIM_3D
        if (flux_normal_axis_string == "y" || flux_normal_axis_string == "Y") {
            flux_normal_axis = 1;
        }
#endif
        if (flux_normal_axis_string == "z" || flux_normal_axis_string == "Z") {
            flux_normal_axis = 2;
        }
#ifdef WARPX_DIM_3D
        const std::string flux_normal_axis_help = "'x', 'y', or 'z'.";
#else
#    ifdef WARPX_DIM_RZ
        const std::string flux_normal_axis_help = "'r' or 'z'.";
#    elif WARPX_DIM_XZ
        const std::string flux_normal_axis_help = "'x' or 'z'.";
#    else
        const std::string flux_normal_axis_help = "'z'.";
#    endif
    #endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(flux_normal_axis >= 0,
            "Error: Invalid value for flux_normal_axis. It must be " + flux_normal_axis_help);
        utils::parser::getWithParser(pp_species, source_name, "flux_direction", flux_direction);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(flux_direction == +1 || flux_direction == -1,
            "Error: flux_direction must be -1 or +1.");
    }

    // Construct InjectorPosition with InjectorPositionRandom.
    h_flux_pos = std::make_unique<InjectorPosition>(
        (InjectorPositionRandomPlane*)nullptr,
        xmin, xmax, ymin, ymax, zmin, zmax,
        flux_normal_axis);
#ifdef AMREX_USE_GPU
    d_flux_pos = static_cast<InjectorPosition*>
        (amrex::The_Arena()->alloc(sizeof(InjectorPosition)));
    amrex::Gpu::htod_memcpy_async(d_flux_pos, h_flux_pos.get(), sizeof(InjectorPosition));
#else
    d_flux_pos = h_flux_pos.get();
#endif

    parseFlux(pp_species);
    SpeciesUtils::parseMomentum(species_name, source_name, "nfluxpercell", h_inj_mom,
                                ux_parser, uy_parser, uz_parser,
                                ux_th_parser, uy_th_parser, uz_th_parser,
                                h_mom_temp, h_mom_vel,
                                flux_normal_axis, flux_direction);
}

void PlasmaInjector::setupNuniformPerCell (amrex::ParmParse const& pp_species)
{
    // Note that for RZ, three numbers are expected, r, theta, and z.
    // For 2D, only two are expected. The third is overwritten with 1.
    // For 1D, only one is expected. The second and third are overwritten with 1.
#if defined(WARPX_DIM_1D_Z)
    constexpr int num_required_ppc_each_dim = 1;
#elif defined(WARPX_DIM_XZ)
    constexpr int num_required_ppc_each_dim = 2;
#else
    constexpr int num_required_ppc_each_dim = 3;
#endif
    utils::parser::getArrWithParser(pp_species, source_name, "num_particles_per_cell_each_dim", num_particles_per_cell_each_dim,
                        0, num_required_ppc_each_dim);
#if WARPX_DIM_XZ
    num_particles_per_cell_each_dim.push_back(1);
#endif
#if WARPX_DIM_1D_Z
    num_particles_per_cell_each_dim.push_back(1); // overwrite 2nd number with 1
    num_particles_per_cell_each_dim.push_back(1); // overwrite 3rd number with 1
#endif
#if WARPX_DIM_RZ
    if (WarpX::n_rz_azimuthal_modes > 1) {
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_particles_per_cell_each_dim[1]>=2*WarpX::n_rz_azimuthal_modes,
        "Error: For accurate use of WarpX cylindrical geometry the number "
        "of particles in the theta direction should be at least two times "
        "n_rz_azimuthal_modes (Please visit PR#765 for more information.)");
    }
#endif
    // Construct InjectorPosition from InjectorPositionRegular.
    h_inj_pos = std::make_unique<InjectorPosition>(
        (InjectorPositionRegular*)nullptr,
        xmin, xmax, ymin, ymax, zmin, zmax,
        amrex::Dim3{num_particles_per_cell_each_dim[0],
            num_particles_per_cell_each_dim[1],
            num_particles_per_cell_each_dim[2]});
#ifdef AMREX_USE_GPU
    d_inj_pos = static_cast<InjectorPosition*>
        (amrex::The_Arena()->alloc(sizeof(InjectorPosition)));
    amrex::Gpu::htod_memcpy_async(d_inj_pos, h_inj_pos.get(), sizeof(InjectorPosition));
#else
    d_inj_pos = h_inj_pos.get();
#endif
    num_particles_per_cell = num_particles_per_cell_each_dim[0] *
                             num_particles_per_cell_each_dim[1] *
                             num_particles_per_cell_each_dim[2];
    SpeciesUtils::parseDensity(species_name, source_name, h_inj_rho, density_parser);
    SpeciesUtils::parseMomentum(species_name, source_name, "nuniformpercell", h_inj_mom,
                                ux_parser, uy_parser, uz_parser,
                                ux_th_parser, uy_th_parser, uz_th_parser,
                                h_mom_temp, h_mom_vel);
}

void PlasmaInjector::setupExternalFile (amrex::ParmParse const& pp_species)
{
#ifndef WARPX_USE_OPENPMD
    WARPX_ABORT_WITH_MESSAGE(
        "WarpX has to be compiled with USE_OPENPMD=TRUE to be able"
        " to read the external openPMD file with species data");
#endif
    external_file = true;
    std::string str_injection_file;
    utils::parser::get(pp_species, source_name, "injection_file", str_injection_file);
    // optional parameters
    utils::parser::queryWithParser(pp_species, source_name, "q_tot", q_tot);
    utils::parser::queryWithParser(pp_species, source_name, "z_shift",z_shift);

#ifdef WARPX_USE_OPENPMD
    const bool charge_is_specified = pp_species.contains("charge");
    const bool mass_is_specified = pp_species.contains("mass");
    const bool species_is_specified = pp_species.contains("species_type");

    if (amrex::ParallelDescriptor::IOProcessor()) {
        m_openpmd_input_series = std::make_unique<openPMD::Series>(
            str_injection_file, openPMD::Access::READ_ONLY);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_openpmd_input_series->iterations.size() == 1u,
            "External file should contain only 1 iteration\n");
        openPMD::Iteration it = m_openpmd_input_series->iterations.begin()->second;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            it.particles.size() == 1u,
            "External file should contain only 1 species\n");
        std::string const ps_name = it.particles.begin()->first;
        openPMD::ParticleSpecies ps = it.particles.begin()->second;

        charge_from_source = ps.contains("charge");
        mass_from_source = ps.contains("mass");

        if (charge_from_source) {
            if (charge_is_specified) {
                ablastr::warn_manager::WMRecordWarning("Species",
                    "Both '" + ps_name + ".charge' and '" +
                        ps_name + ".injection_file' specify a charge.\n'" +
                        ps_name + ".charge' will take precedence.\n");
            }
            else if (species_is_specified) {
                ablastr::warn_manager::WMRecordWarning("Species",
                    "Both '" + ps_name + ".species_type' and '" +
                        ps_name + ".injection_file' specify a charge.\n'" +
                        ps_name + ".species_type' will take precedence.\n");
            }
            else {
                // TODO: Add ASSERT_WITH_MESSAGE to test if charge is a constant record
                auto p_q_ptr =
                    ps["charge"][openPMD::RecordComponent::SCALAR].loadChunk<amrex::ParticleReal>();
                m_openpmd_input_series->flush();
                amrex::ParticleReal const p_q = p_q_ptr.get()[0];
                auto const charge_unit = static_cast<amrex::Real>(ps["charge"][openPMD::RecordComponent::SCALAR].unitSI());
                charge = p_q * charge_unit;
            }
        }
        if (mass_from_source) {
            if (mass_is_specified) {
                ablastr::warn_manager::WMRecordWarning("Species",
                    "Both '" + ps_name + ".mass' and '" +
                        ps_name + ".injection_file' specify a charge.\n'" +
                        ps_name + ".mass' will take precedence.\n");
            }
            else if (species_is_specified) {
                ablastr::warn_manager::WMRecordWarning("Species",
                    "Both '" + ps_name + ".species_type' and '" +
                        ps_name + ".injection_file' specify a mass.\n'" +
                        ps_name + ".species_type' will take precedence.\n");
            }
            else {
                // TODO: Add ASSERT_WITH_MESSAGE to test if mass is a constant record
                auto p_m_ptr =
                    ps["mass"][openPMD::RecordComponent::SCALAR].loadChunk<amrex::ParticleReal>();
                m_openpmd_input_series->flush();
                amrex::ParticleReal const p_m = p_m_ptr.get()[0];
                auto const mass_unit = static_cast<amrex::Real>(ps["mass"][openPMD::RecordComponent::SCALAR].unitSI());
                mass = p_m * mass_unit;
            }
        }
    } // IOProcessor

    // Broadcast charge and mass to non-IO processors if read in from the file
    std::array<int,2> flags{charge_from_source, mass_from_source};
    amrex::ParallelDescriptor::Bcast(flags.data(), flags.size(), amrex::ParallelDescriptor::IOProcessorNumber());
    charge_from_source = flags[0];
    mass_from_source   = flags[1];
    if (charge_from_source) {
        amrex::ParallelDescriptor::Bcast(&charge, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    }
    if (mass_from_source) {
        amrex::ParallelDescriptor::Bcast(&mass, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    }
#else
    WARPX_ABORT_WITH_MESSAGE(
        "Plasma injection via external_file requires openPMD support: "
        "Add USE_OPENPMD=TRUE when compiling WarpX.");
#endif  // WARPX_USE_OPENPMD
}

// Depending on injection type at runtime, initialize inj_flux
// so that inj_flux->getFlux calls
// InjectorFlux[Constant or Parser or etc.].getFlux.
void PlasmaInjector::parseFlux (amrex::ParmParse const& pp_species)
{
    // parse flux information
    std::string flux_prof_s;
    utils::parser::get(pp_species, source_name, "flux_profile", flux_prof_s);
    std::transform(flux_prof_s.begin(), flux_prof_s.end(),
                   flux_prof_s.begin(), ::tolower);
    if (flux_prof_s == "constant") {
        utils::parser::getWithParser(pp_species, source_name, "flux", flux);
        // Construct InjectorFlux with InjectorFluxConstant.
        h_inj_flux.reset(new InjectorFlux((InjectorFluxConstant*)nullptr, flux));
    } else if (flux_prof_s == "parse_flux_function") {
        utils::parser::Store_parserString(pp_species, source_name, "flux_function(x,y,z,t)", str_flux_function);
        // Construct InjectorFlux with InjectorFluxParser.
        flux_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_flux_function,{"x","y","z","t"}));
        h_inj_flux.reset(new InjectorFlux((InjectorFluxParser*)nullptr,
            flux_parser->compile<4>()));
    } else {
        SpeciesUtils::StringParseAbortMessage("Flux profile type", flux_prof_s);
    }
    if (h_inj_flux) {
#ifdef AMREX_USE_GPU
        d_inj_flux = static_cast<InjectorFlux*>
            (amrex::The_Arena()->alloc(sizeof(InjectorFlux)));
        amrex::Gpu::htod_memcpy_async(d_inj_flux, h_inj_flux.get(), sizeof(InjectorFlux));
#else
        d_inj_flux = h_inj_flux.get();
#endif
    }

}

amrex::XDim3 PlasmaInjector::getMomentum (amrex::Real x,
                                          amrex::Real y,
                                          amrex::Real z) const noexcept
{
    return h_inj_mom->getMomentum(x, y, z, amrex::RandomEngine{}); // gamma*beta
}

bool PlasmaInjector::insideBounds (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
{
    return (x < xmax and x >= xmin and
            y < ymax and y >= ymin and
            z < zmax and z >= zmin);
}

bool PlasmaInjector::overlapsWith (const amrex::XDim3& lo,
                                   const amrex::XDim3& hi) const noexcept
{
    return  (    (xmin <= hi.x) && (xmax >= lo.x)
              && (ymin <= hi.y) && (ymax >= lo.y)
              && (zmin <= hi.z) && (zmax >= lo.z) );
}

bool
PlasmaInjector::queryCharge (amrex::ParticleReal& a_charge) const
{
    if (charge_from_source) {
        a_charge = charge;
    }
    return charge_from_source;
}

bool
PlasmaInjector::queryMass (amrex::ParticleReal& a_mass) const
{
    if (mass_from_source) {
        a_mass = mass;
    }
    return mass_from_source;
}

InjectorPosition*
PlasmaInjector::getInjectorPosition () const
{
    return d_inj_pos;
}

InjectorPosition*
PlasmaInjector::getInjectorFluxPosition () const
{
    return d_flux_pos;
}

InjectorDensity*
PlasmaInjector::getInjectorDensity () const
{
    return d_inj_rho;
}

InjectorFlux*
PlasmaInjector::getInjectorFlux () const
{
    return d_inj_flux;
}

InjectorMomentum*
PlasmaInjector::getInjectorMomentumDevice () const
{
    return d_inj_mom;
}

InjectorMomentum*
PlasmaInjector::getInjectorMomentumHost () const
{
    return h_inj_mom.get();
}
