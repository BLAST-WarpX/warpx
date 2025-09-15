/* Copyright 2019-2020 Andrew Myers, Aurore Blelly, Axel Huebl
 * David Grote, Glenn Richardson, Jean-Luc Vay
 * Ligia Diana Amorim, Luca Fedeli, Maxence Thevenet
 * Michael Rowan, Remi Lehe, Revathi Jambunathan
 * Weiqun Zhang, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PhysicalParticleContainer.H"

#include "Fields.H"
#include "Filter/NCIGodfreyFilter.H"
#include "Initialization/PlasmaInjector.H"
#include "MultiParticleContainer.H"
#include "Parallelization/WarpXSumGuardCells.H"
#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H"
#endif
#include "Particles/Deposition/TemperatureDeposition.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/ParticleCreation/DefaultInitialization.H"
#include "Particles/Pusher/CopyParticleAttribs.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/PushSelector.H"
#include "Particles/Pusher/UpdateMomentumBoris.H"
#include "Particles/Pusher/UpdateMomentumBorisWithRadiationReaction.H"
#include "Particles/Pusher/UpdateMomentumHigueraCary.H"
#include "Particles/Pusher/UpdateMomentumVay.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/Physics/IonizationEnergiesTable.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "EmbeddedBoundary/Enabled.H"
#ifdef AMREX_USE_EB
#   include "EmbeddedBoundary/ParticleBoundaryProcess.H"
#   include "EmbeddedBoundary/ParticleScraper.H"
#endif
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuBuffer.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuElixir.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParGDB.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleContainerBase.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_SPACE.H>
#include <AMReX_Scan.H>
#include <AMReX_StructOfArrays.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>
#include <AMReX_Parser.H>

#ifdef AMREX_USE_OMP
#   include <omp.h>
#endif

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace amrex;

PhysicalParticleContainer::PhysicalParticleContainer (AmrCore* amr_core, int ispecies,
                                                      const std::string& name)
    : WarpXParticleContainer(amr_core, ispecies),
      species_name(name)
{
    BackwardCompatibility();

    const ParmParse pp_species_name(species_name);

    std::string injection_style = "none";
    pp_species_name.query("injection_style", injection_style);
    if (injection_style != "none") {
        // The base plasma injector, whose input parameters have no source prefix.
        // Only created if needed
        plasma_injectors.push_back(std::make_unique<PlasmaInjector>(species_id, species_name, amr_core->Geom(0)));
    }

    std::vector<std::string> injection_sources;
    pp_species_name.queryarr("injection_sources", injection_sources);
    for (auto &source_name : injection_sources) {
        plasma_injectors.push_back(std::make_unique<PlasmaInjector>(species_id, species_name, amr_core->Geom(0),
                                                                    source_name));
    }

    // Setup the charge and mass. There are multiple ways that they can be specified, so checks are needed to
    // ensure that a value is specified and warnings given if multiple values are specified.
    // The ordering is that species.charge and species.mass take precedence over all other values.
    // Next is charge and mass determined from species_type.
    // Last is charge and mass from the plasma injector setup
    bool charge_from_source = false;
    bool mass_from_source = false;
    for (auto const& plasma_injector : plasma_injectors) {
        // For now, use the last value for charge and mass that is found.
        // A check could be added for consistency of multiple values, but it'll probably never be needed
        charge_from_source |= plasma_injector->queryCharge(charge);
        mass_from_source |= plasma_injector->queryMass(mass);
    }

    std::string physical_species_s;
    const bool species_is_specified = pp_species_name.query("species_type", physical_species_s);
    if (species_is_specified) {
        const auto physical_species_from_string = species::from_string( physical_species_s );
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(physical_species_from_string,
            physical_species_s + " does not exist!");
        physical_species = physical_species_from_string.value();
        charge = species::get_charge( physical_species );
        mass = species::get_mass( physical_species );
    }

    // parse charge and mass (overriding values above)
    const bool charge_is_specified = utils::parser::queryWithParser(pp_species_name, "charge", charge);
    const bool mass_is_specified = utils::parser::queryWithParser(pp_species_name, "mass", mass);

    if (charge_is_specified && species_is_specified) {
        ablastr::warn_manager::WMRecordWarning("Species",
            "Both '" + species_name +  ".charge' and " +
                species_name + ".species_type' are specified.\n" +
                species_name + ".charge' will take precedence.\n");
    }
    if (mass_is_specified && species_is_specified) {
        ablastr::warn_manager::WMRecordWarning("Species",
            "Both '" + species_name +  ".mass' and " +
                species_name + ".species_type' are specified.\n" +
                species_name + ".mass' will take precedence.\n");
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        charge_from_source ||
        charge_is_specified ||
        species_is_specified,
        "Need to specify at least one of species_type or charge for species '" +
        species_name + "'."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mass_from_source ||
        mass_is_specified ||
        species_is_specified,
        "Need to specify at least one of species_type or mass for species '" +
        species_name + "'."
    );

    utils::parser::queryWithParser(pp_species_name, "do_temperature_deposition", m_do_temperature_deposition);

    pp_species_name.query("boost_adjust_transverse_positions", boost_adjust_transverse_positions);
    pp_species_name.query("do_backward_propagation", do_backward_propagation);
    pp_species_name.query("random_theta", m_rz_random_theta);

    // Initialize splitting
    pp_species_name.query("do_splitting", do_splitting);
    pp_species_name.query("split_type", split_type);
    pp_species_name.query("do_not_deposit", do_not_deposit);
    pp_species_name.query("do_not_gather", do_not_gather);
    pp_species_name.query("do_not_push", do_not_push);

    pp_species_name.query("do_continuous_injection", do_continuous_injection);
    pp_species_name.query("initialize_self_fields", initialize_self_fields);
    utils::parser::queryWithParser(
        pp_species_name, "self_fields_required_precision", self_fields_required_precision);
    utils::parser::queryWithParser(
        pp_species_name, "self_fields_absolute_tolerance", self_fields_absolute_tolerance);
    utils::parser::queryWithParser(
        pp_species_name, "self_fields_max_iters", self_fields_max_iters);
    pp_species_name.query("self_fields_verbosity", self_fields_verbosity);

    pp_species_name.query("do_field_ionization", do_field_ionization);

    pp_species_name.query("do_resampling", do_resampling);
    if (do_resampling) { m_resampler = Resampling(species_name); }

    //check if Radiation Reaction is enabled and do consistency checks
    pp_species_name.query("do_classical_radiation_reaction", do_classical_radiation_reaction);
    //if the species is not a lepton, do_classical_radiation_reaction
    //should be false
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!do_classical_radiation_reaction) ||
        AmIA<PhysicalSpecies::electron>() ||
        AmIA<PhysicalSpecies::positron>(),
        "can't enable classical radiation reaction for non lepton species '"
            + species_name + "'.");

    //Only Boris pusher is compatible with radiation reaction
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!do_classical_radiation_reaction) ||
        WarpX::particle_pusher_algo == ParticlePusherAlgo::Boris,
        "Radiation reaction can be enabled only if Boris pusher is used");
    //_____________________________

#ifdef WARPX_QED
    pp_species_name.query("do_qed_quantum_sync", m_do_qed_quantum_sync);
    if (m_do_qed_quantum_sync) {
        AddRealComp("opticalDepthQSR");
    }

    pp_species_name.query("do_qed_breit_wheeler", m_do_qed_breit_wheeler);
    if (m_do_qed_breit_wheeler) {
        AddRealComp("opticalDepthBW");
    }

    if(m_do_qed_quantum_sync){
        pp_species_name.get("qed_quantum_sync_phot_product_species",
            m_qed_quantum_sync_phot_product_name);
    }
#endif

    // User-defined integer attributes
    pp_species_name.queryarr("addIntegerAttributes", m_user_int_attribs);
    const auto n_user_int_attribs = static_cast<int>(m_user_int_attribs.size());
    std::vector< std::string > str_int_attrib_function;
    str_int_attrib_function.resize(n_user_int_attribs);
    m_user_int_attrib_parser.resize(n_user_int_attribs);
    for (int i = 0; i < n_user_int_attribs; ++i) {
        utils::parser::Store_parserString(
            pp_species_name, "attribute."+m_user_int_attribs.at(i)+"(x,y,z,ux,uy,uz,t)",
            str_int_attrib_function.at(i));
        m_user_int_attrib_parser.at(i) = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_int_attrib_function.at(i),{"x","y","z","ux","uy","uz","t"}));
        AddIntComp(m_user_int_attribs.at(i));
    }

    // User-defined real attributes
    pp_species_name.queryarr("addRealAttributes", m_user_real_attribs);
    const auto n_user_real_attribs = static_cast<int>(m_user_real_attribs.size());
    std::vector< std::string > str_real_attrib_function;
    str_real_attrib_function.resize(n_user_real_attribs);
    m_user_real_attrib_parser.resize(n_user_real_attribs);
    for (int i = 0; i < n_user_real_attribs; ++i) {
        utils::parser::Store_parserString(
            pp_species_name, "attribute."+m_user_real_attribs.at(i)+"(x,y,z,ux,uy,uz,t)",
            str_real_attrib_function.at(i));
        m_user_real_attrib_parser.at(i) = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_real_attrib_function.at(i),{"x","y","z","ux","uy","uz","t"}));
        AddRealComp(m_user_real_attribs.at(i));
    }

    // If old particle positions should be saved add the needed components
    pp_species_name.query("save_previous_position", m_save_previous_position);
    if (m_save_previous_position) {
#if !defined(WARPX_DIM_1D_Z)
        AddRealComp("prev_x");
#endif
#if defined(WARPX_DIM_3D)
        AddRealComp("prev_y");
#endif
#if defined(WARPX_ZINDEX)
        AddRealComp("prev_z");
#endif
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
      amrex::Abort("Saving previous particle positions not yet implemented in RZ");
#endif
    }

    // Read reflection models for absorbing boundaries; defaults to a zero
    pp_species_name.query("reflection_model_xlo(E)", m_boundary_conditions.reflection_model_xlo_str);
    pp_species_name.query("reflection_model_xhi(E)", m_boundary_conditions.reflection_model_xhi_str);
    pp_species_name.query("reflection_model_ylo(E)", m_boundary_conditions.reflection_model_ylo_str);
    pp_species_name.query("reflection_model_yhi(E)", m_boundary_conditions.reflection_model_yhi_str);
    pp_species_name.query("reflection_model_zlo(E)", m_boundary_conditions.reflection_model_zlo_str);
    pp_species_name.query("reflection_model_zhi(E)", m_boundary_conditions.reflection_model_zhi_str);
    m_boundary_conditions.BuildReflectionModelParsers();

    const ParmParse pp_boundary("boundary");
    bool flag = false;
    pp_boundary.query("reflect_all_velocities", flag);
    m_boundary_conditions.Set_reflect_all_velocities(flag);

    // currently supports only isotropic thermal distribution
    // same distribution is applied to all boundaries
    const amrex::ParmParse pp_species_boundary("boundary." + species_name);
    if (WarpX::isAnyParticleBoundaryThermal()) {
        amrex::Real boundary_uth = 0;
        utils::parser::getWithParser(pp_species_boundary,"u_th",boundary_uth);
        m_boundary_conditions.SetThermalVelocity(boundary_uth);
    }
}

void
PhysicalParticleContainer::AllocData ()
{
    // Call Base class Data allocation
    WarpXParticleContainer::AllocData();

    if (m_do_temperature_deposition) {
        using ablastr::fields::Direction;

        auto& warpx = WarpX::GetInstance();
        ablastr::fields::MultiLevelVectorField J_vf =
            warpx.m_fields.get_mr_levels_alldirs(warpx::fields::FieldType::current_fp, warpx.finestLevel());

        const std::string T_field_name = "T_" + species_name;

        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            for (int idir = 0; idir < 3; ++idir) {
                amrex::BoxArray const& ba = J_vf[lev][Direction{idir}]->boxArray();
                amrex::DistributionMapping const& dm = J_vf[lev][Direction{idir}]->DistributionMap();
                amrex::IntVect const& ng = J_vf[lev][Direction{idir}]->nGrowVect();

                warpx.m_fields.alloc_init(T_field_name, Direction{idir},
                    lev, ba, dm, WarpX::ncomps, ng, 0.0_rt);
            }
        }

        ablastr::fields::MultiLevelVectorField T_vf =
            warpx.m_fields.get_mr_levels_alldirs(T_field_name, warpx.finestLevel());

        // Allocate Accumulation Arrays
        local_temperature_arrays = std::make_unique<warpx::particles::deposition::VarianceAccumulationBuffer>(
            T_vf, species_name);
    }
}

PhysicalParticleContainer::PhysicalParticleContainer (AmrCore* amr_core)
    : WarpXParticleContainer(amr_core, 0)
{
}

void
PhysicalParticleContainer::BackwardCompatibility ()
{
    const ParmParse pp_species_name(species_name);
    std::vector<std::string> backward_strings;
    if (pp_species_name.queryarr("plot_vars", backward_strings)){
        WARPX_ABORT_WITH_MESSAGE("<species>.plot_vars is not supported anymore. "
                     "Please use the new syntax for diagnostics, see documentation.");
    }

    int backward_int;
    if (pp_species_name.query("plot_species", backward_int)){
        WARPX_ABORT_WITH_MESSAGE("<species>.plot_species is not supported anymore. "
                     "Please use the new syntax for diagnostics, see documentation.");
    }
}

void PhysicalParticleContainer::InitData ()
{
    AddParticles(0); // Note - add on level 0
    Redistribute();  // We then redistribute
}

void
PhysicalParticleContainer::DefaultInitializeRuntimeAttributes (
    typename ContainerLike<amrex::PinnedArenaAllocator>::ParticleTileType& pinned_tile,
    int n_external_attr_real,
    int n_external_attr_int)
{
    ParticleCreation::DefaultInitializeRuntimeAttributes(pinned_tile,
                                       n_external_attr_real, n_external_attr_int,
                                       m_user_real_attribs, m_user_int_attribs,
                                       GetRealSoANames(), GetIntSoANames(),
                                       amrex::GetVecOfPtrs(m_user_real_attrib_parser),
                                       amrex::GetVecOfPtrs(m_user_int_attrib_parser),
#ifdef WARPX_QED
                                       true,
                                       m_shr_p_bw_engine.get(),
                                       m_shr_p_qs_engine.get(),
#endif
                                       ionization_initial_level,
                                       0,pinned_tile.numParticles());
}

void
PhysicalParticleContainer::Evolve (ablastr::fields::MultiFabRegister& fields,
                                   int lev,
                                   const std::string& current_fp_string,
                                   Real /*t*/, Real dt, DtType a_dt_type, bool skip_deposition,
                                   ImplicitOptions const * implicit_options)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    WARPX_PROFILE("PhysicalParticleContainer::Evolve()");
    WARPX_PROFILE_VAR_NS("PhysicalParticleContainer::Evolve::GatherAndPush", blp_fg);

    BL_ASSERT(OnSameGrids(lev, *fields.get(FieldType::current_fp, Direction{0}, lev)));

    const PushType push_type = (implicit_options == nullptr) ? PushType::Explicit : PushType::Implicit;

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    const iMultiFab* current_masks = WarpX::CurrentBufferMasks(lev);
    const iMultiFab* gather_masks = WarpX::GatherBufferMasks(lev);

    const bool has_rho = fields.has(FieldType::rho_fp, lev);
    const bool has_J_buf = fields.has_vector(FieldType::current_buf, lev);
    const bool has_E_cax = fields.has_vector(FieldType::Efield_cax, lev);
    const bool has_buffer = has_E_cax || has_J_buf;

    amrex::MultiFab & Ex = *fields.get(FieldType::Efield_aux, Direction{0}, lev);
    amrex::MultiFab & Ey = *fields.get(FieldType::Efield_aux, Direction{1}, lev);
    amrex::MultiFab & Ez = *fields.get(FieldType::Efield_aux, Direction{2}, lev);
    amrex::MultiFab & Bx = *fields.get(FieldType::Bfield_aux, Direction{0}, lev);
    amrex::MultiFab & By = *fields.get(FieldType::Bfield_aux, Direction{1}, lev);
    amrex::MultiFab & Bz = *fields.get(FieldType::Bfield_aux, Direction{2}, lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
#ifdef AMREX_USE_OMP
        const int thread_num = omp_get_thread_num();
#else
        const int thread_num = 0;
#endif

        FArrayBox filtered_Ex, filtered_Ey, filtered_Ez;
        FArrayBox filtered_Bx, filtered_By, filtered_Bz;

        for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
        {
            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            const Box& box = pti.validbox();

            // Extract particle data
            auto& attribs = pti.GetAttribs();
            auto&  wp = attribs[PIdx::w];
            auto& uxp = attribs[PIdx::ux];
            auto& uyp = attribs[PIdx::uy];
            auto& uzp = attribs[PIdx::uz];

            const long np = pti.numParticles();

            // Data on the grid
            FArrayBox const* exfab = &Ex[pti];
            FArrayBox const* eyfab = &Ey[pti];
            FArrayBox const* ezfab = &Ez[pti];
            FArrayBox const* bxfab = &Bx[pti];
            FArrayBox const* byfab = &By[pti];
            FArrayBox const* bzfab = &Bz[pti];

            Elixir exeli, eyeli, ezeli, bxeli, byeli, bzeli;

            if (WarpX::use_fdtd_nci_corr)
            {
                // Filter arrays Ex[pti], store the result in
                // filtered_Ex and update pointer exfab so that it
                // points to filtered_Ex (and do the same for all
                // components of E and B).
                applyNCIFilter(lev, pti.tilebox(), exeli, eyeli, ezeli, bxeli, byeli, bzeli,
                               filtered_Ex, filtered_Ey, filtered_Ez,
                               filtered_Bx, filtered_By, filtered_Bz,
                               Ex[pti], Ey[pti], Ez[pti], Bx[pti], By[pti], Bz[pti],
                               exfab, eyfab, ezfab, bxfab, byfab, bzfab);
            }

            // Determine which particles deposit/gather in the buffer, and
            // which particles deposit/gather in the fine patch
            long nfine_deposit = np;
            long nfine_gather = np;
            if (has_buffer && !do_not_push) {
                // - Modify `nfine_deposit` and `nfine_gather` (in place)
                //    so that they correspond to the number of particles
                //    that deposit/gather in the fine patch respectively.
                // - Reorder the particle arrays,
                //    so that the `nfine_deposit`/`nfine_gather` first particles
                //    deposit/gather in the fine patch
                //    and (thus) the `np-nfine_deposit`/`np-nfine_gather` last particles
                //    deposit/gather in the buffer
                PartitionParticlesInBuffers( nfine_deposit, nfine_gather, np,
                    pti, lev, WarpX::n_field_gather_buffer,
                    WarpX::n_current_deposition_buffer, current_masks, gather_masks );
            }

            const long np_to_deposit = has_J_buf ? nfine_deposit : np;

            if (has_rho && ! skip_deposition && ! do_not_deposit) {
                // Deposit charge before particle push, in component 0 of MultiFab rho.

                const int* const AMREX_RESTRICT ion_lev = (do_field_ionization)?
                    pti.GetiAttribs("ionizationLevel").dataPtr():nullptr;

                amrex::MultiFab* rho = fields.get(FieldType::rho_fp, lev);
                DepositCharge(pti, wp, ion_lev, rho, 0, 0,
                              np_to_deposit, thread_num, lev, lev);
                if (has_buffer){
                    amrex::MultiFab* crho = fields.get(FieldType::rho_buf, lev);
                    DepositCharge(pti, wp, ion_lev, crho, 0, np_to_deposit,
                                  np-np_to_deposit, thread_num, lev, lev-1);
                }
            }

            if (! do_not_push)
            {
                const long np_gather = has_E_cax ? nfine_gather : np;

                int e_is_nodal = Ex.is_nodal() and Ey.is_nodal() and Ez.is_nodal();

                // Temporary data used in the implicit advance
                amrex::Gpu::DeviceVector<long> unconverged_indices;
                amrex::Gpu::DeviceVector<amrex::ParticleReal> saved_weights;
                long num_unconverged_particles = 0;
                long num_unconverged_particles_c = 0;

                //
                // Gather and push for particles not in the buffer
                //
                WARPX_PROFILE_VAR_START(blp_fg);
                const auto np_to_push = np_gather;
                const auto gather_lev = lev;
                if (push_type == PushType::Explicit) {
                    PushPX(pti, exfab, eyfab, ezfab,
                           bxfab, byfab, bzfab,
                           Ex.nGrowVect(), e_is_nodal,
                           0, np_to_push, lev, gather_lev, dt, ScaleFields(false), a_dt_type);
                } else if (push_type == PushType::Implicit) {
                    long const offset = 0;
                    ImplicitPushXP(pti, exfab, eyfab, ezfab,
                                   bxfab, byfab, bzfab,
                                   implicit_options,
                                   Ex.nGrowVect(),
                                   offset, np_to_push, lev, gather_lev, dt,
                                   num_unconverged_particles, unconverged_indices, saved_weights);
                }

                if (np_gather < np)
                {
                    const IntVect& ref_ratio = WarpX::RefRatio(lev-1);
                    const Box& cbox = amrex::coarsen(box,ref_ratio);

                    amrex::MultiFab & cEx = *fields.get(FieldType::Efield_cax, Direction{0}, lev);
                    amrex::MultiFab & cEy = *fields.get(FieldType::Efield_cax, Direction{1}, lev);
                    amrex::MultiFab & cEz = *fields.get(FieldType::Efield_cax, Direction{2}, lev);
                    amrex::MultiFab & cBx = *fields.get(FieldType::Bfield_cax, Direction{0}, lev);
                    amrex::MultiFab & cBy = *fields.get(FieldType::Bfield_cax, Direction{1}, lev);
                    amrex::MultiFab & cBz = *fields.get(FieldType::Bfield_cax, Direction{2}, lev);

                    // Data on the grid
                    FArrayBox const* cexfab = &cEx[pti];
                    FArrayBox const* ceyfab = &cEy[pti];
                    FArrayBox const* cezfab = &cEz[pti];
                    FArrayBox const* cbxfab = &cBx[pti];
                    FArrayBox const* cbyfab = &cBy[pti];
                    FArrayBox const* cbzfab = &cBz[pti];

                    if (WarpX::use_fdtd_nci_corr)
                    {
                        // Filter arrays (*cEx)[pti], store the result in
                        // filtered_Ex and update pointer cexfab so that it
                        // points to filtered_Ex (and do the same for all
                        // components of E and B)
                        applyNCIFilter(lev-1, cbox, exeli, eyeli, ezeli, bxeli, byeli, bzeli,
                                       filtered_Ex, filtered_Ey, filtered_Ez,
                                       filtered_Bx, filtered_By, filtered_Bz,
                                       cEx[pti], cEy[pti], cEz[pti],
                                       cBx[pti], cBy[pti], cBz[pti],
                                       cexfab, ceyfab, cezfab, cbxfab, cbyfab, cbzfab);
                    }

                    // Field gather and push for particles in gather buffers
                    e_is_nodal = cEx.is_nodal() and cEy.is_nodal() and cEz.is_nodal();
                    if (push_type == PushType::Explicit) {
                        PushPX(pti, cexfab, ceyfab, cezfab,
                               cbxfab, cbyfab, cbzfab,
                               cEx.nGrowVect(), e_is_nodal,
                               nfine_gather, np-nfine_gather,
                               lev, lev-1, dt, ScaleFields(false), a_dt_type);
                    } else if (push_type == PushType::Implicit) {
                        ImplicitPushXP(pti, cexfab, ceyfab, cezfab,
                                       cbxfab, cbyfab, cbzfab,
                                       implicit_options,
                                       cEx.nGrowVect(),
                                       nfine_gather, np-nfine_gather,
                                       lev, lev-1, dt,
                                       num_unconverged_particles_c, unconverged_indices, saved_weights);
                    }
                }

                WARPX_PROFILE_VAR_STOP(blp_fg);

                // Current Deposition
                if (!skip_deposition)
                {
                    // Deposit at t_{n+1/2} with explicit push
                    const amrex::Real relative_time = (push_type == PushType::Explicit ? -0.5_rt * dt : 0.0_rt);

                    const int* const AMREX_RESTRICT ion_lev = (do_field_ionization)?
                        pti.GetiAttribs("ionizationLevel").dataPtr():nullptr;

                    // Deposit inside domains
                    amrex::MultiFab * jx = fields.get(current_fp_string, Direction{0}, lev);
                    amrex::MultiFab * jy = fields.get(current_fp_string, Direction{1}, lev);
                    amrex::MultiFab * jz = fields.get(current_fp_string, Direction{2}, lev);
                    if (implicit_options && implicit_options->deposit_mass_matrices) {
                        amrex::MultiFab * Sxx = fields.get(FieldType::MassMatrices_X, Direction{0}, lev);
                        amrex::MultiFab * Sxy = fields.get(FieldType::MassMatrices_X, Direction{1}, lev);
                        amrex::MultiFab * Sxz = fields.get(FieldType::MassMatrices_X, Direction{2}, lev);
                        amrex::MultiFab * Syx = fields.get(FieldType::MassMatrices_Y, Direction{0}, lev);
                        amrex::MultiFab * Syy = fields.get(FieldType::MassMatrices_Y, Direction{1}, lev);
                        amrex::MultiFab * Syz = fields.get(FieldType::MassMatrices_Y, Direction{2}, lev);
                        amrex::MultiFab * Szx = fields.get(FieldType::MassMatrices_Z, Direction{0}, lev);
                        amrex::MultiFab * Szy = fields.get(FieldType::MassMatrices_Z, Direction{1}, lev);
                        amrex::MultiFab * Szz = fields.get(FieldType::MassMatrices_Z, Direction{2}, lev);
                        DepositCurrentAndMassMatrices(pti, wp, uxp, uyp, uzp, jx, jy, jz,
                                       Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz,
                                       bxfab, byfab, bzfab, 0, np_to_deposit, thread_num, lev, lev, dt);
                    }
                    else {
                        DepositCurrent(pti, wp, uxp, uyp, uzp, ion_lev, jx, jy, jz,
                                       0, np_to_deposit, thread_num,
                                       lev, lev, dt, relative_time, push_type);
                    }
                    if (has_buffer)
                    {
                        // Deposit in buffers
                        amrex::MultiFab * cjx = fields.get(FieldType::current_buf, Direction{0}, lev);
                        amrex::MultiFab * cjy = fields.get(FieldType::current_buf, Direction{1}, lev);
                        amrex::MultiFab * cjz = fields.get(FieldType::current_buf, Direction{2}, lev);
                        DepositCurrent(pti, wp, uxp, uyp, uzp, ion_lev, cjx, cjy, cjz,
                                       np_to_deposit, np-np_to_deposit, thread_num,
                                       lev, lev-1, dt, relative_time, push_type);
                    }
                } // end of "if skip_deposition"

                if (push_type == PushType::Implicit) {
                    if (num_unconverged_particles > 0) {
                        amrex::MultiFab * jx = fields.get(current_fp_string, Direction{0}, lev);
                        amrex::MultiFab * jy = fields.get(current_fp_string, Direction{1}, lev);
                        amrex::MultiFab * jz = fields.get(current_fp_string, Direction{2}, lev);
                        long const offset = 0;
                        ImplicitPushXPSubOrbits(pti, fields, exfab, eyfab, ezfab,
                                                bxfab, byfab, bzfab,
                                                implicit_options,
                                                Ex.nGrowVect(),
                                                jx, jy, jz,
                                                offset, lev, gather_lev, dt, skip_deposition,
                                                num_unconverged_particles, unconverged_indices, saved_weights);
                    }
                    if (num_unconverged_particles_c > 0) {

                        amrex::MultiFab & cEx = *fields.get(FieldType::Efield_cax, Direction{0}, lev);
                        amrex::MultiFab & cEy = *fields.get(FieldType::Efield_cax, Direction{1}, lev);
                        amrex::MultiFab & cEz = *fields.get(FieldType::Efield_cax, Direction{2}, lev);
                        amrex::MultiFab & cBx = *fields.get(FieldType::Bfield_cax, Direction{0}, lev);
                        amrex::MultiFab & cBy = *fields.get(FieldType::Bfield_cax, Direction{1}, lev);
                        amrex::MultiFab & cBz = *fields.get(FieldType::Bfield_cax, Direction{2}, lev);

                        // Data on the grid
                        FArrayBox const* cexfab = &cEx[pti];
                        FArrayBox const* ceyfab = &cEy[pti];
                        FArrayBox const* cezfab = &cEz[pti];
                        FArrayBox const* cbxfab = &cBx[pti];
                        FArrayBox const* cbyfab = &cBy[pti];
                        FArrayBox const* cbzfab = &cBz[pti];

                        amrex::MultiFab * cjx = fields.get(FieldType::current_buf, Direction{0}, lev);
                        amrex::MultiFab * cjy = fields.get(FieldType::current_buf, Direction{1}, lev);
                        amrex::MultiFab * cjz = fields.get(FieldType::current_buf, Direction{2}, lev);

                        long const offset = num_unconverged_particles;
                        ImplicitPushXPSubOrbits(pti, fields, cexfab, ceyfab, cezfab,
                                                cbxfab, cbyfab, cbzfab,
                                                implicit_options,
                                                cEx.nGrowVect(),
                                                cjx, cjy, cjz,
                                                offset, lev, lev-1, dt, skip_deposition,
                                                num_unconverged_particles_c, unconverged_indices, saved_weights);
                    }
                }

            } // end of "if do_not_push"

            if (has_rho && ! skip_deposition && ! do_not_deposit) {
                // Deposit charge after particle push, in component 1 of MultiFab rho.
                // (Skipped for electrostatic solver, as this may lead to out-of-bounds)
                if (WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::None) {
                    amrex::MultiFab* rho = fields.get(FieldType::rho_fp, lev);
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho->nComp() >= 2,
                        "Cannot deposit charge in rho component 1: only component 0 is allocated!");

                    const int* const AMREX_RESTRICT ion_lev = (do_field_ionization)?
                        pti.GetiAttribs("ionizationLevel").dataPtr():nullptr;

                    DepositCharge(pti, wp, ion_lev, rho, 1, 0,
                                  np_to_deposit, thread_num, lev, lev);
                    if (has_buffer){
                        amrex::MultiFab* crho = fields.get(FieldType::rho_buf, lev);
                        DepositCharge(pti, wp, ion_lev, crho, 1, np_to_deposit,
                                      np-np_to_deposit, thread_num, lev, lev-1);
                    }
                }
            }

            amrex::Gpu::synchronize();

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
            }
        }
    }
    // Split particles at the end of the timestep.
    // When subcycling is ON, the splitting is done on the last call to
    // PhysicalParticleContainer::Evolve on the finest level, i.e., at the
    // end of the large timestep. Otherwise, the pushes on different levels
    // are not consistent, and the call to Redistribute (inside
    // SplitParticles) may result in split particles to deposit twice on the
    // coarse level.
    if (do_splitting && (a_dt_type == DtType::SecondHalf || a_dt_type == DtType::Full) ){
        SplitParticles(lev);
    }
}

void
PhysicalParticleContainer::applyNCIFilter (
    int lev, const Box& box,
    Elixir& exeli, Elixir& eyeli, Elixir& ezeli,
    Elixir& bxeli, Elixir& byeli, Elixir& bzeli,
    FArrayBox& filtered_Ex, FArrayBox& filtered_Ey, FArrayBox& filtered_Ez,
    FArrayBox& filtered_Bx, FArrayBox& filtered_By, FArrayBox& filtered_Bz,
    const FArrayBox& Ex, const FArrayBox& Ey, const FArrayBox& Ez,
    const FArrayBox& Bx, const FArrayBox& By, const FArrayBox& Bz,
    FArrayBox const * & ex_ptr, FArrayBox const * & ey_ptr,
    FArrayBox const * & ez_ptr, FArrayBox const * & bx_ptr,
    FArrayBox const * & by_ptr, FArrayBox const * & bz_ptr)
{

    // Get instances of NCI Godfrey filters
    const auto& nci_godfrey_filter_exeybz = WarpX::GetInstance().nci_godfrey_filter_exeybz;
    const auto& nci_godfrey_filter_bxbyez = WarpX::GetInstance().nci_godfrey_filter_bxbyez;

#if defined(WARPX_DIM_1D_Z)
    const Box& tbox = amrex::grow(box, static_cast<int>(WarpX::noz));
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const Box& tbox = amrex::grow(box, static_cast<int>(WarpX::nox));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const Box& tbox = amrex::grow(box, {static_cast<int>(WarpX::nox),
                static_cast<int>(WarpX::noz)});
#else
    const Box& tbox = amrex::grow(box, {static_cast<int>(WarpX::nox),
                static_cast<int>(WarpX::noy),
                static_cast<int>(WarpX::noz)});
#endif

    // Filter Ex (Both 2D and 3D)
    filtered_Ex.resize(amrex::convert(tbox,Ex.box().ixType()));
    // Safeguard for GPU
    exeli = filtered_Ex.elixir();
    // Apply filter on Ex, result stored in filtered_Ex

    nci_godfrey_filter_exeybz[lev]->ApplyStencil(filtered_Ex, Ex, filtered_Ex.box());
    // Update ex_ptr reference
    ex_ptr = &filtered_Ex;

    // Filter Ez
    filtered_Ez.resize(amrex::convert(tbox,Ez.box().ixType()));
    ezeli = filtered_Ez.elixir();
    nci_godfrey_filter_bxbyez[lev]->ApplyStencil(filtered_Ez, Ez, filtered_Ez.box());
    ez_ptr = &filtered_Ez;

    // Filter By
    filtered_By.resize(amrex::convert(tbox,By.box().ixType()));
    byeli = filtered_By.elixir();
    nci_godfrey_filter_bxbyez[lev]->ApplyStencil(filtered_By, By, filtered_By.box());
    by_ptr = &filtered_By;
#if defined(WARPX_DIM_3D)
    // Filter Ey
    filtered_Ey.resize(amrex::convert(tbox,Ey.box().ixType()));
    eyeli = filtered_Ey.elixir();
    nci_godfrey_filter_exeybz[lev]->ApplyStencil(filtered_Ey, Ey, filtered_Ey.box());
    ey_ptr = &filtered_Ey;

    // Filter Bx
    filtered_Bx.resize(amrex::convert(tbox,Bx.box().ixType()));
    bxeli = filtered_Bx.elixir();
    nci_godfrey_filter_bxbyez[lev]->ApplyStencil(filtered_Bx, Bx, filtered_Bx.box());
    bx_ptr = &filtered_Bx;

    // Filter Bz
    filtered_Bz.resize(amrex::convert(tbox,Bz.box().ixType()));
    bzeli = filtered_Bz.elixir();
    nci_godfrey_filter_exeybz[lev]->ApplyStencil(filtered_Bz, Bz, filtered_Bz.box());
    bz_ptr = &filtered_Bz;
#else
    amrex::ignore_unused(eyeli, bxeli, bzeli,
        filtered_Ey, filtered_Bx, filtered_Bz,
        Ey, Bx, Bz, ey_ptr, bx_ptr, bz_ptr);
#endif
}

// Loop over all particles in the particle container and
// split particles tagged with p.id()=DoSplitParticleID
void
PhysicalParticleContainer::SplitParticles (int lev)
{
    PhysicalParticleContainer pctmp_split(&WarpX::GetInstance());
    RealVector psplit_x, psplit_y, psplit_z, psplit_w;
    RealVector psplit_ux, psplit_uy, psplit_uz;
    long np_split_to_add = 0;
    long np_split;
    if(split_type==0)
    {
        np_split = amrex::Math::powi<AMREX_SPACEDIM>(2);
    } else {
        np_split = 2*AMREX_SPACEDIM;
    }

    // Loop over particle interator
    for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        const auto GetPosition = GetParticlePosition<PIdx>(pti);

        const amrex::Vector<int> ppc_nd = plasma_injectors[0]->num_particles_per_cell_each_dim;
        const std::array<Real,3>& dx = WarpX::CellSize(lev);
        amrex::Vector<Real> split_offset = {dx[0]/2._rt,
                                            dx[1]/2._rt,
                                            dx[2]/2._rt};
        if (ppc_nd[0] > 0){
            // offset for split particles is computed as a function of cell size
            // and number of particles per cell, so that a uniform distribution
            // before splitting results in a uniform distribution after splitting
            split_offset[0] /= ppc_nd[0];
            split_offset[1] /= ppc_nd[1];
            split_offset[2] /= ppc_nd[2];
        }
        // particle Struct Of Arrays data
        auto& attribs = pti.GetAttribs();
        auto& wp  = attribs[PIdx::w ];
        auto& uxp = attribs[PIdx::ux];
        auto& uyp = attribs[PIdx::uy];
        auto& uzp = attribs[PIdx::uz];

        ParticleTileType& ptile = ParticlesAt(lev, pti);
        auto& soa = ptile.GetStructOfArrays();
        uint64_t * const AMREX_RESTRICT idcpu = soa.GetIdCPUData().data();

        const long np = pti.numParticles();
        for(int i=0; i<np; i++){
            ParticleReal xp, yp, zp;
            GetPosition(i, xp, yp, zp);
            if (idcpu[i] == LongParticleIds::DoSplitParticleID){
                // If particle is tagged, split it and put the
                // split particles in local arrays psplit_x etc.
                np_split_to_add += np_split;
#if defined(WARPX_DIM_1D_Z)
                // Split particle in two along z axis
                // 2 particles in 1d, split_type doesn't matter? Discuss with Remi
                for (int ishift = -1; ishift < 2; ishift +=2 ){
                    // Add one particle with offset in z
                    psplit_x.push_back( xp );
                    psplit_y.push_back( yp );
                    psplit_z.push_back( zp + ishift*split_offset[2] );
                    psplit_ux.push_back( uxp[i] );
                    psplit_uy.push_back( uyp[i] );
                    psplit_uz.push_back( uzp[i] );
                    psplit_w.push_back( wp[i]/np_split );
                }
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                // Split particle in two along x axis
                // 2 particles in 1d, split_type doesn't matter? Discuss with Remi
                for (int ishift = -1; ishift < 2; ishift +=2 ){
                    // Add one particle with offset in x
                    psplit_x.push_back( xp + ishift*split_offset[0] );
                    psplit_y.push_back( yp );
                    psplit_x.push_back( zp );
                    psplit_ux.push_back( uxp[i] );
                    psplit_uy.push_back( uyp[i] );
                    psplit_uz.push_back( uzp[i] );
                    psplit_w.push_back( wp[i]/np_split );
                }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                if (split_type==0){
                    // Split particle in two along each diagonals
                    // 4 particles in 2d
                    for (int ishift = -1; ishift < 2; ishift +=2 ){
                        for (int kshift = -1; kshift < 2; kshift +=2 ){
                            // Add one particle with offset in x and z
                            psplit_x.push_back( xp + ishift*split_offset[0] );
                            psplit_y.push_back( yp );
                            psplit_z.push_back( zp + kshift*split_offset[2] );
                            psplit_ux.push_back( uxp[i] );
                            psplit_uy.push_back( uyp[i] );
                            psplit_uz.push_back( uzp[i] );
                            psplit_w.push_back( wp[i]/np_split );
                        }
                    }
                } else {
                    // Split particle in two along each axis
                    // 4 particles in 2d
                    for (int ishift = -1; ishift < 2; ishift +=2 ){
                        // Add one particle with offset in x
                        psplit_x.push_back( xp + ishift*split_offset[0] );
                        psplit_y.push_back( yp );
                        psplit_z.push_back( zp );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                        // Add one particle with offset in z
                        psplit_x.push_back( xp );
                        psplit_y.push_back( yp );
                        psplit_z.push_back( zp + ishift*split_offset[2] );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                    }
                }
#elif defined(WARPX_DIM_3D)
                if (split_type==0){
                    // Split particle in two along each diagonals
                    // 8 particles in 3d
                    for (int ishift = -1; ishift < 2; ishift +=2 ){
                        for (int jshift = -1; jshift < 2; jshift +=2 ){
                            for (int kshift = -1; kshift < 2; kshift +=2 ){
                                // Add one particle with offset in x, y and z
                                psplit_x.push_back( xp + ishift*split_offset[0] );
                                psplit_y.push_back( yp + jshift*split_offset[1] );
                                psplit_z.push_back( zp + kshift*split_offset[2] );
                                psplit_ux.push_back( uxp[i] );
                                psplit_uy.push_back( uyp[i] );
                                psplit_uz.push_back( uzp[i] );
                                psplit_w.push_back( wp[i]/np_split );
                            }
                        }
                    }
                } else {
                    // Split particle in two along each axis
                    // 6 particles in 3d
                    for (int ishift = -1; ishift < 2; ishift +=2 ){
                        // Add one particle with offset in x
                        psplit_x.push_back( xp + ishift*split_offset[0] );
                        psplit_y.push_back( yp );
                        psplit_z.push_back( zp );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                        // Add one particle with offset in y
                        psplit_x.push_back( xp );
                        psplit_y.push_back( yp + ishift*split_offset[1] );
                        psplit_z.push_back( zp );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                        // Add one particle with offset in z
                        psplit_x.push_back( xp );
                        psplit_y.push_back( yp );
                        psplit_z.push_back( zp + ishift*split_offset[2] );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                    }
                }
#endif
                // invalidate the particle
                idcpu[i] = amrex::ParticleIdCpus::Invalid;
            }
        }
    }
    // Add local arrays psplit_x etc. to the temporary
    // particle container pctmp_split. Split particles
    // are tagged with p.id()=NoSplitParticleID so that
    // they are not re-split when entering a higher level
    // AddNParticles calls Redistribute, so that particles
    // in pctmp_split are in the proper grids and tiles
    const amrex::Vector<ParticleReal> xp(psplit_x.data(), psplit_x.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> yp(psplit_y.data(), psplit_y.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> zp(psplit_z.data(), psplit_z.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> uxp(psplit_ux.data(), psplit_ux.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> uyp(psplit_uy.data(), psplit_uy.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> uzp(psplit_uz.data(), psplit_uz.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> wp(psplit_w.data(), psplit_w.data() + np_split_to_add);

    amrex::Vector<amrex::Vector<ParticleReal>> attr;
    attr.push_back(wp);
    const amrex::Vector<amrex::Vector<int>> attr_int;
    pctmp_split.AddNParticles(lev,
                              np_split_to_add,
                              xp,
                              yp,
                              zp,
                              uxp,
                              uyp,
                              uzp,
                              1,
                              attr,
                              0, attr_int,
                              1, LongParticleIds::NoSplitParticleID);
    // Copy particles from tmp to current particle container
    constexpr bool local_flag = true;
    addParticles(pctmp_split,local_flag);
}

void
PhysicalParticleContainer::PushP (int lev, Real dt,
                                  const MultiFab& Ex, const MultiFab& Ey, const MultiFab& Ez,
                                  const MultiFab& Bx, const MultiFab& By, const MultiFab& Bz)
{
    WARPX_PROFILE("PhysicalParticleContainer::PushP()");

    if (do_not_push) { return; }

    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(lev,0));

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
        {
            amrex::Box box = pti.tilebox();
            box.grow(Ex.nGrowVect());

            const long np = pti.numParticles();

            // Data on the grid
            const FArrayBox& exfab = Ex[pti];
            const FArrayBox& eyfab = Ey[pti];
            const FArrayBox& ezfab = Ez[pti];
            const FArrayBox& bxfab = Bx[pti];
            const FArrayBox& byfab = By[pti];
            const FArrayBox& bzfab = Bz[pti];

            const auto getPosition = GetParticlePosition<PIdx>(pti);

            const auto getExternalEB = GetExternalEBField(pti);

            const amrex::ParticleReal Ex_external_particle = m_E_external_particle[0];
            const amrex::ParticleReal Ey_external_particle = m_E_external_particle[1];
            const amrex::ParticleReal Ez_external_particle = m_E_external_particle[2];
            const amrex::ParticleReal Bx_external_particle = m_B_external_particle[0];
            const amrex::ParticleReal By_external_particle = m_B_external_particle[1];
            const amrex::ParticleReal Bz_external_particle = m_B_external_particle[2];

            const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);

            const Dim3 lo = lbound(box);

            const bool galerkin_interpolation = WarpX::galerkin_interpolation;
            const int nox = WarpX::nox;
            const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;

            amrex::Array4<const amrex::Real> const& ex_arr = exfab.array();
            amrex::Array4<const amrex::Real> const& ey_arr = eyfab.array();
            amrex::Array4<const amrex::Real> const& ez_arr = ezfab.array();
            amrex::Array4<const amrex::Real> const& bx_arr = bxfab.array();
            amrex::Array4<const amrex::Real> const& by_arr = byfab.array();
            amrex::Array4<const amrex::Real> const& bz_arr = bzfab.array();

            amrex::IndexType const ex_type = exfab.box().ixType();
            amrex::IndexType const ey_type = eyfab.box().ixType();
            amrex::IndexType const ez_type = ezfab.box().ixType();
            amrex::IndexType const bx_type = bxfab.box().ixType();
            amrex::IndexType const by_type = byfab.box().ixType();
            amrex::IndexType const bz_type = bzfab.box().ixType();

            auto& attribs = pti.GetAttribs();
            ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
            ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
            ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

            int* AMREX_RESTRICT ion_lev = nullptr;
            if (do_field_ionization) {
                ion_lev = pti.GetiAttribs("ionizationLevel").dataPtr();
            }

            // Loop over the particles and update their momentum
            const amrex::ParticleReal q = this->charge;
            const amrex::ParticleReal m = this-> mass;

            const auto pusher_algo = WarpX::particle_pusher_algo;
            const auto do_crr = do_classical_radiation_reaction;

            const auto t_do_not_gather = do_not_gather;

            enum exteb_flags : int { no_exteb, has_exteb };

            const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;

            amrex::ParallelFor(TypeList<CompileTimeOptions<no_exteb,has_exteb>>{},
                               {exteb_runtime_flag},
                               np, [=] AMREX_GPU_DEVICE (long ip, auto exteb_control)
            {
                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                amrex::ParticleReal Exp = Ex_external_particle;
                amrex::ParticleReal Eyp = Ey_external_particle;
                amrex::ParticleReal Ezp = Ez_external_particle;
                amrex::ParticleReal Bxp = Bx_external_particle;
                amrex::ParticleReal Byp = By_external_particle;
                amrex::ParticleReal Bzp = Bz_external_particle;

                if (!t_do_not_gather){
                    // first gather E and B to the particle positions
                    doGatherShapeN(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                   ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                   ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                   dinv, xyzmin, lo, n_rz_azimuthal_modes,
                                   nox, galerkin_interpolation);
                }

                // Externally applied E and B-field in Cartesian co-ordinates
                [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB;
                if constexpr (exteb_control == has_exteb) {
                    getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
                }

                if (do_crr) {
                    amrex::ParticleReal qp = q;
                    if (ion_lev) { qp *= ion_lev[ip]; }
                    UpdateMomentumBorisWithRadiationReaction(ux[ip], uy[ip], uz[ip],
                                                             Exp, Eyp, Ezp, Bxp,
                                                             Byp, Bzp, qp, m, dt);
                } else if (pusher_algo == ParticlePusherAlgo::Boris) {
                    amrex::ParticleReal qp = q;
                    if (ion_lev) { qp *= ion_lev[ip]; }
                    UpdateMomentumBoris( ux[ip], uy[ip], uz[ip],
                                         Exp, Eyp, Ezp, Bxp,
                                         Byp, Bzp, qp, m, dt);
                } else if (pusher_algo == ParticlePusherAlgo::Vay) {
                    amrex::ParticleReal qp = q;
                    if (ion_lev){ qp *= ion_lev[ip]; }
                    UpdateMomentumVay( ux[ip], uy[ip], uz[ip],
                                       Exp, Eyp, Ezp, Bxp,
                                       Byp, Bzp, qp, m, dt);
                } else if (pusher_algo == ParticlePusherAlgo::HigueraCary) {
                    amrex::ParticleReal qp = q;
                    if (ion_lev){ qp *= ion_lev[ip]; }
                    UpdateMomentumHigueraCary( ux[ip], uy[ip], uz[ip],
                                               Exp, Eyp, Ezp, Bxp,
                                               Byp, Bzp, qp, m, dt);
                } else {
                    amrex::Abort("Unknown particle pusher");
                }
            });
        }
    }
}

/* \brief Perform the field gather and particle push operations in one fused kernel
 *
 */
void
PhysicalParticleContainer::PushPX (WarpXParIter& pti,
                                   amrex::FArrayBox const * exfab,
                                   amrex::FArrayBox const * eyfab,
                                   amrex::FArrayBox const * ezfab,
                                   amrex::FArrayBox const * bxfab,
                                   amrex::FArrayBox const * byfab,
                                   amrex::FArrayBox const * bzfab,
                                   const amrex::IntVect ngEB, const int /*e_is_nodal*/,
                                   const long offset,
                                   const long np_to_push,
                                   int lev, int gather_lev,
                                   amrex::Real dt, ScaleFields scaleFields,
                                   DtType a_dt_type)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((gather_lev==(lev-1)) ||
                                     (gather_lev==(lev  )),
                                     "Gather buffers only work for lev-1");
    // If no particles, do not do anything
    if (np_to_push == 0) { return; }

    // Get cell size on gather_lev
    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(gather_lev,0));

    // Get box from which field is gathered.
    // If not gathering from the finest level, the box is coarsened.
    Box box;
    if (lev == gather_lev) {
        box = pti.tilebox();
    } else {
        const IntVect& ref_ratio = WarpX::RefRatio(gather_lev);
        box = amrex::coarsen(pti.tilebox(),ref_ratio);
    }

    // Add guard cells to the box.
    box.grow(ngEB);

    const auto getPosition = GetParticlePosition<PIdx>(pti, offset);
          auto setPosition = SetParticlePosition<PIdx>(pti, offset);

    const auto getExternalEB = GetExternalEBField(pti, offset);

    const amrex::ParticleReal Ex_external_particle = m_E_external_particle[0];
    const amrex::ParticleReal Ey_external_particle = m_E_external_particle[1];
    const amrex::ParticleReal Ez_external_particle = m_E_external_particle[2];
    const amrex::ParticleReal Bx_external_particle = m_B_external_particle[0];
    const amrex::ParticleReal By_external_particle = m_B_external_particle[1];
    const amrex::ParticleReal Bz_external_particle = m_B_external_particle[2];

    // Lower corner of tile box physical domain (take into account Galilean shift)
    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, gather_lev, 0._rt);

    const Dim3 lo = lbound(box);

    const bool galerkin_interpolation = WarpX::galerkin_interpolation;
    const int nox = WarpX::nox;
    const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;

    amrex::Array4<const amrex::Real> const& ex_arr = exfab->array();
    amrex::Array4<const amrex::Real> const& ey_arr = eyfab->array();
    amrex::Array4<const amrex::Real> const& ez_arr = ezfab->array();
    amrex::Array4<const amrex::Real> const& bx_arr = bxfab->array();
    amrex::Array4<const amrex::Real> const& by_arr = byfab->array();
    amrex::Array4<const amrex::Real> const& bz_arr = bzfab->array();

    amrex::IndexType const ex_type = exfab->box().ixType();
    amrex::IndexType const ey_type = eyfab->box().ixType();
    amrex::IndexType const ez_type = ezfab->box().ixType();
    amrex::IndexType const bx_type = bxfab->box().ixType();
    amrex::IndexType const by_type = byfab->box().ixType();
    amrex::IndexType const bz_type = bzfab->box().ixType();

    auto& attribs = pti.GetAttribs();
    ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr() + offset;

    const int do_copy = (m_do_back_transformed_particles && (a_dt_type!=DtType::SecondHalf) );
    CopyParticleAttribs copyAttribs;
    if (do_copy) {
        copyAttribs = CopyParticleAttribs(*this, pti, offset);
    }

    int* AMREX_RESTRICT ion_lev = nullptr;
    if (do_field_ionization) {
        ion_lev = pti.GetiAttribs("ionizationLevel").dataPtr() + offset;
    }

    const bool save_previous_position = m_save_previous_position;
    ParticleReal* x_old = nullptr;
    ParticleReal* y_old = nullptr;
    ParticleReal* z_old = nullptr;
    if (save_previous_position) {
#if !defined(WARPX_DIM_1D_Z)
        x_old = pti.GetAttribs("prev_x").dataPtr() + offset;
#endif
#if defined(WARPX_DIM_3D)
        y_old = pti.GetAttribs("prev_y").dataPtr() + offset;
#endif
#if defined(WARPX_ZINDEX)
        z_old = pti.GetAttribs("prev_z").dataPtr() + offset;
#endif
        amrex::ignore_unused(x_old, y_old, z_old);
    }

    // Loop over the particles and update their momentum
    const amrex::ParticleReal q = this->charge;
    const amrex::ParticleReal m = this-> mass;

    const auto pusher_algo = WarpX::particle_pusher_algo;
    const auto do_crr = do_classical_radiation_reaction;
#ifdef WARPX_QED
    const auto do_sync = m_do_qed_quantum_sync;
    amrex::Real t_chi_max = 0.0;
    if (do_sync) { t_chi_max = m_shr_p_qs_engine->get_minimum_chi_part(); }

    QuantumSynchrotronEvolveOpticalDepth evolve_opt;
    amrex::ParticleReal* AMREX_RESTRICT p_optical_depth_QSR = nullptr;
    const bool local_has_quantum_sync = has_quantum_sync();
    if (local_has_quantum_sync) {
        evolve_opt = m_shr_p_qs_engine->build_evolve_functor();
        p_optical_depth_QSR = pti.GetAttribs("opticalDepthQSR").dataPtr()  + offset;
    }
#endif

    const auto t_do_not_gather = do_not_gather;

    enum exteb_flags : int { no_exteb, has_exteb };
    enum qed_flags : int { no_qed, has_qed };

    const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;
#ifdef WARPX_QED
    const int qed_runtime_flag = (local_has_quantum_sync || do_sync) ? has_qed : no_qed;
#else
    int qed_runtime_flag = no_qed;
#endif

    // Using this version of ParallelFor with compile time options
    // improves performance when qed or external EB are not used by reducing
    // register pressure.
    amrex::ParallelFor(
        TypeList<CompileTimeOptions<no_exteb,has_exteb>, CompileTimeOptions<no_qed  ,has_qed>>{},
        {exteb_runtime_flag, qed_runtime_flag},
        np_to_push,
        [=] AMREX_GPU_DEVICE (long ip, auto exteb_control, auto qed_control)
    {
        amrex::ParticleReal xp, yp, zp;
        getPosition(ip, xp, yp, zp);

        if (save_previous_position) {
#if !defined(WARPX_DIM_1D_Z)
            x_old[ip] = xp;
#endif
#if defined(WARPX_DIM_3D)
            y_old[ip] = yp;
#endif
#if defined(WARPX_ZINDEX)
            z_old[ip] = zp;
#endif
        }

        amrex::ParticleReal Exp = Ex_external_particle;
        amrex::ParticleReal Eyp = Ey_external_particle;
        amrex::ParticleReal Ezp = Ez_external_particle;
        amrex::ParticleReal Bxp = Bx_external_particle;
        amrex::ParticleReal Byp = By_external_particle;
        amrex::ParticleReal Bzp = Bz_external_particle;

        if(!t_do_not_gather){
            // first gather E and B to the particle positions
            doGatherShapeN(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                           ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                           ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                           dinv, xyzmin, lo, n_rz_azimuthal_modes,
                           nox, galerkin_interpolation);
        }

        [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB;
        if constexpr (exteb_control == has_exteb) {
            getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
        }

        scaleFields(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp);

        if (do_copy) {
            //  Copy the old x and u for the BTD
            copyAttribs(ip);
        }

#ifdef WARPX_QED
        if (!do_sync) {
            doParticleMomentumPush<0>(ux[ip], uy[ip], uz[ip],
                                      Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                      ion_lev ? ion_lev[ip] : 1,
                                      m, q, pusher_algo, do_crr,
                                      t_chi_max,
                                      dt);
        } else {
            if constexpr (qed_control == has_qed) {
                doParticleMomentumPush<1>(ux[ip], uy[ip], uz[ip],
                                          Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                          ion_lev ? ion_lev[ip] : 1,
                                          m, q, pusher_algo, do_crr,
                                          t_chi_max,
                                          dt);
            }
        }
#else
        doParticleMomentumPush<0>(ux[ip], uy[ip], uz[ip],
                                  Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                  ion_lev ? ion_lev[ip] : 1,
                                  m, q, pusher_algo, do_crr,
                                  dt);
#endif
        UpdatePosition(xp, yp, zp, ux[ip], uy[ip], uz[ip], dt);
        setPosition(ip, xp, yp, zp);

#ifdef WARPX_QED
        [[maybe_unused]] auto foo_local_has_quantum_sync = local_has_quantum_sync;
        [[maybe_unused]] auto *foo_podq = p_optical_depth_QSR;
        [[maybe_unused]] const auto& foo_evolve_opt = evolve_opt; // have to do all these for nvcc
        if constexpr (qed_control == has_qed) {
            if (local_has_quantum_sync) {
                evolve_opt(ux[ip], uy[ip], uz[ip],
                           Exp, Eyp, Ezp,Bxp, Byp, Bzp,
                           dt, p_optical_depth_QSR[ip]);
            }
        }
#else
            amrex::ignore_unused(qed_control);
#endif
    });
}

void
PhysicalParticleContainer::InitIonizationModule ()
{
    if (!do_field_ionization) { return; }
    const ParmParse pp_species_name(species_name);
    if (charge != PhysConst::q_e){
        ablastr::warn_manager::WMRecordWarning("Species",
            "charge != q_e for ionizable species '" +
            species_name + "':" +
            "overriding user value and setting charge = q_e.");
        charge = PhysConst::q_e;
    }
    utils::parser::queryWithParser(pp_species_name, "do_adk_correction", do_adk_correction);

    utils::parser::queryWithParser(
        pp_species_name, "ionization_initial_level", ionization_initial_level);
    pp_species_name.get("ionization_product_species", ionization_product_name);
    pp_species_name.get("physical_element", physical_element);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        physical_element == "H" || !do_adk_correction,
        "Correction to ADK by Zhang et al., PRA 90, 043410 (2014) only works with Hydrogen");
    // Add runtime integer component for ionization level
    AddIntComp("ionizationLevel");
    // Get atomic number and ionization energies from file
    const int ion_element_id = utils::physics::ion_map_ids.at(physical_element);
    ion_atomic_number = utils::physics::ion_atomic_numbers[ion_element_id];
    Vector<Real> h_ionization_energies(ion_atomic_number);
    const int offset = utils::physics::ion_energy_offsets[ion_element_id];
    for(int i=0; i<ion_atomic_number; i++){
        h_ionization_energies[i] =
            utils::physics::table_ionization_energies[i+offset];
    }
    // Compute ADK prefactors (See Chen, JCP 236 (2013), equation (2))
    // For now, we assume l=0 and m=0.
    // The approximate expressions are used,
    // without Gamma function
    constexpr auto a3 = PhysConst::alpha*PhysConst::alpha*PhysConst::alpha;
    constexpr auto a4 = a3 * PhysConst::alpha;
    constexpr Real wa = a3 * PhysConst::c / PhysConst::r_e;
    constexpr Real Ea = PhysConst::m_e * PhysConst::c*PhysConst::c /PhysConst::q_e *
        a4/PhysConst::r_e;
    constexpr Real UH = utils::physics::table_ionization_energies[0];
    const Real l_eff = std::sqrt(UH/h_ionization_energies[0]) - 1._rt;

    const Real dt = WarpX::GetInstance().getdt(0);

    ionization_energies.resize(ion_atomic_number);
    adk_power.resize(ion_atomic_number);
    adk_prefactor.resize(ion_atomic_number);
    adk_exp_prefactor.resize(ion_atomic_number);

    Gpu::copyAsync(Gpu::hostToDevice,
                   h_ionization_energies.begin(), h_ionization_energies.end(),
                   ionization_energies.begin());

    adk_correction_factors.resize(4);
    if (do_adk_correction) {
        Vector<Real> h_correction_factors(4);
        constexpr int offset_corr = 0; // hard-coded: only Hydrogen
        for(int i=0; i<4; i++){
            h_correction_factors[i] = table_correction_factors[i+offset_corr];
        }
        Gpu::copyAsync(Gpu::hostToDevice,
                       h_correction_factors.begin(), h_correction_factors.end(),
                       adk_correction_factors.begin());
    }

    Real const* AMREX_RESTRICT p_ionization_energies = ionization_energies.data();
    Real * AMREX_RESTRICT p_adk_power = adk_power.data();
    Real * AMREX_RESTRICT p_adk_prefactor = adk_prefactor.data();
    Real * AMREX_RESTRICT p_adk_exp_prefactor = adk_exp_prefactor.data();
    amrex::ParallelFor(ion_atomic_number, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        const Real n_eff = (i+1) * std::sqrt(UH/p_ionization_energies[i]);
        const Real C2 = std::pow(2._rt,2._rt*n_eff)/(n_eff*std::tgamma(n_eff+l_eff+1._rt)*std::tgamma(n_eff-l_eff));
        p_adk_power[i] = -(2._rt*n_eff - 1._rt);
        const Real Uion = p_ionization_energies[i];
        p_adk_prefactor[i] = dt * wa * C2 * ( Uion/(2._rt*UH) )
            * std::pow(2._rt*std::pow((Uion/UH),3._rt/2._rt)*Ea,2._rt*n_eff - 1._rt);
        p_adk_exp_prefactor[i] = -2._rt/3._rt * std::pow( Uion/UH,3._rt/2._rt) * Ea;
    });

    Gpu::synchronize();
}

IonizationFilterFunc
PhysicalParticleContainer::getIonizationFunc (const WarpXParIter& pti,
                                              int lev,
                                              amrex::IntVect ngEB,
                                              const amrex::FArrayBox& Ex,
                                              const amrex::FArrayBox& Ey,
                                              const amrex::FArrayBox& Ez,
                                              const amrex::FArrayBox& Bx,
                                              const amrex::FArrayBox& By,
                                              const amrex::FArrayBox& Bz)
{
    WARPX_PROFILE("PhysicalParticleContainer::getIonizationFunc()");

    return {pti, lev, ngEB, Ex, Ey, Ez, Bx, By, Bz,
                                m_E_external_particle, m_B_external_particle,
                                ionization_energies.dataPtr(),
                                adk_prefactor.dataPtr(),
                                adk_exp_prefactor.dataPtr(),
                                adk_power.dataPtr(),
                                adk_correction_factors.dataPtr(),
                                GetIntCompIndex("ionizationLevel"),
                                ion_atomic_number,
                                do_adk_correction};
}

PlasmaInjector* PhysicalParticleContainer::GetPlasmaInjector (int i)
{
    if (i < 0 || i >= static_cast<int>(plasma_injectors.size())) {
        return nullptr;
    } else {
        return plasma_injectors[i].get();
    }
}

void PhysicalParticleContainer::resample (const amrex::Vector<amrex::Geometry>& geom, const int timestep, const bool verbose)
{
    // In heavily load imbalanced simulations, MPI processes with few particles will spend most of
    // the time at the MPI synchronization in TotalNumberOfParticles(). Having two profiler entries
    // here is thus useful to avoid confusing time spent waiting for other processes with time
    // spent doing actual resampling.
    WARPX_PROFILE_VAR_NS("MultiParticleContainer::doResampling::MPI_synchronization",
                         blp_resample_synchronization);
    WARPX_PROFILE_VAR_NS("MultiParticleContainer::doResampling::ActualResampling",
                         blp_resample_actual);

    WARPX_PROFILE_VAR_START(blp_resample_synchronization);
    const amrex::Real global_numparts = TotalNumberOfParticles();
    WARPX_PROFILE_VAR_STOP(blp_resample_synchronization);

    WARPX_PROFILE_VAR_START(blp_resample_actual);
    if (m_resampler.triggered(timestep, global_numparts))
    {
        Redistribute();
        for (int lev = 0; lev <= maxLevel(); lev++)
        {
            for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
            {
                m_resampler(geom[lev], pti, lev, this);
            }
        }
        deleteInvalidParticles();
        if (verbose) {
            amrex::Print() << Utils::TextMsg::Info(
                "Resampled " + species_name + " at step " + std::to_string(timestep)
                + ": macroparticle count decreased by "
                + std::to_string(static_cast<int>(global_numparts - TotalNumberOfParticles()))
            );
        }
    }
    WARPX_PROFILE_VAR_STOP(blp_resample_actual);
}

bool
PhysicalParticleContainer::findRefinedInjectionBox (amrex::Box& a_fine_injection_box, amrex::IntVect& a_rrfac)
{
    WARPX_PROFILE("PhysicalParticleContainer::findRefinedInjectionBox");

    // This does not work if the mesh is dynamic.  But in that case, we should
    // not use refined injected either.  We also assume there is only one fine level.
    static bool refine_injection = false;
    static Box fine_injection_box;
    static amrex::IntVect rrfac(AMREX_D_DECL(1,1,1));
    if (!refine_injection and WarpX::moving_window_active(WarpX::GetInstance().getistep(0)+1) and WarpX::refine_plasma and do_continuous_injection and numLevels() == 2) {
        refine_injection = true;
        fine_injection_box = ParticleBoxArray(1).minimalBox();
        fine_injection_box.setSmall(WarpX::moving_window_dir, std::numeric_limits<int>::lowest()/2);
        fine_injection_box.setBig(WarpX::moving_window_dir, std::numeric_limits<int>::max()/2);
        rrfac = m_gdb->refRatio(0);
        fine_injection_box.coarsen(rrfac);
    }
    a_fine_injection_box = fine_injection_box;
    a_rrfac = rrfac;
    return refine_injection;
}

#ifdef WARPX_QED


bool PhysicalParticleContainer::has_quantum_sync () const
{
    return m_do_qed_quantum_sync;
}

bool PhysicalParticleContainer::has_breit_wheeler () const
{
    return m_do_qed_breit_wheeler;
}

void
PhysicalParticleContainer::
set_breit_wheeler_engine_ptr (const std::shared_ptr<BreitWheelerEngine>& ptr)
{
    m_shr_p_bw_engine = ptr;
}

void
PhysicalParticleContainer::
set_quantum_sync_engine_ptr (const std::shared_ptr<QuantumSynchrotronEngine>& ptr)
{
    m_shr_p_qs_engine = ptr;
}

PhotonEmissionFilterFunc
PhysicalParticleContainer::getPhotonEmissionFilterFunc ()
{
    WARPX_PROFILE("PhysicalParticleContainer::getPhotonEmissionFunc()");
    return PhotonEmissionFilterFunc{GetRealCompIndex("opticalDepthQSR") - NArrayReal};
}

PairGenerationFilterFunc
PhysicalParticleContainer::getPairGenerationFilterFunc ()
{
    WARPX_PROFILE("PhysicalParticleContainer::getPairGenerationFunc()");
    return PairGenerationFilterFunc{GetRealCompIndex("opticalDepthBW") - NArrayReal};
}

#endif

/* \brief Temperature Deposition for thread thread_num
 * \param pti         Particle iterator
 * \param wp          Array of particle weights
 * \param uxp uyp uzp Array of particle momenta
 * \param Tx Ty Tz    Full array of temperature components
 * \param offset      Index of first particle for which temperature is deposited
 * \param np_to_deposit Number of particles for which temperature is deposited.
                        Particles [offset,offset+np_to_deposit] deposit temperature
 * \param thread_num  Thread number (if tiling)
 * \param lev         Level of box that contains particles
 * \param depos_lev   Level on which particles deposit (if buffers are used)
 * \param dt          Time step for particle level
 * \param relative_time  Time at which to deposit T, relative to the time of the
 *                       current positions of the particles. When different than 0,
 *                       the particle position will be temporarily modified to match
 *                       the time of the deposition.
 */
void
PhysicalParticleContainer::DepositTemperature (
    WarpXParIter& pti,
    RealVector const & wp, RealVector const & uxp,
    RealVector const & uyp, RealVector const & uzp,
    amrex::MultiFab * Tx, amrex::MultiFab * Ty, amrex::MultiFab * Tz,
    long const offset, long const np_to_deposit,
    int const thread_num, const int lev, int const depos_lev,
    amrex::Real const relative_time, PushType push_type,
    const warpx::particles::deposition::TemperatureDepositionType type,
    const warpx::particles::deposition::TemperatureDepositionPass pass)
{
    using ablastr::fields::Direction;

    WARPX_PROFILE("PhysicalParticleContainer::DepositTemperature()");

    // Return if we are not depositing temperature.
    if (!m_do_temperature_deposition) { return; }

    if (WarpX::current_deposition_algo != CurrentDepositionAlgo::Direct
        || push_type != PushType::Explicit
        || WarpX::do_shared_mem_current_deposition
        )
    {
        WARPX_ABORT_WITH_MESSAGE(
            "Temperature Deposition only works with explicit solvers, direct current deposition, "
            "and non-shared memory deposition."
        );
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((depos_lev==(lev-1)) ||
                                     (depos_lev==(lev  )),
                                     "Deposition buffers only work for lev-1");

    // If no particles, do not do anything
    if (np_to_deposit == 0) { return; }

    // If user decides not to deposit
    if (do_not_deposit) { return; }

    // Number of guard cells for local deposition of J
    const WarpX& warpx = WarpX::GetInstance();

    amrex::IntVect ng_J = warpx.get_ng_depos_J();

    // Extract deposition order and check that particles shape fits within the guard cells.
    // NOTE: In specific situations where the staggering of J and the current deposition algorithm
    // are not trivial, this check might be too relaxed and we might include a particle that should
    // deposit part of its current in a neighboring box. However, this should catch particles
    // traveling many cells away, for example with algorithms that allow for large time steps.

#if   defined(WARPX_DIM_1D_Z)
    const amrex::IntVect shape_extent = amrex::IntVect(static_cast<int>(WarpX::noz/2));
#elif   defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const amrex::IntVect shape_extent = amrex::IntVect(static_cast<int>(WarpX::nox/2));
#elif   defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const amrex::IntVect shape_extent = amrex::IntVect(static_cast<int>(WarpX::nox/2),
                                                       static_cast<int>(WarpX::noz/2));
#elif defined(WARPX_DIM_3D)
    const amrex::IntVect shape_extent = amrex::IntVect(static_cast<int>(WarpX::nox/2),
                                                       static_cast<int>(WarpX::noy/2),
                                                       static_cast<int>(WarpX::noz/2));
#endif

    // On GPU: particles deposit directly on the J arrays, which usually have a larger number of guard cells
    // Jx, Jy and Jz have the same number of guard cells, hence it is sufficient to check for Jx
    const amrex::IntVect range = Tx->nGrowVect() - shape_extent;

    amrex::ignore_unused(range); // for release builds
    AMREX_ASSERT_WITH_MESSAGE(
        amrex::numParticlesOutOfRange(pti, range) == 0,
        "Particles shape does not fit within tile (CPU) or guard cells (GPU) used for current deposition");

    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(depos_lev,0));

    // Get tile box where current is deposited.
    // The tile box is different when depositing in the buffers (depos_lev<lev)
    // or when depositing inside the level (depos_lev=lev)
    Box tilebox;
    if (lev == depos_lev) {
        tilebox = pti.tilebox();
    } else {
        const IntVect& ref_ratio = WarpX::RefRatio(depos_lev);
        tilebox = amrex::coarsen(pti.tilebox(),ref_ratio);
    }

    tilebox.grow(ng_J);

    amrex::ignore_unused(thread_num);
    // GPU, no tiling: j<xyz>_arr point to the full j<xyz> arrays
    auto & Tx_fab = Tx->get(pti);
    auto & Ty_fab = Ty->get(pti);
    auto & Tz_fab = Tz->get(pti);

    auto & nx_iab =    local_temperature_arrays->get_n(Direction{0}, lev)->get(pti);
    auto & ny_iab =    local_temperature_arrays->get_n(Direction{1}, lev)->get(pti);
    auto & nz_iab =    local_temperature_arrays->get_n(Direction{2}, lev)->get(pti);
    auto & wx_fab =    local_temperature_arrays->get("w", Direction{0}, lev)->get(pti);
    auto & wy_fab =    local_temperature_arrays->get("w", Direction{1}, lev)->get(pti);
    auto & wz_fab =    local_temperature_arrays->get("w", Direction{2}, lev)->get(pti);
    auto & w2x_fab =   local_temperature_arrays->get("w2", Direction{0}, lev)->get(pti);
    auto & w2y_fab =   local_temperature_arrays->get("w2", Direction{1}, lev)->get(pti);
    auto & w2z_fab =   local_temperature_arrays->get("w2", Direction{2}, lev)->get(pti);
    auto & vxbar_fab = local_temperature_arrays->get("vbar", Direction{0}, lev)->get(pti);
    auto & vybar_fab = local_temperature_arrays->get("vbar", Direction{1}, lev)->get(pti);
    auto & vzbar_fab = local_temperature_arrays->get("vbar", Direction{2}, lev)->get(pti);

    const auto GetPosition = GetParticlePosition<PIdx>(pti, offset);

    // Lower corner of tile box physical domain
    // Note that this includes guard cells since it is after tilebox.ngrow
    const Dim3 lo = lbound(tilebox);
    // Take into account Galilean shift
    const amrex::XDim3 xyzmin = WarpX::LowerCorner(tilebox, depos_lev, 0.0_rt);

    if        (WarpX::nox == 1){
        warpx::particles::deposition::doVarianceDepositionShapeN<1>(
            GetPosition, wp.dataPtr() + offset, uxp.dataPtr() + offset,
            uyp.dataPtr() + offset, uzp.dataPtr() + offset,
            Tx_fab, Ty_fab, Tz_fab,
            nx_iab, ny_iab, nz_iab, wx_fab, wy_fab, wz_fab,
            w2x_fab, w2y_fab, w2z_fab, vxbar_fab, vybar_fab, vzbar_fab,
            type, pass, np_to_deposit, relative_time, dinv,
            xyzmin, lo, WarpX::n_rz_azimuthal_modes);
    } else if (WarpX::nox == 2){
        warpx::particles::deposition::doVarianceDepositionShapeN<2>(
            GetPosition, wp.dataPtr() + offset, uxp.dataPtr() + offset,
            uyp.dataPtr() + offset, uzp.dataPtr() + offset,
            Tx_fab, Ty_fab, Tz_fab,
            nx_iab, ny_iab, nz_iab, wx_fab, wy_fab, wz_fab,
            w2x_fab, w2y_fab, w2z_fab, vxbar_fab, vybar_fab, vzbar_fab,
            type, pass, np_to_deposit, relative_time, dinv,
            xyzmin, lo, WarpX::n_rz_azimuthal_modes);
    } else if (WarpX::nox == 3){
        warpx::particles::deposition::doVarianceDepositionShapeN<3>(
            GetPosition, wp.dataPtr() + offset, uxp.dataPtr() + offset,
            uyp.dataPtr() + offset, uzp.dataPtr() + offset,
            Tx_fab, Ty_fab, Tz_fab,
            nx_iab, ny_iab, nz_iab, wx_fab, wy_fab, wz_fab,
            w2x_fab, w2y_fab, w2z_fab, vxbar_fab, vybar_fab, vzbar_fab,
            type, pass, np_to_deposit, relative_time, dinv,
            xyzmin, lo, WarpX::n_rz_azimuthal_modes);
    } else if (WarpX::nox == 4){
        warpx::particles::deposition::doVarianceDepositionShapeN<4>(
            GetPosition, wp.dataPtr() + offset, uxp.dataPtr() + offset,
            uyp.dataPtr() + offset, uzp.dataPtr() + offset,
            Tx_fab, Ty_fab, Tz_fab,
            nx_iab, ny_iab, nz_iab, wx_fab, wy_fab, wz_fab,
            w2x_fab, w2y_fab, w2z_fab, vxbar_fab, vybar_fab, vzbar_fab,
            type, pass, np_to_deposit, relative_time, dinv,
            xyzmin, lo, WarpX::n_rz_azimuthal_modes);
    }
}

void
PhysicalParticleContainer::AccumulateVelocitiesAndComputeTemperature (
    ablastr::fields::MultiLevelVectorField const & T_vf,
    const amrex::Real relative_time)
{
    using ablastr::fields::Direction;
    using warpx::particles::deposition::TemperatureDepositionType;
    using warpx::particles::deposition::TemperatureDepositionPass;

    // Todo: link this to inputs (hardcoded for the time being)
    // Will fix this in a follow up PR.
    auto depos_type = TemperatureDepositionType::DOUBLE_PASS;

    const auto& warpx = WarpX::GetInstance();

    // Loop over the refinement levels
    auto const finest_level = static_cast<int>(T_vf.size() - 1);
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto const& periodicity = warpx.Geom(lev).periodicity();

        // Clear accumulation arrays
        local_temperature_arrays->reset();

        // Loop over particle tiles and deposit current on each level
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        {
        const int thread_num = omp_get_thread_num();
#else
        const int thread_num = 0;
#endif
        for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            const auto & wp = pti.GetAttribs(PIdx::w);
            const auto & uxp = pti.GetAttribs(PIdx::ux);
            const auto & uyp = pti.GetAttribs(PIdx::uy);
            const auto & uzp = pti.GetAttribs(PIdx::uz);

            DepositTemperature(pti, wp, uxp, uyp, uzp,
                               T_vf[lev][0], T_vf[lev][1], T_vf[lev][2],
                               0, np, thread_num, lev, lev, relative_time, PushType::Explicit,
                               depos_type,
                               TemperatureDepositionPass::FIRST);
        }
#ifdef AMREX_USE_OMP
        }
#endif

        amrex::Gpu::streamSynchronize();

        // Fist pass done, now lets sum the boundaries for the accumulation arrays
        for (int idir=0; idir < 3; ++idir)
        {
            amrex::iMultiFab* n_mf    = local_temperature_arrays->get_n(Direction{idir}, lev);
            amrex::MultiFab*  w_mf    = local_temperature_arrays->get("w", Direction{idir}, lev);
            amrex::MultiFab*  vbar_mf = local_temperature_arrays->get("vbar", Direction{idir}, lev);

            n_mf->SumBoundary(0, 1, n_mf->nGrowVect(), n_mf->nGrowVect(), periodicity);
            WarpXSumGuardCells(*w_mf, periodicity, w_mf->nGrowVect(), 0, 1);
            WarpXSumGuardCells(*vbar_mf, periodicity, vbar_mf->nGrowVect(), 0, 1);
        }

        amrex::Gpu::streamSynchronize();

        if (depos_type == TemperatureDepositionType::DOUBLE_PASS)
        {
            // First step is to clear wv2 for re-accumulation
            for (int idir = 0; idir < 3; ++idir)
            {
                local_temperature_arrays->get("w2", Direction{idir}, lev)->setVal(0.);
            }

            amrex::Gpu::streamSynchronize();

            // Now run deposition loop again
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
            {
            const int thread_num = omp_get_thread_num();
#endif
            for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
            {
                const long np = pti.numParticles();
                const auto & wp = pti.GetAttribs(PIdx::w);
                const auto & uxp = pti.GetAttribs(PIdx::ux);
                const auto & uyp = pti.GetAttribs(PIdx::uy);
                const auto & uzp = pti.GetAttribs(PIdx::uz);

                DepositTemperature(pti, wp, uxp, uyp, uzp,
                                T_vf[lev][0], T_vf[lev][1], T_vf[lev][2],
                                0, np, thread_num, lev, lev, relative_time, PushType::Explicit,
                                depos_type, TemperatureDepositionPass::SECOND);
            }
#ifdef AMREX_USE_OMP
            }
#endif
            amrex::Gpu::streamSynchronize();

        } //if (depos_type == TemperatureDepositionType::DOUBLE_PASS)

        // Do boundary sum for w2
        for (int idir=0; idir < 3; ++idir)
        {
            amrex::MultiFab*  w2_mf    = local_temperature_arrays->get("w2", Direction{idir}, lev);

            WarpXSumGuardCells(*w2_mf, periodicity, w2_mf->nGrowVect(), 0, 1);
        }

        // Get MF pointers for all deposition multifabs
        amrex::iMultiFab* nx_mf    = local_temperature_arrays->get_n(Direction{0}, lev);
        amrex::iMultiFab* ny_mf    = local_temperature_arrays->get_n(Direction{1}, lev);
        amrex::iMultiFab* nz_mf    = local_temperature_arrays->get_n(Direction{2}, lev);
        amrex::MultiFab*  wx_mf    = local_temperature_arrays->get("w", Direction{0}, lev);
        amrex::MultiFab*  wy_mf    = local_temperature_arrays->get("w", Direction{1}, lev);
        amrex::MultiFab*  wz_mf    = local_temperature_arrays->get("w", Direction{2}, lev);
        amrex::MultiFab*  w2x_mf   = local_temperature_arrays->get("w2", Direction{0}, lev);
        amrex::MultiFab*  w2y_mf   = local_temperature_arrays->get("w2", Direction{1}, lev);
        amrex::MultiFab*  w2z_mf   = local_temperature_arrays->get("w2", Direction{2}, lev);
        amrex::MultiFab*  vbarx_mf = local_temperature_arrays->get("vbar", Direction{0}, lev);
        amrex::MultiFab*  vbary_mf = local_temperature_arrays->get("vbar", Direction{1}, lev);
        amrex::MultiFab*  vbarz_mf = local_temperature_arrays->get("vbar", Direction{2}, lev);

        // Normalize variance after accumulating sums cell by cell
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( amrex::MFIter mfi(*T_vf[lev][0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

            amrex::Array4<amrex::Real> const& varx_arr = T_vf[lev][0]->array(mfi);
            amrex::Array4<amrex::Real> const& vary_arr = T_vf[lev][1]->array(mfi);
            amrex::Array4<amrex::Real> const& varz_arr = T_vf[lev][2]->array(mfi);
            const amrex::Array4<const int> & nx_arr = nx_mf->const_array(mfi);
            const amrex::Array4<const int> & ny_arr = ny_mf->const_array(mfi);
            const amrex::Array4<const int> & nz_arr = nz_mf->const_array(mfi);
            const amrex::Array4<const amrex::Real> & wx_arr = wx_mf->const_array(mfi);
            const amrex::Array4<const amrex::Real> & wy_arr = wy_mf->const_array(mfi);
            const amrex::Array4<const amrex::Real> & wz_arr = wz_mf->const_array(mfi);
            const amrex::Array4<const amrex::Real> & w2x_arr = w2x_mf->const_array(mfi);
            const amrex::Array4<const amrex::Real> & w2y_arr = w2y_mf->const_array(mfi);
            const amrex::Array4<const amrex::Real> & w2z_arr = w2z_mf->const_array(mfi);
            amrex::Array4<amrex::Real> const& vxbar_arr = vbarx_mf->array(mfi);
            amrex::Array4<amrex::Real> const& vybar_arr = vbary_mf->array(mfi);
            amrex::Array4<amrex::Real> const& vzbar_arr = vbarz_mf->array(mfi);

            const amrex::Box& tbx  = mfi.growntilebox( T_vf[lev][0]->ixType().toIntVect() );
            const amrex::Box& tby  = mfi.growntilebox( T_vf[lev][1]->ixType().toIntVect() );
            const amrex::Box& tbz  = mfi.growntilebox( T_vf[lev][2]->ixType().toIntVect() );


            bool single_pass = (depos_type == warpx::particles::deposition::TemperatureDepositionType::SINGLE_PASS);

            // Update Mean and Variance values after running through weight deposition loop
            amrex::ParallelFor(tbx, tby, tbz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (nx_arr(i,j,k) > 1) {
                        amrex::Real sumw = wx_arr(i,j,k);
                        amrex::Real sumwv = vxbar_arr(i,j,k);
                        amrex::Real n = static_cast<amrex::Real>(nx_arr(i,j,k));
                        amrex::Real norm = n/((n-1._rt)*sumw);

                        vxbar_arr(i,j,k) = sumwv/sumw;
                        varx_arr(i,j,k) = norm*w2x_arr(i,j,k);
                        if (single_pass)
                            varx_arr(i,j,k) -= norm*sumwv*sumwv/sumw;
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (ny_arr(i,j,k) > 1) {
                        amrex::Real sumw = wy_arr(i,j,k);
                        amrex::Real sumwv = vybar_arr(i,j,k);
                        amrex::Real n = static_cast<amrex::Real>(ny_arr(i,j,k));
                        amrex::Real norm = n/((n-1._rt)*sumw);

                        vybar_arr(i,j,k) = sumwv/sumw;
                        vary_arr(i,j,k) = norm*w2y_arr(i,j,k);
                        if (single_pass)
                            vary_arr(i,j,k) -= norm*sumwv*sumwv/sumw;
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (nz_arr(i,j,k) > 1) {
                        amrex::Real sumw = wz_arr(i,j,k);
                        amrex::Real sumwv = vzbar_arr(i,j,k);
                        amrex::Real n = static_cast<amrex::Real>(nz_arr(i,j,k));
                        amrex::Real norm = n/((n-1._rt)*sumw);

                        vzbar_arr(i,j,k) = sumwv/sumw;
                        varz_arr(i,j,k) = norm*w2z_arr(i,j,k);
                        if (single_pass)
                            varz_arr(i,j,k) -= norm*sumwv*sumwv/sumw;
                    }
                });

        }

        amrex::Gpu::streamSynchronize();

        // Multiply variance by species mass over the Boltzmann constant to convert to temperature in K
        amrex::Real Tnorm = this->getMass()/ablastr::constant::SI::kb;

        // Sum boundaries for accumulation MFs, apply normalization, and filter to end up with
        // temperature in K in T_vf
        local_temperature_arrays->ConvertVarianceToTemperatureAndFilter(T_vf, Tnorm, WarpX::use_filter);
    }
}
