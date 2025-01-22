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
#include "Initialization/InjectorDensity.H"
#include "Initialization/InjectorMomentum.H"
#include "Initialization/InjectorPosition.H"
#include "MultiParticleContainer.H"
#include "Particles/AddPlasmaUtilities.H"
#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H"
#endif
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
#include <sstream>

using namespace amrex;

namespace
{
    using ParticleType = WarpXParticleContainer::ParticleType;

    // Since the user provides the density distribution
    // at t_lab=0 and in the lab-frame coordinates,
    // we need to find the lab-frame position of this
    // particle at t_lab=0, from its boosted-frame coordinates
    // Assuming ballistic motion, this is given by:
    // z0_lab = gamma*( z_boost*(1-beta*betaz_lab) - ct_boost*(betaz_lab-beta) )
    // where betaz_lab is the speed of the particle in the lab frame
    //
    // In order for this equation to be solvable, betaz_lab
    // is explicitly assumed to have no dependency on z0_lab
    //
    // Note that we use the bulk momentum to perform the ballistic correction
    // Assume no z0_lab dependency
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real applyBallisticCorrection(const XDim3& pos, const InjectorMomentum* inj_mom,
                                  Real gamma_boost, Real beta_boost, Real t) noexcept
    {
        const XDim3 u_bulk = inj_mom->getBulkMomentum(pos.x, pos.y, pos.z);
        const Real gamma_bulk = std::sqrt(1._rt +
                  (u_bulk.x*u_bulk.x+u_bulk.y*u_bulk.y+u_bulk.z*u_bulk.z));
        const Real betaz_bulk = u_bulk.z/gamma_bulk;
        const Real z0 = gamma_boost * ( pos.z*(1.0_rt-beta_boost*betaz_bulk)
                             - PhysConst::c*t*(betaz_bulk-beta_boost) );
        return z0;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    XDim3 getCellCoords (const GpuArray<Real, AMREX_SPACEDIM>& lo_corner,
                         const GpuArray<Real, AMREX_SPACEDIM>& dx,
                         const XDim3& r, const IntVect& iv) noexcept
    {
        XDim3 pos;
#if defined(WARPX_DIM_3D)
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = lo_corner[1] + (iv[1]+r.y)*dx[1];
        pos.z = lo_corner[2] + (iv[2]+r.z)*dx[2];
#elif defined(WARPX_DIM_XZ)
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = 0.0_rt;
        pos.z = lo_corner[1] + (iv[1]+r.y)*dx[1];
#elif defined(WARPX_DIM_RZ)
        // Note that for RZ, r.y will be theta
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = 0.0_rt;
        pos.z = lo_corner[1] + (iv[1]+r.z)*dx[1];
#elif defined(WARPX_DIM_1D_Z)
        pos.x = 0.0_rt;
        pos.y = 0.0_rt;
        pos.z = lo_corner[0] + (iv[0]+r.x)*dx[0];
#endif
        return pos;
    }

    /**
     * \brief This function is called in AddPlasma when we want a particle to be removed at the
     * next call to redistribute. It initializes all the particle properties to zero (to be safe
     * and avoid any possible undefined behavior before the next call to redistribute) and sets
     * the particle id to -1 so that it can be effectively deleted.
     *
     * \param idcpu particle id soa data
     * \param pa particle real soa data
     * \param ip index for soa data
     * \param do_field_ionization whether species has ionization
     * \param pi ionization level data
     * \param has_quantum_sync whether species has quantum synchrotron
     * \param p_optical_depth_QSR quantum synchrotron optical depth data
     * \param has_breit_wheeler whether species has Breit-Wheeler
     * \param p_optical_depth_BW Breit-Wheeler optical depth data
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void ZeroInitializeAndSetNegativeID (
        uint64_t * AMREX_RESTRICT idcpu,
        const GpuArray<ParticleReal*,PIdx::nattribs>& pa, long& ip,
        const bool& do_field_ionization, int* pi
#ifdef WARPX_QED
        ,const QEDHelper& qed_helper
#endif
        ) noexcept
    {
        for (int idx=0 ; idx < PIdx::nattribs ; idx++) {
            pa[idx][ip] = 0._rt;
        }
        if (do_field_ionization) {pi[ip] = 0;}
#ifdef WARPX_QED
        if (qed_helper.has_quantum_sync) {qed_helper.p_optical_depth_QSR[ip] = 0._rt;}
        if (qed_helper.has_breit_wheeler) {qed_helper.p_optical_depth_BW[ip] = 0._rt;}
#endif

        idcpu[ip] = amrex::ParticleIdCpus::Invalid;
    }
}

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
        NewRealComp("opticalDepthQSR");
    }

    pp_species_name.query("do_qed_breit_wheeler", m_do_qed_breit_wheeler);
    if (m_do_qed_breit_wheeler) {
        NewRealComp("opticalDepthBW");
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
        NewIntComp(m_user_int_attribs.at(i));
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
        NewRealComp(m_user_real_attribs.at(i));
    }

    // If old particle positions should be saved add the needed components
    pp_species_name.query("save_previous_position", m_save_previous_position);
    if (m_save_previous_position) {
#if (AMREX_SPACEDIM >= 2)
        NewRealComp("prev_x");
#endif
#if defined(WARPX_DIM_3D)
        NewRealComp("prev_y");
#endif
        NewRealComp("prev_z");
#ifdef WARPX_DIM_RZ
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
        amrex::Real boundary_uth;
        utils::parser::getWithParser(pp_species_boundary,"u_th",boundary_uth);
        m_boundary_conditions.SetThermalVelocity(boundary_uth);
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

void PhysicalParticleContainer::MapParticletoBoostedFrame (
    ParticleReal& x, ParticleReal& y, ParticleReal& z, ParticleReal& ux, ParticleReal& uy, ParticleReal& uz, Real t_lab) const
{
    // Map the particles from the lab frame to the boosted frame.
    // This boosts the particle to the lab frame and calculates
    // the particle time in the boosted frame. It then maps
    // the position to the time in the boosted frame.

    // For now, start with the assumption that this will only happen
    // at the start of the simulation.
    const ParticleReal uz_boost = WarpX::gamma_boost*WarpX::beta_boost*PhysConst::c;

    // tpr is the particle's time in the boosted frame
    const ParticleReal tpr = WarpX::gamma_boost*t_lab - uz_boost*z/(PhysConst::c*PhysConst::c);

    // The particle's transformed location in the boosted frame
    const ParticleReal xpr = x;
    const ParticleReal ypr = y;
    const ParticleReal zpr = WarpX::gamma_boost*z - uz_boost*t_lab;

    // transform u and gamma to the boosted frame
    const ParticleReal gamma_lab = std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz)/(PhysConst::c*PhysConst::c));
    // ux = ux;
    // uy = uy;
    uz = WarpX::gamma_boost*uz - uz_boost*gamma_lab;
    const ParticleReal gammapr = std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz)/(PhysConst::c*PhysConst::c));

    const ParticleReal vxpr = ux/gammapr;
    const ParticleReal vypr = uy/gammapr;
    const ParticleReal vzpr = uz/gammapr;

    if (do_backward_propagation){
        uz = -uz;
    }

    //Move the particles to where they will be at t = t0, the current simulation time in the boosted frame
    constexpr int lev = 0;
    const amrex::Real t0 = WarpX::GetInstance().gett_new(lev);
    if (boost_adjust_transverse_positions) {
        x = xpr - (tpr-t0)*vxpr;
        y = ypr - (tpr-t0)*vypr;
    }
    z = zpr - (tpr-t0)*vzpr;

}

void
PhysicalParticleContainer::AddGaussianBeam (PlasmaInjector const& plasma_injector){

    const Real x_m = plasma_injector.x_m;
    const Real y_m = plasma_injector.y_m;
    const Real z_m = plasma_injector.z_m;
    const Real x_rms = plasma_injector.x_rms;
    const Real y_rms = plasma_injector.y_rms;
    const Real z_rms = plasma_injector.z_rms;
    const Real x_cut = plasma_injector.x_cut;
    const Real y_cut = plasma_injector.y_cut;
    const Real z_cut = plasma_injector.z_cut;
    const Real q_tot = plasma_injector.q_tot;
    long npart = plasma_injector.npart;
    const int do_symmetrize = plasma_injector.do_symmetrize;
    const int symmetrization_order = plasma_injector.symmetrization_order;
    const Real focal_distance = plasma_injector.focal_distance;

    // Declare temporary vectors on the CPU
    Gpu::HostVector<ParticleReal> particle_x;
    Gpu::HostVector<ParticleReal> particle_y;
    Gpu::HostVector<ParticleReal> particle_z;
    Gpu::HostVector<ParticleReal> particle_ux;
    Gpu::HostVector<ParticleReal> particle_uy;
    Gpu::HostVector<ParticleReal> particle_uz;
    Gpu::HostVector<ParticleReal> particle_w;

    if (ParallelDescriptor::IOProcessor()) {
        // If do_symmetrize, create either 4x or 8x fewer particles, and
        // Replicate each particle either 4 times (x,y) (-x,y) (x,-y) (-x,-y)
        // or 8 times, additionally (y,x), (-y,x), (y,-x), (-y,-x)
        if (do_symmetrize){
            npart /= symmetrization_order;
        }
        for (long i = 0; i < npart; ++i) {
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            const Real weight = q_tot/(npart*charge);
            Real x = amrex::RandomNormal(x_m, x_rms);
            Real y = amrex::RandomNormal(y_m, y_rms);
            Real z = amrex::RandomNormal(z_m, z_rms);
#elif defined(WARPX_DIM_XZ)
            const Real weight = q_tot/(npart*charge*y_rms);
            Real x = amrex::RandomNormal(x_m, x_rms);
            constexpr Real y = 0._prt;
            Real z = amrex::RandomNormal(z_m, z_rms);
#elif defined(WARPX_DIM_1D_Z)
            const Real weight = q_tot/(npart*charge*x_rms*y_rms);
            constexpr Real x = 0._prt;
            constexpr Real y = 0._prt;
            Real z = amrex::RandomNormal(z_m, z_rms);
#endif
            if (plasma_injector.insideBounds(x, y, z)  &&
                std::abs( x - x_m ) <= x_cut * x_rms     &&
                std::abs( y - y_m ) <= y_cut * y_rms     &&
                std::abs( z - z_m ) <= z_cut * z_rms   ) {
                XDim3 u = plasma_injector.getMomentum(x, y, z);

            if (plasma_injector.do_focusing){
                const XDim3 u_bulk = plasma_injector.getInjectorMomentumHost()->getBulkMomentum(x,y,z);
                const Real u_bulk_norm = std::sqrt( u_bulk.x*u_bulk.x+u_bulk.y*u_bulk.y+u_bulk.z*u_bulk.z );

                // Compute the position of the focal plane
                // (it is located at a distance `focal_distance` from the beam centroid, in the direction of the bulk velocity)
                const Real n_x = u_bulk.x/u_bulk_norm;
                const Real n_y = u_bulk.y/u_bulk_norm;
                const Real n_z = u_bulk.z/u_bulk_norm;
                const Real x_f = x_m + focal_distance * n_x;
                const Real y_f = y_m + focal_distance * n_y;
                const Real z_f = z_m + focal_distance * n_z;
                const Real gamma = std::sqrt( 1._rt + (u.x*u.x+u.y*u.y+u.z*u.z) );

                const Real v_x = u.x / gamma * PhysConst::c;
                const Real v_y = u.y / gamma * PhysConst::c;
                const Real v_z = u.z / gamma * PhysConst::c;

                // Compute the time at which the particle will cross the focal plane
                const Real v_dot_n = v_x * n_x + v_y * n_y + v_z * n_z;
                const Real t = ((x_f-x)*n_x + (y_f-y)*n_y + (z_f-z)*n_z) / v_dot_n;

                // Displace particles in the direction orthogonal to the beam bulk momentum
                // i.e. orthogonal to (n_x, n_y, n_z)
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
                x = x - (v_x - v_dot_n*n_x) * t;
                y = y - (v_y - v_dot_n*n_y) * t;
                z = z - (v_z - v_dot_n*n_z) * t;
#elif defined(WARPX_DIM_XZ)
                x = x - (v_x - v_dot_n*n_x) * t;
                z = z - (v_z - v_dot_n*n_z) * t;
#elif defined(WARPX_DIM_1D_Z)
                z = z - (v_z - v_dot_n*n_z) * t;
#endif
            }
                u.x *= PhysConst::c;
                u.y *= PhysConst::c;
                u.z *= PhysConst::c;

                if (do_symmetrize && symmetrization_order == 8){
                    // Add eight particles to the beam:
                    CheckAndAddParticle(x, y, z, u.x, u.y, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(x, -y, z, u.x, -u.y, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-x, y, z, -u.x, u.y, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-x, -y, z, -u.x, -u.y, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(y, x, z, u.y, u.x, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-y, x, z, -u.y, u.x, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(y, -x, z, u.y, -u.x, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-y, -x, z, -u.y, -u.x, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                } else if (do_symmetrize && symmetrization_order == 4){
                    // Add four particles to the beam:
                    CheckAndAddParticle(x, y, z, u.x, u.y, u.z, weight/4._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(x, -y, z, u.x, -u.y, u.z, weight/4._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-x, y, z, -u.x, u.y, u.z, weight/4._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-x, -y, z, -u.x, -u.y, u.z, weight/4._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                } else {
                    CheckAndAddParticle(x, y, z, u.x, u.y, u.z, weight,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                }
            }
        }
    }
    // Add the temporary CPU vectors to the particle structure
    auto const np = static_cast<long>(particle_z.size());

    const amrex::Vector<ParticleReal> xp(particle_x.data(), particle_x.data() + np);
    const amrex::Vector<ParticleReal> yp(particle_y.data(), particle_y.data() + np);
    const amrex::Vector<ParticleReal> zp(particle_z.data(), particle_z.data() + np);
    const amrex::Vector<ParticleReal> uxp(particle_ux.data(), particle_ux.data() + np);
    const amrex::Vector<ParticleReal> uyp(particle_uy.data(), particle_uy.data() + np);
    const amrex::Vector<ParticleReal> uzp(particle_uz.data(), particle_uz.data() + np);

    amrex::Vector<amrex::Vector<ParticleReal>> attr;
    const amrex::Vector<ParticleReal> wp(particle_w.data(), particle_w.data() + np);
    attr.push_back(wp);

    const amrex::Vector<amrex::Vector<int>> attr_int;

    AddNParticles(0, np, xp,  yp,  zp, uxp, uyp, uzp,
                  1, attr, 0, attr_int, 1);
}

void
PhysicalParticleContainer::AddPlasmaFromFile(PlasmaInjector & plasma_injector,
                                             ParticleReal q_tot,
                                             ParticleReal z_shift)
{
    // Declare temporary vectors on the CPU
    Gpu::HostVector<ParticleReal> particle_x;
    Gpu::HostVector<ParticleReal> particle_z;
    Gpu::HostVector<ParticleReal> particle_ux;
    Gpu::HostVector<ParticleReal> particle_uz;
    Gpu::HostVector<ParticleReal> particle_w;
    Gpu::HostVector<ParticleReal> particle_y;
    Gpu::HostVector<ParticleReal> particle_uy;

#ifdef WARPX_USE_OPENPMD
    //TODO: Make changes for read/write in multiple MPI ranks
    if (ParallelDescriptor::IOProcessor()) {
        // take ownership of the series and close it when done
        auto series = std::move(plasma_injector.m_openpmd_input_series);

        // assumption asserts: see PlasmaInjector
        openPMD::Iteration it = series->iterations.begin()->second;
        const ParmParse pp_species_name(species_name);
        pp_species_name.query("impose_t_lab_from_file", impose_t_lab_from_file);
        double t_lab = 0._prt;
        if (impose_t_lab_from_file) {
            // Impose t_lab as being the time stored in the openPMD file
            t_lab = it.time<double>() * it.timeUnitSI();
        }
        std::string const ps_name = it.particles.begin()->first;
        openPMD::ParticleSpecies ps = it.particles.begin()->second;

        auto const npart = ps["position"]["x"].getExtent()[0];
#if !defined(WARPX_DIM_1D_Z)  // 2D, 3D, and RZ
        const std::shared_ptr<ParticleReal> ptr_x = ps["position"]["x"].loadChunk<ParticleReal>();
        const std::shared_ptr<ParticleReal> ptr_offset_x = ps["positionOffset"]["x"].loadChunk<ParticleReal>();
        auto const position_unit_x = static_cast<ParticleReal>(ps["position"]["x"].unitSI());
        auto const position_offset_unit_x = static_cast<ParticleReal>(ps["positionOffset"]["x"].unitSI());
#endif
#if !(defined(WARPX_DIM_XZ) || defined(WARPX_DIM_1D_Z))
        const std::shared_ptr<ParticleReal> ptr_y = ps["position"]["y"].loadChunk<ParticleReal>();
        const std::shared_ptr<ParticleReal> ptr_offset_y = ps["positionOffset"]["y"].loadChunk<ParticleReal>();
        auto const position_unit_y = static_cast<ParticleReal>(ps["position"]["y"].unitSI());
        auto const position_offset_unit_y = static_cast<ParticleReal>(ps["positionOffset"]["y"].unitSI());
#endif
        const std::shared_ptr<ParticleReal> ptr_z = ps["position"]["z"].loadChunk<ParticleReal>();
        const std::shared_ptr<ParticleReal> ptr_offset_z = ps["positionOffset"]["z"].loadChunk<ParticleReal>();
        auto const position_unit_z = static_cast<ParticleReal>(ps["position"]["z"].unitSI());
        auto const position_offset_unit_z = static_cast<ParticleReal>(ps["positionOffset"]["z"].unitSI());

        const std::shared_ptr<ParticleReal> ptr_ux = ps["momentum"]["x"].loadChunk<ParticleReal>();
        auto const momentum_unit_x = static_cast<ParticleReal>(ps["momentum"]["x"].unitSI());
        const std::shared_ptr<ParticleReal> ptr_uz = ps["momentum"]["z"].loadChunk<ParticleReal>();
        auto const momentum_unit_z = static_cast<ParticleReal>(ps["momentum"]["z"].unitSI());
        const std::shared_ptr<ParticleReal> ptr_w = ps["weighting"][openPMD::RecordComponent::SCALAR].loadChunk<ParticleReal>();
        auto const w_unit = static_cast<ParticleReal>(ps["weighting"][openPMD::RecordComponent::SCALAR].unitSI());
        std::shared_ptr<ParticleReal> ptr_uy = nullptr;
        auto momentum_unit_y = 1.0_prt;
        if (ps["momentum"].contains("y")) {
            ptr_uy = ps["momentum"]["y"].loadChunk<ParticleReal>();
            momentum_unit_y = static_cast<ParticleReal>(ps["momentum"]["y"].unitSI());
        }
        series->flush();  // shared_ptr data can be read now

        if (q_tot != 0.0) {
            std::stringstream warnMsg;
            warnMsg << " Loading particle species from file. " << ps_name << ".q_tot is ignored.";
            ablastr::warn_manager::WMRecordWarning("AddPlasmaFromFile",
               warnMsg.str(), ablastr::warn_manager::WarnPriority::high);
        }

        for (auto i = decltype(npart){0}; i<npart; ++i){

            ParticleReal const weight = ptr_w.get()[i]*w_unit;

#if !defined(WARPX_DIM_1D_Z)
            ParticleReal const x = ptr_x.get()[i]*position_unit_x + ptr_offset_x.get()[i]*position_offset_unit_x;
#else
            ParticleReal const x = 0.0_prt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            ParticleReal const y = ptr_y.get()[i]*position_unit_y + ptr_offset_y.get()[i]*position_offset_unit_y;
#else
            ParticleReal const y = 0.0_prt;
#endif
            ParticleReal const z = ptr_z.get()[i]*position_unit_z + ptr_offset_z.get()[i]*position_offset_unit_z + z_shift;

            if (plasma_injector.insideBounds(x, y, z)) {
                ParticleReal const ux = ptr_ux.get()[i]*momentum_unit_x/mass;
                ParticleReal const uz = ptr_uz.get()[i]*momentum_unit_z/mass;
                ParticleReal uy = 0.0_prt;
                if (ps["momentum"].contains("y")) {
                    uy = ptr_uy.get()[i]*momentum_unit_y/mass;
                }
                CheckAndAddParticle(x, y, z, ux, uy, uz, weight,
                                    particle_x,  particle_y,  particle_z,
                                    particle_ux, particle_uy, particle_uz,
                                    particle_w, static_cast<amrex::Real>(t_lab));
            }
        }
        auto const np = particle_z.size();
        if (np < npart) {
            ablastr::warn_manager::WMRecordWarning("Species",
                "Simulation box doesn't cover all particles",
                ablastr::warn_manager::WarnPriority::high);
        }
    } // IO Processor
    auto const np = static_cast<long>(particle_z.size());
    const amrex::Vector<ParticleReal> xp(particle_x.data(), particle_x.data() + np);
    const amrex::Vector<ParticleReal> yp(particle_y.data(), particle_y.data() + np);
    const amrex::Vector<ParticleReal> zp(particle_z.data(), particle_z.data() + np);
    const amrex::Vector<ParticleReal> uxp(particle_ux.data(), particle_ux.data() + np);
    const amrex::Vector<ParticleReal> uyp(particle_uy.data(), particle_uy.data() + np);
    const amrex::Vector<ParticleReal> uzp(particle_uz.data(), particle_uz.data() + np);

    amrex::Vector<amrex::Vector<ParticleReal>> attr;
    const amrex::Vector<ParticleReal> wp(particle_w.data(), particle_w.data() + np);
    attr.push_back(wp);

    const amrex::Vector<amrex::Vector<int>> attr_int;

    AddNParticles(0, np, xp,  yp,  zp, uxp, uyp, uzp,
                  1, attr, 0, attr_int, 1);
#endif // WARPX_USE_OPENPMD

    ignore_unused(plasma_injector, q_tot, z_shift);
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
                                       particle_comps, particle_icomps,
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
PhysicalParticleContainer::CheckAndAddParticle (
    ParticleReal x, ParticleReal y, ParticleReal z,
    ParticleReal ux, ParticleReal uy, ParticleReal uz,
    ParticleReal weight,
    Gpu::HostVector<ParticleReal>& particle_x,
    Gpu::HostVector<ParticleReal>& particle_y,
    Gpu::HostVector<ParticleReal>& particle_z,
    Gpu::HostVector<ParticleReal>& particle_ux,
    Gpu::HostVector<ParticleReal>& particle_uy,
    Gpu::HostVector<ParticleReal>& particle_uz,
    Gpu::HostVector<ParticleReal>& particle_w,
    Real t_lab) const
{
    if (WarpX::gamma_boost > 1.) {
        MapParticletoBoostedFrame(x, y, z, ux, uy, uz, t_lab);
    }
    particle_x.push_back(x);
    particle_y.push_back(y);
    particle_z.push_back(z);
    particle_ux.push_back(ux);
    particle_uy.push_back(uy);
    particle_uz.push_back(uz);
    particle_w.push_back(weight);
}

void
PhysicalParticleContainer::AddParticles (int lev)
{
    WARPX_PROFILE("PhysicalParticleContainer::AddParticles()");

    for (auto const& plasma_injector : plasma_injectors) {

        if (plasma_injector->add_single_particle) {
            if (WarpX::gamma_boost > 1.) {
                MapParticletoBoostedFrame(plasma_injector->single_particle_pos[0],
                                          plasma_injector->single_particle_pos[1],
                                          plasma_injector->single_particle_pos[2],
                                          plasma_injector->single_particle_u[0],
                                          plasma_injector->single_particle_u[1],
                                          plasma_injector->single_particle_u[2]);
            }
            const amrex::Vector<ParticleReal> xp = {plasma_injector->single_particle_pos[0]};
            const amrex::Vector<ParticleReal> yp = {plasma_injector->single_particle_pos[1]};
            const amrex::Vector<ParticleReal> zp = {plasma_injector->single_particle_pos[2]};
            const amrex::Vector<ParticleReal> uxp = {plasma_injector->single_particle_u[0]};
            const amrex::Vector<ParticleReal> uyp = {plasma_injector->single_particle_u[1]};
            const amrex::Vector<ParticleReal> uzp = {plasma_injector->single_particle_u[2]};
            const amrex::Vector<amrex::Vector<ParticleReal>> attr = {{plasma_injector->single_particle_weight}};
            const amrex::Vector<amrex::Vector<int>> attr_int;
            AddNParticles(lev, 1, xp, yp, zp, uxp, uyp, uzp,
                          1, attr, 0, attr_int, 0);
            return;
        }

        if (plasma_injector->add_multiple_particles) {
            if (WarpX::gamma_boost > 1.) {
                for (int i=0 ; i < plasma_injector->multiple_particles_pos_x.size() ; i++) {
                    MapParticletoBoostedFrame(plasma_injector->multiple_particles_pos_x[i],
                                              plasma_injector->multiple_particles_pos_y[i],
                                              plasma_injector->multiple_particles_pos_z[i],
                                              plasma_injector->multiple_particles_ux[i],
                                              plasma_injector->multiple_particles_uy[i],
                                              plasma_injector->multiple_particles_uz[i]);
                }
            }
            amrex::Vector<amrex::Vector<ParticleReal>> attr;
            attr.push_back(plasma_injector->multiple_particles_weight);
            const amrex::Vector<amrex::Vector<int>> attr_int;
            AddNParticles(lev, static_cast<int>(plasma_injector->multiple_particles_pos_x.size()),
                          plasma_injector->multiple_particles_pos_x,
                          plasma_injector->multiple_particles_pos_y,
                          plasma_injector->multiple_particles_pos_z,
                          plasma_injector->multiple_particles_ux,
                          plasma_injector->multiple_particles_uy,
                          plasma_injector->multiple_particles_uz,
                          1, attr, 0, attr_int, 0);
        }

        if (plasma_injector->gaussian_beam) {
            AddGaussianBeam(*plasma_injector);
        }

        if (plasma_injector->external_file) {
            AddPlasmaFromFile(*plasma_injector,
                              plasma_injector->q_tot,
                              plasma_injector->z_shift);
        }

        if ( plasma_injector->doInjection() ) {
            AddPlasma(*plasma_injector, lev);
        }
    }
}

void
PhysicalParticleContainer::AddPlasma (PlasmaInjector const& plasma_injector, int lev, RealBox part_realbox)
{
    WARPX_PROFILE("PhysicalParticleContainer::AddPlasma()");

    // If no part_realbox is provided, initialize particles in the whole domain
    const Geometry& geom = Geom(lev);
    if (!part_realbox.ok()) { part_realbox = geom.ProbDomain(); }

    const int num_ppc = plasma_injector.num_particles_per_cell;
#ifdef WARPX_DIM_RZ
    const Real rmax = std::min(plasma_injector.xmax, part_realbox.hi(0));
#endif

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();

    defineAllParticleTiles();

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    Box fine_injection_box;
    amrex::IntVect rrfac(AMREX_D_DECL(1,1,1));
    const bool refine_injection = findRefinedInjectionBox(fine_injection_box, rrfac);

    InjectorPosition* inj_pos = plasma_injector.getInjectorPosition();
    InjectorDensity*  inj_rho = plasma_injector.getInjectorDensity();
    InjectorMomentum* inj_mom = plasma_injector.getInjectorMomentumDevice();
    const Real gamma_boost = WarpX::gamma_boost;
    const Real beta_boost = WarpX::beta_boost;
    const Real t = WarpX::GetInstance().gett_new(lev);
    const Real density_min = plasma_injector.density_min;
    const Real density_max = plasma_injector.density_max;

#ifdef WARPX_DIM_RZ
    const int nmodes = WarpX::n_rz_azimuthal_modes;
    const bool radially_weighted = plasma_injector.radially_weighted;
#endif

    auto n_user_int_attribs = static_cast<int>(m_user_int_attribs.size());
    auto n_user_real_attribs = static_cast<int>(m_user_real_attribs.size());
    const PlasmaParserWrapper plasma_parser_wrapper (m_user_int_attribs.size(),
                                                     m_user_real_attribs.size(),
                                                     m_user_int_attrib_parser,
                                                     m_user_real_attrib_parser);

    MFItInfo info;
    if (do_tiling && Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (not WarpX::serialize_initial_conditions)
#endif
    for (MFIter mfi = MakeMFIter(lev, info); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        const Box& tile_box = mfi.tilebox();
        const RealBox tile_realbox = WarpX::getRealBox(tile_box, lev);

        // Find the cells of part_box that overlap with tile_realbox
        // If there is no overlap, just go to the next tile in the loop
        RealBox overlap_realbox;
        Box overlap_box;
        IntVect shifted;
        const bool no_overlap = find_overlap(tile_realbox, part_realbox, dx, problo, overlap_realbox, overlap_box, shifted);
        if (no_overlap) {
            continue; // Go to the next tile
        }

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        const GpuArray<Real,AMREX_SPACEDIM> overlap_corner
            {AMREX_D_DECL(overlap_realbox.lo(0),
                          overlap_realbox.lo(1),
                          overlap_realbox.lo(2))};

        // count the number of particles that each cell in overlap_box could add
        Gpu::DeviceVector<amrex::Long> counts(overlap_box.numPts(), 0);
        Gpu::DeviceVector<amrex::Long> offset(overlap_box.numPts());
        auto *pcounts = counts.data();
        Box fine_overlap_box; // default Box is NOT ok().
        if (refine_injection) {
            fine_overlap_box = overlap_box & amrex::shift(fine_injection_box, -shifted);
        }
        amrex::ParallelFor(overlap_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const IntVect iv(AMREX_D_DECL(i, j, k));
            auto lo = getCellCoords(overlap_corner, dx, {0._rt, 0._rt, 0._rt}, iv);
            auto hi = getCellCoords(overlap_corner, dx, {1._rt, 1._rt, 1._rt}, iv);

            lo.z = applyBallisticCorrection(lo, inj_mom, gamma_boost, beta_boost, t);
            hi.z = applyBallisticCorrection(hi, inj_mom, gamma_boost, beta_boost, t);

            if (inj_pos->overlapsWith(lo, hi))
            {
                auto index = overlap_box.index(iv);
                const amrex::Long r = (fine_overlap_box.ok() && fine_overlap_box.contains(iv))?
                    (AMREX_D_TERM(rrfac[0],*rrfac[1],*rrfac[2])) : (1);
                pcounts[index] = num_ppc*r;
                // update pcount by checking if cell-corners or cell-center
                // has non-zero density
                const auto xlim = GpuArray<Real, 3>{lo.x,(lo.x+hi.x)/2._rt,hi.x};
                const auto ylim = GpuArray<Real, 3>{lo.y,(lo.y+hi.y)/2._rt,hi.y};
                const auto zlim = GpuArray<Real, 3>{lo.z,(lo.z+hi.z)/2._rt,hi.z};

                const auto checker = [&](){
                    for (const auto& x : xlim) {
                        for (const auto& y : ylim) {
                            for (const auto& z : zlim) {
                                if (inj_pos->insideBounds(x,y,z) and (inj_rho->getDensity(x,y,z) > 0) ) {
                                    return 1;
                                }
                            }
                        }
                    }
                    return 0;
                };
                pcounts[index] = checker() ? num_ppc*r : 0;
            }
            amrex::ignore_unused(j,k);
        });

        // Max number of new particles. All of them are created,
        // and invalid ones are then discarded
        const amrex::Long max_new_particles = Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

        // Update NextID to include particles created in this function
        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (add_plasma_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid+max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pid + max_new_particles < LongParticleIds::LastParticleID,
            "ERROR: overflow on particle id numbers");

        const int cpuid = ParallelDescriptor::MyProc();

        auto& particle_tile = GetParticles(lev)[std::make_pair(grid_id,tile_id)];

        if ( (NumRuntimeRealComps()>0) || (NumRuntimeIntComps()>0) ) {
            DefineAndReturnParticleTile(lev, grid_id, tile_id);
        }

        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        auto const new_size = old_size + max_new_particles;
        particle_tile.resize(new_size);

        auto& soa = particle_tile.GetStructOfArrays();
        GpuArray<ParticleReal*,PIdx::nattribs> pa;
        for (int ia = 0; ia < PIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        uint64_t * AMREX_RESTRICT pa_idcpu = soa.GetIdCPUData().data() + old_size;

        PlasmaParserHelper plasma_parser_helper (soa, old_size, m_user_int_attribs, m_user_real_attribs, particle_icomps, particle_comps, plasma_parser_wrapper);
        int** pa_user_int_data = plasma_parser_helper.getUserIntDataPtrs();
        ParticleReal** pa_user_real_data = plasma_parser_helper.getUserRealDataPtrs();
        amrex::ParserExecutor<7> const* user_int_parserexec_data = plasma_parser_helper.getUserIntParserExecData();
        amrex::ParserExecutor<7> const* user_real_parserexec_data = plasma_parser_helper.getUserRealParserExecData();

        int* pi = nullptr;
        if (do_field_ionization) {
            pi = soa.GetIntData(particle_icomps["ionizationLevel"]).data() + old_size;
        }

#ifdef WARPX_QED
        const QEDHelper qed_helper(soa, old_size, particle_comps,
                                   has_quantum_sync(), has_breit_wheeler(),
                                   m_shr_p_qs_engine, m_shr_p_bw_engine);
#endif

        const bool loc_do_field_ionization = do_field_ionization;
        const int loc_ionization_initial_level = ionization_initial_level;

        // Loop over all new particles and inject them (creates too many
        // particles, in particular does not consider xmin, xmax etc.).
        // The invalid ones are given negative ID and are deleted during the
        // next redistribute.
        auto *const poffset = offset.data();
#ifdef WARPX_DIM_RZ
        const bool rz_random_theta = m_rz_random_theta;
#endif
        amrex::ParallelForRNG(overlap_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const IntVect iv = IntVect(AMREX_D_DECL(i, j, k));
            amrex::ignore_unused(j,k);
            const auto index = overlap_box.index(iv);
#ifdef WARPX_DIM_RZ
            Real theta_offset = 0._rt;
            if (rz_random_theta) { theta_offset = amrex::Random(engine) * 2._rt * MathConst::pi; }
#endif

            const Real scale_fac = compute_scale_fac_volume(dx, pcounts[index]);
            for (int i_part = 0; i_part < pcounts[index]; ++i_part)
            {
                long ip = poffset[index] + i_part;
                pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid+ip, cpuid);
                const XDim3 r = (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) ?
                  // In the refined injection region: use refinement ratio `rrfac`
                  inj_pos->getPositionUnitBox(i_part, rrfac, engine) :
                  // Otherwise: use 1 as the refinement ratio
                  inj_pos->getPositionUnitBox(i_part, amrex::IntVect::TheUnitVector(), engine);
                auto pos = getCellCoords(overlap_corner, dx, r, iv);

#if defined(WARPX_DIM_3D)
                bool const box_contains = tile_realbox.contains(XDim3{pos.x,pos.y,pos.z});
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ignore_unused(k);
                bool const box_contains = tile_realbox.contains(XDim3{pos.x,pos.z,0.0_rt});
#elif defined(WARPX_DIM_1D_Z)
                amrex::ignore_unused(j,k);
                bool const box_contains = tile_realbox.contains(XDim3{pos.z,0.0_rt,0.0_rt});
#endif
                if (!box_contains) {
                    ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,qed_helper
#endif
                                                   );
                    continue;
                }

                // Save the x and y values to use in the insideBounds checks.
                // This is needed with WARPX_DIM_RZ since x and y are modified.
                const Real xb = pos.x;
                const Real yb = pos.y;

#ifdef WARPX_DIM_RZ
                // Replace the x and y, setting an angle theta.
                // These x and y are used to get the momentum and density
                // With only 1 mode, the angle doesn't matter so
                // choose it randomly.
                const Real theta = (nmodes == 1 && rz_random_theta)?
                    (2._rt*MathConst::pi*amrex::Random(engine)):
                    (2._rt*MathConst::pi*r.y + theta_offset);
                pos.x = xb*std::cos(theta);
                pos.y = xb*std::sin(theta);
#endif

                Real dens;
                XDim3 u;
                if (gamma_boost == 1._rt) {
                    // Lab-frame simulation
                    // If the particle is not within the species's
                    // xmin, xmax, ymin, ymax, zmin, zmax, go to
                    // the next generated particle.

                    // include ballistic correction for plasma species with bulk motion
                    const Real z0 = applyBallisticCorrection(pos, inj_mom, gamma_boost,
                                                             beta_boost, t);
                    if (!inj_pos->insideBounds(xb, yb, z0)) {
                        ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,qed_helper
#endif
                                                   );
                        continue;
                    }

                    u = inj_mom->getMomentum(pos.x, pos.y, z0, engine);
                    dens = inj_rho->getDensity(pos.x, pos.y, z0);

                    // Remove particle if density below threshold
                    if ( dens < density_min ){
                        ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,qed_helper
#endif
                                                   );
                        continue;
                    }
                    // Cut density if above threshold
                    dens = amrex::min(dens, density_max);
                } else {
                    // Boosted-frame simulation
                    const Real z0_lab = applyBallisticCorrection(pos, inj_mom, gamma_boost,
                                                                 beta_boost, t);

                    // If the particle is not within the lab-frame zmin, zmax, etc.
                    // go to the next generated particle.
                    if (!inj_pos->insideBounds(xb, yb, z0_lab)) {
                        ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,qed_helper
#endif
                                                   );
                        continue;
                    }
                    // call `getDensity` with lab-frame parameters
                    dens = inj_rho->getDensity(pos.x, pos.y, z0_lab);
                    // Remove particle if density below threshold
                    if ( dens < density_min ){
                        ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,qed_helper
#endif
                                                   );
                        continue;
                    }
                    // Cut density if above threshold
                    dens = amrex::min(dens, density_max);

                    // get the full momentum, including thermal motion
                    u = inj_mom->getMomentum(pos.x, pos.y, 0._rt, engine);
                    const Real gamma_lab = std::sqrt( 1._rt+(u.x*u.x+u.y*u.y+u.z*u.z) );
                    const Real betaz_lab = u.z/(gamma_lab);

                    // At this point u and dens are the lab-frame quantities
                    // => Perform Lorentz transform
                    dens = gamma_boost * dens * ( 1.0_rt - beta_boost*betaz_lab );
                    u.z = gamma_boost * ( u.z -beta_boost*gamma_lab );
                }

                if (loc_do_field_ionization) {
                    pi[ip] = loc_ionization_initial_level;
                }

#ifdef WARPX_QED
                if(qed_helper.has_quantum_sync){
                    qed_helper.p_optical_depth_QSR[ip] = qed_helper.quantum_sync_get_opt(engine);
                }

                if(qed_helper.has_breit_wheeler){
                    qed_helper.p_optical_depth_BW[ip] = qed_helper.breit_wheeler_get_opt(engine);
                }
#endif
                // Initialize user-defined integers with user-defined parser
                for (int ia = 0; ia < n_user_int_attribs; ++ia) {
                    pa_user_int_data[ia][ip] = static_cast<int>(user_int_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t));
                }
                // Initialize user-defined real attributes with user-defined parser
                for (int ia = 0; ia < n_user_real_attribs; ++ia) {
                    pa_user_real_data[ia][ip] = user_real_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t);
                }

                u.x *= PhysConst::c;
                u.y *= PhysConst::c;
                u.z *= PhysConst::c;

                Real weight = dens;
                weight *= scale_fac;

#ifdef WARPX_DIM_RZ
                if (radially_weighted) {
                    weight *= 2._rt*MathConst::pi*xb;
                } else {
                    // This is not correct since it might shift the particle
                    // out of the local grid
                    pos.x = std::sqrt(xb*rmax);
                    weight *= dx[0];
                }
#endif
                pa[PIdx::w ][ip] = weight;
                pa[PIdx::ux][ip] = u.x;
                pa[PIdx::uy][ip] = u.y;
                pa[PIdx::uz][ip] = u.z;

#if defined(WARPX_DIM_3D)
                pa[PIdx::x][ip] = pos.x;
                pa[PIdx::y][ip] = pos.y;
                pa[PIdx::z][ip] = pos.z;
#elif defined(WARPX_DIM_XZ)
                pa[PIdx::x][ip] = pos.x;
                pa[PIdx::z][ip] = pos.z;
#elif defined(WARPX_DIM_RZ)
                pa[PIdx::theta][ip] = theta;
                pa[PIdx::x][ip] = xb;
                pa[PIdx::z][ip] = pos.z;
#elif defined(WARPX_DIM_1D_Z)
                pa[PIdx::z][ip] = pos.z;
#endif
            }
        });

        amrex::Gpu::synchronize();

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Remove particles that are inside the embedded boundaries
#ifdef AMREX_USE_EB
    if (EB::enabled())
    {
        using warpx::fields::FieldType;
        auto & warpx = WarpX::GetInstance();
        scrapeParticlesAtEB(
            *this,
            warpx.m_fields.get_mr_levels(FieldType::distance_to_eb, warpx.finestLevel()),
            ParticleBoundaryProcess::Absorb());
    }
#endif

    // The function that calls this is responsible for redistributing particles.
}

void
PhysicalParticleContainer::AddPlasmaFlux (PlasmaInjector const& plasma_injector, amrex::Real dt)
{
    WARPX_PROFILE("PhysicalParticleContainer::AddPlasmaFlux()");

    const Geometry& geom = Geom(0);
    const amrex::RealBox& part_realbox = geom.ProbDomain();

    const amrex::Real num_ppc_real = plasma_injector.num_particles_per_cell_real;
#ifdef WARPX_DIM_RZ
    const Real rmax = std::min(plasma_injector.xmax, geom.ProbDomain().hi(0));
#endif

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();

#ifdef AMREX_USE_EB
    bool const inject_from_eb = plasma_injector.m_inject_from_eb; // whether to inject from EB or from a plane
    // Extract data structures for embedded boundaries
    amrex::FabArray<amrex::EBCellFlagFab> const* eb_flag = nullptr;
    amrex::MultiCutFab const* eb_bnd_area = nullptr;
    amrex::MultiCutFab const* eb_bnd_normal = nullptr;
    amrex::MultiCutFab const* eb_bnd_cent = nullptr;
    if (inject_from_eb) {
        amrex::EBFArrayBoxFactory const& eb_box_factory = WarpX::GetInstance().fieldEBFactory(0);
        eb_flag = &eb_box_factory.getMultiEBCellFlagFab();
        eb_bnd_area = &eb_box_factory.getBndryArea();
        eb_bnd_normal = &eb_box_factory.getBndryNormal();
        eb_bnd_cent = &eb_box_factory.getBndryCent();
    }
#endif

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(0);

    // Create temporary particle container to which particles will be added;
    // we will then call Redistribute on this new container and finally
    // add the new particles to the original container.
    PhysicalParticleContainer tmp_pc(&WarpX::GetInstance());
    for (int ic = 0; ic < NumRuntimeRealComps(); ++ic) { tmp_pc.AddRealComp(false); }
    for (int ic = 0; ic < NumRuntimeIntComps(); ++ic) { tmp_pc.AddIntComp(false); }
    tmp_pc.defineAllParticleTiles();

    Box fine_injection_box;
    amrex::IntVect rrfac(AMREX_D_DECL(1,1,1));
    const bool refine_injection = findRefinedInjectionBox(fine_injection_box, rrfac);

    InjectorPosition* flux_pos = plasma_injector.getInjectorFluxPosition();
    InjectorFlux*  inj_flux = plasma_injector.getInjectorFlux();
    InjectorMomentum* inj_mom = plasma_injector.getInjectorMomentumDevice();
    constexpr int level_zero = 0;
    const amrex::Real t = WarpX::GetInstance().gett_new(level_zero);

#ifdef WARPX_DIM_RZ
    const int nmodes = WarpX::n_rz_azimuthal_modes;
    const bool rz_random_theta = m_rz_random_theta;
    const bool radially_weighted = plasma_injector.radially_weighted;
#endif

    auto n_user_int_attribs = static_cast<int>(m_user_int_attribs.size());
    auto n_user_real_attribs = static_cast<int>(m_user_real_attribs.size());
    const PlasmaParserWrapper plasma_parser_wrapper (m_user_int_attribs.size(),
                                                     m_user_real_attribs.size(),
                                                     m_user_int_attrib_parser,
                                                     m_user_real_attrib_parser);

    MFItInfo info;
    if (do_tiling && Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi = MakeMFIter(0, info); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        const Box& tile_box = mfi.tilebox();
        const RealBox tile_realbox = WarpX::getRealBox(tile_box, 0);

        // Find the cells of part_realbox that overlap with tile_realbox
        // If there is no overlap, just go to the next tile in the loop
        RealBox overlap_realbox;
        Box overlap_box;
        IntVect shifted;
#ifdef AMREX_USE_EB
        if (inject_from_eb) {
            // Injection from EB
            const amrex::FabType fab_type = (*eb_flag)[mfi].getType(tile_box);
            if (fab_type == amrex::FabType::regular) { continue; } // Go to the next tile
            if (fab_type == amrex::FabType::covered) { continue; } // Go to the next tile
            overlap_box = tile_box;
            overlap_realbox = part_realbox;
        } else
#endif
        {
            // Injection from a plane
            const bool no_overlap = find_overlap_flux(tile_realbox, part_realbox, dx, problo, plasma_injector, overlap_realbox, overlap_box, shifted);
            if (no_overlap) { continue; } // Go to the next tile
        }

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        const GpuArray<Real,AMREX_SPACEDIM> overlap_corner
            {AMREX_D_DECL(overlap_realbox.lo(0),
                          overlap_realbox.lo(1),
                          overlap_realbox.lo(2))};

        // count the number of particles that each cell in overlap_box could add
        Gpu::DeviceVector<int> counts(overlap_box.numPts(), 0);
        Gpu::DeviceVector<int> offset(overlap_box.numPts());
        auto *pcounts = counts.data();
        const int flux_normal_axis = plasma_injector.flux_normal_axis;
        Box fine_overlap_box; // default Box is NOT ok().
        if (refine_injection) {
            fine_overlap_box = overlap_box & amrex::shift(fine_injection_box, -shifted);
        }

#ifdef AMREX_USE_EB
        // Extract data structures for embedded boundaries
        amrex::Array4<const typename FabArray<EBCellFlagFab>::value_type> eb_flag_arr;
        amrex::Array4<const amrex::Real> eb_bnd_area_arr;
        amrex::Array4<const amrex::Real> eb_bnd_normal_arr;
        amrex::Array4<const amrex::Real> eb_bnd_cent_arr;
        if (inject_from_eb) {
            eb_flag_arr = eb_flag->array(mfi);
            eb_bnd_area_arr = eb_bnd_area->array(mfi);
            eb_bnd_normal_arr = eb_bnd_normal->array(mfi);
            eb_bnd_cent_arr = eb_bnd_cent->array(mfi);
        }
#endif

        amrex::ParallelForRNG(overlap_box, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const IntVect iv(AMREX_D_DECL(i, j, k));
            amrex::ignore_unused(j,k);

            // Determine the number of macroparticles to inject in this cell (num_ppc_int)
#ifdef AMREX_USE_EB
            amrex::Real num_ppc_real_in_this_cell = num_ppc_real; // user input: number of macroparticles per cell
            if (inject_from_eb) {
                // Injection from EB
                // Skip cells that are not partially covered by the EB
                if (eb_flag_arr(i,j,k).isRegular() || eb_flag_arr(i,j,k).isCovered()) { return; }
                // Scale by the (normalized) area of the EB surface in this cell
                num_ppc_real_in_this_cell *= eb_bnd_area_arr(i,j,k);
            }
#else
            amrex::Real const num_ppc_real_in_this_cell = num_ppc_real; // user input: number of macroparticles per cell
#endif
            // Skip cells that do not overlap with the bounds specified by the user (xmin/xmax, ymin/ymax, zmin/zmax)
            auto lo = getCellCoords(overlap_corner, dx, {0._rt, 0._rt, 0._rt}, iv);
            auto hi = getCellCoords(overlap_corner, dx, {1._rt, 1._rt, 1._rt}, iv);
            if (!flux_pos->overlapsWith(lo, hi)) { return; }

            auto index = overlap_box.index(iv);
            // Take into account refined injection region
            int r = 1;
            if (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) {
                r = compute_area_weights(rrfac, flux_normal_axis);
            }
            const int num_ppc_int = static_cast<int>(num_ppc_real_in_this_cell*r + amrex::Random(engine));
            pcounts[index] = num_ppc_int;

            amrex::ignore_unused(j,k);
        });

        // Max number of new particles. All of them are created,
        // and invalid ones are then discarded
        const amrex::Long max_new_particles = Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

        // Update NextID to include particles created in this function
        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (add_plasma_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid+max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pid + max_new_particles < LongParticleIds::LastParticleID,
            "overflow on particle id numbers");

        const int cpuid = ParallelDescriptor::MyProc();

        auto& particle_tile = tmp_pc.DefineAndReturnParticleTile(0, grid_id, tile_id);

        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        auto const new_size = old_size + max_new_particles;
        particle_tile.resize(new_size);

        auto& soa = particle_tile.GetStructOfArrays();
        GpuArray<ParticleReal*,PIdx::nattribs> pa;
        for (int ia = 0; ia < PIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        uint64_t * AMREX_RESTRICT pa_idcpu = soa.GetIdCPUData().data() + old_size;

        PlasmaParserHelper plasma_parser_helper (soa, old_size, m_user_int_attribs, m_user_real_attribs, particle_icomps, particle_comps, plasma_parser_wrapper);
        int** pa_user_int_data = plasma_parser_helper.getUserIntDataPtrs();
        ParticleReal** pa_user_real_data = plasma_parser_helper.getUserRealDataPtrs();
        amrex::ParserExecutor<7> const* user_int_parserexec_data = plasma_parser_helper.getUserIntParserExecData();
        amrex::ParserExecutor<7> const* user_real_parserexec_data = plasma_parser_helper.getUserRealParserExecData();

        int* p_ion_level = nullptr;
        if (do_field_ionization) {
            p_ion_level = soa.GetIntData(particle_icomps["ionizationLevel"]).data() + old_size;
        }

#ifdef WARPX_QED
        const QEDHelper qed_helper(soa, old_size, particle_comps,
                                   has_quantum_sync(), has_breit_wheeler(),
                                   m_shr_p_qs_engine, m_shr_p_bw_engine);
#endif

        const bool loc_do_field_ionization = do_field_ionization;
        const int loc_ionization_initial_level = ionization_initial_level;
#ifdef WARPX_DIM_RZ
        int const loc_flux_normal_axis = plasma_injector.flux_normal_axis;
#endif

        // Loop over all new particles and inject them (creates too many
        // particles, in particular does not consider xmin, xmax etc.).
        // The invalid ones are given negative ID and are deleted during the
        // next redistribute.
        auto *const poffset = offset.data();
        amrex::ParallelForRNG(overlap_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const IntVect iv = IntVect(AMREX_D_DECL(i, j, k));
            amrex::ignore_unused(j,k);
            const auto index = overlap_box.index(iv);

            Real scale_fac;
#ifdef AMREX_USE_EB
            if (inject_from_eb) {
                scale_fac = compute_scale_fac_area_eb(dx, num_ppc_real, eb_bnd_normal_arr, i, j, k );
            } else
#endif
            {
                scale_fac = compute_scale_fac_area_plane(dx, num_ppc_real, flux_normal_axis);
            }

            if (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) {
                scale_fac /= compute_area_weights(rrfac, flux_normal_axis);
            }

            for (int i_part = 0; i_part < pcounts[index]; ++i_part)
            {
                const long ip = poffset[index] + i_part;
                pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid+ip, cpuid);

                // Determine the position of the particle within the cell
                XDim3 pos;
                XDim3 r;
#ifdef AMREX_USE_EB
                if (inject_from_eb) {
#if defined(WARPX_DIM_3D)
                    pos.x = overlap_corner[0] + (iv[0] + 0.5_rt + eb_bnd_cent_arr(i,j,k,0))*dx[0];
                    pos.y = overlap_corner[1] + (iv[1] + 0.5_rt + eb_bnd_cent_arr(i,j,k,1))*dx[1];
                    pos.z = overlap_corner[2] + (iv[2] + 0.5_rt + eb_bnd_cent_arr(i,j,k,2))*dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    pos.x = overlap_corner[0] + (iv[0] + 0.5_rt + eb_bnd_cent_arr(i,j,k,0))*dx[0];
                    pos.y = 0.0_rt;
                    pos.z = overlap_corner[1] + (iv[1] + 0.5_rt + eb_bnd_cent_arr(i,j,k,1))*dx[1];
#endif
                } else
#endif
                {
                    // Injection from a plane
                    // This assumes the flux_pos is of type InjectorPositionRandomPlane
                    r = (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) ?
                        // In the refined injection region: use refinement ratio `rrfac`
                        flux_pos->getPositionUnitBox(i_part, rrfac, engine) :
                        // Otherwise: use 1 as the refinement ratio
                        flux_pos->getPositionUnitBox(i_part, amrex::IntVect::TheUnitVector(), engine);
                    pos = getCellCoords(overlap_corner, dx, r, iv);
                }
                auto ppos = PDim3(pos);

                // inj_mom would typically be InjectorMomentumGaussianFlux
                XDim3 u;
                u = inj_mom->getMomentum(pos.x, pos.y, pos.z, engine);
                auto pu = PDim3(u);

                pu.x *= PhysConst::c;
                pu.y *= PhysConst::c;
                pu.z *= PhysConst::c;

                // The containsInclusive is used to allow the case of the flux surface
                // being on the boundary of the domain. After the UpdatePosition below,
                // the particles will be within the domain.
#if defined(WARPX_DIM_3D)
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.x,ppos.y,ppos.z})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ignore_unused(k);
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.x,ppos.z,0.0_prt})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#else
                amrex::ignore_unused(j,k);
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.z,0.0_prt,0.0_prt})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#endif
                // Lab-frame simulation
                // If the particle's initial position is not within or on the species's
                // xmin, xmax, ymin, ymax, zmin, zmax, go to the next generated particle.
                if (!flux_pos->insideBoundsInclusive(ppos.x, ppos.y, ppos.z)) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }

#ifdef AMREX_USE_EB
                if (inject_from_eb) {
                    // Injection from EB: rotate momentum according to the normal of the EB surface
                    // (The above code initialized the momentum by assuming that z is the direction
                    // normal to the EB surface. Thus we need to rotate from z to the normal.)
                    rotate_momentum_eb(pu, eb_bnd_normal_arr, i, j , k);
                }
#endif

#ifdef WARPX_DIM_RZ
                // Conversion from cylindrical to Cartesian coordinates
                // Replace the x and y, setting an angle theta.
                // These x and y are used to get the momentum and flux
                // With only 1 mode, the angle doesn't matter so
                // choose it randomly.
                const Real theta = (nmodes == 1 && rz_random_theta)?
                    (2._prt*MathConst::pi*amrex::Random(engine)):
                    (2._prt*MathConst::pi*r.y);
                Real const cos_theta = std::cos(theta);
                Real const sin_theta = std::sin(theta);
                // Rotate the position
                const amrex::Real radial_position = ppos.x;
                ppos.x = radial_position*cos_theta;
                ppos.y = radial_position*sin_theta;
                if ((loc_flux_normal_axis != 2)
#ifdef AMREX_USE_EB
                    || (inject_from_eb)
#endif
                    ) {
                    // Rotate the momentum
                    // This because, when the flux direction is e.g. "r"
                    // the `inj_mom` objects generates a v*Gaussian distribution
                    // along the Cartesian "x" direction by default. This
                    // needs to be rotated along "r".
                    const Real ur = pu.x;
                    const Real ut = pu.y;
                    pu.x = cos_theta*ur - sin_theta*ut;
                    pu.y = sin_theta*ur + cos_theta*ut;
                }
#endif
                const Real flux = inj_flux->getFlux(ppos.x, ppos.y, ppos.z, t);
                // Remove particle if flux is negative or 0
                if (flux <= 0) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }

                if (loc_do_field_ionization) {
                    p_ion_level[ip] = loc_ionization_initial_level;
                }

#ifdef WARPX_QED
                if(qed_helper.has_quantum_sync){
                    qed_helper.p_optical_depth_QSR[ip] = qed_helper.quantum_sync_get_opt(engine);
                }

                if(qed_helper.has_breit_wheeler){
                    qed_helper.p_optical_depth_BW[ip] = qed_helper.breit_wheeler_get_opt(engine);
                }
#endif

                // Initialize user-defined integers with user-defined parser
                for (int ia = 0; ia < n_user_int_attribs; ++ia) {
                    pa_user_int_data[ia][ip] = static_cast<int>(user_int_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t));
                }
                // Initialize user-defined real attributes with user-defined parser
                for (int ia = 0; ia < n_user_real_attribs; ++ia) {
                    pa_user_real_data[ia][ip] = user_real_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t);
                }

#ifdef WARPX_DIM_RZ
                // The particle weight is proportional to the user-specified
                // flux and the emission surface within
                // one cell (captured partially by `scale_fac`).
                // For cylindrical emission (flux_normal_axis==0
                // or flux_normal_axis==2), the emission surface depends on
                // the radius ; thus, the calculation is finalized here
                Real t_weight = flux * scale_fac * dt;
                if (loc_flux_normal_axis != 1) {
                    if (radially_weighted) {
                         t_weight *= 2._rt*MathConst::pi*radial_position;
                    } else {
                         // This is not correct since it might shift the particle
                         // out of the local grid
                         ppos.x = std::sqrt(radial_position*rmax);
                         t_weight *= dx[0];
                    }
                }
                const Real weight = t_weight;
#else
                const Real weight = flux * scale_fac * dt;
#endif
                pa[PIdx::w ][ip] = weight;
                pa[PIdx::ux][ip] = pu.x;
                pa[PIdx::uy][ip] = pu.y;
                pa[PIdx::uz][ip] = pu.z;

                // Update particle position by a random `t_fract`
                // so as to produce a continuous-looking flow of particles
                const amrex::Real t_fract = amrex::Random(engine)*dt;
                UpdatePosition(ppos.x, ppos.y, ppos.z, pu.x, pu.y, pu.z, t_fract);

#if defined(WARPX_DIM_3D)
                pa[PIdx::x][ip] = ppos.x;
                pa[PIdx::y][ip] = ppos.y;
                pa[PIdx::z][ip] = ppos.z;
#elif defined(WARPX_DIM_RZ)
                pa[PIdx::theta][ip] = std::atan2(ppos.y, ppos.x);
                pa[PIdx::x][ip] = std::sqrt(ppos.x*ppos.x + ppos.y*ppos.y);
                pa[PIdx::z][ip] = ppos.z;
#elif defined(WARPX_DIM_XZ)
                pa[PIdx::x][ip] = ppos.x;
                pa[PIdx::z][ip] = ppos.z;
#elif defined(WARPX_DIM_1D_Z)
                pa[PIdx::z][ip] = ppos.z;
#endif
            }
        });

        amrex::Gpu::synchronize();

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Remove particles that are inside the embedded boundaries
#ifdef AMREX_USE_EB
    if (EB::enabled())
    {
        using warpx::fields::FieldType;
        auto & warpx = WarpX::GetInstance();
        scrapeParticlesAtEB(
            tmp_pc,
            warpx.m_fields.get_mr_levels(FieldType::distance_to_eb, warpx.finestLevel()),
            ParticleBoundaryProcess::Absorb());
    }
#endif

    // Redistribute the new particles that were added to the temporary container.
    // (This eliminates invalid particles, and makes sure that particles
    // are in the right tile.)
    tmp_pc.Redistribute();

    // Add the particles to the current container
    this->addParticles(tmp_pc, true);
}

void
PhysicalParticleContainer::Evolve (ablastr::fields::MultiFabRegister& fields,
                                   int lev,
                                   const std::string& current_fp_string,
                                   Real /*t*/, Real dt, DtType a_dt_type, bool skip_deposition,
                                   PushType push_type)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    WARPX_PROFILE("PhysicalParticleContainer::Evolve()");
    WARPX_PROFILE_VAR_NS("PhysicalParticleContainer::Evolve::GatherAndPush", blp_fg);

    BL_ASSERT(OnSameGrids(lev, *fields.get(FieldType::current_fp, Direction{0}, lev)));

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

    if (m_do_back_transformed_particles)
    {
        for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
        {
            const auto np = pti.numParticles();
            const auto t_lev = pti.GetLevel();
            const auto index = pti.GetPairIndex();
            tmp_particle_data.resize(finestLevel()+1);
            for (int i = 0; i < TmpIdx::nattribs; ++i) {
                tmp_particle_data[t_lev][index][i].resize(np);
            }
        }
    }

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
            long nfine_current = np;
            long nfine_gather = np;
            if (has_buffer && !do_not_push) {
                // - Modify `nfine_current` and `nfine_gather` (in place)
                //    so that they correspond to the number of particles
                //    that deposit/gather in the fine patch respectively.
                // - Reorder the particle arrays,
                //    so that the `nfine_current`/`nfine_gather` first particles
                //    deposit/gather in the fine patch
                //    and (thus) the `np-nfine_current`/`np-nfine_gather` last particles
                //    deposit/gather in the buffer
                PartitionParticlesInBuffers( nfine_current, nfine_gather, np,
                    pti, lev, current_masks, gather_masks );
            }

            const long np_current = has_J_buf ? nfine_current : np;

            if (has_rho && ! skip_deposition && ! do_not_deposit) {
                // Deposit charge before particle push, in component 0 of MultiFab rho.

                const int* const AMREX_RESTRICT ion_lev = (do_field_ionization)?
                    pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr():nullptr;

                amrex::MultiFab* rho = fields.get(FieldType::rho_fp, lev);
                DepositCharge(pti, wp, ion_lev, rho, 0, 0,
                              np_current, thread_num, lev, lev);
                if (has_buffer){
                    amrex::MultiFab* crho = fields.get(FieldType::rho_buf, lev);
                    DepositCharge(pti, wp, ion_lev, crho, 0, np_current,
                                  np-np_current, thread_num, lev, lev-1);
                }
            }

            if (! do_not_push)
            {
                const long np_gather = has_E_cax ? nfine_gather : np;

                int e_is_nodal = Ex.is_nodal() and Ey.is_nodal() and Ez.is_nodal();

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
                    ImplicitPushXP(pti, exfab, eyfab, ezfab,
                                   bxfab, byfab, bzfab,
                                   Ex.nGrowVect(), e_is_nodal,
                                   0, np_to_push, lev, gather_lev, dt, ScaleFields(false), a_dt_type);
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
                                       cEx.nGrowVect(), e_is_nodal,
                                       nfine_gather, np-nfine_gather,
                                       lev, lev-1, dt, ScaleFields(false), a_dt_type);
                    }
                }

                WARPX_PROFILE_VAR_STOP(blp_fg);

                // Current Deposition
                if (!skip_deposition)
                {
                    // Deposit at t_{n+1/2} with explicit push
                    const amrex::Real relative_time = (push_type == PushType::Explicit ? -0.5_rt * dt : 0.0_rt);

                    const int* const AMREX_RESTRICT ion_lev = (do_field_ionization)?
                        pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr():nullptr;

                    // Deposit inside domains
                    amrex::MultiFab * jx = fields.get(current_fp_string, Direction{0}, lev);
                    amrex::MultiFab * jy = fields.get(current_fp_string, Direction{1}, lev);
                    amrex::MultiFab * jz = fields.get(current_fp_string, Direction{2}, lev);
                    DepositCurrent(pti, wp, uxp, uyp, uzp, ion_lev, jx, jy, jz,
                                   0, np_current, thread_num,
                                   lev, lev, dt, relative_time, push_type);

                    if (has_buffer)
                    {
                        // Deposit in buffers
                        amrex::MultiFab * cjx = fields.get(FieldType::current_buf, Direction{0}, lev);
                        amrex::MultiFab * cjy = fields.get(FieldType::current_buf, Direction{1}, lev);
                        amrex::MultiFab * cjz = fields.get(FieldType::current_buf, Direction{2}, lev);
                        DepositCurrent(pti, wp, uxp, uyp, uzp, ion_lev, cjx, cjy, cjz,
                                       np_current, np-np_current, thread_num,
                                       lev, lev-1, dt, relative_time, push_type);
                    }
                } // end of "if electrostatic_solver_id == ElectrostaticSolverAlgo::None"
            } // end of "if do_not_push"

            if (has_rho && ! skip_deposition && ! do_not_deposit) {
                // Deposit charge after particle push, in component 1 of MultiFab rho.
                // (Skipped for electrostatic solver, as this may lead to out-of-bounds)
                if (WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::None) {
                    amrex::MultiFab* rho = fields.get(FieldType::rho_fp, lev);
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho->nComp() >= 2,
                        "Cannot deposit charge in rho component 1: only component 0 is allocated!");

                    const int* const AMREX_RESTRICT ion_lev = (do_field_ionization)?
                        pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr():nullptr;

                    DepositCharge(pti, wp, ion_lev, rho, 1, 0,
                                  np_current, thread_num, lev, lev);
                    if (has_buffer){
                        amrex::MultiFab* crho = fields.get(FieldType::rho_buf, lev);
                        DepositCharge(pti, wp, ion_lev, crho, 1, np_current,
                                      np-np_current, thread_num, lev, lev-1);
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
    auto& mypc = WarpX::GetInstance().GetPartContainer();
    auto& pctmp_split = mypc.GetPCtmp();
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
    // Clear tmp container
    pctmp_split.clearParticles();
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
                ion_lev = pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr();
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

/* \brief Inject particles during the simulation
 * \param injection_box: domain where particles should be injected.
 */
void
PhysicalParticleContainer::ContinuousInjection (const RealBox& injection_box)
{
    // Inject plasma on level 0. Particles will be redistributed.
    const int lev=0;
    for (auto const& plasma_injector : plasma_injectors) {
        AddPlasma(*plasma_injector, lev, injection_box);
    }
}

/* \brief Inject a flux of particles during the simulation
 */
void
PhysicalParticleContainer::ContinuousFluxInjection (amrex::Real t, amrex::Real dt)
{
    for (auto const& plasma_injector : plasma_injectors) {
        if (plasma_injector->doFluxInjection()){
            // Check the optional parameters for start and stop of injection
            if ( ((plasma_injector->flux_tmin<0) || (t>=plasma_injector->flux_tmin)) &&
                 ((plasma_injector->flux_tmax<0) || (t< plasma_injector->flux_tmax)) ){

                AddPlasmaFlux(*plasma_injector, dt);

            }
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
        copyAttribs = CopyParticleAttribs(pti, tmp_particle_data, offset);
    }

    int* AMREX_RESTRICT ion_lev = nullptr;
    if (do_field_ionization) {
        ion_lev = pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr() + offset;
    }

    const bool save_previous_position = m_save_previous_position;
    ParticleReal* x_old = nullptr;
    ParticleReal* y_old = nullptr;
    ParticleReal* z_old = nullptr;
    if (save_previous_position) {
#if (AMREX_SPACEDIM >= 2)
        x_old = pti.GetAttribs(particle_comps["prev_x"]).dataPtr() + offset;
#endif
#if defined(WARPX_DIM_3D)
        y_old = pti.GetAttribs(particle_comps["prev_y"]).dataPtr() + offset;
#endif
        z_old = pti.GetAttribs(particle_comps["prev_z"]).dataPtr() + offset;
        amrex::ignore_unused(x_old, y_old);
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
        p_optical_depth_QSR = pti.GetAttribs(particle_comps["opticalDepthQSR"]).dataPtr()  + offset;
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
#if (AMREX_SPACEDIM >= 2)
            x_old[ip] = xp;
#endif
#if defined(WARPX_DIM_3D)
            y_old[ip] = yp;
#endif
            z_old[ip] = zp;
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

#ifdef WARPX_QED
        if (!do_sync)
#endif
        {
            if (do_copy) {
                //  Copy the old x and u for the BTD
                copyAttribs(ip);
            }

            doParticleMomentumPush<0>(ux[ip], uy[ip], uz[ip],
                                      Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                      ion_lev ? ion_lev[ip] : 1,
                                      m, q, pusher_algo, do_crr,
#ifdef WARPX_QED
                                      t_chi_max,
#endif
                                      dt);

            UpdatePosition(xp, yp, zp, ux[ip], uy[ip], uz[ip], dt);
            setPosition(ip, xp, yp, zp);
        }
#ifdef WARPX_QED
        else {
            if constexpr (qed_control == has_qed) {
                if (do_copy) {
                    //  Copy the old x and u for the BTD
                    copyAttribs(ip);
                }

                doParticleMomentumPush<1>(ux[ip], uy[ip], uz[ip],
                                          Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                          ion_lev ? ion_lev[ip] : 1,
                                          m, q, pusher_algo, do_crr,
                                          t_chi_max,
                                          dt);

                UpdatePosition(xp, yp, zp, ux[ip], uy[ip], uz[ip], dt);
                setPosition(ip, xp, yp, zp);
            }
        }
#endif

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

/* \brief Perform the implicit particle push operation in one fused kernel
 *        The main difference from PushPX is the order of operations:
 *         - push position by 1/2 dt
 *         - gather fields
 *         - push velocity by dt
 *         - average old and new velocity to get time centered value
 *        The routines ends with both position and velocity at the half time level.
 */
void
PhysicalParticleContainer::ImplicitPushXP (WarpXParIter& pti,
                                           amrex::FArrayBox const * exfab,
                                           amrex::FArrayBox const * eyfab,
                                           amrex::FArrayBox const * ezfab,
                                           amrex::FArrayBox const * bxfab,
                                           amrex::FArrayBox const * byfab,
                                           amrex::FArrayBox const * bzfab,
                                           amrex::IntVect ngEB, int /*e_is_nodal*/,
                                           long offset,
                                           long np_to_push,
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

    auto setPosition = SetParticlePosition(pti, offset);

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

    const auto depos_type = WarpX::current_deposition_algo;
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

#if (AMREX_SPACEDIM >= 2)
    ParticleReal* x_n = pti.GetAttribs(particle_comps["x_n"]).dataPtr();
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
    ParticleReal* y_n = pti.GetAttribs(particle_comps["y_n"]).dataPtr();
#endif
    ParticleReal* z_n = pti.GetAttribs(particle_comps["z_n"]).dataPtr();
    ParticleReal* ux_n = pti.GetAttribs(particle_comps["ux_n"]).dataPtr();
    ParticleReal* uy_n = pti.GetAttribs(particle_comps["uy_n"]).dataPtr();
    ParticleReal* uz_n = pti.GetAttribs(particle_comps["uz_n"]).dataPtr();

    const int do_copy = (m_do_back_transformed_particles && (a_dt_type!=DtType::SecondHalf) );
    CopyParticleAttribs copyAttribs;
    if (do_copy) {
        copyAttribs = CopyParticleAttribs(pti, tmp_particle_data, offset);
    }

    int* AMREX_RESTRICT ion_lev = nullptr;
    if (do_field_ionization) {
        ion_lev = pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr() + offset;
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
        p_optical_depth_QSR = pti.GetAttribs(particle_comps["opticalDepthQSR"]).dataPtr()  + offset;
    }
#endif

    const auto t_do_not_gather = do_not_gather;

    enum exteb_flags : int { no_exteb, has_exteb };
    enum qed_flags : int { no_qed, has_qed };

    const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;
#ifdef WARPX_QED
    const int qed_runtime_flag = (local_has_quantum_sync || do_sync) ? has_qed : no_qed;
#else
    const int qed_runtime_flag = no_qed;
#endif

    const int max_iterations = WarpX::max_particle_its_in_implicit_scheme;
    const amrex::ParticleReal particle_tolerance = WarpX::particle_tol_in_implicit_scheme;

    amrex::Gpu::Buffer<amrex::Long> unconverged_particles({0});
    amrex::Long* unconverged_particles_ptr = unconverged_particles.data();

    // Using this version of ParallelFor with compile time options
    // improves performance when qed or external EB are not used by reducing
    // register pressure.
    amrex::ParallelFor(TypeList<CompileTimeOptions<no_exteb,has_exteb>,
                                CompileTimeOptions<no_qed  ,has_qed>>{},
                       {exteb_runtime_flag, qed_runtime_flag},
                       np_to_push, [=] AMREX_GPU_DEVICE (long ip, auto exteb_control,
                                                         auto qed_control)
    {
        // Position advance starts from the position at the start of the step
        // but uses the most recent velocity.

#if (AMREX_SPACEDIM >= 2)
        amrex::ParticleReal xp = x_n[ip];
        const amrex::ParticleReal xp_n = x_n[ip];
#else
        const amrex::ParticleReal xp = 0._rt;
        const amrex::ParticleReal xp_n = 0._rt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        amrex::ParticleReal yp = y_n[ip];
        const amrex::ParticleReal yp_n = y_n[ip];
#else
        const amrex::ParticleReal yp = 0._rt;
        const amrex::ParticleReal yp_n = 0._rt;
#endif
        amrex::ParticleReal zp = z_n[ip];
        const amrex::ParticleReal zp_n = z_n[ip];

        amrex::ParticleReal dxp, dxp_save;
        amrex::ParticleReal dyp, dyp_save;
        amrex::ParticleReal dzp, dzp_save;
        auto idxg2 = static_cast<amrex::ParticleReal>(dinv.x*dinv.x);
        auto idyg2 = static_cast<amrex::ParticleReal>(dinv.y*dinv.y);
        auto idzg2 = static_cast<amrex::ParticleReal>(dinv.z*dinv.z);

        amrex::ParticleReal step_norm = 1._prt;
        for (int iter=0; iter<max_iterations;) {

            dxp = 0.0;
            dyp = 0.0;
            dzp = 0.0;
            UpdatePositionImplicit(dxp, dyp, dzp, ux_n[ip], uy_n[ip], uz_n[ip], ux[ip], uy[ip], uz[ip], 0.5_rt*dt);
#if !defined(WARPX_DIM_1D_Z)
            xp = xp_n + dxp;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            yp = yp_n + dyp;
#endif
            zp = zp_n + dzp;
            setPosition(ip, xp, yp, zp);

            PositionNorm( dxp, dyp, dzp, dxp_save, dyp_save, dzp_save,
                          idxg2, idyg2, idzg2, step_norm, iter );
            if( step_norm < particle_tolerance ) { break; }

            amrex::ParticleReal Exp = Ex_external_particle;
            amrex::ParticleReal Eyp = Ey_external_particle;
            amrex::ParticleReal Ezp = Ez_external_particle;
            amrex::ParticleReal Bxp = Bx_external_particle;
            amrex::ParticleReal Byp = By_external_particle;
            amrex::ParticleReal Bzp = Bz_external_particle;

            if(!t_do_not_gather){
                // first gather E and B to the particle positions
                doGatherShapeNImplicit(xp_n, yp_n, zp_n, xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                       ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                       ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                       dinv, xyzmin, lo, n_rz_azimuthal_modes, nox,
                                       depos_type );
            }

            // Externally applied E and B-field in Cartesian co-ordinates
            [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB;
            if constexpr (exteb_control == has_exteb) {
                getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
            }

            scaleFields(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp);

            if (do_copy) {
                //  Copy the old x and u for the BTD
                copyAttribs(ip);
            }

            // The momentum push starts with the velocity at the start of the step
            ux[ip] = ux_n[ip];
            uy[ip] = uy_n[ip];
            uz[ip] = uz_n[ip];

#ifdef WARPX_QED
            if (!do_sync)
#endif
            {
                doParticleMomentumPush<0>(ux[ip], uy[ip], uz[ip],
                                          Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                          ion_lev ? ion_lev[ip] : 1,
                                          m, q, pusher_algo, do_crr,
#ifdef WARPX_QED
                                          t_chi_max,
#endif
                                          dt);
            }
#ifdef WARPX_QED
            else {
                if constexpr (qed_control == has_qed) {
                    doParticleMomentumPush<1>(ux[ip], uy[ip], uz[ip],
                                              Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                              ion_lev ? ion_lev[ip] : 1,
                                              m, q, pusher_algo, do_crr,
                                              t_chi_max,
                                              dt);
                }
            }
#endif

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

            // Take average to get the time centered value
            ux[ip] = 0.5_rt*(ux[ip] + ux_n[ip]);
            uy[ip] = 0.5_rt*(uy[ip] + uy_n[ip]);
            uz[ip] = 0.5_rt*(uz[ip] + uz_n[ip]);

            iter++;

            // particle did not converge
            if ( iter > 1 && iter == max_iterations ) {
#if !defined(AMREX_USE_GPU)
                std::stringstream convergenceMsg;
                convergenceMsg << "Picard solver for particle failed to converge after " <<
                    iter << " iterations.\n";
                convergenceMsg << "Position step norm is " << step_norm <<
                    " and the tolerance is " << particle_tolerance << "\n";
                convergenceMsg << " ux = " << ux[ip] << ", uy = " << uy[ip] << ", uz = " << uz[ip] << "\n";
                convergenceMsg << " xp = " << xp     << ", yp = " << yp     << ", zp = " << zp;
                ablastr::warn_manager::WMRecordWarning("ImplicitPushXP", convergenceMsg.str());
#endif

                // write signaling flag: how many particles did not converge?
                amrex::Gpu::Atomic::Add(unconverged_particles_ptr, amrex::Long(1));
            }

        } // end Picard iterations

    });

    auto const num_unconverged_particles = *(unconverged_particles.copyToHost());
    if (num_unconverged_particles > 0) {
        ablastr::warn_manager::WMRecordWarning("ImplicitPushXP",
            "Picard solver for " +
            std::to_string(num_unconverged_particles) +
            " particles failed to converge after " +
            std::to_string(max_iterations) + " iterations."
         );
    }
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
    NewIntComp("ionizationLevel");
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
                                particle_icomps["ionizationLevel"],
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

void PhysicalParticleContainer::resample (const int timestep, const bool verbose)
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
                m_resampler(pti, lev, this);
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
    return PhotonEmissionFilterFunc{particle_runtime_comps["opticalDepthQSR"]};
}

PairGenerationFilterFunc
PhysicalParticleContainer::getPairGenerationFilterFunc ()
{
    WARPX_PROFILE("PhysicalParticleContainer::getPairGenerationFunc()");
    return PairGenerationFilterFunc{particle_runtime_comps["opticalDepthBW"]};
}

#endif
