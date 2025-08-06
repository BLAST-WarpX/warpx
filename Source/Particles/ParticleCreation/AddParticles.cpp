/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Particles/PhysicalParticleContainer.H"

#include "AddPlasmaUtilities.H"
#include "DefaultInitialization.H"
#include "Initialization/InjectorDensity.H"
#include "Initialization/InjectorMomentum.H"
#include "Initialization/InjectorPosition.H"
#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H"
#endif
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "Utils/WarpXConst.H"
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
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParGDB.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleContainerBase.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_Random.H>
#include <AMReX_SPACE.H>
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

#include <any>
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
    amrex::Real applyBallisticCorrection(const XDim3& pos, const InjectorMomentum* inj_mom,
                                  amrex::Real gamma_boost, amrex::Real beta_boost, amrex::Real t) noexcept
    {
        const XDim3 u_bulk = inj_mom->getBulkMomentum(pos.x, pos.y, pos.z);
        const amrex::Real gamma_bulk = std::sqrt(1._rt +
                             (u_bulk.x*u_bulk.x+u_bulk.y*u_bulk.y+u_bulk.z*u_bulk.z));
        const amrex::Real betaz_bulk = u_bulk.z/gamma_bulk;
        const amrex::Real z0 = gamma_boost * ( pos.z*(1.0_rt-beta_boost*betaz_bulk)
                             - PhysConst::c*t*(betaz_bulk-beta_boost) );
        return z0;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    XDim3 getCellCoords (const amrex::GpuArray<Real, AMREX_SPACEDIM>& lo_corner,
                         const amrex::GpuArray<Real, AMREX_SPACEDIM>& dx,
                         const XDim3& r, const amrex::IntVect& iv) noexcept
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
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = 0.0_rt;
        pos.z = 0.0_rt;
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
        const amrex::GpuArray<ParticleReal*,PIdx::nattribs>& pa, long& ip,
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

/* \brief Inject particles during the simulation
 * \param injection_box: domain where particles should be injected.
 */
void
PhysicalParticleContainer::ContinuousInjection (const amrex::RealBox& injection_box)
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
void PhysicalParticleContainer::MapParticletoBoostedFrame (
    amrex::ParticleReal& x, amrex::ParticleReal& y, amrex::ParticleReal& z,
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz, amrex::Real t_lab) const
{
    // Map the particles from the lab frame to the boosted frame.
    // This boosts the particle to the lab frame and calculates
    // the particle time in the boosted frame. It then maps
    // the position to the time in the boosted frame.

    // For now, start with the assumption that this will only happen
    // at the start of the simulation.
    const amrex::ParticleReal uz_boost = WarpX::gamma_boost*WarpX::beta_boost*PhysConst::c;

    // tpr is the particle's time in the boosted frame
    const amrex::ParticleReal tpr = WarpX::gamma_boost*t_lab - uz_boost*z/(PhysConst::c*PhysConst::c);

    // The particle's transformed location in the boosted frame
    const amrex::ParticleReal xpr = x;
    const amrex::ParticleReal ypr = y;
    const amrex::ParticleReal zpr = WarpX::gamma_boost*z - uz_boost*t_lab;

    // transform u and gamma to the boosted frame
    const amrex::ParticleReal gamma_lab = std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz)/(PhysConst::c*PhysConst::c));
    // ux = ux;
    // uy = uy;
    uz = WarpX::gamma_boost*uz - uz_boost*gamma_lab;
    const amrex::ParticleReal gammapr = std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz)/(PhysConst::c*PhysConst::c));

    const amrex::ParticleReal vxpr = ux/gammapr;
    const amrex::ParticleReal vypr = uy/gammapr;
    const amrex::ParticleReal vzpr = uz/gammapr;

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
PhysicalParticleContainer::CheckAndAddParticle (
    amrex::ParticleReal x, amrex::ParticleReal y, amrex::ParticleReal z,
    amrex::ParticleReal ux, amrex::ParticleReal uy, amrex::ParticleReal uz,
    amrex::ParticleReal weight,
    amrex::Gpu::HostVector<ParticleReal>& particle_x,
    amrex::Gpu::HostVector<ParticleReal>& particle_y,
    amrex::Gpu::HostVector<ParticleReal>& particle_z,
    amrex::Gpu::HostVector<ParticleReal>& particle_ux,
    amrex::Gpu::HostVector<ParticleReal>& particle_uy,
    amrex::Gpu::HostVector<ParticleReal>& particle_uz,
    amrex::Gpu::HostVector<ParticleReal>& particle_w,
    amrex::Real t_lab) const
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
PhysicalParticleContainer::AddGaussianBeam (PlasmaInjector const& plasma_injector){

    const amrex::Real x_m = plasma_injector.x_m;
    const amrex::Real y_m = plasma_injector.y_m;
    const amrex::Real z_m = plasma_injector.z_m;
    const amrex::Real x_rms = plasma_injector.x_rms;
    const amrex::Real y_rms = plasma_injector.y_rms;
    const amrex::Real z_rms = plasma_injector.z_rms;
    const amrex::Real x_cut = plasma_injector.x_cut;
    const amrex::Real y_cut = plasma_injector.y_cut;
    const amrex::Real z_cut = plasma_injector.z_cut;
    const amrex::Real q_tot = plasma_injector.q_tot;
    long npart = plasma_injector.npart;
    const int do_symmetrize = plasma_injector.do_symmetrize;
    const int symmetrization_order = plasma_injector.symmetrization_order;
    const amrex::Real focal_distance = plasma_injector.focal_distance;

    // Declare temporary vectors on the CPU
    amrex::Gpu::HostVector<ParticleReal> particle_x;
    amrex::Gpu::HostVector<ParticleReal> particle_y;
    amrex::Gpu::HostVector<ParticleReal> particle_z;
    amrex::Gpu::HostVector<ParticleReal> particle_ux;
    amrex::Gpu::HostVector<ParticleReal> particle_uy;
    amrex::Gpu::HostVector<ParticleReal> particle_uz;
    amrex::Gpu::HostVector<ParticleReal> particle_w;

    if (ParallelDescriptor::IOProcessor()) {
        // If do_symmetrize, create either 4x or 8x fewer particles, and
        // Replicate each particle either 4 times (x,y) (-x,y) (x,-y) (-x,-y)
        // or 8 times, additionally (y,x), (-y,x), (y,-x), (-y,-x)
        if (do_symmetrize){
            npart /= symmetrization_order;
        }
        for (long i = 0; i < npart; ++i) {
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            const amrex::Real weight = q_tot/(npart*charge);
            amrex::Real x = amrex::RandomNormal(x_m, x_rms);
            amrex::Real y = amrex::RandomNormal(y_m, y_rms);
            amrex::Real z = amrex::RandomNormal(z_m, z_rms);
#elif defined(WARPX_DIM_XZ)
            const amrex::Real weight = q_tot/(npart*charge*y_rms);
            amrex::Real x = amrex::RandomNormal(x_m, x_rms);
            constexpr amrex::Real y = 0._prt;
            amrex::Real z = amrex::RandomNormal(z_m, z_rms);
#elif defined(WARPX_DIM_1D_Z)
            const amrex::Real weight = q_tot/(npart*charge*x_rms*y_rms);
            constexpr amrex::Real x = 0._prt;
            constexpr amrex::Real y = 0._prt;
            amrex::Real z = amrex::RandomNormal(z_m, z_rms);
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            const amrex::Real weight = q_tot/(npart*charge*y_rms*z_rms);
            amrex::Real x = amrex::RandomNormal(x_m, x_rms);
            constexpr amrex::Real y = 0._prt;
            constexpr amrex::Real z = 0._prt;
#endif
            if (plasma_injector.insideBounds(x, y, z)  &&
                std::abs( x - x_m ) <= x_cut * x_rms     &&
                std::abs( y - y_m ) <= y_cut * y_rms     &&
                std::abs( z - z_m ) <= z_cut * z_rms   ) {
                XDim3 u = plasma_injector.getMomentum(x, y, z);

            if (plasma_injector.do_focusing){
                const XDim3 u_bulk = plasma_injector.getInjectorMomentumHost()->getBulkMomentum(x,y,z);
                const amrex::Real u_bulk_norm = std::sqrt( u_bulk.x*u_bulk.x+u_bulk.y*u_bulk.y+u_bulk.z*u_bulk.z );

                // Compute the position of the focal plane
                // (it is located at a distance `focal_distance` from the beam centroid, in the direction of the bulk velocity)
                const amrex::Real n_x = u_bulk.x/u_bulk_norm;
                const amrex::Real n_y = u_bulk.y/u_bulk_norm;
                const amrex::Real n_z = u_bulk.z/u_bulk_norm;
                const amrex::Real x_f = x_m + focal_distance * n_x;
                const amrex::Real y_f = y_m + focal_distance * n_y;
                const amrex::Real z_f = z_m + focal_distance * n_z;
                const amrex::Real gamma = std::sqrt( 1._rt + (u.x*u.x+u.y*u.y+u.z*u.z) );

                const amrex::Real v_x = u.x / gamma * PhysConst::c;
                const amrex::Real v_y = u.y / gamma * PhysConst::c;
                const amrex::Real v_z = u.z / gamma * PhysConst::c;

                // Compute the time at which the particle will cross the focal plane
                const amrex::Real v_dot_n = v_x * n_x + v_y * n_y + v_z * n_z;
                const amrex::Real t = ((x_f-x)*n_x + (y_f-y)*n_y + (z_f-z)*n_z) / v_dot_n;

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
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                x = x - (v_x - v_dot_n*n_x) * t;
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
                                             amrex::ParticleReal q_tot,
                                             amrex::ParticleReal z_shift)
{
    // Declare temporary vectors on the CPU
    amrex::Gpu::HostVector<ParticleReal> particle_x;
    amrex::Gpu::HostVector<ParticleReal> particle_z;
    amrex::Gpu::HostVector<ParticleReal> particle_ux;
    amrex::Gpu::HostVector<ParticleReal> particle_uz;
    amrex::Gpu::HostVector<ParticleReal> particle_w;
    amrex::Gpu::HostVector<ParticleReal> particle_y;
    amrex::Gpu::HostVector<ParticleReal> particle_uy;

#ifdef WARPX_USE_OPENPMD
    //TODO: Make changes for read/write in multiple MPI ranks
    if (ParallelDescriptor::IOProcessor()) {
        // take ownership of the series and close it when done
        auto series = std::any_cast<openPMD::Series>(std::move(plasma_injector.m_openpmd_input_series));

        // assumption asserts: see PlasmaInjector
        openPMD::Iteration it = series.iterations.begin()->second;
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
#if !defined(WARPX_DIM_1D_Z)  // 2D, 3D, RZ, 1D_R
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
#if !defined(WARPX_DIM_RCYLINDER)
        const std::shared_ptr<ParticleReal> ptr_z = ps["position"]["z"].loadChunk<ParticleReal>();
        const std::shared_ptr<ParticleReal> ptr_offset_z = ps["positionOffset"]["z"].loadChunk<ParticleReal>();
        auto const position_unit_z = static_cast<ParticleReal>(ps["position"]["z"].unitSI());
        auto const position_offset_unit_z = static_cast<ParticleReal>(ps["positionOffset"]["z"].unitSI());
#endif

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
        series.flush();  // shared_ptr data can be read now

        if (q_tot != 0.0) {
            std::stringstream warnMsg;
            warnMsg << " Loading particle species from file. " << ps_name << ".q_tot is ignored.";
            ablastr::warn_manager::WMRecordWarning("AddPlasmaFromFile",
               warnMsg.str(), ablastr::warn_manager::WarnPriority::high);
        }

        for (auto i = decltype(npart){0}; i<npart; ++i){

            amrex::ParticleReal const weight = ptr_w.get()[i]*w_unit;

#if !defined(WARPX_DIM_1D_Z)
            amrex::ParticleReal const x = ptr_x.get()[i]*position_unit_x + ptr_offset_x.get()[i]*position_offset_unit_x;
#else
            amrex::ParticleReal const x = 0.0_prt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            amrex::ParticleReal const y = ptr_y.get()[i]*position_unit_y + ptr_offset_y.get()[i]*position_offset_unit_y;
#else
            amrex::ParticleReal const y = 0.0_prt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
            amrex::ParticleReal const z = ptr_z.get()[i]*position_unit_z + ptr_offset_z.get()[i]*position_offset_unit_z + z_shift;
#else
            amrex::ParticleReal const z = 0.0_prt;
#endif

            if (plasma_injector.insideBounds(x, y, z)) {
                amrex::ParticleReal const ux = ptr_ux.get()[i]*momentum_unit_x/mass;
                amrex::ParticleReal const uz = ptr_uz.get()[i]*momentum_unit_z/mass;
                amrex::ParticleReal uy = 0.0_prt;
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
PhysicalParticleContainer::AddPlasma (PlasmaInjector const& plasma_injector, int lev, amrex::RealBox part_realbox)
{
    WARPX_PROFILE("PhysicalParticleContainer::AddPlasma()");

    // If no part_realbox is provided, initialize particles in the whole domain
    const Geometry& geom = Geom(lev);
    if (!part_realbox.ok()) { part_realbox = geom.ProbDomain(); }

    const int num_ppc = plasma_injector.num_particles_per_cell;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const amrex::Real rmax = std::min(plasma_injector.xmax, part_realbox.hi(0));
    const amrex::Real rmin = std::max(plasma_injector.xmin, part_realbox.lo(0));
#endif

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();

    defineAllParticleTiles();

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    amrex::Box fine_injection_box;
    amrex::IntVect rrfac(AMREX_D_DECL(1,1,1));
    const bool refine_injection = findRefinedInjectionBox(fine_injection_box, rrfac);

    InjectorPosition* inj_pos = plasma_injector.getInjectorPosition();
    InjectorDensity*  inj_rho = plasma_injector.getInjectorDensity();
    InjectorMomentum* inj_mom = plasma_injector.getInjectorMomentumDevice();
    const amrex::Real gamma_boost = WarpX::gamma_boost;
    const amrex::Real beta_boost = WarpX::beta_boost;
    const amrex::Real t = WarpX::GetInstance().gett_new(lev);
    const amrex::Real density_min = plasma_injector.density_min;
    const amrex::Real density_max = plasma_injector.density_max;

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    const int nmodes = WarpX::n_rz_azimuthal_modes;
#endif
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const amrex::Real radial_numpercell_power = plasma_injector.radial_numpercell_power;
#endif

    auto n_user_int_attribs = static_cast<int>(m_user_int_attribs.size());
    auto n_user_real_attribs = static_cast<int>(m_user_real_attribs.size());
    const PlasmaParserWrapper plasma_parser_wrapper (m_user_int_attribs.size(),
                                                     m_user_real_attribs.size(),
                                                     m_user_int_attrib_parser,
                                                     m_user_real_attrib_parser);

    MFItInfo info;
    if (do_tiling && amrex::Gpu::notInLaunchRegion()) {
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

        const amrex::Box& tile_box = mfi.tilebox();
        const amrex::RealBox tile_realbox = WarpX::getRealBox(tile_box, lev);

        // Find the cells of part_box that overlap with tile_realbox
        // If there is no overlap, just go to the next tile in the loop
        amrex::RealBox overlap_realbox;
        amrex::Box overlap_box;
        amrex::IntVect shifted;
        const bool no_overlap = find_overlap(tile_realbox, part_realbox, dx, problo, overlap_realbox, overlap_box, shifted);
        if (no_overlap) {
            continue; // Go to the next tile
        }

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> overlap_corner
            {AMREX_D_DECL(overlap_realbox.lo(0),
                          overlap_realbox.lo(1),
                          overlap_realbox.lo(2))};

        // count the number of particles that each cell in overlap_box could add
        amrex::Gpu::DeviceVector<amrex::Long> counts(overlap_box.numPts(), 0);
        amrex::Gpu::DeviceVector<amrex::Long> offset(overlap_box.numPts());
        auto *pcounts = counts.data();
        amrex::Box fine_overlap_box; // default Box is NOT ok().
        if (refine_injection) {
            fine_overlap_box = overlap_box & amrex::shift(fine_injection_box, -shifted);
        }
        amrex::ParallelFor(overlap_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
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
                const auto xlim = amrex::GpuArray<Real, 3>{lo.x,(lo.x+hi.x)/2._rt,hi.x};
                const auto ylim = amrex::GpuArray<Real, 3>{lo.y,(lo.y+hi.y)/2._rt,hi.y};
                const auto zlim = amrex::GpuArray<Real, 3>{lo.z,(lo.z+hi.z)/2._rt,hi.z};

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
        const amrex::Long max_new_particles = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

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
        amrex::GpuArray<ParticleReal*,PIdx::nattribs> pa;
        for (int ia = 0; ia < PIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        uint64_t * AMREX_RESTRICT pa_idcpu = soa.GetIdCPUData().data() + old_size;

        PlasmaParserHelper plasma_parser_helper(soa, old_size, m_user_int_attribs, m_user_real_attribs, plasma_parser_wrapper);
        int** pa_user_int_data = plasma_parser_helper.getUserIntDataPtrs();
        amrex::ParticleReal** pa_user_real_data = plasma_parser_helper.getUserRealDataPtrs();
        amrex::ParserExecutor<7> const* user_int_parserexec_data = plasma_parser_helper.getUserIntParserExecData();
        amrex::ParserExecutor<7> const* user_real_parserexec_data = plasma_parser_helper.getUserRealParserExecData();

        int* pi = nullptr;
        if (do_field_ionization) {
            pi = soa.GetIntData("ionizationLevel").data() + old_size;
        }

#ifdef WARPX_QED
        const QEDHelper qed_helper(soa, old_size,
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
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        const bool rz_random_theta = m_rz_random_theta;
#endif
        amrex::ParallelForRNG(overlap_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const amrex::IntVect iv = amrex::IntVect(AMREX_D_DECL(i, j, k));
            amrex::ignore_unused(j,k);
            const auto index = overlap_box.index(iv);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
            amrex::Real theta_offset = 0._rt;
            if (rz_random_theta) { theta_offset = amrex::Random(engine) * 2._rt * MathConst::pi; }
#endif

            const amrex::Real scale_fac = compute_scale_fac_volume(dx, pcounts[index]);
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
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                amrex::ignore_unused(j,k);
                bool const box_contains = tile_realbox.contains(XDim3{pos.x,0.0_rt,0.0_rt});
#endif
                if (!box_contains) {
                    ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,qed_helper
#endif
                                                   );
                    continue;
                }

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                // Replace the x and y, setting an angle theta.
                // These x and y are used to get the momentum and density
                // With only 1 mode, the angle doesn't matter so
                // choose it randomly.
                const amrex::Real theta = (nmodes == 1 && rz_random_theta)?
#if defined(WARPX_DIM_RZ)
                    // This should be updated to be the same as below, since theta
                    // should range from -pi to +pi. This should be a separate PR
                    // since it will break RZ CI tests.
                    (2._rt*MathConst::pi*amrex::Random(engine)):
#elif defined(WARPX_DIM_RCYLINDER)
                    (MathConst::pi*(2._rt*amrex::Random(engine) - 1._rt)):
#endif
                    (2._rt*MathConst::pi*r.y + theta_offset);

                // Adjust the particle radius to produce the correct distribution.
                // Note that this may shift particles outside of the current tile,
                // but this is Ok since particles will be redistributed afterwards.
                // The tile_realbox.contains check above ensures
                // that the "logical" space is uniformly filled.
                amrex::Real const xu = (pos.x - rmin)/(rmax - rmin);
                amrex::Real const rc = std::pow(rmax, 1._rt + radial_numpercell_power)
                                     - std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const rminp = std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const xb = std::pow(xu*rc + rminp, 1._rt/(1._rt + radial_numpercell_power));
                amrex::Real const yb = theta;

                pos.x = xb*std::cos(theta);
                pos.y = xb*std::sin(theta);
#elif defined(WARPX_DIM_RSPHERE)
                // Replace the x, y, and z, setting angles theta and phi.
                // These x, y, and z are used to get the momentum and density
                const amrex::Real theta = MathConst::pi*(2._rt*amrex::Random(engine) - 1._rt);
                const amrex::Real sin_phi = 2._rt*amrex::Random(engine) - 1._rt;
                const amrex::Real cos_phi = std::sqrt(1._rt - sin_phi*sin_phi);
                const amrex::Real phi = std::atan2(sin_phi, cos_phi);

                // Adjust the particle radius to produce the correct distribution.
                // Note that this may shift particles outside of the current tile,
                // but this is Ok since particles will be redistributed afterwards.
                // The tile_realbox.contains check above ensures
                // that the "logical" space is uniformly filled.
                amrex::Real const xu = (pos.x - rmin)/(rmax - rmin);
                amrex::Real const rc = std::pow(rmax, 1._rt + radial_numpercell_power)
                                     - std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const rminp = std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const xb = std::pow(xu*rc + rminp, 1._rt/(1._rt + radial_numpercell_power));
                amrex::Real const yb = theta;

                pos.x = xb*cos_phi*std::cos(theta);
                pos.y = xb*cos_phi*std::sin(theta);
                pos.z = xb*sin_phi;
#else
                // Save the x and y values to use in the insideBounds checks.
                amrex::Real const xb = pos.x;
                amrex::Real const yb = pos.y;
#endif

                amrex::Real dens;
                XDim3 u;
                if (gamma_boost == 1._rt) {
                    // Lab-frame simulation
                    // If the particle is not within the species's
                    // xmin, xmax, ymin, ymax, zmin, zmax, go to
                    // the next generated particle.

                    // include ballistic correction for plasma species with bulk motion
                    const amrex::Real z0 = applyBallisticCorrection(pos, inj_mom, gamma_boost,
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
                    const amrex::Real z0_lab = applyBallisticCorrection(pos, inj_mom, gamma_boost,
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
                    const amrex::Real gamma_lab = std::sqrt( 1._rt+(u.x*u.x+u.y*u.y+u.z*u.z) );
                    const amrex::Real betaz_lab = u.z/(gamma_lab);

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

                amrex::Real weight = dens;
                weight *= scale_fac;

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                // Update the weight based on the specified power.
                // The coefficient ensures that the correct density distribution is obtained.
                const amrex::Real coeff = 2._rt*MathConst::pi/(1._rt + radial_numpercell_power)
                        *(rmax - std::pow(rmax, -radial_numpercell_power)*std::pow(rmin, 1._rt + radial_numpercell_power));
                weight *= coeff*std::pow(xb/rmax, 1._rt - radial_numpercell_power);
#elif defined(WARPX_DIM_RSPHERE)
                const amrex::Real coeff = 4._rt*MathConst::pi/(1._rt + radial_numpercell_power)
                        *(rmax*rmax - std::pow(rmax, 1._rt - radial_numpercell_power )*std::pow(rmin, 1._rt + radial_numpercell_power));
                weight *= coeff*std::pow(xb/rmax, 2._rt - radial_numpercell_power);
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
#elif defined(WARPX_DIM_RCYLINDER)
                pa[PIdx::theta][ip] = theta;
                pa[PIdx::x][ip] = xb;
#elif defined(WARPX_DIM_RSPHERE)
                pa[PIdx::theta][ip] = theta;
                pa[PIdx::phi][ip] = phi;
                pa[PIdx::x][ip] = xb;
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
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const amrex::Real rmax = std::min(plasma_injector.xmax, geom.ProbDomain().hi(0));
    const amrex::Real rmin = std::max(plasma_injector.xmin, geom.ProbDomain().lo(0));
#endif

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();

#ifdef AMREX_USE_EB
    bool const inject_from_eb = plasma_injector.m_inject_from_eb; // whether to inject from EB or from a plane
    // Extract data structures for embedded boundaries
    amrex::EBFArrayBoxFactory const* eb_factory = nullptr;
    amrex::FabArray<amrex::EBCellFlagFab> const* eb_flag = nullptr;
    if (inject_from_eb) {
        eb_factory = &(WarpX::GetInstance().fieldEBFactory(0));
        eb_flag = &(eb_factory->getMultiEBCellFlagFab());
    }
#endif

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(0);

    // Create temporary particle container to which particles will be added;
    // we will then call Redistribute on this new container and finally
    // add the new particles to the original container.
    PhysicalParticleContainer tmp_pc(&WarpX::GetInstance());
    for (int ic = 0; ic < NumRuntimeRealComps(); ++ic) { tmp_pc.AddRealComp(GetRealSoANames()[ic + NArrayReal], false); }
    for (int ic = 0; ic < NumRuntimeIntComps(); ++ic) { tmp_pc.AddIntComp(GetIntSoANames()[ic + NArrayInt], false); }
    tmp_pc.defineAllParticleTiles();

    amrex::Box fine_injection_box;
    amrex::IntVect rrfac(AMREX_D_DECL(1,1,1));
    const bool refine_injection = findRefinedInjectionBox(fine_injection_box, rrfac);

    InjectorPosition* flux_pos = plasma_injector.getInjectorFluxPosition();
    InjectorFlux*  inj_flux = plasma_injector.getInjectorFlux();
    InjectorMomentum* inj_mom = plasma_injector.getInjectorMomentumDevice();
    constexpr int level_zero = 0;
    const amrex::Real t = WarpX::GetInstance().gett_new(level_zero);

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    const int nmodes = WarpX::n_rz_azimuthal_modes;
    const bool rz_random_theta = m_rz_random_theta;
#endif
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const amrex::Real radial_numpercell_power = plasma_injector.radial_numpercell_power;
#endif

    auto n_user_int_attribs = static_cast<int>(m_user_int_attribs.size());
    auto n_user_real_attribs = static_cast<int>(m_user_real_attribs.size());
    const PlasmaParserWrapper plasma_parser_wrapper (m_user_int_attribs.size(),
                                                     m_user_real_attribs.size(),
                                                     m_user_int_attrib_parser,
                                                     m_user_real_attrib_parser);

    MFItInfo info;
    if (do_tiling && amrex::Gpu::notInLaunchRegion()) {
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

        const amrex::Box& tile_box = mfi.tilebox();
        const amrex::RealBox tile_realbox = WarpX::getRealBox(tile_box, 0);

        // Find the cells of part_realbox that overlap with tile_realbox
        // If there is no overlap, just go to the next tile in the loop
        amrex::RealBox overlap_realbox;
        amrex::Box overlap_box;
        amrex::IntVect shifted;
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

        const amrex::GpuArray<Real,AMREX_SPACEDIM> overlap_corner
            {AMREX_D_DECL(overlap_realbox.lo(0),
                          overlap_realbox.lo(1),
                          overlap_realbox.lo(2))};

        // count the number of particles that each cell in overlap_box could add
        amrex::Gpu::DeviceVector<int> counts(overlap_box.numPts(), 0);
        amrex::Gpu::DeviceVector<int> offset(overlap_box.numPts());
        auto *pcounts = counts.data();
        const int flux_normal_axis = plasma_injector.flux_normal_axis;
        amrex::Box fine_overlap_box; // default Box is NOT ok().
        if (refine_injection) {
            fine_overlap_box = overlap_box & amrex::shift(fine_injection_box, -shifted);
        }

#ifdef AMREX_USE_EB
        auto eb_flag_arr = eb_flag ? eb_flag->const_array(mfi) : Array4<EBCellFlag const>{};
        auto eb_data = eb_factory ? eb_factory->getEBData(mfi) : EBData{};
#endif

        amrex::ParallelForRNG(overlap_box, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            amrex::ignore_unused(j,k);

            // Determine the number of macroparticles to inject in this cell (num_ppc_int)
#ifdef AMREX_USE_EB
            amrex::Real num_ppc_real_in_this_cell = num_ppc_real; // user input: number of macroparticles per cell
            if (inject_from_eb) {
                // Injection from EB
                // Skip cells that are not partially covered by the EB
                if (eb_flag_arr(i,j,k).isRegular() || eb_flag_arr(i,j,k).isCovered()) { return; }
                // Scale by the (normalized) area of the EB surface in this cell
                num_ppc_real_in_this_cell *= eb_data.get<amrex::EBData_t::bndryarea>(i,j,k);
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
        const amrex::Long max_new_particles = amrex::Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

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
        amrex::GpuArray<ParticleReal*,PIdx::nattribs> pa;
        for (int ia = 0; ia < PIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        uint64_t * AMREX_RESTRICT pa_idcpu = soa.GetIdCPUData().data() + old_size;

        PlasmaParserHelper plasma_parser_helper(soa, old_size, m_user_int_attribs, m_user_real_attribs, plasma_parser_wrapper);
        int** pa_user_int_data = plasma_parser_helper.getUserIntDataPtrs();
        amrex::ParticleReal** pa_user_real_data = plasma_parser_helper.getUserRealDataPtrs();
        amrex::ParserExecutor<7> const* user_int_parserexec_data = plasma_parser_helper.getUserIntParserExecData();
        amrex::ParserExecutor<7> const* user_real_parserexec_data = plasma_parser_helper.getUserRealParserExecData();

        int* p_ion_level = nullptr;
        if (do_field_ionization) {
            p_ion_level = soa.GetIntData("ionizationLevel").data() + old_size;
        }

#ifdef WARPX_QED
        const QEDHelper qed_helper(soa, old_size,
                                   has_quantum_sync(), has_breit_wheeler(),
                                   m_shr_p_qs_engine, m_shr_p_bw_engine);
#endif

        const bool loc_do_field_ionization = do_field_ionization;
        const int loc_ionization_initial_level = ionization_initial_level;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
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
            const amrex::IntVect iv = amrex::IntVect(AMREX_D_DECL(i, j, k));
            amrex::ignore_unused(j,k);
            const auto index = overlap_box.index(iv);

            amrex::Real scale_fac;
#ifdef AMREX_USE_EB
            if (inject_from_eb) {
                scale_fac = compute_scale_fac_area_eb(dx, num_ppc_real,
                                                      AMREX_D_DECL(eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,0),
                                                                   eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,1),
                                                                   eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,2)));
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
                    auto const& pt = eb_data.randomPointOnEB(i,j,k,engine);
#if defined(WARPX_DIM_3D)
                    pos.x = overlap_corner[0] + (iv[0] + 0.5_rt + pt[0])*dx[0];
                    pos.y = overlap_corner[1] + (iv[1] + 0.5_rt + pt[1])*dx[1];
                    pos.z = overlap_corner[2] + (iv[2] + 0.5_rt + pt[2])*dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    pos.x = overlap_corner[0] + (iv[0] + 0.5_rt + pt[0])*dx[0];
                    pos.y = 0.0_rt;
                    pos.z = overlap_corner[1] + (iv[1] + 0.5_rt + pt[1])*dx[1];
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
#elif defined(WARPX_DIM_1D_Z)
                amrex::ignore_unused(j,k);
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.z,0.0_prt,0.0_prt})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                amrex::ignore_unused(j,k);
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.x,0.0_prt,0.0_prt})) {
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
                    rotate_momentum_eb(pu, AMREX_D_DECL(eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,0),
                                                        eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,1),
                                                        eb_data.get<amrex::EBData_t::bndrynorm>(i,j,k,2)));
                }
#endif

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                // Adjust the particle radius to produce the correct distribution.
                // Note that this may shift particles outside of the current tile,
                // but this is Ok since particles will be redistributed afterwards.
                // The containsInclusive check above ensures
                // that the "logical" space is uniformly filled.
                amrex::Real const xu = (ppos.x - rmin)/(rmax - rmin);
                amrex::Real const rc = std::pow(rmax, 1._rt + radial_numpercell_power)
                                     - std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const rminp = std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const radial_position = std::pow(xu*rc + rminp, 1._rt/(1._rt + radial_numpercell_power));

                // Conversion from cylindrical to Cartesian coordinates
                // Replace the x and y, setting an angle theta.
                // These x and y are used to get the momentum and flux
                // With only 1 mode, the angle doesn't matter so
                // choose it randomly.
                const amrex::Real theta = (nmodes == 1 && rz_random_theta)?
#if defined(WARPX_DIM_RZ)
                    // This should be updated to be the same as below, since theta
                    // should range from -pi to +pi. This should be a separate PR
                    // since it will break RZ CI tests.
                    (2._rt*MathConst::pi*amrex::Random(engine)):
#elif defined(WARPX_DIM_RCYLINDER)
                    (MathConst::pi*(2._rt*amrex::Random(engine) - 1._rt)):
#endif
                    (2._prt*MathConst::pi*r.y);
                amrex::Real const cos_theta = std::cos(theta);
                amrex::Real const sin_theta = std::sin(theta);
                // Rotate the position
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
                    const amrex::Real ur = pu.x;
                    const amrex::Real ut = pu.y;
                    pu.x = cos_theta*ur - sin_theta*ut;
                    pu.y = sin_theta*ur + cos_theta*ut;
                }
#elif defined(WARPX_DIM_RSPHERE)
                // Adjust the particle radius to produce the correct distribution.
                // Note that this may shift particles outside of the current tile,
                // but this is Ok since particles will be redistributed afterwards.
                // The containsInclusive check above ensures
                // that the "logical" space is uniformly filled.
                amrex::Real const xu = (ppos.x - rmin)/(rmax - rmin);
                amrex::Real const rc = std::pow(rmax, 1._rt + radial_numpercell_power)
                                     - std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const rminp = std::pow(rmin, 1._rt + radial_numpercell_power);
                amrex::Real const radial_position = std::pow(xu*rc + rminp, 1._rt/(1._rt + radial_numpercell_power));

                // Replace the x, y, and z, setting angles theta and phi.
                // These x, y, and z are used to get the momentum and flux
                amrex::Real const theta = MathConst::pi*(2._rt*amrex::Random(engine) - 1._rt);
                amrex::Real const sin_phi = 2._rt*amrex::Random(engine) - 1._rt;
                amrex::Real const cos_phi = std::sqrt(1._rt - sin_phi*sin_phi);
                amrex::Real const phi = std::atan2(sin_phi, cos_phi);
                amrex::Real const cos_theta = std::cos(theta);
                amrex::Real const sin_theta = std::sin(theta);
                pos.x = radial_position*cos_phi*std::cos(theta);
                pos.y = radial_position*cos_phi*std::sin(theta);
                pos.z = radial_position*sin_phi;
                // Rotate the momentum
                // This because, when the flux direction is e.g. "r"
                // the `inj_mom` objects generates a v*Gaussian distribution
                // along the Cartesian "x" direction by default. This
                // needs to be rotated along "r".
                amrex::Real const ur = pu.x;
                amrex::Real const ut = pu.y;
                amrex::Real const up = pu.z;
                pu.x = cos_phi*cos_theta*ur - sin_theta*ut - sin_phi*cos_theta*up;
                pu.y = cos_phi*sin_theta*ur + cos_theta*ut - sin_phi*sin_theta*up;
                pu.z = sin_phi*ur + cos_phi*up;
#endif
                const amrex::Real flux = inj_flux->getFlux(ppos.x, ppos.y, ppos.z, t);
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

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                // The particle weight is proportional to the user-specified
                // flux and the emission surface within
                // one cell (captured partially by `scale_fac`).
                // For cylindrical emission (flux_normal_axis==0
                // or flux_normal_axis==2), the emission surface depends on
                // the radius ; thus, the calculation is finalized here
                amrex::Real t_weight = flux * scale_fac * dt;
                if (loc_flux_normal_axis != 1) {
                    // Update the weight based on the specified power.
                    // The coefficient ensures that the correct density distribution is obtained.
                    const amrex::Real coeff = 2._rt*MathConst::pi/(1._rt + radial_numpercell_power)
                        *(rmax - std::pow(rmax, -radial_numpercell_power)*std::pow(rmin, 1._rt + radial_numpercell_power));
                    t_weight *= coeff*std::pow(radial_position/rmax, 1._rt - radial_numpercell_power);
                }

                const amrex::Real weight = t_weight;
#elif defined(WARPX_DIM_RSPHERE)
                // The particle weight is proportional to the user-specified
                // flux and the emission surface within
                // one cell (captured partially by `scale_fac`).
                // For spherical emission (flux_normal_axis==0),
                // the emission surface depends on
                // the radius ; thus, the calculation is finalized here
                amrex::Real t_weight = flux * scale_fac * dt;
                if (loc_flux_normal_axis == 0) {
                    // Update the weight based on the specified power.
                    // The coefficient ensures that the correct density distribution is obtained.
                    const amrex::Real coeff = 4._rt*MathConst::pi/(1._rt + radial_numpercell_power)
                        *(rmax*rmax - std::pow(rmax, 1._rt - radial_numpercell_power)*std::pow(rmin, 1._rt + radial_numpercell_power));
                    t_weight *= coeff*std::pow(radial_position/rmax, 2._rt - radial_numpercell_power);
                }
                const amrex::Real weight = t_weight;
#else
                const amrex::Real weight = flux * scale_fac * dt;
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
#elif defined(WARPX_DIM_RCYLINDER)
                pa[PIdx::theta][ip] = theta;
                pa[PIdx::x][ip] = radial_position;
#elif defined(WARPX_DIM_RSPHERE)
                pa[PIdx::theta][ip] = theta;
                pa[PIdx::phi][ip] = phi;
                pa[PIdx::x][ip] = radial_position;
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
