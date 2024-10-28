/* Copyright 2019-2020 Andrew Myers, David Grote, Glenn Richardson
 * Ligia Diana Amorim, Luca Fedeli, Maxence Thevenet
 * Remi Lehe, Revathi Jambunathan, Weiqun Zhang
 * Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Pusher/GetAndSetPosition.H"
#include "Pusher/UpdateMomentumBoris.H"
#include "Pusher/UpdateMomentumBorisWithRadiationReaction.H"
#include "Pusher/UpdateMomentumHigueraCary.H"
#include "Pusher/UpdateMomentumVay.H"
#include "RigidInjectedParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_StructOfArrays.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

using namespace amrex;

RigidInjectedParticleContainer::RigidInjectedParticleContainer (AmrCore* amr_core, int ispecies,
                                                                const std::string& name)
    : PhysicalParticleContainer(amr_core, ispecies, name)
{

    const ParmParse pp_species_name(species_name);

    utils::parser::getWithParser(
        pp_species_name, "zinject_plane", zinject_plane);
    pp_species_name.query("rigid_advance", rigid_advance);

}

void RigidInjectedParticleContainer::InitData()
{
    // Perform Lorentz transform of `z_inject_plane`
    const amrex::Real t_boost = WarpX::GetInstance().gett_new(0);
    const amrex::Real zinject_plane_boost = zinject_plane/WarpX::gamma_boost
                                    - WarpX::beta_boost*PhysConst::c*t_boost;
    zinject_plane_levels.resize(finestLevel()+1, zinject_plane_boost);

    AddParticles(0); // Note - add on level 0

    // Particles added by AddParticles should already be in the boosted frame
    RemapParticles();

    Redistribute();  // We then redistribute
}

void
RigidInjectedParticleContainer::RemapParticles()
{
    // For rigid_advance == false, nothing needs to be done

    if (rigid_advance) {

        // The particle z positions are adjusted to account for the difference between
        // advancing with vzbar and wih vz[i] before injection

        // For now, start with the assumption that this will only happen
        // at the start of the simulation.
        const ParticleReal t_lab = 0._prt;

        const ParticleReal uz_boost = WarpX::gamma_boost*WarpX::beta_boost*PhysConst::c;
        const ParticleReal csqi = 1._prt/(PhysConst::c*PhysConst::c);

        vzbeam_ave_boosted = meanParticleVelocity(false)[2];

        for (int lev = 0; lev <= finestLevel(); lev++) {

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            {
                // Get the average beam velocity in the boosted frame.
                // Note that the particles are already in the boosted frame.
                // This value is saved to advance the particles not injected yet

                for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
                {
                    const auto& attribs = pti.GetAttribs();
                    const auto *const uxp = attribs[PIdx::ux].dataPtr();
                    const auto *const uyp = attribs[PIdx::uy].dataPtr();
                    const auto *const uzp = attribs[PIdx::uz].dataPtr();

                    const auto GetPosition = GetParticlePosition<PIdx>(pti);
                          auto SetPosition = SetParticlePosition<PIdx>(pti);

                    // Loop over particles
                    const long np = pti.numParticles();
                    const ParticleReal lvzbeam_ave_boosted = vzbeam_ave_boosted;
                    const ParticleReal gamma_boost = WarpX::gamma_boost;
                    amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long i)
                    {
                        ParticleReal xp, yp, zp;
                        GetPosition(i, xp, yp, zp);

                        const ParticleReal gammapr = std::sqrt(1._prt + (uxp[i]*uxp[i] + uyp[i]*uyp[i] + uzp[i]*uzp[i])*csqi);
                        const ParticleReal vzpr = uzp[i]/gammapr;

                        // Back out the value of z_lab
                        const ParticleReal z_lab = (zp + uz_boost*t_lab + gamma_boost*t_lab*vzpr)/(gamma_boost + uz_boost*vzpr*csqi);

                        // Time of the particle in the boosted frame given its position in the lab frame at t=0.
                        const ParticleReal tpr = gamma_boost*t_lab - uz_boost*z_lab*csqi;

                        // Adjust the position, taking away its motion from its own velocity and adding
                        // the motion from the average velocity
                        zp += tpr*vzpr - tpr*lvzbeam_ave_boosted;

                        SetPosition(i, xp, yp, zp);
                    });
                }
            }
        }
    }
}

void
RigidInjectedParticleContainer::PushPX (WarpXParIter& pti,
                                        amrex::FArrayBox const * exfab,
                                        amrex::FArrayBox const * eyfab,
                                        amrex::FArrayBox const * ezfab,
                                        amrex::FArrayBox const * bxfab,
                                        amrex::FArrayBox const * byfab,
                                        amrex::FArrayBox const * bzfab,
                                        const amrex::IntVect ngEB, const int e_is_nodal,
                                        const long offset,
                                        const long np_to_push,
                                        int lev, int gather_lev,
                                        amrex::Real dt, ScaleFields /*scaleFields*/,
                                        DtType a_dt_type)
{
    auto& attribs = pti.GetAttribs();
    auto& uxp = attribs[PIdx::ux];
    auto& uyp = attribs[PIdx::uy];
    auto& uzp = attribs[PIdx::uz];

    // Save the position, momentum and optical depth, making copies
    amrex::Gpu::DeviceVector<ParticleReal> xp_save, yp_save, zp_save;
    amrex::Gpu::DeviceVector<ParticleReal> uxp_save, uyp_save, uzp_save;
#ifdef WARPX_QED
    amrex::Gpu::DeviceVector<ParticleReal> optical_depth_save;
#endif

    const auto GetPosition = GetParticlePosition<PIdx>(pti, offset);
          auto SetPosition = SetParticlePosition<PIdx>(pti, offset);

    amrex::ParticleReal* const AMREX_RESTRICT ux = uxp.dataPtr() + offset;
    amrex::ParticleReal* const AMREX_RESTRICT uy = uyp.dataPtr() + offset;
    amrex::ParticleReal* const AMREX_RESTRICT uz = uzp.dataPtr() + offset;

#ifdef WARPX_QED
    const bool loc_has_quantum_sync = has_quantum_sync();
    amrex::ParticleReal* AMREX_RESTRICT p_optical_depth = nullptr;
    amrex::ParticleReal* AMREX_RESTRICT p_optical_depth_save = nullptr;
#endif

    if (!done_injecting_lev)
    {
        // If the old values are not already saved, create copies here.
        xp_save.resize(np_to_push);
        yp_save.resize(np_to_push);
        zp_save.resize(np_to_push);

        uxp_save.resize(np_to_push);
        uyp_save.resize(np_to_push);
        uzp_save.resize(np_to_push);

        amrex::ParticleReal* const AMREX_RESTRICT xp_save_ptr = xp_save.dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT yp_save_ptr = yp_save.dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT zp_save_ptr = zp_save.dataPtr();

        amrex::ParticleReal* const AMREX_RESTRICT uxp_save_ptr = uxp_save.dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT uyp_save_ptr = uyp_save.dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT uzp_save_ptr = uzp_save.dataPtr();

#ifdef WARPX_QED
        if(loc_has_quantum_sync){
            p_optical_depth = pti.GetAttribs(particle_comps["opticalDepthQSR"]).dataPtr()
                              + offset;
            optical_depth_save.resize(np_to_push);
            p_optical_depth_save = optical_depth_save.dataPtr();
        }
#endif

        amrex::ParallelFor( np_to_push,
                            [=] AMREX_GPU_DEVICE (long i) {
                                amrex::ParticleReal xp, yp, zp;
                                GetPosition(i, xp, yp, zp);
                                xp_save_ptr[i] = xp;
                                yp_save_ptr[i] = yp;
                                zp_save_ptr[i] = zp;
                                uxp_save_ptr[i] = ux[i];
                                uyp_save_ptr[i] = uy[i];
                                uzp_save_ptr[i] = uz[i];
#ifdef WARPX_QED
                                if(loc_has_quantum_sync){
                                    p_optical_depth_save[i] = p_optical_depth[i];}
#endif
                            });
    }

    const bool do_scale = not done_injecting_lev;
    const Real v_boost = WarpX::beta_boost*PhysConst::c;
    PhysicalParticleContainer::PushPX(pti, exfab, eyfab, ezfab, bxfab, byfab, bzfab,
                                      ngEB, e_is_nodal, offset, np_to_push, lev, gather_lev, dt,
                                      ScaleFields(do_scale, dt, zinject_plane_lev_previous,
                                                  vzbeam_ave_boosted, v_boost),
                                      a_dt_type);

    if (!done_injecting_lev) {

        amrex::ParticleReal* AMREX_RESTRICT x_save = xp_save.dataPtr();
        amrex::ParticleReal* AMREX_RESTRICT y_save = yp_save.dataPtr();
        amrex::ParticleReal* AMREX_RESTRICT z_save = zp_save.dataPtr();
        amrex::ParticleReal* AMREX_RESTRICT ux_save = uxp_save.dataPtr();
        amrex::ParticleReal* AMREX_RESTRICT uy_save = uyp_save.dataPtr();
        amrex::ParticleReal* AMREX_RESTRICT uz_save = uzp_save.dataPtr();

        // Undo the push for particles not injected yet.
        // The zp are advanced a fixed amount.
        const amrex::ParticleReal z_plane_lev = zinject_plane_lev;
        const amrex::ParticleReal vz_ave_boosted = vzbeam_ave_boosted;
        const bool rigid = rigid_advance;
        constexpr amrex::ParticleReal inv_csq = 1._prt/(PhysConst::c*PhysConst::c);
        amrex::ParallelFor( np_to_push,
                            [=] AMREX_GPU_DEVICE (long i) {
                                amrex::ParticleReal xp, yp, zp;
                                GetPosition(i, xp, yp, zp);
                                if (zp <= z_plane_lev) {
                                    ux[i] = ux_save[i];
                                    uy[i] = uy_save[i];
                                    uz[i] = uz_save[i];
                                    xp = x_save[i];
                                    yp = y_save[i];
                                    if (rigid) {
                                        zp = z_save[i] + dt*vz_ave_boosted;
                                    }
                                    else {
                                        const amrex::ParticleReal gi = 1._prt/std::sqrt(1._prt + (ux[i]*ux[i]
                                                             + uy[i]*uy[i] + uz[i]*uz[i])*inv_csq);
                                        zp = z_save[i] + dt*uz[i]*gi;
                                    }
                                    SetPosition(i, xp, yp, zp);
#ifdef WARPX_QED
                                    if(loc_has_quantum_sync){
                                        p_optical_depth[i] = p_optical_depth_save[i];}
#endif
                                }
                            });
    }
}

void
RigidInjectedParticleContainer::Evolve (ablastr::fields::MultiFabRegister& fields,
                                        int lev,
                                        const std::string& current_fp_string,
                                        Real t, Real dt, DtType a_dt_type, bool skip_deposition,
                                        PushType push_type)
{

    // Update location of injection plane in the boosted frame
    zinject_plane_lev_previous = zinject_plane_levels[lev];
    zinject_plane_levels[lev] -= dt*WarpX::beta_boost*PhysConst::c;
    zinject_plane_lev = zinject_plane_levels[lev];

    // Set the done injecting flag when the inject plane moves out of the
    // simulation domain.
    // It is much easier to do this check, rather than checking if all of the
    // particles have crossed the inject plane.
    const Real* plo = Geom(lev).ProbLo();
    const Real* phi = Geom(lev).ProbHi();
    done_injecting_lev = ((zinject_plane_levels[lev] < plo[WARPX_ZINDEX] && WarpX::moving_window_v + WarpX::beta_boost*PhysConst::c >= 0.) ||
                           (zinject_plane_levels[lev] > phi[WARPX_ZINDEX] && WarpX::moving_window_v + WarpX::beta_boost*PhysConst::c <= 0.));

    PhysicalParticleContainer::Evolve (fields,
                                       lev,
                                       current_fp_string,
                                       t, dt, a_dt_type, skip_deposition, push_type);
}

void
RigidInjectedParticleContainer::PushP (int lev, Real dt,
                                       const MultiFab& Ex, const MultiFab& Ey, const MultiFab& Ez,
                                       const MultiFab& Bx, const MultiFab& By, const MultiFab& Bz)
{
    WARPX_PROFILE("RigidInjectedParticleContainer::PushP");

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


            const Dim3 lo = lbound(box);

            const bool galerkin_interpolation = WarpX::galerkin_interpolation;
            const int nox = WarpX::nox;
            const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;

            const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);

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
            amrex::ParticleReal* const AMREX_RESTRICT uxpp = attribs[PIdx::ux].dataPtr();
            amrex::ParticleReal* const AMREX_RESTRICT uypp = attribs[PIdx::uy].dataPtr();
            amrex::ParticleReal* const AMREX_RESTRICT uzpp = attribs[PIdx::uz].dataPtr();

            int* AMREX_RESTRICT ion_lev = nullptr;
            if (do_field_ionization) {
                ion_lev = pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr();
            }

            // Save the position and momenta, making copies
            amrex::Gpu::DeviceVector<ParticleReal> uxp_save(np);
            amrex::Gpu::DeviceVector<ParticleReal> uyp_save(np);
            amrex::Gpu::DeviceVector<ParticleReal> uzp_save(np);
            ParticleReal* const AMREX_RESTRICT ux_save = uxp_save.dataPtr();
            ParticleReal* const AMREX_RESTRICT uy_save = uyp_save.dataPtr();
            ParticleReal* const AMREX_RESTRICT uz_save = uzp_save.dataPtr();

            // Loop over the particles and update their momentum
            const amrex::ParticleReal q = this->charge;
            const amrex::ParticleReal m = this-> mass;

            const auto pusher_algo = WarpX::particle_pusher_algo;
            const auto do_crr = do_classical_radiation_reaction;

            enum exteb_flags : int { no_exteb, has_exteb };

            const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;

            amrex::ParallelFor(TypeList<CompileTimeOptions<no_exteb,has_exteb>>{},
                               {exteb_runtime_flag},
                               np, [=] AMREX_GPU_DEVICE (long ip, auto exteb_control)
            {
                ux_save[ip] = uxpp[ip];
                uy_save[ip] = uypp[ip];
                uz_save[ip] = uzpp[ip];

                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                amrex::ParticleReal Exp = Ex_external_particle;
                amrex::ParticleReal Eyp = Ey_external_particle;
                amrex::ParticleReal Ezp = Ez_external_particle;
                amrex::ParticleReal Bxp = Bx_external_particle;
                amrex::ParticleReal Byp = By_external_particle;
                amrex::ParticleReal Bzp = Bz_external_particle;

                // first gather E and B to the particle positions
                doGatherShapeN(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                               ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                               ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                               dinv, xyzmin, lo, n_rz_azimuthal_modes,
                               nox, galerkin_interpolation);

                [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB;
                if constexpr (exteb_control == has_exteb) {
                    getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
                }

                amrex::ParticleReal qp = q;
                if (ion_lev) { qp *= ion_lev[ip]; }

                if (do_crr) {
                    UpdateMomentumBorisWithRadiationReaction(uxpp[ip], uypp[ip], uzpp[ip],
                                                             Exp, Eyp, Ezp, Bxp,
                                                             Byp, Bzp, qp, m, dt);
                } else if (pusher_algo == ParticlePusherAlgo::Boris) {
                    UpdateMomentumBoris( uxpp[ip], uypp[ip], uzpp[ip],
                                         Exp, Eyp, Ezp, Bxp,
                                         Byp, Bzp, qp, m, dt);
                } else if (pusher_algo == ParticlePusherAlgo::Vay) {
                    UpdateMomentumVay( uxpp[ip], uypp[ip], uzpp[ip],
                                       Exp, Eyp, Ezp, Bxp,
                                       Byp, Bzp, qp, m, dt);
                } else if (pusher_algo == ParticlePusherAlgo::HigueraCary) {
                    UpdateMomentumHigueraCary( uxpp[ip], uypp[ip], uzpp[ip],
                                               Exp, Eyp, Ezp, Bxp,
                                               Byp, Bzp, qp, m, dt);
                } else {
                    amrex::Abort("Unknown particle pusher");
                }
            });

            // Undo the push for particles not injected yet.
            // It is assumed that PushP will only be called on the first and last steps
            // and that no particles will cross zinject_plane.
            const ParticleReal zz = zinject_plane_levels[lev];
            amrex::ParallelFor( pti.numParticles(), [=] AMREX_GPU_DEVICE (long i)
            {
                ParticleReal xp, yp, zp;
                getPosition(i, xp, yp, zp);
                if (zp <= zz) {
                    uxpp[i] = ux_save[i];
                    uypp[i] = uy_save[i];
                    uzpp[i] = uz_save[i];
                }
            });

            amrex::Gpu::synchronize();
        }
    }
}
