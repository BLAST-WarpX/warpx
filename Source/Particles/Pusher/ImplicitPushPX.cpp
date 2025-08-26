/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Particles/PhysicalParticleContainer.H"

#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H"
#endif
#include "CopyParticleAttribs.H"
#include "GetAndSetPosition.H"
#include "PushSelector.H"
#include "UpdatePosition.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Dim3.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuBuffer.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleContainerBase.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_Scan.H>
#include <AMReX_Utility.H>

using namespace amrex::literals;
using namespace amrex;

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

#if !defined(WARPX_DIM_1D_Z)
    ParticleReal* x_n = pti.GetAttribs("x_n").dataPtr();
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    ParticleReal* y_n = pti.GetAttribs("y_n").dataPtr();
#endif
#if !defined(WARPX_DIM_RCYLINDER)
    ParticleReal* z_n = pti.GetAttribs("z_n").dataPtr();
#endif
    ParticleReal* ux_n = pti.GetAttribs("ux_n").dataPtr();
    ParticleReal* uy_n = pti.GetAttribs("uy_n").dataPtr();
    ParticleReal* uz_n = pti.GetAttribs("uz_n").dataPtr();

    const int do_copy = (m_do_back_transformed_particles && (a_dt_type!=DtType::SecondHalf) );
    CopyParticleAttribs copyAttribs;
    if (do_copy) {
        copyAttribs = CopyParticleAttribs(*this, pti, offset);
    }

    int* AMREX_RESTRICT ion_lev = nullptr;
    if (do_field_ionization) {
        ion_lev = pti.GetiAttribs("ionizationLevel").dataPtr() + offset;
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

#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal xp = x_n[ip];
        const amrex::ParticleReal xp_n = x_n[ip];
#else
        const amrex::ParticleReal xp = 0._rt;
        const amrex::ParticleReal xp_n = 0._rt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ParticleReal yp = y_n[ip];
        const amrex::ParticleReal yp_n = y_n[ip];
#else
        const amrex::ParticleReal yp = 0._rt;
        const amrex::ParticleReal yp_n = 0._rt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
        amrex::ParticleReal zp = z_n[ip];
        const amrex::ParticleReal zp_n = z_n[ip];
#else
        amrex::ParticleReal zp = 0._rt;
        const amrex::ParticleReal zp_n = 0._rt;
#endif

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
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            yp = yp_n + dyp;
#endif
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
