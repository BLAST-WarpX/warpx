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

namespace {

    enum exteb_flags : int { no_exteb, has_exteb };
    enum qed_flags : int { no_qed, has_qed };
    enum depos_order_flags : int { order_one = 1, order_two, order_three, order_four };

    template<int exteb_control, int qed_control>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    bool PushXPSingleStep (
        int const & ip,
        amrex::Real const & dt,
        SetParticlePosition<PIdx> const &  setPosition,
        amrex::ParticleReal & xp,
        amrex::ParticleReal & yp,
        amrex::ParticleReal & zp,
        amrex::ParticleReal * const ux,
        amrex::ParticleReal * const uy,
        amrex::ParticleReal * const uz,
        amrex::ParticleReal const & xp_n,
        amrex::ParticleReal const & yp_n,
        amrex::ParticleReal const & zp_n,
        amrex::ParticleReal const & uxp_n,
        amrex::ParticleReal const & uyp_n,
        amrex::ParticleReal const & uzp_n,
        amrex::ParticleReal & step_norm,
        amrex::ParticleReal const & particle_tolerance,
        int const & max_iterations,
        amrex::ParticleReal const & Ex_external_particle,
        amrex::ParticleReal const & Ey_external_particle,
        amrex::ParticleReal const & Ez_external_particle,
        amrex::ParticleReal const & Bx_external_particle,
        amrex::ParticleReal const & By_external_particle,
        amrex::ParticleReal const & Bz_external_particle,
        int const & t_do_not_gather,
        amrex::Array4<const amrex::Real> const & ex_arr,
        amrex::Array4<const amrex::Real> const & ey_arr,
        amrex::Array4<const amrex::Real> const & ez_arr,
        amrex::Array4<const amrex::Real> const & bx_arr,
        amrex::Array4<const amrex::Real> const & by_arr,
        amrex::Array4<const amrex::Real> const & bz_arr,
        amrex::IndexType const & ex_type,
        amrex::IndexType const & ey_type,
        amrex::IndexType const & ez_type,
        amrex::IndexType const & bx_type,
        amrex::IndexType const & by_type,
        amrex::IndexType const & bz_type,
        amrex::XDim3 const & dinv,
        amrex::XDim3 const & xyzmin,
        amrex::Dim3 const & lo,
        int const & n_rz_azimuthal_modes,
        int const & depos_order,
        CurrentDepositionAlgo const & depos_type,
        GetExternalEBField const & getExternalEB,
        ScaleFields const & scaleFields,
        int const * const ion_lev,
        amrex::ParticleReal const & m,
        amrex::ParticleReal const & q,
        ParticlePusherAlgo const & pusher_algo,
        bool const & do_crr
#ifdef WARPX_QED
        , bool const & do_sync,
        amrex::Real t_chi_max,
        amrex::ParticleReal * p_optical_depth_QSR,
        QuantumSynchrotronEvolveOpticalDepth const & evolve_opt
#endif
    )
    {
        amrex::ParticleReal dxp_save;
        amrex::ParticleReal dyp_save;
        amrex::ParticleReal dzp_save;

        auto idxg2 = static_cast<amrex::ParticleReal>(dinv.x*dinv.x);
        auto idyg2 = static_cast<amrex::ParticleReal>(dinv.y*dinv.y);
        auto idzg2 = static_cast<amrex::ParticleReal>(dinv.z*dinv.z);

        bool convergence = false;
        for (int iter=0; iter < max_iterations; iter++) {

            // Position is advanced using the time-centered velocity.
            // A converged advance will have self-consistent position and velocity.
            amrex::ParticleReal dxp = 0.0_prt;
            amrex::ParticleReal dyp = 0.0_prt;
            amrex::ParticleReal dzp = 0.0_prt;
            UpdatePositionImplicit(dxp, dyp, dzp, uxp_n, uyp_n, uzp_n, ux[ip], uy[ip], uz[ip], 0.5_rt*dt);
            xp = xp_n + dxp;
            yp = yp_n + dyp;
            zp = zp_n + dzp;
            setPosition(ip, xp, yp, zp);

            PositionNorm(dxp, dyp, dzp, dxp_save, dyp_save, dzp_save,
                         idxg2, idyg2, idzg2, step_norm, iter);
            if (step_norm < particle_tolerance) {
                convergence = true;
                break;
            }

            amrex::ParticleReal Exp = Ex_external_particle;
            amrex::ParticleReal Eyp = Ey_external_particle;
            amrex::ParticleReal Ezp = Ez_external_particle;
            amrex::ParticleReal Bxp = Bx_external_particle;
            amrex::ParticleReal Byp = By_external_particle;
            amrex::ParticleReal Bzp = Bz_external_particle;

            if (!t_do_not_gather){
                // first gather E and B to the particle positions
                doGatherShapeNImplicit(xp_n, yp_n, zp_n, xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                       ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                       ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                       dinv, xyzmin, lo, n_rz_azimuthal_modes, depos_order,
                                       depos_type );
            }

            // Externally applied E and B-field in Cartesian co-ordinates
            [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB;
            if constexpr (exteb_control == has_exteb) {
                getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
            }

            scaleFields(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp);

            // The momentum push starts with the velocity at the start of the step
            ux[ip] = uxp_n;
            uy[ip] = uyp_n;
            uz[ip] = uzp_n;

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
            [[maybe_unused]] auto *foo_podq = p_optical_depth_QSR;
            [[maybe_unused]] const auto& foo_evolve_opt = evolve_opt; // have to do all these for nvcc
            if constexpr (qed_control == has_qed) {
                if (p_optical_depth_QSR) {
                    evolve_opt(ux[ip], uy[ip], uz[ip],
                               Exp, Eyp, Ezp,Bxp, Byp, Bzp,
                               dt, p_optical_depth_QSR[ip]);
                }
            }
#else
            amrex::ignore_unused(qed_control);
#endif

            // Take average to get the time centered value
            ux[ip] = 0.5_rt*(ux[ip] + uxp_n);
            uy[ip] = 0.5_rt*(uy[ip] + uyp_n);
            uz[ip] = 0.5_rt*(uz[ip] + uzp_n);

        }

        return convergence;
    }
}

/* \brief Perform the implicit particle push operation in one fused kernel
 *        The main difference from PushPX is the order of operations:
 *         - push position by 1/2 dt
 *         - gather fields
 *         - push velocity by dt
 *         - average old and new velocity to get time centered value
 *        The routines ends with both position and velocity at the half time level.
 *
 * \param[in] pti The WarpXParIter holding the particles to push
 * \param[in] exfab, eyfab, ezfab The E fields
 * \param[in] bxfab, byfab, bzfab The B fields
 * \param[in] ngEB The number of guard cells in the E and B fields
 * \param[in] offset The particle index offset for the particles to be pushed
 * \param[in] np_to_push The number of particles to push
 * \param[in] lev The refinement level
 * \param[in] gather_lev The refinement level at which to do the field gather
 * \param[in] dt The time step size
 * \param[in] scaleFields Allows scale factor to the fields (for rigid injection)
 * \param[in] a_dt_type The push type (which part of the time step)
 */
void
PhysicalParticleContainer::ImplicitPushXP (WarpXParIter& pti,
                                           amrex::FArrayBox const * exfab,
                                           amrex::FArrayBox const * eyfab,
                                           amrex::FArrayBox const * ezfab,
                                           amrex::FArrayBox const * bxfab,
                                           amrex::FArrayBox const * byfab,
                                           amrex::FArrayBox const * bzfab,
                                           ImplicitOptions const * implicit_options,
                                           amrex::IntVect const & ngEB,
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
    const int depos_order = WarpX::nox;
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

    const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;
#ifdef WARPX_QED
    const int qed_runtime_flag = (local_has_quantum_sync || do_sync) ? has_qed : no_qed;
#else
    const int qed_runtime_flag = no_qed;
#endif

    const int max_iterations = implicit_options->max_particle_iterations;
    const amrex::ParticleReal particle_tolerance = implicit_options->particle_tolerance;

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

        if (do_copy) {
            //  Copy the old x and u for the BTD
            copyAttribs(ip);
        }

#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal xp = x_n[ip];
        const amrex::ParticleReal xp_n = x_n[ip];
#else
        amrex::ParticleReal xp = 0._prt;
        const amrex::ParticleReal xp_n = 0._prt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ParticleReal yp = y_n[ip];
        const amrex::ParticleReal yp_n = y_n[ip];
#else
        amrex::ParticleReal yp = 0._prt;
        const amrex::ParticleReal yp_n = 0._prt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
        amrex::ParticleReal zp = z_n[ip];
        const amrex::ParticleReal zp_n = z_n[ip];
#else
        amrex::ParticleReal zp = 0._prt;
        const amrex::ParticleReal zp_n = 0._prt;
#endif

#ifdef WARPX_QED
        amrex::ParticleReal p_optical_depth_QSR0 = 0.0_prt;
        if (p_optical_depth_QSR) {
            p_optical_depth_QSR0 = p_optical_depth_QSR[ip];
        }
#endif

        amrex::ParticleReal step_norm = 1._prt;
        bool convergence = PushXPSingleStep<exteb_control, qed_control>(ip, dt, setPosition,
                             xp, yp, zp, ux, uy, uz, xp_n, yp_n, zp_n, ux_n[ip], uy_n[ip], uz_n[ip],
                             step_norm, particle_tolerance, max_iterations,
                             Ex_external_particle, Ey_external_particle, Ez_external_particle,
                             Bx_external_particle, By_external_particle, Bz_external_particle,
                             t_do_not_gather, ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                             ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                             dinv, xyzmin, lo, n_rz_azimuthal_modes, depos_order, depos_type,
                             getExternalEB, scaleFields, ion_lev, m, q, pusher_algo, do_crr
#ifdef WARPX_QED
                             , do_sync, t_chi_max, p_optical_depth_QSR, evolve_opt
#endif
                             );

        // particle did not converge
        if (max_iterations > 1 && !convergence) {
#if !defined(AMREX_USE_GPU)
            std::stringstream convergenceMsg;
            convergenceMsg << "Picard solver for particle failed to converge after " <<
                max_iterations << " iterations.\n";
            convergenceMsg << "Position step norm is " << step_norm <<
                " and the tolerance is " << particle_tolerance << "\n";
            convergenceMsg << " ux = " << ux[ip] << ", uy = " << uy[ip] << ", uz = " << uz[ip] << "\n";
            convergenceMsg << " xp = " << xp << ", yp = " << yp << ", zp = " << zp;
            ablastr::warn_manager::WMRecordWarning("ImplicitPushXP", convergenceMsg.str());
#endif

#ifdef WARPX_QED
                // Reset the QED parameter to what is was at the start of the step
                if (p_optical_depth_QSR) {
                    p_optical_depth_QSR[ip] = p_optical_depth_QSR0;
                }
#endif

            // Write signaling flag: how many particles did not converge?
            amrex::Gpu::Atomic::Add(unconverged_particles_ptr, amrex::Long(1));
        }
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
