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
#include "Particles/Deposition/CurrentDeposition.H"
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
#include <AMReX_Print.H>
#include <AMReX_Scan.H>
#include <AMReX_Utility.H>

using namespace amrex::literals;

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
                                           long & num_unconverged_particles,
                                           amrex::Gpu::DeviceVector<long> & unconverged_indices,
                                           amrex::Gpu::DeviceVector<amrex::ParticleReal> & saved_weights,
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
    amrex::Box box;
    if (lev == gather_lev) {
        box = pti.tilebox();
    } else {
        const amrex::IntVect& ref_ratio = WarpX::RefRatio(gather_lev);
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
    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, gather_lev, 0.0_rt);

    const amrex::Dim3 lo = lbound(box);

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
    amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr() + offset;
    amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr() + offset;
    amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr() + offset;
    amrex::ParticleReal* const AMREX_RESTRICT w  = attribs[PIdx::w ].dataPtr() + offset;

    auto * const AMREX_RESTRICT idcpu = pti.GetStructOfArrays().GetIdCPUData().data() + offset;

#if !defined(WARPX_DIM_1D_Z)
    amrex::ParticleReal* x_n = pti.GetAttribs("x_n").dataPtr() + offset;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ParticleReal* y_n = pti.GetAttribs("y_n").dataPtr() + offset;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
    amrex::ParticleReal* z_n = pti.GetAttribs("z_n").dataPtr() + offset;
#endif
    amrex::ParticleReal* ux_n = pti.GetAttribs("ux_n").dataPtr() + offset;
    amrex::ParticleReal* uy_n = pti.GetAttribs("uy_n").dataPtr() + offset;
    amrex::ParticleReal* uz_n = pti.GetAttribs("uz_n").dataPtr() + offset;

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
    amrex::ParallelFor(amrex::TypeList<amrex::CompileTimeOptions<no_exteb,has_exteb>,
                                       amrex::CompileTimeOptions<no_qed  ,has_qed>>{},
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

#ifdef WARPX_QED
        amrex::ParticleReal p_optical_depth_QSR0 = 0.0_prt;
        if (local_has_quantum_sync) {
            p_optical_depth_QSR0 = p_optical_depth_QSR[ip];
        }
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
#if !defined(WARPX_DIM_RCYLINDER)
            zp = zp_n + dzp;
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

                // Flag the particle as invalid. It will be handled later in a special
                // loop with suborbiting.
                amrex::ParticleIDWrapper{idcpu[ip]}.make_invalid();

#ifdef WARPX_QED
                // Reset the QED parameter to what is was at the start of the step
                if (local_has_quantum_sync) {
                    p_optical_depth_QSR[ip] = p_optical_depth_QSR0;
                }
#endif

                // write signaling flag: how many particles did not converge?
                amrex::Gpu::Atomic::Add(unconverged_particles_ptr, amrex::Long(1));
            }

        } // end Picard iterations

    });

    amrex::Gpu::synchronize();

    // Setup for handling the unconverged particles. A list of their indices is
    // gathered, their weights saved, and their weight set to zero (so they
    // don't contribute to the current density).
    num_unconverged_particles = *(unconverged_particles.copyToHost());

    auto num_previous = unconverged_indices.size();
    unconverged_indices.resize(num_previous + num_unconverged_particles);
    saved_weights.resize(num_previous + num_unconverged_particles);

    long * unconverged_i = unconverged_indices.data() + num_previous;
    amrex::ParticleReal * saved_w = saved_weights.data() + num_previous;

    long num_flagged = amrex::Scan::PrefixSum<long>(np_to_push,
        [=] AMREX_GPU_DEVICE (long ip) -> long
            {
                auto pidw = amrex::ParticleIDWrapper{idcpu[ip]};
                return !pidw.is_valid();
            },
        [=] AMREX_GPU_DEVICE (long ip, long x) // x is the exclusive sum at position ip
            {
                auto pidw = amrex::ParticleIDWrapper{idcpu[ip]};
                if (!pidw.is_valid()) {
                    if (x < num_unconverged_particles)  {
                        // The index saved is relative to the full array
                        unconverged_i[x] = ip + offset;
                        saved_w[x] = w[ip];
                        w[ip] = 0.0_prt;
                    }
                }
            },
         amrex::Scan::Type::exclusive, amrex::Scan::retSum);

     WARPX_ALWAYS_ASSERT_WITH_MESSAGE(num_flagged == num_unconverged_particles,
                                      "ImplicitPushXP: wrong number of invalid particles found");

    if (num_unconverged_particles > 0) {
        ablastr::warn_manager::WMRecordWarning("ImplicitPushXP",
            "Picard solver for " +
            std::to_string(num_unconverged_particles) +
            " particles failed to converge after " +
            std::to_string(max_iterations) + " iterations."
         );
    }
}

/* \brief Perform the implicit particle push operation in one fused kernel
 *        using suborbits. This routine is used for particles where the
 *        iteration in ImplicitPushXP failed to converge.
 */
void
PhysicalParticleContainer::ImplicitPushXPSubOrbits (WarpXParIter& pti,
                                                    amrex::FArrayBox const * exfab,
                                                    amrex::FArrayBox const * eyfab,
                                                    amrex::FArrayBox const * ezfab,
                                                    amrex::FArrayBox const * bxfab,
                                                    amrex::FArrayBox const * byfab,
                                                    amrex::FArrayBox const * bzfab,
                                                    amrex::IntVect ngEB,
                                                    amrex::MultiFab * const jx,
                                                    amrex::MultiFab * const jy,
                                                    amrex::MultiFab * const jz,
                                                    long index_offset,
                                                    long num_unconverged_particles,
                                                    int lev, int gather_lev,
                                                    amrex::Real dt, ScaleFields scaleFields,
                                                    bool skip_deposition,
                                                    amrex::Gpu::DeviceVector<long> & unconverged_indices,
                                                    amrex::Gpu::DeviceVector<amrex::ParticleReal> & saved_weights)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((gather_lev==(lev-1)) ||
                                     (gather_lev==(lev  )),
                                     "Gather buffers only work for lev-1");
    // If no particles, do not do anything
    if (num_unconverged_particles == 0) { return; }

    // Get cell size on gather_lev
    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(gather_lev,0));
    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    // Get box from which field is gathered.
    // If not gathering from the finest level, the box is coarsened.
    amrex::Box box;
    if (lev == gather_lev) {
        box = pti.tilebox();
    } else {
        const amrex::IntVect& ref_ratio = WarpX::RefRatio(gather_lev);
        box = amrex::coarsen(pti.tilebox(),ref_ratio);
    }

    // Add guard cells to the box.
    box.grow(ngEB);

    auto setPosition = SetParticlePosition(pti, 0);

    const auto getExternalEB = GetExternalEBField(pti, 0);

    const amrex::ParticleReal Ex_external_particle = m_E_external_particle[0];
    const amrex::ParticleReal Ey_external_particle = m_E_external_particle[1];
    const amrex::ParticleReal Ez_external_particle = m_E_external_particle[2];
    const amrex::ParticleReal Bx_external_particle = m_B_external_particle[0];
    const amrex::ParticleReal By_external_particle = m_B_external_particle[1];
    const amrex::ParticleReal Bz_external_particle = m_B_external_particle[2];

    // Lower corner of tile box physical domain (take into account Galilean shift)
    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, gather_lev, 0.0_rt);

    const amrex::Dim3 lo = lbound(box);

    const auto depos_type = WarpX::current_deposition_algo;
    const int nox = WarpX::nox;
    const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;

    amrex::Array4<const amrex::Real> const & ex_arr = exfab->array();
    amrex::Array4<const amrex::Real> const & ey_arr = eyfab->array();
    amrex::Array4<const amrex::Real> const & ez_arr = ezfab->array();
    amrex::Array4<const amrex::Real> const & bx_arr = bxfab->array();
    amrex::Array4<const amrex::Real> const & by_arr = byfab->array();
    amrex::Array4<const amrex::Real> const & bz_arr = bzfab->array();
    amrex::Array4<amrex::Real> const & Jx_arr = jx->array(pti);
    amrex::Array4<amrex::Real> const & Jy_arr = jy->array(pti);
    amrex::Array4<amrex::Real> const & Jz_arr = jz->array(pti);

    amrex::IndexType const ex_type = exfab->box().ixType();
    amrex::IndexType const ey_type = eyfab->box().ixType();
    amrex::IndexType const ez_type = ezfab->box().ixType();
    amrex::IndexType const bx_type = bxfab->box().ixType();
    amrex::IndexType const by_type = byfab->box().ixType();
    amrex::IndexType const bz_type = bzfab->box().ixType();

    auto& attribs = pti.GetAttribs();
    amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT w  = attribs[PIdx::w ].dataPtr();

    auto * const AMREX_RESTRICT idcpu = pti.GetStructOfArrays().GetIdCPUData().data();

#if !defined(WARPX_DIM_1D_Z)
    amrex::ParticleReal* x_n = pti.GetAttribs("x_n").dataPtr();
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ParticleReal* y_n = pti.GetAttribs("y_n").dataPtr();
#endif
#if !defined(WARPX_DIM_RCYLINDER)
    amrex::ParticleReal* z_n = pti.GetAttribs("z_n").dataPtr();
#endif
    amrex::ParticleReal* ux_n = pti.GetAttribs("ux_n").dataPtr();
    amrex::ParticleReal* uy_n = pti.GetAttribs("uy_n").dataPtr();
    amrex::ParticleReal* uz_n = pti.GetAttribs("uz_n").dataPtr();

    int* AMREX_RESTRICT ion_lev = nullptr;
    if (do_field_ionization) {
        ion_lev = pti.GetiAttribs("ionizationLevel").dataPtr();
    }
    bool const do_ionization = do_field_ionization;

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
        p_optical_depth_QSR = pti.GetAttribs("opticalDepthQSR").dataPtr();
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

    int constexpr max_suborbits = 10;

    long * unconverged_i = unconverged_indices.data() + index_offset;
    amrex::ParticleReal * saved_w = saved_weights.data() + index_offset;

    // Using this version of ParallelFor with compile time options
    // improves performance when qed or external EB are not used by reducing
    // register pressure.
    amrex::ParallelFor(amrex::TypeList<amrex::CompileTimeOptions<no_exteb,has_exteb>,
                                       amrex::CompileTimeOptions<no_qed  ,has_qed>>{},
                       {exteb_runtime_flag, qed_runtime_flag},
                       num_unconverged_particles, [=] AMREX_GPU_DEVICE (long i,
                                                                 auto exteb_control, auto qed_control)
    {

        long ip = unconverged_i[i];

        // Restore the particle weight
        w[ip] = saved_w[i];

        // Reset valid flag
        auto pidw = amrex::ParticleIDWrapper{idcpu[ip]};
        pidw.make_valid();

        // Create temporary arrays to hold the particle suborbit data
        // which is used to deposit the current of the suborbits after
        // convergence is found
        amrex::GpuArray<amrex::Real, max_suborbits + 1> x_n_save;
        amrex::GpuArray<amrex::Real, max_suborbits + 1> y_n_save;
        amrex::GpuArray<amrex::Real, max_suborbits + 1> z_n_save;
        amrex::GpuArray<amrex::Real, max_suborbits + 1> ux_n_save;
        amrex::GpuArray<amrex::Real, max_suborbits + 1> uy_n_save;
        amrex::GpuArray<amrex::Real, max_suborbits + 1> uz_n_save;

#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal const xp_n0 = x_n[ip];
#else
        amrex::ParticleReal const xp_n0 = 0.0_rt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ParticleReal const yp_n0 = y_n[ip];
#else
        amrex::ParticleReal const yp_n0 = 0.0_rt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
        amrex::ParticleReal const zp_n0 = z_n[ip];
#else
        amrex::ParticleReal const zp_n0 = 0.0_rt;
#endif

#if WARPX_QED
        amrex::ParticleReal p_optical_depth_QSR0 = 0.0_prt;
        if (local_has_quantum_sync) {
            p_optical_depth_QSR0 = p_optical_depth_QSR[ip];
        }
#endif

        amrex::ParticleReal const uxp_n0 = ux_n[ip];
        amrex::ParticleReal const uyp_n0 = uy_n[ip];
        amrex::ParticleReal const uzp_n0 = uz_n[ip];

        amrex::ParticleReal xp_n = xp_n0;
        amrex::ParticleReal yp_n = yp_n0;
        amrex::ParticleReal zp_n = zp_n0;

        amrex::ParticleReal uxp_n = uxp_n0;
        amrex::ParticleReal uyp_n = uyp_n0;
        amrex::ParticleReal uzp_n = uzp_n0;

        amrex::ParticleReal xp = xp_n;
        amrex::ParticleReal yp = yp_n;
        amrex::ParticleReal zp = zp_n;

        amrex::ParticleReal dxp, dxp_save;
        amrex::ParticleReal dyp, dyp_save;
        amrex::ParticleReal dzp, dzp_save;
        auto idxg2 = static_cast<amrex::ParticleReal>(dinv.x*dinv.x);
        auto idyg2 = static_cast<amrex::ParticleReal>(dinv.y*dinv.y);
        auto idzg2 = static_cast<amrex::ParticleReal>(dinv.z*dinv.z);

        // A single step advance already failed, so start here with two suborbit steps
        int num_suborbits = 2;

        int isuborbit = 0;
        while (isuborbit < num_suborbits) {
            x_n_save[isuborbit] = xp_n;
            y_n_save[isuborbit] = yp_n;
            z_n_save[isuborbit] = zp_n;
            ux_n_save[isuborbit] = uxp_n;
            uy_n_save[isuborbit] = uyp_n;
            uz_n_save[isuborbit] = uzp_n;

            amrex::Real const dt_suborbit = dt/num_suborbits;

            // Try advancing the particle one suborbit step
            amrex::ParticleReal step_norm = 1.0_prt;
            int iter;
            for (iter = 0; iter < max_iterations;) {

                dxp = 0.0_prt;
                dyp = 0.0_prt;
                dzp = 0.0_prt;
                UpdatePositionImplicit(dxp, dyp, dzp, uxp_n, uyp_n, uzp_n, ux[ip], uy[ip], uz[ip], 0.5_rt*dt_suborbit);
                xp = xp_n + dxp;
                yp = yp_n + dyp;
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
                                              dt_suborbit);
                }
#ifdef WARPX_QED
                else {
                    if constexpr (qed_control == has_qed) {
                        doParticleMomentumPush<1>(ux[ip], uy[ip], uz[ip],
                                                  Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                                  ion_lev ? ion_lev[ip] : 1,
                                                  m, q, pusher_algo, do_crr,
                                                  t_chi_max,
                                                  dt_suborbit);
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
                                   dt_suborbit, p_optical_depth_QSR[ip]);
                    }
                }
#else
                amrex::ignore_unused(qed_control);
#endif

                // Take average to get the time centered value
                ux[ip] = 0.5_rt*(ux[ip] + uxp_n);
                uy[ip] = 0.5_rt*(uy[ip] + uyp_n);
                uz[ip] = 0.5_rt*(uz[ip] + uzp_n);

                iter++;

            } // end Picard iterations

            if ( iter == max_iterations ) {

                // particle did not converge
                // Increase the number of suborbits and start over
                num_suborbits++;
                isuborbit = 0;

                xp_n = xp_n0;
                yp_n = yp_n0;
                zp_n = zp_n0;

                uxp_n = uxp_n0;
                uyp_n = uyp_n0;
                uzp_n = uzp_n0;

                ux[ip] = uxp_n0;
                uy[ip] = uyp_n0;
                uz[ip] = uzp_n0;

#ifdef WARPX_QED
                if (local_has_quantum_sync) {
                    p_optical_depth_QSR[ip] = p_optical_depth_QSR0;
                }
#endif

            } else {

                isuborbit++;

                // That step was successful, update the starting values for the next suborbit
                // interpolating to the end of the step.
                xp_n = 2.0_prt*xp - xp_n;
                yp_n = 2.0_prt*yp - yp_n;
                zp_n = 2.0_prt*zp - zp_n;
                uxp_n = 2.0_prt*ux[ip] - uxp_n;
                uyp_n = 2.0_prt*uy[ip] - uyp_n;
                uzp_n = 2.0_prt*uz[ip] - uzp_n;
                ux[ip] = uxp_n;
                uy[ip] = uyp_n;
                uz[ip] = uzp_n;

            }

            if (num_suborbits >= max_suborbits) {
                // This is very bad
                amrex::Gpu::Atomic::Add(unconverged_particles_ptr, amrex::Long(1));
                break;
            }

        } // end suborbits

        // Set position and momentum to be at the half time level relative to the
        // full time step
        xp = 0.5_prt*(xp_n0 + xp_n);
        yp = 0.5_prt*(yp_n0 + yp_n);
        zp = 0.5_prt*(zp_n0 + zp_n);
        setPosition(ip, xp, yp, zp);

        ux[ip] = 0.5_prt*(uxp_n0 + uxp_n);
        uy[ip] = 0.5_prt*(uyp_n0 + uyp_n);
        uz[ip] = 0.5_prt*(uzp_n0 + uzp_n);


        if (!skip_deposition) {
            // Save the values at the end of the orbit
            x_n_save[isuborbit] = xp_n;
            y_n_save[isuborbit] = yp_n;
            z_n_save[isuborbit] = zp_n;
            ux_n_save[isuborbit] = uxp_n;
            uy_n_save[isuborbit] = uyp_n;
            uz_n_save[isuborbit] = uzp_n;

            amrex::Real wq = q*w[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            wq /= num_suborbits;

            amrex::Real const dt_suborbit = dt/num_suborbits;

            // Deposit the current density from the suborbit steps
            for (int is=0 ; is < num_suborbits ; is++) {
                const amrex::ParticleReal xp_old = x_n_save[is];
                const amrex::ParticleReal yp_old = y_n_save[is];
                const amrex::ParticleReal zp_old = z_n_save[is];
                const amrex::ParticleReal xp_new = x_n_save[is+1];
                const amrex::ParticleReal yp_new = y_n_save[is+1];
                const amrex::ParticleReal zp_new = z_n_save[is+1];

                const amrex::ParticleReal uxp_old = ux_n_save[is];
                const amrex::ParticleReal uyp_old = uy_n_save[is];
                const amrex::ParticleReal uzp_old = uz_n_save[is];
                const amrex::ParticleReal uxp_new = ux_n_save[is+1];
                const amrex::ParticleReal uyp_new = uy_n_save[is+1];
                const amrex::ParticleReal uzp_new = uz_n_save[is+1];
                const amrex::ParticleReal uxp_nph = 0.5_prt*(uxp_old + uxp_new);
                const amrex::ParticleReal uyp_nph = 0.5_prt*(uyp_old + uyp_new);
                const amrex::ParticleReal uzp_nph = 0.5_prt*(uzp_old + uzp_new);

#if !defined(WARPX_DIM_3D)
                constexpr amrex::ParticleReal inv_c2 = 1.0_prt/(PhysConst::c*PhysConst::c);

                // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
                // The uxp,uyp,uzp are the velocities at time level n+1/2
                const amrex::ParticleReal gamma_old = std::sqrt(1.0_prt + (uxp_old*uxp_old + uyp_old*uyp_old + uzp_old*uzp_old)*inv_c2);
                const amrex::ParticleReal gamma_new = std::sqrt(1.0_prt + (uxp_new*uxp_new + uyp_new*uyp_new + uzp_new*uzp_new)*inv_c2);
                const amrex::ParticleReal gaminv = 2.0_prt/(gamma_old + gamma_new);
#else
                // unused
                const amrex::ParticleReal gaminv = 1.;
#endif

                if (depos_type == CurrentDepositionAlgo::Esirkepov) {
                    if (nox == 1) {
                        EsirkepovDepositionShapeNKernel<1>(xp_old, yp_old, zp_old, xp_new, yp_new, zp_new, wq,
                                                           uxp_nph, uyp_nph, uzp_nph, gaminv,
                                                           Jx_arr, Jy_arr, Jz_arr,
                                                           dt_suborbit, dinv, xyzmin, lo, n_rz_azimuthal_modes);
                    } else if (nox == 2) {
                        EsirkepovDepositionShapeNKernel<2>(xp_old, yp_old, zp_old, xp_new, yp_new, zp_new, wq,
                                                           uxp_nph, uyp_nph, uzp_nph, gaminv,
                                                           Jx_arr, Jy_arr, Jz_arr,
                                                           dt_suborbit, dinv, xyzmin, lo, n_rz_azimuthal_modes);
                    } else if (nox == 3) {
                        EsirkepovDepositionShapeNKernel<3>(xp_old, yp_old, zp_old, xp_new, yp_new, zp_new, wq,
                                                           uxp_nph, uyp_nph, uzp_nph, gaminv,
                                                           Jx_arr, Jy_arr, Jz_arr,
                                                           dt_suborbit, dinv, xyzmin, lo, n_rz_azimuthal_modes);
                    } else if (nox == 4) {
                        EsirkepovDepositionShapeNKernel<4>(xp_old, yp_old, zp_old, xp_new, yp_new, zp_new, wq,
                                                           uxp_nph, uyp_nph, uzp_nph, gaminv,
                                                           Jx_arr, Jy_arr, Jz_arr,
                                                           dt_suborbit, dinv, xyzmin, lo, n_rz_azimuthal_modes);
                    }

                } else if (depos_type == CurrentDepositionAlgo::Villasenor) {
                    if (nox == 1) {
                        VillasenorDepositionShapeNKernel<1>(xp_old, yp_old, zp_old, xp_new, yp_new, zp_new, wq,
                                                            uxp_nph, uyp_nph, uzp_nph, gaminv,
                                                            Jx_arr, Jy_arr, Jz_arr,
                                                            dt_suborbit, dinv, xyzmin, lo, invvol, n_rz_azimuthal_modes);
                    }
                    else if (nox == 2) {
                        VillasenorDepositionShapeNKernel<2>(xp_old, yp_old, zp_old, xp_new, yp_new, zp_new, wq,
                                                            uxp_nph, uyp_nph, uzp_nph, gaminv,
                                                            Jx_arr, Jy_arr, Jz_arr,
                                                            dt_suborbit, dinv, xyzmin, lo, invvol, n_rz_azimuthal_modes);
                    }
                    else if (nox == 3) {
                        VillasenorDepositionShapeNKernel<3>(xp_old, yp_old, zp_old, xp_new, yp_new, zp_new, wq,
                                                            uxp_nph, uyp_nph, uzp_nph, gaminv,
                                                            Jx_arr, Jy_arr, Jz_arr,
                                                            dt_suborbit, dinv, xyzmin, lo, invvol, n_rz_azimuthal_modes);
                    }
                    else if (nox == 4) {
                        VillasenorDepositionShapeNKernel<4>(xp_old, yp_old, zp_old, xp_new, yp_new, zp_new, wq,
                                                            uxp_nph, uyp_nph, uzp_nph, gaminv,
                                                            Jx_arr, Jy_arr, Jz_arr,
                                                            dt_suborbit, dinv, xyzmin, lo, invvol, n_rz_azimuthal_modes);
                    }

                }
            }
        }


    });

    long const num_failed_particles = *(unconverged_particles.copyToHost());
    if (num_failed_particles > 0) {
        ablastr::warn_manager::WMRecordWarning("ImplicitPushXPSubOrbits",
            "Picard solver for " +
            std::to_string(num_failed_particles) +
            " particles failed to converge after " +
            std::to_string(max_iterations) + " iterations and " + std::to_string(max_suborbits) + " sub-orbits."
         );
    }
}
