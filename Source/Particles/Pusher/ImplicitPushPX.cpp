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
#include "Particles/Deposition/MassMatricesDeposition.H"
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
        amrex::ParticleReal & Bxp,
        amrex::ParticleReal & Byp,
        amrex::ParticleReal & Bzp,
        int const & do_gather,
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
        int const * const ion_lev,
        amrex::ParticleReal const & mass,
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

        auto idxg2 = static_cast<amrex::ParticleReal>(dinv.x*dinv.x);
        auto idyg2 = static_cast<amrex::ParticleReal>(dinv.y*dinv.y);
        auto idzg2 = static_cast<amrex::ParticleReal>(dinv.z*dinv.z);

        // Picard fixed-point iteration method for self-consistent update of position and velocity
        //     Compute initial value of dxp (xp_np1 = xp_n + dxp)
        //     Picard iterations {
        //         velocity push
        //         position push
        //         check step norm of dxp for convergence
        //     }
        // The initial velocity used to compute the intial value of dxp is either the time-centered
        // velocity from the end of the previous nonlinear iteration or the velocity at the start of
        // the step if being called from the suborbit routine.
        //
        // Note: The charge-conserving deposits assume the change in position is consistent with
        // the velocity: (xp^{n+1}-xp^n)/dt = vp^{n+1/2}. This requires finishing the iterations
        // with the position updated, even in situations where convergence is not obtained.

        // Perform an initial position push to set the initial guess for dxp
        amrex::ParticleReal dxp = 0.0_prt;
        amrex::ParticleReal dyp = 0.0_prt;
        amrex::ParticleReal dzp = 0.0_prt;
        UpdatePositionImplicit(dxp, dyp, dzp, uxp_n, uyp_n, uzp_n, ux[ip], uy[ip], uz[ip], 0.5_rt*dt);
        xp = xp_n + dxp;
        yp = yp_n + dyp;
        zp = zp_n + dzp;
        setPosition(ip, xp, yp, zp);

        bool convergence = false;
        for (int iter=0; iter < max_iterations; iter++) {

            amrex::ParticleReal Exp = Ex_external_particle;
            amrex::ParticleReal Eyp = Ey_external_particle;
            amrex::ParticleReal Ezp = Ez_external_particle;
            Bxp = Bx_external_particle;
            Byp = By_external_particle;
            Bzp = Bz_external_particle;

            if (do_gather) {
                // first gather E and B to the particle positions
                doGatherShapeNImplicit(xp_n, yp_n, zp_n, xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                       ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                       ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                       dinv, xyzmin, lo, n_rz_azimuthal_modes, depos_order, depos_type);
            }

            // Externally applied E and B-field in Cartesian co-ordinates
            [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB;
            if constexpr (exteb_control == has_exteb) {
                getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
            }

            // The momentum push starts with the velocity at the start of the step
            ux[ip] = uxp_n;
            uy[ip] = uyp_n;
            uz[ip] = uzp_n;

#ifdef WARPX_QED
            if (!do_sync) {
                doParticleMomentumPush<0>(ux[ip], uy[ip], uz[ip],
                                          Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                          ion_lev ? ion_lev[ip] : 1,
                                          mass, q, pusher_algo, do_crr,
                                          t_chi_max,
                                          dt);
            } else {
                if constexpr (qed_control == has_qed) {
                    doParticleMomentumPush<1>(ux[ip], uy[ip], uz[ip],
                                              Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                              ion_lev ? ion_lev[ip] : 1,
                                              mass, q, pusher_algo, do_crr,
                                              t_chi_max,
                                              dt);
                }
            }
#else
            doParticleMomentumPush<0>(ux[ip], uy[ip], uz[ip],
                                      Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                      ion_lev ? ion_lev[ip] : 1,
                                      mass, q, pusher_algo, do_crr,
                                      dt);
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

            // Take average to get the time-centered value
            ux[ip] = 0.5_prt*(ux[ip] + uxp_n);
            uy[ip] = 0.5_prt*(uy[ip] + uyp_n);
            uz[ip] = 0.5_prt*(uz[ip] + uzp_n);

            // Save position change from previous position push for step norm calculation
            const amrex::ParticleReal dxp_save = dxp;
            const amrex::ParticleReal dyp_save = dyp;
            const amrex::ParticleReal dzp_save = dzp;

            // Update the particle position using the time-centered velocity
            dxp = 0.0_prt;
            dyp = 0.0_prt;
            dzp = 0.0_prt;
            UpdatePositionImplicit(dxp, dyp, dzp, uxp_n, uyp_n, uzp_n, ux[ip], uy[ip], uz[ip], 0.5_rt*dt);
            xp = xp_n + dxp;
            yp = yp_n + dyp;
            zp = zp_n + dzp;
            setPosition(ip, xp, yp, zp);

            // Check for convergence based on the step norm of the position change
            PositionNorm(dxp, dyp, dzp, dxp_save, dyp_save, dzp_save, idxg2, idyg2, idzg2, step_norm);
            if (step_norm < particle_tolerance) {
                convergence = true;
                break;
            }
        }

        return convergence;
    }
}

/*
 * \brief The routine finds particles that have previously been marked for suborbiting.
 *        The indices and the weights of such particles are saved.
 *
 * \param[in] pti                            The WarpXParIter holding the particles to push
 * \param[in] offset                         The particle index offset for the particles to be pushed
 * \param[in] np_to_push                     The number of particles to push
 * \param[in/out] num_unconverged_particles  Number of unconverged particles
 * \param[in/out] unconverged_indices        The list of indices of unconverged particles
 * \param[in/out] saved_weights              The saved weights of the unconverged particles
 */
void
PhysicalParticleContainer::FindSuborbitParticles (WarpXParIter & pti,
                                                  long offset,
                                                  long np_to_push,
                                                  long & num_unconverged_particles,
                                                  amrex::Gpu::DeviceVector<long> & unconverged_indices,
                                                  amrex::Gpu::DeviceVector<amrex::ParticleReal> & saved_weights)
{
    // If no particles, do not do anything
    if (np_to_push == 0) { return; }

    amrex::Gpu::Buffer<amrex::Long> unconverged_particles({0});
    amrex::Long* unconverged_particles_ptr = unconverged_particles.data();
    int *nsuborbits = (HasiAttrib("nsuborbits") ? pti.GetiAttribs("nsuborbits").dataPtr() : nullptr);

    amrex::ParallelFor(
        np_to_push, [=] AMREX_GPU_DEVICE (long ip)
    {

        if (nsuborbits && nsuborbits[ip] > 1) {
            // write signaling flag: how many particles did not converge?
            amrex::Gpu::Atomic::Add(unconverged_particles_ptr, amrex::Long(1));
            return;
        }

    });

    // Setup for handling the suborbit particles. A list of their indices is
    // gathered, their weights saved, and their weight set to zero (so they
    // don't contribute to the current density).
    num_unconverged_particles = *(unconverged_particles.copyToHost());
    SetupSuborbitParticles(pti, offset, np_to_push, num_unconverged_particles,
                           unconverged_indices, saved_weights);

}

/*
 * \brief Setup for handling the suborbit particles. A list of their indices is
 *        gathered, their weights saved, and their weight set to zero (so they
 *        don't contribute to the current density).
 *
 * \param[in] pti                        The WarpXParIter holding the particles to push
 * \param[in] offset                     The particle index offset for the particles to be pushed
 * \param[in] np_to_push                 The number of particles to push
 * \param[in] num_unconverged_particles  Number of unconverged particles
 * \param[in/out] unconverged_indices    The list of indices of unconverged particles
 * \param[in/out] saved_weights          The saved weights of the unconverged particles
 */
void
PhysicalParticleContainer::SetupSuborbitParticles (WarpXParIter & pti,
                                                   long offset,
                                                   long np_to_push,
                                                   long num_unconverged_particles,
                                                   amrex::Gpu::DeviceVector<long> & unconverged_indices,
                                                   amrex::Gpu::DeviceVector<amrex::ParticleReal> & saved_weights)
{
    // If no unconverged particles, do not do anything
    if (num_unconverged_particles == 0) { return; }

    auto& attribs = pti.GetAttribs();
    amrex::ParticleReal* const AMREX_RESTRICT w  = attribs[PIdx::w ].dataPtr() + offset;

    int *nsuborbits = (HasiAttrib("nsuborbits") ? pti.GetiAttribs("nsuborbits").dataPtr() : nullptr);

    auto num_previous = unconverged_indices.size();
    unconverged_indices.resize(num_previous + num_unconverged_particles);
    saved_weights.resize(num_previous + num_unconverged_particles);

    long * unconverged_i = unconverged_indices.data() + num_previous;
    amrex::ParticleReal * saved_w = saved_weights.data() + num_previous;

    if (nsuborbits) {
        // This looks for the particles that require suborbits to converge.
        long num_flagged = amrex::Scan::PrefixSum<long>(np_to_push,
            [=] AMREX_GPU_DEVICE (long ip) -> long
                {
                    return nsuborbits[ip] > 1;
                },
            [=] AMREX_GPU_DEVICE (long ip, long x) // x is the exclusive sum at position ip
                {
                    if (nsuborbits[ip] > 1) {
                        // This check of x should always be true but is here for memory safety
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
                                          "SetupSuborbitParticles: wrong number of invalid particles found");
    }

}

/* \brief Perform the implicit particle push operation in one fused kernel
 *        The main difference from PushPX is the order of operations:
 *         - push position by 1/2 dt
 *         - gather fields
 *         - push velocity by dt
 *         - average old and new velocity to get time centered value
 *        The routines ends with both position and velocity at the half time level.
 *        The routine iterates the advance until the position and velocity pushes
 *        (which depend on each other) are consistent. Any unconverged particles
 *        are flagged for later processing.
 *
 * \param[in] pti The WarpXParIter holding the particles to push
 * \param[in] exfab, eyfab, ezfab The E fields
 * \param[in] bxfab, byfab, bzfab The B fields
 * \param[in] implicit_options Specifies options for implicit push
 * \param[in] ngEB The number of guard cells in the E and B fields
 * \param[in] offset The particle index offset for the particles to be pushed
 * \param[in] np_to_push The number of particles to push
 * \param[in] lev The refinement level
 * \param[in] gather_lev The refinement level at which to do the field gather
 * \param[in] dt The time step size
 * \param[in/out] num_unconverged_particles Number of unconverged particles
 * \param[in/out] unconverged_indices The list of indices of unconverged particles
 * \param[in/out] saved_weights The saved weights of the unconverged particles
 */
void
PhysicalParticleContainer::ImplicitPushXP (WarpXParIter & pti,
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
                                           amrex::Real dt,
                                           long & num_unconverged_particles,
                                           amrex::Gpu::DeviceVector<long> & unconverged_indices,
                                           amrex::Gpu::DeviceVector<amrex::ParticleReal> & saved_weights)
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
    amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr() + offset;
    amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr() + offset;
    amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr() + offset;

    // The x/y/z_n are the positions and velocities saved at the start of the step
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

    if (m_do_back_transformed_particles) { //  Copy the old x and u for the BTD
        CopyParticleAttribs copyAttribs = CopyParticleAttribs(*this, pti, offset);
        amrex::ParallelFor(np_to_push, [copyAttribs] AMREX_GPU_DEVICE (long ip)
        {
            copyAttribs(ip);
        });
    }

    int* AMREX_RESTRICT ion_lev = nullptr;
    if (do_field_ionization) {
        ion_lev = pti.GetiAttribs("ionizationLevel").dataPtr() + offset;
    }

    // Loop over the particles and update their momentum
    const amrex::ParticleReal q = this->charge;
    const amrex::ParticleReal mass = this->m_mass;

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

    const auto do_gather = !do_not_gather;

    const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;
#ifdef WARPX_QED
    const int qed_runtime_flag = (local_has_quantum_sync || do_sync) ? has_qed : no_qed;
#else
    const int qed_runtime_flag = no_qed;
#endif

    const int max_iterations = implicit_options->max_particle_iterations;
    const amrex::ParticleReal particle_tolerance = implicit_options->particle_tolerance;
    [[maybe_unused]] const bool print_unconverged_particle_details = implicit_options->print_unconverged_particle_details;

    amrex::Gpu::Buffer<amrex::Long> unconverged_particles({0});
    amrex::Long* unconverged_particles_ptr = unconverged_particles.data();
    int *nsuborbits = (HasiAttrib("nsuborbits") ? pti.GetiAttribs("nsuborbits").dataPtr() : nullptr);

    // Using this version of ParallelFor with compile time options
    // improves performance when qed or external EB are not used by reducing
    // register pressure.
    amrex::ParallelFor(amrex::TypeList<amrex::CompileTimeOptions<no_exteb,has_exteb>,
                                       amrex::CompileTimeOptions<no_qed  ,has_qed>>{},
                       {exteb_runtime_flag, qed_runtime_flag},
                       np_to_push, [=] AMREX_GPU_DEVICE (long ip, auto exteb_control,
                                                         auto qed_control)
    {

        // Skip any particles that require suborbits
        if (nsuborbits && nsuborbits[ip] > 1) {
            // write signaling flag: how many particles did not converge?
            amrex::Gpu::Atomic::Add(unconverged_particles_ptr, amrex::Long(1));
            return;
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

        amrex::ParticleReal Bxp = 0.0_prt;
        amrex::ParticleReal Byp = 0.0_prt;
        amrex::ParticleReal Bzp = 0.0_prt;
        amrex::ParticleReal step_norm = 1._prt;

        bool convergence = PushXPSingleStep<exteb_control, qed_control>(ip, dt, setPosition,
                             xp, yp, zp, ux, uy, uz, xp_n, yp_n, zp_n, ux_n[ip], uy_n[ip], uz_n[ip],
                             step_norm, particle_tolerance, max_iterations,
                             Ex_external_particle, Ey_external_particle, Ez_external_particle,
                             Bx_external_particle, By_external_particle, Bz_external_particle,
                             Bxp, Byp, Bzp,
                             do_gather, ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                             ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                             dinv, xyzmin, lo, n_rz_azimuthal_modes, depos_order, depos_type,
                             getExternalEB, ion_lev, mass, q, pusher_algo, do_crr
#ifdef WARPX_QED
                             , do_sync, t_chi_max, p_optical_depth_QSR, evolve_opt
#endif
                             );

        // check if particle did not converge
        if (max_iterations > 1 && !convergence) {

            if (nsuborbits) {
                // Suborbits are required for this particle to converge.
                // It will be handled later in a special loop with suborbiting.
                nsuborbits[ip] = 2;
            }

#if !defined(AMREX_USE_GPU)
            if (print_unconverged_particle_details) {
                std::stringstream convergenceMsg;
                convergenceMsg << "Picard solver for particle failed to converge after " <<
                    max_iterations << " iterations.\n";
                convergenceMsg << "Position step norm is " << step_norm <<
                    " and the tolerance is " << particle_tolerance << "\n";
                convergenceMsg << " ux = " << ux[ip] << ", uy = " << uy[ip] << ", uz = " << uz[ip] << "\n";
                convergenceMsg << " xp = " << xp << ", yp = " << yp << ", zp = " << zp;
                ablastr::warn_manager::WMRecordWarning("ImplicitPushXP", convergenceMsg.str());
            }
#endif

#ifdef WARPX_QED
            // Reset the QED parameter to what is was at the start of the step
            if (p_optical_depth_QSR) {
                p_optical_depth_QSR[ip] = p_optical_depth_QSR0;
            }
#endif

            // write signaling flag: how many particles did not converge?
            amrex::Gpu::Atomic::Add(unconverged_particles_ptr, amrex::Long(1));
        }

    });

    // Setup for handling the unconverged particles. A list of their indices is
    // gathered, their weights saved, and their weight set to zero (so they
    // don't contribute to the current density).
    num_unconverged_particles = *(unconverged_particles.copyToHost());
    SetupSuborbitParticles(pti, offset, np_to_push, num_unconverged_particles,
                           unconverged_indices, saved_weights);

    if (num_unconverged_particles > 0) {
        ablastr::warn_manager::WMRecordWarning("ImplicitPushXP",
            "Picard solver for " +
            std::to_string(num_unconverged_particles) +
            " particles failed to converge after " +
            std::to_string(max_iterations) + " iterations."
         );
    }
}

/* \brief Perform the implicit particle push operation for unconverged particles
 *        in one fused kernel using suborbits.
 *        These are particles that failed to converge in ImplicitPushXP.
 *
 * \param[in] pti The WarpXParIter holding the particles to push
 * \param[in] fields The MultiFab register instance
 * \param[in] exfab, eyfab, ezfab The E fields
 * \param[in] bxfab, byfab, bzfab The B fields
 * \param[in] implicit_options Specifies options for implicit push
 * \param[in] ngEB The number of guard cells in the E and B fields
 * \param[in/out] jx, jy, jz The current densities to be deposited into
 * \param[in] index_offset Offset in the list of unconverged particles
 * \param[in] lev The refinement level
 * \param[in] gather_lev The refinement level at which to do the field gather
 * \param[in] dt The time step size
 * \param[in] skip_deposition Whether to do the deposition
 * \param[in] num_unconverged_particles Number of unconverged particles to push
 * \param[in] unconverged_indices The list of indices of unconverged particles
 * \param[in] saved_weights The saved weights of the unconverged particles
 */
void
PhysicalParticleContainer::ImplicitPushXPSubOrbits (WarpXParIter& pti,
                                                    ablastr::fields::MultiFabRegister& fields,
                                                    amrex::FArrayBox const * exfab,
                                                    amrex::FArrayBox const * eyfab,
                                                    amrex::FArrayBox const * ezfab,
                                                    amrex::FArrayBox const * bxfab,
                                                    amrex::FArrayBox const * byfab,
                                                    amrex::FArrayBox const * bzfab,
                                                    ImplicitOptions const * implicit_options,
                                                    amrex::IntVect ngEB,
                                                    amrex::MultiFab * const jx,
                                                    amrex::MultiFab * const jy,
                                                    amrex::MultiFab * const jz,
                                                    long index_offset,
                                                    int lev, int gather_lev,
                                                    amrex::Real dt,
                                                    bool skip_deposition,
                                                    long num_unconverged_particles,
                                                    amrex::Gpu::DeviceVector<long> & unconverged_indices,
                                                    amrex::Gpu::DeviceVector<amrex::ParticleReal> & saved_weights)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((gather_lev==(lev-1)) ||
                                     (gather_lev==(lev  )),
                                     "Gather buffers only work for lev-1");

    // If not doing suborbits, do not do anything
    if (!HasiAttrib("nsuborbits")) { return; }

    // If no particles, do not do anything
    if (num_unconverged_particles == 0) { return; }

    const auto depos_type = WarpX::current_deposition_algo;

    if (depos_type != CurrentDepositionAlgo::Villasenor ) {
        ablastr::warn_manager::WMRecordWarning("ImplicitPushXPSubOrbits",
            "When particle suborbits are used during the implicit particle push, only Villasenor "
            "current deposition is supported.");
    }

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

    const int depos_order = WarpX::nox;
    const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;
    const int max_crossings = WarpX::particle_max_grid_crossings;

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

    const bool deposit_mass_matrices = implicit_options->deposit_mass_matrices;
    amrex::MultiFab *Sxx, *Sxy, *Sxz, *Syx, *Syy, *Syz, *Szx, *Szy, *Szz;
    if (deposit_mass_matrices) {
        // Mass matrices deposit for suborbit particles is only for the preconditioner,
        // which currently only has the diagonal elements. The off-diagonal components
        // (i.e., Sxy and Szx) will not be used in the deposit below, but it is required
        // to pass them anyway. When the PC is more mature, the off-diagonal components
        // of the MM for the PC will have unique containers that will replace those used
        // here, which are used for the Jacobian calculation of non-suborbit particles.
        Sxx = fields.get(FieldType::MassMatrices_PC, Direction{0}, lev);
        Sxy = fields.get(FieldType::MassMatrices_X, Direction{1}, lev);
        Sxz = fields.get(FieldType::MassMatrices_X, Direction{2}, lev);
        Syx = fields.get(FieldType::MassMatrices_Y, Direction{0}, lev);
        Syy = fields.get(FieldType::MassMatrices_PC, Direction{1}, lev);
        Syz = fields.get(FieldType::MassMatrices_Y, Direction{2}, lev);
        Szx = fields.get(FieldType::MassMatrices_Z, Direction{0}, lev);
        Szy = fields.get(FieldType::MassMatrices_Z, Direction{1}, lev);
        Szz = fields.get(FieldType::MassMatrices_PC, Direction{2}, lev);
    }

    amrex::Array4<amrex::Real> const & Sxx_arr = (deposit_mass_matrices ? Sxx->array(pti) : amrex::Array4<amrex::Real>());
    amrex::Array4<amrex::Real> const & Sxy_arr = (deposit_mass_matrices ? Sxy->array(pti) : amrex::Array4<amrex::Real>());
    amrex::Array4<amrex::Real> const & Sxz_arr = (deposit_mass_matrices ? Sxz->array(pti) : amrex::Array4<amrex::Real>());
    amrex::Array4<amrex::Real> const & Syx_arr = (deposit_mass_matrices ? Syx->array(pti) : amrex::Array4<amrex::Real>());
    amrex::Array4<amrex::Real> const & Syy_arr = (deposit_mass_matrices ? Syy->array(pti) : amrex::Array4<amrex::Real>());
    amrex::Array4<amrex::Real> const & Syz_arr = (deposit_mass_matrices ? Syz->array(pti) : amrex::Array4<amrex::Real>());
    amrex::Array4<amrex::Real> const & Szx_arr = (deposit_mass_matrices ? Szx->array(pti) : amrex::Array4<amrex::Real>());
    amrex::Array4<amrex::Real> const & Szy_arr = (deposit_mass_matrices ? Szy->array(pti) : amrex::Array4<amrex::Real>());
    amrex::Array4<amrex::Real> const & Szz_arr = (deposit_mass_matrices ? Szz->array(pti) : amrex::Array4<amrex::Real>());

    // Create Gpu::Buffer for mass matrices to reduce kernel argument size
    amrex::Gpu::Buffer<amrex::Array4<amrex::Real>> Sbuf({Sxx_arr, Sxy_arr, Sxz_arr,
                                                         Syx_arr, Syy_arr, Syz_arr,
                                                         Szx_arr, Szy_arr, Szz_arr});
    auto const* pSbuf = Sbuf.data();

    auto& attribs = pti.GetAttribs();
    amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT w  = attribs[PIdx::w ].dataPtr();

    int *nsuborbits = pti.GetiAttribs("nsuborbits").dataPtr();

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

    // Loop over the particles and update their momentum
    const amrex::ParticleReal q = this->charge;
    const amrex::ParticleReal mass = this->m_mass;

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

    const auto do_gather = !do_not_gather;

    const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;
#ifdef WARPX_QED
    const int qed_runtime_flag = (local_has_quantum_sync || do_sync) ? has_qed : no_qed;
#else
    const int qed_runtime_flag = no_qed;
#endif
    const int depos_order_flag = depos_order;

    // The number of suborbits are not permitted to change during the linear stage of jfnk.
    // A buffer is given to the max_iterations to decrease the chance of non-convergence.
    const bool linear_stage_of_jfnk = implicit_options->linear_stage_of_jfnk;
    const int iter_buffer = linear_stage_of_jfnk ? 10 : 0;
    const int max_iterations = implicit_options->max_particle_iterations + iter_buffer;
    const amrex::ParticleReal particle_tolerance = implicit_options->particle_tolerance;

    long * unconverged_i = unconverged_indices.data() + index_offset;
    amrex::ParticleReal * saved_w = saved_weights.data() + index_offset;

    // Using this version of ParallelFor with compile time options
    // improves performance when qed or external EB are not used by reducing
    // register pressure.
    amrex::ParallelFor(amrex::TypeList<amrex::CompileTimeOptions<no_exteb,has_exteb>,
                                       amrex::CompileTimeOptions<no_qed  ,has_qed>,
                                       amrex::CompileTimeOptions<order_one, order_two, order_three, order_four >>{},
                       {exteb_runtime_flag, qed_runtime_flag, depos_order_flag},
                       num_unconverged_particles, [=] AMREX_GPU_DEVICE (long i,
                                                                 auto exteb_control, auto qed_control, auto depos_order_control)
    {

        long ip = unconverged_i[i];

        // Restore the particle weight
        w[ip] = saved_w[i];

        amrex::Real wq = q*w[ip];
        if (ion_lev) {
            wq *= ion_lev[ip];
        }

        // The _n0 variables save the position and velocity at the start
        // of the full time step.
        // The _n variables save the position and velocity at the start
        // of each sub step.

#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal const xp_n0 = x_n[ip];
#else
        amrex::ParticleReal const xp_n0 = 0.0_prt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ParticleReal const yp_n0 = y_n[ip];
#else
        amrex::ParticleReal const yp_n0 = 0.0_prt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
        amrex::ParticleReal const zp_n0 = z_n[ip];
#else
        amrex::ParticleReal const zp_n0 = 0.0_prt;
#endif

#if WARPX_QED
        amrex::ParticleReal p_optical_depth_QSR0 = 0.0_prt;
        if (p_optical_depth_QSR) {
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

        // For nonlinear stage, we first loop over all suborbits doing the push only to
        // check for non-convergence. If a suborbit does not convegence, then the number
        // of suborbits is increased and the loop starts over. This proceeds until all
        // suborbits converge. For the linear stage, the number of suborbits are not
        // permitted to change because this can cause non-convergence of GMRES.
        bool doing_deposition = linear_stage_of_jfnk ? true : false;
        int isuborbit = 0;
        while (isuborbit < nsuborbits[ip]) {

            amrex::Real const dt_suborbit = dt/nsuborbits[ip];

            amrex::ParticleReal Bxp = 0.0_prt;
            amrex::ParticleReal Byp = 0.0_prt;
            amrex::ParticleReal Bzp = 0.0_prt;
            amrex::ParticleReal step_norm = 1._prt;

            // Try advancing the particle one suborbit step
            bool convergence = PushXPSingleStep<exteb_control, qed_control>(ip, dt_suborbit, setPosition,
                                 xp, yp, zp, ux, uy, uz, xp_n, yp_n, zp_n, uxp_n, uyp_n, uzp_n,
                                 step_norm, particle_tolerance, max_iterations,
                                 Ex_external_particle, Ey_external_particle, Ez_external_particle,
                                 Bx_external_particle, By_external_particle, Bz_external_particle,
                                 Bxp, Byp, Bzp,
                                 do_gather, ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                 ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                 dinv, xyzmin, lo, n_rz_azimuthal_modes, depos_order, depos_type,
                                 getExternalEB, ion_lev, mass, q, pusher_algo, do_crr
#ifdef WARPX_QED
                                 , do_sync, t_chi_max, p_optical_depth_QSR, evolve_opt
#endif
                                 );

            // Don't change number of suborbits during linear stage of jfnk
            if (linear_stage_of_jfnk) { convergence = true; }

            if (doing_deposition) {

                const amrex::ParticleReal xp_np1 = 2.0_prt*xp - xp_n;
                const amrex::ParticleReal yp_np1 = 2.0_prt*yp - yp_n;
                const amrex::ParticleReal zp_np1 = 2.0_prt*zp - zp_n;

                const amrex::ParticleReal gaminv = GetImplicitGammaInverse(uxp_n, uyp_n, uzp_n, ux[ip], uy[ip], uz[ip]);

                if (deposit_mass_matrices) {
                    const amrex::Real wq_invvol = wq*invvol/nsuborbits[ip];
                    const amrex::Real rhop = 2.0_rt*wq_invvol*gaminv; // approximation when neglecting MM coupling terms

                    // Set the Mass Matrices kernels
                    amrex::ParticleReal fpxx, fpxy, fpxz;
                    amrex::ParticleReal fpyx, fpyy, fpyz;
                    amrex::ParticleReal fpzx, fpzy, fpzz;
                    setMassMatricesKernels(q, mass, dt_suborbit, rhop,
                                           ux[ip], uy[ip], uz[ip],
                                           Bxp, Byp, Bzp,
                                           fpxx, fpxy, fpxz,
                                           fpyx, fpyy, fpyz,
                                           fpzx, fpzy, fpzz);

                    // The ignore_unused is needed so that the variables are not first-captured
                    // in a constexpr-if context.
                    amrex::ignore_unused(max_crossings);
                    amrex::ignore_unused(Jx_arr, Jy_arr, Jz_arr, invvol);
                    amrex::ignore_unused(pSbuf);
                    if constexpr (depos_order_control == order_one) {
                        doVillasenorJandSigmaDepositionKernel<1,false>(
                                                              xp_n, yp_n, zp_n, xp_np1, yp_np1, zp_np1,
                                                              wq_invvol, ux[ip], uy[ip], uz[ip], gaminv,
                                                              fpxx, fpxy, fpxz,
                                                              fpyx, fpyy, fpyz,
                                                              fpzx, fpzy, fpzz,
                                                              Jx_arr, Jy_arr, Jz_arr,
                                                              max_crossings,
                                                              pSbuf[0], pSbuf[1], pSbuf[2],
                                                              pSbuf[3], pSbuf[4], pSbuf[5],
                                                              pSbuf[6], pSbuf[7], pSbuf[8],
                                                              dt_suborbit, dinv, xyzmin, lo );
                    } else if constexpr (depos_order_control == order_two) {
                        doVillasenorJandSigmaDepositionKernel<2,false>(
                                                              xp_n, yp_n, zp_n, xp_np1, yp_np1, zp_np1,
                                                              wq_invvol, ux[ip], uy[ip], uz[ip], gaminv,
                                                              fpxx, fpxy, fpxz,
                                                              fpyx, fpyy, fpyz,
                                                              fpzx, fpzy, fpzz,
                                                              Jx_arr, Jy_arr, Jz_arr,
                                                              max_crossings,
                                                              pSbuf[0], pSbuf[1], pSbuf[2],
                                                              pSbuf[3], pSbuf[4], pSbuf[5],
                                                              pSbuf[6], pSbuf[7], pSbuf[8],
                                                              dt_suborbit, dinv, xyzmin, lo );
                    } else if constexpr (depos_order_control == order_three) {
                        doVillasenorJandSigmaDepositionKernel<3,false>(
                                                              xp_n, yp_n, zp_n, xp_np1, yp_np1, zp_np1,
                                                              wq_invvol, ux[ip], uy[ip], uz[ip], gaminv,
                                                              fpxx, fpxy, fpxz,
                                                              fpyx, fpyy, fpyz,
                                                              fpzx, fpzy, fpzz,
                                                              Jx_arr, Jy_arr, Jz_arr,
                                                              max_crossings,
                                                              pSbuf[0], pSbuf[1], pSbuf[2],
                                                              pSbuf[3], pSbuf[4], pSbuf[5],
                                                              pSbuf[6], pSbuf[7], pSbuf[8],
                                                              dt_suborbit, dinv, xyzmin, lo );
                    } else if constexpr (depos_order_control == order_four) {
                        doVillasenorJandSigmaDepositionKernel<4,false>(
                                                              xp_n, yp_n, zp_n, xp_np1, yp_np1, zp_np1,
                                                              wq_invvol, ux[ip], uy[ip], uz[ip], gaminv,
                                                              fpxx, fpxy, fpxz,
                                                              fpyx, fpyy, fpyz,
                                                              fpzx, fpzy, fpzz,
                                                              Jx_arr, Jy_arr, Jz_arr,
                                                              max_crossings,
                                                              pSbuf[0], pSbuf[1], pSbuf[2],
                                                              pSbuf[3], pSbuf[4], pSbuf[5],
                                                              pSbuf[6], pSbuf[7], pSbuf[8],
                                                              dt_suborbit, dinv, xyzmin, lo );
                    }

                } else {

                    amrex::ParticleReal wq_n = wq/nsuborbits[ip];

                    // Only CurrentDepositionAlgo::Villasenor is supported
                    // The ignore_unused is needed so that the variables are not first-captured
                    // in a constexpr-if context.
                    amrex::ignore_unused(Jx_arr, Jy_arr, Jz_arr, invvol);
                    if constexpr (depos_order_control == order_one) {
                        VillasenorDepositionShapeNKernel<1>(xp_n, yp_n, zp_n, xp_np1, yp_np1, zp_np1, wq_n,
                                                            ux[ip], uy[ip], uz[ip], gaminv,
                                                            Jx_arr, Jy_arr, Jz_arr,
                                                            dt_suborbit, dinv, xyzmin, lo, invvol, n_rz_azimuthal_modes);
                    }
                    else if constexpr (depos_order_control == order_two) {
                        VillasenorDepositionShapeNKernel<2>(xp_n, yp_n, zp_n, xp_np1, yp_np1, zp_np1, wq_n,
                                                            ux[ip], uy[ip], uz[ip], gaminv,
                                                            Jx_arr, Jy_arr, Jz_arr,
                                                            dt_suborbit, dinv, xyzmin, lo, invvol, n_rz_azimuthal_modes);
                    }
                    else if constexpr (depos_order_control == order_three) {
                        VillasenorDepositionShapeNKernel<3>(xp_n, yp_n, zp_n, xp_np1, yp_np1, zp_np1, wq_n,
                                                            ux[ip], uy[ip], uz[ip], gaminv,
                                                            Jx_arr, Jy_arr, Jz_arr,
                                                            dt_suborbit, dinv, xyzmin, lo, invvol, n_rz_azimuthal_modes);
                    }
                    else if constexpr (depos_order_control == order_four) {
                        VillasenorDepositionShapeNKernel<4>(xp_n, yp_n, zp_n, xp_np1, yp_np1, zp_np1, wq_n,
                                                            ux[ip], uy[ip], uz[ip], gaminv,
                                                            Jx_arr, Jy_arr, Jz_arr,
                                                            dt_suborbit, dinv, xyzmin, lo, invvol, n_rz_azimuthal_modes);
                    }
                }
            }

            isuborbit++;

            if (!convergence || (isuborbit == nsuborbits[ip] && !doing_deposition)) {

                if (!convergence) {
                    // particle did not converge
                    // Increase the number of suborbits and start over
                    nsuborbits[ip]++;
                } else if (skip_deposition) {
                    break;
                } else {
                    // Convergence was reached for all suborbits, now redo the loop
                    // and do the deposition
                    doing_deposition = true;
                }

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
                if (p_optical_depth_QSR) {
                    p_optical_depth_QSR[ip] = p_optical_depth_QSR0;
                }
#endif

            } else {

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

    });

    amrex::Gpu::streamSynchronize();
}
