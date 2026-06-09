/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HybridResistiveDrag.H"

#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/particles/NodalFieldGather.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <cmath>
#include <string>

HybridResistiveDrag::HybridResistiveDrag (std::string const& collision_name)
    : CollisionBase(collision_name)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_species_names.size() == 1,
        "hybrid_resistive_drag must have exactly one species.");
}

void
HybridResistiveDrag::doCollisions (amrex::Real /*cur_time*/, amrex::Real dt, MultiParticleContainer* mypc)
{
    ABLASTR_PROFILE("HybridResistiveDrag::doCollisions()");
    using namespace amrex::literals;
    using warpx::fields::FieldType;

    // Drag-only operator: shifts each ion macroparticle toward the local
    // electron fluid velocity V_e at the eta-derived rate
    //     nu_{s,e} = Z_s * e^2 * eta(rho, |J|, t) * n_e / m_s
    //
    // Update:
    //     v_p <- V_e + (v_p - V_e) * exp(-nu * dt)
    //
    // i.e. a uniform per-cell bulk shift (V_s - V_e)(1 - exp(-nu dt))
    // applied identically to every particle in the cell -- preserves the
    // thermal moment around V_s. The corresponding W_dot dissipation into
    // T_e is NOT computed here; the Joule-heating source on T_e is the
    // separate, gridded HybridPICModel::QDSMCAddJouleHeating call, which
    // evaluates the same Σ_s n_s m_s ν |V_s - V_e|^2 expression per cell
    // from the existing fields. Decoupling lets the user pick:
    //   * register this drag operator -> ions feel the back-reaction to
    //                                    qE_eta in Ohm's law
    //                                    (anti-friction cancelled)
    //   * don't register it            -> ions feel only qE
    //                                    (Topanga convention)
    // independently of include_joule_heating.

    auto & warpx = WarpX::GetInstance();
    auto & species = mypc->GetParticleContainerFromName(m_species_names[0]);

    auto const * hybrid_model = warpx.get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(hybrid_model != nullptr,
        "HybridResistiveDrag requires the hybrid-PIC solver to be active.");

    auto const eta_func = hybrid_model->m_eta;
    auto const t_now    = warpx.gett_new(0);

    amrex::Real const species_mass = species.getMass();
    amrex::Real const Z_s          = species.getCharge() / PhysConst::q_e;
    amrex::Real const Z_e2_over_ms = Z_s * PhysConst::q_e * PhysConst::q_e / species_mass;
    amrex::Real const inv_qe       = 1.0_rt / PhysConst::q_e;

    // Match the density floor used by Calculate{Ion,Electron}FluidVelocity
    // and QDSMCAddJouleHeating[Kinetic]: skip particles in cells where rho
    // is below the floor. In low-density cells eta ~ 1/sqrt(rho) blows up
    // and the gridded V_s/V_e use a floored rho, so their gathered values
    // can be noise rather than physical bulk velocities. Skipping keeps
    // the drag honest at the same locations where the heat source is
    // already gated off.
    amrex::Real const rho_floor = PhysConst::q_e * hybrid_model->m_n_floor;

    int const nox = WarpX::nox;
    int const n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;
    bool const galerkin_interpolation = WarpX::galerkin_interpolation;

    for (int lev = 0; lev <= species.finestLevel(); ++lev) {
        ablastr::fields::VectorField Ve_fp = warpx.m_fields.get_alldirs("Ve_fp", lev);
        ablastr::fields::VectorField Vs_fp = warpx.m_fields.get_alldirs("Vs_fp_" + m_species_names[0], lev);
        ablastr::fields::VectorField B_fp  = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        ablastr::fields::VectorField J_fp  = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
        amrex::MultiFab const & rho_fp     = *warpx.m_fields.get(FieldType::rho_fp, lev);

        amrex::XDim3 const dinv = WarpX::InvCellSize(lev);
        auto const dxi = warpx.Geom(lev).InvCellSizeArray();
        auto const plo = warpx.Geom(lev).ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(species, lev); pti.isValid(); ++pti)
        {
            amrex::Box const & box = pti.validbox();
            amrex::XDim3 const xyzmin = WarpX::LowerCorner(box, lev, 0._rt);
            amrex::Dim3 const lo = amrex::lbound(box);

            auto const Vex_arr = Ve_fp[0]->const_array(pti);
            auto const Vey_arr = Ve_fp[1]->const_array(pti);
            auto const Vez_arr = Ve_fp[2]->const_array(pti);
            auto const Vsx_arr = Vs_fp[0]->const_array(pti);
            auto const Vsy_arr = Vs_fp[1]->const_array(pti);
            auto const Vsz_arr = Vs_fp[2]->const_array(pti);
            auto const Bx_arr  = B_fp[0]->const_array(pti);
            auto const By_arr  = B_fp[1]->const_array(pti);
            auto const Bz_arr  = B_fp[2]->const_array(pti);
            auto const Jx_arr  = J_fp[0]->const_array(pti);
            auto const Jy_arr  = J_fp[1]->const_array(pti);
            auto const Jz_arr  = J_fp[2]->const_array(pti);
            auto const rho_arr = rho_fp.const_array(pti);

            amrex::IndexType const Vex_type = Ve_fp[0]->ixType();
            amrex::IndexType const Vey_type = Ve_fp[1]->ixType();
            amrex::IndexType const Vez_type = Ve_fp[2]->ixType();
            amrex::IndexType const Vsx_type = Vs_fp[0]->ixType();
            amrex::IndexType const Vsy_type = Vs_fp[1]->ixType();
            amrex::IndexType const Vsz_type = Vs_fp[2]->ixType();
            amrex::IndexType const Bx_type  = B_fp[0]->ixType();
            amrex::IndexType const By_type  = B_fp[1]->ixType();
            amrex::IndexType const Bz_type  = B_fp[2]->ixType();
            amrex::IndexType const Jx_type  = J_fp[0]->ixType();
            amrex::IndexType const Jy_type  = J_fp[1]->ixType();
            amrex::IndexType const Jz_type  = J_fp[2]->ixType();

            auto& attribs = pti.GetAttribs();
            amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
            amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
            amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

            auto const getPosition = GetParticlePosition<PIdx>(pti);
            long const np = pti.numParticles();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
            {
                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                amrex::ParticleReal Vex = 0._prt, Vey = 0._prt, Vez = 0._prt;
                amrex::ParticleReal Bxp = 0._prt, Byp = 0._prt, Bzp = 0._prt;
                amrex::ParticleReal Vsx = 0._prt, Vsy = 0._prt, Vsz = 0._prt;
                amrex::ParticleReal Jxp = 0._prt, Jyp = 0._prt, Jzp = 0._prt;

                // Gather V_e (E-slot) and B (B-slot).
                doGatherShapeN(xp, yp, zp, Vex, Vey, Vez, Bxp, Byp, Bzp,
                               Vex_arr, Vey_arr, Vez_arr, Bx_arr, By_arr, Bz_arr,
                               Vex_type, Vey_type, Vez_type, Bx_type, By_type, Bz_type,
                               dinv, xyzmin, lo, n_rz_azimuthal_modes,
                               nox, galerkin_interpolation);

                // Gather V_s (E-slot) and J_plasma (B-slot). J is used only
                // to evaluate eta(rho, |J|, t) below; it is not needed for
                // the drag shift itself.
                doGatherShapeN(xp, yp, zp, Vsx, Vsy, Vsz, Jxp, Jyp, Jzp,
                               Vsx_arr, Vsy_arr, Vsz_arr, Jx_arr, Jy_arr, Jz_arr,
                               Vsx_type, Vsy_type, Vsz_type, Jx_type, Jy_type, Jz_type,
                               dinv, xyzmin, lo, n_rz_azimuthal_modes,
                               nox, galerkin_interpolation);

                amrex::Real const rho_val = ablastr::particles::doGatherScalarFieldNodal(
                    xp, yp, zp, rho_arr, dxi, plo);

                if (rho_val <= rho_floor) { return; }

                amrex::Real const Jmag = std::sqrt(Jxp*Jxp + Jyp*Jyp + Jzp*Jzp);
                amrex::Real const eta_val = eta_func(rho_val, Jmag, t_now);

                // ν_{s,e} = Z_s * e^2 * eta * n_e / m_s
                amrex::Real const n_e = rho_val * inv_qe;
                amrex::Real const nu  = Z_e2_over_ms * eta_val * n_e;

                amrex::Real const one_minus_fac = -std::expm1(-nu * dt);

                // Uniform per-cell shift toward V_e; preserves thermal moment.
                amrex::ParticleReal const dVx = Vsx - Vex;
                amrex::ParticleReal const dVy = Vsy - Vey;
                amrex::ParticleReal const dVz = Vsz - Vez;
                ux[ip] -= dVx * one_minus_fac;
                uy[ip] -= dVy * one_minus_fac;
                uz[ip] -= dVz * one_minus_fac;
            });
        }
    }
}
