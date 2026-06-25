/* Copyright 2026 The WarpX Community
 * Authors: Prabhat Kumar, Eric Clark (Helion Energy)
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

    // Drag-only operator: relax the ion bulk velocity toward the electron
    // fluid V_e at the eta-derived rate
    //     nu_{s,e} = Z_s e^2 eta_s_eff n_e / m_s
    // via a uniform per-cell shift, applied identically to every particle so
    // the thermal spread (T_i) is preserved:
    //     v_p -= (V_s - V_e)(1 - exp(-nu dt)).
    // No heat is deposited here; the eta J^2 source on T_e is the separate
    // gridded HybridPICModel::QDSMCAddJouleHeating call. Registering this
    // operator gives ions the back-reaction to the eta*J term in Ohm's law
    // (cancels the anti-friction), independently of include_joule_heating.

    auto & warpx = WarpX::GetInstance();
    auto & species = mypc->GetParticleContainerFromName(m_species_names[0]);

    auto const * hybrid_model = warpx.get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(hybrid_model != nullptr,
        "HybridResistiveDrag requires the hybrid-PIC solver to be active.");

    auto const eta_func = hybrid_model->m_eta;
    auto const t_now    = warpx.gett_new(0);

    // Per-species resistivity overlay (Phys. Plasmas 31, 012902 (2024), Eq. 10):
    // if registered, added to the global eta in the kernel so the drag uses
    //   eta_s_eff = eta_global + eta_s_per
    // (the same effective resistivity Ohm's law and QDSMCAddJouleHeating use).
    auto const eta_per_it     = hybrid_model->m_eta_per_species.find(m_species_names[0]);
    bool const has_eta_per    = (eta_per_it != hybrid_model->m_eta_per_species.end());
    amrex::ParserExecutor<7> eta_s_per{};
    if (has_eta_per) { eta_s_per = eta_per_it->second; }

    amrex::Real const species_mass = species.getMass();
    amrex::Real const Z_s          = species.getCharge() / PhysConst::q_e;
    amrex::Real const Z_e2_over_ms = Z_s * PhysConst::q_e * PhysConst::q_e / species_mass;
    amrex::Real const inv_qe       = 1.0_rt / PhysConst::q_e;

    // Density floor (matches the fluid-velocity and Joule-heat paths): skip
    // cells below the floor, where eta ~ 1/sqrt(rho) blows up and the gridded
    // V_s/V_e are noise rather than physical bulk velocities.
    amrex::Real const rho_floor = PhysConst::q_e * hybrid_model->m_n_floor;

    int const nox = WarpX::nox;
    int const n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;
    bool const galerkin_interpolation = WarpX::galerkin_interpolation;

    for (int lev = 0; lev <= species.finestLevel(); ++lev) {
        ablastr::fields::VectorField Ve_fp  = warpx.m_fields.get_alldirs("Ve_fp", lev);
        ablastr::fields::VectorField Vs_fp  = warpx.m_fields.get_alldirs("Vs_fp_" + m_species_names[0], lev);
        ablastr::fields::VectorField B_fp   = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        ablastr::fields::VectorField J_fp   = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
        // Per-species J_s and rho_s, plus T_e (nodal). Only consulted inside
        // the kernel when has_eta_per is true; looked up here unconditionally
        // so the kernel can capture their arrays cheaply (the lookup is a
        // const-time field-registry query).
        ablastr::fields::VectorField Js_fp  = warpx.m_fields.get_alldirs("current_fp_" + m_species_names[0], lev);
        amrex::MultiFab const & rho_fp      = *warpx.m_fields.get(FieldType::rho_fp, lev);
        amrex::MultiFab const & rhos_fp     = *warpx.m_fields.get("rho_fp_" + m_species_names[0], lev);
        amrex::MultiFab const & Te_fp       = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);

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

            auto const Vex_arr  = Ve_fp[0]->const_array(pti);
            auto const Vey_arr  = Ve_fp[1]->const_array(pti);
            auto const Vez_arr  = Ve_fp[2]->const_array(pti);
            auto const Vsx_arr  = Vs_fp[0]->const_array(pti);
            auto const Vsy_arr  = Vs_fp[1]->const_array(pti);
            auto const Vsz_arr  = Vs_fp[2]->const_array(pti);
            auto const Bx_arr   = B_fp[0]->const_array(pti);
            auto const By_arr   = B_fp[1]->const_array(pti);
            auto const Bz_arr   = B_fp[2]->const_array(pti);
            auto const Jx_arr   = J_fp[0]->const_array(pti);
            auto const Jy_arr   = J_fp[1]->const_array(pti);
            auto const Jz_arr   = J_fp[2]->const_array(pti);
            auto const Jsx_arr  = Js_fp[0]->const_array(pti);
            auto const Jsy_arr  = Js_fp[1]->const_array(pti);
            auto const Jsz_arr  = Js_fp[2]->const_array(pti);
            auto const rho_arr  = rho_fp.const_array(pti);
            auto const rhos_arr = rhos_fp.const_array(pti);
            auto const Te_arr   = Te_fp.const_array(pti);

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
            amrex::IndexType const Jsx_type = Js_fp[0]->ixType();
            amrex::IndexType const Jsy_type = Js_fp[1]->ixType();
            amrex::IndexType const Jsz_type = Js_fp[2]->ixType();

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
                amrex::Real eta_s_eff = eta_func(rho_val, Jmag, t_now);

                // Per-species overlay: add eta_s_per(rho_s, rho, Te, |J|,
                // |J_s|, |B|, t) to eta_s_eff when registered for this
                // species. When not registered, eta_s_eff stays at the
                // global eta and the drag rate matches the pre-Phase-4
                // path exactly.
                if (has_eta_per) {
                    amrex::Real const rhos_val = ablastr::particles::doGatherScalarFieldNodal(
                        xp, yp, zp, rhos_arr, dxi, plo);
                    amrex::Real const Te_val   = ablastr::particles::doGatherScalarFieldNodal(
                        xp, yp, zp, Te_arr, dxi, plo);
                    amrex::ParticleReal Jsxp = 0._prt, Jsyp = 0._prt, Jszp = 0._prt;
                    amrex::ParticleReal _dx = 0._prt, _dy = 0._prt, _dz = 0._prt;
                    doGatherShapeN(xp, yp, zp, Jsxp, Jsyp, Jszp, _dx, _dy, _dz,
                                   Jsx_arr, Jsy_arr, Jsz_arr, Bx_arr, By_arr, Bz_arr,
                                   Jsx_type, Jsy_type, Jsz_type, Bx_type, By_type, Bz_type,
                                   dinv, xyzmin, lo, n_rz_azimuthal_modes,
                                   nox, galerkin_interpolation);
                    amrex::Real const Jsmag = std::sqrt(Jsxp*Jsxp + Jsyp*Jsyp + Jszp*Jszp);
                    amrex::Real const Bmag  = std::sqrt(Bxp*Bxp + Byp*Byp + Bzp*Bzp);
                    eta_s_eff += eta_s_per(rhos_val, rho_val, Te_val,
                                           Jmag, Jsmag, Bmag, t_now);
                }

                // ν_{s,e} = Z_s * e^2 * eta_s_eff * n_e / m_s
                amrex::Real const n_e = rho_val * inv_qe;
                amrex::Real const nu  = Z_e2_over_ms * eta_s_eff * n_e;

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
