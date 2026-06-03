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
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/particles/NodalFieldGather.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_GpuAtomic.H>
#include <AMReX_Math.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <string>

HybridResistiveDrag::HybridResistiveDrag (std::string const& collision_name)
    : CollisionBase(collision_name)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_species_names.size() == 1,
        "hybrid_resistive_drag must have exactly one species.");

    const amrex::ParmParse pp_collision_name(collision_name);

    std::string expression_type_str = "collision_frequency";
    pp_collision_name.query("expression_type", expression_type_str);
    if (expression_type_str == "collision_frequency") {
        m_expression_type = HybridResistiveDragExpressionType::CollisionFrequency;
    } else if (expression_type_str == "resistivity") {
        m_expression_type = HybridResistiveDragExpressionType::Resistivity;
    } else {
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_resistive_drag: expression_type must be 'collision_frequency' or 'resistivity'.");
    }

    std::string drag_function_str;
    pp_collision_name.get("drag_function(rho,Vs,Ve,Ts,Te,B)", drag_function_str);
    m_drag_parser = utils::parser::makeParser(
        drag_function_str, {"rho", "Vs", "Ve", "Ts", "Te", "B"});
    m_drag_func = m_drag_parser.compile<6>();
}

void
HybridResistiveDrag::doCollisions (amrex::Real /*cur_time*/, amrex::Real dt, MultiParticleContainer* mypc)
{
    ABLASTR_PROFILE("HybridResistiveDrag::doCollisions()");
    using namespace amrex::literals;
    using warpx::fields::FieldType;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_expression_type == HybridResistiveDragExpressionType::CollisionFrequency,
        "hybrid_resistive_drag: expression_type = 'resistivity' is not yet implemented; "
        "use 'collision_frequency'.");

    auto & warpx = WarpX::GetInstance();
    auto & species = mypc->GetParticleContainerFromName(m_species_names[0]);

    const auto drag_func = m_drag_func;
    auto const * hybrid_model = warpx.get_pointer_HybridPICModel();
    const amrex::Real electron_temp = hybrid_model->m_elec_temp;
    const bool include_resistive_heating = hybrid_model->m_include_resistive_heating;
    // Species mass per macroparticle is needed to convert the per-particle
    // friction work into a per-cell W_dot contribution to the electron-pressure
    // heating equation.
    const amrex::Real species_mass = species.getMass();
    const int nox = WarpX::nox;
    const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;
    const bool galerkin_interpolation = WarpX::galerkin_interpolation;

    // If the species has do_temperature_deposition = 1 in the input, WarpX
    // accumulates T_<species> as a 3-component face-staggered field via
    // DepositTemperatures. We gather it at the particle position and pass the
    // scalar Ts = (Txx + Tyy + Tzz)/3 to the parser. Otherwise pass Ts = 0
    // (which is fine for Spitzer, since Ts contributes at order m_e/m_s).
    const bool have_Ts = species.getTemperatureDepositionFlag();

    for (int lev = 0; lev <= species.finestLevel(); ++lev) {
        ablastr::fields::VectorField Ve_fp = warpx.m_fields.get_alldirs("Ve_fp", lev);
        ablastr::fields::VectorField Vs_fp = warpx.m_fields.get_alldirs("Vs_fp_" + m_species_names[0], lev);
        ablastr::fields::VectorField B_fp  = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        amrex::MultiFab const & rho_fp    = *warpx.m_fields.get(FieldType::rho_fp, lev);
        amrex::MultiFab       & heating_fp = *warpx.m_fields.get(FieldType::hybrid_drag_heating_fp, lev);

        // T_<species> is allocated only if do_temperature_deposition is on; if
        // not, alias Ts arrays to B (any const Real Array4 will do) and gate
        // their use behind have_Ts in the kernel.
        ablastr::fields::VectorField Ts_fp =
            have_Ts ? warpx.m_fields.get_alldirs("T_" + m_species_names[0], lev) : B_fp;

        const amrex::XDim3 dinv = WarpX::InvCellSize(lev);
        const auto dxi = warpx.Geom(lev).InvCellSizeArray();
        const auto plo = warpx.Geom(lev).ProbLoArray();

        // Cell volume (per unit out-of-plane length in lower dimensions).
        amrex::Real cell_volume = 1.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { cell_volume /= dxi[d]; }
        const amrex::Real inv_cell_volume = 1.0_rt / cell_volume;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(species, lev); pti.isValid(); ++pti)
        {
            const amrex::Box& box = pti.validbox();
            const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);
            const amrex::Dim3 lo = amrex::lbound(box);

            const auto Vex_arr = Ve_fp[0]->const_array(pti);
            const auto Vey_arr = Ve_fp[1]->const_array(pti);
            const auto Vez_arr = Ve_fp[2]->const_array(pti);
            const auto Vsx_arr = Vs_fp[0]->const_array(pti);
            const auto Vsy_arr = Vs_fp[1]->const_array(pti);
            const auto Vsz_arr = Vs_fp[2]->const_array(pti);
            const auto Bx_arr  = B_fp[0]->const_array(pti);
            const auto By_arr  = B_fp[1]->const_array(pti);
            const auto Bz_arr  = B_fp[2]->const_array(pti);
            const auto rho_arr = rho_fp.const_array(pti);
            // T_<species> is a face-staggered 3-comp field (T_xx, T_yy, T_zz);
            // gather all three and average to get a scalar isotropic Ts.
            const auto Tsx_arr = Ts_fp[0]->const_array(pti);
            const auto Tsy_arr = Ts_fp[1]->const_array(pti);
            const auto Tsz_arr = Ts_fp[2]->const_array(pti);
            const amrex::IndexType Tsx_type = Ts_fp[0]->ixType();
            const amrex::IndexType Tsy_type = Ts_fp[1]->ixType();
            const amrex::IndexType Tsz_type = Ts_fp[2]->ixType();

            const amrex::IndexType Vex_type = Ve_fp[0]->ixType();
            const amrex::IndexType Vey_type = Ve_fp[1]->ixType();
            const amrex::IndexType Vez_type = Ve_fp[2]->ixType();
            const amrex::IndexType Vsx_type = Vs_fp[0]->ixType();
            const amrex::IndexType Vsy_type = Vs_fp[1]->ixType();
            const amrex::IndexType Vsz_type = Vs_fp[2]->ixType();
            const amrex::IndexType Bx_type  = B_fp[0]->ixType();
            const amrex::IndexType By_type  = B_fp[1]->ixType();
            const amrex::IndexType Bz_type  = B_fp[2]->ixType();

            auto& attribs = pti.GetAttribs();
            amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
            amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
            amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();
            // Macro-particle weight (number of physical particles per macro);
            // used to weight the per-particle heating contribution into the
            // hybrid_drag_heating_fp accumulator.
            amrex::ParticleReal const * const AMREX_RESTRICT wp_arr =
                attribs[PIdx::w].dataPtr();

            // Writable heating accumulator (NGP atomic deposit into the cell
            // containing the particle). Only written when include_resistive_heating is on.
            amrex::Array4<amrex::Real> const heating_arr = heating_fp.array(pti);

            const auto getPosition = GetParticlePosition<PIdx>(pti);
            const long np = pti.numParticles();

            // Dummy outputs for the second gather (E-slot reused for Vs).
            amrex::Array4<amrex::Real const> const dummy_arr = Bx_arr;

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
            {
                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                amrex::ParticleReal Vex = 0._prt, Vey = 0._prt, Vez = 0._prt;
                amrex::ParticleReal Bxp = 0._prt, Byp = 0._prt, Bzp = 0._prt;
                amrex::ParticleReal Vsx = 0._prt, Vsy = 0._prt, Vsz = 0._prt;
                amrex::ParticleReal _dx = 0._prt, _dy = 0._prt, _dz = 0._prt;

                // Gather Ve (E-slot) and B (B-slot).
                doGatherShapeN(xp, yp, zp, Vex, Vey, Vez, Bxp, Byp, Bzp,
                               Vex_arr, Vey_arr, Vez_arr, Bx_arr, By_arr, Bz_arr,
                               Vex_type, Vey_type, Vez_type, Bx_type, By_type, Bz_type,
                               dinv, xyzmin, lo, n_rz_azimuthal_modes,
                               nox, galerkin_interpolation);

                // Gather Vs in the E-slot; B-slot is wired to dummies and ignored.
                doGatherShapeN(xp, yp, zp, Vsx, Vsy, Vsz, _dx, _dy, _dz,
                               Vsx_arr, Vsy_arr, Vsz_arr, dummy_arr, dummy_arr, dummy_arr,
                               Vsx_type, Vsy_type, Vsz_type, Bx_type, By_type, Bz_type,
                               dinv, xyzmin, lo, n_rz_azimuthal_modes,
                               nox, galerkin_interpolation);

                // Gather T_<species> components if available (3-comp vector field
                // T_xx, T_yy, T_zz at face staggering). Scalar Ts = trace/3 is
                // passed to the parser. If do_temperature_deposition is off for
                // this species, Ts = 0 (Spitzer is insensitive to Ts at order m_e/m_s).
                amrex::Real Ts_scalar = 0._rt;
                if (have_Ts) {
                    amrex::ParticleReal Txx = 0._prt, Tyy = 0._prt, Tzz = 0._prt;
                    amrex::ParticleReal _e1 = 0._prt, _e2 = 0._prt, _e3 = 0._prt;
                    doGatherShapeN(xp, yp, zp, Txx, Tyy, Tzz, _e1, _e2, _e3,
                                   Tsx_arr, Tsy_arr, Tsz_arr,
                                   dummy_arr, dummy_arr, dummy_arr,
                                   Tsx_type, Tsy_type, Tsz_type,
                                   Bx_type, By_type, Bz_type,
                                   dinv, xyzmin, lo, n_rz_azimuthal_modes,
                                   nox, galerkin_interpolation);
                    Ts_scalar = (Txx + Tyy + Tzz) / 3._rt;
                }

                const amrex::Real rho_val = ablastr::particles::doGatherScalarFieldNodal(
                    xp, yp, zp, rho_arr, dxi, plo);

                const amrex::Real Vsmag = std::sqrt(Vsx*Vsx + Vsy*Vsy + Vsz*Vsz);
                const amrex::Real Vemag = std::sqrt(Vex*Vex + Vey*Vey + Vez*Vez);
                const amrex::Real Bmag  = std::sqrt(Bxp*Bxp + Byp*Byp + Bzp*Bzp);

                const amrex::Real nu = drag_func(rho_val, Vsmag, Vemag, Ts_scalar, electron_temp, Bmag);
                const amrex::Real one_minus_fac = -std::expm1(-nu * dt);

                // Uniform per-cell shift: applies the bulk-fluid drag to
                // every particle without contracting the thermal moment.
                const amrex::ParticleReal dVx = Vsx - Vex;
                const amrex::ParticleReal dVy = Vsy - Vey;
                const amrex::ParticleReal dVz = Vsz - Vez;
                ux[ip] -= dVx * one_minus_fac;
                uy[ip] -= dVy * one_minus_fac;
                uz[ip] -= dVz * one_minus_fac;

                // Drag-dissipation heating: each macroparticle contributes
                //   W_dot_p = w_p * m_s * nu * |V_s - V_e|^2   [J/s]
                // to the cell containing it. Sum over particles in the cell
                // gives W_dot per cell; dividing by cell_volume yields the
                // per-cell power density (W/m^3) consumed by QDSMCAddResistiveHeating.
                // NGP scatter is sufficient here: the heating is an integrated
                // quantity and the shape-factor smoothing already lives in the
                // gathered V_s, V_e via the binomial filter on those fields.
                if (include_resistive_heating) {
                    const amrex::Real dV2 = dVx*dVx + dVy*dVy + dVz*dVz;
                    const amrex::Real Wdot_p =
                        wp_arr[ip] * species_mass * nu * dV2 * inv_cell_volume;
                    // Find containing cell via floor of (xp - plo)/dx.
                    int i_cell = 0, j_cell = 0, k_cell = 0;
                    i_cell = static_cast<int>(amrex::Math::floor((xp - plo[0]) * dxi[0]));
#if AMREX_SPACEDIM >= 2
                    j_cell = static_cast<int>(amrex::Math::floor((zp - plo[1]) * dxi[1]));
#endif
#if AMREX_SPACEDIM == 3
                    // For 3D Cartesian the second in-plane coord is yp and
                    // the out-of-plane is zp; remap to the AMReX (i,j,k) tuple.
                    j_cell = static_cast<int>(amrex::Math::floor((yp - plo[1]) * dxi[1]));
                    k_cell = static_cast<int>(amrex::Math::floor((zp - plo[2]) * dxi[2]));
#endif
                    amrex::HostDevice::Atomic::Add(
                        &heating_arr(i_cell, j_cell, k_cell), Wdot_p);
                }
            });
        }
    }
}
