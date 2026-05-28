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
    const amrex::Real electron_temp = warpx.get_pointer_HybridPICModel()->m_elec_temp;
    const int nox = WarpX::nox;
    const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;
    const bool galerkin_interpolation = WarpX::galerkin_interpolation;

    for (int lev = 0; lev <= species.finestLevel(); ++lev) {
        ablastr::fields::VectorField Ve_fp = warpx.m_fields.get_alldirs("Ve_fp", lev);
        ablastr::fields::VectorField Vs_fp = warpx.m_fields.get_alldirs("Vs_fp_" + m_species_names[0], lev);
        ablastr::fields::VectorField B_fp  = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        amrex::MultiFab const & rho_fp    = *warpx.m_fields.get(FieldType::rho_fp, lev);

        const amrex::XDim3 dinv = WarpX::InvCellSize(lev);
        const auto dxi = warpx.Geom(lev).InvCellSizeArray();
        const auto plo = warpx.Geom(lev).ProbLoArray();

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

                const amrex::Real rho_val = ablastr::particles::doGatherScalarFieldNodal(
                    xp, yp, zp, rho_arr, dxi, plo);

                const amrex::Real Vsmag = std::sqrt(Vsx*Vsx + Vsy*Vsy + Vsz*Vsz);
                const amrex::Real Vemag = std::sqrt(Vex*Vex + Vey*Vey + Vez*Vez);
                const amrex::Real Bmag  = std::sqrt(Bxp*Bxp + Byp*Byp + Bzp*Bzp);

                // Per-species temperature is not yet on the grid; pass 0 as Ts.
                const amrex::Real nu = drag_func(rho_val, Vsmag, Vemag, 0._rt, electron_temp, Bmag);
                const amrex::Real one_minus_fac = -std::expm1(-nu * dt);

                // Uniform per-cell shift: applies the bulk-fluid drag to
                // every particle without contracting the thermal moment.
                ux[ip] -= (Vsx - Vex) * one_minus_fac;
                uy[ip] -= (Vsy - Vey) * one_minus_fac;
                uz[ip] -= (Vsz - Vez) * one_minus_fac;
            });
        }
    }
}
