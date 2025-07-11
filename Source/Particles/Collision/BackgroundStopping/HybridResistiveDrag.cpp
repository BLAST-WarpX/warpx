/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HybridResistiveDrag.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <string>

namespace warpx::particles::collision::backgroundstopping {

HybridResistiveDrag::HybridResistiveDrag (std::string const& collision_name)
    : CollisionBase(collision_name)
{
    using namespace amrex::literals;

    const amrex::ParmParse pp_collision_name(collision_name);

    // Default type is "resistivity", but can also be expressed as collision frequency "nu"
    // depending on this definition the expression may be normalized to eta by eta = m_e * nu / (e * rho)
    // in Hybrid the total rho deposited for all species is the electron rho.
    std::string expression_type_string = "resistivity";
    pp_collision_name.query("expression_type", expression_type_string);

    if (expression_type_string == "resistivity") {
        m_type = HybridResistiveDragType::RESISTIVITY;
    } else if (expression_type_string == "collision_frequency") {
        m_type = HybridResistiveDragType::COLLISION_FREQUENCY;
    } else {
        WARPX_ABORT_WITH_MESSAGE("expression_type for hybrid_resistive_drag must be either resisistivity or collision_frequency.");
    }

    // Set up parser for eta with rho, v_i, v_e, T_i, T_e, B inputs
    const std::string & function_string;
    pp_collision_name.query("drag_function(rho,v_i,v_e,T_i,T_e,B)", function_string);
    m_expression_parser = utils::parser::makeParser(function_string,{"rho","v_i","v_e","T_i","T_e","B"});
    m_expression_func = m_expression_parser->compile<6>();
}

void
HybridResistiveDrag::doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc)
{
    WARPX_PROFILE("HybridResistiveDrag::doCollisions()");
    using namespace amrex::literals;

    // Iterate over each species specified in the collision operator
    for (auto & species_name : m_species_names) {
        auto& species = mypc->GetParticleContainerFromName(species_name);
        amrex::ParticleReal const species_mass = species.getMass();
        amrex::ParticleReal const species_charge = species.getCharge();

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(species_mass > 0_prt, "Error: With hybrid resistive drag, the species mass must be > 0");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(species_charge != 0_prt, "Error: With hybrid resistive drag, the species charge must be nonzero");

        // Loop over refinement levels
        auto const flvl = species.finestLevel();
        for (int lev = 0; lev <= flvl; ++lev) {

            auto *cost = WarpX::getCosts(lev);

            // loop over particles box by box
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (WarpXParIter pti(species, lev); pti.isValid(); ++pti) {
                if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
                {
                    amrex::Gpu::synchronize();
                }
                auto wt = static_cast<amrex::Real>(amrex::second());

                doHybridResistiveDragOnIonsWithinTile(pti, dt, cur_time, species_mass, species_charge);

                if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
                {
                    amrex::Gpu::synchronize();
                    wt = static_cast<amrex::Real>(amrex::second()) - wt;
                    amrex::HostDevice::Atomic::Add(&(*cost)[pti.index()], wt);
                }
            }

        }
    }
}

void HybridResistiveDrag::doHybridResistiveDragOnIonsWithinTile (WarpXParIter& pti, amrex::Real dt, amrex::Real t,
                                                               amrex::ParticleReal species_mass, amrex::ParticleReal species_charge)
{
    using namespace amrex::literals;

    // So that GPU code gets its intrinsic, not the host-only C++ library version
    using std::sqrt, std::abs, std::log, std::exp, std::pow;

    // get particle count
    long const np = pti.numParticles();

    // get background particle mass
    amrex::ParticleReal const mass_i = m_background_mass;
    amrex::ParticleReal const charge_state_i = m_background_charge_state;

    // setup parsers for the background density and temperature
    auto const n_i_func = m_background_density_func;
    auto const T_i_func = m_background_temperature_func;

    // get Struct-Of-Array particle data, also called attribs
    auto& attribs = pti.GetAttribs();
    amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

    // May be needed to evaluate the density function
    auto const GetPosition = GetParticlePosition<PIdx>(pti);

    amrex::ParallelFor(np,
        [=] AMREX_GPU_HOST_DEVICE (long ip)
        {

            amrex::ParticleReal x, y, z;
            GetPosition.AsStored(ip, x, y, z);
            amrex::ParticleReal const n_i = n_i_func(x, y, z, t);
            amrex::ParticleReal const T_i = T_i_func(x, y, z, t)*PhysConst::kb;

            AMREX_ASSERT(n_i > 0_prt);
            AMREX_ASSERT(T_i > 0_prt);

            // This implements the equation 14.20 from Introduction to Plasma Physics,
            // Goldston and Rutherford, the slowing down of beam ions due to collisions with electrons.
            // The equation is written with energy, W, as dW/dt = -alpha/W**0.5, and integrated to
            // give W(t+dt) = (W(t)**1.5 - 3./2.*alpha*dt)**(2/3)

            amrex::ParticleReal constexpr pi = MathConst::pi;
            amrex::ParticleReal constexpr q_e = PhysConst::q_e;
            amrex::ParticleReal constexpr q_e2 = q_e*q_e;
            amrex::ParticleReal constexpr ep0 = PhysConst::ep0;
            amrex::ParticleReal constexpr ep02 = ep0*ep0;

            amrex::ParticleReal const qi2 = charge_state_i*charge_state_i*q_e2;
            amrex::ParticleReal const qb2 = species_charge*species_charge;
            amrex::ParticleReal const Zb = abs(species_charge/q_e);

            amrex::ParticleReal const vth = sqrt(3_prt*T_i/mass_i);
            amrex::ParticleReal const wp = sqrt(n_i*q_e2/(ep0*mass_i));
            amrex::ParticleReal const lambdadb = vth/wp;
            amrex::ParticleReal const lambdadb3 = lambdadb*lambdadb*lambdadb;
            amrex::ParticleReal const loglambda = log((12_prt*pi/Zb)*(n_i*lambdadb3));

            AMREX_ASSERT(loglambda > 0_prt);

            amrex::ParticleReal const alpha = sqrt(2_prt)*n_i*qi2*qb2*sqrt(species_mass)*loglambda/(8_prt*pi*ep02*mass_i);

            amrex::ParticleReal const W0 = 0.5_prt*species_mass*(ux[ip]*ux[ip] + uy[ip]*uy[ip] + uz[ip]*uz[ip]);
            amrex::ParticleReal const f1 = pow(W0, 1.5_prt) - 1.5_prt*alpha*dt;
            // If f1 goes negative, the particle has fully stopped, so set W1 to 0.
            amrex::ParticleReal const W1 = pow((f1 > 0_prt ? f1 : 0_prt), 2_prt/3_prt);
            amrex::ParticleReal const vscale = (W0 > 0_prt ? std::sqrt(W1/W0) : 0_prt);

            ux[ip] *= vscale;
            uy[ip] *= vscale;
            uz[ip] *= vscale;

        }
        );
}

} // namespace warpx::particles::collision::backgroundstopping
