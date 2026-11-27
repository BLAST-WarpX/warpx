/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/FiniteMassPhotonCollision.H"
#include "Radiation/FrozenPhotonCollisionRate.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using namespace warpx::radiation;
        using Direction = amrex::GpuArray<amrex::Real, 3>;
        std::array<Direction, 6> const directions{{
            {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
        for (Direction const beta : {Direction{0.01, 0, 0}, Direction{0.3, -0.2, 0.1}})
        {
            amrex::Real b2 = 0;
            for (auto const value : beta) { b2 += value*value; }
            auto const gamma = 1/std::sqrt(1-b2);
            FourVector lab_drag{};
            FourVector equilibrium{};
            FourVector incorrect_lab_rate{};
            FourVector bath_moments{};
            for (auto const& n : directions)
            {
                FourVector const lab_incoming{1./6, n[0]/6, n[1]/6, n[2]/6};
                auto const rate = EvaluateFrozenPhotonCollisionRate(1, beta, n);
                AMREX_ALWAYS_ASSERT(rate.valid);
                auto boosted = BoostFourVector(lab_incoming, beta);
                auto const doppler = boosted[0]/lab_incoming[0];
                // A fixed lab-volume comoving-isotropic bath needs BOTH the
                // individual photon energy and macroparticle number weight
                // transformed. Constant boosted weights are not this bath.
                for (auto& value : boosted) { value *= doppler; }
                Direction boosted_direction{};
                for (int d = 0; d < 3; ++d)
                {
                    boosted_direction[d] = boosted[d+1]/boosted[0];
                }
                auto const bath_rate = EvaluateFrozenPhotonCollisionRate(
                    1, beta, boosted_direction);
                AMREX_ALWAYS_ASSERT(bath_rate.valid);
                for (int c = 0; c < 4; ++c) { bath_moments[c] += boosted[c]; }
                for (auto const& outgoing : directions)
                {
                    auto const drag = CoherentPhotonCollision(lab_incoming, outgoing, beta);
                    auto const bath = CoherentPhotonCollision(boosted, outgoing, beta);
                    AMREX_ALWAYS_ASSERT(drag.valid && bath.valid);
                    lab_drag[0] += rate.inverse_length*drag.radiation_debit/6;
                    equilibrium[0] += bath_rate.inverse_length*bath.radiation_debit/6;
                    incorrect_lab_rate[0] += bath.radiation_debit/6;
                    for (int d = 0; d < 3; ++d)
                    {
                        lab_drag[d+1] += rate.inverse_length*PhysConst::c*drag.material_impulse[d]/6;
                        equilibrium[d+1] += bath_rate.inverse_length*PhysConst::c
                            *bath.material_impulse[d]/6;
                        incorrect_lab_rate[d+1] += PhysConst::c*bath.material_impulse[d]/6;
                    }
                }
            }
            auto const energy_scale = bath_moments[0];
            AMREX_ALWAYS_ASSERT(std::abs(bath_moments[0]-gamma*gamma*(1+b2/3)) < 2.e-14*energy_scale);
            AMREX_ALWAYS_ASSERT(std::abs(lab_drag[0]+4./3*gamma*gamma*gamma*b2) < 2.e-14*energy_scale);
            for (int d = 0; d < 3; ++d)
            {
                auto const expected = -4./3*gamma*gamma*gamma*beta[d];
                AMREX_ALWAYS_ASSERT(std::abs(lab_drag[d+1]-expected) < 2.e-14*energy_scale);
                AMREX_ALWAYS_ASSERT(std::abs(bath_moments[d+1]-4./3*gamma*gamma*beta[d])
                                    < 2.e-14*energy_scale);
                AMREX_ALWAYS_ASSERT(std::abs(incorrect_lab_rate[d+1]-gamma*gamma*beta[d]/3)
                                    < 2.e-14*energy_scale);
            }
            for (auto const value : equilibrium)
            {
                AMREX_ALWAYS_ASSERT(std::abs(value) < 2.e-14*energy_scale);
            }
        }
        AMREX_ALWAYS_ASSERT(!EvaluateFrozenPhotonCollisionRate(1, {1, 0, 0}, {0, 0, 1}).valid);
        AMREX_ALWAYS_ASSERT(!EvaluateFrozenPhotonCollisionRate(-1, {0, 0, 0}, {0, 0, 1}).valid);
        AMREX_ALWAYS_ASSERT(!EvaluateFrozenPhotonCollisionRate(1, {0, 0, 0}, {0, 0, 2}).valid);
        amrex::Print() << "Frozen proper-rate source: exact angular drag and boosted-bath "
                         "equilibrium quadrature passed; constant lab-rate control differs.\n";
    }
    amrex::Finalize();
}
