/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteMassPhotonReference.H"

#include "Radiation/FiniteMassPhotonCollision.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using namespace warpx::radiation;
        auto const incoming_direction = finite_mass_reference::Unit({1, 2, 3});
        auto const frame_direction = finite_mass_reference::Unit({-2, 1, -0.5L});
        amrex::GpuArray<amrex::Real, 3> const beta{0.11, -0.07, 0.03};
        amrex::GpuArray<amrex::Real, 3> direction{};
        for (int d = 0; d < 3; ++d) { direction[d] = frame_direction[d]; }

        // Scaling covers squared-norm overflow/underflow without invalidating
        // the finite input four-vector itself.
        for (amrex::Real const energy : {1.e-200, 0.03, 0.2, 1.e200}) {
            FourVector incoming{energy, 0, 0, 0};
            for (int d = 0; d < 3; ++d) {
                incoming[d+1] = energy*incoming_direction[d];
            }
            auto const collision = CoherentPhotonCollision(incoming, direction, beta);
            AMREX_ALWAYS_ASSERT(collision.valid);
            finite_mass_reference::Solution expected;
            finite_mass_reference::Replay({{energy, incoming_direction, frame_direction}},
                                          {beta[0], beta[1], beta[2]}, expected);
            AMREX_ALWAYS_ASSERT(std::abs(collision.outgoing_photon[0]/energy
                - expected.outgoing_energy[0]/energy) < 2.e-14L);
            for (int d = 0; d < 3; ++d) {
                AMREX_ALWAYS_ASSERT(std::abs(collision.outgoing_photon[d+1]/energy
                    - expected.outgoing_energy[0]/energy*expected.outgoing_direction[0][d])
                    < 2.e-14L);
                AMREX_ALWAYS_ASSERT(std::abs(collision.material_impulse[d]*PhysConst::c/energy
                    - (incoming[d+1]/energy-collision.outgoing_photon[d+1]/energy)) < 2.e-14L);
            }
            AMREX_ALWAYS_ASSERT(std::abs(collision.radiation_debit/energy
                - (1-expected.outgoing_energy[0]/energy)) < 2.e-14L);
        }

        finite_mass_reference::Vector initial_u{beta[0], beta[1], beta[2]};
        auto const initial_gamma = 1/std::sqrt(1-finite_mass_reference::Dot(initial_u, initial_u));
        for (auto& value : initial_u) { value *= initial_gamma; }
        for (long double const energy : {0.03L, 0.2L}) {
            finite_mass_reference::Photon const photon{energy, incoming_direction, frame_direction};
            auto const reference = finite_mass_reference::Solve({{1, initial_u}}, {photon});
            AMREX_ALWAYS_ASSERT(reference.valid);
            FourVector incoming{static_cast<amrex::Real>(energy), 0, 0, 0};
            amrex::GpuArray<amrex::Real, 3> secant{};
            for (int d = 0; d < 3; ++d) {
                incoming[d+1] = energy*incoming_direction[d];
                secant[d] = reference.frame_velocity[d];
            }
            auto const collision = CoherentPhotonCollision(incoming, direction, secant);
            AMREX_ALWAYS_ASSERT(collision.valid);
            finite_mass_reference::Vector lab_direction{};
            for (int d = 0; d < 3; ++d) {
                lab_direction[d] = collision.outgoing_photon[d+1]/collision.outgoing_photon[0];
            }
            auto const mass_shell = finite_mass_reference::MassShellOutgoingEnergy(
                initial_u, photon, lab_direction);
            AMREX_ALWAYS_ASSERT(std::abs(collision.outgoing_photon[0]-mass_shell) < 2.e-14L*energy);
            AMREX_ALWAYS_ASSERT(std::abs(collision.radiation_debit-reference.material_work)
                                < 2.e-14L*energy);
        }
        auto const cold = finite_mass_reference::Solve({{1, {0, 0, 0}}},
                                                       {{0.2L, {0, 0, 1}, {0, 0, -1}}});
        AMREX_ALWAYS_ASSERT(cold.valid);
        auto const backscatter = CoherentPhotonCollision({0.2, 0, 0, 0.2}, {0, 0, -1},
            {static_cast<amrex::Real>(cold.frame_velocity[0]),
             static_cast<amrex::Real>(cold.frame_velocity[1]),
             static_cast<amrex::Real>(cold.frame_velocity[2])});
        AMREX_ALWAYS_ASSERT(backscatter.valid);
        AMREX_ALWAYS_ASSERT(std::abs(backscatter.outgoing_photon[0]-0.2L/1.4L) < 2.e-14L*0.2L);

        auto reject = [&] (FourVector const& incoming,
                           amrex::GpuArray<amrex::Real, 3> const& outgoing_direction,
                           amrex::GpuArray<amrex::Real, 3> const& frame_beta)
        {
            FourVector output{2, 3, 5, 7};
            amrex::GpuArray<amrex::Real, 3> impulse{11, 13, 17};
            amrex::Real debit = 19;
            auto const old_output = output;
            auto const old_impulse = impulse;
            AMREX_ALWAYS_ASSERT(!CoherentPhotonCollision(
                incoming, outgoing_direction, frame_beta, output, impulse, debit));
            for (int d = 0; d < 4; ++d) { AMREX_ALWAYS_ASSERT(output[d] == old_output[d]); }
            for (int d = 0; d < 3; ++d) { AMREX_ALWAYS_ASSERT(impulse[d] == old_impulse[d]); }
            AMREX_ALWAYS_ASSERT(debit == 19);
        };
        for (amrex::Real const scale : {1.e-200, 1., 1.e200}) {
            reject({scale, 2*scale, 0, 0}, {1, 0, 0}, beta);
        }
        reject({0, 0, 0, 0}, {1, 0, 0}, beta);
        reject({-1, 1, 0, 0}, {1, 0, 0}, beta);
        reject({1, 1, 0, 0}, {2, 0, 0}, beta);
        reject({1, 1, 0, 0}, {1, 0, 0}, {1, 0, 0});
        reject({1, 1, 0, 0}, {1.e200, 0, 0}, beta);
        auto const nan = std::numeric_limits<amrex::Real>::quiet_NaN();
        for (int component = 0; component < 4; ++component) {
            FourVector invalid{1, 1, 0, 0};
            invalid[component] = nan;
            reject(invalid, {1, 0, 0}, beta);
        }
        amrex::Print() << "Coherent photon transform agrees with independent Doppler/aberration "
                      << "reference; invalid input preserves all outputs.\n";
    }
    amrex::Finalize();
}
