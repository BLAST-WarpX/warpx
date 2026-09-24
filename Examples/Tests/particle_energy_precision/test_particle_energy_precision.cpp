/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Diagnostics/ReducedDiags/ParticleEnergy.H"
#include "Initialization/WarpXInit.H"
#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <limits>

int main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        auto& ions = simulation.GetPartContainer().GetParticleContainerFromName("ions");
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            auto& soa = iterator.GetStructOfArrays();
            auto* w = soa.GetRealData(PIdx::w).data();
            auto* ux = soa.GetRealData(PIdx::ux).data();
            auto* uy = soa.GetRealData(PIdx::uy).data();
            auto* uz = soa.GetRealData(PIdx::uz).data();
            amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                w[ip] = 1.1 + (ip % 3) * 0.2;
                ux[ip] = 12345.5 + ip * 0.25;
                uy[ip] = -45678.25 - ip * 0.5;
                uz[ip] = 98765.75 + ip * 0.125;
            });
        }
        long double reference_energy = 0;
        long double reference_weight = 0;
        for (auto const& [key, tile] : ions.GetParticles(0)) {
            amrex::ignore_unused(key);
            auto const count = tile.numParticles();
            std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, 4> host;
            std::array<int, 4> const components{PIdx::w, PIdx::ux, PIdx::uy, PIdx::uz};
            for (int c = 0; c < 4; ++c) {
                host[c].resize(count);
                auto const& values = tile.GetStructOfArrays().GetRealData(components[c]);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + count, host[c].begin());
            }
            for (long i = 0; i < count; ++i) {
                // Keep the existing per-particle kinetic-energy calculation.
                // Only weighting and reduction are independently evaluated.
                auto const energy = Algorithms::KineticEnergy(
                    host[1][i], host[2][i], host[3][i], ions.getMass());
                reference_energy += static_cast<long double>(host[0][i]) * energy;
                reference_weight += static_cast<long double>(host[0][i]);
            }
        }
        AMREX_ALWAYS_ASSERT(reference_energy > 0 && reference_weight > 0);
        auto const check = [] (amrex::Real actual, long double expected) {
            auto const bound = 128 * std::numeric_limits<amrex::Real>::epsilon()
                * std::abs(expected);
            amrex::Print().SetPrecision(17) << "actual=" << actual
                << " reference=" << static_cast<double>(expected) << '\n';
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(static_cast<long double>(actual) - expected) <= bound,
                "Particle diagnostics must preserve field-precision weighted reductions.");
        };
        auto const [energy, weight] = ions.sumParticleWeightAndEnergy();
        check(energy, reference_energy);
        check(weight, reference_weight);
        check(ions.sumParticleEnergy(), reference_energy);
        ParticleEnergy diagnostic("energy");
        diagnostic.ComputeDiags(0);
        AMREX_ALWAYS_ASSERT(diagnostic.m_data.size() == 4);
        check(diagnostic.m_data[0], reference_energy);
        check(diagnostic.m_data[1], reference_energy);
        check(diagnostic.m_data[2], reference_energy / reference_weight);
        check(diagnostic.m_data[3], reference_energy / reference_weight);
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
