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
        bool const photons = ions.AmIA<PhysicalSpecies::photon>();
        auto const mass = ions.getMass();
        for (auto const& [key, tile] : ions.GetParticles(0)) {
            amrex::ignore_unused(key);
            auto const count = tile.numParticles();
            amrex::Gpu::HostVector<amrex::ParticleReal> host_weight(count);
            auto const& weights = tile.GetStructOfArrays().GetRealData(PIdx::w);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, weights.begin(),
                weights.begin() + count, host_weight.begin());
            // Evaluate native per-particle energy on the same backend as the
            // reduction; isolate weighting/summing from host/device math differences.
            amrex::Gpu::DeviceVector<amrex::ParticleReal> energies(count);
            auto* energy = energies.data();
            auto const& soa = tile.GetStructOfArrays();
            auto const* ux = soa.GetRealData(PIdx::ux).data();
            auto const* uy = soa.GetRealData(PIdx::uy).data();
            auto const* uz = soa.GetRealData(PIdx::uz).data();
            amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(long i) {
                energy[i] = photons
                    ? Algorithms::KineticEnergyPhotons(ux[i], uy[i], uz[i])
                    : Algorithms::KineticEnergy(ux[i], uy[i], uz[i], mass);
            });
            amrex::Gpu::HostVector<amrex::ParticleReal> host_energy(count);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, energies.begin(), energies.end(),
                host_energy.begin());
            for (long i = 0; i < count; ++i) {
                reference_energy += static_cast<long double>(host_weight[i]) * host_energy[i];
                reference_weight += static_cast<long double>(host_weight[i]);
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
