/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <cmath>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        constexpr int count = 256;
        amrex::Gpu::DeviceVector<int> device_results(count);
        int* const results = device_results.dataPtr();
        amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE (int const i)
        {
            ElectronThermodynamicsExecutor eos;
            if (i % 2 != 0) {
                eos.m_model = ElectronThermodynamicsModel::FixedChargeLatentEnergy;
                eos.m_num_latent_transitions = 1;
                eos.m_latent_transition_temperature[0] = amrex::Real(1.234e5);
                eos.m_latent_energy_per_electron[0] = amrex::Real(2.34e-17);
                eos.m_latent_transition_sharpness[0] = amrex::Real(3.5);
            }
            amrex::Real const density = amrex::Real(1.234567)
                * std::pow(amrex::Real(10), amrex::Real(i % 19 - 5));
            amrex::Real const temperature = amrex::Real(0.9876543)
                * std::pow(amrex::Real(10), amrex::Real(i % 13 - 2));
            ElectronThermodynamicsExecutor::MaterialMassDensities const masses{};
            auto const update = eos.applyEnergyDensityIncrement(
                density, masses, temperature, amrex::Real(0));
            auto const invalid_density = eos.applyEnergyDensityIncrement(
                -density, masses, temperature, amrex::Real(0));
            auto const invalid_temperature = eos.applyEnergyDensityIncrement(
                density, masses, -temperature, amrex::Real(0));
            results[i] = update.valid && update.temperature == temperature
                && update.energy_change_density == amrex::Real(0)
                && update.energy_residual_density == amrex::Real(0)
                && !invalid_density.valid && !invalid_temperature.valid;
        });
        amrex::Gpu::HostVector<int> host_results(count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_results.begin(),
                         device_results.end(), host_results.begin());
        int failures = 0;
        for (int const passed : host_results) { failures += passed == 0 ? 1 : 0; }
        amrex::Print() << "Zero-source caloric checks: " << count - failures
                       << '/' << count << '\n';
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(failures == 0,
            "Zero electron-energy source must preserve temperature and all ledgers exactly.");
    }
    amrex::Finalize();
}
