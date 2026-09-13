/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "HybridPICModel.H"

#include "ElectronHeatConduction.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Print.H>

#include <utility>
#include <vector>

using namespace amrex::literals;

void
HybridPICModel::AdvanceElectronHeatConduction (int const lev, amrex::Real const dt) const
{
    if (!m_electron_heat_conduction)
    {
        return;
    }
    ABLASTR_PROFILE("HybridPICModel::AdvanceElectronHeatConduction()");
    auto& simulation = WarpX::GetInstance();
    using warpx::fields::FieldType;
    auto& temperature = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    auto& redistribution = *simulation.m_fields.get("hybrid_conduction_energy_fp", lev);
    auto const& charge = *simulation.m_fields.get(FieldType::rho_fp, lev);
    amrex::MultiFab* mean_charge = nullptr;
    amrex::MultiFab* effective_charge = nullptr;
    if (m_conductivity_uses_charge_moments)
    {
        mean_charge = simulation.m_fields.get("hybrid_conduction_mean_charge_fp", lev);
        effective_charge = simulation.m_fields.get("hybrid_conduction_effective_charge_fp", lev);
        std::vector<std::pair<amrex::MultiFab const*, amrex::Real>> species_densities;
        auto& particles = simulation.GetPartContainer();
        for (auto const& name : particles.GetSpeciesNames())
        {
            auto const& species = particles.GetParticleContainerFromName(name);
            if (species.getCharge() == 0.0_prt || species.do_not_deposit)
            {
                continue;
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                species.getMass() > 0.0_prt && species.getCharge() > 0.0_prt &&
                    !species.HasEvolvingChargeState(),
                "Composition-dependent conduction requires massive fixed-charge positive ions.");
            species_densities.emplace_back(
                simulation.m_fields.get("rho_fp_" + name, lev),
                static_cast<amrex::Real>(species.getCharge() / PhysConst::q_e));
        }
        warpx::hybrid::computeElectronChargeMoments(*mean_charge, *effective_charge, charge,
                                                    species_densities);
    }
    auto const result = warpx::hybrid::advanceElectronHeatConduction(
        temperature, redistribution, charge, simulation.Geom(lev), m_gamma, m_electron_conductivity,
        m_electron_conduction_flux_limiter, dt, m_electron_conduction_max_substeps,
        simulation.verboncoeurAxisCorrection() ? 1.0_rt / 3.0_rt : 1.0_rt / 4.0_rt, mean_charge,
        effective_charge, m_electron_composition_conductivity);
    if (m_electron_conduction_verbosity > 0)
    {
        amrex::Print() << "Electron heat conduction: substeps=" << result.substeps
                       << " graph_iterations=" << result.iterations << '\n';
    }
}
