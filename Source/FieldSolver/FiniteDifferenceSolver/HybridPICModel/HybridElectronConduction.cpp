/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "HybridPICModel.H"

#include "ElectronHeatConduction.H"
#include "Fields.H"
#include "WarpX.H"

#include <AMReX_Print.H>

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
    auto const result = warpx::hybrid::advanceElectronHeatConduction(
        temperature, redistribution, charge, simulation.Geom(lev), m_gamma, m_electron_conductivity,
        m_electron_conduction_flux_limiter, dt, m_electron_conduction_max_substeps,
        simulation.verboncoeurAxisCorrection() ? 1.0_rt / 3.0_rt : 1.0_rt / 4.0_rt);
    if (m_electron_conduction_verbosity > 0)
    {
        amrex::Print() << "Electron heat conduction: substeps=" << result.substeps
                       << " graph_iterations=" << result.iterations << '\n';
    }
}
