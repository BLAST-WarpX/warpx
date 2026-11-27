/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

namespace
{
// Independently inspect native nodal rho/T and evaluate the EOS there. Do not
// invert cell-averaged temperature or use the conduction solver's heat ledger.
std::vector<amrex::Real>
ReadProfile (WarpX& simulation)
{
    using warpx::fields::FieldType;
    auto const& geometry = simulation.Geom(0);
    int const nodes = geometry.Domain().length(0) + 1;
    int const axial_cells = geometry.Domain().length(1);
    auto const& temperature =
        *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& charge = *simulation.m_fields.get(FieldType::rho_fp, 0);
    auto const* model = simulation.get_pointer_HybridPICModel();
    auto const eos = model->electronThermodynamicsExecutor();
    amrex::MultiFab const* unit_charge = nullptr;
    amrex::Real ion_mass = 0;
    if (eos.isSingularitySpiner())
    {
        AMREX_ALWAYS_ASSERT(model->electronThermodynamicsNumMaterials() == 1);
        unit_charge = simulation.m_fields.get(
            "ni_charge_fp_" + model->electronThermodynamicsMaterialSpeciesName(0), 0);
        ion_mass =
            simulation.GetPartContainer()
                .GetParticleContainerFromName(model->electronThermodynamicsMaterialSpeciesName(0))
                .getMass();
    }
    std::vector<amrex::Real> values(4 * nodes, 0);
    auto const ownership = temperature.OwnerMask(geometry.periodicity());
    for (amrex::MFIter it(temperature); it.isValid(); ++it)
    {
        auto const copy = [&] (amrex::MultiFab const& field)
        {
            amrex::FArrayBox host(field[it].box(), 1, amrex::The_Pinned_Arena());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, field[it].dataPtr(),
                             field[it].dataPtr() + field[it].box().numPts(), host.dataPtr());
            return host;
        };
        auto const t = copy(temperature), rho = copy(charge);
        amrex::FArrayBox ni;
        if (unit_charge)
        {
            ni = copy(*unit_charge);
        }
        amrex::IArrayBox mask((*ownership)[it].box(), 1, amrex::The_Pinned_Arena());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, (*ownership)[it].dataPtr(),
                         (*ownership)[it].dataPtr() + (*ownership)[it].box().numPts(),
                         mask.dataPtr());
        auto const box = it.validbox();
        for (int j = box.smallEnd(1); j <= box.bigEnd(1) && j < axial_cells; ++j)
        {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i)
            {
                amrex::IntVect const cell(i, j);
                if (!mask(cell))
                {
                    continue;
                }
                ElectronThermodynamicsExecutor::MaterialMassDensities materials{};
                if (unit_charge)
                {
                    materials[0] = ion_mass * ni(cell) / PhysConst::q_e;
                }
                auto const state =
                    eos.stateFromMaterialMassDensitiesTemperature(rho(cell), materials, t(cell));
                values[4 * i] += t(cell) / axial_cells;
                values[4 * i + 1] += rho(cell) / axial_cells;
                values[4 * i + 2] += state.internal_energy_density / axial_cells;
                values[4 * i + 3] += materials[0] / axial_cells;
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(values.data(), static_cast<int>(values.size()));
    return values;
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        simulation.HybridPICPrepareElectronStateForDiagnostics();
        auto const initial = ReadProfile(simulation);
        simulation.Evolve();
        auto const final = ReadProfile(simulation);
        AMREX_ALWAYS_ASSERT(initial.size() == final.size());
        auto const& geometry = simulation.Geom(0);
        int const cells = geometry.Domain().length(0);
        auto const dr = geometry.CellSize(0);
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            std::ofstream output("native_nonlinear_conduction.csv");
            output << "r,T0,T1,rho0,rho1,U0,U1,mass_density,volume\n" << std::setprecision(17);
            long double change = 0, scale = 0;
            for (int i = 0; i <= cells; ++i)
            {
                auto const area =
                    i == 0 ? MathConst::pi * dr * dr *
                                 (simulation.verboncoeurAxisCorrection() ? 1.0 / 3 : 1.0 / 4)
                           : 2 * MathConst::pi * i * dr * dr * (i == cells ? 0.5 : 1.0);
                auto const volume = area * geometry.ProbLength(1);
                for (int d = 0; d < 4; ++d)
                {
                    AMREX_ALWAYS_ASSERT(std::isfinite(initial[4 * i + d]) &&
                                        std::isfinite(final[4 * i + d]));
                }
                AMREX_ALWAYS_ASSERT(std::abs(final[4 * i + 1] - initial[4 * i + 1]) <=
                                    1.e-12 * initial[4 * i + 1]);
                change +=
                    volume * (static_cast<long double>(final[4 * i + 2]) - initial[4 * i + 2]);
                scale += volume * (std::abs(initial[4 * i + 2]) + std::abs(final[4 * i + 2]));
                output << i * dr << ',' << initial[4 * i] << ',' << final[4 * i] << ','
                       << initial[4 * i + 1] << ',' << final[4 * i + 1] << ',' << initial[4 * i + 2]
                       << ',' << final[4 * i + 2] << ',' << initial[4 * i + 3] << ',' << volume
                       << '\n';
            }
            AMREX_ALWAYS_ASSERT(output.good() && std::abs(change) <= 1.e-10L * scale);
            amrex::Print() << "Native RZ nonlinear conduction energy residual=" << change / scale
                           << '\n';
        }
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
