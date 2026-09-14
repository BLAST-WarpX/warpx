/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronHeatConduction.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace amrex::literals;

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse test("test");
        std::string table_file;
        test.query("table_file", table_file);
        bool const table = !table_file.empty();
        amrex::ParmParse parameters("conduction_eos");
        parameters.add("electron_thermodynamics",
                       table ? "singularity_spiner" : "fixed_charge_latent_energy");
        if (table)
        {
            parameters.addarr("electron_eos_species", std::vector<std::string>{"ions"});
            parameters.add("electron_eos_ions_table_file", table_file);
            parameters.add("electron_eos_ions_material_id", 7400);
        }
        else
        {
            parameters.addarr("electron_latent_transition_temperature_eV",
                              std::vector<amrex::Real>{10});
            parameters.addarr("electron_latent_energy_eV", std::vector<amrex::Real>{40});
            parameters.addarr("electron_latent_sharpness", std::vector<amrex::Real>{8});
        }
        ElectronThermodynamics owner;
        owner.ReadParameters(parameters, 5._rt / 3);
        auto const eos = owner.executor();
        constexpr int cells = 64, steps = 64;
        amrex::Geometry geometry(amrex::Box(amrex::IntVect(0), amrex::IntVect(cells - 1)),
                                 amrex::RealBox({0._rt}, {1._rt}), 0, std::array<int, 1>{1});
        amrex::BoxArray boxes(geometry.Domain());
        boxes.maxSize(16);
        boxes.convert(amrex::IntVect::TheNodeVector());
        amrex::DistributionMapping distribution(boxes);
        amrex::MultiFab temperature(boxes, distribution, 1, 1), charge(boxes, distribution, 1, 1);
        amrex::MultiFab mass(boxes, distribution, 1, 0), delta(boxes, distribution, 1, 1);
        auto const number =
            table ? 20 * 1000 / (183.84_rt * PhysConst::m_u) : 0.5_rt * (2._rt / 3) / PhysConst::kb;
        charge.setVal(PhysConst::q_e * number);
        mass.setVal(table ? 1000 : 0);
        auto const base = table ? 1.e4_rt : 10 * PhysConst::q_e / PhysConst::kb;
        for (amrex::MFIter it(temperature); it.isValid(); ++it)
        {
            auto const t = temperature.array(it);
            amrex::ParallelFor(
                it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                { t(i, j, k) = base * (1 + 0.5_rt * std::cos(2 * MathConst::pi * i / cells)); });
        }
        temperature.FillBoundary(geometry.periodicity());
        amrex::Parser conductivity(table ? "1.2e8" : "0.5");
        conductivity.registerVariables({"rho", "Te"});
        auto const parser = conductivity.compile<2>();
        int substeps = 0, iterations = 0;
        bool reject_budget = false;
        test.query("reject_budget", reject_budget);
        if (reject_budget) {
            amrex::MultiFab saved(boxes, distribution, 1, 1);
            amrex::MultiFab::Copy(saved, temperature, 0, 0, 1, 1);
            delta.setVal(79);
            bool caught = false;
            try {
                warpx::hybrid::advanceElectronHeatConduction(temperature, delta, charge,
                    geometry, 5._rt/3, parser, 0, 0.03_rt, 1, 1._rt/3,
                    nullptr, nullptr, {}, &eos, table ? &mass : nullptr);
            } catch (std::exception const& error) {
                caught = std::string(error.what()).find("exhausted its substep budget")
                    != std::string::npos;
            }
            AMREX_ALWAYS_ASSERT(caught);
            amrex::MultiFab::Subtract(saved, temperature, 0, 0, 1, 1);
            AMREX_ALWAYS_ASSERT(saved.norm0(0,1) == 0);
            AMREX_ALWAYS_ASSERT(delta.min(0,1) == 79 && delta.max(0,1) == 79);
            amrex::Print() << "Nonlinear conduction interval rejection preserves live state.\n";
        }
        for (int step = 0; step < (reject_budget ? 0 : steps); ++step)
        {
            auto const result = warpx::hybrid::advanceElectronHeatConduction(
                temperature, delta, charge, geometry, 5._rt / 3, parser, 0, 0.03_rt / steps, 4096,
                1._rt / 3, nullptr, nullptr, {}, &eos, table ? &mass : nullptr);
            substeps += result.substeps;
            iterations += result.iterations;
        }
        // Each periodic node is a finite-volume thermal degree of freedom.
        // Pack only owned nodes, excluding the duplicate upper periodic node.
        std::vector<amrex::Real> values(cells, 0);
        auto const owns = temperature.OwnerMask(geometry.periodicity());
        for (amrex::MFIter it(temperature); it.isValid(); ++it)
        {
            auto const box = it.validbox();
            amrex::FArrayBox packed(box, 1, amrex::The_Arena());
            auto const t = temperature.const_array(it);
            auto const owner_mask = owns->const_array(it);
            auto const out = packed.array();
            amrex::ParallelFor(
                box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                { out(i, j, k) = owner_mask(i, j, k) && i < cells ? t(i, j, k) : 0; });
            amrex::FArrayBox host(box, 1, amrex::The_Pinned_Arena());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, packed.dataPtr(),
                             packed.dataPtr() + packed.size(), host.dataPtr());
            for (int i = box.smallEnd(0); i <= box.bigEnd(0) && i < cells; ++i)
            {
                values[i] += host(amrex::IntVect(i));
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum(values.data(), cells);
        if (amrex::ParallelDescriptor::IOProcessor() && !reject_budget)
        {
            std::ofstream output("nonlinear_conduction.csv");
            output << "x,T_initial,T_final,U_initial,U_final\n" << std::setprecision(17);
            long double before = 0, after = 0, absolute = 0;
            for (int i = 0; i < cells; ++i)
            {
                auto const initial = base * (1 + 0.5_rt * std::cos(2 * MathConst::pi * i / cells));
                ElectronThermodynamicsExecutor::MaterialMassDensities materials{};
                if (table)
                {
                    materials[0] = 1000;
                }
                auto const old = eos.stateFromMaterialMassDensitiesTemperature(
                                        number * PhysConst::q_e, materials, initial)
                                     .internal_energy_density;
                auto const current = eos.stateFromMaterialMassDensitiesTemperature(
                                            number * PhysConst::q_e, materials, values[i])
                                         .internal_energy_density;
                before += old;
                after += current;
                absolute += std::abs(old) + std::abs(current);
                output << static_cast<amrex::Real>(i) / cells << ',' << initial << ',' << values[i]
                       << ',' << old << ',' << current << '\n';
            }
            AMREX_ALWAYS_ASSERT(output.good());
            AMREX_ALWAYS_ASSERT(std::abs(after - before) < 1.e-10L * absolute);
            amrex::Print() << "Nonlinear EOS conduction: table=" << table
                           << " substeps=" << substeps << " iterations=" << iterations
                           << " energy_residual=" << std::abs(after - before) / absolute << '\n';
        }
    }
    amrex::Finalize();
}
