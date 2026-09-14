/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Radiation/RadiationTransport.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

using namespace amrex::literals;

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        simulation.HybridPICPrepareElectronStateForDiagnostics();
        using warpx::fields::FieldType;
        auto& te = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& radiation = *simulation.m_fields.get(FieldType::radiation_diffusion_energy, 0);
        auto const& material_charge = *simulation.m_fields.get("ni_charge_fp_ions", 0);
        auto const& geometry = simulation.Geom(0);
        amrex::Real const dx = geometry.CellSize(0);
        constexpr amrex::Real mass = 2.1618558138269478e-26_rt;
        // ni_charge_fp is q_e*n_i, not the physical species charge density
        // Z*q_e*n_i. Keep this distinct from the total electron charge field.
        constexpr amrex::Real charge = PhysConst::q_e;
        constexpr amrex::Real dt = 1.e-13_rt;
        std::string filename;
        amrex::ParmParse pp("hybrid_pic_model");
        pp.get("electron_eos_ions_table_file", filename);
        // Direct ElectronOnly library queries, bypassing the WarpX caloric
        // wrapper and source ledger, audit units and fixed-Z mass mapping.
        singularity::SpinerEOSDependsRhoT oracle(filename, 7400,
                                                 singularity::TableSplit::ElectronOnly);
        auto caloric = [&]
        {
            amrex::Real total = 0;
            for (amrex::MFIter mfi(radiation); mfi.isValid(); ++mfi)
            {
                auto const t = te.const_array(mfi);
                auto const rho = material_charge.const_array(mfi);
                for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell)
                {
                    auto const i = cell()[0];
                    for (int di = 0; di < 2; ++di)
                    {
                        auto const density = rho(i + di, 0, 0) * mass / charge;
                        auto const specific = oracle.InternalEnergyFromDensityTemperature(
                            density * 1.e-3_rt, t(i + di, 0, 0));
                        total += 0.5_rt * dx * density * 1.e-4_rt * specific;
                    }
                }
            }
            amrex::ParallelDescriptor::ReduceRealSum(total);
            return total;
        };
        auto const initial_material = caloric();
        auto const initial = initial_material + radiation.sum(0, false);
        amrex::Real worst = 0;
        std::ofstream history;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            history.open("coupled_spiner_history.csv");
            history << std::setprecision(17)
                    << "step,time_s,radiation_J,material_J,min_T_K,max_T_K,raw_relative_error\n";
        }
        for (int step = 0; step <= 100; ++step)
        {
            if (step > 0)
            {
                auto const old = caloric();
                simulation.GetRadiationTransport().Advance(
                    simulation.GetPartContainer(), simulation.m_fields, (step - 1) * dt, dt);
                auto const source =
                    simulation.m_fields.get(FieldType::radiation_material_energy, 0)->sum(0, false);
                auto const realized = caloric() - old;
                if (std::abs(realized - source) >= 1.e-10_rt * initial)
                {
                    amrex::Print() << "Spiner material mismatch: step=" << step << " old=" << old
                                   << " oracle_delta=" << realized << " source=" << source
                                   << " initial=" << initial << " Te=" << te.min(0) << ','
                                   << te.max(0) << " material_charge=" << material_charge.min(0)
                                   << ',' << material_charge.max(0) << '\n';
                }
                AMREX_ALWAYS_ASSERT(std::abs(realized - source) < 1.e-10_rt * initial);
            }
            auto const material = caloric();
            auto const energy = radiation.sum(0, false);
            auto const residual = std::abs(material + energy - initial) / initial;
            worst = amrex::max(worst, residual);
            AMREX_ALWAYS_ASSERT(residual < 1.e-10_rt && radiation.min(0) >= 0);
            auto const minimum = te.min(0);
            auto const maximum = te.max(0);
            AMREX_ALWAYS_ASSERT(minimum >= oracle.MinimumTemperature() && maximum <= oracle.TMax());
            if (amrex::ParallelDescriptor::IOProcessor())
            {
                history << step << ',' << step * dt << ',' << energy << ',' << material << ','
                        << minimum << ',' << maximum << ',' << residual << '\n';
            }
        }
        AMREX_ALWAYS_ASSERT(caloric() - initial_material > 0.01_rt * initial_material);
        // Independent uniform BE root at the table's actual interpolated U(T).
        te.setVal(300);
        radiation.setVal(2.e9_rt * dx);
        auto u = [&] (amrex::Real t)
        { return 1000.0_rt * 1.e-4_rt * oracle.InternalEnergyFromDensityTemperature(1.0, t); };
        amrex::Real low = 300;
        amrex::Real high = 1000;
        for (int iteration = 0; iteration < 100; ++iteration)
        {
            auto const t = 0.5_rt * (low + high);
            auto const e = 2.e9_rt - (u(t) - u(300));
            auto const equation =
                e - 2.e9_rt +
                dt * PhysConst::c * 1000 * (e - 7.565733250280007e-16_rt * t * t * t * t);
            if (equation > 0)
            {
                low = t;
            }
            else
            {
                high = t;
            }
        }
        auto const expected = 0.5_rt * (low + high);
        simulation.GetRadiationTransport().Advance(simulation.GetPartContainer(),
                                                   simulation.m_fields, 100 * dt, dt);
        AMREX_ALWAYS_ASSERT(std::abs(te.max(0) - expected) < 1.e-8_rt * (expected - 300));
        AMREX_ALWAYS_ASSERT(std::abs(te.min(0) - expected) < 1.e-8_rt * (expected - 300));
        amrex::Print() << "100-stage single-material Spiner radiation: raw=" << worst
                       << " uniform heating root error="
                       << (te.max(0) - expected) / (expected - 300) << '\n';
        oracle.Finalize();
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
