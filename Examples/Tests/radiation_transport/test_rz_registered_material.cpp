/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

// Limited frozen-material / ideal-EOS qualification test driver in native RZ.
// Verifies 100-stage stationary implicit radiation-electron heating against
// independent nodal caloric quadrature (not calibrated tungsten or moving PIC).

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Radiation/RadiationTransport.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#ifdef WARPX_USE_SINGULARITY_EOS
#include <AMReX_BoxIterator.H>
#endif
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

using namespace amrex::literals;

namespace {
amrex::Real
compute_material_caloric_energy (const amrex::MultiFab& mf_rho,
                                 const amrex::MultiFab& mf_T,
                                 const amrex::MultiFab& mf_grid,
                                 const amrex::Geometry& geom) {
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    const amrex::Real dr = geom.CellSize(0);
    const amrex::Real dz = geom.CellSize(1);
    const amrex::Real prob_lo_r = geom.ProbLo(0);
    const int dom_lo_i = geom.Domain().smallEnd(0);
    const amrex::Real coef = 1.5 * PhysConst::kb / PhysConst::q_e;
    const amrex::Real pi = MathConst::pi;

    for (amrex::MFIter mfi(mf_grid); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& rho = mf_rho.const_array(mfi);
        const auto& T = mf_T.const_array(mfi);
        reduce_op.eval(
            bx, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                amrex::Real cell_energy = 0.0;
                const amrex::Real r0 = prob_lo_r + (i - dom_lo_i) * dr;
                for (int di = 0; di <= 1; ++di) {
                    const amrex::Real ra = r0 + di * dr * 0.5;
                    const amrex::Real rb = r0 + (di + 1) * dr * 0.5;
                    const amrex::Real corner_vol =
                        pi * (rb * rb - ra * ra) * dz * 0.5;
                    for (int dj = 0; dj <= 1; ++dj) {
                        const amrex::Real u = coef * rho(i + di, j + dj, k) *
                                              T(i + di, j + dj, k);
                        cell_energy += u * corner_vol;
                    }
                }
                return {cell_energy};
            });
    }
    amrex::Real total_energy = amrex::get<0>(reduce_data.value(reduce_op));
    amrex::ParallelDescriptor::ReduceRealSum(total_energy);
    return total_energy;
}
} // namespace

int
main (int argc, char* argv[]) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& s = WarpX::GetInstance();
        s.InitData();
        s.HybridPICPrepareElectronStateForDiagnostics();

        const auto& geom = s.Geom(0);
        auto* mf_T = s.m_fields.get(
            warpx::fields::FieldType::hybrid_electron_temperature_fp, 0);
        auto* mf_rho = s.m_fields.get(warpx::fields::FieldType::rho_fp, 0);
        auto* mf_rad = s.m_fields.get(
            warpx::fields::FieldType::radiation_diffusion_energy, 0);
        auto* mf_mat_src = s.m_fields.get(
            warpx::fields::FieldType::radiation_material_energy, 0);

        AMREX_ALWAYS_ASSERT(mf_T && mf_rho && mf_rad && mf_mat_src);
        const auto electron_eos = s.get_pointer_HybridPICModel()
                                      ->electronThermodynamicsExecutor();
        const bool ideal_gas = electron_eos.isIdealGas();
        AMREX_ALWAYS_ASSERT(ideal_gas || electron_eos.isSingularitySpiner());
#ifdef WARPX_USE_SINGULARITY_EOS
        using SpinerOracle = singularity::SpinerEOSDependsRhoT;
        std::unique_ptr<SpinerOracle> oracle;
        const amrex::MultiFab* mf_ni_charge = nullptr;
        if (!ideal_gas) {
            AMREX_ALWAYS_ASSERT(s.get_pointer_HybridPICModel()
                                    ->electronThermodynamicsNumMaterials() == 1);
            std::string table_file;
            amrex::ParmParse pp_eos("hybrid_pic_model");
            pp_eos.get("electron_eos_ions_table_file", table_file);
            oracle = std::make_unique<SpinerOracle>(table_file, 7400,
                                                    singularity::TableSplit::ElectronOnly);
            mf_ni_charge = s.m_fields.get("ni_charge_fp_ions", 0);
            AMREX_ALWAYS_ASSERT(oracle && mf_ni_charge && mf_ni_charge->nComp() == 1);
        }
#else
        AMREX_ALWAYS_ASSERT(ideal_gas);
#endif
        AMREX_ALWAYS_ASSERT(
            s.get_pointer_HybridPICModel()
                ->transportsElectronInternalEnergyAtRawDensity() == !ideal_gas);
        if (ideal_gas) {
            AMREX_ALWAYS_ASSERT(
                std::abs(electron_eos.m_gamma - 5.0 / 3.0) < 1.e-14);
        }
        AMREX_ALWAYS_ASSERT(mf_rad->nComp() == 2 && mf_rho->nComp() == 1);

        amrex::ParmParse pp_test("test");
        int steps = 100;
        amrex::Real dt = 1.0e-11;
        std::string ref_dir;
        pp_test.query("steps", steps);
        pp_test.query("dt", dt);
        pp_test.query("reference_directory", ref_dir);
        AMREX_ALWAYS_ASSERT(steps > 0 && dt > 0 && std::isfinite(dt));

        amrex::MultiFab rho_init(mf_rho->boxArray(), mf_rho->DistributionMap(),
                                 1, 0);
        amrex::MultiFab::Copy(rho_init, *mf_rho, 0, 0, 1, 0);

#ifdef WARPX_USE_SINGULARITY_EOS
        auto spiner_caloric = [&] (bool reference_floor) {
            amrex::Real total = 0.0;
            const amrex::Real dr = geom.CellSize(0), dz = geom.CellSize(1),
                              prob_lo_r = geom.ProbLo(0);
            const int dom_lo_i = geom.Domain().smallEnd(0);
            const int outer_i = geom.Domain().bigEnd(0) + 1;
            const auto radial_upper = geom.ProbHi(0);
            const auto axis_factor = s.verboncoeurAxisCorrection() ? 1.0/3.0 : 1.0/4.0;
            constexpr amrex::Real ion_mass = 183.84_rt * 1.66053906892e-27_rt;
            // Iterate physical cells; nodal boxes would double-count shared
            // nodes and introduce fictitious cells at the domain boundary.
            for (amrex::MFIter mfi(*mf_rad); mfi.isValid(); ++mfi) {
                const auto charge = mf_ni_charge->const_array(mfi);
                const auto T = mf_T->const_array(mfi);
                for (amrex::BoxIterator cell(mfi.validbox()); cell.ok();
                     ++cell) {
                    const int i = cell()[0], j = cell()[1];
                    const amrex::Real r0 = prob_lo_r + (i - dom_lo_i) * dr;
                    for (int di = 0; di <= 1; ++di) {
                        const amrex::Real ra = r0 + di * dr * 0.5;
                        const amrex::Real rb = r0 + (di + 1) * dr * 0.5;
                        amrex::Real corner_vol =
                            MathConst::pi * (rb * rb - ra * ra) * dz * 0.5;
                        // Partition the native finite-volume electron inventory
                        // between adjacent physical cells. This independent
                        // measure includes the deposition-axis correction and
                        // reflective wall half volumes, irrespective of EOS.
                        const int node = i + di;
                        const auto r_node = prob_lo_r + (node - dom_lo_i)*dr;
                        const auto node_lo = std::max(prob_lo_r, r_node - dr/2);
                        const auto node_hi = std::min(radial_upper, r_node + dr/2);
                        const auto geometric_volume = MathConst::pi
                            * (node_hi*node_hi - node_lo*node_lo)*dz;
                        auto native_volume = r_node == 0
                            ? MathConst::pi*dr*dr*dz*axis_factor
                            : 2*MathConst::pi*r_node*dr*dz;
                        if (node == outer_i || (node == dom_lo_i && r_node > 0)) {
                            native_volume *= 0.5;
                        }
                        corner_vol *= native_volume / geometric_volume;
                        for (int dj = 0; dj <= 1; ++dj) {
                            const amrex::Real density = charge(i + di, j + dj, 0)
                                * ion_mass / PhysConst::q_e;
                            AMREX_ALWAYS_ASSERT(density * 1.e-3 >= oracle->MinimumDensity()
                                && density * 1.e-3 <= oracle->MaximumDensity());
                            const amrex::Real temperature = reference_floor
                                ? oracle->MinimumTemperature()
                                : T(i + di, j + dj, 0);
                            total += density * 1.0e-4 *
                                oracle->InternalEnergyFromDensityTemperature(
                                    density * 1.0e-3, temperature) *
                                corner_vol;
                        }
                    }
                }
            }
            amrex::ParallelDescriptor::ReduceRealSum(total);
            return total;
        };
#endif
        auto material_caloric = [&] () {
#ifdef WARPX_USE_SINGULARITY_EOS
            if (!ideal_gas) { return spiner_caloric(false); }
#endif
            return compute_material_caloric_energy(*mf_rho, *mf_T, *mf_rad, geom);
        };
        amrex::Real time = 0.0;
        amrex::Real mat_energy = material_caloric();
        amrex::Real rad_energy = mf_rad->sum(0, false) + mf_rad->sum(1, false);
        const amrex::Real init_mat_energy = mat_energy;
        const amrex::Real conserved_initial = init_mat_energy + rad_energy;
        const amrex::Real energy_norm = std::abs(init_mat_energy) + rad_energy;
        AMREX_ALWAYS_ASSERT(std::isfinite(energy_norm) && energy_norm > 0);
#ifdef WARPX_USE_SINGULARITY_EOS
        const amrex::Real init_thermal_energy =
            ideal_gas ? 0.0 : init_mat_energy - spiner_caloric(true);
        AMREX_ALWAYS_ASSERT(ideal_gas || init_thermal_energy > 0);
#endif

        amrex::Real T_min = mf_T->min(0);
        amrex::Real T_max = mf_T->max(0);
#ifdef WARPX_USE_SINGULARITY_EOS
        AMREX_ALWAYS_ASSERT(ideal_gas || (T_min >= oracle->MinimumTemperature()
                                          && T_max <= oracle->TMax()));
#endif
        amrex::Print() << "Electron EOS backend: "
                       << (ideal_gas ? "ideal_gas" : "singularity_spiner")
                       << '\n';

        amrex::Print() << "Initial nodal charge density range: " << mf_rho->min(0)
                       << " .. " << mf_rho->max(0) << "; temperature range: "
                       << T_min << " .. " << T_max << '\n';

        std::ofstream csv;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            csv.open("trajectory.csv");
            csv.precision(16);
            csv << "step,time,material_energy,radiation_energy,Tmin,Tmax\n";
            csv << 0 << "," << time << "," << mat_energy << "," << rad_energy
                << "," << T_min << "," << T_max << "\n";
            csv.flush();
        }

        amrex::Real max_defect = 0.0;
        amrex::Real max_source_error = 0.0;
        for (int step = 1; step <= steps; ++step) {
            const amrex::Real old_mat_energy = mat_energy;
            s.GetRadiationTransport().Advance(s.GetPartContainer(), s.m_fields,
                                              time, dt);
            time += dt;

            mat_energy =
                material_caloric();
            rad_energy = mf_rad->sum(0, false) + mf_rad->sum(1, false);

            const amrex::Real total_energy = mat_energy + rad_energy;
            const amrex::Real defect =
                std::abs(total_energy - conserved_initial) / energy_norm;
            max_defect = std::max(max_defect, defect);
            if (!(defect < 1.0e-10)) {
                amrex::Print() << "Native caloric inventory mismatch: step=" << step
                    << " relative_total_defect=" << defect << '\n';
            }
            AMREX_ALWAYS_ASSERT(defect < 1.0e-10);

            const amrex::Real delta_caloric = mat_energy - old_mat_energy;
            const amrex::Real source_sum = mf_mat_src->sum(0, false);
            max_source_error = std::max(max_source_error,
                std::abs(delta_caloric - source_sum) / energy_norm);
            AMREX_ALWAYS_ASSERT(max_source_error < 1.0e-10);

            T_min = mf_T->min(0);
            T_max = mf_T->max(0);
            const amrex::Real E0_min = mf_rad->min(0);
            const amrex::Real E1_min = mf_rad->min(1);
            AMREX_ALWAYS_ASSERT(mf_T->is_finite() && mf_rad->is_finite());
            AMREX_ALWAYS_ASSERT(std::isfinite(T_min) && T_min > 0.0 &&
                                std::isfinite(T_max));
            AMREX_ALWAYS_ASSERT(E0_min >= 0.0 && E1_min >= 0.0);
#ifdef WARPX_USE_SINGULARITY_EOS
            if (!ideal_gas) {
                AMREX_ALWAYS_ASSERT(
                    T_min >= oracle->MinimumTemperature() &&
                    T_max <= oracle->TMax());
            }
#endif

            amrex::MultiFab diff_rho(mf_rho->boxArray(),
                                     mf_rho->DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(diff_rho, *mf_rho, 0, 0, 1, 0);
            amrex::MultiFab::Subtract(diff_rho, rho_init, 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT(diff_rho.norm0(0) == 0.0);

            if (amrex::ParallelDescriptor::IOProcessor()) {
                csv << step << "," << time << "," << mat_energy << ","
                    << rad_energy << "," << T_min << "," << T_max << "\n";
                csv.flush();
            }
        }

        if (ideal_gas) {
            AMREX_ALWAYS_ASSERT(std::abs(mat_energy - init_mat_energy) >
                                0.001 * std::abs(init_mat_energy));
        }
#ifdef WARPX_USE_SINGULARITY_EOS
        else {
            AMREX_ALWAYS_ASSERT(std::abs(mat_energy - init_mat_energy) >
                                0.05 * init_thermal_energy);
        }
#endif

        mf_T->FillBoundary(geom.periodicity());
        mf_rho->FillBoundary(geom.periodicity());
        mf_rad->FillBoundary(geom.periodicity());
        amrex::VisMF::Write(*mf_T, "state_T");
        amrex::VisMF::Write(*mf_rho, "state_rho");
        amrex::VisMF::Write(*mf_rad, "state_radiation");

        amrex::Real max_reference_error = 0.0;
        if (!ref_dir.empty()) {
            auto check_ref = [&] (const amrex::MultiFab& mf,
                                  const std::string& name) {
                const std::string path = ref_dir + "/" + name;
                amrex::MultiFab ref_mf;
                amrex::VisMF::Read(ref_mf, path);
                amrex::MultiFab aligned_mf(mf.boxArray(), mf.DistributionMap(),
                                           mf.nComp(), 0);
                aligned_mf.ParallelCopy(ref_mf, 0, 0, mf.nComp(), 0, 0,
                                        geom.periodicity());
                for (int c = 0; c < mf.nComp(); ++c) {
                    amrex::MultiFab diff(mf.boxArray(), mf.DistributionMap(), 1,
                                         0);
                    amrex::MultiFab::Copy(diff, mf, c, 0, 1, 0);
                    amrex::MultiFab::Subtract(diff, aligned_mf, c, 0, 1, 0);
                    const amrex::Real err = diff.norm0(0);
                    const amrex::Real scale =
                        std::max({mf.norm0(c), aligned_mf.norm0(c),
                                  std::numeric_limits<amrex::Real>::min()});
                    max_reference_error =
                        std::max(max_reference_error, err / scale);
                    AMREX_ALWAYS_ASSERT(max_reference_error < 1.0e-11);
                }
            };
            check_ref(*mf_T, "state_T");
            check_ref(*mf_rho, "state_rho");
            check_ref(*mf_rad, "state_radiation");
        }

        amrex::Print() << "RZ registered-material test passed. Max defect: "
                       << max_defect << ", max source error: "
                       << max_source_error << ", max reference error: "
                       << max_reference_error << "\n";
#ifdef WARPX_USE_SINGULARITY_EOS
        if (oracle) { oracle->Finalize(); }
        oracle.reset();
#endif
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
    return 0;
}
