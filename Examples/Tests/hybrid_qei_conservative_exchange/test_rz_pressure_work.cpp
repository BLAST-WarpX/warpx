/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/RZPressureWork.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

using namespace amrex::literals;

namespace
{
void
operator_check (bool const periodic, amrex::Real const axis_factor)
{
    constexpr int nr = 8, nz = 12;
    constexpr amrex::Real dr = 0.17_rt, dz = 0.23_rt;
    amrex::Box const box(amrex::IntVect(0, 0), amrex::IntVect(nr, nz),
                         amrex::IntVect::TheNodeVector());
    amrex::FArrayBox state(amrex::grow(box, 1), 2);
    amrex::FArrayBox radial(amrex::grow(box, 1), 1);
    amrex::FArrayBox axial(amrex::grow(box, 1), 1);
    amrex::FArrayBox output(box, 3);
    auto const s = state.array(), r = radial.array(), z = axial.array();
    amrex::ParallelFor(state.box(),
                       [=] AMREX_GPU_DEVICE(int i, int j, int k)
                       {
                           int const jj = periodic ? (j + nz) % nz : j;
                           s(i, j, k, 0) = 3.0_rt + std::sin(0.41_rt * i + 0.73_rt * jj);
                           s(i, j, k, 1) = 0.7_rt + 0.01_rt * i + 0.02_rt * jj;
                           r(i, j, k) = 1.2_rt + std::cos(0.33_rt * i - 0.41_rt * jj);
                           z(i, j, k) = -0.7_rt + std::sin(0.19_rt * i + 0.31_rt * jj);
                       });
    auto const out = output.array();
    auto const sc = state.const_array(), rc = radial.const_array(), zc = axial.const_array();
    amrex::ParallelFor(
        box,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Real volume = MathConst::pi * dr * dr * dz * (i == 0 ? axis_factor : 2.0_rt * i);
            if (i == nr)
            {
                volume *= 0.5_rt;
            }
            if (!periodic && (j == 0 || j == nz))
            {
                volume *= 0.5_rt;
            }
            out(i, j, k, 0) =
                warpx::hybrid::rzPressureGradient(sc, i, j, 0, 0, nr, 0, nz, periodic, 1.0_rt / dr);
            out(i, j, k, 1) =
                warpx::hybrid::rzPressureGradient(sc, i, j, 2, 0, nr, 0, nz, periodic, 1.0_rt / dz);
            out(i, j, k, 2) = warpx::hybrid::rzPressureWorkDivergence(
                rc, zc, sc, i, j, 0, nr, 0, nz, periodic, 1.0_rt / dr, 1.0_rt / dz, volume);
        });
    amrex::Gpu::HostVector<amrex::Real> values(output.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, output.dataPtr(), output.dataPtr() + output.size(),
                     values.begin());

    // Independent host sparse-matrix assembly. Construct each row of G
    // then scatter -G^T onto its pressure columns; do not call the runtime
    // divergence or metric helpers in this reference.
    int const height = periodic ? nz : nz + 1;
    constexpr int stride = (nr + 1) * (nz + 1);
    std::vector<double> transpose(stride, 0);
    auto const index = [=] (int i, int j) { return i + (nr + 1) * (periodic ? (j + nz) % nz : j); };
    auto const pressure = [] (int i, int j) { return 3.0 + std::sin(0.41 * i + 0.73 * j); };
    double particle_power = 0, electron_power = 0, scale = 0;
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i <= nr; ++i)
        {
            int const n = index(i, j);
            double const inverse_rho = 0.7 + 0.01 * i + 0.02 * j;
            std::array<double, 2> const current{1.2 + std::cos(0.33 * i - 0.41 * j),
                                                -0.7 + std::sin(0.19 * i + 0.31 * j)};
            for (int d = 0; d < 2; ++d)
            {
                bool const active =
                    i != nr && (periodic || (j != 0 && j != nz)) && (d != 0 || i != 0);
                double gradient = 0;
                if (active)
                {
                    int const il = i - (d == 0), ih = i + (d == 0);
                    int const jl = periodic ? (j - (d == 1) + nz) % nz : j - (d == 1);
                    int const jh = periodic ? (j + (d == 1)) % nz : j + (d == 1);
                    double const coefficient = 0.5 / (d == 0 ? dr : dz);
                    gradient = coefficient * (pressure(ih, jh) - pressure(il, jl));
                    transpose[index(ih, jh)] -= coefficient * current[d] * inverse_rho;
                    transpose[index(il, jl)] += coefficient * current[d] * inverse_rho;
                }
                AMREX_ALWAYS_ASSERT(std::abs(values[n + d * stride] - gradient) < 2.e-13);
                particle_power -= gradient * current[d] * inverse_rho;
                scale += std::abs(gradient * current[d] * inverse_rho);
            }
        }
    }
    double maximum_error = 0;
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i <= nr; ++i)
        {
            int const n = index(i, j);
            double volume = MathConst::pi * dr * dr * dz * (i == 0 ? axis_factor : 2.0 * i);
            if (i == nr)
            {
                volume *= 0.5;
            }
            if (!periodic && (j == 0 || j == nz))
            {
                volume *= 0.5;
            }
            double const actual = values[n + 2 * stride] * volume;
            maximum_error = std::max(maximum_error, std::abs(actual - transpose[n]));
            electron_power -= values[n + 2 * stride] * pressure(i, j) * volume;
        }
    }
    AMREX_ALWAYS_ASSERT(maximum_error < 2.e-13);
    AMREX_ALWAYS_ASSERT(std::abs(particle_power + electron_power) <
                        512 * std::numeric_limits<double>::epsilon() * scale);
    amrex::Print() << "RZ G/-W^-1 G^T: periodic_z=" << periodic << " axis_factor=" << axis_factor
                   << " max_error=" << maximum_error
                   << " power_residual=" << std::abs(particle_power + electron_power) / scale
                   << '\n';
}

std::array<amrex::Real, 2>
inventory (WarpX& simulation)
{
    using warpx::fields::FieldType;
    auto const& geom = simulation.Geom(0);
    auto const dx = geom.CellSizeArray();
    auto const hi = amrex::ubound(geom.Domain());
    bool const periodic = geom.isPeriodic(1);
    amrex::Real const axis_factor =
        simulation.verboncoeurAxisCorrection() ? 1.0_rt / 3.0_rt : 1.0_rt / 4.0_rt;
    auto const& te = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& rho = *simulation.m_fields.get(FieldType::rho_fp, 0);
    auto const owner = te.OwnerMask(geom.periodicity());
    amrex::MultiFab energy(te.boxArray(), te.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi)
    {
        auto const out = energy.array(mfi);
        auto const t = te.const_array(mfi);
        auto const charge = rho.const_array(mfi);
        auto const owned = owner->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               amrex::Real volume = MathConst::pi * dx[0] * dx[0] * dx[1] *
                                                    (i == 0 ? axis_factor : 2.0_rt * i);
                               if (i == hi.x + 1)
                               {
                                   volume *= 0.5_rt;
                               }
                               if (!periodic && (j == 0 || j == hi.y + 1))
                               {
                                   volume *= 0.5_rt;
                               }
                               out(i, j, k) = owned(i, j, k)
                                                  ? 1.5_rt * charge(i, j, k) / PhysConst::q_e *
                                                        PhysConst::kb * t(i, j, k) * volume
                                                  : 0.0_rt;
                           });
    }
    return {simulation.GetPartContainer().GetParticleContainerFromName("ions").sumParticleEnergy(),
            energy.sum(0)};
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        for (bool periodic : {false, true})
        {
            for (amrex::Real axis : {1.0_rt / 3.0_rt, 1.0_rt / 4.0_rt})
            {
                operator_check(periodic, axis);
            }
        }
        bool operator_only = false;
        amrex::ParmParse("test").query("operator_only", operator_only);
        if (!operator_only)
        {
            auto& simulation = WarpX::GetInstance();
            simulation.InitData();
            bool inject_magnetic_current = false;
            amrex::ParmParse("test").query("inject_magnetic_current", inject_magnetic_current);
            if (inject_magnetic_current)
            {
                auto& bt = *simulation.m_fields.get(warpx::fields::FieldType::Bfield_fp,
                                                    ablastr::fields::Direction{1}, 0);
                for (amrex::MFIter mfi(bt); mfi.isValid(); ++mfi)
                {
                    auto const field = bt.array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                       { field(i, j, k) = 0.01_rt * i; });
                }
            }
            simulation.HybridPICPrepareElectronStateForDiagnostics();
            auto const initial = inventory(simulation);
            simulation.Evolve();
            auto const final = inventory(simulation);
            auto const ion_work = final[0] - initial[0];
            auto const electron_work = final[1] - initial[1];
            auto const energy = initial[0] + initial[1];
            amrex::Print() << "RZ native pressure work: ion=" << ion_work
                           << " electron=" << electron_work << " initial=" << energy
                           << " residual=" << (ion_work + electron_work) / energy << '\n';
            AMREX_ALWAYS_ASSERT(std::abs(ion_work) > 1.e-4_rt * energy);
            AMREX_ALWAYS_ASSERT(std::abs(ion_work + electron_work) < 1.e-10_rt * energy);
            bool require_heating = false;
            amrex::ParmParse("test").query("require_electron_heating", require_heating);
            if (require_heating)
            {
                AMREX_ALWAYS_ASSERT(electron_work > 0.01_rt * energy);
            }
        }
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
