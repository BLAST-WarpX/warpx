/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QeiThermalSupport.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace
{
    constexpr int exchange_steps = 3;
    constexpr amrex::Real default_exchange_dt = 1.e-11_rt;
    constexpr int n_corners = 1 << AMREX_SPACEDIM;

    std::vector<std::array<int, 3>> SupportCells ()
    {
#if defined(WARPX_DIM_1D_Z)
        return {{0, 0, 0}, {1, 0, 0}, {4, 0, 0}, {6, 0, 0}};
#elif defined(WARPX_DIM_XZ)
        return {{0, 0, 0}, {1, 1, 0}, {3, 2, 0}, {2, 3, 0}};
#elif defined(WARPX_DIM_3D)
        return {{0, 0, 0}, {1, 1, 1}, {3, 2, 1}, {2, 3, 2}, {1, 1, 3}};
#else
        std::abort();
#endif
    }

    std::array<int, 3> SingletonCell ()
    {
#if defined(WARPX_DIM_1D_Z)
        return {2, 0, 0};
#elif defined(WARPX_DIM_XZ)
        return {4, 4, 0};
#elif defined(WARPX_DIM_3D)
        return {3, 3, 3};
#else
        std::abort();
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real CartesianNodeVolume (
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
        amrex::Dim3 const lo, amrex::Dim3 const hi,
        amrex::GpuArray<int, 3> const& periodic)
    {
        amrex::Real volume =
            periodic[0] || (i != lo.x && i != hi.x + 1) ? dx[0] : 0.5_rt * dx[0];
#if AMREX_SPACEDIM >= 2
        volume *= periodic[1] || (j != lo.y && j != hi.y + 1) ? dx[1] : 0.5_rt * dx[1];
#else
        amrex::ignore_unused(j);
#endif
#if AMREX_SPACEDIM == 3
        volume *= periodic[2] || (k != lo.z && k != hi.z + 1) ? dx[2] : 0.5_rt * dx[2];
#else
        amrex::ignore_unused(k);
#endif
        return volume;
    }

    amrex::Real CartesianCellVolume (
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx)
    {
        return AMREX_D_TERM(dx[0], * dx[1], * dx[2]);
    }

    bool CellIsAddressable (
        int const i, int const j, int const k,
        amrex::Dim3 const lo, amrex::Dim3 const hi,
        amrex::GpuArray<int, 3> const& periodic)
    {
        bool const addressable =
            (periodic[0] || (i >= lo.x && i <= hi.x))
#if AMREX_SPACEDIM >= 2
            && (periodic[1] || (j >= lo.y && j <= hi.y))
#endif
#if AMREX_SPACEDIM == 3
            && (periodic[2] || (k >= lo.z && k <= hi.z))
#endif
            ;
        amrex::ignore_unused(j, k);
        return addressable;
    }

    std::array<int, 3> WrapCell (
        int i, int j, int k,
        amrex::Dim3 const lo, amrex::Dim3 const hi,
        amrex::GpuArray<int, 3> const& periodic)
    {
        auto wrap = [] (int index, int const low, int const high, int const is_periodic) {
            if (!is_periodic) { return index; }
            int const length = high - low + 1;
            while (index < low) { index += length; }
            while (index > high) { index -= length; }
            return index;
        };
        std::array<int, 3> cell{
            wrap(i, lo.x, hi.x, periodic[0]), 0, 0};
#if AMREX_SPACEDIM >= 2
        cell[1] = wrap(j, lo.y, hi.y, periodic[1]);
#else
        amrex::ignore_unused(j);
#endif
#if AMREX_SPACEDIM == 3
        cell[2] = wrap(k, lo.z, hi.z, periodic[2]);
#else
        amrex::ignore_unused(k);
#endif
        return cell;
    }

    void FillUniform (amrex::MultiFab& field, amrex::Real const value,
                      amrex::Geometry const& geometry)
    {
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            auto const array = field.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (
                                                       int i, int j, int k) noexcept {
                array(i, j, k) = value;
            });
        }
        field.FillBoundary(field.nGrowVect(), geometry.periodicity());
    }

    struct ParticleRecord
    {
        std::uint64_t id = 0;
        std::array<int, 3> cell{};
        amrex::ParticleReal weight = 0.0_prt;
        std::array<amrex::ParticleReal, 3> velocity{};
    };

    bool operator== (ParticleRecord const& a, ParticleRecord const& b)
    {
        return a.id == b.id && a.cell == b.cell && a.weight == b.weight
            && a.velocity == b.velocity;
    }

    bool RecordsUnchanged (
        std::vector<ParticleRecord> const& before,
        std::vector<ParticleRecord> const& after)
    {
        if (before.size() != after.size()) { return false; }
        std::map<std::uint64_t, ParticleRecord> old_records;
        std::map<std::uint64_t, ParticleRecord> new_records;
        for (auto const& record : before) { old_records.emplace(record.id, record); }
        for (auto const& record : after) { new_records.emplace(record.id, record); }
        return old_records.size() == before.size() && old_records == new_records;
    }

    std::vector<ParticleRecord> ParticleSnapshot (
        WarpXParticleContainer& pc, amrex::Geometry const& geometry)
    {
        std::vector<ParticleRecord> result;
        auto const lower = geometry.ProbLoArray();
        auto const inverse_dx = geometry.InvCellSizeArray();
        for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
            long const np = pti.numParticles();
            auto const& soa = pti.GetParticleTile().GetStructOfArrays();
            amrex::Gpu::HostVector<std::uint64_t> ids(np);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
                             soa.GetIdCPUData().begin() + np, ids.begin());
            std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, AMREX_SPACEDIM> positions;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                auto const& values = pti.GetAttribs(d);
                positions[d].resize(np);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                                 values.begin() + np, positions[d].begin());
            }
            amrex::Gpu::HostVector<amrex::ParticleReal> weights(np);
            amrex::Gpu::HostVector<amrex::ParticleReal> ux(np);
            amrex::Gpu::HostVector<amrex::ParticleReal> uy(np);
            amrex::Gpu::HostVector<amrex::ParticleReal> uz(np);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, pti.GetAttribs(PIdx::w).begin(),
                             pti.GetAttribs(PIdx::w).begin() + np, weights.begin());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, pti.GetAttribs(PIdx::ux).begin(),
                             pti.GetAttribs(PIdx::ux).begin() + np, ux.begin());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, pti.GetAttribs(PIdx::uy).begin(),
                             pti.GetAttribs(PIdx::uy).begin() + np, uy.begin());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, pti.GetAttribs(PIdx::uz).begin(),
                             pti.GetAttribs(PIdx::uz).begin() + np, uz.begin());
            for (long ip = 0; ip < np; ++ip) {
                if (!amrex::ParticleIDWrapper{ids[ip]}.is_valid()) { continue; }
                std::array<int, 3> cell{0, 0, 0};
                cell[0] = static_cast<int>(amrex::Math::floor(
                    (positions[0][ip] - lower[0]) * inverse_dx[0]));
#if AMREX_SPACEDIM >= 2
                cell[1] = static_cast<int>(amrex::Math::floor(
                    (positions[1][ip] - lower[1]) * inverse_dx[1]));
#endif
#if AMREX_SPACEDIM == 3
                cell[2] = static_cast<int>(amrex::Math::floor(
                    (positions[2][ip] - lower[2]) * inverse_dx[2]));
#endif
                result.push_back({ids[ip], cell, weights[ip],
                                  {ux[ip], uy[ip], uz[ip]}});
            }
        }
        return result;
    }

    amrex::Real KineticEnergy (
        WarpXParticleContainer& pc, amrex::Real const mass)
    {
        amrex::ReduceOps<amrex::ReduceOpSum> ops;
        amrex::ReduceData<amrex::Real> data(ops);
        using Tuple = typename decltype(data)::Type;
        for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
            auto const* weight = pti.GetAttribs(PIdx::w).dataPtr();
            auto const* ux = pti.GetAttribs(PIdx::ux).dataPtr();
            auto const* uy = pti.GetAttribs(PIdx::uy).dataPtr();
            auto const* uz = pti.GetAttribs(PIdx::uz).dataPtr();
            ops.eval(pti.numParticles(), data,
                     [=] AMREX_GPU_DEVICE (int const p) -> Tuple
                     {
                         return {0.5_rt * mass * static_cast<amrex::Real>(weight[p]) *
                                 static_cast<amrex::Real>(
                                     ux[p] * ux[p] + uy[p] * uy[p] + uz[p] * uz[p])};
                     });
        }
        auto result = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealSum(result);
        return result;
    }

    void CheckCellMomentumAndSingleton (
        std::vector<ParticleRecord> const& before,
        std::vector<ParticleRecord> const& after,
        std::array<int, 3> const& singleton)
    {
        std::map<std::uint64_t, ParticleRecord const*> old_records;
        std::map<std::uint64_t, ParticleRecord const*> new_records;
        for (auto const& record : before) { old_records.emplace(record.id, &record); }
        for (auto const& record : after) { new_records.emplace(record.id, &record); }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(old_records.size() == new_records.size(),
                                         "Resolved Qei changed the particle census.");
        std::map<std::array<int, 3>, std::array<amrex::Real, 3>> old_momentum;
        std::map<std::array<int, 3>, std::array<amrex::Real, 3>> new_momentum;
        amrex::Real scale = 1.0_rt;
        for (auto const& [id, old_record] : old_records) {
            auto const new_record = new_records.at(id);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(old_record->cell == new_record->cell,
                "The native exchange moved a particle between NGP cells.");
            auto& old_sum = old_momentum[old_record->cell];
            auto& new_sum = new_momentum[new_record->cell];
            for (int d = 0; d < 3; ++d) {
                old_sum[d] += static_cast<amrex::Real>(old_record->weight) *
                              static_cast<amrex::Real>(old_record->velocity[d]);
                new_sum[d] += static_cast<amrex::Real>(new_record->weight) *
                              static_cast<amrex::Real>(new_record->velocity[d]);
                scale = std::max(scale, std::abs(old_sum[d]));
            }
        }
        amrex::Real maximum_error = 0.0_rt;
        for (auto const& [cell, momentum] : new_momentum) {
            auto const expected = old_momentum.at(cell);
            for (int d = 0; d < 3; ++d) {
                maximum_error = std::max(maximum_error,
                    std::abs(momentum[d] - expected[d]));
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(maximum_error);
        amrex::ParallelDescriptor::ReduceRealMax(scale);
        amrex::Print() << "Cartesian Qei unscaled sum(w*u) absolute error="
                       << maximum_error << '\n';
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            maximum_error < 1.e-12_rt * scale,
            "Resolved Qei exchange changed cell ion momentum.");
        bool singleton_unchanged = true;
        for (auto const& record : after) {
            if (record.cell != singleton) { continue; }
            singleton_unchanged = singleton_unchanged && record == *old_records.at(record.id);
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(singleton_unchanged);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(singleton_unchanged,
            "A singleton cell acquired a resolved thermal exchange.");
    }

    struct ExchangeOracle
    {
        amrex::Dim3 lo{};
        amrex::Dim3 hi{};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
        amrex::GpuArray<int, 3> periodic{};
        int nnode_x = 0;
        int nnode_y = 1;
        int nnode_z = 1;
        std::vector<amrex::Real> temperature;
        std::set<std::array<int, 3>> support;
        std::map<std::array<int, 3>, amrex::Real> light_ion_temperature;
        std::map<std::array<int, 3>, amrex::Real> heavy_ion_temperature;

        ExchangeOracle (
            amrex::Geometry const& geometry,
            std::vector<std::array<int, 3>> const& cells,
            amrex::Real const initial_temperature)
        {
            auto const domain = geometry.Domain();
            lo = amrex::lbound(domain);
            hi = amrex::ubound(domain);
            dx = geometry.CellSizeArray();
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                periodic[d] = geometry.isPeriodic(d) ? 1 : 0;
            }
            nnode_x = hi.x - lo.x + 2;
#if AMREX_SPACEDIM >= 2
            nnode_y = hi.y - lo.y + 2;
#endif
#if AMREX_SPACEDIM == 3
            nnode_z = hi.z - lo.z + 2;
#endif
            temperature.assign(static_cast<std::size_t>(nnode_x * nnode_y * nnode_z),
                               initial_temperature);
            support.insert(cells.begin(), cells.end());
        }

        void SetIonTemperatures (amrex::Real const light, amrex::Real const heavy)
        {
            for (auto const& cell : support) {
                light_ion_temperature[cell] = light;
                heavy_ion_temperature[cell] = heavy;
            }
        }

        amrex::Real& At (int const i, int const j, int const k)
        {
            int const ii = i - lo.x;
            int const jj = AMREX_SPACEDIM >= 2 ? j - lo.y : 0;
            int const kk = AMREX_SPACEDIM == 3 ? k - lo.z : 0;
            return temperature[static_cast<std::size_t>(
                ii + nnode_x * (jj + nnode_y * kk))];
        }

        amrex::Real ExchangeSpecies (
            amrex::Real const weight,
            std::map<std::array<int, 3>, amrex::Real>& ion_temperature,
            amrex::Real const electron_number_density,
            amrex::Real const rate, amrex::Real const dt)
        {
            amrex::Real ion_energy_change = 0.0_rt;
            amrex::Real const fraction = 1.0_rt / static_cast<amrex::Real>(n_corners);
            amrex::Real const fragment_number = fraction * 2.0_rt * weight;
            amrex::Real const cell_number = 2.0_rt * weight;
            std::map<std::array<int, 3>, amrex::Real> cell_energy_change;
            int const i_last = hi.x + (periodic[0] ? 0 : 1);
#if AMREX_SPACEDIM >= 2
            int const j_last = hi.y + (periodic[1] ? 0 : 1);
#else
            int const j_last = lo.y;
#endif
#if AMREX_SPACEDIM == 3
            int const k_last = hi.z + (periodic[2] ? 0 : 1);
#else
            int const k_last = lo.z;
#endif
            for (int k = lo.z; k <= k_last; ++k) {
                for (int j = lo.y; j <= j_last; ++j) {
                    for (int i = lo.x; i <= i_last; ++i) {
                        amrex::Real const volume = CartesianNodeVolume(
                            i, j, k, dx, lo, hi, periodic);
                        if (!(volume > 0.0_rt)) { continue; }
                        for (int dk = 0; dk < (AMREX_SPACEDIM == 3 ? 2 : 1); ++dk) {
                            for (int dj = 0; dj < (AMREX_SPACEDIM >= 2 ? 2 : 1); ++dj) {
                                for (int di = 0; di < 2; ++di) {
                                    int const ci = i - di;
                                    int const cj = j - dj;
                                    int const ck = k - dk;
                                    if (!CellIsAddressable(ci, cj, ck, lo, hi, periodic)) {
                                        continue;
                                    }
                                    auto const cell = WrapCell(ci, cj, ck, lo, hi, periodic);
                                    if (support.find(cell) == support.end()) { continue; }
                                    amrex::Real const fragment_density = fragment_number / volume;
                                    amrex::Real const electron_capacity =
                                        1.5_rt * electron_number_density * PhysConst::kb;
                                    amrex::Real const ion_capacity =
                                        1.5_rt * fragment_density * PhysConst::kb;
                                    amrex::Real const conductance =
                                        3.0_rt * fragment_density * PhysConst::kb * rate;
                                    if (conductance == 0.0_rt) { continue; }
                                    amrex::Real const coupled_rate = conductance *
                                        (1.0_rt / electron_capacity + 1.0_rt / ion_capacity);
                                    amrex::Real const exchange_fraction =
                                        -std::expm1(-coupled_rate * dt);
                                    amrex::Real const series_capacity = electron_capacity *
                                        ion_capacity / (electron_capacity + ion_capacity);
                                    amrex::Real const electron_change_density = series_capacity *
                                        (ion_temperature.at(cell) - At(i, j, k)) * exchange_fraction;
                                    At(i, j, k) += electron_change_density / electron_capacity;
                                    amrex::Real const ion_change = -electron_change_density * volume;
                                    ion_energy_change += ion_change;
                                    cell_energy_change[cell] += ion_change;
                                }
                            }
                        }
                    }
                }
            }
            amrex::Real const cell_capacity = 1.5_rt * cell_number * PhysConst::kb;
            for (auto const& [cell, change] : cell_energy_change) {
                ion_temperature.at(cell) += change / cell_capacity;
            }
            return ion_energy_change;
        }
    };

    amrex::Real ElectronEnergy (
        amrex::MultiFab const& te, amrex::MultiFab const& rho,
        amrex::Geometry const& geometry)
    {
        auto const dx = geometry.CellSizeArray();
        auto const lo = amrex::lbound(geometry.Domain());
        auto const hi = amrex::ubound(geometry.Domain());
        amrex::GpuArray<int, 3> periodic{0, 0, 0};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            periodic[d] = geometry.isPeriodic(d) ? 1 : 0;
        }
        auto owner = te.OwnerMask(geometry.periodicity());
        amrex::MultiFab energy(te.boxArray(), te.DistributionMap(), 1, 0);
        for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
            auto const output = energy.array(mfi);
            auto const temperature = te.const_array(mfi);
            auto const charge = rho.const_array(mfi);
            auto const owned = owner->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (
                                                       int i, int j, int k) {
                amrex::Real const volume = CartesianNodeVolume(
                    i, j, k, dx, lo, hi, periodic);
                output(i, j, k) = owned(i, j, k)
                    ? 1.5_rt * charge(i, j, k) / PhysConst::q_e * PhysConst::kb *
                          temperature(i, j, k) * volume
                    : 0.0_rt;
            });
        }
        return energy.sum(0);
    }
}

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        std::array<double, 4> moments{};
        double component_cross = 0.0;
        double counter_cross = 0.0;
        constexpr std::uint64_t samples = 1000000;
        for (std::uint64_t particle = 0; particle < samples; ++particle) {
            auto const x = warpx::hybrid::qeiNormal(1, 0, 0, particle, 0);
            auto const y = warpx::hybrid::qeiNormal(1, 0, 0, particle, 1);
            auto const next = warpx::hybrid::qeiNormal(1, 1, 0, particle, 0);
            AMREX_ALWAYS_ASSERT(std::isfinite(x) && std::isfinite(y) && std::isfinite(next));
            auto power = x;
            for (auto& moment : moments) {
                moment += power / static_cast<double>(samples);
                power *= x;
            }
            component_cross += x * y / static_cast<double>(samples);
            counter_cross += x * next / static_cast<double>(samples);
        }
        AMREX_ALWAYS_ASSERT(std::abs(moments[0]) < 0.01);
        AMREX_ALWAYS_ASSERT(std::abs(moments[1] - 1.0) < 0.02);
        AMREX_ALWAYS_ASSERT(std::abs(moments[2]) < 0.04);
        AMREX_ALWAYS_ASSERT(std::abs(moments[3] - 3.0) < 0.1);
        AMREX_ALWAYS_ASSERT(std::abs(component_cross) < 0.01);
        AMREX_ALWAYS_ASSERT(std::abs(counter_cross) < 0.01);

        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        simulation.HybridPICPrepareElectronStateForDiagnostics();
        auto& model = *simulation.get_pointer_HybridPICModel();
        auto& mypc = simulation.GetPartContainer();
        auto const& geom = simulation.Geom(0);
        auto const dx = geom.CellSizeArray();
        amrex::Real const cell_volume = CartesianCellVolume(dx);
        auto const support = SupportCells();
        auto const singleton = SingletonCell();
        amrex::ParticleReal const light_weight = 2.e8_prt;
        amrex::ParticleReal const heavy_weight = 3.e8_prt;
        amrex::ParticleReal const speed_of_light = static_cast<amrex::ParticleReal>(PhysConst::c);
        std::array<amrex::ParticleReal, 3> const bulk{
            2.e-3_prt * speed_of_light, -1.e-3_prt * speed_of_light, 3.e-3_prt * speed_of_light};
        std::array<amrex::ParticleReal, 3> const light_spread{
            2.e-4_prt * speed_of_light, 1.e-4_prt * speed_of_light, 3.e-4_prt * speed_of_light};
        std::array<amrex::ParticleReal, 3> const heavy_spread{
            2.e-5_prt * speed_of_light, 1.e-5_prt * speed_of_light, 2.e-5_prt * speed_of_light};

        auto add_particles = [&] (WarpXParticleContainer& pc,
                                  amrex::ParticleReal const weight,
                                  std::array<amrex::ParticleReal, 3> const& spread)
        {
            amrex::Vector<amrex::ParticleReal> x, y, z, ux, uy, uz;
            auto add_at = [&] (std::array<int, 3> const& cell, bool const pair) {
                std::array<amrex::Real, 3> position{};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    position[d] = geom.ProbLo(d) + (cell[d] + 0.5_rt) * dx[d];
                }
#if defined(WARPX_DIM_1D_Z)
                x.push_back(0.0_prt);
                y.push_back(0.0_prt);
                z.push_back(static_cast<amrex::ParticleReal>(position[0]));
#elif defined(WARPX_DIM_XZ)
                x.push_back(static_cast<amrex::ParticleReal>(position[0]));
                y.push_back(0.0_prt);
                z.push_back(static_cast<amrex::ParticleReal>(position[1]));
#else
                x.push_back(static_cast<amrex::ParticleReal>(position[0]));
                y.push_back(static_cast<amrex::ParticleReal>(position[1]));
                z.push_back(static_cast<amrex::ParticleReal>(position[2]));
#endif
                ux.push_back(bulk[0]);
                uy.push_back(bulk[1]);
                uz.push_back(bulk[2]);
                if (!pair) { return; }
#if defined(WARPX_DIM_1D_Z)
                x.push_back(0.0_prt);
                y.push_back(0.0_prt);
                z.push_back(static_cast<amrex::ParticleReal>(position[0]));
#elif defined(WARPX_DIM_XZ)
                x.push_back(static_cast<amrex::ParticleReal>(position[0]));
                y.push_back(0.0_prt);
                z.push_back(static_cast<amrex::ParticleReal>(position[1]));
#else
                x.push_back(static_cast<amrex::ParticleReal>(position[0]));
                y.push_back(static_cast<amrex::ParticleReal>(position[1]));
                z.push_back(static_cast<amrex::ParticleReal>(position[2]));
#endif
                ux.push_back(bulk[0]);
                uy.push_back(bulk[1]);
                uz.push_back(bulk[2]);
                ux[ux.size() - 2] += spread[0];
                uy[uy.size() - 2] += spread[1];
                uz[uz.size() - 2] += spread[2];
                ux[ux.size() - 1] -= spread[0];
                uy[uy.size() - 1] -= spread[1];
                uz[uz.size() - 1] -= spread[2];
            };
            for (auto const& cell : support) { add_at(cell, true); }
            add_at(singleton, false);
            amrex::Vector<amrex::Vector<amrex::ParticleReal>> attributes{
                amrex::Vector<amrex::ParticleReal>(x.size(), weight)};
            amrex::Vector<amrex::Vector<int>> no_integers;
            pc.AddNParticles(0, static_cast<long>(x.size()), x, y, z, ux, uy, uz,
                             1, attributes, 0, no_integers, 0);
        };

        auto& light = mypc.GetParticleContainerFromName("light");
        auto& heavy = mypc.GetParticleContainerFromName("heavy");
        add_particles(light, light_weight, light_spread);
        add_particles(heavy, heavy_weight, heavy_spread);

        using warpx::fields::FieldType;
        auto& te = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rho = *simulation.m_fields.get(FieldType::rho_fp, 0);
        auto& rho_sum = *simulation.m_fields.get("hybrid_rho_species_sum_fp", 0);
        auto& rho_light = *simulation.m_fields.get("rho_fp_light", 0);
        auto& rho_heavy = *simulation.m_fields.get("rho_fp_heavy", 0);

        amrex::Real const light_number_density =
            2.0_rt * static_cast<amrex::Real>(light_weight) / cell_volume;
        amrex::Real const heavy_number_density =
            2.0_rt * static_cast<amrex::Real>(heavy_weight) / cell_volume;
        amrex::ignore_unused(light_number_density, heavy_number_density);
        amrex::Real const electron_number_density =
            2.0_rt * static_cast<amrex::Real>(light_weight) / cell_volume +
            4.0_rt * static_cast<amrex::Real>(heavy_weight) / cell_volume;
        bool expect_no_exchange = false;
        bool expect_heavy_noop = false;
        amrex::Real exchange_dt = default_exchange_dt;
        amrex::ParmParse("test").query("exchange_dt", exchange_dt);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(exchange_dt) && exchange_dt > 0.0_rt,
            "The Qei fixture exchange interval must be finite and positive.");
        amrex::ParmParse("test").query("expect_no_exchange", expect_no_exchange);
        amrex::ParmParse("test").query("expect_heavy_noop", expect_heavy_noop);
        amrex::Real const light_rate = expect_no_exchange ? 0.0_rt : 1.3e10_rt;
        amrex::Real const heavy_rate =
            (expect_no_exchange || expect_heavy_noop) ? 0.0_rt : 0.7e10_rt;
        auto spread2 = [] (std::array<amrex::ParticleReal, 3> const& spread) {
            return static_cast<amrex::Real>(
                spread[0] * spread[0] + spread[1] * spread[1] + spread[2] * spread[2]);
        };
        amrex::Real const light_temperature =
            static_cast<amrex::Real>(light.getMass()) * spread2(light_spread) /
            (3.0_rt * PhysConst::kb);
        amrex::Real const heavy_temperature =
            static_cast<amrex::Real>(heavy.getMass()) * spread2(heavy_spread) /
            (3.0_rt * PhysConst::kb);

        FillUniform(rho, PhysConst::q_e * electron_number_density, geom);
        FillUniform(rho_light, PhysConst::q_e * 2.0_rt *
                                   static_cast<amrex::Real>(light_weight) / cell_volume, geom);
        FillUniform(rho_heavy, 2.0_rt * PhysConst::q_e * 2.0_rt *
                                   static_cast<amrex::Real>(heavy_weight) / cell_volume, geom);
        FillUniform(rho_sum, PhysConst::q_e * electron_number_density, geom);
        amrex::Real const initial_temperature =
            100.0_rt * PhysConst::q_e / PhysConst::kb;
        FillUniform(te, initial_temperature, geom);

        amrex::BoxArray const cell_boxes = amrex::convert(
            te.boxArray(), amrex::IntVect::TheCellVector());
        amrex::MultiFab light_temperature_field(cell_boxes, te.DistributionMap(), 1, 1);
        amrex::MultiFab heavy_temperature_field(cell_boxes, te.DistributionMap(), 1, 1);
        light_temperature_field.setVal(0.0_rt);
        heavy_temperature_field.setVal(0.0_rt);
        std::map<std::string, amrex::MultiFab*> temperatures{
            {"light", &light_temperature_field}, {"heavy", &heavy_temperature_field}};

        auto const initial_light = ParticleSnapshot(light, geom);
        auto const initial_heavy = ParticleSnapshot(heavy, geom);
        amrex::Real const initial_light_energy = KineticEnergy(
            light, static_cast<amrex::Real>(light.getMass()));
        amrex::Real const initial_heavy_energy = KineticEnergy(
            heavy, static_cast<amrex::Real>(heavy.getMass()));
        amrex::Real const initial_electron = ElectronEnergy(te, rho, geom);
        amrex::Real const initial_total_energy =
            initial_electron + initial_light_energy + initial_heavy_energy;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(initial_total_energy) && initial_total_energy > 0.0_rt,
            "Cartesian Qei fixture energy inventory is invalid.");

        ExchangeOracle oracle(geom, support, initial_temperature);
        oracle.SetIonTemperatures(light_temperature, heavy_temperature);
        amrex::Real expected_light_ion_change = 0.0_rt;
        amrex::Real expected_heavy_ion_change = 0.0_rt;
        int const applied_steps = expect_no_exchange ? 1 : exchange_steps;

        auto apply_exchange = [&] () {
            model.QDSMCAddTemperatureRelaxation(0, exchange_dt, temperatures);
            model.QDSMCApplyIonHeating(0, exchange_dt, nullptr, &temperatures);
        };

        if (expect_no_exchange) {
            apply_exchange();
            bool unchanged = RecordsUnchanged(initial_light, ParticleSnapshot(light, geom)) &&
                RecordsUnchanged(initial_heavy, ParticleSnapshot(heavy, geom));
            amrex::ParallelDescriptor::ReduceBoolAnd(unchanged);
            amrex::Real const total = ElectronEnergy(te, rho, geom) +
                KineticEnergy(light, static_cast<amrex::Real>(light.getMass())) +
                KineticEnergy(heavy, static_cast<amrex::Real>(heavy.getMass()));
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                unchanged && total == initial_total_energy,
                "Zero Cartesian Qei rates must be an exact no-op.");
        } else {
            for (int step = 0; step < exchange_steps; ++step) {
                auto const before_light = ParticleSnapshot(light, geom);
                auto const before_heavy = ParticleSnapshot(heavy, geom);
                expected_light_ion_change += oracle.ExchangeSpecies(
                    static_cast<amrex::Real>(light_weight), oracle.light_ion_temperature,
                    electron_number_density, light_rate, exchange_dt);
                expected_heavy_ion_change += oracle.ExchangeSpecies(
                    static_cast<amrex::Real>(heavy_weight), oracle.heavy_ion_temperature,
                    electron_number_density, heavy_rate, exchange_dt);
                apply_exchange();
                auto const after_light = ParticleSnapshot(light, geom);
                auto const after_heavy = ParticleSnapshot(heavy, geom);
                CheckCellMomentumAndSingleton(before_light, after_light, singleton);
                CheckCellMomentumAndSingleton(before_heavy, after_heavy, singleton);
                amrex::Real const light_change = KineticEnergy(
                    light, static_cast<amrex::Real>(light.getMass())) -
                    initial_light_energy;
                amrex::Real const heavy_change = KineticEnergy(
                    heavy, static_cast<amrex::Real>(heavy.getMass())) -
                    initial_heavy_energy;
                amrex::Print() << "Cartesian Qei expected/actual light gain="
                               << expected_light_ion_change << '/' << light_change
                               << " heavy gain=" << expected_heavy_ion_change << '/'
                               << heavy_change << '\n';
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::abs(light_change - expected_light_ion_change) <
                        1.e-10_rt * std::max(std::abs(expected_light_ion_change),
                                             initial_total_energy),
                    "The unequal-mass light-ion ledger differs from its independent exchange.");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::abs(heavy_change - expected_heavy_ion_change) <
                        1.e-10_rt * std::max(std::abs(expected_heavy_ion_change),
                                             initial_total_energy),
                    "The unequal-mass heavy-ion ledger differs from its independent exchange.");
                if (expect_heavy_noop) {
                    bool heavy_unchanged =
                        RecordsUnchanged(initial_heavy, after_heavy);
                    amrex::ParallelDescriptor::ReduceBoolAnd(heavy_unchanged);
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        heavy_unchanged && heavy_change == 0.0_rt,
                        "A zero heavy-species Qei rate must leave that species bitwise unchanged.");
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        std::abs(light_change) > 1.e-8_rt * initial_total_energy,
                        "The independent light-species rate produced no ion energy exchange.");
                } else {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        std::abs(light_change) > 1.e-8_rt * initial_total_energy &&
                            std::abs(heavy_change) > 1.e-8_rt * initial_total_energy &&
                            light_change * heavy_change > 0.0_rt,
                        "Unequal-mass Cartesian Qei must exchange independently on both ions.");
                }
                amrex::Real const total = ElectronEnergy(te, rho, geom) +
                    KineticEnergy(light, static_cast<amrex::Real>(light.getMass())) +
                    KineticEnergy(heavy, static_cast<amrex::Real>(heavy.getMass()));
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::abs(total - initial_total_energy) < 1.e-10_rt * initial_total_energy,
                    "Cartesian resolved Qei failed total energy conservation.");
            }
            amrex::Real oracle_min = initial_temperature;
            amrex::Real oracle_max = initial_temperature;
            for (amrex::Real const node_temperature : oracle.temperature) {
                oracle_min = std::min(oracle_min, node_temperature);
                oracle_max = std::max(oracle_max, node_temperature);
            }
            amrex::Real const temperature_scale =
                std::max(initial_temperature, std::abs(oracle_max));
            amrex::Real const temperature_tolerance =
                64.0_rt * std::numeric_limits<amrex::Real>::epsilon() * temperature_scale;
            amrex::Print() << "Cartesian Qei expected Te range=["
                           << oracle_min * PhysConst::kb / PhysConst::q_e << ", "
                           << oracle_max * PhysConst::kb / PhysConst::q_e
                           << "] eV actual=[" << te.min(0) * PhysConst::kb / PhysConst::q_e
                           << ", " << te.max(0) * PhysConst::kb / PhysConst::q_e << "]\n";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(te.min(0) - oracle_min) < temperature_tolerance &&
                    std::abs(te.max(0) - oracle_max) < temperature_tolerance,
                "Cartesian nodal electron temperature differs from the independent pairwise oracle.");
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory("cartesian_qei_checkpoint", 0755);
        }
        amrex::ParallelDescriptor::Barrier();
        model.WriteMomentHistory("cartesian_qei_checkpoint");
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ifstream support_file("cartesian_qei_checkpoint/HybridQeiSupport.txt");
            std::string version;
            int seed = 0;
            unsigned long long counter = 0;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (support_file >> version >> seed >> counter) && version == "resolved_pairwise_v3"
                    && seed == 1 && counter == static_cast<unsigned long long>(applied_steps),
                "Cartesian resolved Qei checkpoint counter is not the applied exchange count.");
        }
        model.ReadMomentHistory("cartesian_qei_checkpoint");
        amrex::Print() << "Cartesian resolved Qei: light gain=" << expected_light_ion_change
                       << " J, heavy gain=" << expected_heavy_ion_change << " J\n";
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
