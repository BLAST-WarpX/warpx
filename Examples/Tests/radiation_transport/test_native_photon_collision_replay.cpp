/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/NativePhotonCollisionReplay.H"
#include "Radiation/PacketMaterialWorkIteration.H"
#include "Radiation/ParticleImpulse.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace
{
using warpx::radiation::NativePhotonCollisionReplay;
using warpx::radiation::NativePhotonCollisionReport;
using warpx::radiation::ParticleImpulseTransaction;
using warpx::radiation::PhotonPacketTransaction;
using Vector = std::array<long double, 3>;

constexpr int number_k_cells = 4;
constexpr std::uint64_t frozen_seed = 4;
constexpr std::uint64_t frozen_interval = 7;
constexpr amrex::Real frozen_tau = 16.0_rt;

struct TileState
{
    std::pair<int, int> key{};
    long count = 0;
    std::vector<amrex::ParticleReal> real_data;
    std::vector<int> int_data;
    std::vector<std::uint64_t> idcpu;
};

struct ParticleState
{
    std::vector<amrex::ParticleReal> real_data;
    std::vector<int> int_data;
    int cell = 0;
};

template <typename T>
bool
SameBits (std::vector<T> const& a, std::vector<T> const& b)
{
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

std::vector<TileState>
Snapshot (WarpXParticleContainer::Base const& particles)
{
    std::vector<TileState> result;
    for (auto const& [key, tile] : particles.GetParticles(0))
    {
        TileState state;
        state.key = key;
        state.count = tile.numParticles();
        if (state.count != 0)
        {
            auto const& soa = tile.GetStructOfArrays();
            state.real_data.resize(
                static_cast<std::size_t>(state.count) * particles.NumRealComps());
            state.int_data.resize(
                static_cast<std::size_t>(state.count) * particles.NumIntComps());
            state.idcpu.resize(static_cast<std::size_t>(state.count));
            for (int component = 0; component < particles.NumRealComps(); ++component)
            {
                auto const& values = soa.GetRealData(component);
                amrex::Gpu::copy(
                    amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + state.count,
                    state.real_data.begin() + static_cast<std::size_t>(component) *
                        static_cast<std::size_t>(state.count));
            }
            for (int component = 0; component < particles.NumIntComps(); ++component)
            {
                auto const& values = soa.GetIntData(component);
                amrex::Gpu::copy(
                    amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + state.count,
                    state.int_data.begin() + static_cast<std::size_t>(component) *
                        static_cast<std::size_t>(state.count));
            }
            auto const& ids = soa.GetIdCPUData();
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost, ids.begin(), ids.begin() + state.count,
                state.idcpu.begin());
        }
        result.push_back(std::move(state));
    }
    return result;
}

bool
SameSnapshot (std::vector<TileState> const& a, std::vector<TileState> const& b)
{
    if (a.size() != b.size()) { return false; }
    for (std::size_t tile = 0; tile < a.size(); ++tile)
    {
        if (a[tile].key != b[tile].key || a[tile].count != b[tile].count ||
            !SameBits(a[tile].real_data, b[tile].real_data) ||
            !SameBits(a[tile].int_data, b[tile].int_data) ||
            !SameBits(a[tile].idcpu, b[tile].idcpu))
        {
            return false;
        }
    }
    return true;
}

bool
SameNonmomentumState (ParticleState const& a, ParticleState const& b)
{
    if (a.real_data.size() != b.real_data.size() ||
        a.int_data.size() != b.int_data.size())
    {
        return false;
    }
    for (int component = 0; component < static_cast<int>(a.real_data.size());
         ++component)
    {
        if (component == PIdx::ux || component == PIdx::uy || component == PIdx::uz)
        {
            continue;
        }
        if (std::memcmp(&a.real_data[static_cast<std::size_t>(component)],
                        &b.real_data[static_cast<std::size_t>(component)],
                        sizeof(amrex::ParticleReal)) != 0)
        {
            return false;
        }
    }
    return SameBits(a.int_data, b.int_data);
}

std::map<std::uint64_t, ParticleState>
ParticleRecords (WarpXParticleContainer::Base const& particles)
{
    std::map<std::uint64_t, ParticleState> records;
    auto const& geometry = WarpX::GetInstance().Geom(0);
    for (auto const& [key, tile] : particles.GetParticles(0))
    {
        long const count = tile.numParticles();
        if (count == 0) { continue; }
        auto const& soa = tile.GetStructOfArrays();
        std::vector<std::uint64_t> ids(static_cast<std::size_t>(count));
        std::vector<amrex::ParticleReal> z_values(static_cast<std::size_t>(count));
        std::vector<amrex::ParticleReal> real_values(
            static_cast<std::size_t>(count) * particles.NumRealComps());
        std::vector<int> int_values(
            static_cast<std::size_t>(count) * particles.NumIntComps());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
            soa.GetIdCPUData().begin() + count, ids.begin());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::z).begin(),
            soa.GetRealData(PIdx::z).begin() + count, z_values.begin());
        for (int component = 0; component < particles.NumRealComps(); ++component)
        {
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost, soa.GetRealData(component).begin(),
                soa.GetRealData(component).begin() + count,
                real_values.begin() + static_cast<std::size_t>(component) *
                    static_cast<std::size_t>(count));
        }
        for (int component = 0; component < particles.NumIntComps(); ++component)
        {
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost, soa.GetIntData(component).begin(),
                soa.GetIntData(component).begin() + count,
                int_values.begin() + static_cast<std::size_t>(component) *
                    static_cast<std::size_t>(count));
        }
        for (long particle = 0; particle < count; ++particle)
        {
            ParticleState state;
            state.real_data.resize(static_cast<std::size_t>(particles.NumRealComps()));
            state.int_data.resize(static_cast<std::size_t>(particles.NumIntComps()));
            for (int component = 0; component < particles.NumRealComps(); ++component)
            {
                state.real_data[static_cast<std::size_t>(component)] =
                    real_values[static_cast<std::size_t>(component) *
                                    static_cast<std::size_t>(count) +
                                static_cast<std::size_t>(particle)];
            }
            for (int component = 0; component < particles.NumIntComps(); ++component)
            {
                state.int_data[static_cast<std::size_t>(component)] =
                    int_values[static_cast<std::size_t>(component) *
                                   static_cast<std::size_t>(count) +
                               static_cast<std::size_t>(particle)];
            }
            long double const coordinate = static_cast<long double>(z_values[particle]) -
                static_cast<long double>(geometry.ProbLo(0));
            state.cell = static_cast<int>(std::floor(
                coordinate / static_cast<long double>(geometry.CellSize(0))));
            records.emplace(ids[static_cast<std::size_t>(particle)], std::move(state));
        }
        amrex::ignore_unused(key);
    }
    return records;
}

struct MaterialCell
{
    long double mass = 0;
    Vector momentum{};
    long double energy = 0;
};

std::array<MaterialCell, number_k_cells>
MaterialCells (WarpXParticleContainer& ions, std::array<int, 4> carry_components)
{
    auto const& geometry = WarpX::GetInstance().Geom(0);
    long double const speed = static_cast<long double>(PhysConst::c);
    long double const rest_mass = static_cast<long double>(ions.getMass());
    std::array<MaterialCell, number_k_cells> cells{};
    for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator)
    {
        long const count = iterator.numParticles();
        auto const& soa = iterator.GetStructOfArrays();
        std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, 9> host{};
        std::array<int, 9> components{PIdx::w, PIdx::z, PIdx::ux, PIdx::uy,
            PIdx::uz, carry_components[0], carry_components[1],
            carry_components[2], carry_components[3]};
        for (int component = 0; component < 9; ++component)
        {
            host[static_cast<std::size_t>(component)].resize(count);
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost, soa.GetRealData(components[component]).begin(),
                soa.GetRealData(components[component]).begin() + count,
                host[static_cast<std::size_t>(component)].begin());
        }
        for (long particle = 0; particle < count; ++particle)
        {
            long double const coordinate =
                static_cast<long double>(host[1][static_cast<std::size_t>(particle)]) -
                static_cast<long double>(geometry.ProbLo(0));
            int const cell = static_cast<int>(std::floor(
                coordinate / static_cast<long double>(geometry.CellSize(0))));
            if (cell < 0 || cell >= number_k_cells) { continue; }
            MaterialCell& material = cells[static_cast<std::size_t>(cell)];
            long double const weighted_mass =
                static_cast<long double>(host[0][static_cast<std::size_t>(particle)]) *
                rest_mass;
            Vector const velocity{
                static_cast<long double>(host[2][static_cast<std::size_t>(particle)]),
                static_cast<long double>(host[3][static_cast<std::size_t>(particle)]),
                static_cast<long double>(host[4][static_cast<std::size_t>(particle)])};
            long double const velocity_squared = velocity[0] * velocity[0] +
                velocity[1] * velocity[1] + velocity[2] * velocity[2];
            long double const gamma = std::sqrt(1 + velocity_squared /
                (speed * speed));
            material.mass += weighted_mass;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                material.momentum[dimension] += weighted_mass * velocity[dimension] +
                    weighted_mass * static_cast<long double>(
                        host[5 + static_cast<std::size_t>(
                                    dimension)][static_cast<std::size_t>(particle)]);
            }
            material.energy += weighted_mass * speed * speed * gamma +
                weighted_mass * static_cast<long double>(
                    host[8][static_cast<std::size_t>(particle)]);
        }
    }
#ifdef AMREX_USE_MPI
    std::array<long double, number_k_cells*5> values{};
    for (int cell = 0; cell < number_k_cells; ++cell)
    {
        values[5*cell] = cells[cell].mass;
        values[5*cell+1] = cells[cell].energy;
        for (int d = 0; d < 3; ++d) { values[5*cell+2+d] = cells[cell].momentum[d]; }
    }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
                  MPI_LONG_DOUBLE, MPI_SUM, amrex::ParallelDescriptor::Communicator());
    for (int cell = 0; cell < number_k_cells; ++cell)
    {
        cells[cell].mass = values[5*cell];
        cells[cell].energy = values[5*cell+1];
        for (int d = 0; d < 3; ++d) { cells[cell].momentum[d] = values[5*cell+2+d]; }
    }
#endif
    return cells;
}

struct PhotonCell
{
    long double energy = 0;
    Vector momentum{};
    Vector direction{};
};

std::array<PhotonCell, number_k_cells>
PhotonCells (std::map<std::uint64_t, ParticleState> const& records)
{
    long double const speed = static_cast<long double>(PhysConst::c);
    std::array<PhotonCell, number_k_cells> cells{};
    for (auto const& [id, state] : records)
    {
        if (state.cell < 0 || state.cell >= number_k_cells) { continue; }
        long double const weight = static_cast<long double>(
            state.real_data[static_cast<std::size_t>(PIdx::w)]);
        Vector momentum{};
        for (int dimension = 0; dimension < 3; ++dimension)
        {
            momentum[dimension] = static_cast<long double>(
                state.real_data[static_cast<std::size_t>(PIdx::ux + dimension)]) *
                weight * static_cast<long double>(PhysConst::m_e);
        }
        long double const norm = std::sqrt(momentum[0] * momentum[0] +
            momentum[1] * momentum[1] + momentum[2] * momentum[2]);
        PhotonCell& photon = cells[static_cast<std::size_t>(state.cell)];
        photon.energy = norm * speed;
        photon.momentum = momentum;
        for (int dimension = 0; dimension < 3; ++dimension)
        {
            photon.direction[dimension] = momentum[dimension] / norm;
        }
    }
#ifdef AMREX_USE_MPI
    std::array<long double, number_k_cells*4> values{};
    for (int cell = 0; cell < number_k_cells; ++cell)
    {
        values[4*cell] = cells[cell].energy;
        for (int d = 0; d < 3; ++d) { values[4*cell+1+d] = cells[cell].momentum[d]; }
    }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
                  MPI_LONG_DOUBLE, MPI_SUM, amrex::ParallelDescriptor::Communicator());
    for (int cell = 0; cell < number_k_cells; ++cell)
    {
        cells[cell].energy = values[4*cell];
        for (int d = 0; d < 3; ++d)
        {
            cells[cell].momentum[d] = values[4*cell+1+d];
            cells[cell].direction[d] = speed*cells[cell].momentum[d]/cells[cell].energy;
        }
    }
#endif
    return cells;
}

long double
Dot (Vector const& a, Vector const& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

long double
IndependentOutgoingEnergy (
    MaterialCell const& initial, PhotonCell const& incoming,
    Vector const& outgoing_direction, long double speed)
{
    return incoming.energy*(initial.energy-speed*Dot(initial.momentum, incoming.direction)) /
        (initial.energy-speed*Dot(initial.momentum, outgoing_direction)
         +incoming.energy*(1-Dot(incoming.direction, outgoing_direction)));
}

class WorkCallback
{
public:
    bool perturb = false;
    int calls = 0;

    WorkCallback (
        NativePhotonCollisionReplay const& replay,
        WarpXParticleContainer& photons)
        : m_replay(replay), m_photons(photons) {}

    bool operator() (
        ParticleImpulseTransaction const& staged,
        amrex::MultiFab& total_impulse, amrex::MultiFab& radiation_debit,
        amrex::MultiFab& uncancelled_energy_scale)
    {
        ++calls;
        m_transaction = std::make_unique<PhotonPacketTransaction>(m_photons);
        amrex::MultiFab kinetic_impulse(
            total_impulse.boxArray(), total_impulse.DistributionMap(), 3, 0);
        amrex::MultiFab kinetic_debit(
            total_impulse.boxArray(), total_impulse.DistributionMap(), 1, 0);
        amrex::MultiFab kinetic_scale(
            total_impulse.boxArray(), total_impulse.DistributionMap(), 1, 0);
        NativePhotonCollisionReport replay_report;
        if (!m_replay.Replay(
                staged.WorkVelocity(), m_transaction->Particles(), kinetic_impulse,
                kinetic_debit, kinetic_scale, replay_report))
        {
            return false;
        }
        amrex::MultiFab::Copy(total_impulse, kinetic_impulse, 0, 0, 3, 0);
        amrex::MultiFab::Copy(radiation_debit, kinetic_debit, 0, 0, 1, 0);
        amrex::MultiFab::Copy(uncancelled_energy_scale, kinetic_scale, 0, 0, 1, 0);
        if (perturb)
        {
            amrex::Real const sign = calls % 2 == 0 ? -1.0_rt : 1.0_rt;
            for (amrex::MFIter iterator(total_impulse); iterator.isValid(); ++iterator)
            {
                auto field = total_impulse.array(iterator);
                amrex::For(iterator.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (i < number_k_cells)
                    {
                        for (int dimension = 0; dimension < 3; ++dimension)
                        {
                            field(i, j, k, dimension) += sign * 0.01_rt;
                        }
                    }
                });
            }
        }
        return true;
    }

    void DiscardTransaction () { m_transaction.reset(); }
    [[nodiscard]] PhotonPacketTransaction& Transaction () const
    { return *m_transaction; }

private:
    NativePhotonCollisionReplay const& m_replay;
    WarpXParticleContainer& m_photons;
    std::unique_ptr<PhotonPacketTransaction> m_transaction;
};

amrex::Real
MaximumDiffusionField (amrex::MultiFab const& field, int component)
{
    amrex::MultiFab masked(
        field.boxArray(), field.DistributionMap(), field.nComp(), 0);
    amrex::MultiFab::Copy(masked, field, 0, 0, field.nComp(), 0);
    int const components = masked.nComp();
    for (amrex::MFIter iterator(masked); iterator.isValid(); ++iterator)
    {
        auto values = masked.array(iterator);
        amrex::For(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (i < number_k_cells)
            {
                for (int c = 0; c < components; ++c)
                {
                    values(i, j, k, c) = 0;
                }
            }
        });
    }
    return masked.max(component);
}
}

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        auto& simulation = WarpX::GetInstance();
        auto& particles = simulation.GetPartContainer();
        auto& ions = particles.GetParticleContainerFromName("ions");
        auto& photons_before_init =
            particles.GetParticleContainerFromName("photons");
        photons_before_init.AddRealComp("collision_real_sentinel", 1);
        photons_before_init.AddIntComp("collision_int_sentinel", 1);
        std::string const path = "native_collision";
        RegisterParticleImpulseState(ions, path);
        simulation.InitData();
        auto& photons = particles.GetParticleContainerFromName("photons");
        int const sentinel_real = photons.GetRealCompIndex("collision_real_sentinel");
        int const sentinel_int = photons.GetIntCompIndex("collision_int_sentinel");

        long double const speed = static_cast<long double>(PhysConst::c);
        long double const gamma_moving = 1 / std::sqrt(
            1 - 0.11L * 0.11L - 0.07L * 0.07L - 0.03L * 0.03L);
        amrex::GpuArray<amrex::ParticleReal, 3> const moving_u{
            static_cast<amrex::ParticleReal>(0.11L * gamma_moving * speed),
            static_cast<amrex::ParticleReal>(-0.07L * gamma_moving * speed),
            static_cast<amrex::ParticleReal>(0.03L * gamma_moving * speed)};

        auto& tiles = photons.GetParticles(0);
        for (auto& [key, tile] : tiles)
        {
            long const count = tile.numParticles();
            auto& soa = tile.GetStructOfArrays();
            std::vector<std::uint64_t> ids(static_cast<std::size_t>(count));
            std::vector<amrex::ParticleReal> z_values(static_cast<std::size_t>(count));
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost, soa.GetIdCPUData().begin(),
                soa.GetIdCPUData().begin() + count, ids.begin());
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost, soa.GetRealData(PIdx::z).begin(),
                soa.GetRealData(PIdx::z).begin() + count, z_values.begin());
            auto const& geometry = simulation.Geom(0);
            std::array<int, 4> kept_per_cell{};
            for (long particle = 0; particle < count; ++particle)
            {
                int const cell = static_cast<int>(std::floor(
                    (static_cast<long double>(z_values[static_cast<std::size_t>(particle)]) -
                     static_cast<long double>(geometry.ProbLo(0))) /
                    static_cast<long double>(geometry.CellSize(0))));
                bool const keep = cell >= 0 && cell < number_k_cells &&
                    kept_per_cell[static_cast<std::size_t>(cell)]++ == 0;
                if (!keep)
                {
                    ids[static_cast<std::size_t>(particle)] =
                        amrex::ParticleIdCpus::Invalid;
                }
            }
            amrex::Gpu::copy(
                amrex::Gpu::hostToDevice, ids.begin(), ids.end(),
                soa.GetIdCPUData().begin());
        }
        photons.Redistribute();
        for (auto& [key, tile] : photons.GetParticles(0))
        {
            long const count = tile.numParticles();
            auto data = tile.getParticleTileData();
            auto& soa = tile.GetStructOfArrays();
            auto* real_sentinel = soa.GetRealData(sentinel_real).data();
            auto* int_sentinel = soa.GetIntData(sentinel_int).data();
            auto const& geometry = simulation.Geom(0);
            long double const cell_mass = 1.0e20L *
                static_cast<long double>(geometry.CellSize(0))*ions.getMass();
            auto const lower = geometry.ProbLo(0);
            auto const inverse_dx = geometry.InvCellSize(0);
            amrex::ParticleReal const weights[number_k_cells] = {
                static_cast<amrex::ParticleReal>(
                    0.03L * cell_mass / static_cast<long double>(PhysConst::m_e)),
                static_cast<amrex::ParticleReal>(
                    0.20L * cell_mass / static_cast<long double>(PhysConst::m_e)),
                static_cast<amrex::ParticleReal>(
                    0.03L * cell_mass / static_cast<long double>(PhysConst::m_e)),
                static_cast<amrex::ParticleReal>(
                    0.20L * cell_mass / static_cast<long double>(PhysConst::m_e))};
            amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE (long particle)
            {
                int const cell = static_cast<int>(std::floor(
                    (data.m_rdata[PIdx::z][particle] - lower)*inverse_dx));
                data.m_rdata[PIdx::w][particle] = weights[cell];
                data.m_rdata[PIdx::ux][particle] = 0.0_prt;
                data.m_rdata[PIdx::uy][particle] = 0.0_prt;
                data.m_rdata[PIdx::uz][particle] =
                    static_cast<amrex::ParticleReal>(PhysConst::c);
                real_sentinel[particle] = 17.25_prt + particle + cell;
                int_sentinel[particle] = 31 + static_cast<int>(particle) + cell;
            });
        }
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator)
        {
            auto data = iterator.GetParticleTile().getParticleTileData();
            auto const& geometry = simulation.Geom(0);
            auto const lower = geometry.ProbLo(0);
            auto const inverse_dx = geometry.InvCellSize(0);
            amrex::ParallelFor(iterator.numParticles(),
                [=] AMREX_GPU_DEVICE (long particle)
            {
                int const cell = static_cast<int>(std::floor(
                    (data.m_rdata[PIdx::z][particle] - lower)*inverse_dx));
                bool const moving = cell >= 2 && cell < number_k_cells;
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    data.m_rdata[PIdx::ux + dimension][particle] =
                        moving ? moving_u[dimension] : 0.0_prt;
                }
            });
        }
        amrex::Gpu::streamSynchronize();

        auto const live_before = Snapshot(photons);
        auto const ions_before = Snapshot(ions);
        auto const& box_array = photons.ParticleBoxArray(0);
        auto const& distribution_map = photons.ParticleDistributionMap(0);
        amrex::iMultiFab diffusion_mask(box_array, distribution_map, 1, 0);
        amrex::MultiFab opacity(box_array, distribution_map, 1, 0);
        amrex::Real const dt = simulation.getdt()[0];
        for (amrex::MFIter iterator(diffusion_mask); iterator.isValid(); ++iterator)
        {
            auto mask = diffusion_mask.array(iterator);
            auto chi = opacity.array(iterator);
            amrex::For(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                mask(i, j, k) = i < number_k_cells ? 0 : 1;
                chi(i, j, k) = i < number_k_cells ? frozen_tau / (PhysConst::c * dt) : 0;
            });
        }

        NativePhotonCollisionOptions options;
        options.seed = frozen_seed;
        options.interval = frozen_interval;
        options.max_events_per_photon = 64;
        options.max_optical_depth = 16;
        NativePhotonCollisionReport report;
        NativePhotonCollisionReplay flag_replay;
        auto unsupported = options;
        unsupported.absorption = true;
        AMREX_ALWAYS_ASSERT(!flag_replay.Prepare(
            photons, diffusion_mask, opacity, simulation.Geom(0), dt, unsupported,
            report));
        unsupported = options;
        unsupported.emission = true;
        AMREX_ALWAYS_ASSERT(!flag_replay.Prepare(
            photons, diffusion_mask, opacity, simulation.Geom(0), dt, unsupported,
            report));
        unsupported = options;
        unsupported.moving_cross_sections = true;
        AMREX_ALWAYS_ASSERT(!flag_replay.Prepare(
            photons, diffusion_mask, opacity, simulation.Geom(0), dt, unsupported,
            report));
        auto too_thick = options;
        too_thick.max_optical_depth = frozen_tau - 0.01_rt;
        AMREX_ALWAYS_ASSERT(!flag_replay.Prepare(
            photons, diffusion_mask, opacity, simulation.Geom(0), dt, too_thick,
            report));
        AMREX_ALWAYS_ASSERT(SameSnapshot(Snapshot(photons), live_before));

        NativePhotonCollisionReplay isotropic_replay;
        auto isotropic_options = options;
        isotropic_options.angular_law = NativePhotonCollisionAngularLaw::Isotropic;
        bool const isotropic_prepared = isotropic_replay.Prepare(
            photons, diffusion_mask, opacity, simulation.Geom(0), dt,
            isotropic_options, report);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(isotropic_prepared, report.reason.c_str());
        AMREX_ALWAYS_ASSERT(report.valid && report.colliding_photons == number_k_cells &&
            report.collision_events >= number_k_cells &&
            report.collision_events <= number_k_cells*options.max_events_per_photon);
        amrex::MultiFab zero_velocity(box_array, distribution_map, 3, 0);
        zero_velocity.setVal(0);
        NativePhotonCollisionReport first_report;
        std::vector<TileState> first_state;
        amrex::MultiFab first_impulse(box_array, distribution_map, 3, 0);
        amrex::MultiFab first_debit(box_array, distribution_map, 1, 0);
        amrex::MultiFab first_scale(box_array, distribution_map, 1, 0);
        {
            PhotonPacketTransaction transaction(photons);
            AMREX_ALWAYS_ASSERT(isotropic_replay.Replay(
                zero_velocity, transaction.Particles(), first_impulse, first_debit,
                first_scale, first_report));
            first_state = Snapshot(transaction.Particles());
        }
        NativePhotonCollisionReport second_report;
        {
            PhotonPacketTransaction transaction(photons);
            amrex::MultiFab impulse(box_array, distribution_map, 3, 0);
            amrex::MultiFab debit(box_array, distribution_map, 1, 0);
            amrex::MultiFab scale(box_array, distribution_map, 1, 0);
            AMREX_ALWAYS_ASSERT(isotropic_replay.Replay(
                zero_velocity, transaction.Particles(), impulse, debit, scale,
                second_report));
            AMREX_ALWAYS_ASSERT(SameSnapshot(Snapshot(transaction.Particles()), first_state));
            for (int component = 0; component < 3; ++component)
            {
                AMREX_ALWAYS_ASSERT(first_impulse.max(component) ==
                                    impulse.max(component));
                AMREX_ALWAYS_ASSERT(first_impulse.min(component) ==
                                    impulse.min(component));
            }
            AMREX_ALWAYS_ASSERT(first_debit.max(0) == debit.max(0));
            AMREX_ALWAYS_ASSERT(first_scale.max(0) == scale.max(0));
        }
        AMREX_ALWAYS_ASSERT(first_report.collision_events ==
                            second_report.collision_events);

        // Ordinary interval-start photons in D are transparent to this source.
        // Keeping full snapshot indexing also covers mixed K/D tiles.
        {
            amrex::iMultiFab all_diffusion(box_array, distribution_map, 1, 0);
            amrex::MultiFab transparent(box_array, distribution_map, 1, 0);
            all_diffusion.setVal(1);
            transparent.setVal(0);
            NativePhotonCollisionReplay diffusion_replay;
            AMREX_ALWAYS_ASSERT(diffusion_replay.Prepare(photons, all_diffusion,
                transparent, simulation.Geom(0), dt, options, report));
            PhotonPacketTransaction transaction(photons);
            AMREX_ALWAYS_ASSERT(diffusion_replay.Replay(zero_velocity,
                transaction.Particles(), first_impulse, first_debit, first_scale, report));
            AMREX_ALWAYS_ASSERT(SameSnapshot(Snapshot(transaction.Particles()), live_before));
            AMREX_ALWAYS_ASSERT(report.collision_events == 0 && report.photon_energy_debit == 0);
            AMREX_ALWAYS_ASSERT(first_scale.norm0() == 0 && first_debit.norm0() == 0);
            for (int d = 0; d < 3; ++d) { AMREX_ALWAYS_ASSERT(first_impulse.norm0(d) == 0); }
        }

        ParticleImpulseMaterial material(particles, {"ions"});
        amrex::MultiFab initial_impulse(box_array, distribution_map, 3, 0);
        initial_impulse.setVal(0);
        ParticleImpulseTransaction seed_stage;
        AMREX_ALWAYS_ASSERT(seed_stage.Stage(material, path, initial_impulse, true));

        auto proper_options = isotropic_options;
        proper_options.moving_cross_sections = true;
        NativePhotonCollisionReplay proper_zero_replay;
        AMREX_ALWAYS_ASSERT(proper_zero_replay.Prepare(photons, diffusion_mask,
            opacity, simulation.Geom(0), dt, zero_velocity, proper_options, report));
        AMREX_ALWAYS_ASSERT(report.collision_events == first_report.collision_events);
        {
            PhotonPacketTransaction transaction(photons);
            AMREX_ALWAYS_ASSERT(proper_zero_replay.Replay(zero_velocity,
                transaction.Particles(), first_impulse, first_debit, first_scale, report));
            AMREX_ALWAYS_ASSERT(SameSnapshot(Snapshot(transaction.Particles()), first_state));
        }

        NativePhotonCollisionReplay replay;
        options.angular_law = NativePhotonCollisionAngularLaw::FixedFrameDirection;
        options.fixed_frame_direction = {0, 0, -1};
        options.moving_cross_sections = true;
        options.max_optical_depth = 32; // Bounds the boosted envelope, not its sampling rate.
        AMREX_ALWAYS_ASSERT(replay.Prepare(
            photons, diffusion_mask, opacity, simulation.Geom(0), dt,
            seed_stage.WorkVelocity(), options,
            report));
        AMREX_ALWAYS_ASSERT(report.valid && report.colliding_photons == number_k_cells &&
            report.collision_events >= number_k_cells &&
            report.collision_events <= number_k_cells*options.max_events_per_photon);
        {
            PhotonPacketTransaction transaction(photons);
            auto const candidate_before = Snapshot(transaction.Particles());
            amrex::MultiFab fast_velocity(box_array, distribution_map, 3, 0);
            fast_velocity.setVal(PhysConst::c);
            amrex::MultiFab impulse(box_array, distribution_map, 3, 0);
            amrex::MultiFab debit(box_array, distribution_map, 1, 0);
            amrex::MultiFab scale(box_array, distribution_map, 1, 0);
            amrex::Real const poison = std::numeric_limits<amrex::Real>::quiet_NaN();
            impulse.setVal(poison);
            debit.setVal(poison);
            scale.setVal(poison);
            NativePhotonCollisionReport invalid_report;
            AMREX_ALWAYS_ASSERT(!replay.Replay(
                fast_velocity, transaction.Particles(), impulse, debit, scale,
                invalid_report));
            AMREX_ALWAYS_ASSERT(!invalid_report.valid);
            AMREX_ALWAYS_ASSERT(SameSnapshot(Snapshot(transaction.Particles()),
                                             candidate_before));
        }

        std::array<int, 4> carry{};
        carry[0] = ions.GetRealCompIndex("radiation_impulse_" + path + "_ux");
        carry[1] = ions.GetRealCompIndex("radiation_impulse_" + path + "_uy");
        carry[2] = ions.GetRealCompIndex("radiation_impulse_" + path + "_uz");
        carry[3] = ions.GetRealCompIndex("radiation_impulse_" + path + "_work");
        auto const initial_material = MaterialCells(ions, carry);
        auto const initial_photon_records = ParticleRecords(photons);
        auto const initial_photons = PhotonCells(initial_photon_records);

        PacketMaterialWorkIterationOptions iteration_options;
        iteration_options.impulse_relative_tolerance = 1.e-13_rt;
        iteration_options.work_relative_tolerance = 1.e-13_rt;
        WorkCallback callback(replay, photons);
        auto staged = std::make_unique<ParticleImpulseTransaction>();
        PacketMaterialWorkIterationReport iteration_report;
        callback.perturb = true;
        iteration_options.max_iterations = 2;
        AMREX_ALWAYS_ASSERT(!TryIteratePacketMaterialWork(
            material, path, initial_impulse, std::ref(callback), iteration_options,
            staged, iteration_report));
        AMREX_ALWAYS_ASSERT(iteration_report.reason ==
                            PacketMaterialWorkIterationReason::NotConverged);
        AMREX_ALWAYS_ASSERT(SameSnapshot(Snapshot(photons), live_before));
        AMREX_ALWAYS_ASSERT(SameSnapshot(Snapshot(ions), ions_before));
        callback.DiscardTransaction();

        callback.perturb = false;
        callback.calls = 0;
        iteration_options.max_iterations = 64;
        AMREX_ALWAYS_ASSERT(TryIteratePacketMaterialWork(
            material, path, initial_impulse, std::ref(callback), iteration_options,
            staged, iteration_report));
        AMREX_ALWAYS_ASSERT(iteration_report.converged);
        for (int dimension = 0; dimension < 3; ++dimension)
        {
            AMREX_ALWAYS_ASSERT(MaximumDiffusionField(
                iteration_report.candidate_total_impulse, dimension) == 0);
        }
        AMREX_ALWAYS_ASSERT(MaximumDiffusionField(
            iteration_report.candidate_radiation_debit, 0) == 0);
        AMREX_ALWAYS_ASSERT(MaximumDiffusionField(
            iteration_report.candidate_uncancelled_energy_scale, 0) == 0);

        AMREX_ALWAYS_ASSERT(callback.Transaction().ValidateLive());
        staged->Commit();
        AMREX_ALWAYS_ASSERT(material.Commit());
        callback.Transaction().Commit();
        auto const final_material = MaterialCells(ions, carry);
        auto const final_photon_records = ParticleRecords(photons);
        auto const final_photons = PhotonCells(final_photon_records);
        AMREX_ALWAYS_ASSERT(final_photon_records.size() ==
                            initial_photon_records.size());
        for (auto const& [id, initial] : initial_photon_records)
        {
            auto const found = final_photon_records.find(id);
            AMREX_ALWAYS_ASSERT(found != final_photon_records.end());
            AMREX_ALWAYS_ASSERT(found->second.cell == initial.cell);
            AMREX_ALWAYS_ASSERT(SameNonmomentumState(found->second, initial));
        }

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            for (int cell = 0; cell < number_k_cells; ++cell)
            {
                PhotonCell const& incoming = initial_photons[
                    static_cast<std::size_t>(cell)];
                PhotonCell const& outgoing = final_photons[
                    static_cast<std::size_t>(cell)];
                long double const expected_energy = IndependentOutgoingEnergy(
                    initial_material[static_cast<std::size_t>(cell)], incoming,
                    outgoing.direction, speed);
                AMREX_ALWAYS_ASSERT(std::abs(outgoing.energy - expected_energy) <=
                    1.e-11L * (outgoing.energy + expected_energy));
                if (cell < 2)
                {
                    long double const cold_expected = incoming.energy /
                        (1 + 2 * incoming.energy /
                         (initial_material[static_cast<std::size_t>(cell)].mass *
                          speed * speed));
                    AMREX_ALWAYS_ASSERT(std::abs(outgoing.energy - cold_expected) <=
                        1.e-11L * (outgoing.energy + cold_expected));
                }
                else
                {
                    AMREX_ALWAYS_ASSERT(std::abs(outgoing.direction[0]) > 0 ||
                                        std::abs(outgoing.direction[1]) > 0);
                }

                MaterialCell const& initial_state =
                    initial_material[static_cast<std::size_t>(cell)];
                MaterialCell const& final_state =
                    final_material[static_cast<std::size_t>(cell)];
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    long double const material_change =
                        final_state.momentum[dimension] -
                        initial_state.momentum[dimension];
                    long double const photon_change =
                        incoming.momentum[dimension] - outgoing.momentum[dimension];
                    AMREX_ALWAYS_ASSERT(std::abs(material_change - photon_change) <=
                        1.e-11L * (std::abs(material_change) +
                                   std::abs(photon_change)));
                }
                long double const material_energy_change =
                    final_state.energy - initial_state.energy;
                long double const photon_energy_change =
                    incoming.energy - outgoing.energy;
                AMREX_ALWAYS_ASSERT(std::abs(
                    material_energy_change - photon_energy_change) <=
                    1.e-11L * (std::abs(material_energy_change) +
                               std::abs(photon_energy_change)));
            }
        }

        amrex::Print() << "Native coherent photon replay passed with "
                       << report.collision_events << " frozen events and "
                       << iteration_report.iterations << " material iterations; "
                       << "D-cell outputs are zero.\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
