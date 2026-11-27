/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/FiniteMassPhotonCollision.H"
#include "Radiation/PacketMaterialWorkIteration.H"
#include "Radiation/ParticleImpulse.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace amrex::literals;

namespace
{
    using Vector = std::array<long double, 3>;

    constexpr int num_photons = 2;

    long double Dot (Vector const& a, Vector const& b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    Vector Direction (long double x, long double y, long double z)
    {
        auto const norm = std::sqrt(x * x + y * y + z * z);
        return {x / norm, y / norm, z / norm};
    }

    amrex::GpuArray<amrex::Real, 3> RealDirection (
        long double x, long double y, long double z)
    {
        auto const value = Direction(x, y, z);
        return {static_cast<amrex::Real>(value[0]),
                static_cast<amrex::Real>(value[1]),
                static_cast<amrex::Real>(value[2])};
    }

    std::vector<amrex::ParticleReal> ParticleSnapshot (WarpXParticleContainer& ions)
    {
        std::vector<amrex::ParticleReal> snapshot;
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator)
        {
            for (int component = 0; component < ions.NumRealComps(); ++component)
            {
                auto const& values = iterator.GetStructOfArrays().GetRealData(component);
                auto const start = snapshot.size();
                snapshot.resize(start + iterator.numParticles());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + iterator.numParticles(), snapshot.begin() + start);
            }
        }
        return snapshot;
    }

    bool SameSnapshot (
        std::vector<amrex::ParticleReal> const& a,
        std::vector<amrex::ParticleReal> const& b)
    {
        return a.size() == b.size() && (a.empty() ||
            std::memcmp(a.data(), b.data(), a.size() * sizeof(amrex::ParticleReal)) == 0);
    }

    amrex::Real FieldSum (amrex::MultiFab const& field, int component)
    {
        return field.sum(component);
    }

    amrex::Real FieldMaxDifference (
        amrex::MultiFab const& a, amrex::MultiFab const& b, int component)
    {
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<amrex::Real> data(ops);
        using Tuple = typename decltype(data)::Type;
        for (amrex::MFIter iterator(a); iterator.isValid(); ++iterator)
        {
            auto const left = a.const_array(iterator);
            auto const right = b.const_array(iterator);
            ops.eval(iterator.validbox(), data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple
            {
                return {std::abs(left(i, j, k, component) - right(i, j, k, component))};
            });
        }
        amrex::Real error = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealMax(error);
        return error;
    }

    struct MaterialTotals
    {
        long double mass = 0;
        Vector momentum{};
        long double kinetic_energy = 0;
    };

    MaterialTotals MaterialInventory (
        WarpXParticleContainer& ions, std::string const& path, long double speed)
    {
        std::array<int, 4> carry{};
        carry[0] = ions.GetRealCompIndex("radiation_impulse_" + path + "_ux");
        carry[1] = ions.GetRealCompIndex("radiation_impulse_" + path + "_uy");
        carry[2] = ions.GetRealCompIndex("radiation_impulse_" + path + "_uz");
        carry[3] = ions.GetRealCompIndex("radiation_impulse_" + path + "_work");
        MaterialTotals result;
        auto const rest_mass = static_cast<long double>(ions.getMass());
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator)
        {
            auto const count = iterator.numParticles();
            std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, 8> host{};
            auto const& soa = iterator.GetStructOfArrays();
            std::array<int, 8> components{PIdx::w, PIdx::ux, PIdx::uy, PIdx::uz,
                carry[0], carry[1], carry[2], carry[3]};
            for (int c = 0; c < 8; ++c)
            {
                auto const& values = soa.GetRealData(components[c]);
                host[c].resize(count);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + count, host[c].begin());
            }
            for (long p = 0; p < static_cast<long>(count); ++p)
            {
                long double const weighted_mass =
                    static_cast<long double>(host[0][p]) * rest_mass;
                Vector const velocity{static_cast<long double>(host[1][p]),
                    static_cast<long double>(host[2][p]),
                    static_cast<long double>(host[3][p])};
                result.mass += weighted_mass;
                for (int d = 0; d < 3; ++d)
                {
                    result.momentum[d] += weighted_mass * velocity[d];
                }
                result.kinetic_energy += weighted_mass * Dot(velocity, velocity) /
                    (std::sqrt(1.0L + Dot(velocity, velocity) / (speed * speed)) + 1.0L);
            }
        }
#ifdef AMREX_USE_MPI
        std::array<long double, 5> sums{result.mass, result.momentum[0], result.momentum[1],
                                        result.momentum[2], result.kinetic_energy};
        MPI_Allreduce(MPI_IN_PLACE, sums.data(), 5, MPI_LONG_DOUBLE, MPI_SUM,
                      amrex::ParallelDescriptor::Communicator());
        result.mass = sums[0];
        result.momentum = {sums[1], sums[2], sums[3]};
        result.kinetic_energy = sums[4];
#endif
        return result;
    }

    struct AnalyticPhoton
    {
        long double energy = 0;
        Vector incoming{};
        Vector frame_outgoing{};
    };

    struct AnalyticSolution
    {
        Vector impulse{};
        Vector frame_velocity{};
        long double radiation_debit = 0;
        long double material_work = 0;
        int iterations = 0;
    };

    Vector SecantVelocity (
        Vector const& initial_u, Vector const& impulse, long double cell_mass,
        long double speed)
    {
        Vector increment{};
        Vector next{};
        long double old_squared = 0;
        long double new_squared = 0;
        for (int d = 0; d < 3; ++d)
        {
            increment[d] = impulse[d] / cell_mass;
            next[d] = initial_u[d] + increment[d];
            old_squared += initial_u[d] * initial_u[d];
            new_squared += next[d] * next[d];
        }
        long double const denominator =
            std::sqrt(1.0L + old_squared / (speed * speed)) +
            std::sqrt(1.0L + new_squared / (speed * speed));
        return {(2.0L * initial_u[0] + increment[0]) / denominator,
                (2.0L * initial_u[1] + increment[1]) / denominator,
                (2.0L * initial_u[2] + increment[2]) / denominator};
    }

    AnalyticSolution SolveAnalytic (
        std::vector<AnalyticPhoton> const& photons, Vector const& initial_u,
        long double cell_mass, long double speed, long double damping)
    {
        AnalyticSolution result;
        for (int iteration = 0; iteration < 256; ++iteration)
        {
            ++result.iterations;
            Vector const velocity =
                SecantVelocity(initial_u, result.impulse, cell_mass, speed);
            Vector const beta{velocity[0] / speed, velocity[1] / speed,
                              velocity[2] / speed};
            long double const gamma = 1.0L / std::sqrt(1.0L - Dot(beta, beta));
            Vector candidate{};
            result.radiation_debit = 0;
            for (auto const& photon : photons)
            {
                long double const frame_energy =
                    photon.energy * gamma * (1.0L - Dot(beta, photon.incoming));
                long double const projection = Dot(beta, photon.frame_outgoing);
                Vector outgoing{};
                for (int d = 0; d < 3; ++d)
                {
                    outgoing[d] = (photon.frame_outgoing[d] / gamma + beta[d] +
                        gamma / (gamma + 1.0L) * projection * beta[d]) /
                        (1.0L + projection);
                }
                long double const outgoing_energy =
                    frame_energy * gamma * (1.0L + projection);
                result.radiation_debit += photon.energy - outgoing_energy;
                for (int d = 0; d < 3; ++d)
                {
                    candidate[d] += (photon.energy * photon.incoming[d] -
                        outgoing_energy * outgoing[d]) / speed;
                }
            }
            long double residual = 0;
            long double scale = 0;
            for (int d = 0; d < 3; ++d)
            {
                residual += std::abs(candidate[d] - result.impulse[d]);
                scale += std::abs(candidate[d]) + std::abs(result.impulse[d]);
            }
            if (residual <=
                256.0L * std::numeric_limits<long double>::epsilon() * scale)
            {
                result.frame_velocity = velocity;
                Vector next{};
                long double old_squared = 0;
                long double new_squared = 0;
                for (int d = 0; d < 3; ++d)
                {
                    next[d] = initial_u[d] + result.impulse[d] / cell_mass;
                    old_squared += initial_u[d] * initial_u[d];
                    new_squared += next[d] * next[d];
                }
                result.material_work = cell_mass * speed * speed *
                    (std::sqrt(1.0L + new_squared / (speed * speed)) -
                     std::sqrt(1.0L + old_squared / (speed * speed)));
                return result;
            }
            for (int d = 0; d < 3; ++d)
            {
                result.impulse[d] =
                    (1.0L - damping) * result.impulse[d] + damping * candidate[d];
            }
        }
        return {};
    }

    enum class ReplayMode { Physical, RejectOneRank, NonConverging, OffsettingEnergyErrors };

    class ReplayFixture
    {
    public:
        ReplayMode mode = ReplayMode::Physical;
        int calls = 0;

        ReplayFixture (
            amrex::BoxArray const& boxes,
            amrex::DistributionMapping const& distribution, amrex::Real total_photon_energy)
            : m_last_impulse(boxes, distribution, 3, 0),
              m_last_debit(boxes, distribution, 1, 0),
              m_last_scale(boxes, distribution, 1, 0),
              m_last_work_velocity(boxes, distribution, 3, 0),
              m_last_requested_work(boxes, distribution, 1, 0)
        {
            m_photon_energy[0] = total_photon_energy * 5 / 14;
            m_photon_energy[1] = total_photon_energy * 9 / 14;
            m_incoming[0] = RealDirection(0.2L, -0.3L, 0.87L);
            m_incoming[1] = RealDirection(-0.1L, 0.45L, 0.75L);
            m_outgoing[0] = RealDirection(0.35L, 0.25L, 0.8L);
            m_outgoing[1] = RealDirection(-0.2L, -0.4L, 0.8L);
            Reset();
        }

        void Reset ()
        {
            mode = ReplayMode::Physical;
            calls = 0;
            m_last_impulse.setVal(0);
            m_last_debit.setVal(0);
            m_last_scale.setVal(0);
            m_last_work_velocity.setVal(0);
            m_last_requested_work.setVal(0);
        }

        bool operator() (
            warpx::radiation::ParticleImpulseTransaction const& staged,
            amrex::MultiFab& new_total_impulse,
            amrex::MultiFab& total_radiation_debit,
            amrex::MultiFab& uncancelled_energy_scale)
        {
            ++calls;
            if (mode == ReplayMode::RejectOneRank &&
                amrex::ParallelDescriptor::MyProc() == 0)
            {
                return false;
            }
            int const source_cells =
                WarpX::GetInstance().Geom(0).Domain().length(0) / 2;
            amrex::GpuArray<warpx::radiation::FourVector, num_photons> incoming{};
            for (int p = 0; p < num_photons; ++p)
            {
                incoming[p][0] = m_photon_energy[p];
                for (int d = 0; d < 3; ++d)
                {
                    incoming[p][d + 1] = incoming[p][0] * m_incoming[p][d];
                }
            }
            auto const outgoing = m_outgoing;
            amrex::Gpu::DeviceScalar<int> invalid(0);
            auto* invalid_ptr = invalid.dataPtr();
            new_total_impulse.setVal(0);
            total_radiation_debit.setVal(0);
            uncancelled_energy_scale.setVal(0);
            for (amrex::MFIter iterator(new_total_impulse);
                 iterator.isValid(); ++iterator)
            {
                auto const velocity = staged.WorkVelocity().const_array(iterator);
                auto const impulse = new_total_impulse.array(iterator);
                auto const debit = total_radiation_debit.array(iterator);
                auto const scale = uncancelled_energy_scale.array(iterator);
                amrex::For(iterator.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (i >= source_cells) { return; }
                    amrex::GpuArray<amrex::Real, 3> beta{};
                    for (int d = 0; d < 3; ++d)
                    {
                        beta[d] = velocity(i, j, k, d) / PhysConst::c;
                    }
                    for (int p = 0; p < num_photons; ++p)
                    {
                        warpx::radiation::FourVector photon{};
                        amrex::GpuArray<amrex::Real, 3> collision_impulse{};
                        amrex::Real photon_debit = 0;
                        if (!warpx::radiation::CoherentPhotonCollision(
                                incoming[p], outgoing[p], beta, photon,
                                collision_impulse, photon_debit))
                        {
                            amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                            continue;
                        }
                        for (int d = 0; d < 3; ++d)
                        {
                            impulse(i, j, k, d) += collision_impulse[d];
                        }
                        debit(i, j, k) += photon_debit;
                        scale(i, j, k) += incoming[p][0] + photon[0];
                    }
                });
            }
            if (invalid.dataValue() != 0) { return false; }
            if (mode == ReplayMode::OffsettingEnergyErrors)
            {
                auto const error = (m_photon_energy[0]+m_photon_energy[1])/10;
                for (amrex::MFIter iterator(total_radiation_debit); iterator.isValid(); ++iterator)
                {
                    auto const debit = total_radiation_debit.array(iterator);
                    amrex::ParallelFor(iterator.validbox(),
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (i < source_cells) { debit(i, j, k) += i % 2 == 0 ? error : -error; }
                    });
                }
            }
            if (mode == ReplayMode::NonConverging)
            {
                amrex::Real const sign = (calls % 2 == 0) ? -1.0_rt : 1.0_rt;
                for (amrex::MFIter iterator(new_total_impulse);
                     iterator.isValid(); ++iterator)
                {
                    auto const field = new_total_impulse.array(iterator);
                    amrex::For(iterator.validbox(),
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for (int d = 0; d < 3; ++d)
                        {
                            field(i, j, k, d) += sign * 0.01_rt;
                        }
                    });
                }
            }
            amrex::MultiFab::Copy(m_last_impulse, new_total_impulse, 0, 0, 3, 0);
            amrex::MultiFab::Copy(m_last_debit, total_radiation_debit, 0, 0, 1, 0);
            amrex::MultiFab::Copy(m_last_scale, uncancelled_energy_scale, 0, 0, 1, 0);
            amrex::MultiFab::Copy(
                m_last_work_velocity, staged.WorkVelocity(), 0, 0, 3, 0);
            amrex::MultiFab::Copy(
                m_last_requested_work, staged.RequestedWork(), 0, 0, 1, 0);
            return true;
        }

        [[nodiscard]] amrex::MultiFab const& LastImpulse () const
        { return m_last_impulse; }
        [[nodiscard]] amrex::MultiFab const& LastDebit () const { return m_last_debit; }
        [[nodiscard]] amrex::MultiFab const& LastScale () const { return m_last_scale; }
        [[nodiscard]] amrex::MultiFab const& LastWorkVelocity () const
        { return m_last_work_velocity; }
        [[nodiscard]] amrex::Real PhotonEnergy (int p) const { return m_photon_energy[p]; }

    private:
        amrex::GpuArray<amrex::Real, num_photons> m_photon_energy{};
        amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, num_photons> m_incoming{};
        amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, num_photons> m_outgoing{};
        amrex::MultiFab m_last_impulse;
        amrex::MultiFab m_last_debit;
        amrex::MultiFab m_last_scale;
        amrex::MultiFab m_last_work_velocity;
        amrex::MultiFab m_last_requested_work;
    };
}

int main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        auto& simulation = WarpX::GetInstance();
        auto& particles = simulation.GetPartContainer();
        std::vector<std::string> material_names{"ions"};
        amrex::ParmParse("test").queryarr("material_species", material_names);
        std::vector<WarpXParticleContainer*> species;
        std::string const path = "finite_mass";
        for (auto const& name : material_names) {
            species.push_back(&particles.GetParticleContainerFromName(name));
            RegisterParticleImpulseState(*species.back(), path);
        }
        auto& ions = *species.front();
        simulation.InitData();

        long double const speed = static_cast<long double>(PhysConst::c);
        auto const gamma_initial = 1/std::sqrt(1-0.11L*0.11L-0.07L*0.07L-0.03L*0.03L);
        amrex::GpuArray<amrex::ParticleReal, 3> const initial_u{
            static_cast<amrex::ParticleReal>(0.11L*gamma_initial*speed),
            static_cast<amrex::ParticleReal>(-0.07L*gamma_initial*speed),
            static_cast<amrex::ParticleReal>(0.03L*gamma_initial*speed)};
        Vector const initial_u_long{initial_u[0], initial_u[1], initial_u[2]};
        for (auto* material_species : species) {
            for (WarpXParIter iterator(*material_species, 0); iterator.isValid(); ++iterator)
            {
                auto data = iterator.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(iterator.numParticles(),
                    [=] AMREX_GPU_DEVICE (long p)
                {
                    data.m_rdata[PIdx::ux][p] = initial_u[0];
                    data.m_rdata[PIdx::uy][p] = initial_u[1];
                    data.m_rdata[PIdx::uz][p] = initial_u[2];
                });
            }
        }
        amrex::Gpu::streamSynchronize();

        auto const snapshot = [&] () {
            std::vector<amrex::ParticleReal> values;
            for (auto* material_species : species) {
                auto const next = ParticleSnapshot(*material_species);
                values.insert(values.end(), next.begin(), next.end());
            }
            return values;
        };
        auto const inventory = [&] () {
            MaterialTotals total;
            for (auto* material_species : species) {
                auto const next = MaterialInventory(*material_species, path, speed);
                total.mass += next.mass;
                total.kinetic_energy += next.kinetic_energy;
                for (int d = 0; d < 3; ++d) { total.momentum[d] += next.momentum[d]; }
            }
            return total;
        };
        std::vector<MaterialTotals> species_before;
        for (auto* material_species : species) {
            species_before.push_back(MaterialInventory(*material_species, path, speed));
        }
        auto const before = snapshot();
        MaterialTotals const initial_material = inventory();
        int const n_cells = simulation.Geom(0).Domain().length(0);
        int const source_cells = n_cells / 2;
        amrex::Real recoil_fraction = 0.03;
        amrex::ParmParse("test").query("recoil_fraction", recoil_fraction);
        auto const total_photon_energy = static_cast<amrex::Real>(
            recoil_fraction*initial_material.mass/n_cells*speed*speed);
        ParticleImpulseMaterial material(particles, material_names);
        amrex::MultiFab initial_impulse(
            ions.ParticleBoxArray(0), ions.ParticleDistributionMap(0), 3, 0);
        initial_impulse.setVal(0);
        ReplayFixture replay(
            initial_impulse.boxArray(), initial_impulse.DistributionMap(), total_photon_energy);

        auto sentinel = std::make_unique<ParticleImpulseTransaction>();
        auto* const sentinel_pointer = sentinel.get();
        PacketMaterialWorkIterationOptions options;
        PacketMaterialWorkIterationReport report;

        auto bad_options = options;
        if (amrex::ParallelDescriptor::MyProc() == 0) { bad_options.damping = 0; }
        AMREX_ALWAYS_ASSERT(!TryIteratePacketMaterialWork(material, path, initial_impulse,
            std::ref(replay), bad_options, sentinel, report));
        AMREX_ALWAYS_ASSERT(report.reason == PacketMaterialWorkIterationReason::InvalidOptions);
        AMREX_ALWAYS_ASSERT(replay.calls == 0 && sentinel.get() == sentinel_pointer);
        auto const invalid_path = amrex::ParallelDescriptor::MyProc() == 0 ? "" : path;
        AMREX_ALWAYS_ASSERT(!TryIteratePacketMaterialWork(material, invalid_path, initial_impulse,
            std::ref(replay), options, sentinel, report));
        AMREX_ALWAYS_ASSERT(report.reason == PacketMaterialWorkIterationReason::InvalidLayout);
        AMREX_ALWAYS_ASSERT(replay.calls == 0 && sentinel.get() == sentinel_pointer);

        replay.mode = ReplayMode::RejectOneRank;
        AMREX_ALWAYS_ASSERT(!TryIteratePacketMaterialWork(
            material, path, initial_impulse, std::ref(replay), options, sentinel, report));
        AMREX_ALWAYS_ASSERT(sentinel.get() == sentinel_pointer);
        AMREX_ALWAYS_ASSERT(report.reason ==
            PacketMaterialWorkIterationReason::ReplayRejected);
        AMREX_ALWAYS_ASSERT(SameSnapshot(snapshot(), before));

        replay.Reset();
        replay.mode = ReplayMode::NonConverging;
        options.max_iterations = 2;
        options.damping = 1.0_rt;
        AMREX_ALWAYS_ASSERT(!TryIteratePacketMaterialWork(
            material, path, initial_impulse, std::ref(replay), options, sentinel, report));
        AMREX_ALWAYS_ASSERT(sentinel.get() == sentinel_pointer);
        AMREX_ALWAYS_ASSERT(report.reason ==
            PacketMaterialWorkIterationReason::NotConverged);
        AMREX_ALWAYS_ASSERT(report.iterations == 2);
        AMREX_ALWAYS_ASSERT(SameSnapshot(snapshot(), before));

        replay.Reset();
        options.max_iterations = 64;
        options.damping = 0.5_rt;
        options.impulse_relative_tolerance = 1.e-13_rt;
        options.work_relative_tolerance = 1.e-13_rt;
        AMREX_ALWAYS_ASSERT(TryIteratePacketMaterialWork(
            material, path, initial_impulse, std::ref(replay), options, sentinel, report));
        AMREX_ALWAYS_ASSERT(sentinel.get() != sentinel_pointer);
        AMREX_ALWAYS_ASSERT(report.converged);
        AMREX_ALWAYS_ASSERT(report.reason ==
            PacketMaterialWorkIterationReason::Success);
        AMREX_ALWAYS_ASSERT(replay.calls == report.iterations);
        for (int d = 0; d < 3; ++d)
        {
            AMREX_ALWAYS_ASSERT(FieldMaxDifference(
                report.candidate_total_impulse, replay.LastImpulse(), d) == 0);
            AMREX_ALWAYS_ASSERT(FieldMaxDifference(
                sentinel->WorkVelocity(), replay.LastWorkVelocity(), d) == 0);
        }
        AMREX_ALWAYS_ASSERT(FieldMaxDifference(
            report.candidate_radiation_debit, replay.LastDebit(), 0) == 0);
        AMREX_ALWAYS_ASSERT(FieldMaxDifference(
            report.candidate_uncancelled_energy_scale, replay.LastScale(), 0) == 0);
        AMREX_ALWAYS_ASSERT(SameSnapshot(snapshot(), before));

        // Opposite local energy errors cancel globally but must reject cell-wise.
        auto* const accepted_pointer = sentinel.get();
        auto reject_options = options;
        reject_options.max_iterations = 1;
        replay.Reset();
        replay.mode = ReplayMode::OffsettingEnergyErrors;
        PacketMaterialWorkIterationReport reject_report;
        AMREX_ALWAYS_ASSERT(!TryIteratePacketMaterialWork(material, path,
            report.candidate_total_impulse, std::ref(replay), reject_options, sentinel,
            reject_report));
        AMREX_ALWAYS_ASSERT(sentinel.get() == accepted_pointer);
        AMREX_ALWAYS_ASSERT(reject_report.reason ==
            PacketMaterialWorkIterationReason::NotConverged);
        AMREX_ALWAYS_ASSERT(reject_report.max_radiation_work_residual > 0.01);
        AMREX_ALWAYS_ASSERT(SameSnapshot(snapshot(), before));

        amrex::Real const cell_mass = FieldSum(sentinel->MaterialMass(), 0) / n_cells;
        AMREX_ALWAYS_ASSERT(std::abs(cell_mass*n_cells-initial_material.mass) <=
                            1.e-13L*initial_material.mass);
        std::vector<AnalyticPhoton> photons(num_photons);
        photons[0].energy = replay.PhotonEnergy(0);
        photons[0].incoming = Direction(0.2L, -0.3L, 0.87L);
        photons[0].frame_outgoing = Direction(0.35L, 0.25L, 0.8L);
        photons[1].energy = replay.PhotonEnergy(1);
        photons[1].incoming = Direction(-0.1L, 0.45L, 0.75L);
        photons[1].frame_outgoing = Direction(-0.2L, -0.4L, 0.8L);
        AnalyticSolution const analytic = SolveAnalytic(
            photons, initial_u_long, static_cast<long double>(cell_mass), speed, 0.5L);
        AMREX_ALWAYS_ASSERT(analytic.iterations > 0);
        for (int d = 0; d < 3; ++d)
        {
            long double const production = static_cast<long double>(
                FieldSum(report.candidate_total_impulse, d) / source_cells);
            AMREX_ALWAYS_ASSERT(std::abs(production - analytic.impulse[d]) <=
                1.e-11L * (std::abs(analytic.impulse[d]) + std::abs(production)));
        }

        sentinel->Commit();
        AMREX_ALWAYS_ASSERT(SameSnapshot(snapshot(), before));
        AMREX_ALWAYS_ASSERT(material.Commit());
        int changed = !SameSnapshot(snapshot(), before);
        amrex::ParallelDescriptor::ReduceIntMax(changed);
        AMREX_ALWAYS_ASSERT(changed != 0);

        MaterialTotals const final_material = inventory();
        auto const carry = ParticleImpulseInventory(particles, material_names, path);
        for (std::size_t s = 0; s < species.size(); ++s) {
            auto const current = MaterialInventory(*species[s], path, speed);
            auto const remainder = ParticleImpulseInventory(particles, {material_names[s]}, path);
            auto const share = species_before[s].mass/initial_material.mass;
            for (int d = 0; d < 3; ++d) {
                auto const actual = current.momentum[d]-species_before[s].momentum[d]+remainder[d];
                auto const expected = share*source_cells*analytic.impulse[d];
                AMREX_ALWAYS_ASSERT(std::abs(actual-expected) <=
                                    1.e-11L*(std::abs(actual)+std::abs(expected)));
            }
            auto const actual = current.kinetic_energy-species_before[s].kinetic_energy+remainder[3];
            auto const expected = share*source_cells*analytic.radiation_debit;
            AMREX_ALWAYS_ASSERT(std::abs(actual-expected) <=
                                1.e-11L*(std::abs(actual)+std::abs(expected)));
        }
        for (int d = 0; d < 3; ++d)
        {
            long double const momentum_change =
                final_material.momentum[d] - initial_material.momentum[d] + carry[d];
            long double const photon_impulse =
                static_cast<long double>(source_cells) * analytic.impulse[d];
            AMREX_ALWAYS_ASSERT(std::abs(momentum_change - photon_impulse) <=
                1.e-11L * (std::abs(photon_impulse) + std::abs(momentum_change)));
        }
        long double const material_energy_change =
            final_material.kinetic_energy - initial_material.kinetic_energy;
        long double const expected_energy_debit =
            static_cast<long double>(source_cells) * analytic.radiation_debit;
        long double const production_debit = static_cast<long double>(
            FieldSum(report.candidate_radiation_debit, 0));
        AMREX_ALWAYS_ASSERT(std::abs(material_energy_change + carry[3] - expected_energy_debit) <=
            1.e-11L * (std::abs(expected_energy_debit) +
                       std::abs(material_energy_change)));
        AMREX_ALWAYS_ASSERT(std::abs(production_debit - expected_energy_debit) <=
            1.e-11L * (std::abs(expected_energy_debit) + std::abs(production_debit)));
        AMREX_ALWAYS_ASSERT(std::abs(
            static_cast<long double>(report.actual_work) -
            material_energy_change) <=
            1.e-11L * (std::abs(material_energy_change) +
                       std::abs(static_cast<long double>(report.actual_work))));
        AMREX_ALWAYS_ASSERT(report.work_partition_residual == 0);
        AMREX_ALWAYS_ASSERT(std::abs(report.energy_carry_change-carry[3]) <=
            256*std::numeric_limits<amrex::Real>::epsilon()
                *(std::abs(report.energy_carry_change)+std::abs(carry[3])));

        amrex::Print() << "Packet/material finite-mass fixed point passed with "
                       << report.iterations << " coordinator iterations and "
                       << analytic.iterations << " long-double reference iterations; "
                       << "residuals: impulse=" << report.max_impulse_residual
                       << " photon_work=" << report.max_radiation_work_residual
                       << " material_work=" << report.max_material_work_residual
                       << " material_impulse=" << report.max_material_impulse_residual << '\n';
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
