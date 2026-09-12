/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleImpulseBoundary.H"

#include "MaterialKineticWork.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaries.H"
#include "Particles/ParticleBoundaries_K.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "RadiationTransport.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>

namespace warpx::radiation
{
    std::vector<std::string> RegisteredParticleImpulsePaths (WarpXParticleContainer &species)
    {
        std::string const prefix = "radiation_impulse_";
        std::string const suffix = "_work";
        std::vector<std::string> paths;
        for (auto const &name : species.GetRealSoANames()) {
            if (name.starts_with(prefix) && name.ends_with(suffix)) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(name.size() > prefix.size() + suffix.size(),
                                                 "Empty or malformed radiation carry path.");
                auto const path =
                    name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!path.empty(), "Empty radiation carry path.");
                for (auto const *component : {"_ux", "_uy", "_uz"}) {
                    amrex::ignore_unused(species.GetRealCompIndex(prefix + path + component));
                }
                paths.push_back(path);
            }
        }
        return paths;
    }

    namespace
    {
        struct Candidate {
            amrex::GpuArray<amrex::ParticleReal, 3> position{}, velocity{};
            amrex::GpuArray<bool, 3> reflected{};
            int valid = 0;
        };
        using CarryPointers = amrex::GpuArray<amrex::ParticleReal *, 4>;
        struct TileTrial {
            WarpXParticleContainer::ParticleTileType *tile = nullptr;
            amrex::Gpu::DeviceVector<Candidate> candidates;
            amrex::Gpu::DeviceVector<CarryPointers> carries;
        };
    } // namespace

    bool TryReflectParticleImpulseState (
        WarpXParticleContainer &species, ParticleBoundaries const &boundaries,
        std::vector<ParticleImpulseBoundaryTransfer> &transfers,
        std::function<bool(std::vector<ParticleImpulseBoundaryTransfer> const &)> const &accept)
    {
        auto const paths = RegisteredParticleImpulsePaths(species);
        if (paths.empty() || WarpX::do_moving_window || species.finestLevel() != 0 ||
            species.Geom(0).Coord() != 0 || std::numeric_limits<amrex::Real>::digits < 53 ||
            std::numeric_limits<amrex::ParticleReal>::digits < 53) {
            return false;
        }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(boundaries, transfers, accept);
        return false;
#else
        auto const settings = boundaries.data;
        amrex::GpuArray<ParticleBoundaryType, 3> const lower_type{
            settings.xmin_bc, settings.ymin_bc, settings.zmin_bc};
        amrex::GpuArray<ParticleBoundaryType, 3> const upper_type{
            settings.xmax_bc, settings.ymax_bc, settings.zmax_bc};
        auto const &geometry = species.Geom(0);
        amrex::XDim3 lo{}, hi{};
        amrex::GpuArray<bool, 3> periodic{true, true, true};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
#if defined(WARPX_DIM_1D_Z)
            int const axis = 2;
#elif defined(WARPX_DIM_XZ)
            int const axis = d == 0 ? 0 : 2;
#else
            int const axis = d;
#endif
            auto const expected = geometry.isPeriodic(d) ? ParticleBoundaryType::Periodic
                                                         : ParticleBoundaryType::Reflecting;
            // Use the actual species boundary settings, including reflect-all.
            if (lower_type[axis] != expected || upper_type[axis] != expected) {
                return false;
            }
            periodic[axis] = geometry.isPeriodic(d);
        }
#ifndef WARPX_DIM_1D_Z
        lo.x = geometry.ProbLo(0);
        hi.x = geometry.ProbHi(0);
#endif
#ifdef WARPX_DIM_3D
        lo.y = geometry.ProbLo(1);
        hi.y = geometry.ProbHi(1);
#endif
#if defined(WARPX_ZINDEX)
        lo.z = geometry.ProbLo(WARPX_ZINDEX);
        hi.z = geometry.ProbHi(WARPX_ZINDEX);
#endif
        amrex::GpuArray<amrex::Real, 3> const lower{lo.x, lo.y, lo.z}, upper{hi.x, hi.y, hi.z};
        auto const mass = species.getMass();
        if (!(mass > 0) || !std::isfinite(mass)) {
            return false;
        }
        int const count = static_cast<int>(paths.size());
        std::vector<std::unique_ptr<TileTrial>> trials;
        for (WarpXParIter iterator(species, 0); iterator.isValid(); ++iterator) {
            auto trial = std::make_unique<TileTrial>();
            trial->tile = &iterator.GetParticleTile();
            auto const np = iterator.numParticles();
            trial->candidates.resize(np);
            amrex::Gpu::HostVector<CarryPointers> host(count);
            for (int group = 0; group < count; ++group) {
                std::array<std::string, 4> const suffix{"ux", "uy", "uz", "work"};
                for (int d = 0; d < 4; ++d) {
                    host[group][d] = iterator.GetStructOfArrays()
                                         .GetRealData(species.GetRealCompIndex(
                                             "radiation_impulse_" + paths[group] + "_" + suffix[d]))
                                         .data();
                }
            }
            trial->carries.resize(count);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(),
                             trial->carries.begin());
            auto const data = iterator.GetParticleTile().getParticleTileData();
            auto const get_position = GetParticlePosition<PIdx>(iterator);
            auto *output = trial->candidates.data();
            auto const *carry = trial->carries.data();
            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE(long ip,
                                                           amrex::RandomEngine const &engine) {
                Candidate next;
                get_position.AsStored(ip, next.position[0], next.position[1], next.position[2]);
                next.valid = amrex::ParticleIDWrapper{data.m_idcpu[ip]}.is_valid();
                for (int d = 0; d < 3; ++d) {
                    next.velocity[d] = data.m_rdata[PIdx::ux + d][ip];
                    next.valid = next.valid && amrex::Math::isfinite(next.position[d]) &&
                                 amrex::Math::isfinite(next.velocity[d]);
                }
                auto const weight_mass = data.m_rdata[PIdx::w][ip] * mass;
                next.valid = next.valid && weight_mass > 0 && amrex::Math::isfinite(weight_mass);
                if (!next.valid) {
                    output[ip] = next;
                    return;
                }
                bool lost = false;
                ApplyParticleBoundaries::BoundaryEvent event;
                ApplyParticleBoundaries::apply_boundaries(
                    next.position[0], next.position[1], next.position[2], lo, hi, next.velocity[0],
                    next.velocity[1], next.velocity[2], lost, settings, engine, &event);
                next.reflected = event.coordinate_reflection;
                for (int d = 0; d < 3; ++d) {
                    next.valid = next.valid && (periodic[d] || (next.position[d] >= lower[d] &&
                                                                next.position[d] <= upper[d]));
                }
                for (int group = 0; group < count; ++group) {
                    amrex::GpuArray<amrex::Real, 4> values{};
                    for (int d = 0; d < 4; ++d) {
                        values[d] = carry[group][d][ip];
                    }
                    auto const candidate = EvaluateMaterialCarryReflection(values, next.reflected,
                                                                           event.thermalized, lost);
                    next.valid = next.valid && candidate.valid;
                    for (int d = 0; d < 3; ++d) {
                        next.valid =
                            next.valid &&
                            amrex::Math::isfinite(weight_mass * candidate.boundary_transfer[d]);
                    }
                }
                output[ip] = next;
            });
            trials.push_back(std::move(trial));
        }
        std::vector<ParticleImpulseBoundaryTransfer> accepted;
        for (int group = 0; group < count; ++group) {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpMin>
                ops;
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, int> sums(ops);
            using Tuple = typename decltype(sums)::Type;
            for (auto const &trial : trials) {
                auto const *candidates = trial->candidates.data();
                auto const *carries = trial->carries.data();
                auto const data = trial->tile->getParticleTileData();
                ops.eval(trial->candidates.size(), sums, [=] AMREX_GPU_DEVICE(long ip) -> Tuple {
                    auto const &candidate = candidates[ip];
                    if (!candidate.valid) {
                        return {0, 0, 0, 0};
                    }
                    amrex::GpuArray<amrex::Real, 3> transfer{};
                    for (int d = 0; d < 3; ++d) {
                        if (candidate.reflected[d]) {
                            transfer[d] =
                                data.m_rdata[PIdx::w][ip] * mass * (2 * carries[group][d][ip]);
                        }
                    }
                    return {transfer[0], transfer[1], transfer[2], 1};
                });
            }
            auto const sum = sums.value();
            int valid = amrex::get<3>(sum);
            amrex::ParallelDescriptor::ReduceIntMin(valid);
            amrex::GpuArray<amrex::Real, 3> momentum{amrex::get<0>(sum), amrex::get<1>(sum),
                                                     amrex::get<2>(sum)};
            amrex::ParallelDescriptor::ReduceRealSum(momentum.data(), 3);
            for (auto value : momentum) {
                valid = valid && std::isfinite(value);
            }
            if (!valid) {
                return false;
            }
            accepted.push_back({paths[group], momentum});
        }
        int accepted_by_caller = !accept || accept(accepted);
        amrex::ParallelDescriptor::ReduceIntMin(accepted_by_caller);
        if (!accepted_by_caller) {
            return false;
        }
        // Only exact sign changes are repeated here, not a force/work solve or sum.
        for (auto const &trial : trials) {
            auto const *candidates = trial->candidates.data();
            auto const *carry = trial->carries.data();
            auto const data = trial->tile->getParticleTileData();
            amrex::ParallelFor(trial->candidates.size(), [=] AMREX_GPU_DEVICE(long ip) {
                auto p = WarpXParticleContainer::ParticleType(data, ip);
                auto const &candidate = candidates[ip];
#if defined(WARPX_DIM_1D_Z)
                p.pos(0) = candidate.position[2];
#elif defined(WARPX_DIM_XZ)
                p.pos(0) = candidate.position[0]; p.pos(1) = candidate.position[2];
#else
                for (int d = 0; d < 3; ++d) { p.pos(d) = candidate.position[d]; }
#endif
                for (int d = 0; d < 3; ++d) {
                    data.m_rdata[PIdx::ux + d][ip] = candidate.velocity[d];
                    if (candidate.reflected[d]) {
                        for (int group = 0; group < count; ++group) {
                            carry[group][d][ip] = -carry[group][d][ip];
                        }
                    }
                }
            });
        }
        amrex::Gpu::streamSynchronize();
        transfers = std::move(accepted);
        return true;
#endif
    }
} // namespace warpx::radiation

namespace
{
    using CarryWallKey = std::pair<std::string, std::string>;
    std::vector<CarryWallKey> CarryWallKeys ()
    {
        std::vector<CarryWallKey> keys;
        auto &particles = WarpX::GetInstance().GetPartContainer();
        for (auto const &name : particles.GetSpeciesNames()) {
            auto &species = particles.GetParticleContainerFromName(name);
            for (auto const &path : warpx::radiation::RegisteredParticleImpulsePaths(species)) {
                keys.emplace_back(name, path);
            }
        }
        return keys;
    }
} // namespace

bool RadiationTransport::ReflectParticleCarryBoundaries (WarpXParticleContainer &species,
                                                         ParticleBoundaries const &boundaries)
{
    auto candidate = m_particle_carry_wall_momentum;
    std::vector<warpx::radiation::ParticleImpulseBoundaryTransfer> transfers;
    auto accept = [&] (auto const &proposed) {
        for (auto const &transfer : proposed) {
            auto &value = candidate[{species.getName(), transfer.path}];
            for (int d = 0; d < 3; ++d) {
                auto const old = value.sum[d];
                auto const increment = transfer.momentum[d];
                auto const next = old + increment;
                if (!std::isfinite(next)) {
                    return false;
                }
                value.correction[d] += std::abs(old) >= std::abs(increment)
                                           ? (old - next) + increment
                                           : (increment - next) + old;
                value.sum[d] = next;
                if (!std::isfinite(value.correction[d]) ||
                    !std::isfinite(next + value.correction[d])) {
                    return false;
                }
            }
        }
        return true;
    };
    if (!warpx::radiation::TryReflectParticleImpulseState(species, boundaries, transfers, accept)) {
        return false;
    }
    m_particle_carry_wall_momentum.swap(candidate);
    return true;
}

amrex::GpuArray<amrex::Real, 3>
RadiationTransport::particleCarryWallMomentum (std::string const &species,
                                               std::string const &path) const
{
    auto const found = m_particle_carry_wall_momentum.find({species, path});
    amrex::GpuArray<amrex::Real, 3> result{};
    if (found != m_particle_carry_wall_momentum.end()) {
        for (int d = 0; d < 3; ++d) {
            result[d] = found->second.sum[d] + found->second.correction[d];
        }
    }
    return result;
}

void RadiationTransport::WriteParticleCarryWallCheckpoint (std::string const &directory) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }
    auto const keys = CarryWallKeys();
    for (auto const &[key, value] : m_particle_carry_wall_momentum) {
        amrex::ignore_unused(value);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::find(keys.begin(), keys.end(), key) != keys.end(),
                                         "Cannot drop particle carry wall owners at checkpoint.");
    }
    if (keys.empty()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_particle_carry_wall_momentum.empty(),
                                         "Cannot drop particle carry wall owners at checkpoint.");
        return;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::numeric_limits<amrex::Real>::digits >= 53 &&
                                         std::numeric_limits<amrex::ParticleReal>::digits >= 53,
                                     "Particle carry wall history requires double precision.");
    std::ofstream output(directory + "/RadiationCarryWallMomentum_data.txt");
    output << "particle_carry_wall_v2 " << keys.size() << '\n'
           << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    for (auto const &[species, path] : keys) {
        auto const found = m_particle_carry_wall_momentum.find({species, path});
        auto const value =
            found == m_particle_carry_wall_momentum.end() ? ParticleCarryWallSum{} : found->second;
        output << std::quoted(species) << ' ' << std::quoted(path);
        for (int d = 0; d < 3; ++d) {
            output << ' ' << value.sum[d] << ' ' << value.correction[d];
        }
        output << '\n';
    }
    output.flush();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(output.good(),
                                     "Could not checkpoint particle carry wall ledger.");
}

void RadiationTransport::ReadParticleCarryWallCheckpoint (std::string const &directory)
{
    auto const keys = CarryWallKeys();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(keys.empty() ||
                                         (std::numeric_limits<amrex::Real>::digits >= 53 &&
                                          std::numeric_limits<amrex::ParticleReal>::digits >= 53),
                                     "Particle carry wall history requires double precision.");
    auto const file = directory + "/RadiationCarryWallMomentum_data.txt";
    int exists = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        exists = amrex::FileExists(file);
    }
    amrex::ParallelDescriptor::Bcast(&exists, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    decltype(m_particle_carry_wall_momentum) restored;
    if (!exists) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            keys.empty() || WarpX::GetInstance().Geom(0).isAllPeriodic(),
            "Nonperiodic particle carry restart requires a wall ledger.");
        m_particle_carry_wall_momentum.swap(restored);
        return;
    }
    amrex::Vector<char> buffer;
    amrex::ParallelDescriptor::ReadAndBcastFile(file, buffer);
    std::istringstream input(std::string(buffer.data()));
    std::string version;
    std::size_t count = 0;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (input >> version >> count) &&
            (version == "particle_carry_wall_v1" || version == "particle_carry_wall_v2") &&
            count == keys.size(),
        "Invalid particle carry wall ledger schema or owners.");
    for (auto const &key : keys) {
        std::string species, path;
        ParticleCarryWallSum momentum;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((input >> std::quoted(species) >> std::quoted(path)) &&
                                             (key == CarryWallKey{species, path}),
                                         "Particle carry wall ledger owner changed.");
        for (int d = 0; d < 3; ++d) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE((input >> momentum.sum[d]) &&
                                                 std::isfinite(momentum.sum[d]),
                                             "Invalid particle carry wall momentum.");
            if (version == "particle_carry_wall_v2") {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (input >> momentum.correction[d]) && std::isfinite(momentum.correction[d]) &&
                        std::isfinite(momentum.sum[d] + momentum.correction[d]),
                    "Invalid particle carry wall correction.");
            }
        }
        restored.emplace(key, momentum);
    }
    input >> std::ws;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(input.eof(), "Trailing particle carry wall ledger metadata.");
    m_particle_carry_wall_momentum.swap(restored);
}
