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
#include "RZSpecularTrajectory.H"
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
            amrex::GpuArray<amrex::Real, 4> transverse_reflection{1, 0, 0, 1};
            bool angular_reflection = false;
            int valid = 0;
        };
        using CarryPointers = amrex::GpuArray<amrex::ParticleReal *, 4>;
        struct TileTrial {
            WarpXParticleContainer::ParticleTileType *tile = nullptr;
            amrex::Gpu::DeviceVector<Candidate> candidates;
            amrex::Gpu::DeviceVector<CarryPointers> carries;
        };

        [[maybe_unused]] AMREX_GPU_HOST_DEVICE MaterialCarryReflection
        ReflectCarry (amrex::GpuArray<amrex::Real, 4> const& values,
                      Candidate const& candidate, bool thermalized = false, bool lost = false)
        {
#if defined(WARPX_DIM_RZ)
            if (candidate.angular_reflection) {
                auto result = EvaluateMaterialCarryReflection(values, {false, false, false},
                                                              thermalized, lost);
                if (!result.valid) { return result; }
                auto const& matrix = candidate.transverse_reflection;
                result.carry[0] = matrix[0] * values[0] + matrix[1] * values[1];
                result.carry[1] = matrix[2] * values[0] + matrix[3] * values[1];
                for (int d = 0; d < 2; ++d) {
                    result.boundary_transfer[d] = values[d] - result.carry[d];
                    if (!std::isfinite(result.carry[d]) ||
                        !std::isfinite(result.boundary_transfer[d])) { return {}; }
                }
                return result;
            }
            return EvaluateCylindricalMaterialCarryReflection(
                values, candidate.reflected, candidate.position[1], thermalized, lost);
#else
            return EvaluateMaterialCarryReflection(values, candidate.reflected, thermalized, lost);
#endif
        }
    } // namespace

    bool TryReflectParticleImpulseState (
        WarpXParticleContainer &species, ParticleBoundaries const &boundaries,
        std::vector<ParticleImpulseBoundaryTransfer> &transfers,
        std::function<bool(std::vector<ParticleImpulseBoundaryTransfer> const &)> const &accept,
        bool const angular_trajectory)
    {
        auto const paths = RegisteredParticleImpulsePaths(species);
        if (paths.empty() || WarpX::do_moving_window || species.finestLevel() != 0 ||
            std::numeric_limits<amrex::Real>::digits < 53 ||
            std::numeric_limits<amrex::ParticleReal>::digits < 53) {
            return false;
        }
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(boundaries, transfers, accept, angular_trajectory);
        return false;
#else
        auto const settings = boundaries.data;
        amrex::GpuArray<ParticleBoundaryType, 3> const lower_type{
            settings.xmin_bc, settings.ymin_bc, settings.zmin_bc};
        amrex::GpuArray<ParticleBoundaryType, 3> const upper_type{
            settings.xmax_bc, settings.ymax_bc, settings.zmax_bc};
        auto const &geometry = species.Geom(0);
#if !defined(WARPX_DIM_RZ)
        if (angular_trajectory) { return false; }
#endif
#if defined(WARPX_DIM_RZ)
        if (angular_trajectory && (!geometry.isPeriodic(1) || settings.reflect_all_velocities)) {
            return false;
        }
        if (angular_trajectory) {
            auto const& names = species.GetRealSoANames();
            for (auto const* required : {"prev_x", "prev_y"}) {
                if (std::find(names.begin(), names.end(), required) == names.end()) { return false; }
            }
        }
        // WarpX's compiled RZ geometry supplies the cylindrical interpretation;
        // its AMReX Geometry need not carry CoordType::RZ.
        if (geometry.ProbLo(0) != 0 || geometry.isPeriodic(0) ||
            lower_type[0] != ParticleBoundaryType::None ||
            upper_type[0] != ParticleBoundaryType::Reflecting) { return false; }
#else
        if (geometry.Coord() != 0) { return false; }
#endif
        amrex::XDim3 lo{}, hi{};
        amrex::GpuArray<bool, 3> periodic{true, true, true};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
#if defined(WARPX_DIM_1D_Z)
            int const axis = 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            int const axis = d == 0 ? 0 : 2;
#else
            int const axis = d;
#endif
#if defined(WARPX_DIM_RZ)
            if (d == 0) {
                periodic[axis] = false;
                continue;
            }
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
#if defined(WARPX_DIM_RZ)
            amrex::GpuArray<amrex::ParticleReal const*, 2> previous{};
            if (angular_trajectory) {
                previous = {iterator.GetAttribs("prev_x").dataPtr(),
                            iterator.GetAttribs("prev_y").dataPtr()};
            }
#endif
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
#if defined(WARPX_DIM_RZ)
                if (angular_trajectory && next.position[0] > upper[0]) {
                    auto const theta = next.position[1];
                    // The trajectory/ledger uses field precision. The native angular
                    // contract requires DP fields; mixed SP/DP builds must still compile.
                    auto const path = ReflectRZTrajectory(
                        {static_cast<amrex::Real>(previous[0][ip]),
                         static_cast<amrex::Real>(previous[1][ip])},
                        {static_cast<amrex::Real>(next.position[0] * std::cos(theta)),
                         static_cast<amrex::Real>(next.position[0] * std::sin(theta))},
                        upper[0]);
                    if (!path.valid) { next.valid = false; output[ip] = next; return; }
                    next.angular_reflection = true;
                    next.transverse_reflection = path.reflection;
                    next.position[0] = static_cast<amrex::ParticleReal>(amrex::min(upper[0],
                        std::hypot(path.position[0], path.position[1])));
                    next.position[1] = static_cast<amrex::ParticleReal>(std::atan2(path.position[1], path.position[0]));
                    auto const ux = next.velocity[0], uy = next.velocity[1];
                    next.velocity[0] = static_cast<amrex::ParticleReal>(path.reflection[0] * ux + path.reflection[1] * uy);
                    next.velocity[1] = static_cast<amrex::ParticleReal>(path.reflection[2] * ux + path.reflection[3] * uy);
                }
                bool const outside = next.position[0] < lower[0] || next.position[0] > upper[0]
                    || next.position[2] < lower[2] || next.position[2] > upper[2];
                auto const cosine = std::cos(next.position[1]);
                auto const sine = std::sin(next.position[1]);
                if (outside) {
                    auto const x = next.velocity[0];
                    auto const y = next.velocity[1];
                    next.velocity[0] = x * cosine + y * sine;
                    next.velocity[1] = -x * sine + y * cosine;
                }
#endif
                ApplyParticleBoundaries::apply_boundaries(
                    next.position[0], next.position[1], next.position[2], lo, hi, next.velocity[0],
                    next.velocity[1], next.velocity[2], lost, settings, engine, &event);
#if defined(WARPX_DIM_RZ)
                if (outside) {
                    auto const radial = next.velocity[0];
                    auto const azimuthal = next.velocity[1];
                    next.velocity[0] = radial * cosine - azimuthal * sine;
                    next.velocity[1] = radial * sine + azimuthal * cosine;
                }
#endif
                next.reflected = event.coordinate_reflection;
                for (int d = 0; d < 3; ++d) {
                    next.valid = next.valid && amrex::Math::isfinite(next.velocity[d]) &&
                        (periodic[d] || (next.position[d] >= lower[d] &&
                                                                next.position[d] <= upper[d]));
                }
                for (int group = 0; group < count; ++group) {
                    amrex::GpuArray<amrex::Real, 4> values{};
                    for (int d = 0; d < 4; ++d) {
                        values[d] = carry[group][d][ip];
                    }
                    auto const candidate = ReflectCarry(values, next, event.thermalized, lost);
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
                    amrex::GpuArray<amrex::Real, 4> values{};
                    for (int d = 0; d < 4; ++d) { values[d] = carries[group][d][ip]; }
                    auto const reflected = ReflectCarry(values, candidate);
                    for (int d = 0; d < 3; ++d) {
                        transfer[d] = data.m_rdata[PIdx::w][ip] * mass
                            * reflected.boundary_transfer[d];
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
        // Repeat only the accepted reflection map, not a force/work solve or sum.
        for (auto const &trial : trials) {
            auto const *candidates = trial->candidates.data();
            auto const *carry = trial->carries.data();
            auto const data = trial->tile->getParticleTileData();
            amrex::ParallelFor(trial->candidates.size(), [=] AMREX_GPU_DEVICE(long ip) {
                auto p = WarpXParticleContainer::ParticleType(data, ip);
                auto const &candidate = candidates[ip];
#if defined(WARPX_DIM_1D_Z)
                p.pos(0) = candidate.position[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                p.pos(0) = candidate.position[0]; p.pos(1) = candidate.position[2];
#if defined(WARPX_DIM_RZ)
                data.m_rdata[PIdx::theta][ip] = candidate.position[1];
#endif
#else
                for (int d = 0; d < 3; ++d) { p.pos(d) = candidate.position[d]; }
#endif
                for (int d = 0; d < 3; ++d) {
                    data.m_rdata[PIdx::ux + d][ip] = candidate.velocity[d];
                }
                for (int group = 0; group < count; ++group) {
                    amrex::GpuArray<amrex::Real, 4> values{};
                    for (int d = 0; d < 4; ++d) { values[d] = carry[group][d][ip]; }
                    auto const reflected = ReflectCarry(values, candidate);
                    for (int d = 0; d < 3; ++d) { carry[group][d][ip] = reflected.carry[d]; }
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
    if (!warpx::radiation::TryReflectParticleImpulseState(
            species, boundaries, transfers, accept, m_rz_angular_transport)) {
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
