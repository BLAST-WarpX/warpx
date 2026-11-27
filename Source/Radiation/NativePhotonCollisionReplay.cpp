/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativePhotonCollisionReplay.H"

#include "FiniteMassPhotonCollision.H"
#include "FrozenPhotonCollisionRate.H"
#include "NativePhotonReplayInternals.H"
#include "PhotonFourMomentum.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Box.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace warpx::radiation
{
using photon_replay_detail::CapturedTile;

using namespace amrex::literals;
using photon_replay_detail::CaptureParticles;
using photon_replay_detail::GlobalFailureCode;
using photon_replay_detail::NearestCell;
using photon_replay_detail::OpenUnitInterval;
using photon_replay_detail::PhotonDrawKey;
using photon_replay_detail::SameBits;
namespace
{
constexpr amrex::Real pi = 3.141592653589793238462643383279502884_rt;

void
SetRejection (NativePhotonCollisionReport& report, int code)
{
    report = NativePhotonCollisionReport{};
    switch (code)
    {
    case 1: report.reason = "invalid collective inputs"; break;
    case 2: report.reason = "invalid native source photon"; break;
    case 3: report.reason = "optical depth exceeded frozen bound"; break;
    case 4: report.reason = "collision event count exceeded frozen bound"; break;
    case 5: report.reason = "invalid frozen random draw"; break;
    case 6: report.reason = "invalid replay layout"; break;
    case 7: report.reason = "private photon clone does not match frozen snapshot"; break;
    case 8: report.reason = "invalid staged material velocity"; break;
    case 9: report.reason = "invalid replay collision"; break;
    case 10: report.reason = "replay conservation identity failed"; break;
    default: report.reason = "unknown native photon replay rejection"; break;
    }
}

struct LocalGrid
{
    amrex::Box box{};
    amrex::Vector<int> diffusion_mask;
    amrex::Vector<amrex::Real> opacity;
    amrex::Vector<amrex::Real> rate_velocity;
};

std::unordered_map<int, LocalGrid>
CaptureLocalGrids (
    amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& kinetic_scattering_opacity,
    amrex::MultiFab const* rate_velocity)
{
    amrex::iMultiFab compact_mask(diffusion_mask.boxArray(),
        diffusion_mask.DistributionMap(), 1, 0);
    amrex::MultiFab compact_opacity(kinetic_scattering_opacity.boxArray(),
        kinetic_scattering_opacity.DistributionMap(), 1, 0);
    amrex::iMultiFab::Copy(compact_mask, diffusion_mask, 0, 0, 1, 0);
    amrex::MultiFab::Copy(compact_opacity, kinetic_scattering_opacity, 0, 0, 1, 0);
    amrex::MultiFab compact_velocity(diffusion_mask.boxArray(),
        diffusion_mask.DistributionMap(), 3, 0);
    compact_velocity.setVal(0);
    if (rate_velocity != nullptr)
    {
        amrex::MultiFab::Copy(compact_velocity, *rate_velocity, 0, 0, 3, 0);
    }
    std::unordered_map<int, LocalGrid> grids;
    for (amrex::MFIter iterator(compact_mask); iterator.isValid(); ++iterator)
    {
        LocalGrid grid;
        grid.box = iterator.validbox();
        long const values = grid.box.numPts();
        grid.diffusion_mask.resize(static_cast<std::size_t>(values));
        grid.opacity.resize(static_cast<std::size_t>(values));
        grid.rate_velocity.resize(static_cast<std::size_t>(3*values));
        auto const& mask = compact_mask[iterator];
        auto const& opacity = compact_opacity[iterator];
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, mask.dataPtr(), mask.dataPtr() + values,
            grid.diffusion_mask.begin());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, opacity.dataPtr(), opacity.dataPtr() + values,
            grid.opacity.begin());
        auto const* velocity = compact_velocity[iterator].dataPtr();
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, velocity, velocity+3*values,
                         grid.rate_velocity.begin());
        grids.emplace(iterator.index(), std::move(grid));
    }
    return grids;
}

} // namespace

struct NativePhotonCollisionReplay::Impl
{
    amrex::Geometry geometry{};
    NativePhotonCollisionOptions options{};
    amrex::Real dt = 0;
    amrex::BoxArray box_array;
    amrex::DistributionMapping distribution_map;
    std::vector<CapturedTile> baseline;
    int number_real_components = 0;
    int number_int_components = 0;
    std::vector<std::string> real_names;
    std::vector<std::string> int_names;
    amrex::Long colliding_photons = 0;
    amrex::Long collision_events = 0;
    bool prepared = false;

    amrex::Gpu::DeviceVector<std::uint64_t> idcpu;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> weight;
    amrex::Gpu::DeviceVector<amrex::Real> incoming;
    amrex::Gpu::DeviceVector<int> event_count;
    amrex::Gpu::DeviceVector<long> event_offset;
    amrex::Gpu::DeviceVector<amrex::Real> event_direction;
    amrex::Gpu::DeviceVector<int> cell;
};

NativePhotonCollisionReplay::NativePhotonCollisionReplay ()
    : m_impl(std::make_unique<Impl>()) {}

NativePhotonCollisionReplay::~NativePhotonCollisionReplay () = default;

bool
NativePhotonCollisionReplay::Prepare (
    WarpXParticleContainer& photons, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& kinetic_scattering_opacity,
    amrex::Geometry const& geometry, amrex::Real dt,
    amrex::MultiFab const& rate_velocity, NativePhotonCollisionOptions const& options,
    NativePhotonCollisionReport& report)
{
    return PrepareImpl(photons, diffusion_mask, kinetic_scattering_opacity,
        geometry, dt, &rate_velocity, options, report);
}

bool
NativePhotonCollisionReplay::Prepare (
    WarpXParticleContainer& photons, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& kinetic_scattering_opacity,
    amrex::Geometry const& geometry, amrex::Real dt,
    NativePhotonCollisionOptions const& options,
    NativePhotonCollisionReport& report)
{
    return PrepareImpl(photons, diffusion_mask, kinetic_scattering_opacity,
        geometry, dt, nullptr, options, report);
}

bool
NativePhotonCollisionReplay::PrepareImpl (
    WarpXParticleContainer& photons, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& kinetic_scattering_opacity,
    amrex::Geometry const& geometry, amrex::Real dt,
    amrex::MultiFab const* rate_velocity, NativePhotonCollisionOptions const& options,
    NativePhotonCollisionReport& report)
{
    // Clear any previous plan first: a rejected Prepare never leaves a usable plan.
    m_impl = std::make_unique<Impl>();
    report = NativePhotonCollisionReport{};

    int invalid_inputs = 0;
    if constexpr (std::numeric_limits<amrex::Real>::digits < 53 ||
                  std::numeric_limits<amrex::ParticleReal>::digits < 53)
    {
        invalid_inputs = 1;
    }
    if (options.absorption || options.emission ||
        options.moving_cross_sections != (rate_velocity != nullptr) ||
        (options.angular_law != NativePhotonCollisionAngularLaw::Isotropic &&
         options.angular_law != NativePhotonCollisionAngularLaw::FixedFrameDirection) ||
        options.max_events_per_photon <= 0 ||
        !amrex::Math::isfinite(options.max_optical_depth) ||
        !(options.max_optical_depth > 0) || !amrex::Math::isfinite(dt) || !(dt > 0))
    {
        invalid_inputs = 1;
    }
    amrex::Real direction_squared = 0;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        if (!amrex::Math::isfinite(options.fixed_frame_direction[dimension]))
        {
            invalid_inputs = 1;
        }
        direction_squared += options.fixed_frame_direction[dimension] *
            options.fixed_frame_direction[dimension];
    }
    if (!amrex::Math::isfinite(direction_squared) || !(direction_squared > 0) ||
        std::abs(direction_squared - 1) >
            256 * std::numeric_limits<amrex::Real>::epsilon() * direction_squared)
    {
        invalid_inputs = 1;
    }

    amrex::Long integer_options[6] = {
        static_cast<amrex::Long>(options.seed),
        static_cast<amrex::Long>(options.interval),
        static_cast<amrex::Long>(options.max_events_per_photon),
        static_cast<amrex::Long>(options.angular_law ==
                                 NativePhotonCollisionAngularLaw::Isotropic),
        static_cast<amrex::Long>(options.absorption),
        static_cast<amrex::Long>(options.moving_cross_sections)};
    amrex::Long minimum_integers[6];
    amrex::Long maximum_integers[6];
    std::copy(std::begin(integer_options), std::end(integer_options), minimum_integers);
    std::copy(std::begin(integer_options), std::end(integer_options), maximum_integers);
    amrex::ParallelDescriptor::ReduceLongMin(minimum_integers, 6);
    amrex::ParallelDescriptor::ReduceLongMax(maximum_integers, 6);
    if (!std::equal(std::begin(minimum_integers), std::end(minimum_integers),
                    std::begin(maximum_integers)))
    {
        invalid_inputs = 1;
    }

    amrex::Real real_options[5] = {
        dt, options.max_optical_depth, options.fixed_frame_direction[0],
        options.fixed_frame_direction[1], options.fixed_frame_direction[2]};
    amrex::Real minimum_reals[5];
    amrex::Real maximum_reals[5];
    std::copy(std::begin(real_options), std::end(real_options), minimum_reals);
    std::copy(std::begin(real_options), std::end(real_options), maximum_reals);
    amrex::ParallelDescriptor::ReduceRealMin(minimum_reals, 5);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_reals, 5);
    if (!std::equal(std::begin(minimum_reals), std::end(minimum_reals),
                    std::begin(maximum_reals)))
    {
        invalid_inputs = 1;
    }
    int const global_invalid_inputs = GlobalFailureCode(invalid_inputs);
    if (global_invalid_inputs != 0)
    {
        SetRejection(report, global_invalid_inputs);
        return false;
    }

    int invalid_layout = 0;
    if (photons.finestLevel() != 0 || geometry.Coord() != 0 ||
        !photons.AmIA<PhysicalSpecies::photon>() || diffusion_mask.nComp() != 1 ||
        kinetic_scattering_opacity.nComp() != 1)
    {
        invalid_layout = 1;
    }
    auto const int_names = photons.GetIntSoANames();
    if (std::find(int_names.begin(), int_names.end(),
                  "__photon_birth_provisional") != int_names.end())
    {
        invalid_layout = 1;
    }
    auto const& photon_boxes = photons.ParticleBoxArray(0);
    auto const& photon_distribution = photons.ParticleDistributionMap(0);
    if (photon_boxes != diffusion_mask.boxArray() ||
        photon_distribution != diffusion_mask.DistributionMap() ||
        photon_boxes != kinetic_scattering_opacity.boxArray() ||
        photon_distribution != kinetic_scattering_opacity.DistributionMap())
    {
        invalid_layout = 1;
    }
    if (rate_velocity != nullptr && (rate_velocity->nComp() != 3 ||
        rate_velocity->boxArray() != photon_boxes ||
        rate_velocity->DistributionMap() != photon_distribution))
    {
        invalid_layout = 1;
    }
    int const global_invalid_layout = GlobalFailureCode(invalid_layout);
    if (global_invalid_layout != 0)
    {
        SetRejection(report, global_invalid_layout);
        return false;
    }

    amrex::Long geometry_integers[2 * AMREX_SPACEDIM];
    amrex::Real geometry_reals[2 * AMREX_SPACEDIM];
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        geometry_integers[dimension] = geometry.Domain().smallEnd(dimension);
        geometry_integers[AMREX_SPACEDIM + dimension] =
            geometry.Domain().length(dimension);
        geometry_reals[dimension] = geometry.ProbLo(dimension);
        geometry_reals[AMREX_SPACEDIM + dimension] = geometry.ProbHi(dimension);
    }
    amrex::Long minimum_geometry_integers[2 * AMREX_SPACEDIM];
    amrex::Long maximum_geometry_integers[2 * AMREX_SPACEDIM];
    amrex::Real minimum_geometry_reals[2 * AMREX_SPACEDIM];
    amrex::Real maximum_geometry_reals[2 * AMREX_SPACEDIM];
    std::copy(std::begin(geometry_integers), std::end(geometry_integers),
              minimum_geometry_integers);
    std::copy(std::begin(geometry_integers), std::end(geometry_integers),
              maximum_geometry_integers);
    std::copy(std::begin(geometry_reals), std::end(geometry_reals),
              minimum_geometry_reals);
    std::copy(std::begin(geometry_reals), std::end(geometry_reals),
              maximum_geometry_reals);
    amrex::ParallelDescriptor::ReduceLongMin(
        minimum_geometry_integers, 2 * AMREX_SPACEDIM);
    amrex::ParallelDescriptor::ReduceLongMax(
        maximum_geometry_integers, 2 * AMREX_SPACEDIM);
    amrex::ParallelDescriptor::ReduceRealMin(
        minimum_geometry_reals, 2 * AMREX_SPACEDIM);
    amrex::ParallelDescriptor::ReduceRealMax(
        maximum_geometry_reals, 2 * AMREX_SPACEDIM);
    if (!std::equal(std::begin(minimum_geometry_integers),
                    std::end(minimum_geometry_integers),
                    std::begin(maximum_geometry_integers)) ||
        !std::equal(std::begin(minimum_geometry_reals),
                    std::end(minimum_geometry_reals),
                    std::begin(maximum_geometry_reals)))
    {
        SetRejection(report, 1);
        return false;
    }

    auto const captured = CaptureParticles(photons);
    auto const local_grids = CaptureLocalGrids(
        diffusion_mask, kinetic_scattering_opacity, rate_velocity);
    auto const number_real_components = photons.NumRealComps();
    auto const number_int_components = photons.NumIntComps();

    std::vector<std::uint64_t> host_idcpu;
    std::vector<amrex::ParticleReal> host_weight;
    std::vector<amrex::Real> host_incoming;
    std::vector<int> host_event_count;
    std::vector<long> host_event_offset{0};
    std::vector<amrex::Real> host_event_direction;
    std::vector<int> host_cell;
    amrex::Long colliding_photons = 0;
    amrex::Long collision_events = 0;
    amrex::GpuArray<amrex::Real, 4> incoming_inventory{};

    int local_failure = 0;
    for (auto const& [grid_index, grid] : local_grids)
    {
        amrex::ignore_unused(grid_index);
        for (std::size_t cell_index = 0;
             cell_index < static_cast<std::size_t>(grid.opacity.size()); ++cell_index)
        {
            auto const chi = grid.opacity[cell_index];
            auto const region = grid.diffusion_mask[cell_index];
            amrex::Real b2 = 0;
            for (int d = 0; d < 3; ++d)
            {
                auto const beta = grid.rate_velocity[cell_index +
                    static_cast<std::size_t>(d)*grid.opacity.size()]/PhysConst::c;
                b2 += beta*beta;
            }
            if ((region != 0 && region != 1) || !std::isfinite(chi) || chi < 0 ||
                (region == 1 && chi != 0) || !std::isfinite(b2) || !(b2 < 1))
            {
                local_failure = 1;
            }
        }
    }
    long global_particle = 0;
    for (auto const& tile : captured.tiles)
    {
        long const count = tile.count;
        auto const component_value = [&](int component, long particle) {
            return tile.real_data[static_cast<std::size_t>(component) *
                                         static_cast<std::size_t>(count) +
                                     static_cast<std::size_t>(particle)];
        };
        for (long particle = 0; particle < count; ++particle, ++global_particle)
        {
            if (local_failure != 0) { continue; }
            if (!amrex::ConstParticleIDWrapper(
                    tile.idcpu[static_cast<std::size_t>(particle)]).is_valid())
            {
                local_failure = 2;
                continue;
            }
            auto const found_grid = local_grids.find(tile.key.first);
            if (found_grid == local_grids.end() ||
                photon_distribution[tile.key.first] !=
                    amrex::ParallelDescriptor::MyProc())
            {
                local_failure = 2;
                continue;
            }
#if defined(WARPX_DIM_3D)
            amrex::ParticleReal const position_x = component_value(PIdx::x, particle);
            amrex::ParticleReal const position_y = component_value(PIdx::y, particle);
#elif defined(WARPX_DIM_XZ)
            amrex::ParticleReal const position_x = component_value(PIdx::x, particle);
            amrex::ParticleReal const position_y = 0;
#else
            amrex::ParticleReal const position_x = 0;
            amrex::ParticleReal const position_y = 0;
#endif
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            amrex::ParticleReal const position_z = 0;
#else
            amrex::ParticleReal const position_z = component_value(PIdx::z, particle);
#endif
            amrex::IntVect cell{};
            if (!NearestCell(geometry, position_x, position_y, position_z, cell) ||
                !found_grid->second.box.contains(cell))
            {
                local_failure = 2;
                continue;
            }
            long const field_index = found_grid->second.box.index(cell);
            int const mask = found_grid->second.diffusion_mask[
                static_cast<std::size_t>(field_index)];
            amrex::Real const opacity = found_grid->second.opacity[
                static_cast<std::size_t>(field_index)];
            if ((mask != 0 && mask != 1) || !amrex::Math::isfinite(opacity) ||
                opacity < 0)
            {
                local_failure = 2;
                continue;
            }

            amrex::ParticleReal const weight = component_value(PIdx::w, particle);
            amrex::ParticleReal const ux = component_value(PIdx::ux, particle);
            amrex::ParticleReal const uy = component_value(PIdx::uy, particle);
            amrex::ParticleReal const uz = component_value(PIdx::uz, particle);
            auto const represented = PacketConversionFourMomentumFromPhoton(
                weight, ux, uy, uz);
            if (!represented.valid)
            {
                local_failure = 2;
                continue;
            }

            amrex::GpuArray<amrex::Real, 3> beta0{};
            amrex::GpuArray<amrex::Real, 3> planning_direction{};
            for (int d = 0; d < 3; ++d)
            {
                beta0[d] = found_grid->second.rate_velocity[
                    static_cast<std::size_t>(field_index) +
                    static_cast<std::size_t>(d)*found_grid->second.opacity.size()]/PhysConst::c;
                planning_direction[d] = represented.value[d+1]/represented.value[0];
            }
            auto rate = EvaluateFrozenPhotonCollisionRate(opacity, beta0, planning_direction);
            amrex::Real const optical_depth_target =
                rate.maximum_inverse_length * PhysConst::c * dt;
            if (!rate.valid || !amrex::Math::isfinite(optical_depth_target) ||
                optical_depth_target < 0 ||
                optical_depth_target > options.max_optical_depth)
            {
                local_failure = 3;
                continue;
            }

            int events = 0;
            amrex::Real elapsed = 0;
            std::uint64_t const idcpu = tile.idcpu[
                static_cast<std::size_t>(particle)];
            while (opacity > 0)
            {
                amrex::Real const waiting_depth = -std::log(OpenUnitInterval(
                    PhotonDrawKey(options.seed, idcpu, options.interval,
                                  static_cast<std::uint64_t>(events), 0)));
                if (!amrex::Math::isfinite(waiting_depth) || !(waiting_depth > 0))
                {
                    local_failure = 5;
                    break;
                }
                // The envelope above is only a rejection bound, never the
                // sampling rate. Recompute the directional rate after every
                // comoving outgoing draw using the frozen initial frame.
                auto const remaining_depth = (dt-elapsed)*PhysConst::c*rate.inverse_length;
                if (waiting_depth >= remaining_depth) { break; }
                if (events == options.max_events_per_photon)
                {
                    local_failure = 4;
                    break;
                }
                auto const waiting_time = waiting_depth/(PhysConst::c*rate.inverse_length);
                if (!std::isfinite(waiting_time) || !(waiting_time > 0) ||
                    !(elapsed+waiting_time > elapsed))
                {
                    local_failure = 5;
                    break;
                }
                elapsed += waiting_time;
                amrex::GpuArray<amrex::Real, 3> direction{};
                if (options.angular_law ==
                    NativePhotonCollisionAngularLaw::FixedFrameDirection)
                {
                    direction = options.fixed_frame_direction;
                }
                else
                {
                    amrex::Real const cosine =
                        2 * OpenUnitInterval(PhotonDrawKey(
                            options.seed, idcpu, options.interval,
                            static_cast<std::uint64_t>(events), 1)) - 1;
                    amrex::Real const azimuth = 2 * pi *
                        OpenUnitInterval(PhotonDrawKey(
                            options.seed, idcpu, options.interval,
                            static_cast<std::uint64_t>(events), 2));
                    amrex::Real const sine = std::sqrt(1 - cosine * cosine);
                    direction = {sine * std::cos(azimuth),
                                 sine * std::sin(azimuth), cosine};
                    amrex::Real norm_squared = 0;
                    for (int dimension = 0; dimension < 3; ++dimension)
                    {
                        norm_squared += direction[dimension] * direction[dimension];
                    }
                    if (!amrex::Math::isfinite(norm_squared) || !(norm_squared > 0))
                    {
                        local_failure = 5;
                        break;
                    }
                    amrex::Real const inverse_norm = 1 / std::sqrt(norm_squared);
                    for (int dimension = 0; dimension < 3; ++dimension)
                    {
                        direction[dimension] *= inverse_norm;
                    }
                }
                amrex::Real norm_squared = 0;
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    if (!amrex::Math::isfinite(direction[dimension]))
                    {
                        local_failure = 5;
                        break;
                    }
                    norm_squared += direction[dimension] * direction[dimension];
                }
                if (local_failure != 0) { break; }
                if (!amrex::Math::isfinite(norm_squared) || !(norm_squared > 0) ||
                    std::abs(norm_squared - 1) >
                        256 * std::numeric_limits<amrex::Real>::epsilon() *
                            norm_squared)
                {
                    local_failure = 5;
                    break;
                }
                host_event_direction.push_back(direction[0]);
                host_event_direction.push_back(direction[1]);
                host_event_direction.push_back(direction[2]);
                ++events;
                auto const laboratory = BoostFourVector(
                    FourVector{1, direction[0], direction[1], direction[2]}, beta0);
                for (int d = 0; d < 3; ++d)
                {
                    planning_direction[d] = laboratory[d+1]/laboratory[0];
                }
                rate = EvaluateFrozenPhotonCollisionRate(opacity, beta0, planning_direction);
                if (!rate.valid)
                {
                    local_failure = 5;
                    break;
                }
            }
            if (local_failure != 0) { continue; }
            host_event_count.push_back(events);
            host_event_offset.push_back(static_cast<long>(host_event_direction.size()) / 3);

            host_idcpu.push_back(idcpu);
            host_weight.push_back(weight);
            for (int component = 0; component < 4; ++component)
            {
                host_incoming.push_back(represented.value[component]);
                incoming_inventory[component] += represented.value[component];
            }
            for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
            {
                host_cell.push_back(cell[dimension]);
            }
            for (int dimension = AMREX_SPACEDIM; dimension < 3; ++dimension)
            {
                host_cell.push_back(0);
            }
            collision_events += events;
            if (events != 0) { ++colliding_photons; }
        }
    }

    int const global_failure = GlobalFailureCode(local_failure);
    if (global_failure != 0)
    {
        SetRejection(report, global_failure);
        return false;
    }

    auto plan = std::make_unique<Impl>();
    plan->geometry = geometry;
    plan->options = options;
    plan->dt = dt;
    plan->box_array = photon_boxes;
    plan->distribution_map = photon_distribution;
    plan->baseline = captured.tiles;
    plan->number_real_components = number_real_components;
    plan->number_int_components = number_int_components;
    plan->real_names = photons.GetRealSoANames();
    plan->int_names = photons.GetIntSoANames();
    plan->colliding_photons = colliding_photons;
    plan->collision_events = collision_events;
    plan->prepared = true;
    auto copy_to_device = [] (auto const& source, auto& destination) {
        destination.resize(source.size());
        if (!source.empty())
        {
            amrex::Gpu::copy(
                amrex::Gpu::hostToDevice, source.data(), source.data() + source.size(),
                destination.begin());
        }
    };
    copy_to_device(host_idcpu, plan->idcpu);
    copy_to_device(host_weight, plan->weight);
    copy_to_device(host_incoming, plan->incoming);
    copy_to_device(host_event_count, plan->event_count);
    copy_to_device(host_event_offset, plan->event_offset);
    copy_to_device(host_event_direction, plan->event_direction);
    copy_to_device(host_cell, plan->cell);

    amrex::ParallelDescriptor::ReduceLongSum(colliding_photons);
    amrex::ParallelDescriptor::ReduceLongSum(collision_events);
    plan->colliding_photons = colliding_photons;
    plan->collision_events = collision_events;
    amrex::ParallelDescriptor::ReduceRealSum(incoming_inventory.data(), 4);
    for (auto value : incoming_inventory)
    {
        if (!amrex::Math::isfinite(value))
        {
            SetRejection(report, 2);
            return false;
        }
    }
    report.valid = true;
    report.colliding_photons = colliding_photons;
    report.collision_events = collision_events;
    report.incoming_inventory = incoming_inventory;
    report.outgoing_inventory = incoming_inventory;
    m_impl = std::move(plan);
    return true;
}

bool
NativePhotonCollisionReplay::Replay (
    amrex::MultiFab const& work_velocity,
    WarpXParticleContainer::Base& private_photons,
    amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
    amrex::MultiFab& uncancelled_energy_scale,
    NativePhotonCollisionReport& report) const
{
    report = NativePhotonCollisionReport{};
    if (GlobalFailureCode(m_impl != nullptr && m_impl->prepared ? 0 : 1) != 0)
    {
        SetRejection(report, 1);
        return false;
    }

    int invalid_layout = 0;
    if (material_impulse.nComp() != 3 || photon_energy_debit.nComp() != 1 ||
        uncancelled_energy_scale.nComp() != 1 || work_velocity.nComp() != 3 ||
        material_impulse.boxArray() != m_impl->box_array ||
        photon_energy_debit.boxArray() != m_impl->box_array ||
        uncancelled_energy_scale.boxArray() != m_impl->box_array ||
        work_velocity.boxArray() != m_impl->box_array ||
        material_impulse.DistributionMap() != m_impl->distribution_map ||
        photon_energy_debit.DistributionMap() != m_impl->distribution_map ||
        uncancelled_energy_scale.DistributionMap() != m_impl->distribution_map ||
        work_velocity.DistributionMap() != m_impl->distribution_map ||
        private_photons.ParticleBoxArray(0) != m_impl->box_array ||
        private_photons.ParticleDistributionMap(0) != m_impl->distribution_map ||
        private_photons.NumRealComps() != m_impl->number_real_components ||
        private_photons.NumIntComps() != m_impl->number_int_components ||
        private_photons.GetRealSoANames() != m_impl->real_names ||
        private_photons.GetIntSoANames() != m_impl->int_names)
    {
        invalid_layout = 1;
    }
    int const global_invalid_layout = GlobalFailureCode(invalid_layout);
    if (global_invalid_layout != 0)
    {
        SetRejection(report, global_invalid_layout);
        return false;
    }

    auto const candidate_capture = CaptureParticles(private_photons);
    bool same_particles = true;
    if (candidate_capture.tiles.size() != m_impl->baseline.size())
    {
        same_particles = false;
    }
    else
    {
        for (std::size_t tile = 0; tile < m_impl->baseline.size(); ++tile)
        {
            if (candidate_capture.tiles[tile].key != m_impl->baseline[tile].key ||
                candidate_capture.tiles[tile].count != m_impl->baseline[tile].count)
            {
                same_particles = false;
                break;
            }
        }
    }
    if (same_particles)
    {
        for (std::size_t tile = 0; tile < m_impl->baseline.size(); ++tile)
        {
            if (!SameBits(candidate_capture.tiles[tile].real_data,
                          m_impl->baseline[tile].real_data) ||
                !SameBits(candidate_capture.tiles[tile].int_data,
                          m_impl->baseline[tile].int_data) ||
                !SameBits(candidate_capture.tiles[tile].idcpu,
                          m_impl->baseline[tile].idcpu))
            {
                same_particles = false;
                break;
            }
        }
    }
    int mismatched_particles = same_particles ? 0 : 7;
    int const global_mismatched_particles = GlobalFailureCode(mismatched_particles);
    if (global_mismatched_particles != 0)
    {
        SetRejection(report, global_mismatched_particles);
        return false;
    }

    amrex::Gpu::DeviceScalar<int> invalid_velocity(0);
    auto* const invalid_velocity_ptr = invalid_velocity.dataPtr();
    for (amrex::MFIter iterator(work_velocity); iterator.isValid(); ++iterator)
    {
        auto const velocity = work_velocity.const_array(iterator);
        amrex::For(iterator.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real velocity_squared = 0;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                amrex::Real const value = velocity(i, j, k, dimension);
                if (!amrex::Math::isfinite(value))
                {
                    amrex::HostDevice::Atomic::Add(invalid_velocity_ptr, 1);
                    return;
                }
                velocity_squared += value * value;
            }
            if (!(velocity_squared < PhysConst::c * PhysConst::c))
            {
                amrex::HostDevice::Atomic::Add(invalid_velocity_ptr, 1);
            }
        });
    }
    int invalid_staged_velocity = invalid_velocity.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(invalid_staged_velocity);
    if (invalid_staged_velocity != 0)
    {
        SetRejection(report, 8);
        return false;
    }

    amrex::MultiFab candidate_impulse(
        m_impl->box_array, m_impl->distribution_map, 3, 0);
    amrex::MultiFab candidate_debit(
        m_impl->box_array, m_impl->distribution_map, 1, 0);
    amrex::MultiFab candidate_scale(
        m_impl->box_array, m_impl->distribution_map, 1, 0);
    candidate_impulse.setVal(0);
    candidate_debit.setVal(0);
    candidate_scale.setVal(0);

    long const number_particles = static_cast<long>(m_impl->idcpu.size());
    amrex::Gpu::DeviceVector<amrex::Real> candidate_u(
        static_cast<std::size_t>(number_particles) * 3, 0);
    amrex::Gpu::DeviceVector<amrex::Real> sums(20, 0);
    amrex::Gpu::DeviceScalar<int> invalid_collision(0);

    std::map<std::pair<int, int>, long> tile_offsets;
    long next_offset = 0;
    for (auto const& tile : m_impl->baseline)
    {
        tile_offsets.emplace(tile.key, next_offset);
        next_offset += tile.count;
    }

    for (auto const& tile : m_impl->baseline)
    {
        long const offset = tile_offsets.at(tile.key);
        long const count = tile.count;
        if (count == 0) { continue; }

        auto const velocity = work_velocity[tile.key.first].const_array();
        auto const impulse = candidate_impulse[tile.key.first].array();
        auto const debit = candidate_debit[tile.key.first].array();
        auto const scale = candidate_scale[tile.key.first].array();
        auto const original = private_photons.GetParticles(0).at(tile.key).getParticleTileData();
        amrex::Real const* const incoming = m_impl->incoming.dataPtr();
        amrex::ParticleReal const* const weight = m_impl->weight.dataPtr();
        int const* const event_count = m_impl->event_count.dataPtr();
        long const* const event_offset = m_impl->event_offset.dataPtr();
        amrex::Real const* const event_direction = m_impl->event_direction.dataPtr();
        int const* const cell = m_impl->cell.dataPtr();
        amrex::Real* const next_u = candidate_u.dataPtr();
        amrex::Real* const sum = sums.dataPtr();
        auto* const invalid_ptr = invalid_collision.dataPtr();
        amrex::For(count, [=] AMREX_GPU_DEVICE (long particle)
        {
            long const source = offset + particle;
            FourVector current{};
            for (int component = 0; component < 4; ++component)
            {
                current[component] = incoming[4 * source + component];
            }
            amrex::GpuArray<amrex::Real, 3> beta{};
            int const i = cell[3 * source];
            int const j = cell[3 * source + 1];
            int const k = cell[3 * source + 2];
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                beta[dimension] = velocity(i, j, k, dimension) / PhysConst::c;
            }
            int const events = event_count[source];
            long const first_event = event_offset[source];
            for (int event = 0; event < events; ++event)
            {
                amrex::GpuArray<amrex::Real, 3> direction{};
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    direction[dimension] =
                        event_direction[3 * (first_event + event) + dimension];
                }
                FourVector outgoing{};
                amrex::GpuArray<amrex::Real, 3> event_impulse{};
                amrex::Real event_debit = 0;
                if (!CoherentPhotonCollision(
                        current, direction, beta, outgoing, event_impulse, event_debit))
                {
                    amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                    return;
                }
                current = outgoing;
            }

            amrex::ParticleReal rounded_u[3] = {};
            amrex::Real const denominator =
                weight[source] * PhysConst::m_e * PhysConst::c;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                rounded_u[dimension] = static_cast<amrex::ParticleReal>(
                    events == 0 ? original.m_rdata[PIdx::ux+dimension][particle]
                                : current[dimension + 1] / denominator);
                next_u[3 * source + dimension] = rounded_u[dimension];
            }
            auto const represented_outgoing = PacketConversionFourMomentumFromPhoton(
                weight[source], rounded_u[0], rounded_u[1], rounded_u[2]);
            if (!represented_outgoing.valid)
            {
                amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                return;
            }
            FourVector const actual_outgoing = events == 0 ? current : represented_outgoing.value;
            amrex::GpuArray<amrex::Real, 3> photon_impulse{};
            amrex::Real photon_debit = incoming[4 * source] - actual_outgoing[0];
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                photon_impulse[dimension] =
                    (incoming[4 * source + dimension + 1] -
                     actual_outgoing[dimension + 1]) / PhysConst::c;
                if (!amrex::Math::isfinite(photon_impulse[dimension]))
                {
                    amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                    return;
                }
            }
            if (!amrex::Math::isfinite(photon_debit))
            {
                amrex::HostDevice::Atomic::Add(invalid_ptr, 1);
                return;
            }

            for (int component = 0; component < 4; ++component)
            {
                amrex::HostDevice::Atomic::Add(
                    &sum[component], incoming[4 * source + component]);
                amrex::HostDevice::Atomic::Add(
                    &sum[4 + component], actual_outgoing[component]);
            }
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                amrex::HostDevice::Atomic::Add(
                    &impulse(i, j, k, dimension), photon_impulse[dimension]);
                amrex::HostDevice::Atomic::Add(&sum[8 + dimension], photon_impulse[dimension]);
                amrex::Real const residual = std::abs(
                    incoming[4 * source + dimension + 1] -
                    actual_outgoing[dimension + 1] -
                    PhysConst::c * photon_impulse[dimension]);
                amrex::HostDevice::Atomic::Add(&sum[14 + dimension], residual);
                amrex::HostDevice::Atomic::Add(
                    &sum[17 + dimension],
                    std::abs(incoming[4 * source + dimension + 1]) +
                        std::abs(actual_outgoing[dimension + 1]));
            }
            amrex::HostDevice::Atomic::Add(&debit(i, j, k), photon_debit);
            amrex::HostDevice::Atomic::Add(&sum[11], photon_debit);
            if (events != 0)
            {
                amrex::HostDevice::Atomic::Add(
                    &scale(i, j, k), incoming[4 * source] + actual_outgoing[0]);
            }
            amrex::HostDevice::Atomic::Add(
                &sum[12], std::abs(
                              incoming[4 * source] - actual_outgoing[0] - photon_debit));
            amrex::HostDevice::Atomic::Add(
                &sum[13], std::abs(incoming[4 * source]) +
                              std::abs(actual_outgoing[0]));
        });
    }
    amrex::Gpu::streamSynchronize();

    int invalid_replay_collision = invalid_collision.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(invalid_replay_collision);
    if (invalid_replay_collision != 0)
    {
        SetRejection(report, 9);
        return false;
    }

    std::vector<amrex::Real> host_sums(sums.size());
    if (!host_sums.empty())
    {
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, sums.begin(), sums.end(), host_sums.begin());
    }
    amrex::ParallelDescriptor::ReduceRealSum(host_sums.data(), 20);
    amrex::Real const roundoff = 1024 * std::numeric_limits<amrex::Real>::epsilon();
    bool valid_identities = true;
    for (auto value : host_sums)
    {
        valid_identities = valid_identities && amrex::Math::isfinite(value);
    }
    if (host_sums[12] > roundoff * host_sums[13])
    {
        valid_identities = false;
    }
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        if (host_sums[14 + dimension] > roundoff * host_sums[17 + dimension])
        {
            valid_identities = false;
        }
    }
    // Check the actual deposited fields independently of packet-local identities.
    auto const deposited_energy = candidate_debit.sum(0);
    valid_identities = valid_identities && std::isfinite(deposited_energy) &&
        std::abs(deposited_energy-host_sums[11]) <= roundoff*host_sums[13];
    for (int d = 0; d < 3; ++d)
    {
        auto const deposited_impulse = candidate_impulse.sum(d);
        valid_identities = valid_identities && std::isfinite(deposited_impulse) &&
            std::abs(PhysConst::c*(deposited_impulse-host_sums[8+d])) <=
                roundoff*host_sums[17+d];
    }
    int identity_failure = valid_identities ? 0 : 10;
    int const global_identity_failure = GlobalFailureCode(identity_failure);
    if (global_identity_failure != 0)
    {
        SetRejection(report, global_identity_failure);
        return false;
    }

    // Global validation is complete; only now may a supplied candidate be mutated.
    for (auto const& tile : m_impl->baseline)
    {
        auto& found = private_photons.GetParticles(0).at(tile.key);
        AMREX_ALWAYS_ASSERT(found.numParticles() == tile.count);
        long const count = tile.count;
        if (count == 0) { continue; }
        long const offset = tile_offsets.at(tile.key);
        auto data = found.getParticleTileData();
        amrex::Real const* const next_u = candidate_u.dataPtr();
        amrex::For(count, [=] AMREX_GPU_DEVICE (long particle)
        {
            long const source = offset + particle;
            data.m_rdata[PIdx::ux][particle] =
                static_cast<amrex::ParticleReal>(next_u[3 * source]);
            data.m_rdata[PIdx::uy][particle] =
                static_cast<amrex::ParticleReal>(next_u[3 * source + 1]);
            data.m_rdata[PIdx::uz][particle] =
                static_cast<amrex::ParticleReal>(next_u[3 * source + 2]);
        });
    }
    amrex::Gpu::streamSynchronize();

    amrex::MultiFab::Copy(
        material_impulse, candidate_impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(
        photon_energy_debit, candidate_debit, 0, 0, 1, 0);
    amrex::MultiFab::Copy(
        uncancelled_energy_scale, candidate_scale, 0, 0, 1, 0);

    report.valid = true;
    report.colliding_photons = m_impl->colliding_photons;
    report.collision_events = m_impl->collision_events;
    for (int component = 0; component < 4; ++component)
    {
        report.incoming_inventory[component] = host_sums[component];
        report.outgoing_inventory[component] = host_sums[4 + component];
    }
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        report.material_impulse[dimension] = host_sums[8 + dimension];
    }
    report.photon_energy_debit = host_sums[11];
    return true;
}

} // namespace warpx::radiation
