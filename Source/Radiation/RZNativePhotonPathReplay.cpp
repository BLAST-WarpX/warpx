/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RZNativePhotonPathReplay.H"

#include "FiniteMassPhotonCollision.H"
#include "NativePhotonCarryRouting.H"
#include "NativePhotonReplayInternals.H"
#include "NativeRZThermalHistoryAdmission.H"
#include "PhotonFourMomentum.H"
#include "PhotonLogicalDrawKey.H"
#include "PhotonPacketTransaction.H"
#include "RZFaceMarching.H"
#include "RZMomentGeometry.H"
#include "RZSpecularTrajectory.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Box.H>
#include <AMReX_BoxIterator.H>
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
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace warpx::radiation
{
using namespace amrex::literals;
using photon_replay_detail::CaptureParticles;
using photon_replay_detail::GlobalFailureCode;
using photon_replay_detail::OpenUnitInterval;
using photon_replay_detail::PhotonDrawKey;
using photon_replay_detail::SameBits;

namespace
{
using RoutingIdentity = std::pair<std::uint64_t, std::uint64_t>;

[[maybe_unused]] RoutingIdentity
CarryIdentity (NativePhotonPathCarry const& carry)
{
    return {carry.provisional_draw_key == 0 ? 0 : 1,
            carry.provisional_draw_key == 0 ? carry.idcpu : carry.provisional_draw_key};
}

bool
SameCapturedTiles (std::vector<photon_replay_detail::CapturedTile> const& a,
                   std::vector<photon_replay_detail::CapturedTile> const& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t n = 0; n < a.size(); ++n) {
        if (a[n].key != b[n].key || a[n].count != b[n].count || !SameBits(a[n].idcpu, b[n].idcpu) ||
            !SameBits(a[n].real_data, b[n].real_data) || !SameBits(a[n].int_data, b[n].int_data)) {
            return false;
        }
    }
    return true;
}

void
SetPathRejection (RZNativePhotonPathReport& report, int code)
{
    report = RZNativePhotonPathReport{};
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
    case 11: report.reason = "replay face march failed"; break;
    case 12: report.reason = "replay output aliases an input"; break;
    default: report.reason = "unknown RZ native photon path replay rejection"; break;
    }
}

#if defined(WARPX_DIM_RZ)
// NVCC requires an extended-device lambda's enclosing function to be accessible.
// Keep the authenticated replay implementation private; move only its final
// independent per-particle copy kernel to this translation-unit-local helper.
template <typename TileData>
void
WriteReplayTile (long count, long offset, TileData data,
                 amrex::ParticleReal const* position, amrex::ParticleReal const* momentum,
                 int const* event_count, bool census_available,
                 amrex::GpuArray<amrex::ParticleReal*, 3> census_data)
{
    amrex::For(count, [=] AMREX_GPU_DEVICE (long particle)
    {
        long const source = offset + particle;
        data.m_rdata[PIdx::r][particle] = position[3 * source];
        data.m_rdata[PIdx::theta][particle] = position[3 * source + 1];
        data.m_rdata[PIdx::z][particle] = position[3 * source + 2];
        data.m_rdata[PIdx::ux][particle] = momentum[3 * source];
        data.m_rdata[PIdx::uy][particle] = momentum[3 * source + 1];
        data.m_rdata[PIdx::uz][particle] = momentum[3 * source + 2];
        if (census_available && event_count[source] == 0)
        {
            census_data[0][particle] = momentum[3 * source];
            census_data[1][particle] = momentum[3 * source + 1];
            census_data[2][particle] = momentum[3 * source + 2];
        }
    });
}

constexpr amrex::Real two_pi =
    6.283185307179586476925286766559005768394_rt;

bool
ValidPathPhoton (amrex::ParticleReal weight, amrex::ParticleReal ux,
                 amrex::ParticleReal uy, amrex::ParticleReal uz,
                 PacketConversionFourMomentum const& account)
{
    if (NeedsScaledNativePhotonMomentum(weight))
    {
        auto const scaled = ScaledNativePhotonMomentumFromPhoton(weight, ux, uy, uz);
        return scaled.valid && ValidFiniteMassPhoton(scaled.physical) &&
            ConsistentScaledNativePhotonAccounts(scaled);
    }
    return account.valid;
}

FourVector
ScaledPathAccounts (amrex::ParticleReal weight, amrex::Real u_scale,
                    FourVector const& physical)
{
    // Measure the prospective actually represented native momentum with the
    // same factory as Prepare/final publication. Reconstructing its account
    // by multiplying u_scale*(u/u_scale) can leave a formation ULP unowned.
    amrex::ParticleReal const ux = static_cast<amrex::ParticleReal>(physical[1] * u_scale);
    amrex::ParticleReal const uy = static_cast<amrex::ParticleReal>(physical[2] * u_scale);
    amrex::ParticleReal const uz = static_cast<amrex::ParticleReal>(physical[3] * u_scale);
    return PacketConversionFourMomentumFromPhoton(weight, ux, uy, uz).value;
}

struct HostField
{
    amrex::Box box{};
    int components = 0;
    std::vector<amrex::Real> values;

    [[nodiscard]] bool contains (amrex::IntVect const& cell) const
    {
        return box.contains(cell);
    }

    [[nodiscard]] amrex::Real operator() (
        amrex::IntVect const& cell, int component) const
    {
        auto const offset = static_cast<std::size_t>(
            box.index(cell) +
            box.numPts() * static_cast<long long>(component));
        return values[offset];
    }

    void add (amrex::IntVect const& cell, int component, amrex::Real value)
    {
        auto const offset = static_cast<std::size_t>(
            box.index(cell) +
            box.numPts() * static_cast<long long>(component));
        values[offset] += value;
    }
};

std::unordered_map<int, HostField>
CaptureHostFields (amrex::MultiFab const& fields, int number_components)
{
    std::unordered_map<int, HostField> result;
    for (amrex::MFIter iterator(fields); iterator.isValid(); ++iterator)
    {
        amrex::Box const box = iterator.fabbox();
        HostField host;
        host.box = box;
        host.components = number_components;
        host.values.resize(
            static_cast<std::size_t>(box.numPts()) *
            static_cast<std::size_t>(number_components), 0);
        auto const& fab = fields[iterator];
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, fab.dataPtr(),
            fab.dataPtr() + box.numPts() * number_components,
            host.values.begin());
        result.emplace(iterator.index(), std::move(host));
    }
    return result;
}

void
RestoreHostFields (
    amrex::MultiFab& fields,
    std::unordered_map<int, HostField> const& hosts)
{
    for (amrex::MFIter iterator(fields); iterator.isValid(); ++iterator)
    {
        auto found = hosts.find(iterator.index());
        if (found == hosts.end()) { continue; }
        HostField const& host = found->second;
        auto& fab = fields[iterator];
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, host.values.begin(), host.values.end(),
            fab.dataPtr());
    }
}

std::pair<std::unordered_map<int, HostField>,
          std::unordered_map<int, HostField>>
CaptureFrozenGrids (
    amrex::MultiFab const& opacity, amrex::MultiFab const& rate_velocity,
    amrex::Geometry const& geometry)
{
    amrex::MultiFab compact_opacity(
        opacity.boxArray(), opacity.DistributionMap(), 1, 1);
    amrex::MultiFab compact_velocity(
        rate_velocity.boxArray(), rate_velocity.DistributionMap(), 3, 1);
    amrex::MultiFab::Copy(compact_opacity, opacity, 0, 0, 1, 0);
    amrex::MultiFab::Copy(compact_velocity, rate_velocity, 0, 0, 3, 0);
    compact_opacity.FillBoundary(geometry.periodicity());
    compact_velocity.FillBoundary(geometry.periodicity());
    return {CaptureHostFields(compact_opacity, 1),
            CaptureHostFields(compact_velocity, 3)};
}

[[nodiscard]] amrex::Real
SampleOpticalDepth (
    NativePhotonCollisionOptions const& options, std::uint64_t idcpu,
    int event)
{
    return -std::log(OpenUnitInterval(PhotonDrawKey(
        options.seed, idcpu, options.interval,
        static_cast<std::uint64_t>(event), 0)));
}

[[nodiscard]] amrex::GpuArray<amrex::Real, 3>
SampleEventDirection (
    NativePhotonCollisionOptions const& options, std::uint64_t idcpu,
    int event)
{
    if (options.angular_law ==
        NativePhotonCollisionAngularLaw::FixedFrameDirection)
    {
        return options.fixed_frame_direction;
    }
    amrex::Real const cosine = 2 * OpenUnitInterval(PhotonDrawKey(
        options.seed, idcpu, options.interval,
        static_cast<std::uint64_t>(event), 1)) - 1;
    amrex::Real const azimuth = two_pi * OpenUnitInterval(PhotonDrawKey(
        options.seed, idcpu, options.interval,
        static_cast<std::uint64_t>(event), 2));
    amrex::Real const sine = std::sqrt(1 - cosine * cosine);
    return {sine * std::cos(azimuth), sine * std::sin(azimuth), cosine};
}

[[nodiscard]] amrex::Real
WrapPeriodicZ (amrex::Geometry const& geometry, amrex::Real coordinate)
{
    amrex::Real const lower = geometry.ProbLo(1);
    amrex::Real const upper = geometry.ProbHi(1);
    amrex::Real const length = upper - lower;
    long double const exact = static_cast<long double>(lower) +
        std::fmod(
            static_cast<long double>(coordinate) -
                static_cast<long double>(lower),
            static_cast<long double>(length));
    long double const canonical =
        exact < static_cast<long double>(lower) ?
            exact + static_cast<long double>(length) : exact;
    amrex::Real wrapped = static_cast<amrex::Real>(canonical);
    if (wrapped >= upper) { wrapped = std::nextafter(upper, lower); }
    return wrapped;
}

[[nodiscard]] amrex::Real
CanonicalTheta (amrex::Real theta)
{
    long double canonical = std::remainder(
        static_cast<long double>(theta), static_cast<long double>(two_pi));
    if (canonical < 0.0L)
    {
        canonical += static_cast<long double>(two_pi);
    }
    return static_cast<amrex::Real>(canonical);
}

/** Optical depth along one Cartesian straight path inside one RZ cell.
 * The frozen cell velocity is (vr, vtheta-conjugate, vz). Its physical
 * tangential speed is vtheta*r/rbar, while h=x*ny-y*nx stays constant.
 * Integrating the radial beta projection by endpoints avoids the cusp when
 * a ray passes through the axis. The remaining gamma(r(s)) is smooth.
 */
[[nodiscard]] amrex::Real
IntegratedRZOpticalDepth (
    amrex::Real chi, amrex::GpuArray<amrex::Real, 3> const& velocity,
    amrex::Real rbar, amrex::Real x, amrex::Real y,
    amrex::GpuArray<amrex::Real, 3> const& direction, amrex::Real distance,
    bool moving)
{
    if (!std::isfinite(chi) || chi < 0 || !std::isfinite(distance) || distance < 0 ||
        !(rbar > 0)) { return std::numeric_limits<amrex::Real>::quiet_NaN(); }
    if (chi == 0 || distance == 0) { return 0; }
    if (!moving) { return chi * distance; }
    amrex::Real const br = velocity[0] / PhysConst::c;
    amrex::Real const bt = velocity[1] / PhysConst::c;
    amrex::Real const bz = velocity[2] / PhysConst::c;
    amrex::Real const a = 1 - br * br - bz * bz;
    amrex::Real const b = (bt / rbar) * (bt / rbar);
    amrex::Real const h = x * direction[1] - y * direction[0];
    amrex::Real const factor = 1 - bz * direction[2] - bt * h / rbar;
    amrex::Real const initial_radius = std::hypot(x, y);
    amrex::Real const final_radius = std::hypot(
        x + distance * direction[0], y + distance * direction[1]);
    if (!std::isfinite(a) || !(a > 0) || !std::isfinite(b) ||
        !std::isfinite(factor) || !(factor > 0))
    {
        return std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    if (b == 0)
    {
        return chi / std::sqrt(a) *
            (factor * distance - br * (final_radius - initial_radius));
    }
    // Eight-point Gauss-Legendre integration is checked against two half
    // intervals. The admitted beta<=1e-3 makes gamma analytic and nearly
    // constant throughout the bounded half-cell path.
    constexpr std::array<amrex::Real, 4> nodes{
        0.18343464249564980494_rt, 0.52553240991632898582_rt,
        0.79666647741362673959_rt, 0.96028985649753623168_rt};
    constexpr std::array<amrex::Real, 4> weights{
        0.36268378337836198297_rt, 0.31370664587788728734_rt,
        0.22238103445337447054_rt, 0.10122853629037625915_rt};
    auto const integrate_gamma = [&] (amrex::Real lower, amrex::Real upper,
                                     bool along_ray) -> amrex::Real
    {
        amrex::Real const midpoint = 0.5_rt * (lower + upper);
        amrex::Real const half = 0.5_rt * (upper - lower);
        amrex::Real sum = 0;
        for (int n = 0; n < 4; ++n)
        {
            for (int side : {-1, 1})
            {
                amrex::Real const coordinate = midpoint + side * half * nodes[n];
                amrex::Real const radius = along_ray ? std::hypot(
                    x + coordinate * direction[0],
                    y + coordinate * direction[1]) : coordinate;
                amrex::Real const radicand = a - b * radius * radius;
                if (!(radicand > 0) || !std::isfinite(radicand))
                {
                    return std::numeric_limits<amrex::Real>::quiet_NaN();
                }
                sum += weights[n] / std::sqrt(radicand);
            }
        }
        return half * sum;
    };
    amrex::Real const whole = integrate_gamma(0, distance, true);
    amrex::Real const half = 0.5_rt * distance;
    amrex::Real const refined = integrate_gamma(0, half, true) +
        integrate_gamma(half, distance, true);
    amrex::Real const radial = integrate_gamma(
        initial_radius, final_radius, false);
    amrex::Real const tolerance = 512 *
        std::numeric_limits<amrex::Real>::epsilon() * std::abs(refined);
    if (!std::isfinite(whole) || !std::isfinite(refined) ||
        !std::isfinite(radial) || std::abs(whole - refined) > tolerance)
    {
        return std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    amrex::Real const depth = chi * (factor * refined - br * radial);
    return std::isfinite(depth) && depth >= 0 ? depth :
        std::numeric_limits<amrex::Real>::quiet_NaN();
}

[[nodiscard]] bool
SameOptionsAcrossRanks (
    NativePhotonCollisionOptions const& options, amrex::Real dt)
{
    amrex::Long integers[7] = {
        static_cast<amrex::Long>(options.seed),
        static_cast<amrex::Long>(options.interval),
        static_cast<amrex::Long>(options.max_events_per_photon),
        static_cast<amrex::Long>(options.angular_law ==
                                 NativePhotonCollisionAngularLaw::Isotropic),
        static_cast<amrex::Long>(options.absorption),
        static_cast<amrex::Long>(options.emission),
        static_cast<amrex::Long>(options.moving_cross_sections)};
    amrex::Long minimum_integers[7];
    amrex::Long maximum_integers[7];
    std::copy(std::begin(integers), std::end(integers), minimum_integers);
    std::copy(std::begin(integers), std::end(integers), maximum_integers);
    amrex::ParallelDescriptor::ReduceLongMin(minimum_integers, 7);
    amrex::ParallelDescriptor::ReduceLongMax(maximum_integers, 7);
    if (!std::equal(std::begin(minimum_integers), std::end(minimum_integers),
                    std::begin(maximum_integers)))
    {
        return false;
    }
    amrex::Real reals[5] = {
        dt, options.max_optical_depth, options.fixed_frame_direction[0],
        options.fixed_frame_direction[1], options.fixed_frame_direction[2]};
    amrex::Real minimum_reals[5];
    amrex::Real maximum_reals[5];
    std::copy(std::begin(reals), std::end(reals), minimum_reals);
    std::copy(std::begin(reals), std::end(reals), maximum_reals);
    amrex::ParallelDescriptor::ReduceRealMin(minimum_reals, 5);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_reals, 5);
    return std::equal(std::begin(minimum_reals), std::end(minimum_reals),
                      std::begin(maximum_reals));
}

[[nodiscard]] bool
SameGeometryAcrossRanks (amrex::Geometry const& geometry)
{
    amrex::Long integers[2 + 2 * AMREX_SPACEDIM] = {geometry.Coord(), 0};
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        integers[2 + 2 * dimension] =
            geometry.Domain().smallEnd(dimension);
        integers[3 + 2 * dimension] =
            geometry.Domain().bigEnd(dimension);
        integers[1] = 2 * integers[1] +
                      (geometry.isPeriodic(dimension) ? 1 : 0);
    }
    amrex::Long minimum_integers[2 + 2 * AMREX_SPACEDIM];
    amrex::Long maximum_integers[2 + 2 * AMREX_SPACEDIM];
    std::copy(std::begin(integers), std::end(integers), minimum_integers);
    std::copy(std::begin(integers), std::end(integers), maximum_integers);
    amrex::ParallelDescriptor::ReduceLongMin(
        minimum_integers, 2 + 2 * AMREX_SPACEDIM);
    amrex::ParallelDescriptor::ReduceLongMax(
        maximum_integers, 2 + 2 * AMREX_SPACEDIM);
    if (!std::equal(std::begin(minimum_integers), std::end(minimum_integers),
                    std::begin(maximum_integers)))
    {
        return false;
    }
    constexpr int geometry_values=3*AMREX_SPACEDIM;
    amrex::Real cell_sizes[geometry_values];
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        cell_sizes[3*dimension] = geometry.CellSize(dimension);
        cell_sizes[3*dimension+1] = geometry.ProbLo(dimension);
        cell_sizes[3*dimension+2] = geometry.ProbHi(dimension);
    }
    amrex::Real minimum_sizes[geometry_values];
    amrex::Real maximum_sizes[geometry_values];
    std::copy(std::begin(cell_sizes), std::end(cell_sizes), minimum_sizes);
    std::copy(std::begin(cell_sizes), std::end(cell_sizes), maximum_sizes);
    amrex::ParallelDescriptor::ReduceRealMin(
        minimum_sizes, geometry_values);
    amrex::ParallelDescriptor::ReduceRealMax(
        maximum_sizes, geometry_values);
    return std::equal(std::begin(minimum_sizes), std::end(minimum_sizes),
                      std::begin(maximum_sizes));
}

struct PointerRange
{
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

[[nodiscard]] std::vector<PointerRange>
LocalFieldRanges (amrex::MultiFab const& field, int number_components)
{
    std::vector<PointerRange> ranges;
    for (amrex::MFIter iterator(field); iterator.isValid(); ++iterator)
    {
        auto const& fab = field[iterator];
        long long const points = fab.box().numPts();
        std::size_t const bytes = static_cast<std::size_t>(points) *
                                  sizeof(amrex::Real);
        for (int component = 0; component < number_components; ++component)
        {
            auto const begin = reinterpret_cast<std::uintptr_t>(
                fab.dataPtr(component));
            if (begin==0 || bytes>std::numeric_limits<std::uintptr_t>::max()-begin) {
                return {{0,std::numeric_limits<std::uintptr_t>::max()}};
            }
            ranges.push_back({begin, begin + bytes});
        }
    }
    return ranges;
}

[[nodiscard]] bool
RangesOverlap (PointerRange const& first, PointerRange const& second)
{
    return first.begin < second.end && second.begin < first.end;
}

[[nodiscard]] bool
LocalSuppliedFieldsAlias (
    amrex::MultiFab const& work_velocity, amrex::MultiFab& material_impulse,
    amrex::MultiFab& photon_energy_debit,
    amrex::MultiFab& uncancelled_energy_scale)
{
    std::array<std::vector<PointerRange>, 4> ranges = {
        LocalFieldRanges(work_velocity, 3),
        LocalFieldRanges(material_impulse, 3),
        LocalFieldRanges(photon_energy_debit, 1),
        LocalFieldRanges(uncancelled_energy_scale, 1)};
    for (int first = 0; first < 4; ++first)
    {
        for (int second = first + 1; second < 4; ++second)
        {
            for (auto const& a : ranges[static_cast<std::size_t>(first)])
            {
                for (auto const& b : ranges[static_cast<std::size_t>(second)])
                {
                    if (RangesOverlap(a, b)) { return true; }
                }
            }
        }
    }
    return false;
}

[[nodiscard]] amrex::Real
AngularPhotonMomentum (
    amrex::Real x, amrex::Real y, FourVector const& momentum)
{
    return x * momentum[2] - y * momentum[1];
}

[[nodiscard]] amrex::GpuArray<amrex::Real, 4>
RadialReflectionMatrix (amrex::Real normal_x, amrex::Real normal_y)
{
    amrex::GpuArray<amrex::Real, 4> reflection{};
    reflection[0] = 1 - 2 * normal_x * normal_x;
    reflection[1] = -2 * normal_x * normal_y;
    reflection[2] = reflection[1];
    reflection[3] = 1 - 2 * normal_y * normal_y;
    return reflection;
}

void
ReflectMomentum (FourVector& momentum, amrex::GpuArray<amrex::Real, 4> const& m)
{
    amrex::Real const x = momentum[1];
    amrex::Real const y = momentum[2];
    momentum[1] = m[0] * x + m[1] * y;
    momentum[2] = m[2] * x + m[3] * y;
}

[[nodiscard]] amrex::Real
SumValidHost (
    amrex::MultiFab const& field,
    std::unordered_map<int, HostField> const& hosts, int component)
{
    amrex::Real local = 0;
    for (amrex::MFIter iterator(field); iterator.isValid(); ++iterator)
    {
        auto found = hosts.find(iterator.index());
        if (found == hosts.end()) { continue; }
        for (amrex::BoxIterator cell(iterator.validbox()); cell.ok(); ++cell)
        {
            local += found->second(*cell, component);
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(local);
    return local;
}

#if defined(WARPX_DIM_RZ)
[[nodiscard]] amrex::Real
SumAngularValidHost (
    amrex::MultiFab const& field,
    std::unordered_map<int, HostField> const& hosts,
    amrex::Geometry const& geometry)
{
    RZMomentMetric const metric(geometry);
    amrex::Real local = 0;
    for (amrex::MFIter iterator(field); iterator.isValid(); ++iterator)
    {
        auto found = hosts.find(iterator.index());
        if (found == hosts.end()) { continue; }
        for (amrex::BoxIterator cell(iterator.validbox()); cell.ok(); ++cell)
        {
            local += metric.Angular((*cell)[0]).mean_radius *
                     found->second(*cell, 1);
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(local);
    return local;
}
#endif
#endif // WARPX_DIM_RZ: private trajectory implementation
}

struct RZNativePhotonPathReplay::Impl
{
    WarpXParticleContainer const* photon_species = nullptr;
    amrex::Geometry geometry{};
    NativePhotonCollisionOptions options{};
    amrex::Real dt = 0;
    amrex::BoxArray box_array;
    amrex::DistributionMapping distribution_map;
    std::vector<photon_replay_detail::CapturedTile> baseline;
    int number_real_components = 0;
    int number_int_components = 0;
    std::vector<std::string> real_names;
    std::vector<std::string> int_names;

    std::vector<std::uint64_t> idcpu;
    std::vector<std::uint64_t> draw_key;
    std::vector<amrex::ParticleReal> weight;
    std::vector<amrex::Real> incoming;
    std::vector<amrex::ParticleReal> initial_u;
    std::vector<amrex::Real> initial_position; // r,theta,z
    std::vector<int> initial_cell;             // i,j
    std::vector<NativePhotonPathCarry> initial_carries;
    amrex::MultiFab frozen_opacity;
    amrex::MultiFab frozen_rate_velocity;
    bool census_available = false;
    std::array<int, 3> census_components{};
    NativePhotonPathSnapshotPolicy snapshot_policy = NativePhotonPathSnapshotPolicy::NativeOnly;
    bool authenticated_empty_duration = false;
    NativeRZThermalHistoryStepContext const* duration_context = nullptr;
    NativeRZThermalAllDNoKEventWitness const* duration_witness = nullptr;
    bool prepared = false;
};

RZNativePhotonPathReplay::RZNativePhotonPathReplay ()
    : m_impl(std::make_unique<Impl>()) {}

RZNativePhotonPathReplay::~RZNativePhotonPathReplay () = default;

bool
RZNativePhotonPathReplay::Prepare (
    WarpXParticleContainer& photons, amrex::MultiFab const& scattering_opacity,
    amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
    amrex::Real dt, NativePhotonCollisionOptions const& options,
    RZNativePhotonPathReport& report)
{
    return Prepare(photons, photons, scattering_opacity,
        initial_rate_velocity, geometry, dt, options, report);
}

bool
RZNativePhotonPathReplay::Prepare (WarpXParticleContainer::Base const& private_snapshot,
                                   WarpXParticleContainer const& photon_species,
                                   amrex::MultiFab const& scattering_opacity,
                                   amrex::MultiFab const& initial_rate_velocity,
                                   amrex::Geometry const& geometry, amrex::Real dt,
                                   NativePhotonCollisionOptions const& options,
                                   std::vector<NativePhotonPathCarry> const& initial_carries,
                                   RZNativePhotonPathReport& report,
                                   NativePhotonPathSnapshotPolicy policy)
{
    return PrepareImpl(private_snapshot, photon_species, scattering_opacity, initial_rate_velocity,
                       geometry, dt, options, initial_carries, true, report, policy, false);
}

bool
RZNativePhotonPathReplay::Prepare (WarpXParticleContainer::Base const& private_snapshot,
                                   WarpXParticleContainer const& photon_species,
                                   amrex::MultiFab const& scattering_opacity,
                                   amrex::MultiFab const& initial_rate_velocity,
                                   amrex::Geometry const& geometry, amrex::Real dt,
                                   NativePhotonCollisionOptions const& options,
                                   RZNativePhotonPathReport& report,
                                   NativePhotonPathSnapshotPolicy policy)
{
    return PrepareImpl(private_snapshot, photon_species, scattering_opacity, initial_rate_velocity,
                       geometry, dt, options, {}, false, report, policy, false);
}

bool
RZNativePhotonPathReplay::PrepareAllDNoPhoton (
    PhotonPacketTransaction const& private_snapshot,
    WarpXParticleContainer const& photon_species,
    amrex::MultiFab const& scattering_opacity,
    amrex::MultiFab const& initial_rate_velocity, amrex::Geometry const& geometry,
    amrex::Real dt, NativePhotonCollisionOptions const& options,
    std::vector<NativePhotonPathCarry> const& initial_carries,
    NativeRZThermalHistoryStepContext const& context,
    NativeRZThermalAllDNoKEventWitness const& witness,
    RZNativePhotonPathReport& report)
{
    m_impl = std::make_unique<Impl>();
    report = RZNativePhotonPathReport{};
    std::string reason;
    bool const authority = witness.ValidatePhotonDuration(context, dt, options.seed,
        options.interval, photon_species, private_snapshot, witness.CanonicalMask(),
        geometry, reason);
    bool const clocks = GlobalFailureCode(initial_carries.empty() ? 0 : 1) == 0;
    if (!authority || !clocks)
    {
        SetPathRejection(report, 1);
        report.reason = authority ? "all-D implicit path has nonempty actual clocks" : reason;
        return false;
    }
    if (!PrepareImpl(private_snapshot.Particles(), photon_species, scattering_opacity,
            initial_rate_velocity, geometry, dt, options, initial_carries, true, report,
            NativePhotonPathSnapshotPolicy::KeyedProvisional, true))
    { return false; }
    // The genuine preparation graph ran. Revalidate before retaining borrowed
    // authority; no shortened duration, synthetic plan or event draw is issued.
    if (!witness.ValidatePhotonDuration(context, dt, options.seed, options.interval,
            photon_species, private_snapshot, witness.CanonicalMask(), geometry, reason))
    {
        m_impl = std::make_unique<Impl>();
        SetPathRejection(report, 1);
        report.reason = reason;
        return false;
    }
    m_impl->duration_context = &context;
    m_impl->duration_witness = &witness;
    return true;
}

bool
RZNativePhotonPathReplay::PrepareImpl (WarpXParticleContainer::Base const& private_snapshot,
                                       WarpXParticleContainer const& photon_species,
                                       amrex::MultiFab const& scattering_opacity,
                                       amrex::MultiFab const& initial_rate_velocity,
                                       amrex::Geometry const& geometry, amrex::Real dt,
                                       NativePhotonCollisionOptions const& options,
                                       std::vector<NativePhotonPathCarry> const& initial_carries,
                                       bool resume_carries, RZNativePhotonPathReport& report,
                                       NativePhotonPathSnapshotPolicy policy,
                                       bool authenticated_empty_duration)
{
    m_impl = std::make_unique<Impl>();
    report = RZNativePhotonPathReport{};

    int invalid_inputs = 0;
    int minimum_policy = static_cast<int>(policy), maximum_policy = minimum_policy;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_policy);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_policy);
    if (minimum_policy != maximum_policy ||
        (policy != NativePhotonPathSnapshotPolicy::NativeOnly &&
         policy != NativePhotonPathSnapshotPolicy::KeyedProvisional)) {
        invalid_inputs = 1;
    }
#if !defined(WARPX_DIM_RZ)
    invalid_inputs = 1;
#endif
    if constexpr (std::numeric_limits<amrex::Real>::digits < 53 ||
                  std::numeric_limits<amrex::ParticleReal>::digits < 53)
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
    if (options.absorption || options.emission ||
        options.max_events_per_photon <= 0 ||
        (options.angular_law !=
             NativePhotonCollisionAngularLaw::Isotropic &&
         options.angular_law !=
             NativePhotonCollisionAngularLaw::FixedFrameDirection) ||
        !amrex::Math::isfinite(options.max_optical_depth) ||
        !(options.max_optical_depth > 0) ||
        !amrex::Math::isfinite(dt) || !(dt > 0) ||
        !amrex::Math::isfinite(direction_squared) ||
        !(direction_squared > 0) ||
        std::abs(direction_squared - 1) >
            256 * std::numeric_limits<amrex::Real>::epsilon() *
                direction_squared)
    {
        invalid_inputs = 1;
    }
#if defined(WARPX_DIM_RZ)
    bool const same_inputs = SameOptionsAcrossRanks(options, dt) &&
        SameGeometryAcrossRanks(geometry);
#else
    bool const same_inputs = false;
#endif
    int const global_invalid_inputs = GlobalFailureCode(invalid_inputs);
    if (global_invalid_inputs != 0 || !same_inputs)
    {
        SetPathRejection(
            report, global_invalid_inputs == 0 ? 1 : global_invalid_inputs);
        return false;
    }

#if defined(WARPX_DIM_RZ)
    auto const& photons = private_snapshot;
    auto const& photon_boxes = photons.ParticleBoxArray(0);
    auto const& photon_distribution = photons.ParticleDistributionMap(0);
    int invalid_layout = 0;
    if (photons.finestLevel() != 0 || (geometry.Coord() != 0 && geometry.Coord() != 1) ||
        !geometry.Domain().ok() || !photon_boxes.ok() || !photon_boxes.isDisjoint() ||
        !photon_boxes.ixType().cellCentered() ||
        !scattering_opacity.ixType().cellCentered() ||
        !initial_rate_velocity.ixType().cellCentered() ||
        geometry.isPeriodic(0) ||
        photon_species.finestLevel() != 0 ||
        !photon_species.AmIA<PhysicalSpecies::photon>() ||
        photon_species.ParticleBoxArray(0) != photon_boxes ||
        photon_species.ParticleDistributionMap(0) != photon_distribution ||
        scattering_opacity.nComp() != 1 ||
        initial_rate_velocity.nComp() != 3 ||
        scattering_opacity.boxArray() != photon_boxes ||
        scattering_opacity.DistributionMap() != photon_distribution ||
        initial_rate_velocity.boxArray() != photon_boxes ||
        initial_rate_velocity.DistributionMap() != photon_distribution)
    {
        invalid_layout = 1;
    }

    amrex::Box const domain = geometry.Domain();
    long covered_points = 0;
    for (int box_index = 0; box_index < photon_boxes.size(); ++box_index)
    {
        amrex::Box intersection = photon_boxes[box_index] & domain;
        if (intersection != photon_boxes[box_index])
        {
            invalid_layout = 1;
        }
        covered_points += photon_boxes[box_index].numPts();
    }
    if (covered_points != domain.numPts())
    {
        invalid_layout = 1;
    }
    amrex::Real maximum_path_length =
        std::numeric_limits<amrex::Real>::infinity();
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        amrex::Real const cell_size = geometry.CellSize(dimension);
        if (!amrex::Math::isfinite(cell_size) || !(cell_size > 0))
        {
            invalid_layout = 1;
        }
        for (auto const* particle_geometry : {&photons.Geom(0), &photon_species.Geom(0)}) {
            if (particle_geometry->Domain()!=geometry.Domain() ||
                particle_geometry->Coord()!=geometry.Coord() ||
                particle_geometry->isPeriodic(dimension)!=geometry.isPeriodic(dimension) ||
                particle_geometry->ProbLo(dimension)!=geometry.ProbLo(dimension) ||
                particle_geometry->ProbHi(dimension)!=geometry.ProbHi(dimension)) {
                invalid_layout=1;
            }
        }
        maximum_path_length =
            std::min(maximum_path_length, 0.5_rt * cell_size);
    }
    amrex::Real const path_length = PhysConst::c * dt;
    if (!amrex::Math::isfinite(path_length) || !(path_length > 0) ||
        (!authenticated_empty_duration && path_length > maximum_path_length) ||
        !amrex::Math::isfinite(geometry.ProbLo(0)) ||
        !amrex::Math::isfinite(geometry.ProbHi(0)) ||
        !(geometry.ProbLo(0) >= 0) ||
        !(geometry.ProbHi(0) > geometry.ProbLo(0)) ||
        !amrex::Math::isfinite(geometry.ProbLo(1)) ||
        !amrex::Math::isfinite(geometry.ProbHi(1)) ||
        !(geometry.ProbHi(1) > geometry.ProbLo(1)))
    {
        invalid_layout = 1;
    }
    int const global_invalid_layout = GlobalFailureCode(invalid_layout);
    if (global_invalid_layout != 0)
    {
        SetPathRejection(report, global_invalid_layout);
        return false;
    }

    auto const captured = CaptureParticles(photons);
    std::vector<std::uint64_t> native_ids, provisional_keys, draw_keys;
    if (GlobalFailureCode(
            photon_replay_detail::CapturePhotonRoutingKeys(photons, photon_species, policy,
                                                           native_ids, provisional_keys, draw_keys)
                ? 0
                : 2) != 0) {
        SetPathRejection(report, 2);
        return false;
    }
    if (policy == NativePhotonPathSnapshotPolicy::KeyedProvisional) {
        std::vector<NativePhotonPathCarry> probes, routed;
        for (std::size_t n = 0; n < native_ids.size(); ++n) {
            probes.push_back({native_ids[n], 1, 0, 1, provisional_keys[n]});
        }
        if (!photon_replay_detail::RouteKeyedPhotonCarries(probes, native_ids, provisional_keys,
                                                           options.max_events_per_photon, routed)) {
            SetPathRejection(report, 2);
            return false;
        }
    }
    std::map<RoutingIdentity, NativePhotonPathCarry> carry_by_id;
    long captured_particles = 0;
    for (auto const& tile : captured.tiles) {
        captured_particles += tile.count;
    }
    int invalid_carries = (resume_carries && initial_carries.size() !=
                                                 static_cast<std::size_t>(captured_particles)) ||
                          (!resume_carries && !initial_carries.empty());
    for (auto const& carry : initial_carries) {
        if (!carry_by_id.emplace(CarryIdentity(carry), carry).second ||
            (policy == NativePhotonPathSnapshotPolicy::NativeOnly &&
             carry.provisional_draw_key != 0)) {
            invalid_carries = 1;
        }
    }
    if (GlobalFailureCode(invalid_carries) != 0) {
        SetPathRejection(report, 1);
        return false;
    }
    auto const [opacity_fields, rate_fields] = CaptureFrozenGrids(
        scattering_opacity, initial_rate_velocity, geometry);
    RZMomentMetric const metric(geometry);
    int local_failure = 0;
    for (amrex::MFIter iterator(scattering_opacity); iterator.isValid();
         ++iterator)
    {
        auto const& opacity_field = opacity_fields.at(iterator.index());
        auto const& rate_field = rate_fields.at(iterator.index());
        for (amrex::BoxIterator cell(iterator.validbox()); cell.ok(); ++cell)
        {
            amrex::Real const chi = opacity_field(*cell, 0);
            amrex::Real const rbar =
                metric.Angular((*cell)[0]).mean_radius;
            amrex::Real const outer_radius = metric.Radius((*cell)[0] + 1);
            amrex::Real beta_squared = 0;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                amrex::Real velocity = rate_field(*cell, dimension);
                if (dimension == 1)
                {
                    velocity *= outer_radius / rbar;
                }
                amrex::Real const beta = velocity / PhysConst::c;
                if (!amrex::Math::isfinite(beta)) { local_failure = 1; }
                beta_squared += beta * beta;
            }
            if (!amrex::Math::isfinite(chi) || chi < 0 ||
                !amrex::Math::isfinite(beta_squared) ||
                !(beta_squared < 1) ||
                (options.moving_cross_sections && beta_squared > 1.e-6_rt))
            {
                local_failure = 1;
            }
            amrex::Real const beta = std::sqrt(beta_squared);
            amrex::Real const hazard = chi * PhysConst::c * dt *
                (options.moving_cross_sections ?
                    (1 + beta) / std::sqrt(1 - beta_squared) : amrex::Real(1));
            if (!amrex::Math::isfinite(hazard) || hazard < 0 ||
                hazard > options.max_optical_depth)
            {
                local_failure = 3;
            }
        }
    }

    auto plan = std::make_unique<Impl>();
    plan->photon_species = &photon_species;
    plan->geometry = geometry;
    plan->options = options;
    plan->snapshot_policy = policy;
    plan->authenticated_empty_duration = authenticated_empty_duration;
    plan->dt = dt;
    plan->box_array = photon_boxes;
    plan->distribution_map = photon_distribution;
    plan->baseline = captured.tiles;
    plan->number_real_components = photons.NumRealComps();
    plan->number_int_components = photons.NumIntComps();
    plan->real_names = photons.GetRealSoANames();
    plan->int_names = photons.GetIntSoANames();
    plan->frozen_opacity.define(photon_boxes, photon_distribution, 1, 1);
    plan->frozen_rate_velocity.define(
        photon_boxes, photon_distribution, 3, 1);
    amrex::MultiFab::Copy(
        plan->frozen_opacity, scattering_opacity, 0, 0, 1, 0);
    amrex::MultiFab::Copy(
        plan->frozen_rate_velocity, initial_rate_velocity, 0, 0, 3, 0);
    plan->frozen_opacity.FillBoundary(geometry.periodicity());
    plan->frozen_rate_velocity.FillBoundary(geometry.periodicity());

    static std::array<std::string, 3> const census_names = {
        "radiation_diffusion_census_ux0",
        "radiation_diffusion_census_uy0",
        "radiation_diffusion_census_uz0"};
    auto const real_names = photons.GetRealSoANames();
    bool census_available = true;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        auto const found = std::find(
            real_names.begin(), real_names.end(), census_names[
                static_cast<std::size_t>(dimension)]);
        if (found == real_names.end())
        {
            census_available = false;
            break;
        }
        plan->census_components[static_cast<std::size_t>(dimension)] =
            static_cast<int>(std::distance(real_names.begin(), found));
    }
    plan->census_available = census_available;

    long global_particle = 0;
    amrex::Real const radial_lo = geometry.ProbLo(0);
    amrex::Real const axial_lo = geometry.ProbLo(1);
    amrex::Real const dr = geometry.CellSize(0);
    amrex::Real const dz = geometry.CellSize(1);
    for (auto const& tile : captured.tiles)
    {
        long const count = tile.count;
        auto const value = [&](int component, long particle) {
            return tile.real_data[
                static_cast<std::size_t>(component) *
                    static_cast<std::size_t>(count) +
                static_cast<std::size_t>(particle)];
        };
        for (long particle = 0; particle < count;
             ++particle, ++global_particle)
        {
            if (local_failure != 0) {
                continue;
            }
            std::uint64_t const idcpu = tile.idcpu[static_cast<std::size_t>(particle)];
            NativePhotonPathCarry carry{};
            carry.idcpu = idcpu;
            carry.provisional_draw_key =
                provisional_keys[static_cast<std::size_t>(global_particle)];
            if (resume_carries) {
                auto const found = carry_by_id.find(CarryIdentity(carry));
                if (found == carry_by_id.end() || found->second.idcpu != idcpu ||
                    !std::isfinite(found->second.residual_optical_depth) ||
                    !(found->second.residual_optical_depth > 0) ||
                    found->second.completed_events < 0 ||
                    found->second.completed_events > options.max_events_per_photon ||
                    found->second.next_draw != found->second.completed_events + 1) {
                    local_failure = 1;
                    continue;
                }
                carry = found->second;
            }
            if (!amrex::ConstParticleIDWrapper(idcpu).is_valid() ||
                photon_distribution[tile.key.first] !=
                    amrex::ParallelDescriptor::MyProc())
            {
                local_failure = 2;
                continue;
            }
            std::uint64_t const draw_key = draw_keys[static_cast<std::size_t>(global_particle)];
            amrex::ParticleReal const radius = value(PIdx::r, particle);
            amrex::ParticleReal const theta = value(PIdx::theta, particle);
            amrex::ParticleReal const z = value(PIdx::z, particle);
            if (!std::isfinite(radius) || radius<radial_lo || radius>=geometry.ProbHi(0) ||
                !std::isfinite(theta) || !std::isfinite(z) || z<axial_lo || z>=geometry.ProbHi(1)) {
                local_failure=2;
                continue;
            }
            auto const represented = PacketConversionFourMomentumFromPhoton(
                value(PIdx::w, particle), value(PIdx::ux, particle),
                value(PIdx::uy, particle), value(PIdx::uz, particle));
            amrex::Real const radial_offset =
                static_cast<amrex::Real>(radius) - radial_lo;
            amrex::Real const axial_offset =
                static_cast<amrex::Real>(z) - axial_lo;
            amrex::Real const radial_index = std::floor(radial_offset / dr)+domain.smallEnd(0);
            amrex::Real const axial_index = std::floor(axial_offset / dz)+domain.smallEnd(1);
            bool const valid_cell =
                std::isfinite(radial_index) && std::isfinite(axial_index) &&
                radial_index >= domain.smallEnd(0) &&
                radial_index <= domain.bigEnd(0) &&
                axial_index >= domain.smallEnd(1) &&
                axial_index <= domain.bigEnd(1);
            amrex::IntVect cell = domain.smallEnd();
            if (valid_cell)
            {
                cell[0] = static_cast<int>(radial_index);
                cell[1] = static_cast<int>(axial_index);
            }
            if (!ValidPathPhoton(value(PIdx::w, particle), value(PIdx::ux, particle),
                    value(PIdx::uy, particle), value(PIdx::uz, particle), represented) ||
                !amrex::Math::isfinite(radius) ||
                !(radius >= radial_lo) || !(radius < geometry.ProbHi(0)) ||
                !amrex::Math::isfinite(theta) || !amrex::Math::isfinite(z) ||
                !(z >= axial_lo) || !(z < geometry.ProbHi(1)) ||
                !valid_cell || !photon_boxes[tile.key.first].contains(cell))
            {
                local_failure = 2;
                continue;
            }
            plan->idcpu.push_back(idcpu);
            plan->draw_key.push_back(draw_key);
            plan->initial_carries.push_back(carry);
            plan->weight.push_back(value(PIdx::w, particle));
            plan->initial_u.push_back(value(PIdx::ux, particle));
            plan->initial_u.push_back(value(PIdx::uy, particle));
            plan->initial_u.push_back(value(PIdx::uz, particle));
            plan->incoming.insert(
                plan->incoming.end(), represented.value.begin(),
                represented.value.end());
            plan->initial_position.push_back(static_cast<amrex::Real>(radius));
            plan->initial_position.push_back(static_cast<amrex::Real>(theta));
            plan->initial_position.push_back(static_cast<amrex::Real>(z));
            plan->initial_cell.push_back(cell[0]);
            plan->initial_cell.push_back(cell[1]);
        }
    }
    int const global_failure = GlobalFailureCode(local_failure);
    if (global_failure != 0)
    {
        SetPathRejection(report, global_failure);
        return false;
    }
    plan->prepared = true;
    m_impl = std::move(plan);
    report.valid = true;
    return true;
#else
    amrex::ignore_unused(
        private_snapshot, photon_species, scattering_opacity,
        initial_rate_velocity, geometry, dt, options, initial_carries,
        resume_carries, authenticated_empty_duration);
    SetPathRejection(report, 1);
    return false;
#endif
}

bool
RZNativePhotonPathReplay::Replay (
    amrex::MultiFab const& work_velocity,
    WarpXParticleContainer::Base& private_photons,
    amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
    amrex::MultiFab& uncancelled_energy_scale,
    RZNativePhotonPathReport& report) const
{
    return ReplayImpl(work_velocity, private_photons, material_impulse, photon_energy_debit,
                      uncancelled_energy_scale, report, nullptr);
}

bool
RZNativePhotonPathReplay::ReplayAllDNoPhoton (
    amrex::MultiFab const& work_velocity, PhotonPacketTransaction& private_photons,
    NativeRZThermalHistoryStepContext const& context,
    NativeRZThermalAllDNoKEventWitness const& witness,
    amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
    amrex::MultiFab& uncancelled_energy_scale, RZNativePhotonPathReport& report) const
{
    report = RZNativePhotonPathReport{};
    bool const plan = m_impl != nullptr && m_impl->prepared &&
        m_impl->authenticated_empty_duration && m_impl->duration_context == &context &&
        m_impl->duration_witness == &witness && m_impl->photon_species != nullptr;
    if (GlobalFailureCode(plan ? 0 : 1) != 0)
    {
        SetPathRejection(report, 1);
        return false;
    }
    std::string reason;
    if (!witness.ValidatePhotonDuration(context, m_impl->dt, m_impl->options.seed,
            m_impl->options.interval, *m_impl->photon_species, private_photons,
            witness.CanonicalMask(), m_impl->geometry, reason))
    {
        SetPathRejection(report, 1);
        report.reason = reason;
        return false;
    }
    return ReplayImpl(work_velocity, private_photons.Particles(), material_impulse,
                      photon_energy_debit, uncancelled_energy_scale, report, &private_photons);
}

bool
RZNativePhotonPathReplay::ReplayImpl (
    amrex::MultiFab const& work_velocity,
    WarpXParticleContainer::Base& private_photons,
    amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
    amrex::MultiFab& uncancelled_energy_scale,
    RZNativePhotonPathReport& report, PhotonPacketTransaction* authenticated_empty) const
{
    report = RZNativePhotonPathReport{};
    if (GlobalFailureCode(
            m_impl != nullptr && m_impl->prepared &&
                m_impl->authenticated_empty_duration == (authenticated_empty != nullptr) ? 0 : 1)
        != 0)
    {
        SetPathRejection(report, 1);
        return false;
    }

#if defined(WARPX_DIM_RZ)
    int invalid_layout = 0;
    if (private_photons.finestLevel()!=0 ||
        work_velocity.nComp() != 3 || material_impulse.nComp() != 3 ||
        photon_energy_debit.nComp() != 1 ||
        uncancelled_energy_scale.nComp() != 1 ||
        !work_velocity.ixType().cellCentered() ||
        !material_impulse.ixType().cellCentered() ||
        !photon_energy_debit.ixType().cellCentered() ||
        !uncancelled_energy_scale.ixType().cellCentered() ||
        work_velocity.boxArray() != m_impl->box_array ||
        work_velocity.DistributionMap() != m_impl->distribution_map ||
        material_impulse.boxArray() != m_impl->box_array ||
        material_impulse.DistributionMap() != m_impl->distribution_map ||
        photon_energy_debit.boxArray() != m_impl->box_array ||
        photon_energy_debit.DistributionMap() != m_impl->distribution_map ||
        uncancelled_energy_scale.boxArray() != m_impl->box_array ||
        uncancelled_energy_scale.DistributionMap() !=
            m_impl->distribution_map ||
        private_photons.ParticleBoxArray(0) != m_impl->box_array ||
        private_photons.ParticleDistributionMap(0) !=
            m_impl->distribution_map ||
        private_photons.NumRealComps() != m_impl->number_real_components ||
        private_photons.NumIntComps() != m_impl->number_int_components ||
        private_photons.GetRealSoANames() != m_impl->real_names ||
        private_photons.GetIntSoANames() != m_impl->int_names)
    {
        invalid_layout = 1;
    }
    auto const& private_geometry=private_photons.Geom(0);
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        if (private_geometry.Domain()!=m_impl->geometry.Domain() ||
            private_geometry.Coord()!=m_impl->geometry.Coord() ||
            private_geometry.isPeriodic(d)!=m_impl->geometry.isPeriodic(d) ||
            private_geometry.ProbLo(d)!=m_impl->geometry.ProbLo(d) ||
            private_geometry.ProbHi(d)!=m_impl->geometry.ProbHi(d)) { invalid_layout=1; }
    }
    int const global_invalid_layout = GlobalFailureCode(invalid_layout);
    if (global_invalid_layout != 0)
    {
        SetPathRejection(report, global_invalid_layout);
        return false;
    }

    auto const candidate_capture = CaptureParticles(private_photons);
    bool same_particles =
        candidate_capture.tiles.size() == m_impl->baseline.size();
    if (same_particles)
    {
        for (std::size_t tile = 0; tile < m_impl->baseline.size(); ++tile)
        {
            if (candidate_capture.tiles[tile].key !=
                    m_impl->baseline[tile].key ||
                candidate_capture.tiles[tile].count !=
                    m_impl->baseline[tile].count ||
                !SameBits(candidate_capture.tiles[tile].real_data,
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
    if (GlobalFailureCode(same_particles ? 0 : 7) != 0)
    {
        SetPathRejection(report, 7);
        return false;
    }
    if (GlobalFailureCode(
            LocalSuppliedFieldsAlias(
                work_velocity, material_impulse, photon_energy_debit,
                uncancelled_energy_scale) ? 12 : 0) != 0)
    {
        SetPathRejection(report, 12);
        return false;
    }
    if (GlobalFailureCode(
            work_velocity.is_finite() ? 0 : 8) != 0)
    {
        SetPathRejection(report, 8);
        return false;
    }

    amrex::MultiFab staged_velocity(
        m_impl->box_array, m_impl->distribution_map, 3, 1);
    amrex::MultiFab::Copy(staged_velocity, work_velocity, 0, 0, 3, 0);
    staged_velocity.FillBoundary(m_impl->geometry.periodicity());
    m_impl->frozen_opacity.FillBoundary(m_impl->geometry.periodicity());
    m_impl->frozen_rate_velocity.FillBoundary(
        m_impl->geometry.periodicity());
    auto const staged_fields = CaptureHostFields(staged_velocity, 3);
    auto const opacity_fields = CaptureHostFields(m_impl->frozen_opacity, 1);
    auto const rate_fields = CaptureHostFields(m_impl->frozen_rate_velocity, 3);

    int invalid_velocity = 0;
    RZMomentMetric const metric(m_impl->geometry);
    for (amrex::MFIter iterator(staged_velocity); iterator.isValid();
         ++iterator)
    {
        auto const& staged = staged_fields.at(iterator.index());
        for (amrex::BoxIterator cell(iterator.validbox()); cell.ok(); ++cell)
        {
            amrex::Real const rbar =
                metric.Angular((*cell)[0]).mean_radius;
            amrex::Real const outer_radius = metric.Radius((*cell)[0] + 1);
            amrex::Real beta_squared = 0;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                amrex::Real velocity = staged(*cell, dimension);
                if (dimension == 1)
                {
                    velocity *= outer_radius / rbar;
                }
                amrex::Real const beta = velocity / PhysConst::c;
                if (!amrex::Math::isfinite(beta))
                {
                    invalid_velocity = 1;
                }
                beta_squared += beta * beta;
            }
            if (!amrex::Math::isfinite(beta_squared) ||
                !(beta_squared < 1))
            {
                invalid_velocity = 1;
            }
        }
    }
    if (GlobalFailureCode(invalid_velocity) != 0)
    {
        SetPathRejection(report, 8);
        return false;
    }

    amrex::MultiFab candidate_impulse(
        m_impl->box_array, m_impl->distribution_map, 3, 1);
    amrex::MultiFab candidate_debit(
        m_impl->box_array, m_impl->distribution_map, 1, 1);
    amrex::MultiFab candidate_scale(
        m_impl->box_array, m_impl->distribution_map, 1, 1);
    candidate_impulse.setVal(0);
    candidate_debit.setVal(0);
    candidate_scale.setVal(0);
    auto candidate_impulse_host = CaptureHostFields(candidate_impulse, 3);
    auto candidate_debit_host = CaptureHostFields(candidate_debit, 1);
    auto candidate_scale_host = CaptureHostFields(candidate_scale, 1);

    std::vector<amrex::ParticleReal> final_position(
        m_impl->idcpu.size() * 3, 0);
    std::vector<amrex::ParticleReal> final_u(m_impl->idcpu.size() * 3, 0);
    std::vector<int> final_events(m_impl->idcpu.size(), 0);
    std::vector<NativePhotonPathCarry> final_carries;
    final_carries.reserve(m_impl->idcpu.size());
    amrex::GpuArray<amrex::Real, 4> incoming_inventory{};
    amrex::GpuArray<amrex::Real, 4> outgoing_inventory{};
    amrex::GpuArray<amrex::Real, 3> deposited_impulse{};
    amrex::Real deposited_debit = 0;
    amrex::Real deposited_angular_momentum = 0;
    amrex::Real incoming_angular_inventory = 0;
    amrex::Real outgoing_angular_inventory = 0;
    amrex::Real axial_wall_momentum = 0;
    amrex::GpuArray<amrex::Real, 2> axial_wall_by_side{};
    amrex::Real axial_wall_operand = 0;
    amrex::Long axial_wall_transfers = 0;
    std::vector<std::uint64_t> axial_wall_draw_keys;
    amrex::Long collision_events = 0;
    amrex::Long colliding_photons = 0;
    amrex::Long face_crossings = 0;
    amrex::Long wall_reflections = 0;
    int local_failure = 0;
    int const domain_lo_i = m_impl->geometry.Domain().smallEnd(0);
    int const domain_hi_i = m_impl->geometry.Domain().bigEnd(0);
    int const domain_lo_j = m_impl->geometry.Domain().smallEnd(1);
    int const domain_hi_j = m_impl->geometry.Domain().bigEnd(1);
    amrex::Real const radial_lo = m_impl->geometry.ProbLo(0);
    amrex::Real const radial_hi = m_impl->geometry.ProbHi(0);
    amrex::Real const axial_lo = m_impl->geometry.ProbLo(1);
    amrex::Real const axial_hi = m_impl->geometry.ProbHi(1);
    amrex::Real const dr = m_impl->geometry.CellSize(0);
    amrex::Real const dz = m_impl->geometry.CellSize(1);
    bool const periodic_z = m_impl->geometry.isPeriodic(1);

    long global_particle = 0;
    for (auto const& tile : m_impl->baseline)
    {
        long const count = tile.count;
        for (long particle = 0; particle < count;
             ++particle, ++global_particle)
        {
            if (local_failure != 0) { continue; }
            auto const& staged = staged_fields.at(tile.key.first);
            auto const& opacity_field = opacity_fields.at(tile.key.first);
            auto const& rate_field = rate_fields.at(tile.key.first);
            auto& impulse_field = candidate_impulse_host.at(tile.key.first);
            auto& debit_field = candidate_debit_host.at(tile.key.first);
            auto& scale_field = candidate_scale_host.at(tile.key.first);

            auto const weight = m_impl->weight[static_cast<std::size_t>(global_particle)];
            bool const scaled_path = NeedsScaledNativePhotonMomentum(weight);
            ScaledNativePhotonMomentum scaled;
            if (scaled_path)
            {
                scaled = ScaledNativePhotonMomentumFromPhoton(weight,
                    m_impl->initial_u[3 * global_particle],
                    m_impl->initial_u[3 * global_particle + 1],
                    m_impl->initial_u[3 * global_particle + 2]);
            }
            auto represented_vector = [&] (FourVector const& physical)
            {
                return scaled_path ? ScaledPathAccounts(weight, scaled.u_scale, physical) : physical;
            };
            FourVector current{}, initial_account{};
            for (int component = 0; component < 4; ++component)
            {
                initial_account[component] =
                    m_impl->incoming[4 * global_particle + component];
                incoming_inventory[component] += initial_account[component];
                current[component] = scaled_path ? scaled.physical[component] : initial_account[component];
            }
            // The first represented account is exact Prepare readback, not a
            // second formation from normalized kinematics. Thereafter every
            // event/mirror owns its represented before/after account.
            FourVector accounted_current = initial_account;
            amrex::Real radius =
                m_impl->initial_position[3 * global_particle + 0];
            amrex::Real theta =
                m_impl->initial_position[3 * global_particle + 1];
            amrex::Real z =
                m_impl->initial_position[3 * global_particle + 2];
            int radial_cell = m_impl->initial_cell[2 * global_particle + 0];
            int axial_cell = m_impl->initial_cell[2 * global_particle + 1];
            amrex::Real x = radius * std::cos(theta);
            amrex::Real y = radius * std::sin(theta);
            amrex::Real const initial_angular=AngularPhotonMomentum(x,y,initial_account);
            incoming_angular_inventory += initial_angular;
            amrex::Real remaining_time = m_impl->dt;
            std::uint64_t const draw_key =
                m_impl->draw_key[static_cast<std::size_t>(global_particle)];
            NativePhotonPathCarry carry =
                m_impl->initial_carries[static_cast<std::size_t>(global_particle)];
            amrex::Real remaining_depth = carry.residual_optical_depth;
            if (!(remaining_depth > 0))
            {
                remaining_depth = SampleOpticalDepth(
                    m_impl->options, draw_key, carry.next_draw);
                ++carry.next_draw;
            }
            if (!amrex::Math::isfinite(remaining_depth) ||
                !(remaining_depth > 0))
            {
                local_failure = 5;
                break;
            }
            int events = 0;
            int reflections = 0;
            // Axial sign since the last collision (or the initial native
            // state). Rounding after a mirror must return to that event frame.
            int axial_parity = 1;
            int last_axial_wall_side = -1;
            std::array<int, 2> last_event_cell{radial_cell, axial_cell};
            amrex::Real last_event_rbar =
                metric.Angular(radial_cell).mean_radius;
            FourVector last_unrounded{};
            amrex::Real event_debit_sum = 0;
            amrex::GpuArray<amrex::Real, 3> event_impulse_sum{};
            amrex::Real event_angular_momentum = 0;
            long crossings = 0;
            long const maximum_crossings =
                4L * (static_cast<long>(m_impl->options.max_events_per_photon) + 2L)+16;

            while (true)
            {
                amrex::GpuArray<amrex::Real, 3> direction{};
                amrex::Real direction_squared = 0;
                for (int dimension = 0; dimension < 3; ++dimension)
                {
                    direction[dimension] =
                        current[dimension + 1] / current[0];
                    direction_squared += direction[dimension] *
                                         direction[dimension];
                }
                if (!amrex::Math::isfinite(direction_squared) ||
                    !(direction_squared > 0))
                {
                    local_failure = 9;
                    break;
                }
                amrex::IntVect local_cell{radial_cell, axial_cell};
                if (!staged.contains(local_cell) ||
                    !opacity_field.contains(local_cell) ||
                    !rate_field.contains(local_cell))
                {
                    local_failure = 11;
                    break;
                }
                amrex::Real const chi = opacity_field(local_cell, 0);
                if (!amrex::Math::isfinite(chi) || chi < 0)
                {
                    local_failure = 9;
                    break;
                }
                amrex::GpuArray<amrex::Real, 3> frozen_velocity{
                    rate_field(local_cell, 0), rate_field(local_cell, 1),
                    rate_field(local_cell, 2)};
                amrex::Real const end_distance =
                    remaining_time * PhysConst::c;
                if (!std::isfinite(end_distance) || end_distance < 0)
                {
                    local_failure = 9;
                    break;
                }
                auto const face = FindNextRZFace(
                    static_cast<amrex::ParticleReal>(x),
                    static_cast<amrex::ParticleReal>(y),
                    static_cast<amrex::ParticleReal>(z),
                    static_cast<amrex::ParticleReal>(direction[0]),
                    static_cast<amrex::ParticleReal>(direction[1]),
                    static_cast<amrex::ParticleReal>(direction[2]),
                    radial_cell, axial_cell, domain_lo_i, domain_hi_i,
                    domain_lo_j, domain_hi_j,
                    static_cast<amrex::ParticleReal>(radial_lo),
                    static_cast<amrex::ParticleReal>(axial_lo),
                    static_cast<amrex::ParticleReal>(dr),
                    static_cast<amrex::ParticleReal>(dz), periodic_z,
                    static_cast<amrex::ParticleReal>(end_distance));
                if (!face.valid_input || !amrex::Math::isfinite(face.distance) ||
                    face.distance < 0 || face.distance > end_distance *
                        (1 + 64 * std::numeric_limits<amrex::Real>::epsilon()))
                {
                    local_failure = 11;
                    break;
                }

                amrex::Real const march_distance =
                    std::min(face.distance, end_distance);
                amrex::Real const rbar =
                    metric.Angular(radial_cell).mean_radius;
                amrex::Real const marched_depth = IntegratedRZOpticalDepth(
                    chi, frozen_velocity, rbar, x, y, direction,
                    march_distance, m_impl->options.moving_cross_sections);
                if (!std::isfinite(marched_depth) || marched_depth < 0)
                {
                    local_failure = 9;
                    break;
                }
                bool const collision = chi > 0 &&
                    remaining_depth <= marched_depth;
                amrex::Real collision_distance =
                    std::numeric_limits<amrex::Real>::infinity();
                if (collision)
                {
                    if (!m_impl->options.moving_cross_sections)
                    {
                        collision_distance = remaining_depth / chi;
                    }
                    else
                    {
                        amrex::Real lower = 0;
                        amrex::Real upper = march_distance;
                        for (int iteration = 0; iteration < 64; ++iteration)
                        {
                            amrex::Real const middle = lower + 0.5_rt *
                                (upper - lower);
                            if (middle == lower || middle == upper) { break; }
                            amrex::Real const depth = IntegratedRZOpticalDepth(
                                chi, frozen_velocity, rbar, x, y, direction,
                                middle, true);
                            if (!std::isfinite(depth) || depth < 0)
                            {
                                local_failure = 9;
                                break;
                            }
                            if (depth < remaining_depth) { lower = middle; }
                            else { upper = middle; }
                        }
                        collision_distance = upper;
                    }
                }
                if (local_failure != 0) { break; }

                if (collision && collision_distance <= face.distance &&
                    collision_distance <= end_distance)
                {
                    if (carry.completed_events == m_impl->options.max_events_per_photon)
                    {
                        local_failure = 4;
                        break;
                    }
                    x += collision_distance * direction[0];
                    y += collision_distance * direction[1];
                    z += collision_distance * direction[2];
                    remaining_time -= collision_distance / PhysConst::c;
                    remaining_depth = 0;
                    if (!std::isfinite(remaining_time) ||
                        remaining_time < 0)
                    {
                        local_failure = 9;
                        break;
                    }
                    radius = std::hypot(x,y);

                    amrex::Real const cosine = radius > 0 ? x / radius : 1;
                    amrex::Real const sine = radius > 0 ? y / radius : 0;
                    amrex::GpuArray<amrex::Real, 3> beta{};
                    beta[0] = staged(local_cell, 0) * cosine -
                              staged(local_cell, 1) * (radius / rbar) * sine;
                    beta[1] = staged(local_cell, 0) * sine +
                              staged(local_cell, 1) * (radius / rbar) * cosine;
                    beta[2] = staged(local_cell, 2);
                    for (int dimension = 0; dimension < 3; ++dimension)
                    {
                        beta[dimension] /= PhysConst::c;
                    }
                    amrex::GpuArray<amrex::Real, 3> frame_direction =
                        SampleEventDirection(
                            m_impl->options, draw_key, carry.next_draw - 1);
                    if (m_impl->options.angular_law ==
                        NativePhotonCollisionAngularLaw::FixedFrameDirection)
                    {
                        amrex::GpuArray<amrex::Real, 3> local_direction =
                            frame_direction;
                        frame_direction[0] = local_direction[0] * cosine -
                                             local_direction[1] * sine;
                        frame_direction[1] = local_direction[0] * sine +
                                             local_direction[1] * cosine;
                        frame_direction[2] = local_direction[2];
                    }
                    FourVector outgoing{};
                    amrex::GpuArray<amrex::Real, 3> cartesian_impulse{};
                    amrex::Real event_debit = 0;
                    if (!CoherentPhotonCollision(
                            current, frame_direction, beta, outgoing,
                            cartesian_impulse, event_debit))
                    {
                        local_failure = 9;
                        break;
                    }
                    auto const current_account = scaled_path ? accounted_current : current;
                    auto const outgoing_account = represented_vector(outgoing);
                    if (scaled_path)
                    {
                        // Kinematics remain in the unchanged physical null
                        // vector/collision contract. Recoil owns represented
                        // before-after accounts, never the scaled amplitude.
                        for (int d = 0; d < 3; ++d)
                        {
                            cartesian_impulse[d] =
                                (current_account[d + 1] - outgoing_account[d + 1]) / PhysConst::c;
                        }
                        event_debit = current_account[0] - outgoing_account[0];
                    }
                    amrex::Real const radial_impulse =
                        radius > 0 ?
                            (x * cartesian_impulse[0] +
                             y * cartesian_impulse[1]) / radius :
                            cartesian_impulse[0];
                    amrex::Real const conjugate_theta_impulse =
                        (x * cartesian_impulse[1] -
                         y * cartesian_impulse[0]) / rbar;
                    amrex::IntVect event_cell{radial_cell, axial_cell};
                    impulse_field.add(event_cell, 0, radial_impulse);
                    impulse_field.add(
                        event_cell, 1, conjugate_theta_impulse);
                    impulse_field.add(event_cell, 2, cartesian_impulse[2]);
                    debit_field.add(event_cell, 0, event_debit);
                    scale_field.add(
                        event_cell, 0,
                        std::abs(current_account[0]) + std::abs(outgoing_account[0]));
                    deposited_impulse[0] += radial_impulse;
                    deposited_impulse[1] += conjugate_theta_impulse;
                    deposited_impulse[2] += cartesian_impulse[2];
                    deposited_debit += event_debit;
                    deposited_angular_momentum +=
                        rbar * conjugate_theta_impulse;
                    event_impulse_sum[0] += radial_impulse;
                    event_impulse_sum[1] += conjugate_theta_impulse;
                    event_impulse_sum[2] += cartesian_impulse[2];
                    event_debit_sum += event_debit;
                    event_angular_momentum += rbar * conjugate_theta_impulse;
                    last_event_cell = {radial_cell, axial_cell};
                    last_event_rbar = rbar;
                    last_unrounded = outgoing_account;
                    accounted_current = outgoing_account;
                    current = outgoing;
                    axial_parity = 1;
                    last_axial_wall_side = -1;
                    ++events;
                    ++carry.completed_events;
                    ++collision_events;
                    remaining_depth = SampleOpticalDepth(
                        m_impl->options, draw_key, carry.next_draw);
                    ++carry.next_draw;
                    if (!amrex::Math::isfinite(remaining_depth) ||
                        !(remaining_depth > 0))
                    {
                        local_failure = 5;
                        break;
                    }
                    if (remaining_time == 0) { break; }
                    continue;
                }

                amrex::Real const face_distance = face.distance;
                if (face.crossed && face_distance <= end_distance)
                {
                    remaining_depth -= marched_depth;
                    if (!amrex::Math::isfinite(remaining_depth) ||
                        remaining_depth < 0)
                    {
                        local_failure = 9;
                        break;
                    }
                    amrex::Real const previous_x = x;
                    amrex::Real const previous_y = y;
                    x += face_distance * direction[0];
                    y += face_distance * direction[1];
                    z += face_distance * direction[2];
                    if (face.next_radial_cell != radial_cell)
                    {
                        bool const outer_wall =
                            face.next_radial_cell > domain_hi_i;
                        bool const inner_wall =
                            face.next_radial_cell < domain_lo_i;
                        amrex::GpuArray<amrex::Real, 4> reflection{};
                        if (outer_wall)
                        {
                            amrex::GpuArray<amrex::Real, 2> previous{
                                previous_x, previous_y};
                            amrex::GpuArray<amrex::Real, 2> trial{x, y};
                            auto const reflected = ReflectRZTrajectory(
                                previous, trial, radial_hi);
                            if (!reflected.valid)
                            {
                                local_failure = 11;
                                break;
                            }
                            x = reflected.position[0];
                            y = reflected.position[1];
                            amrex::Real const norm=std::hypot(x,y);
                            if (!(norm>0)) { local_failure=11; break; }
                            reflection = RadialReflectionMatrix(x/norm,y/norm);
                        }
                        else if (inner_wall)
                        {
                            amrex::Real const norm =
                                std::sqrt(x * x + y * y);
                            if (!(norm > 0) || !std::isfinite(norm))
                            {
                                local_failure = 11;
                                break;
                            }
                            reflection = RadialReflectionMatrix(
                                x / norm, y / norm);
                        }
                        else
                        {
                            reflection = {1, 0, 0, 1};
                        }
                        if (outer_wall || inner_wall)
                        {
                            ReflectMomentum(current, reflection);
                            if (scaled_path)
                            {
                                accounted_current = represented_vector(current);
                                if (events == 0 && !periodic_z)
                                {
                                    accounted_current[3] = axial_parity * initial_account[3];
                                }
                            }
                            ++reflections;
                            ++wall_reflections;
                        }
                        else
                        {
                            radial_cell = face.next_radial_cell;
                        }
                    }
                    if (face.crossed &&
                        face.next_axial_cell != axial_cell)
                    {
                        bool const end_wall = !periodic_z &&
                            (face.next_axial_cell < domain_lo_j ||
                             face.next_axial_cell > domain_hi_j);
                        if (end_wall)
                        {
                            z = face.next_axial_cell < domain_lo_j ? axial_lo : axial_hi;
                            auto const current_account = scaled_path ? accounted_current : current;
                            auto const transfer = 2 * current_account[3] / PhysConst::c;
                            axial_wall_momentum += transfer;
                            last_axial_wall_side = face.next_axial_cell < domain_lo_j ? 0 : 1;
                            axial_wall_by_side[last_axial_wall_side] += transfer;
                            axial_wall_operand += std::abs(transfer);
                            ++axial_wall_transfers;
                            axial_wall_draw_keys.push_back(draw_key);
                            current[3] = -current[3];
                            if (scaled_path) { accounted_current[3] = -accounted_current[3]; }
                            axial_parity = -axial_parity;
                            ++wall_reflections;
                        }
                        else
                        {
                            axial_cell = face.next_axial_cell;
                        }
                    }
                    remaining_time -= face_distance / PhysConst::c;
                    ++crossings;
                    ++face_crossings;
                    if (crossings > maximum_crossings ||
                        remaining_time < 0)
                    {
                        local_failure = 11;
                        break;
                    }
                    if (remaining_time == 0) { break; }
                    continue;
                }

                remaining_depth -= marched_depth;
                if (!std::isfinite(remaining_depth) || !(remaining_depth > 0))
                {
                    local_failure = 9;
                    break;
                }
                x += end_distance * direction[0];
                y += end_distance * direction[1];
                z += end_distance * direction[2];
                remaining_time = 0;
                break;
            }
            if (local_failure != 0) { break; }
            carry.residual_optical_depth = remaining_depth;
            final_carries.push_back(carry);

            amrex::ParticleReal rounded_u[3] = {};
            amrex::Real const denominator = scaled_path ? 1 : m_impl->weight[
                static_cast<std::size_t>(global_particle)] *
                PhysConst::m_e * PhysConst::c;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                rounded_u[dimension] = static_cast<amrex::ParticleReal>(
                    events == 0 && (reflections == 0 || (!periodic_z && dimension == 2)) ?
                        (dimension == 2 ? axial_parity : 1) *
                            m_impl->initial_u[3*global_particle+dimension] :
                        (scaled_path ? current[dimension + 1] * scaled.u_scale :
                                       current[dimension + 1] / denominator));
                final_u[3 * global_particle + dimension] =
                    rounded_u[dimension];
            }
            auto const represented = PacketConversionFourMomentumFromPhoton(
                m_impl->weight[static_cast<std::size_t>(global_particle)],
                rounded_u[0], rounded_u[1], rounded_u[2]);
            if (!ValidPathPhoton(weight, rounded_u[0], rounded_u[1], rounded_u[2], represented))
            {
                local_failure = 9;
                break;
            }
            radius = std::hypot(x,y);
            amrex::Real const radial_roundoff=64*std::numeric_limits<amrex::Real>::epsilon()*radial_hi;
            if (radius>=radial_hi && radius-radial_hi<=radial_roundoff) {
                radius=std::nextafter(radial_hi,radial_lo);
            }
            if (radius<radial_lo && radial_lo-radius<=radial_roundoff) { radius=radial_lo; }
            theta = CanonicalTheta(std::atan2(y, x));
            if (periodic_z)
            {
                z = WrapPeriodicZ(m_impl->geometry, z);
            }
            else
            {
                amrex::Real const allowance = 64 * std::numeric_limits<amrex::Real>::epsilon() *
                    std::max({std::abs(axial_lo), std::abs(axial_hi), dz});
                if (z >= axial_hi && z - axial_hi <= allowance)
                {
                    z = std::nextafter(axial_hi, axial_lo);
                }
                if (z < axial_lo && axial_lo - z <= allowance) { z = axial_lo; }
            }
            amrex::Real const published_theta =
                static_cast<amrex::Real>(theta);
            amrex::Real const published_z = static_cast<amrex::Real>(z);
            amrex::Real const published_x =
                static_cast<amrex::Real>(radius) * std::cos(published_theta);
            amrex::Real const published_y =
                static_cast<amrex::Real>(radius) * std::sin(published_theta);
            if (!std::isfinite(radius) || radius < radial_lo ||
                radius > radial_hi || !std::isfinite(published_theta) ||
                !std::isfinite(published_z) || published_z < axial_lo ||
                published_z >= axial_hi)
            {
                local_failure = 11;
                break;
            }
            final_position[3 * global_particle + 0] =
                static_cast<amrex::ParticleReal>(radius);
            final_position[3 * global_particle + 1] =
                static_cast<amrex::ParticleReal>(published_theta);
            final_position[3 * global_particle + 2] =
                static_cast<amrex::ParticleReal>(published_z);
            final_events[static_cast<std::size_t>(global_particle)] =
                carry.completed_events;

            for (int component = 0; component < 4; ++component)
            {
                outgoing_inventory[component] += represented.value[component];
            }
            amrex::Real const final_angular=AngularPhotonMomentum(
                published_x, published_y, represented.value);
            outgoing_angular_inventory += final_angular;

            if (events != 0)
            {
                auto const current_account = scaled_path ? accounted_current : current;
                amrex::IntVect correction_cell{
                    last_event_cell[0], last_event_cell[1]};
                if (!impulse_field.contains(correction_cell) ||
                    !debit_field.contains(correction_cell) ||
                    !scale_field.contains(correction_cell))
                {
                    local_failure = 11;
                    break;
                }
                amrex::Real const axial_correction = periodic_z ?
                    (m_impl->incoming[4 * global_particle + 3] -
                     represented.value[3]) / PhysConst::c -
                    event_impulse_sum[2] :
                    (last_unrounded[3] - axial_parity * represented.value[3]) / PhysConst::c;
                if (!periodic_z)
                {
                    auto const correction =
                        (current_account[3] - represented.value[3]) / PhysConst::c - axial_correction;
                    axial_wall_momentum += correction;
                    // No mirror since the last collision gives exactly the
                    // same two subtractions above, hence zero correction.
                    if (last_axial_wall_side >= 0)
                    {
                        axial_wall_by_side[last_axial_wall_side] += correction;
                        axial_wall_operand += std::abs(correction);
                        ++axial_wall_transfers;
                    }
                    else if (correction != 0)
                    {
                        local_failure = 11;
                        break;
                    }
                }
                amrex::Real const energy_correction =
                    m_impl->incoming[4 * global_particle + 0] -
                    represented.value[0] - event_debit_sum;
                amrex::Real const angular_correction =
                    (initial_angular-final_angular)/PhysConst::c;
                amrex::Real const angular_correction_momentum =
                    angular_correction - event_angular_momentum;
                amrex::Real const theta_correction =
                    angular_correction_momentum / last_event_rbar;
                impulse_field.add(correction_cell, 1, theta_correction);
                impulse_field.add(correction_cell, 2, axial_correction);
                debit_field.add(correction_cell, 0, energy_correction);
                scale_field.add(
                    correction_cell, 0,
                    std::abs(last_unrounded[0]) +
                        std::abs(represented.value[0]));
                deposited_impulse[1] += theta_correction;
                deposited_impulse[2] += axial_correction;
                deposited_debit += energy_correction;
                deposited_angular_momentum +=
                    last_event_rbar * theta_correction;
                ++colliding_photons;
            }
        }
        if (local_failure != 0) { break; }
    }
    int const global_failure = GlobalFailureCode(local_failure);
    if (global_failure != 0)
    {
        SetPathRejection(report, global_failure);
        return false;
    }

    RestoreHostFields(candidate_impulse, candidate_impulse_host);
    RestoreHostFields(candidate_debit, candidate_debit_host);
    RestoreHostFields(candidate_scale, candidate_scale_host);
    candidate_impulse.SumBoundary(m_impl->geometry.periodicity());
    candidate_debit.SumBoundary(m_impl->geometry.periodicity());
    candidate_scale.SumBoundary(m_impl->geometry.periodicity());
    auto const summed_impulse_host = CaptureHostFields(candidate_impulse, 3);
    auto const summed_debit_host = CaptureHostFields(candidate_debit, 1);
    amrex::Real field_impulse[3] = {};
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        field_impulse[dimension] = SumValidHost(
            candidate_impulse, summed_impulse_host, dimension);
    }
    amrex::Real const field_debit = SumValidHost(
        candidate_debit, summed_debit_host, 0);
    amrex::Real const field_angular_momentum = PhysConst::c *
        SumAngularValidHost(
            candidate_impulse, summed_impulse_host, m_impl->geometry);

    amrex::Real reduced_values[16] = {};
    for (int component = 0; component < 4; ++component)
    {
        reduced_values[component] = incoming_inventory[component];
        reduced_values[4 + component] = outgoing_inventory[component];
    }
    amrex::ParallelDescriptor::ReduceRealSum(reduced_values, 8);
    amrex::ParallelDescriptor::ReduceRealSum(incoming_angular_inventory);
    amrex::ParallelDescriptor::ReduceRealSum(outgoing_angular_inventory);
    amrex::ParallelDescriptor::ReduceRealSum(axial_wall_momentum);
    amrex::ParallelDescriptor::ReduceRealSum(axial_wall_by_side.data(), 2);
    amrex::ParallelDescriptor::ReduceRealSum(axial_wall_operand);
    amrex::Real deposited_values[5] = {
        deposited_impulse[0], deposited_impulse[1], deposited_impulse[2],
        deposited_debit, deposited_angular_momentum};
    amrex::ParallelDescriptor::ReduceRealSum(deposited_values, 5);

    amrex::Real const energy_residual =
        field_debit - (reduced_values[0] - reduced_values[4]);
    amrex::Real const energy_scale = std::abs(field_debit) +
        reduced_values[0] + reduced_values[4];
    amrex::Real const axial_residual = PhysConst::c * (field_impulse[2] + axial_wall_momentum) -
        (reduced_values[3] - reduced_values[7]);
    amrex::Real const axial_scale = std::abs(PhysConst::c * field_impulse[2]) +
        reduced_values[0] + reduced_values[4];
    amrex::Real const angular_residual = field_angular_momentum -
        (incoming_angular_inventory - outgoing_angular_inventory);
    amrex::Real const angular_scale = std::abs(field_angular_momentum) +
        radial_hi*(reduced_values[0]+reduced_values[4]);
    bool valid = amrex::Math::isfinite(energy_residual) &&
        amrex::Math::isfinite(energy_scale) &&
        amrex::Math::isfinite(axial_residual) &&
        amrex::Math::isfinite(axial_scale) &&
        amrex::Math::isfinite(angular_residual) &&
        amrex::Math::isfinite(angular_scale);
    amrex::Real const roundoff = 4096 *
        std::numeric_limits<amrex::Real>::epsilon();
    valid = valid && std::abs(energy_residual) <= roundoff * energy_scale;
    valid = valid && std::abs(axial_residual) <= roundoff * axial_scale;
    valid = valid && std::abs(angular_residual) <= roundoff * angular_scale;
    valid = valid && candidate_impulse.is_finite() &&
        candidate_debit.is_finite() && candidate_scale.is_finite();
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        valid = valid && std::abs(PhysConst::c *
            (field_impulse[dimension] - deposited_values[dimension])) <=
            roundoff * axial_scale;
    }
    valid = valid && std::abs(field_debit - deposited_values[3]) <=
        roundoff * energy_scale;
    valid = valid && std::abs(field_angular_momentum -
        PhysConst::c * deposited_values[4]) <= roundoff * angular_scale;
    if (GlobalFailureCode(valid ? 0 : 10) != 0)
    {
        SetPathRejection(report, 10);
        return false;
    }

    if (authenticated_empty != nullptr)
    {
        std::string reason;
        bool const authority = m_impl->duration_witness->ValidatePhotonDuration(
            *m_impl->duration_context, m_impl->dt, m_impl->options.seed, m_impl->options.interval,
            *m_impl->photon_species, *authenticated_empty,
            m_impl->duration_witness->CanonicalMask(), m_impl->geometry, reason);
        // Inspect genuine graph results, not manufactured empty accounts.
        // Read every norm collectively before combining local predicates.
        auto const impulse_norm = candidate_impulse.norm0(0, 3, candidate_impulse.nGrowVect());
        auto const debit_norm = candidate_debit.norm0(0, 1, candidate_debit.nGrowVect());
        auto const scale_norm = candidate_scale.norm0(0, 1, candidate_scale.nGrowVect());
        bool empty = m_impl->idcpu.empty() && m_impl->initial_carries.empty() &&
            final_carries.empty() && final_position.empty() && final_u.empty() &&
            final_events.empty() && axial_wall_draw_keys.empty() && collision_events == 0 &&
            colliding_photons == 0 && face_crossings == 0 && wall_reflections == 0 &&
            axial_wall_transfers == 0 && axial_wall_momentum == 0 && axial_wall_operand == 0 &&
            incoming_angular_inventory == 0 && outgoing_angular_inventory == 0 &&
            field_angular_momentum == 0 && impulse_norm == 0 && debit_norm == 0 && scale_norm == 0;
        for (auto value : reduced_values) { empty = value == 0 && empty; }
        for (auto value : deposited_values) { empty = value == 0 && empty; }
        for (auto value : axial_wall_by_side) { empty = value == 0 && empty; }
        if (GlobalFailureCode(authority && empty ? 0 : 1) != 0)
        {
            SetPathRejection(report, 1);
            report.reason = authority
                ? "all-D implicit path produced nonempty real accounts" : reason;
            return false;
        }
    }

    auto replay_rollback = std::make_unique<WarpXParticleContainer::Base>(
        private_photons.make_alike<>());
    replay_rollback->SetArena(private_photons.arena());
    replay_rollback->copyParticles(private_photons, true);

    std::map<std::pair<int, int>, long> tile_offsets;
    long next_offset = 0;
    for (auto const& tile : m_impl->baseline)
    {
        tile_offsets.emplace(tile.key, next_offset);
        next_offset += tile.count;
    }
    amrex::Gpu::DeviceVector<amrex::ParticleReal> device_position(
        final_position.size());
    amrex::Gpu::DeviceVector<amrex::ParticleReal> device_u(final_u.size());
    amrex::Gpu::DeviceVector<int> device_events(final_events.size());
    if (!final_position.empty())
    {
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, final_position.begin(),
            final_position.end(), device_position.begin());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, final_u.begin(), final_u.end(),
            device_u.begin());
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, final_events.begin(),
            final_events.end(), device_events.begin());
    }
    bool const census_available = m_impl->census_available;
    amrex::GpuArray<int,3> const census_components{m_impl->census_components[0],
        m_impl->census_components[1],m_impl->census_components[2]};
    for (auto const& tile : m_impl->baseline)
    {
        long const count = tile.count;
        if (count == 0) { continue; }
        auto& particle_tile=private_photons.GetParticles(0).at(tile.key);
        auto data = particle_tile.getParticleTileData();
        amrex::GpuArray<amrex::ParticleReal*,3> census_data{};
        if (census_available) {
            for (int d=0; d<3; ++d) {
                census_data[d]=particle_tile.GetStructOfArrays().GetRealData(
                    census_components[d]).dataPtr();
            }
        }
        amrex::ParticleReal const* const position = device_position.dataPtr();
        amrex::ParticleReal const* const momentum = device_u.dataPtr();
        int const* const event_count = device_events.dataPtr();
        long const offset = tile_offsets.at(tile.key);
        WriteReplayTile(count, offset, data, position, momentum, event_count,
                        census_available, census_data);
    }
    amrex::Gpu::streamSynchronize();
    private_photons.Redistribute();
    std::vector<std::uint64_t> migrated_ids, migrated_keys, migrated_draws;
    bool const routing_valid = photon_replay_detail::CapturePhotonRoutingKeys(
        private_photons, *m_impl->photon_species, m_impl->snapshot_policy, migrated_ids,
        migrated_keys, migrated_draws);
    std::vector<NativePhotonPathCarry> migrated_carries;
    if (GlobalFailureCode(routing_valid ? 0 : 1) != 0 ||
        !(m_impl->snapshot_policy == NativePhotonPathSnapshotPolicy::NativeOnly
              ? photon_replay_detail::RoutePhotonCarries(final_carries, migrated_ids,
                                                         m_impl->options.max_events_per_photon,
                                                         migrated_carries)
              : photon_replay_detail::RouteKeyedPhotonCarries(
                    final_carries, migrated_ids, migrated_keys,
                    m_impl->options.max_events_per_photon, migrated_carries))) {
        private_photons.copyParticles(*replay_rollback, true);
        SetPathRejection(report, 7);
        return false;
    }

    if (authenticated_empty != nullptr)
    {
        std::string reason;
        bool const authority = m_impl->duration_witness->ValidatePhotonDuration(
            *m_impl->duration_context, m_impl->dt, m_impl->options.seed, m_impl->options.interval,
            *m_impl->photon_species, *authenticated_empty,
            m_impl->duration_witness->CanonicalMask(), m_impl->geometry, reason);
        bool const empty = GlobalFailureCode(migrated_carries.empty() && migrated_ids.empty() &&
            migrated_keys.empty() && migrated_draws.empty() ? 0 : 1) == 0;
        if (!authority || !empty)
        {
            private_photons.copyParticles(*replay_rollback, true);
            SetPathRejection(report, 1);
            report.reason = authority
                ? "all-D implicit path migrated unexpected actual records" : reason;
            return false;
        }
    }

    amrex::MultiFab::Copy(material_impulse, candidate_impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(
        photon_energy_debit, candidate_debit, 0, 0, 1, 0);
    amrex::MultiFab::Copy(
        uncancelled_energy_scale, candidate_scale, 0, 0, 1, 0);

    amrex::ParallelDescriptor::ReduceLongSum(collision_events);
    amrex::ParallelDescriptor::ReduceLongSum(colliding_photons);
    amrex::ParallelDescriptor::ReduceLongSum(face_crossings);
    amrex::ParallelDescriptor::ReduceLongSum(wall_reflections);
    report.valid = true;
    report.photon_carries = std::move(migrated_carries);
    report.colliding_photons = colliding_photons;
    report.collision_events = collision_events;
    report.face_crossings = face_crossings;
    report.wall_reflections = wall_reflections;
    report.wall_momentum[2] = axial_wall_momentum;
    report.axial_wall_momentum_by_side = axial_wall_by_side;
    report.axial_wall_momentum_operand = axial_wall_operand;
    auto replayed_photons = static_cast<amrex::Long>(m_impl->idcpu.size());
    amrex::ParallelDescriptor::ReduceLongSum(replayed_photons);
    amrex::ParallelDescriptor::ReduceLongSum(axial_wall_transfers);
    report.replayed_photons = replayed_photons;
    report.axial_wall_transfers = axial_wall_transfers;
    report.axial_wall_draw_keys = std::move(axial_wall_draw_keys);
    for (int component = 0; component < 4; ++component)
    {
        report.incoming_inventory[component] = reduced_values[component];
        report.outgoing_inventory[component] =
            reduced_values[4 + component];
    }
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        report.material_impulse[dimension] = field_impulse[dimension];
    }
    report.photon_energy_debit = field_debit;
    return true;
#else
    amrex::ignore_unused(
        work_velocity, private_photons, material_impulse,
        photon_energy_debit, uncancelled_energy_scale, authenticated_empty);
    SetPathRejection(report, 1);
    return false;
#endif
}

struct RZNativePhotonPathSubsegmentReplay::Impl
{
    WarpXParticleContainer* photon_species = nullptr;
    std::unique_ptr<amrex::MultiFab> scattering_opacity;
    std::unique_ptr<amrex::MultiFab> initial_rate_velocity;
    amrex::Geometry geometry{};
    NativePhotonCollisionOptions options{};
    amrex::Real subsegment_dt = 0;
    int subsegments = 0;
    NativePhotonPathSnapshotPolicy snapshot_policy = NativePhotonPathSnapshotPolicy::NativeOnly;
    bool freeze_private_snapshot = false;
    std::vector<photon_replay_detail::CapturedTile> baseline;
    bool prepared = false;
};

RZNativePhotonPathSubsegmentReplay::RZNativePhotonPathSubsegmentReplay ()
    : m_impl(std::make_unique<Impl>()) {}

RZNativePhotonPathSubsegmentReplay::~RZNativePhotonPathSubsegmentReplay () = default;

bool
RZNativePhotonPathSubsegmentReplay::Prepare (WarpXParticleContainer& photon_species,
                                             amrex::MultiFab const& scattering_opacity,
                                             amrex::MultiFab const& initial_rate_velocity,
                                             amrex::Geometry const& geometry, amrex::Real dt,
                                             NativePhotonCollisionOptions const& options,
                                             int subsegment_limit, RZNativePhotonPathReport& report)
{
    return PrepareImpl(photon_species, photon_species, scattering_opacity, initial_rate_velocity,
                       geometry, dt, options, subsegment_limit, report,
                       NativePhotonPathSnapshotPolicy::NativeOnly, false);
}

bool
RZNativePhotonPathSubsegmentReplay::Prepare (
    WarpXParticleContainer::Base const& private_snapshot, WarpXParticleContainer& photon_species,
    amrex::MultiFab const& scattering_opacity, amrex::MultiFab const& initial_rate_velocity,
    amrex::Geometry const& geometry, amrex::Real dt, NativePhotonCollisionOptions const& options,
    int subsegment_limit, RZNativePhotonPathReport& report, NativePhotonPathSnapshotPolicy policy)
{
    return PrepareImpl(private_snapshot, photon_species, scattering_opacity, initial_rate_velocity,
                       geometry, dt, options, subsegment_limit, report, policy, true);
}

bool
RZNativePhotonPathSubsegmentReplay::PrepareImpl (
    WarpXParticleContainer::Base const& private_snapshot, WarpXParticleContainer& photon_species,
    amrex::MultiFab const& scattering_opacity, amrex::MultiFab const& initial_rate_velocity,
    amrex::Geometry const& geometry, amrex::Real dt, NativePhotonCollisionOptions const& options,
    int subsegment_limit, RZNativePhotonPathReport& report, NativePhotonPathSnapshotPolicy policy,
    bool freeze_private_snapshot)
{
    m_impl = std::make_unique<Impl>();
    report = RZNativePhotonPathReport{};
    int invalid = 0;
    int minimum_limit = subsegment_limit;
    int maximum_limit = subsegment_limit;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_limit);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_limit);
#if defined(WARPX_DIM_RZ)
    if (!SameOptionsAcrossRanks(options, dt) ||
        !SameGeometryAcrossRanks(geometry)) { invalid = 1; }
#else
    invalid = 1;
#endif
    if (minimum_limit != maximum_limit || subsegment_limit <= 0 ||
        !std::isfinite(dt) || !(dt > 0) ||
        !std::isfinite(options.max_optical_depth) ||
        !(options.max_optical_depth > 0)) { invalid = 1; }
    amrex::Real minimum_cell = geometry.CellSize(0);
    for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
    {
        amrex::Real const cell = geometry.CellSize(dimension);
        if (!std::isfinite(cell) || !(cell > 0)) { invalid = 1; }
        minimum_cell = std::min(minimum_cell, cell);
    }
    amrex::Real const length = PhysConst::c * dt;
    if (!std::isfinite(length) || !(length > 0)) { invalid = 1; }
    long subsegments = 0;
    if (invalid == 0)
    {
        long double const ratio = 2 * static_cast<long double>(length) /
            static_cast<long double>(minimum_cell);
        if (!std::isfinite(ratio) || ratio > subsegment_limit)
        {
            invalid = 1;
        }
        else
        {
            subsegments = std::max(1L, static_cast<long>(std::ceil(ratio)));
            auto const represented_safe = [&] (long count)
            {
                amrex::Real const segment_length =
                    PhysConst::c * (dt / static_cast<amrex::Real>(count));
                for (int dimension = 0; dimension < AMREX_SPACEDIM; ++dimension)
                {
                    if (segment_length > 0.5_rt * geometry.CellSize(dimension))
                    {
                        return false;
                    }
                }
                return true;
            };
            if (!represented_safe(subsegments)) { ++subsegments; }
            if (subsegments > subsegment_limit || !represented_safe(subsegments))
            {
                invalid = 1;
            }
        }
    }
    if (GlobalFailureCode(invalid) != 0)
    {
        report.reason = "invalid collective RZ packet subsegment controls";
        return false;
    }
    RZNativePhotonPathReplay preflight;
    if (!preflight.Prepare(private_snapshot, photon_species, scattering_opacity,
            initial_rate_velocity, geometry,
            dt / static_cast<amrex::Real>(subsegments), options, report, policy))
    {
        return false;
    }
#if defined(WARPX_DIM_RZ)
    auto const [opacity_fields, rate_fields] = CaptureFrozenGrids(
        scattering_opacity, initial_rate_velocity, geometry);
    RZMomentMetric const metric(geometry);
    for (amrex::MFIter iterator(scattering_opacity); iterator.isValid(); ++iterator)
    {
        auto const& opacity = opacity_fields.at(iterator.index());
        auto const& rate = rate_fields.at(iterator.index());
        for (amrex::BoxIterator cell(iterator.validbox()); cell.ok(); ++cell)
        {
            amrex::Real const chi = opacity(*cell, 0);
            amrex::Real const rbar = metric.Angular((*cell)[0]).mean_radius;
            amrex::Real const outer = metric.Radius((*cell)[0] + 1);
            amrex::Real beta2 = 0;
            for (int dimension = 0; dimension < 3; ++dimension)
            {
                amrex::Real velocity = rate(*cell, dimension);
                if (dimension == 1) { velocity *= outer / rbar; }
                amrex::Real const beta = velocity / PhysConst::c;
                beta2 += beta * beta;
            }
            if (!std::isfinite(chi) || chi < 0 || !std::isfinite(beta2) ||
                !(beta2 < 1) ||
                (options.moving_cross_sections && beta2 > 1.e-6_rt)) { invalid = 1; }
            amrex::Real const beta = std::sqrt(beta2);
            amrex::Real const hazard = chi * length *
                (options.moving_cross_sections ?
                    (1 + beta) / std::sqrt(1 - beta2) : amrex::Real(1));
            if (!std::isfinite(hazard) || hazard < 0 ||
                hazard > options.max_optical_depth) { invalid = 3; }
        }
    }
#endif
    if (GlobalFailureCode(invalid) != 0)
    {
        report.valid = false;
        report.reason = "invalid outer RZ packet optical-depth envelope";
        return false;
    }
    m_impl->photon_species = &photon_species;
    m_impl->scattering_opacity = std::make_unique<amrex::MultiFab>(
        scattering_opacity.boxArray(), scattering_opacity.DistributionMap(), 1, 0);
    m_impl->initial_rate_velocity = std::make_unique<amrex::MultiFab>(
        initial_rate_velocity.boxArray(), initial_rate_velocity.DistributionMap(), 3, 0);
    amrex::MultiFab::Copy(*m_impl->scattering_opacity, scattering_opacity, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*m_impl->initial_rate_velocity, initial_rate_velocity,
        0, 0, 3, 0);
    m_impl->geometry = geometry;
    m_impl->options = options;
    m_impl->subsegment_dt = dt / static_cast<amrex::Real>(subsegments);
    m_impl->subsegments = static_cast<int>(subsegments);
    m_impl->snapshot_policy = policy;
    m_impl->freeze_private_snapshot = freeze_private_snapshot;
    if (freeze_private_snapshot) { m_impl->baseline = CaptureParticles(private_snapshot).tiles; }
    m_impl->prepared = true;
    report.valid = true;
    return true;
}

bool
RZNativePhotonPathSubsegmentReplay::Replay (
    amrex::MultiFab const& work_velocity,
    WarpXParticleContainer::Base& private_photons,
    amrex::MultiFab& material_impulse, amrex::MultiFab& photon_energy_debit,
    amrex::MultiFab& uncancelled_energy_scale,
    NativePhotonPathSubsegmentReport& report) const
{
    report = NativePhotonPathSubsegmentReport{};
    if (GlobalFailureCode(m_impl && m_impl->prepared ? 0 : 1) != 0)
    {
        report.path.reason = "unprepared RZ packet subsegment replay";
        return false;
    }
    int invalid = private_photons.finestLevel() != 0 ||
        private_photons.ParticleBoxArray(0) != m_impl->scattering_opacity->boxArray() ||
        private_photons.ParticleDistributionMap(0) !=
            m_impl->scattering_opacity->DistributionMap() ||
        work_velocity.nComp() != 3 || material_impulse.nComp() != 3 ||
        photon_energy_debit.nComp() != 1 || uncancelled_energy_scale.nComp() != 1;
    for (amrex::MultiFab const* field :
         std::array<amrex::MultiFab const*, 4>{
             &work_velocity, &material_impulse, &photon_energy_debit,
             &uncancelled_energy_scale})
    {
        if (field->boxArray() != m_impl->scattering_opacity->boxArray() ||
            field->DistributionMap() != m_impl->scattering_opacity->DistributionMap() ||
            !field->ixType().cellCentered()) { invalid = 1; }
    }
#if defined(WARPX_DIM_RZ)
    if (LocalSuppliedFieldsAlias(work_velocity, material_impulse,
            photon_energy_debit, uncancelled_energy_scale)) { invalid = 1; }
#endif
    if (GlobalFailureCode(invalid) != 0)
    {
        report.path.reason = "invalid RZ packet subsegment output layout or alias";
        return false;
    }
    std::vector<std::uint64_t> ids, keys, draws;
    if (GlobalFailureCode(
            photon_replay_detail::CapturePhotonRoutingKeys(
                private_photons, *m_impl->photon_species, m_impl->snapshot_policy, ids, keys, draws)
                ? 0
                : 1) != 0 ||
        (m_impl->freeze_private_snapshot &&
         GlobalFailureCode(
             SameCapturedTiles(CaptureParticles(private_photons).tiles, m_impl->baseline)
                 ? 0
                 : 1) != 0)) {
        report.path.reason = "private RZ photon schema or frozen snapshot mismatch";
        return false;
    }
    auto private_trial =
        std::make_unique<WarpXParticleContainer::Base>(private_photons.make_alike<>());
    private_trial->SetArena(private_photons.arena());
    private_trial->copyParticles(private_photons, true);
    amrex::MultiFab accumulated_impulse(material_impulse.boxArray(),
        material_impulse.DistributionMap(), 3, 0);
    amrex::MultiFab accumulated_debit(photon_energy_debit.boxArray(),
        photon_energy_debit.DistributionMap(), 1, 0);
    amrex::MultiFab accumulated_scale(uncancelled_energy_scale.boxArray(),
        uncancelled_energy_scale.DistributionMap(), 1, 0);
    accumulated_impulse.setVal(0);
    accumulated_debit.setVal(0);
    accumulated_scale.setVal(0);
    std::vector<NativePhotonPathCarry> carries;
    for (int segment = 0; segment < m_impl->subsegments; ++segment)
    {
        RZNativePhotonPathReplay path;
        NativePhotonPathReport path_report;
        bool const prepared =
            segment == 0 ? path.Prepare(*private_trial, *m_impl->photon_species,
                                        *m_impl->scattering_opacity, *m_impl->initial_rate_velocity,
                                        m_impl->geometry, m_impl->subsegment_dt, m_impl->options,
                                        path_report, m_impl->snapshot_policy)
                         : path.Prepare(*private_trial, *m_impl->photon_species,
                                        *m_impl->scattering_opacity, *m_impl->initial_rate_velocity,
                                        m_impl->geometry, m_impl->subsegment_dt, m_impl->options,
                                        carries, path_report, m_impl->snapshot_policy);
        if (!prepared) {
            report.path = path_report;
            return false;
        }
        amrex::MultiFab segment_impulse(material_impulse.boxArray(),
            material_impulse.DistributionMap(), 3, 0);
        amrex::MultiFab segment_debit(photon_energy_debit.boxArray(),
            photon_energy_debit.DistributionMap(), 1, 0);
        amrex::MultiFab segment_scale(uncancelled_energy_scale.boxArray(),
            uncancelled_energy_scale.DistributionMap(), 1, 0);
        if (!path.Replay(work_velocity, *private_trial,
                segment_impulse, segment_debit, segment_scale, path_report))
        {
            report.path = path_report;
            return false;
        }
        amrex::MultiFab::Add(accumulated_impulse, segment_impulse, 0, 0, 3, 0);
        amrex::MultiFab::Add(accumulated_debit, segment_debit, 0, 0, 1, 0);
        amrex::MultiFab::Add(accumulated_scale, segment_scale, 0, 0, 1, 0);
        if (segment == 0)
        {
            report.path.incoming_inventory = path_report.incoming_inventory;
        }
        report.path.outgoing_inventory = path_report.outgoing_inventory;
        report.path.collision_events += path_report.collision_events;
        report.path.face_crossings += path_report.face_crossings;
        report.path.wall_reflections += path_report.wall_reflections;
        for (int side = 0; side < 2; ++side)
        {
            report.path.axial_wall_momentum_by_side[side] +=
                path_report.axial_wall_momentum_by_side[side];
        }
        report.path.axial_wall_momentum_operand += path_report.axial_wall_momentum_operand;
        report.path.replayed_photons += path_report.replayed_photons;
        report.path.axial_wall_transfers += path_report.axial_wall_transfers;
        report.path.axial_wall_draw_keys.insert(report.path.axial_wall_draw_keys.end(),
            path_report.axial_wall_draw_keys.begin(), path_report.axial_wall_draw_keys.end());
        for (int d = 0; d < 3; ++d)
        {
            report.path.material_impulse[d] += path_report.material_impulse[d];
            report.path.wall_momentum[d] += path_report.wall_momentum[d];
        }
        report.path.photon_energy_debit += path_report.photon_energy_debit;
        carries = std::move(path_report.photon_carries);
    }
    if (GlobalFailureCode(accumulated_impulse.is_finite() &&
            accumulated_debit.is_finite() && accumulated_scale.is_finite() ? 0 : 1) != 0)
    {
        report.path.reason = "nonfinite accumulated RZ packet exchange";
        return false;
    }
    amrex::Long colliding_photons = 0;
    for (auto const& carry : carries)
    {
        if (carry.completed_events > 0) { ++colliding_photons; }
    }
    amrex::ParallelDescriptor::ReduceLongSum(colliding_photons);
    private_photons.copyParticles(*private_trial, true);
    private_photons.Redistribute();
    amrex::MultiFab::Copy(material_impulse, accumulated_impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(photon_energy_debit, accumulated_debit, 0, 0, 1, 0);
    amrex::MultiFab::Copy(uncancelled_energy_scale, accumulated_scale, 0, 0, 1, 0);
    report.subsegments = m_impl->subsegments;
    report.path.valid = true;
    report.path.colliding_photons = colliding_photons;
    report.path.photon_carries = carries;
    report.photon_carries = std::move(carries);
    return true;
}
}
