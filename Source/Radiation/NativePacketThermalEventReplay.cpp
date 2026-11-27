/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativePacketThermalEventReplay.H"

#include "FiniteMassPhotonCollision.H"
#include "FluxRelaxedPackets.H"
#include "NativePhotonReplayInternals.H"
#include "PhotonFourMomentum.H"
#include "PhotonLogicalDrawKey.H"
#include "PhotonPacketTransaction.H"
#include "RZMomentGeometry.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace warpx::radiation
{
using namespace amrex::literals;
using photon_replay_detail::CaptureParticles;
using photon_replay_detail::NearestCell;
using photon_replay_detail::OpenUnitInterval;
using photon_replay_detail::PhotonDrawKey;
using photon_replay_detail::SameBits;

namespace
{
amrex::MultiFab
HostCopy (amrex::MultiFab const& source)
{
    amrex::MultiFab host(source.boxArray(), source.DistributionMap(), source.nComp(),
        source.nGrowVect(), amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(host, source);
    amrex::Gpu::streamSynchronize();
    return host;
}

bool
SameSnapshot (photon_replay_detail::CapturedParticles const& a,
              photon_replay_detail::CapturedParticles const& b)
{
    if (a.tiles.size() != b.tiles.size()) { return false; }
    for (std::size_t n = 0; n < a.tiles.size(); ++n) {
        auto const& x = a.tiles[n];
        auto const& y = b.tiles[n];
        if (x.key != y.key || x.count != y.count || !SameBits(x.real_data, y.real_data) ||
            !SameBits(x.int_data, y.int_data) || !SameBits(x.idcpu, y.idcpu)) { return false; }
    }
    return true;
}

bool
LowBeta (amrex::GpuArray<amrex::Real, 3> const& beta)
{
    amrex::Real norm = 0;
    for (auto const value : beta) {
        if (!std::isfinite(value)) { return false; }
        norm += value * value;
    }
    return std::isfinite(norm) && norm <=
#if defined(WARPX_DIM_RZ)
        1.e-6_rt;
#else
        1.e-4_rt;
#endif
}

amrex::ParticleReal
RealAt (photon_replay_detail::CapturedTile const& tile, int component, long p)
{
    return tile.real_data[static_cast<std::size_t>(component * tile.count + p)];
}

bool
ValidCapturePhoton (amrex::ParticleReal weight, amrex::ParticleReal ux,
                    amrex::ParticleReal uy, amrex::ParticleReal uz,
                    PacketConversionFourMomentum const& account)
{
#if defined(WARPX_DIM_RZ)
    if (NeedsScaledNativePhotonMomentum(weight))
    {
        auto const scaled = ScaledNativePhotonMomentumFromPhoton(weight, ux, uy, uz);
        return scaled.valid && ValidFiniteMassPhoton(scaled.physical) &&
            ConsistentScaledNativePhotonAccounts(scaled);
    }
#else
    amrex::ignore_unused(weight, ux, uy, uz);
#endif
    return account.valid && ValidFiniteMassPhoton(account.value);
}

#if defined(WARPX_DIM_RZ)
bool
EndpointCell (amrex::Geometry const& geometry, amrex::Real radius, amrex::Real z,
              amrex::IntVect& cell)
{
    amrex::Real const coordinates[2]{radius, z};
    cell = geometry.Domain().smallEnd();
    for (int d = 0; d < 2; ++d)
    {
        if (!std::isfinite(coordinates[d]) || coordinates[d] < geometry.ProbLo(d) ||
            !(coordinates[d] < geometry.ProbHi(d)))
        {
            return false;
        }
        cell[d] += static_cast<int>(std::floor(
            (coordinates[d] - geometry.ProbLo(d)) / geometry.CellSize(d)));
    }
    return geometry.Domain().contains(cell);
}

amrex::Real
AnnularGaussRadius (RZMomentMetric const& metric, int radial_cell, int point)
{
    auto const lower = metric.Radius(radial_cell);
    auto const upper = metric.Radius(radial_cell + 1);
    auto const delta_s = (upper - lower) * (upper + lower);
    auto const middle_s = lower * lower + .5_rt * delta_s;
    auto const offset_s = .5_rt * delta_s / std::sqrt(3._rt);
    return std::sqrt(middle_s + (point == 0 ? -offset_s : offset_s));
}
#endif

FourVector
EndpointConjugate (FourVector const& cartesian, amrex::Real radius, amrex::Real angle,
                   amrex::Geometry const& geometry, amrex::IntVect const& cell,
                   bool absolute_operands = false)
{
#if defined(WARPX_DIM_RZ)
    auto const cosine = std::cos(angle), sine = std::sin(angle);
    auto const ratio = radius / RZMomentMetric(geometry).Angular(cell[0]).mean_radius;
    if (absolute_operands)
    {
        return {std::abs(cartesian[0]),
                std::abs(cartesian[1] * cosine) + std::abs(cartesian[2] * sine),
                ratio * (std::abs(cartesian[2] * cosine) + std::abs(cartesian[1] * sine)),
                std::abs(cartesian[3])};
    }
    return {cartesian[0], cartesian[1] * cosine + cartesian[2] * sine,
            ratio * (cartesian[2] * cosine - cartesian[1] * sine), cartesian[3]};
#else
    amrex::ignore_unused(radius, angle, geometry, cell);
    auto result = cartesian;
    if (absolute_operands)
    {
        for (auto& value : result) { value = std::abs(value); }
    }
    return result;
#endif
}

#if defined(WARPX_DIM_RZ)
amrex::GpuArray<amrex::Real, 3>
EndpointVelocity (amrex::GpuArray<amrex::Real, 3> const& conjugate,
                  amrex::Real radius, amrex::Real angle,
                  amrex::Geometry const& geometry, amrex::IntVect const& cell)
{
    auto const cosine = std::cos(angle), sine = std::sin(angle);
    auto const tangential = conjugate[1] * radius /
                            RZMomentMetric(geometry).Angular(cell[0]).mean_radius;
    return {conjugate[0] * cosine - tangential * sine,
            conjugate[0] * sine + tangential * cosine, conjugate[2]};
}
#endif

bool
ParticleCell (photon_replay_detail::CapturedTile const& tile, long p,
              amrex::Geometry const& geometry, amrex::IntVect& cell)
{
#if defined(WARPX_DIM_RZ)
    return std::isfinite(RealAt(tile, PIdx::theta, p)) &&
           EndpointCell(geometry, RealAt(tile, PIdx::r, p), RealAt(tile, PIdx::z, p), cell);
#else
    amrex::ParticleReal x = 0, y = 0, z = RealAt(tile, PIdx::z, p);
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
    x = RealAt(tile, PIdx::x, p);
#endif
#if defined(WARPX_DIM_3D)
    y = RealAt(tile, PIdx::y, p);
#endif
    return NearestCell(geometry, x, y, z, cell);
#endif
}

bool
SameControls (amrex::Geometry const& geometry, amrex::Real dt,
              NativePacketThermalEventOptions const& options)
{
    amrex::Long low[] = {static_cast<amrex::Long>(options.seed & UINT64_C(0xffffffff)),
        static_cast<amrex::Long>(options.seed >> 32U),
        static_cast<amrex::Long>(options.interval & UINT64_C(0xffffffff)),
        static_cast<amrex::Long>(options.interval >> 32U), options.emission_rays_per_cell,
        options.emission ? 1L : 0L, geometry.Coord(), options.survival_weighted_emission ? 1L : 0L,
        options.rz_angular_conjugate ? 1L : 0L,
        options.rz_reciprocal_corner_emission ? 1L : 0L,
        options.rz_reflecting_axial_frozen_bath ? 1L : 0L,
        options.rz_reflecting_axial_reciprocal_endpoint ? 1L : 0L};
    amrex::Long high[12];
    std::copy(std::begin(low), std::end(low), high);
    amrex::ParallelDescriptor::ReduceLongMin(low, 12);
    amrex::ParallelDescriptor::ReduceLongMax(high, 12);
    bool valid = std::equal(std::begin(low), std::end(low), high);
    amrex::Real minimum[4 + 2 * AMREX_SPACEDIM] = {
        dt, options.photon_energy, options.radiation_constant,
        options.max_absorption_optical_depth};
    int lower[3 * AMREX_SPACEDIM];
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        minimum[4 + 2 * d] = geometry.ProbLo(d);
        minimum[5 + 2 * d] = geometry.ProbHi(d);
        lower[3 * d] = geometry.Domain().smallEnd(d);
        lower[3 * d + 1] = geometry.Domain().bigEnd(d);
        lower[3 * d + 2] = geometry.isPeriodic(d) ? 1 : 0;
    }
    amrex::Real maximum[4 + 2 * AMREX_SPACEDIM];
    int upper[3 * AMREX_SPACEDIM];
    std::copy(std::begin(minimum), std::end(minimum), maximum);
    std::copy(std::begin(lower), std::end(lower), upper);
    amrex::ParallelDescriptor::ReduceRealMin(minimum, 4 + 2 * AMREX_SPACEDIM);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 4 + 2 * AMREX_SPACEDIM);
    amrex::ParallelDescriptor::ReduceIntMin(lower, 3 * AMREX_SPACEDIM);
    amrex::ParallelDescriptor::ReduceIntMax(upper, 3 * AMREX_SPACEDIM);
    return std::equal(std::begin(minimum), std::end(minimum), maximum) &&
           std::equal(std::begin(lower), std::end(lower), upper) && valid;
}

std::uint64_t
BirthKey (amrex::Geometry const& geometry, amrex::IntVect const& cell, int ray,
          NativePacketThermalEventOptions const& options)
{
    auto const site = static_cast<std::uint64_t>(geometry.Domain().index(cell));
    auto key = PhotonDrawKey(options.seed, site, options.interval,
        static_cast<std::uint64_t>(ray), UINT64_C(0x544845524d414c));
    return key == 0 ? UINT64_C(0x9e3779b97f4a7c15) : key;
}

amrex::GpuArray<amrex::Real, 3>
EmissionDirection (std::uint64_t key)
{
    constexpr amrex::Real pi = 3.141592653589793238462643383279502884_rt;
    auto const mu = 2 * OpenUnitInterval(PhotonDrawKey(key, 0, 0, 0, 0)) - 1;
    auto const phi = 2 * pi * OpenUnitInterval(PhotonDrawKey(key, 0, 0, 0, 1));
    auto const transverse = std::sqrt(1 - mu * mu);
    return {transverse * std::cos(phi), transverse * std::sin(phi), mu};
}
} // namespace

struct NativePacketThermalEventReplay::Impl
{
    bool prepared = false;
    amrex::Geometry geometry;
    amrex::Real dt = 0;
    NativePacketThermalEventOptions options;
    amrex::MultiFab coefficients; // host: kinetic mask, proper chi_a, initial rate velocity
    photon_replay_detail::CapturedParticles baseline;
    std::vector<std::string> real_names, int_names;
    std::array<int, 2> key_components{-1, -1};
};

NativePacketThermalEventReplay::NativePacketThermalEventReplay ()
    : m_impl(std::make_unique<Impl>()) {}
NativePacketThermalEventReplay::~NativePacketThermalEventReplay () = default;

bool
NativePacketThermalEventReplay::Prepare (
    WarpXParticleContainer::Base const& snapshot, WarpXParticleContainer const& species,
    amrex::iMultiFab const& mask, amrex::MultiFab const& opacity,
    amrex::MultiFab const& rate, amrex::Geometry const& geometry, amrex::Real dt,
    NativePacketThermalEventOptions const& options, std::string& reason,
    amrex::iMultiFab const* capture_mask)
{
    m_impl = std::make_unique<Impl>();
    reason = "invalid native gray thermal source inputs";
    bool valid = SameControls(geometry, dt, options);
    int capture_low = capture_mask ? 1 : 0, capture_high = capture_low;
    amrex::ParallelDescriptor::ReduceIntMin(capture_low);
    amrex::ParallelDescriptor::ReduceIntMax(capture_high);
    valid = valid && capture_low == capture_high;
    auto const& boxes = snapshot.ParticleBoxArray(0);
    auto const& mapping = snapshot.ParticleDistributionMap(0);
    valid = valid && species.AmIA<PhysicalSpecies::photon>() && species.finestLevel() == 0 &&
        snapshot.finestLevel() == 0 &&
        species.ParticleBoxArray(0) == boxes && species.ParticleDistributionMap(0) == mapping &&
        species.GetRealSoANames() == snapshot.GetRealSoANames() &&
        (species.GetIntSoANames() == snapshot.GetIntSoANames() ||
         (snapshot.NumIntComps() == species.NumIntComps() + 1 &&
          std::equal(species.GetIntSoANames().begin(), species.GetIntSoANames().end(),
                     snapshot.GetIntSoANames().begin()) &&
          snapshot.GetIntSoANames().back() == "__photon_birth_provisional")) &&
        mask.nComp() == 1 && opacity.nComp() == 1 && rate.nComp() == 3 &&
        mask.boxArray() == boxes && opacity.boxArray() == boxes && rate.boxArray() == boxes &&
        mask.DistributionMap() == mapping && opacity.DistributionMap() == mapping &&
        rate.DistributionMap() == mapping && std::isfinite(dt) && dt > 0 &&
        std::isfinite(options.photon_energy) && options.photon_energy > 0 &&
        std::isfinite(options.radiation_constant) && options.radiation_constant > 0 &&
        std::isfinite(options.max_absorption_optical_depth) &&
        options.max_absorption_optical_depth > 0 && options.emission_rays_per_cell > 0 &&
        options.emission_rays_per_cell <= 1024 &&
        std::numeric_limits<amrex::Real>::digits >= 53 &&
        std::numeric_limits<amrex::ParticleReal>::digits >= 53;
    if (capture_mask) {
        valid = valid && capture_mask->nComp() == 1 && capture_mask->boxArray() == boxes &&
            capture_mask->DistributionMap() == mapping;
    }
#if defined(WARPX_DIM_RZ)
    bool axial = geometry.isPeriodic(1);
#if !defined(AMREX_USE_GPU)
    axial = axial || (((options.rz_reflecting_axial_frozen_bath &&
                       !options.rz_reciprocal_corner_emission) ||
                      (options.rz_reflecting_axial_reciprocal_endpoint &&
                       options.rz_reciprocal_corner_emission)) &&
        WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
        WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
        WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting);
#else
    valid = valid && !options.rz_reflecting_axial_frozen_bath &&
                    !options.rz_reflecting_axial_reciprocal_endpoint;
#endif
    valid = valid && options.rz_angular_conjugate &&
        !(options.rz_reflecting_axial_frozen_bath &&
          options.rz_reflecting_axial_reciprocal_endpoint) &&
        (!options.rz_reflecting_axial_reciprocal_endpoint ||
         options.rz_reciprocal_corner_emission) &&
        (!options.rz_reflecting_axial_frozen_bath || !options.rz_reciprocal_corner_emission) &&
        (geometry.Coord() == 0 || geometry.Coord() == 1) &&
        WarpX::n_rz_azimuthal_modes == 1 && geometry.ProbLo(0) == 0 &&
        !geometry.isPeriodic(0) && axial &&
        WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
        WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting;
#elif !defined(WARPX_DIM_1D_Z) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_3D)
    valid = false;
#else
    valid = valid && !options.rz_angular_conjugate &&
        !options.rz_reciprocal_corner_emission &&
        !options.rz_reflecting_axial_frozen_bath &&
        !options.rz_reflecting_axial_reciprocal_endpoint && geometry.Coord() == 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { valid &= geometry.isPeriodic(d); }
#endif
    valid = FindPhotonDrawKeyComponents(snapshot.GetRealSoANames(), m_impl->key_components) &&
        (!options.emission || m_impl->key_components[0] >= 0) && valid;
    if (options.rz_reciprocal_corner_emission)
    {
        valid = valid && WarpX::GetInstance().finestLevel() == 0 &&
                WarpX::GetInstance().getdo_moving_window() == 0;
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return false; }
    auto& s = *m_impl;
    s.coefficients.define(boxes, mapping, 6, 0,
        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    auto const chi = HostCopy(opacity), velocity = HostCopy(rate);
    for (amrex::MFIter it(s.coefficients); it.isValid(); ++it) {
        auto const source = chi.const_array(it), v = velocity.const_array(it);
        auto const target = s.coefficients.array(it);
        std::vector<int> flags(static_cast<std::size_t>(mask[it].box().numPts()));
        auto const* src = mask[it].dataPtr();
        // Mask may have ghosts; copy the complete native fab then use its own
        // box indexing, never assume valid cells are a contiguous subarray.
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, src, src + flags.size(), flags.begin());
        std::vector<int> capture_flags;
        if (capture_mask) {
            capture_flags.resize(static_cast<std::size_t>((*capture_mask)[it].box().numPts()));
            auto const* values = (*capture_mask)[it].dataPtr();
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values, values + capture_flags.size(),
                             capture_flags.begin());
        }
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell) {
            auto const index = cell();
            auto const flag = flags[static_cast<std::size_t>(mask[it].box().index(index))];
            auto const capture = capture_mask ? capture_flags[static_cast<std::size_t>(
                (*capture_mask)[it].box().index(index))] : flag;
            amrex::GpuArray<amrex::Real, 3> beta{};
            for (int d = 0; d < 3; ++d) { beta[d] = v(index, d) / PhysConst::c; }
#if defined(WARPX_DIM_RZ)
            beta[1] *= RZMomentMetric(geometry).Radius(index[0] + 1) /
                       RZMomentMetric(geometry).Angular(index[0]).mean_radius;
#endif
            valid = valid && (flag == 0 || flag == 1) && (capture == 0 || capture == 1) && LowBeta(beta) &&
                std::isfinite(source(index)) && source(index) >= 0;
            target(index, 0) = flag;
            target(index, 1) = source(index);
            target(index, 5) = capture;
            for (int d = 0; d < 3; ++d) { target(index, d + 2) = v(index, d); }
        }
    }
    s.geometry = geometry;
    s.dt = dt;
    s.options = options;
    s.baseline = CaptureParticles(snapshot);
    s.real_names = snapshot.GetRealSoANames();
    s.int_names = snapshot.GetIntSoANames();
    for (auto const& tile : s.baseline.tiles) {
        if (tile.count == 0) { continue; }
        auto const coefficients = s.coefficients.const_array(tile.key.first);
        for (long p = 0; p < tile.count; ++p) {
            amrex::IntVect cell;
            auto const photon = PacketConversionFourMomentumFromPhoton(RealAt(tile, PIdx::w, p),
                RealAt(tile, PIdx::ux, p), RealAt(tile, PIdx::uy, p), RealAt(tile, PIdx::uz, p));
            valid = valid && ValidCapturePhoton(RealAt(tile, PIdx::w, p),
                RealAt(tile, PIdx::ux, p), RealAt(tile, PIdx::uy, p),
                RealAt(tile, PIdx::uz, p), photon) &&
                ParticleCell(tile, p, geometry, cell) && coefficients.contains(cell);
            if (!valid) { break; }
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        reason = "invalid frozen source photons, opacity, mask or rate frame";
        return false;
    }
    s.prepared = true;
    reason.clear();
    return true;
}

NativePacketThermalEventResult
NativePacketThermalEventReplay::Replay (amrex::MultiFab const& temperature,
                                        amrex::MultiFab const& velocity,
                                        PhotonPacketTransaction& transaction,
                                        NativeThermalCornerEmission const* reciprocal) const
{
    NativePacketThermalEventResult rejected;
    rejected.reason = "unprepared native gray thermal replay";
    auto const& s = *m_impl;
    bool valid = s.prepared;
    bool const rz_reciprocal = s.options.rz_reciprocal_corner_emission;
    valid = valid && (!s.options.rz_angular_conjugate ||
                      rz_reciprocal == (reciprocal != nullptr));
    int mode_low = reciprocal ? 1 : 0, mode_high = mode_low;
    amrex::ParallelDescriptor::ReduceIntMin(mode_low);
    amrex::ParallelDescriptor::ReduceIntMax(mode_high);
    valid = valid && mode_low == mode_high;
    int ray_low = reciprocal ? reciprocal->rays_per_corner : s.options.emission_rays_per_cell;
    int ray_high = ray_low;
    amrex::ParallelDescriptor::ReduceIntMin(ray_low);
    amrex::ParallelDescriptor::ReduceIntMax(ray_high);
    valid = valid && ray_low == ray_high && ray_low > 0 && ray_low <= 1024;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { return rejected; }
    auto& photons = transaction.Particles();
    valid = photons.ParticleBoxArray(0) == s.coefficients.boxArray() &&
        photons.ParticleDistributionMap(0) == s.coefficients.DistributionMap() &&
        photons.GetRealSoANames() == s.real_names && photons.GetIntSoANames() == s.int_names &&
        temperature.nComp() == 1 && velocity.nComp() == 3 &&
        temperature.boxArray() == s.coefficients.boxArray() &&
        velocity.boxArray() == s.coefficients.boxArray() &&
        temperature.DistributionMap() == s.coefficients.DistributionMap() &&
        velocity.DistributionMap() == s.coefficients.DistributionMap();
    if (reciprocal)
    {
        valid = valid && reciprocal->weights && reciprocal->nodal_temperature;
        if (valid)
        {
            valid = reciprocal->weights->nComp() == (1 << AMREX_SPACEDIM) &&
                    reciprocal->weights->boxArray() == s.coefficients.boxArray() &&
                    reciprocal->weights->DistributionMap() == s.coefficients.DistributionMap() &&
                    reciprocal->nodal_temperature->nComp() == 1 &&
                    reciprocal->nodal_temperature->boxArray() ==
                        amrex::convert(s.coefficients.boxArray(), amrex::IntVect(1)) &&
                    reciprocal->nodal_temperature->DistributionMap() ==
                        s.coefficients.DistributionMap();
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    rejected.reason = "invalid native gray thermal replay layout";
    if (!valid) { return rejected; }
    valid = SameSnapshot(CaptureParticles(photons), s.baseline);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    rejected.reason = "private photons changed from frozen thermal snapshot";
    if (!valid) { return rejected; }
    auto const te = HostCopy(temperature), frame = HostCopy(velocity);
    amrex::MultiFab corner_weights, node_temperature, corner_debit;
    if (reciprocal)
    {
        corner_weights = HostCopy(*reciprocal->weights);
        node_temperature = HostCopy(*reciprocal->nodal_temperature);
        corner_debit.define(s.coefficients.boxArray(), s.coefficients.DistributionMap(),
                            4 * (1 << AMREX_SPACEDIM), 0,
                            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        corner_debit.setVal(0);
        amrex::Gpu::streamSynchronize();
    }
    amrex::MultiFab debit(s.coefficients.boxArray(), s.coefficients.DistributionMap(), 4, 0,
        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::MultiFab scale(s.coefficients.boxArray(), s.coefficients.DistributionMap(), 4, 0,
        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    debit.setVal(0); scale.setVal(0);
    std::vector<amrex::Vector<amrex::ParticleReal>> weights;
    weights.reserve(s.baseline.tiles.size());
    NativePacketThermalEventResult result;
    auto observe_refusal = [&] (bool before, char const* stage)
    {
        auto* trace = s.options.refusal_trace;
        if (trace == nullptr || trace->captured || !before || valid) { return false; }
        trace->captured = true;
        trace->stage = stage;
        trace->dt = s.dt;
        trace->depth_limit = s.options.max_absorption_optical_depth;
        return true;
    };
    auto observe_particle = [&] (photon_replay_detail::CapturedTile const& tile, long p)
    {
        auto& trace = *s.options.refusal_trace;
        trace.record = p;
        trace.idcpu = tile.idcpu[p];
        trace.old_weight = RealAt(tile, PIdx::w, p);
        trace.momentum_u = {RealAt(tile, PIdx::ux, p), RealAt(tile, PIdx::uy, p),
                            RealAt(tile, PIdx::uz, p)};
#if defined(WARPX_DIM_RZ)
        trace.cylindrical_coordinates = true;
        trace.coordinates = {RealAt(tile, PIdx::r, p), RealAt(tile, PIdx::z, p),
                             RealAt(tile, PIdx::theta, p)};
#else
        trace.coordinates[2] = RealAt(tile, PIdx::z, p);
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
        trace.coordinates[0] = RealAt(tile, PIdx::x, p);
#endif
#if defined(WARPX_DIM_3D)
        trace.coordinates[1] = RealAt(tile, PIdx::y, p);
#endif
#endif
    };
    std::set<std::uint64_t> existing_keys;
    for (auto const& tile : s.baseline.tiles) {
        weights.emplace_back(static_cast<std::size_t>(tile.count));
        if (tile.count == 0) { continue; }
        auto const coefficients = s.coefficients.const_array(tile.key.first);
        auto const d = debit.array(tile.key.first), bound = scale.array(tile.key.first);
        for (long p = 0; p < tile.count; ++p) {
            auto const old_weight = RealAt(tile, PIdx::w, p);
            auto& next_weight = weights.back()[static_cast<std::size_t>(p)];
            next_weight = old_weight;
            if (s.key_components[0] >= 0) {
                std::uint64_t key = 0;
                auto const low = RealAt(tile, s.key_components[0], p);
                auto const high = RealAt(tile, s.key_components[1], p);
                auto const before_key = valid;
                valid = DecodePhotonDrawKey(low, high, tile.idcpu[p], key) && valid;
                if (low != 0 || high != 0)
                {
                    auto const inserted = existing_keys.insert(key).second;
                    valid = valid && (!rz_reciprocal || inserted);
                }
                if (observe_refusal(before_key, "existing_key_decode_or_duplicate"))
                { observe_particle(tile, p); s.options.refusal_trace->logical_key = key; }
            }
            amrex::IntVect cell;
            if (!ParticleCell(tile, p, s.geometry, cell))
            {
                auto const before_cell = valid;
                valid = false;
                if (observe_refusal(before_cell, "particle_cell")) { observe_particle(tile, p); }
                continue;
            }
            if (coefficients(cell, 5) == 0 || coefficients(cell, 1) == 0) { continue; }
            auto const incoming = PacketConversionFourMomentumFromPhoton(old_weight,
                RealAt(tile, PIdx::ux, p), RealAt(tile, PIdx::uy, p), RealAt(tile, PIdx::uz, p));
            amrex::Real beta2 = 0, projection = 0;
            amrex::GpuArray<amrex::Real, 3> rate_beta{};
            for (int q = 0; q < 3; ++q)
            {
                rate_beta[q] = coefficients(cell, q + 2) / PhysConst::c;
            }
            amrex::Real radius = 0, angle = 0;
#if defined(WARPX_DIM_RZ)
            radius = RealAt(tile, PIdx::r, p);
            angle = RealAt(tile, PIdx::theta, p);
            rate_beta = EndpointVelocity(rate_beta, radius, angle, s.geometry, cell);
#endif
            for (int q = 0; q < 3; ++q) {
                auto const beta = rate_beta[q];
                beta2 += beta * beta;
#if defined(WARPX_DIM_RZ)
                if (NeedsScaledNativePhotonMomentum(old_weight))
                {
                    auto const scaled = ScaledNativePhotonMomentumFromPhoton(old_weight,
                        RealAt(tile, PIdx::ux, p), RealAt(tile, PIdx::uy, p),
                        RealAt(tile, PIdx::uz, p));
                    projection += beta * scaled.physical[q + 1] / scaled.physical[0];
                }
                else
#endif
                {
                    projection += beta * incoming.value[q + 1] / incoming.value[0];
                }
            }
            auto const depth = PhysConst::c * s.dt * coefficients(cell, 1) *
                (1 - projection) / std::sqrt(1 - beta2);
            auto const before_depth = valid;
            valid = valid && std::isfinite(depth) && depth >= 0 &&
                depth <= s.options.max_absorption_optical_depth;
            if (observe_refusal(before_depth, "attenuation_depth"))
            {
                observe_particle(tile, p);
                auto& trace = *s.options.refusal_trace;
                trace.cell = cell; trace.alpha = coefficients(cell, 1);
                trace.rate_beta = {rate_beta[0], rate_beta[1], rate_beta[2]};
                trace.depth = depth; trace.incoming = incoming.value;
                trace.incoming_valid = incoming.valid;
            }
            if (!valid) { continue; }
            next_weight = static_cast<amrex::ParticleReal>(old_weight * std::exp(-depth));
            auto const outgoing = PacketConversionFourMomentumFromPhoton(next_weight,
                RealAt(tile, PIdx::ux, p), RealAt(tile, PIdx::uy, p), RealAt(tile, PIdx::uz, p));
            auto const before_outgoing = valid;
            bool const exact_zero_endpoint = s.options.rz_angular_conjugate && next_weight == 0;
            valid = valid && (exact_zero_endpoint || ValidCapturePhoton(next_weight,
                RealAt(tile, PIdx::ux, p), RealAt(tile, PIdx::uy, p),
                RealAt(tile, PIdx::uz, p), outgoing)) &&
                next_weight <= old_weight;
            if (observe_refusal(before_outgoing, "attenuation_outgoing_momentum_or_weight"))
            {
                observe_particle(tile, p);
                auto& trace = *s.options.refusal_trace;
                trace.cell = cell; trace.alpha = coefficients(cell, 1);
                trace.rate_beta = {rate_beta[0], rate_beta[1], rate_beta[2]};
                trace.next_weight = next_weight; trace.depth = depth;
                trace.incoming = incoming.value; trace.outgoing = outgoing.value;
                trace.incoming_valid = incoming.valid; trace.outgoing_valid = outgoing.valid;
            }
            FourVector difference{};
            for (int q = 0; q < 4; ++q)
            {
                difference[q] = incoming.value[q] - outgoing.value[q];
            }
            auto const transfer = EndpointConjugate(difference, radius, angle, s.geometry, cell);
            auto const before_operands =
                EndpointConjugate(incoming.value, radius, angle, s.geometry, cell, true);
            auto const after_operands =
                EndpointConjugate(outgoing.value, radius, angle, s.geometry, cell, true);
            for (int q = 0; q < 4; ++q) {
                d(cell, q) += transfer[q];
                bound(cell, q) += before_operands[q] + after_operands[q];
            }
            result.absorbed_energy += incoming.value[0] - outgoing.value[0];
            result.attenuated_packets += next_weight != old_weight ? 1 : 0;
        }
    }
    detail::EmissionPlan plan;
    std::set<std::uint64_t> proposed_keys;
    std::map<std::uint64_t, int> proposed_origins;
    std::map<std::uint64_t, amrex::Real> proposed_radii;
    std::map<std::uint64_t, amrex::IntVect> proposed_cells;
    auto const dx = s.geometry.CellSizeArray();
    [[maybe_unused]] auto const volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    for (amrex::MFIter it(s.coefficients); it.isValid(); ++it) {
        auto const coefficients = s.coefficients.const_array(it), t = te.const_array(it);
        auto const v = frame.const_array(it);
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell) {
            auto const index = cell();
            amrex::GpuArray<amrex::Real, 3> beta{};
            for (int q = 0; q < 3; ++q) { beta[q] = v(index, q) / PhysConst::c; }
            auto bound_beta = beta;
#if defined(WARPX_DIM_RZ)
            auto const metric = RZMomentMetric(s.geometry);
            auto const cell_volume = metric.Volume(index[0]);
            bound_beta[1] *= metric.Radius(index[0] + 1) /
                             metric.Angular(index[0]).mean_radius;
#else
            auto const cell_volume = volume;
#endif
            auto const before_frame = valid;
            valid = valid && LowBeta(bound_beta) && std::isfinite(t(index)) && t(index) >= 0;
            if (observe_refusal(before_frame, "emission_work_frame_or_temperature"))
            {
                auto& trace = *s.options.refusal_trace;
                trace.cell = index; trace.temperature = t(index);
                trace.work_beta = {bound_beta[0], bound_beta[1], bound_beta[2]};
            }
            if (!valid || !s.options.emission || coefficients(index, 0) == 0 ||
                coefficients(index, 1) == 0 || t(index) == 0) { continue; }
            int const origins = reciprocal ? (1 << AMREX_SPACEDIM) : 1;
            amrex::Real incidence_sum = 0;
            for (int origin = 0; origin < origins; ++origin)
            {
                auto node = index;
                for (int axis = 0; axis < AMREX_SPACEDIM; ++axis)
                {
                    node[axis] += (origin >> axis) & 1;
                }
                auto const incidence = reciprocal ? corner_weights[it](index, origin) : 1;
                auto const source_t = reciprocal ? node_temperature[it](node) : t(index);
                auto const before_corner = valid;
                valid =
                    valid && std::isfinite(incidence + source_t) && incidence >= 0 && source_t >= 0;
                if (observe_refusal(before_corner, "emission_corner_incidence_or_temperature"))
                {
                    auto& trace = *s.options.refusal_trace;
                    trace.cell = index; trace.temperature = source_t; trace.incidence = incidence;
                }
                incidence_sum += incidence;
                if (incidence == 0 || source_t == 0)
                {
                    continue;
                }
                auto const t2 = source_t * source_t;
                auto const energy = PhysConst::c * s.dt * cell_volume * coefficients(index, 1) *
                                    s.options.radiation_constant * t2 * t2 * incidence;
                int const radial_points = rz_reciprocal ? 2 : 1;
                auto const weight = static_cast<amrex::ParticleReal>(
                    energy / (radial_points * ray_low) / s.options.photon_energy);
                auto const before_energy = valid;
                valid = valid && std::isfinite(energy) && energy > 0 && std::isfinite(weight) &&
                        weight > 0;
                if (observe_refusal(before_energy, "emission_energy_or_weight"))
                {
                    auto& trace = *s.options.refusal_trace;
                    trace.cell = index; trace.temperature = source_t; trace.incidence = incidence;
                    trace.source_energy = energy; trace.next_weight = weight;
                }
                if (!valid)
                {
                    continue;
                }
                amrex::ParticleReal x = 0, y = 0, z = 0;
                auto center = [&] (int axis)
                {
                    return static_cast<amrex::ParticleReal>(
                        s.geometry.ProbLo(axis) +
                        (index[axis] - s.geometry.Domain().smallEnd(axis) + 0.5_rt) * dx[axis]);
                };
#if defined(WARPX_DIM_1D_Z)
                z = center(0);
#elif defined(WARPX_DIM_XZ)
                x = center(0);
                z = center(1);
#elif defined(WARPX_DIM_3D)
                x = center(0);
                y = center(1);
                z = center(2);
#elif defined(WARPX_DIM_RZ)
                x = static_cast<amrex::ParticleReal>(metric.Angular(index[0]).mean_radius);
                z = center(1);
#endif
                for (int radial_point = 0; radial_point < radial_points; ++radial_point)
                {
                    auto birth_x = x;
                    auto birth_beta = beta;
#if defined(WARPX_DIM_RZ)
                    if (rz_reciprocal)
                    {
                        // W owns the corner's caloric source, not a radial
                        // sub-volume. Every corner samples the full annulus;
                        // otherwise native axis FV weights bias the swirl
                        // lever even for a spatially uniform LTE bath.
                        birth_x = static_cast<amrex::ParticleReal>(
                            AnnularGaussRadius(metric, index[0], radial_point));
                        birth_beta = EndpointVelocity(
                            beta, birth_x, 0, s.geometry, index);
                        auto const before_radius = valid;
                        valid = valid && birth_x > metric.Radius(index[0]) &&
                                birth_x < metric.Radius(index[0] + 1);
                        if (observe_refusal(before_radius, "emission_annular_radius"))
                        {
                            auto& trace = *s.options.refusal_trace;
                            trace.cell = index; trace.cylindrical_coordinates = true;
                            trace.coordinates = {birth_x, z, 0};
                        }
                    }
#endif
                    amrex::IntVect represented_cell;
                    auto const before_birth_cell = valid;
#if defined(WARPX_DIM_RZ)
                    valid = EndpointCell(s.geometry, std::hypot(birth_x, y), z,
                                         represented_cell) &&
                            represented_cell == index && valid;
#else
                    valid = NearestCell(s.geometry, birth_x, y, z, represented_cell) &&
                            represented_cell == index && valid;
#endif
                    if (observe_refusal(before_birth_cell, "emission_represented_cell"))
                    {
                        auto& trace = *s.options.refusal_trace;
                        trace.cell = index;
#if defined(WARPX_DIM_RZ)
                        trace.cylindrical_coordinates = true;
                        trace.coordinates = {birth_x, z, 0};
#else
                        trace.coordinates = {birth_x, y, z};
#endif
                    }
                    for (int ray = 0; ray < ray_low; ++ray)
                    {
                        auto const base_key = BirthKey(s.geometry, index, ray, s.options);
                        auto const origin_key =
                            PhotonDrawKey(base_key, static_cast<std::uint64_t>(origin), 0, 0,
                                          UINT64_C(0x4e4f44414c424154));
                        auto const radial_key =
                            PhotonDrawKey(base_key, static_cast<std::uint64_t>(origin),
                                          static_cast<std::uint64_t>(radial_point), 0,
                                          UINT64_C(0x525a524543495052));
                        auto const selected = rz_reciprocal ? radial_key : origin_key;
                        auto const key = reciprocal
                                             ? (selected == 0 ? UINT64_C(0x9e3779b97f4a7c15)
                                                              : selected)
                                             : base_key;
                        auto const before_birth_key = valid;
                        valid = valid && proposed_keys.insert(key).second &&
                                !existing_keys.contains(key);
                        if (observe_refusal(before_birth_key, "emission_birth_key"))
                        { s.options.refusal_trace->cell = index; s.options.refusal_trace->logical_key = key; }
                        if (reciprocal)
                        {
                            proposed_origins.emplace(key, origin);
                        }
                        if (rz_reciprocal)
                        {
                            proposed_radii.emplace(key, birth_x);
                            proposed_cells.emplace(key, index);
                        }
                        auto const direction = EmissionDirection(key);
                        FourVector comoving{s.options.photon_energy,
                                            s.options.photon_energy * direction[0],
                                            s.options.photon_energy * direction[1],
                                            s.options.photon_energy * direction[2]};
                        auto const laboratory = BoostFourVector(comoving, birth_beta);
                        auto const before_boost = valid;
                        valid = ValidFiniteMassPhoton(laboratory) && valid;
                        if (observe_refusal(before_boost, "emission_boost_momentum"))
                        {
                            auto& trace = *s.options.refusal_trace;
                            trace.cell = index; trace.logical_key = key; trace.outgoing = laboratory;
                            trace.work_beta = {birth_beta[0], birth_beta[1], birth_beta[2]};
                        }
                        plan.x.push_back(birth_x);
                        plan.y.push_back(y);
                        plan.z.push_back(z);
                        auto const ux = static_cast<amrex::ParticleReal>(
                            laboratory[1] / (PhysConst::m_e * PhysConst::c));
                        auto const uy = static_cast<amrex::ParticleReal>(
                            laboratory[2] / (PhysConst::m_e * PhysConst::c));
                        auto const uz = static_cast<amrex::ParticleReal>(
                            laboratory[3] / (PhysConst::m_e * PhysConst::c));
                        amrex::Real survival = 1;
                        if (s.options.survival_weighted_emission)
                        {
                            auto const ray_momentum =
                                PacketConversionFourMomentumFromPhoton(1, ux, uy, uz);
                            amrex::GpuArray<amrex::Real, 3> rate_beta{};
                            for (int q = 0; q < 3; ++q)
                            {
                                rate_beta[q] = coefficients(index, q + 2) / PhysConst::c;
                            }
#if defined(WARPX_DIM_RZ)
                            if (rz_reciprocal)
                            {
                                rate_beta = EndpointVelocity(
                                    rate_beta, birth_x, 0, s.geometry, index);
                            }
#endif
                            amrex::Real beta2 = 0, projection = 0;
                            for (int q = 0; q < 3; ++q)
                            {
                                beta2 += rate_beta[q] * rate_beta[q];
                                projection += rate_beta[q] * ray_momentum.value[q + 1] /
                                              ray_momentum.value[0];
                            }
                            auto const depth = PhysConst::c * s.dt * coefficients(index, 1) *
                                               (1 - projection) / std::sqrt(1 - beta2);
                            auto const before_survival = valid;
                            valid = valid && ray_momentum.valid && std::isfinite(depth) &&
                                    depth >= 0 &&
                                    depth <= s.options.max_absorption_optical_depth;
                            if (observe_refusal(before_survival, "emission_survival_depth_or_momentum"))
                            {
                                auto& trace = *s.options.refusal_trace;
                                trace.cell = index; trace.logical_key = key;
                                trace.alpha = coefficients(index, 1); trace.depth = depth;
                                trace.incoming = ray_momentum.value; trace.incoming_valid = ray_momentum.valid;
                                trace.rate_beta = {rate_beta[0], rate_beta[1], rate_beta[2]};
                            }
                            survival = depth > 0 ? -std::expm1(-depth) / depth : 1;
                        }
                        plan.weight.push_back(
                            static_cast<amrex::ParticleReal>(weight * survival));
                        plan.ux.push_back(ux);
                        plan.uy.push_back(uy);
                        plan.uz.push_back(uz);
                        plan.logical_draw_keys.push_back(key);
                    }
                }
            }
            auto const before_incidence = valid;
            valid = valid && (incidence_sum == 1 || incidence_sum == 0);
            if (observe_refusal(before_incidence, "emission_incidence_sum"))
            { s.options.refusal_trace->cell = index; s.options.refusal_trace->incidence = incidence_sum; }
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    rejected.reason = "invalid gray capture depth, temperature, work frame or birth key";
    if (!valid)
    {
        return rejected;
    }
    for (std::size_t n = 0; n < s.baseline.tiles.size(); ++n)
    {
        auto const& before = s.baseline.tiles[n];
        auto& tile = photons.GetParticles(0).at(before.key);
        bool const retired = std::any_of(weights[n].begin(), weights[n].end(),
                                        [] (auto weight) { return weight == 0; });
        if (!retired)
        {
            auto& values = tile.GetStructOfArrays().GetRealData(PIdx::w);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, weights[n].begin(), weights[n].end(),
                             values.begin());
            continue;
        }
        // Only an exactly zero stored attenuation endpoint is retired, after
        // its full incoming->zero represented debit above. Compact this caller
        // private tile, preserving every survivor's pose/u/key/census/ID byte.
        // Existing history reconciliation owns dropping absorbed clocks.
        std::vector<long> keep;
        for (long p = 0; p < before.count; ++p)
        {
            if (weights[n][static_cast<std::size_t>(p)] != 0) { keep.push_back(p); }
        }
        tile.resize(keep.size());
        auto& soa = tile.GetStructOfArrays();
        for (int d = 0; d < photons.NumRealComps(); ++d)
        {
            amrex::Vector<amrex::ParticleReal> values;
            for (auto p : keep)
            {
                values.push_back(d == PIdx::w ? weights[n][static_cast<std::size_t>(p)] :
                    before.real_data[d * before.count + p]);
            }
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, values.begin(), values.end(),
                             soa.GetRealData(d).begin());
        }
        for (int d = 0; d < photons.NumIntComps(); ++d)
        {
            amrex::Vector<int> values;
            for (auto p : keep) { values.push_back(before.int_data[d * before.count + p]); }
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, values.begin(), values.end(),
                             soa.GetIntData(d).begin());
        }
        amrex::Vector<std::uint64_t> ids;
        for (auto p : keep) { ids.push_back(before.idcpu[p]); }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, ids.begin(), ids.end(), soa.GetIdCPUData().begin());
    }
    amrex::Gpu::streamSynchronize();
    if (s.options.emission && !detail::StageEmissionPlan(transaction, plan))
    {
        rejected.reason = "native thermal birth staging failed; discard caller private trial";
        return rejected;
    }
    // Remeasure only identified newborns. Unchanged photons may carry an
    // arbitrarily large background; never infer emission from bulk differences.
    auto const stored = CaptureParticles(photons);
    std::set<std::uint64_t> found;
    for (auto const& tile : stored.tiles) {
        if (!s.options.emission) { break; }
        if (tile.count == 0) { continue; }
        auto const d = debit.array(tile.key.first), bound = scale.array(tile.key.first);
        for (long p = 0; p < tile.count; ++p) {
            std::uint64_t key = 0;
            valid = DecodePhotonDrawKey(RealAt(tile, s.key_components[0], p),
                RealAt(tile, s.key_components[1], p), tile.idcpu[p], key) && valid;
            if (!proposed_keys.contains(key)) { continue; }
            auto const outgoing = PacketConversionFourMomentumFromPhoton(RealAt(tile, PIdx::w, p),
                RealAt(tile, PIdx::ux, p), RealAt(tile, PIdx::uy, p), RealAt(tile, PIdx::uz, p));
            amrex::IntVect cell;
            valid = valid && found.insert(key).second && outgoing.valid &&
                ValidFiniteMassPhoton(outgoing.value) && ParticleCell(tile, p, s.geometry, cell) &&
                d.contains(cell);
            if (!valid) { continue; }
            amrex::Real radius = 0, angle = 0;
#if defined(WARPX_DIM_RZ)
            radius = RealAt(tile, PIdx::r, p);
            angle = RealAt(tile, PIdx::theta, p);
#endif
            if (rz_reciprocal)
            {
                auto const expected_radius = proposed_radii.find(key);
                auto const expected_cell = proposed_cells.find(key);
                valid = valid && expected_radius != proposed_radii.end() &&
                        expected_cell != proposed_cells.end();
                if (valid)
                {
                    auto const radius_scale = std::max({1._rt, std::abs(radius),
                                                        std::abs(expected_radius->second)});
                    valid = std::abs(radius - expected_radius->second) <=
                                8 * std::numeric_limits<amrex::Real>::epsilon() * radius_scale &&
                            cell == expected_cell->second && angle == 0;
                }
                if (!valid) { continue; }
            }
            auto const conjugate =
                EndpointConjugate(outgoing.value, radius, angle, s.geometry, cell);
            auto const operands =
                EndpointConjugate(outgoing.value, radius, angle, s.geometry, cell, true);
            for (int q = 0; q < 4; ++q) {
                d(cell, q) -= conjugate[q];
                bound(cell, q) += operands[q];
                if (reciprocal)
                {
                    corner_debit[tile.key.first](cell, 4 * proposed_origins.at(key) + q) -=
                        rz_reciprocal ? conjugate[q] : outgoing.value[q];
                }
            }
            result.emitted_energy += outgoing.value[0];
        }
    }
    valid = valid && found.size() == proposed_keys.size();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    rejected.reason = "native thermal birth remeasurement failed; discard caller private trial";
    if (!valid) { return rejected; }
    result.emission_rays = static_cast<amrex::Long>(found.size());
    amrex::ParallelDescriptor::ReduceLongSum(result.attenuated_packets);
    amrex::ParallelDescriptor::ReduceLongSum(result.emission_rays);
    amrex::ParallelDescriptor::ReduceRealSum(result.absorbed_energy);
    amrex::ParallelDescriptor::ReduceRealSum(result.emitted_energy);
    bool const finite_debit = debit.is_finite();
    bool const finite_scale = scale.is_finite();
    valid = finite_debit && finite_scale && (!reciprocal || corner_debit.is_finite()) &&
            std::isfinite(result.absorbed_energy) &&
            std::isfinite(result.emitted_energy);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    rejected.reason = "nonfinite thermal source accounting; discard caller private trial";
    if (!valid) { return rejected; }
    result.radiation_debit.define(debit.boxArray(), debit.DistributionMap(), 4, 0);
    result.operand_scale.define(scale.boxArray(), scale.DistributionMap(), 4, 0);
    amrex::htod_memcpy(result.radiation_debit, debit);
    amrex::htod_memcpy(result.operand_scale, scale);
    if (reciprocal)
    {
        result.corner_emission_debit.define(corner_debit.boxArray(), corner_debit.DistributionMap(),
                                            corner_debit.nComp(), 0);
        amrex::htod_memcpy(result.corner_emission_debit, corner_debit);
    }
    amrex::Gpu::streamSynchronize();
    result.emission_birth_keys = std::move(plan.logical_draw_keys);
    result.valid = true;
    return result;
}
} // namespace warpx::radiation
