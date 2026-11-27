/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "AdaptivePacketDiffusionTransition.H"

#include "FluxRelaxedPackets.H"
#include "PacketConversionSampling.H"
#include "PacketMomentReconstruction.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "PhotonPacketTransaction.H"
#include "NativePhotonReplayInternals.H"
#include "RZAngularGeometry.H"
#include "Utils/WarpXConst.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <vector>

namespace warpx::radiation
{
namespace
{
bool
All (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

bool
Physical (FourVector const& value)
{
    for (auto const component : value) {
        if (!std::isfinite(component)) {
            return false;
        }
    }
    if (value[0] < 0) {
        return false;
    }
    auto const norm = std::hypot(std::hypot(value[1], value[2]), value[3]);
    return value[0] == 0 ? norm == 0 : norm <= value[0];
}

void
Clone (amrex::MultiFab& target, amrex::MultiFab const& source)
{
    target.define(source.boxArray(), source.DistributionMap(), source.nComp(), source.nGrowVect());
    amrex::MultiFab::Copy(target, source, 0, 0, source.nComp(), source.nGrowVect());
}

amrex::Real
AngularSum (amrex::MultiFab const& field, int component, amrex::Geometry const& geometry)
{
#if defined(WARPX_DIM_RZ)
    auto const lower = geometry.ProbLo(0), dx = geometry.CellSize(0);
    int const domain_lo = geometry.Domain().smallEnd(0);
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real> data(ops);
    for (amrex::MFIter it(field); it.isValid(); ++it) {
        auto const value = field.const_array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real> {
                     auto const lo = lower + (i - domain_lo) * dx;
                     return {value(i, j, k, component) *
                             RZAngularGeometry::MeanRadius(lo, lo + dx)};
                 });
    }
    auto result = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
#else
    amrex::ignore_unused(geometry);
    return field.sum(component);
#endif
}

FourVector
Fields (amrex::MultiFab const& energy, amrex::MultiFab const& momentum,
        amrex::Geometry const& geometry)
{
    return {energy.sum(0),
#if defined(WARPX_DIM_RZ)
            0,
#else
            momentum.sum(0),
#endif
            AngularSum(momentum, 1, geometry), momentum.sum(2)};
}

FourVector
Carry (amrex::MultiFab const& carry, amrex::Geometry const& geometry)
{
    return {carry.sum(0),
#if defined(WARPX_DIM_RZ)
            0,
#else
            carry.sum(1),
#endif
            AngularSum(carry, 2, geometry), carry.sum(3)};
}

bool
Packets (WarpXParticleContainer::Base& photons, FourVector& inventory)
{
    if (!detail::PhotonInventory(photons, inventory)) {
        return false;
    }
#if defined(WARPX_DIM_RZ)
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real> data(ops);
    for (WarpXParIter it(photons, 0); it.isValid(); ++it) {
        auto const p = it.GetParticleTile().getParticleTileData();
        auto const position = GetParticlePosition<PIdx>(it);
        ops.eval(it.numParticles(), data,
                 [=] AMREX_GPU_DEVICE(long n) -> amrex::GpuTuple<amrex::Real> {
                     if (!amrex::ParticleIDWrapper{p.m_idcpu[n]}.is_valid() ||
                         !(p.m_rdata[PIdx::w][n] > 0)) {
                         return {0};
                     }
                     amrex::ParticleReal x = 0, y = 0, z = 0;
                     position(n, x, y, z);
                     return {p.m_rdata[PIdx::w][n] * PhysConst::m_e * PhysConst::c *
                             (x * p.m_rdata[PIdx::uy][n] - y * p.m_rdata[PIdx::ux][n])};
                 });
    }
    auto angular = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(angular);
    inventory[1] = 0;       // No global radial linear invariant under cylindrical streaming.
    inventory[2] = angular; // c*Lz [J m], not the local theta conjugate [J].
    return std::isfinite(angular);
#else
    return true;
#endif
}
} // namespace

AdaptiveRepresentationInventoryWitness
CaptureAdaptiveRepresentationInventory (
    WarpXParticleContainer::Base const& photons, amrex::MultiFab const& energy,
    amrex::MultiFab const& momentum, amrex::MultiFab const& carry,
    amrex::Geometry const& geometry)
{
    AdaptiveRepresentationInventoryWitness out;
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU)
    amrex::ignore_unused(photons, energy, momentum, carry, geometry);
    return out;
#else
    bool valid = energy.nComp() == 1 && momentum.nComp() == 3 && carry.nComp() == 4 &&
        energy.boxArray() == momentum.boxArray() && energy.boxArray() == carry.boxArray() &&
        energy.DistributionMap() == momentum.DistributionMap() &&
        energy.DistributionMap() == carry.DistributionMap() &&
        energy.ixType().cellCentered() && momentum.ixType().cellCentered() &&
        carry.ixType().cellCentered() && energy.boxArray().isDisjoint() &&
        energy.boxArray().contains(geometry.Domain()) &&
        energy.boxArray().numPts() == geometry.Domain().numPts() &&
        photons.ParticleBoxArray(0) == energy.boxArray() &&
        photons.ParticleDistributionMap(0) == energy.DistributionMap() &&
        std::fegetround() == FE_TONEAREST &&
        std::numeric_limits<amrex::Real>::is_iec559 &&
        std::numeric_limits<amrex::Real>::digits == 53 &&
        std::numeric_limits<amrex::ParticleReal>::digits == 53 &&
        std::numeric_limits<long double>::digits >= 53;
#ifdef __FAST_MATH__
    valid = false;
#endif
    if (!All(valid)) { return out; }
    auto normal_or_zero = [] (amrex::Real value)
    { return std::isfinite(value) && (value == 0 || std::isnormal(value)); };
    auto normal_product = [&] (amrex::Real a, amrex::Real b)
    {
        auto const value = a*b;
        return normal_or_zero(a) && normal_or_zero(b) && normal_or_zero(value) &&
            (value != 0 || a == 0 || b == 0);
    };
    auto normal_quotient = [&] (amrex::Real a, amrex::Real b)
    {
        auto const value = a/b;
        return normal_or_zero(a) && normal_or_zero(b) && b != 0 &&
            normal_or_zero(value) && (value != 0 || a == 0);
    };
    valid = geometry.ProbLo(0) >= 0 && normal_or_zero(geometry.ProbLo(0)) &&
        normal_or_zero(geometry.ProbHi(0)) && normal_or_zero(geometry.CellSize(0)) &&
        geometry.CellSize(0) > 0;
    // Admission-only evaluations below follow the ORIGINAL Real readback
    // intermediates. They do not replace that readback or supply its values.
    // Reject under/overflow before conversion; relative gamma is not an
    // additive underflow model. Signed sums of normal binary operands may
    // cancel exactly to zero; a nonzero subnormal product/division is refused.
    for (auto const& tile : photon_replay_detail::CaptureParticles(photons).tiles)
    for (long p = 0; p < tile.count; ++p)
    {
        if (!amrex::ConstParticleIDWrapper(tile.idcpu[p]).is_valid()) { continue; }
        auto get = [&] (int n) -> long double { return tile.real_data[n * tile.count + p]; };
        auto const w = get(PIdx::w), x = get(PIdx::ux), y = get(PIdx::uy), z = get(PIdx::uz);
        auto const rw = static_cast<amrex::Real>(w), rx = static_cast<amrex::Real>(x),
            ry = static_cast<amrex::Real>(y), rz = static_cast<amrex::Real>(z);
        auto const rr = static_cast<amrex::Real>(get(PIdx::r));
        auto const theta = static_cast<amrex::Real>(get(PIdx::theta));
        auto const coefficient = rw*PhysConst::m_e*PhysConst::c;
        auto const scale = std::max({std::abs(rx), std::abs(ry), std::abs(rz)});
        auto const cosine = std::cos(theta), sine = std::sin(theta);
        auto const px = rr*cosine, py = rr*sine;
        amrex::Real norm2 = 0;
        bool ray_range = rw == 0 || scale > 0;
        if (scale > 0)
        {
            for (auto u : {rx, ry, rz})
            {
                auto const ratio = u/scale;
                ray_range = ray_range && normal_quotient(u, scale) &&
                    normal_product(ratio, ratio);
                norm2 += ratio*ratio;
            }
        }
        auto const root = std::sqrt(norm2);
        valid = valid && normal_or_zero(theta) && normal_product(rw, PhysConst::m_e) &&
            normal_product(rw*PhysConst::m_e, PhysConst::c) && ray_range &&
            normal_or_zero(norm2) && normal_or_zero(root) &&
            normal_product(coefficient, scale) && normal_product(coefficient*scale, root) &&
            normal_product(coefficient, rx) && normal_product(coefficient, ry) &&
            normal_product(coefficient, rz) && rr >= 0 &&
            normal_product(rr, cosine) && normal_product(rr, sine) &&
            normal_product(px, ry) && normal_product(py, rx) &&
            normal_or_zero(px*ry - py*rx) &&
            normal_product(coefficient, px*ry - py*rx);
        auto const f = w * PhysConst::m_e * PhysConst::c;
        std::array<long double, 4> const value{
            f * std::sqrt(x*x + y*y + z*z), 0,
            f * get(PIdx::r) * (std::cos(get(PIdx::theta))*y - std::sin(get(PIdx::theta))*x),
            f*z};
        valid = valid && w >= 0;
        for (int n = 0; n < 4; ++n)
        {
            if (n == 1) { continue; } // Exact absent radial global currency.
            valid = valid && std::isfinite(value[n]);
            out.packet[n] += value[n];
            out.packet_operands[n] += n == 2 ?
                std::abs(f*get(PIdx::r))*(std::abs(x)+std::abs(y)) : std::abs(value[n]);
        }
        valid = valid && out.packet_records < std::numeric_limits<amrex::Long>::max();
        if (valid) { ++out.packet_records; }
    }
    auto host = [&] (amrex::MultiFab const& input)
    {
        amrex::MultiFab result(input.boxArray(), input.DistributionMap(), input.nComp(),
            input.nGrowVect(),
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        amrex::dtoh_memcpy(result, input);
        amrex::Gpu::streamSynchronize();
        return result;
    };
    auto e = host(energy), q = host(momentum), c = host(carry);
    for (amrex::MFIter it(e); it.isValid(); ++it)
    for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
    {
        auto const lo = static_cast<long double>(geometry.ProbLo(0)) +
            (cell()[0]-geometry.Domain().smallEnd(0)) *
                static_cast<long double>(geometry.CellSize(0));
        auto const hi = lo + geometry.CellSize(0);
        auto const rbar = (2.L/3)*(lo*lo + lo*hi + hi*hi)/(lo+hi);
        auto const real_lo = geometry.ProbLo(0) +
            (cell()[0]-geometry.Domain().smallEnd(0))*geometry.CellSize(0);
        auto const real_hi = real_lo+geometry.CellSize(0);
        auto const midpoint = (real_lo+real_hi)/2;
        auto const width = real_hi-real_lo;
        auto const real_mean = RZAngularGeometry::MeanRadius(real_lo, real_hi);
        valid = valid && real_lo >= 0 && real_hi > real_lo &&
            normal_product(static_cast<amrex::Real>(cell()[0]-geometry.Domain().smallEnd(0)),
                           geometry.CellSize(0)) &&
            normal_or_zero(real_lo) && normal_or_zero(real_hi) &&
            normal_or_zero(real_lo+real_hi) && normal_or_zero(midpoint) &&
            normal_or_zero(width) && normal_product(width, width) &&
            normal_product(12, midpoint) && normal_quotient(width*width, 12*midpoint) &&
            normal_or_zero(real_mean);
        for (auto value : {e[it](cell()), q[it](cell(),0), q[it](cell(),1), q[it](cell(),2),
                          c[it](cell(),0), c[it](cell(),1), c[it](cell(),2), c[it](cell(),3)})
        { valid = valid && normal_or_zero(value); }
        valid = valid && normal_product(real_mean, q[it](cell(),1)) &&
            normal_product(real_mean, c[it](cell(),2));
        std::array<long double, 4> const fields{e[it](cell()), 0,
            rbar*q[it](cell(),1), q[it](cell(),2)};
        std::array<long double, 4> const carries{c[it](cell(),0), 0,
            rbar*c[it](cell(),2), c[it](cell(),3)};
        for (int n = 0; n < 4; ++n)
        {
            if (n == 1) { continue; }
            valid = valid && std::isfinite(fields[n]) && std::isfinite(carries[n]);
            out.field[n] += fields[n]; out.field_operands[n] += std::abs(fields[n]);
            out.carry[n] += carries[n]; out.carry_operands[n] += std::abs(carries[n]);
        }
        valid = valid && out.field_records < std::numeric_limits<amrex::Long>::max() &&
            out.carry_records < std::numeric_limits<amrex::Long>::max();
        if (valid) { ++out.field_records; ++out.carry_records; }
    }
    if (!All(valid)) { return out; }
    // Fail closed before integer sum overflow, including empty owners.
    auto maximum_local_records = std::max({out.packet_records, out.field_records,
                                           out.carry_records});
    amrex::ParallelDescriptor::ReduceLongMax(maximum_local_records);
    out.ranks = amrex::ParallelDescriptor::NProcs();
    if (out.ranks <= 0 || maximum_local_records >
        std::numeric_limits<amrex::Long>::max()/out.ranks) { return out; }
#ifdef AMREX_USE_MPI
    for (auto* values : {&out.packet, &out.field, &out.carry, &out.packet_operands,
                        &out.field_operands, &out.carry_operands})
    { MPI_Allreduce(MPI_IN_PLACE, values->data(), 4, MPI_LONG_DOUBLE, MPI_SUM,
        amrex::ParallelDescriptor::Communicator()); }
#endif
    amrex::ParallelDescriptor::ReduceLongSum(out.packet_records);
    amrex::ParallelDescriptor::ReduceLongSum(out.field_records);
    amrex::ParallelDescriptor::ReduceLongSum(out.carry_records);
    out.readback_operations = 64.L*out.packet_records +
        32.L*(static_cast<long double>(out.field_records)+out.carry_records) +
        32.L*out.ranks;
    out.valid = std::isfinite(out.readback_operations);
    auto const product = out.readback_operations*std::numeric_limits<amrex::Real>::epsilon();
    out.valid = out.valid && product >= 0 && product < .25L;
    if (out.valid)
    {
        // Positive accumulations receive a count-derived upper collar before
        // becoming immutable operands. epsReal >= epsLD; no defect is used.
        for (auto* values : {&out.packet_operands, &out.field_operands, &out.carry_operands})
        { for (auto& value : *values) { if (value > 0) { value = std::nextafter(
            value/(1-product), std::numeric_limits<long double>::infinity()); } } }
    }
    for (auto const* values : {&out.packet, &out.field, &out.carry, &out.packet_operands,
                              &out.field_operands, &out.carry_operands})
    { for (auto value : *values) { out.valid = out.valid && std::isfinite(value); } }
    return out;
#endif
}

bool
TryProposeAdaptiveDiffusionMask (amrex::iMultiFab const& old, amrex::MultiFab const& opacity,
                                 amrex::Geometry const& geometry, amrex::iMultiFab& candidate,
                                 AdaptiveDiffusionMaskPolicy const& policy)
{
    bool valid = std::isfinite(policy.demote_depth) && policy.demote_depth >= 0 &&
                 std::isfinite(policy.promote_depth) &&
                 policy.promote_depth > policy.demote_depth && old.nComp() == 1 &&
                 candidate.nComp() == 1 && old.ixType().cellCentered() &&
                 candidate.ixType() == old.ixType() && old.boxArray() == candidate.boxArray() &&
                 old.boxArray().isDisjoint() && old.boxArray().contains(geometry.Domain()) &&
                 old.boxArray().numPts() == geometry.Domain().numPts() &&
                 old.DistributionMap() == candidate.DistributionMap() && opacity.nComp() == 1 &&
                 opacity.ixType() == old.ixType() && opacity.boxArray() == old.boxArray() &&
                 opacity.DistributionMap() == old.DistributionMap();
    amrex::Real dx = geometry.CellSize(0);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        valid = valid && std::isfinite(geometry.CellSize(d)) && geometry.CellSize(d) > 0;
        dx = std::min(dx, geometry.CellSize(d));
    }
    if (!All(valid)) {
        return false;
    }
    if (!opacity.is_finite(0, 1, 0) || opacity.min(0) < 0 || old.min(0) < 0 || old.max(0) > 1) {
        return false;
    }
    amrex::Real limits[2]{policy.demote_depth, policy.promote_depth};
    amrex::Real high[2]{limits[0], limits[1]};
    amrex::ParallelDescriptor::ReduceRealMin(limits, 2);
    amrex::ParallelDescriptor::ReduceRealMax(high, 2);
    if (!All(limits[0] == high[0] && limits[1] == high[1])) {
        return false;
    }
    amrex::iMultiFab proposed(old.boxArray(), old.DistributionMap(), 1, 0);
    for (amrex::MFIter it(proposed); it.isValid(); ++it) {
        auto const previous = old.const_array(it);
        auto const chi = opacity.const_array(it);
        auto const output = proposed.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto const depth = chi(i, j, k) * dx;
            output(i, j, k) = previous(i, j, k) == 1 ? depth >= policy.demote_depth
                                                     : depth >= policy.promote_depth;
        });
    }
    amrex::iMultiFab::Copy(candidate, proposed, 0, 0, 1, 0);
    return true;
}

AdaptivePacketDiffusionTransitionResult
TryConvertAdaptivePacketDiffusionRepresentation (
    PhotonPacketTransaction& photons, amrex::MultiFab const& old_energy,
    amrex::MultiFab const& old_momentum, amrex::iMultiFab const& old_mask,
    amrex::iMultiFab const& desired_mask, amrex::MultiFab const& old_carry,
    amrex::Geometry const& geometry, CollidedDiffusionAdmissionComponents const& census,
    LogicalPhotonBirthKeyContext const& birth_keys, amrex::Real const photon_energy,
    CollidedDiffusionAdmissionControls const& admission, bool const capture_accounting_witness)
{
    AdaptivePacketDiffusionTransitionResult result;
    auto reject = [&result] (std::string reason) {
        result.reason = std::move(reason);
        return std::move(result);
    };
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(photons, old_energy, old_momentum, old_mask, desired_mask, old_carry,
                         geometry, census, birth_keys, photon_energy, admission,
                         capture_accounting_witness);
    return reject("Adaptive representation transfer currently requires "
                  "Cartesian or native RZ geometry");
#else
    auto const& boxes = old_energy.boxArray();
    auto const& dm = old_energy.DistributionMap();
    bool valid =
#if defined(WARPX_DIM_RZ)
        (geometry.Coord() == 0 || geometry.Coord() == 1) && geometry.ProbLo(0) >= 0 &&
        !geometry.isPeriodic(0) &&
#else
        geometry.Coord() == 0 &&
#endif
        std::isfinite(photon_energy) && photon_energy > 0 &&
        std::numeric_limits<amrex::Real>::digits >= 53 &&
        std::numeric_limits<amrex::ParticleReal>::digits >= 53 && old_energy.nComp() == 1 &&
        old_energy.ixType().cellCentered() && old_momentum.nComp() == 3 &&
        old_momentum.ixType().cellCentered() && old_momentum.boxArray() == boxes &&
        old_momentum.DistributionMap() == dm && old_carry.nComp() == 4 &&
        old_carry.ixType().cellCentered() && old_carry.boxArray() == boxes &&
        old_carry.DistributionMap() == dm && old_mask.nComp() == 1 &&
        old_mask.ixType().cellCentered() && old_mask.boxArray() == boxes &&
        old_mask.DistributionMap() == dm && desired_mask.nComp() == 1 &&
        desired_mask.ixType().cellCentered() && desired_mask.boxArray() == boxes &&
        desired_mask.DistributionMap() == dm && photons.Particles().ParticleBoxArray(0) == boxes &&
        photons.Particles().ParticleDistributionMap(0) == dm && boxes.isDisjoint() &&
        boxes.contains(geometry.Domain()) && boxes.numPts() == geometry.Domain().numPts() &&
        geometry.Domain() == photons.Particles().Geom(0).Domain();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        valid = valid && std::isfinite(geometry.ProbLo(d)) && std::isfinite(geometry.ProbHi(d)) &&
                geometry.ProbHi(d) > geometry.ProbLo(d) && std::isfinite(geometry.CellSize(d)) &&
                geometry.CellSize(d) > 0 &&
                geometry.ProbLo(d) == photons.Particles().Geom(0).ProbLo(d) &&
                geometry.ProbHi(d) == photons.Particles().Geom(0).ProbHi(d) &&
                geometry.isPeriodic(d) == photons.Particles().Geom(0).isPeriodic(d);
    }
    if (!All(valid)) {
        return reject("Invalid adaptive representation layout, precision or controls");
    }
    amrex::Real minimum[3]{photon_energy, admission.max_flux_fraction,
                           admission.max_pressure_anisotropy};
    amrex::Real maximum[3]{minimum[0], minimum[1], minimum[2]};
    amrex::ParallelDescriptor::ReduceRealMin(minimum, 3);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 3);
    auto expected_keys = birth_keys;
    amrex::ParallelDescriptor::Bcast(&expected_keys.seed, 1, 0);
    amrex::ParallelDescriptor::Bcast(&expected_keys.interval, 1, 0);
    valid = expected_keys.seed == birth_keys.seed && expected_keys.interval == birth_keys.interval;
    for (int d = 0; d < 3; ++d) {
        valid = valid && minimum[d] == maximum[d];
    }
    if (!All(valid)) {
        return reject("Adaptive representation controls or draw context differ "
                      "between ranks");
    }
    if (!old_carry.is_finite(0, 4, 0)) {
        return reject("Invalid persisted representation carry");
    }
    int capture_low = capture_accounting_witness, capture_high = capture_low;
    amrex::ParallelDescriptor::ReduceIntMin(capture_low);
    amrex::ParallelDescriptor::ReduceIntMax(capture_high);
    if (capture_low != capture_high)
    { return reject("Conversion witness controls differ between ranks"); }
    if (capture_accounting_witness)
    {
        result.accounting_witness.before = CaptureAdaptiveRepresentationInventory(
            photons.Particles(), old_energy, old_momentum, old_carry, geometry);
        if (!result.accounting_witness.before.valid)
        { return reject("Unsupported or invalid actual conversion input witness"); }
    }
    FourVector packet_before{};
    if (!Packets(photons.Particles(), packet_before)) {
        return reject("Invalid initial native packet inventory");
    }
    auto const field_before = Fields(old_energy, old_momentum, geometry);
    Clone(result.energy, old_energy);
    Clone(result.momentum, old_momentum);
    Clone(result.representation_carry, old_carry);
    result.mask.define(boxes, dm, 1, old_mask.nGrowVect());
    amrex::iMultiFab::Copy(result.mask, old_mask, 0, 0, 1, old_mask.nGrowVect());
    result.carry_change.define(boxes, dm, 4, 0);
    result.carry_change.setVal(0);
    result.realized_demotion_inventory.define(boxes, dm, 4, 0);
    result.realized_demotion_inventory.setVal(0);
    amrex::iMultiFab promotion_mask(boxes, dm, 1, 0);
    promotion_mask.setVal(0);
    amrex::MultiFab host_e(boxes, dm, 1, old_energy.nGrowVect(),
                           amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::MultiFab host_q(boxes, dm, 3, old_momentum.nGrowVect(),
                           amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::iMultiFab host_old(boxes, dm, 1, old_mask.nGrowVect(),
                              amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::iMultiFab host_desired(boxes, dm, 1, desired_mask.nGrowVect(),
                                  amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(host_e, old_energy);
    amrex::dtoh_memcpy(host_q, old_momentum);
    amrex::dtoh_memcpy(host_old, old_mask);
    amrex::dtoh_memcpy(host_desired, desired_mask);
    amrex::Gpu::streamSynchronize();
    detail::EmissionPlan plan;
    auto const lower = geometry.ProbLoArray(), dx = geometry.CellSizeArray();
    auto const domain_lo = geometry.Domain().smallEnd();
    auto demotion_keys = birth_keys;
    demotion_keys.seed = MixPhotonBirthKey(birth_keys.seed ^ UINT64_C(0xa72ed14ef96b7103));
    for (amrex::MFIter it(host_e); it.isValid(); ++it) {
        auto const e = host_e.const_array(it), q = host_q.const_array(it);
        auto const previous = host_old.const_array(it), desired = host_desired.const_array(it);
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell) {
            auto const index = cell();
            FourVector value{e(index), q(index, 0), q(index, 1), q(index, 2)};
            int const old = previous(index), next = desired(index);
            valid =
                valid && (old == 0 || old == 1) && (next == 0 || next == 1) && Physical(value) &&
                (old == 1 || (value[0] == 0 && value[1] == 0 && value[2] == 0 && value[3] == 0));
            if (!valid || old != 1 || next != 0) {
                continue;
            }
            ++result.demoted_cells;
            auto const rays = ReconstructPacketMoments(value, photon_energy);
            valid = valid && rays.valid;
            if (!rays.valid) {
                continue;
            }
            amrex::ParticleReal position[3]{0, 0, 0};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                auto const lo = lower[d] + (index[d] - domain_lo[d]) * dx[d];
                auto const center =
#if defined(WARPX_DIM_RZ)
                    d == 0 ? RZAngularGeometry::MeanRadius(lo, lo + dx[d]) :
#endif
                           lo + dx[d] / 2;
                auto const coordinate =
                    detail::ClampParticleConversionPosition(center, lo, lo + dx[d]);
#if AMREX_SPACEDIM == 1
                position[2] = coordinate;
#elif AMREX_SPACEDIM == 2
                position[d == 0 ? 0 : 2] = coordinate;
#else
                position[d] = coordinate;
#endif
            }
            for (int n = 0; n < 8; ++n) {
                auto const& ray = rays.photons[n];
                if (ray.energy_weight == 0) {
                    continue;
                }
                std::uint64_t key = 0;
                valid =
                    valid && MakePhotonBirthDrawKey(geometry, index, 0, 1, n, demotion_keys, key);
                plan.x.push_back(position[0]);
                plan.y.push_back(position[1]);
                plan.z.push_back(position[2]);
                plan.ux.push_back(ray.proper_velocity[0]);
                plan.uy.push_back(ray.proper_velocity[1]);
                plan.uz.push_back(ray.proper_velocity[2]);
                plan.weight.push_back(ray.energy_weight);
                plan.logical_draw_keys.push_back(key);
            }
        }
    }
    if (!All(valid)) {
        return reject("Invalid mask/off-mask inventory or non-realizable E/q");
    }
    for (amrex::MFIter it(promotion_mask); it.isValid(); ++it) {
        auto const old = old_mask.const_array(it), desired = desired_mask.const_array(it);
        auto const output = promotion_mask.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            output(i, j, k) = old(i, j, k) == 0 && desired(i, j, k) == 1;
        });
    }
    auto promoted = TryAdmitCollidedPhotons(photons.Particles(), census, result.energy,
                                            result.momentum, promotion_mask, geometry, admission);
    if (!promoted.valid) {
        return reject("Collided-census representation admission rejected");
    }
    result.promoted_photons = promoted.admitted;
    result.realized_promotion_inventory = std::move(promoted.admitted_four_inventory);
    if (!detail::StageEmissionPlan(photons, plan) ||
        !photons.ProvisionalBirthInventory(result.realized_demotion_inventory, geometry,
                                           result.demoted_photons, &plan.logical_draw_keys)) {
        return reject("Native demotion births or stored inventory realization rejected");
    }
    amrex::Long expected_births = static_cast<amrex::Long>(plan.x.size());
    amrex::ParallelDescriptor::ReduceLongSum(expected_births);
    if (result.demoted_photons != expected_births) {
        return reject("Actual stored demotion births differ from the constructive batch");
    }
    amrex::MultiFab host_born(boxes, dm, 4, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::MultiFab host_promoted(boxes, dm, 4, 0,
                                  amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(host_born, result.realized_demotion_inventory);
    amrex::dtoh_memcpy(host_promoted, result.realized_promotion_inventory);
    amrex::Gpu::streamSynchronize();
    constexpr auto bound = 128 * std::numeric_limits<amrex::Real>::epsilon();
    for (amrex::MFIter it(host_e); it.isValid(); ++it) {
        auto const e = host_e.const_array(it), q = host_q.const_array(it);
        auto const old = host_old.const_array(it), desired = host_desired.const_array(it);
        auto const born = host_born.const_array(it), admitted = host_promoted.const_array(it);
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell) {
            auto const index = cell();
            if (old(index) == 1 && desired(index) == 0) {
                FourVector before{e(index), q(index, 0), q(index, 1), q(index, 2)};
                for (int d = 0; d < 4; ++d) {
                    valid = valid && std::isfinite(born(index, d)) &&
                            std::abs(before[d] - born(index, d)) <= bound * before[0];
                }
            } else {
                valid = valid && born(index, 0) == 0 && born(index, 1) == 0 &&
                        born(index, 2) == 0 && born(index, 3) == 0;
            }
            if (old(index) == 0 && desired(index) == 1) {
                if (admitted(index, 0) > 0) {
                    ++result.promoted_cells;
                } else {
                    ++result.deferred_cells;
                }
            }
        }
    }
    if (!All(valid)) {
        return reject("Actual native demotion E/q exceeds the cell arithmetic bound");
    }
    for (amrex::MFIter it(result.energy); it.isValid(); ++it) {
        auto const e = result.energy.array(it), q = result.momentum.array(it);
        auto const old_e = old_energy.const_array(it), old_q = old_momentum.const_array(it);
        auto const old = old_mask.const_array(it), desired = desired_mask.const_array(it);
        auto const born = result.realized_demotion_inventory.const_array(it);
        auto const admitted = result.realized_promotion_inventory.const_array(it);
        auto const mask = result.mask.array(it);
        auto const carry = result.representation_carry.array(it);
        auto const change = result.carry_change.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            bool const demote = old(i, j, k) == 1 && desired(i, j, k) == 0;
            bool const promote =
                old(i, j, k) == 0 && desired(i, j, k) == 1 && admitted(i, j, k, 0) > 0;
            mask(i, j, k) = demote ? 0 : promote ? 1 : old(i, j, k);
            for (int d = 0; d < 4; ++d) {
                auto const before = d == 0 ? old_e(i, j, k) : old_q(i, j, k, d - 1);
                auto const after = d == 0 ? e(i, j, k) : q(i, j, k, d - 1);
                change(i, j, k, d) = demote    ? before - born(i, j, k, d)
                                     : promote ? (before + admitted(i, j, k, d)) - after
                                               : 0;
                carry(i, j, k, d) += change(i, j, k, d);
            }
            if (demote) {
                e(i, j, k) = 0;
                for (int d = 0; d < 3; ++d) {
                    q(i, j, k, d) = 0;
                }
            }
        });
    }
    FourVector packet_after{};
    if (!Packets(photons.Particles(), packet_after) ||
        !result.representation_carry.is_finite(0, 4, 0)) {
        return reject("Invalid realized private inventory after adaptive transition");
    }
    auto const field_after = Fields(result.energy, result.momentum, geometry);
    auto const carry_before = Carry(old_carry, geometry);
    auto const carry_after = Carry(result.representation_carry, geometry);
    for (int d = 0; d < 4; ++d) {
        result.raw_inventory_residual[d] =
            (packet_after[d] - packet_before[d]) + (field_after[d] - field_before[d]);
        auto const carry_delta = carry_after[d] - carry_before[d];
        result.carried_inventory_residual[d] = result.raw_inventory_residual[d] + carry_delta;
        valid = valid && std::isfinite(result.carried_inventory_residual[d]) &&
                std::abs(result.carried_inventory_residual[d]) <=
                    bound * (packet_before[0] + field_before[0]) *
#if defined(WARPX_DIM_RZ)
                        (d == 2 ? geometry.ProbHi(0) : 1);
#else
                        1;
#endif
    }
    if (!All(valid)) {
        return reject("Adaptive representation four-inventory ledger rejected");
    }
    amrex::ParallelDescriptor::ReduceLongSum(result.demoted_cells);
    amrex::ParallelDescriptor::ReduceLongSum(result.promoted_cells);
    amrex::ParallelDescriptor::ReduceLongSum(result.deferred_cells);
    if (capture_accounting_witness)
    {
        auto& witness = result.accounting_witness;
        witness.after = CaptureAdaptiveRepresentationInventory(
            photons.Particles(), result.energy, result.momentum,
            result.representation_carry, geometry);
        if (!witness.after.valid) { return reject("Invalid actual conversion endpoint witness"); }
        for (int d = 0; d < 4; ++d)
        {
            witness.declared_conversion_roundoff_bound[d] =
                d == 1 ? 0.L : static_cast<long double>(bound *
                    (packet_before[0] + field_before[0]) * (d == 2 ? geometry.ProbHi(0) : 1));
            witness.carried_residual[d] =
                (witness.after.packet[d] - witness.before.packet[d]) +
                (witness.after.field[d] - witness.before.field[d]) +
                (witness.after.carry[d] - witness.before.carry[d]);
        }
        witness.valid = true;
    }
    result.valid = true;
    return result;
#endif
}
} // namespace warpx::radiation
