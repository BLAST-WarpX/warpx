/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeResolvedThermalSource.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/OwnedNodalCaloricDifference.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcMetricTransport.H"
#include "HybridCellMaterialState.H"
#include "NativeCompositeTableCornerCapture.H"
#include "NativePhotonReplayInternals.H"
#include "NativeRZIndependentGrayOpacitySlice.H"
#include "NativeRZOwnedNodalCaloric.H"
#include "Particles/MultiParticleContainer.H"
#include "PhotonLogicalDrawKey.H"
#include "PureGrayAbsorptionSource.H"
#include "RZMomentGeometry.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace warpx::radiation
{
using namespace amrex::literals;
namespace
{
using Vector = std::vector<amrex::Real>;

amrex::MultiFab
Copy (amrex::MultiFab const& field)
{
    amrex::MultiFab result(field.boxArray(), field.DistributionMap(), field.nComp(),
                           field.nGrowVect());
    amrex::MultiFab::Copy(result, field, 0, 0, field.nComp(), field.nGrowVect());
    return result;
}

amrex::MultiFab
Host (amrex::MultiFab const& field)
{
    amrex::MultiFab result(field.boxArray(), field.DistributionMap(), field.nComp(),
                           field.nGrowVect(), amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(result, field);
    amrex::Gpu::streamSynchronize();
    return result;
}

#if defined(WARPX_DIM_RZ)
// Match NativeRZOwnedNodalCaloric's corrected physical native-FV measure,
// not the geometric cell-corner partition alone.
amrex::Real
RZNativeCornerVolume (amrex::IntVect const& cell, int corner,
                      amrex::Geometry const& geometry)
{
    auto const radial_node = cell[0] + (corner & 1);
    auto const dr = geometry.CellSize(0), dz = geometry.CellSize(1);
    auto const radius = geometry.ProbLo(0) +
        (radial_node - geometry.Domain().smallEnd(0)) * dr;
    auto const axis_factor = WarpX::GetInstance().verboncoeurAxisCorrection()
        ? 1.0_rt / 3 : 1.0_rt / 4;
    auto transport = warpx::hybrid::cylindricalTransportNodeVolume(
        radius, dr, dz, axis_factor);
    if (radial_node == geometry.Domain().bigEnd(0) + 1)
    {
        transport *= 0.5_rt;
    }
    auto const lo = std::max(geometry.ProbLo(0), radius - 0.5_rt * dr);
    auto const hi = std::min(geometry.ProbHi(0), radius + 0.5_rt * dr);
    auto const geometric = MathConst::pi * (hi * hi - lo * lo) * dz;
    auto const corner_geometric = detail::HybridCellCornerVolume(
        cell[0], cell[1], 0, corner & 1, (corner >> 1) & 1, 0,
        geometry.ProbLoArray(), geometry.CellSizeArray(),
        amrex::lbound(geometry.Domain()));
    return corner_geometric * transport / geometric;
}
#endif

// OwnerMask and OverrideSync share AMReX's global-box/periodic precedence.
// Krylov coordinates exist only at this rank's unique valid owner nodes;
// fields supply the duplicate/halo corner values used by native cell physics.
struct OwnedCaloricNodes
{
    struct Node
    {
        int fab;
        amrex::IntVect index;
    };
    std::vector<Node> owned;
    amrex::Long global_size = 0;

    void
    Define (amrex::MultiFab const& field, amrex::Geometry const& geometry)
    {
        auto const owners = field.OwnerMask(geometry.periodicity());
        owned.clear();
        for (amrex::MFIter it(field); it.isValid(); ++it)
        {
            auto const& mask = (*owners)[it];
            std::vector<int> flags(static_cast<std::size_t>(mask.box().numPts()));
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, mask.dataPtr(),
                             mask.dataPtr() + flags.size(), flags.begin());
            for (amrex::BoxIterator node(it.validbox()); node.ok(); ++node)
            {
                if (flags[static_cast<std::size_t>(mask.box().index(node()))] != 0)
                {
                    owned.push_back({it.index(), node()});
                }
            }
        }
        global_size = static_cast<amrex::Long>(owned.size());
        amrex::ParallelDescriptor::ReduceLongSum(global_size);
    }

    Vector
    GatherHost (amrex::MultiFab const& host, int component = 0) const
    {
        Vector values(owned.size());
        for (std::size_t n = 0; n < owned.size(); ++n)
        {
            values[n] = host[owned[n].fab](owned[n].index, component);
        }
        return values;
    }

    Vector
    Gather (amrex::MultiFab const& field) const
    {
        return GatherHost(Host(field));
    }

    void
    Scatter (Vector const& values, amrex::Geometry const& geometry, amrex::MultiFab& field) const
    {
        auto host = Host(field);
        host.setVal(0);
        for (std::size_t n = 0; n < owned.size(); ++n)
        {
            host[owned[n].fab](owned[n].index, 0) = values[n];
        }
        amrex::htod_memcpy(field, host);
        amrex::Gpu::streamSynchronize();
        field.OverrideSync(geometry.periodicity());
        field.FillBoundary(geometry.periodicity());
    }
};

amrex::Real
Dot (Vector const& first, Vector const& second)
{
    amrex::Real sum = 0;
    for (std::size_t n = 0; n < first.size(); ++n)
    {
        sum += first[n] * second[n];
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}

bool
SameSourceControls (NativeResolvedThermalSourceOptions const& options)
{
    int low[] = {options.max_temperature_iterations,
                 options.max_linear_iterations,
                 options.max_material_iterations,
                 options.max_line_search,
                 options.allow_vacuum_noop ? 1 : 0,
                 static_cast<int>(options.caloric_backend),
                 options.reciprocal_emission_rays_per_corner,
                 options.reciprocal_composite_old_temperature ? 1 : 0,
                 options.rz_reciprocal_implicit ? 1 : 0,
                 options.rz_closed_reciprocal_endpoint_trial ? 1 : 0,
                 options.rz_closed_reciprocal_spatial_endpoint_trial ? 1 : 0,
                 options.rz_closed_reciprocal_table_trial ? 1 : 0,
                 options.rz_single_selected_table_trial ? 1 : 0,
                 options.rz_independent_gray_opacity != nullptr,
                 options.rz_single_selected_table_vacuum,
                 options.cartesian_native_nodal_force,
                 options.rz_native_nodal_force,
                 options.rz_centered_ideal_temperature_trial};
    int high[18];
    std::copy(std::begin(low), std::end(low), high);
    amrex::ParallelDescriptor::ReduceIntMin(low, 18);
    amrex::ParallelDescriptor::ReduceIntMax(high, 18);
    amrex::Real minimum[] = {options.material_tolerance, options.work_tolerance,
                             options.energy_tolerance, options.minimum_electron_density};
    amrex::Real maximum[4];
    std::copy(std::begin(minimum), std::end(minimum), maximum);
    amrex::ParallelDescriptor::ReduceRealMin(minimum, 4);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 4);
    return std::equal(std::begin(low), std::end(low), high) &&
           std::equal(std::begin(minimum), std::end(minimum), maximum);
}

bool
UniqueLogicalKeys (WarpXParticleContainer::Base const& photons)
{
    return photon_replay_detail::UniqueLogicalPhotonKeys(photons);
}

struct SourceContext
{
    WarpXParticleContainer* live;
    PhotonPacketTransaction const* endpoint;
    ParticleImpulseMaterial* material;
    HybridPICModel const* model;
    std::string path;
    amrex::Geometry geometry;
    NativeResolvedThermalSourceOptions options;
    amrex::Real dt, volume, floor;
    ElectronThermodynamicsExecutor eos;
    amrex::MultiFab old_temperature, density, old_diffusion, opacity;
    amrex::MultiFab trial_work;
    NativeResolvedThermalTrialBackend const* backend = nullptr;
    bool rz_reciprocal = false;
    bool composite_table = false;
    std::unique_ptr<NativeRZOwnedNodalCaloricState> rz_caloric;
    std::unique_ptr<detail::NativeCompositeTableCornerCapture> table_caloric;
    amrex::iMultiFab kinetic_mask, capture_mask;
    amrex::MultiFab diffusion_flags;
    NativePacketThermalEventReplay packets;
    Vector old_nodes, charge, old_capacity, old_energy, available;
    Vector old_corner_energy_offset;
    std::vector<unsigned char> active_nodes;
    OwnedCaloricNodes node_layout;
    amrex::MultiFab node_state;
    amrex::MultiFab composite_state, corner_weights;
    int peak_basis_vectors = 0;
    int table_inverse_newton_fallbacks = 0;
    int table_unaffordable_newton_fallbacks = 0;
    int table_converged_nondescent_probes = 0;
    bool table_initial_inverse_unaffordable = false;
};

struct Trial
{
    bool valid = false;
    std::string reason;
    NativeResolvedThermalSourceResult candidate;
    Vector residual, heat, roundoff;
    amrex::Real norm = std::numeric_limits<amrex::Real>::infinity();
};

amrex::Real
PhysicalNodeTemperature (SourceContext const& s, Vector const& nodes, std::size_t n)
{
    return s.options.rz_centered_ideal_temperature_trial
               ? s.old_nodes[n] + nodes[n] : nodes[n];
}

amrex::Real
CaloricTemperatureScale (SourceContext const& s, Vector const& nodes, Trial const& base,
                         std::size_t n)
{
    return std::max({std::abs(PhysicalNodeTemperature(s, nodes, n)), std::abs(s.old_nodes[n]),
                     std::abs(base.residual[n]) / s.old_capacity[n], 1._rt});
}

amrex::Real
FrozenCaloricMerit (SourceContext const& s, Vector const& reference_nodes, Trial const& base,
                    Trial const& candidate)
{
    amrex::Real sum = 0;
    for (std::size_t n = 0; n < reference_nodes.size(); ++n)
    {
        if (!s.active_nodes[n])
        {
            continue;
        }
        auto const energy_scale =
            s.old_capacity[n] * CaloricTemperatureScale(s, reference_nodes, base, n);
        auto const value = candidate.residual[n] / energy_scale;
        sum += value * value;
    }
    // Empty ranks contribute zero but participate in the same scalar reduction.
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}

#if defined(WARPX_DIM_RZ)
// Reconstruct accepted D transport and cylindrical stress using the returned
// physical faces, independently of the field solver's private residual. A
// D->K face remains an explicit outgoing account for the outer interval; this
// source neither turns it into heat nor creates a packet to absorb it.
bool
CheckRZFaceAndGlobalAccounts (SourceContext const& s,
                              NativeResolvedThermalSourceResult const& result)
{
    auto const old = Host(s.old_diffusion);
    auto const next = Host(result.diffusion_radiation);
    auto const raw = Host(result.raw_debit);
    auto const packet = Host(result.rz_packet_debit);
    auto const correction = Host(result.rz_leakage->donor_radiation_debit_correction);
    auto const stress = Host(result.rz_outward_pressure_stress);
    auto const connection = Host(result.rz_theta_connection);
    auto const roundoff = Host(result.storage_roundoff_bound);
    amrex::MultiFab device_flags(s.old_diffusion.boxArray(),
                                 s.old_diffusion.DistributionMap(), 1, 1);
    device_flags.setVal(0);
    amrex::MultiFab::Copy(device_flags, s.diffusion_flags, 0, 0, 1, 0);
    device_flags.FillBoundary(s.geometry.periodicity());
    auto const flags = Host(device_flags);
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> energy_face, stress_face, leak_face;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        energy_face[d] = Host(*result.rz_face_lab_energy_current[d]);
        stress_face[d] = Host(*result.rz_face_lab_stress[d]);
        leak_face[d] = Host(*result.rz_face_leaked_energy[d]);
    }
    auto const metric = RZMomentMetric(s.geometry);
    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real totals[13]{};
    long double axial_operands[2]{}; // owned-D face and D/K interface incidences
    bool valid = true;
    for (amrex::MFIter it(old); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            if (flags[it](index) == 0) { continue; }
            amrex::Real transfer = 0, face_scale = 0;
            amrex::Real interface_energy = 0, interface_pz = 0, interface_lz = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                auto upper = index;
                ++upper[d];
                auto lower_neighbor = index, upper_neighbor = index;
                --lower_neighbor[d]; ++upper_neighbor[d];
                auto const area_lo = metric.Area(d, index[0]);
                auto const area_hi = metric.Area(d, upper[0]);
                if (!s.options.rz_closed_reciprocal_spatial_endpoint_trial)
                {
                    // Each owned D cell uses these four face incidences in its
                    // axial stress divergence. D/D faces deliberately occur
                    // twice: two arithmetic uses, not duplicate cell owners.
                    axial_operands[0] += static_cast<long double>(s.dt) *
                        (static_cast<long double>(area_lo) *
                             std::abs(static_cast<long double>(stress_face[d][it](index, 2))) +
                         static_cast<long double>(area_hi) *
                             std::abs(static_cast<long double>(stress_face[d][it](upper, 2))));
                }
                auto const low = s.dt * area_lo * energy_face[d][it](index);
                auto const high = s.dt * area_hi * energy_face[d][it](upper);
                transfer += high - low;
                face_scale += std::abs(high) + std::abs(low);
                for (int side = 0; side < 2; ++side)
                {
                    auto const face = side == 0 ? index : upper;
                    auto const neighbor = side == 0 ? lower_neighbor : upper_neighbor;
                    if (d == 0 &&
                        (neighbor[0] < s.geometry.Domain().smallEnd(0) ||
                         neighbor[0] > s.geometry.Domain().bigEnd(0)))
                    {
                        valid = valid && energy_face[d][it](face) == 0 &&
                            stress_face[d][it](face, 1) == 0 &&
                            stress_face[d][it](face, 2) == 0 &&
                            leak_face[d][it](face) == 0;
                        continue;
                    }
                    if (s.options.rz_closed_reciprocal_spatial_endpoint_trial && d == 1 &&
                        (neighbor[1] < s.geometry.Domain().smallEnd(1) ||
                         neighbor[1] > s.geometry.Domain().bigEnd(1)))
                    {
                        valid = valid && energy_face[d][it](face) == 0 &&
                            leak_face[d][it](face) == 0 &&
                            stress_face[d][it](face, 0) == 0 &&
                            stress_face[d][it](face, 1) == 0 &&
                            std::isfinite(stress_face[d][it](face, 2));
                        auto const wall = (side == 0 ? -1._rt : 1._rt) *
                            s.dt * metric.Area(d, face[0]) * stress_face[d][it](face, 2);
                        totals[9 + side] += wall;
                        totals[11 + side] += std::abs(wall);
                        continue; // External traction, never a D/K interface.
                    }
                    if (flags[it](neighbor) != 0) { continue; }
                    auto const sign = side == 0 ? -1._rt : 1._rt;
                    auto const area = metric.Area(d, face[0]);
                    auto const flux = s.dt * area * sign * energy_face[d][it](face);
                    auto const leak = leak_face[d][it](face);
                    auto const flux_bound = 128 * epsilon *
                        (std::abs(flux) + std::abs(leak));
                    valid = valid && leak >= 0 &&
                        std::abs(flux - leak) <= flux_bound;
                    interface_energy += leak;
                    interface_pz += s.dt * area * sign * stress_face[d][it](face, 2);
                    if (!s.options.rz_closed_reciprocal_spatial_endpoint_trial)
                    {
                        axial_operands[1] += std::abs(static_cast<long double>(s.dt) *
                            area * static_cast<long double>(stress_face[d][it](face, 2)));
                    }
                    auto const lever = d == 0 ? metric.Radius(face[0])
                                              : metric.Angular(index[0]).mean_radius;
                    interface_lz += s.dt * area * sign * lever *
                                    stress_face[d][it](face, 1);
                }
            }
            auto const field_raw = raw[it](index, 0) - packet[it](index, 0) -
                                   correction[it](index, 0);
            auto const energy_defect = old[it](index, 0) - next[it](index, 0) -
                                       transfer - field_raw;
            auto const energy_scale = std::abs(old[it](index, 0)) +
                std::abs(next[it](index, 0)) + face_scale +
                std::abs(raw[it](index, 0)) + std::abs(packet[it](index, 0)) +
                std::abs(correction[it](index, 0));
            auto const energy_bound = roundoff[it](index, 0) +
                s.options.energy_tolerance * energy_scale + 512 * epsilon * energy_scale;
            valid = valid && std::isfinite(energy_defect + energy_bound) &&
                std::abs(energy_defect) <= energy_bound;
            for (int component = 0; component < 3; ++component)
            {
                auto const field_source = raw[it](index, component + 1) -
                    packet[it](index, component + 1) -
                    correction[it](index, component + 1);
                auto const defect = old[it](index, component + 1) -
                    next[it](index, component + 1) -
                    stress[it](index, component) - field_source;
                auto const scale = std::abs(old[it](index, component + 1)) +
                    std::abs(next[it](index, component + 1)) +
                    std::abs(stress[it](index, component)) +
                    std::abs(raw[it](index, component + 1)) +
                    std::abs(packet[it](index, component + 1)) +
                    std::abs(correction[it](index, component + 1));
                auto const bound = roundoff[it](index, component + 1) +
                    512 * epsilon * scale;
                valid = valid && std::isfinite(defect + bound) &&
                    std::abs(defect) <= bound;
            }
            auto radial_high = index;
            ++radial_high[0];
            auto const expected_connection = metric.Angular(index[0]).Geometric(
                stress_face[0][it](index, 1),
                stress_face[0][it](radial_high, 1), s.dt);
            auto const connection_scale = std::abs(expected_connection) +
                std::abs(connection[it](index));
            valid = valid && std::abs(expected_connection - connection[it](index)) <=
                128 * epsilon * connection_scale;
            totals[0] += transfer;
            totals[1] += interface_energy;
            totals[2] += face_scale;
            totals[3] += stress[it](index, 2);
            totals[4] += interface_pz;
            totals[5] += std::abs(stress[it](index, 2)) + std::abs(interface_pz);
            auto const rbar = metric.Angular(index[0]).mean_radius;
            totals[6] += rbar * stress[it](index, 1);
            totals[7] += interface_lz;
            totals[8] += std::abs(rbar * stress[it](index, 1)) +
                         std::abs(interface_lz);
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    amrex::ParallelDescriptor::ReduceRealSum(totals, 13);
    bool axial_account = std::abs(totals[3] - totals[4]) <= 512 * epsilon * totals[5];
    if (!s.options.rz_closed_reciprocal_spatial_endpoint_trial)
    {
        // Per owned D cell: field geometry/stress formation <=27 operations;
        // audit face geometry/interface/scalar sums <=72 (including separately
        // recomputed interface areas); this additional
        // positive face/interface-incidence accumulation <=26. The 128/cell capacity
        // therefore covers both representations of the SAME stored faces.
        // All domain cells bound the active D subset without a new count
        // collective; 16/rank covers the existing 13 scalar reductions plus
        // these two operand reductions, and 256 covers final scalar arithmetic.
#ifdef AMREX_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, axial_operands, 2, MPI_LONG_DOUBLE, MPI_SUM,
                      amrex::ParallelDescriptor::Communicator());
#endif
        auto const operations = 256.L + 128.L * s.geometry.Domain().numPts() +
            16.L * amrex::ParallelDescriptor::NProcs();
        auto const product = operations * epsilon;
        bool const arithmetic_valid = std::isfinite(product) && product >= 0 && product < .25L &&
            std::isfinite(axial_operands[0]) && axial_operands[0] >= 0 &&
            std::isfinite(axial_operands[1]) && axial_operands[1] >= 0;
        // totals[5] includes the actual interface/scalar-add operands. It is
        // NOT enough alone: stress and interface sums may almost cancel.
        // p/(1-2p)=gamma/(1-gamma) also covers positive operand summation.
        auto const face_scale = std::nextafter(axial_operands[0] + axial_operands[1],
            std::numeric_limits<long double>::infinity());
        auto const operand = std::nextafter(face_scale + totals[5],
            std::numeric_limits<long double>::infinity());
        auto const factor = std::nextafter(product / (1-2*product),
            std::numeric_limits<long double>::infinity());
        auto const allowance = std::nextafter(factor*operand,
            std::numeric_limits<long double>::infinity());
        // Retain every original periodic pass; only this arithmetic identity
        // gains its missing uncancelled scale. No physical tolerance enters.
        axial_account = arithmetic_valid && std::isfinite(allowance) &&
            (axial_account || std::abs(static_cast<long double>(totals[3])-totals[4]) <= allowance);
    }
    if (s.options.rz_closed_reciprocal_spatial_endpoint_trial)
    {
        auto const neps = (256._rt + 64._rt * s.geometry.Domain().numPts() +
                          16._rt * amrex::ParallelDescriptor::NProcs()) * epsilon;
        bool const safe_arithmetic = std::isfinite(neps) && neps >= 0 && neps < .5_rt;
        auto const gamma = safe_arithmetic ? neps / (1 - neps) : 0;
        auto const lo = PhysConst::c * result.rz_axial_wall_momentum_by_side[0];
        auto const hi = PhysConst::c * result.rz_axial_wall_momentum_by_side[1];
        auto const sum = PhysConst::c * result.rz_axial_wall_momentum;
        axial_account = safe_arithmetic && std::isfinite(lo + hi + sum) &&
            std::abs(lo - totals[9]) <= gamma * (std::abs(lo) + totals[11]) &&
            std::abs(hi - totals[10]) <= gamma * (std::abs(hi) + totals[12]) &&
            std::abs(sum - lo - hi) <= gamma * (std::abs(sum) + std::abs(lo) + std::abs(hi)) &&
            std::abs(totals[3] - totals[4] - totals[9] - totals[10]) <=
                gamma * (totals[5] + totals[11] + totals[12]);
    }
    valid = valid && std::abs(totals[0] - totals[1]) <=
        512 * epsilon * (totals[2] + std::abs(totals[1])) &&
        axial_account &&
        std::abs(totals[6] - totals[7]) <= 512 * epsilon * totals[8];
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

// Independent global E, axial cP and axial cLz inventory reconstructed from
// old/new D storage, actual endpoint packet debit and actual stored leakage
// rays. The material side is the ONE staged native finite kick/carry plus the
// independent physical old-corner caloric realization. No radial sum is used.
bool
CheckRZGlobalNativeInventory (SourceContext const& s,
                              NativeResolvedThermalSourceResult const& result)
{
    auto const old = Host(s.old_diffusion);
    auto const next = Host(result.diffusion_radiation);
    auto const packet = Host(result.rz_packet_debit);
    auto const ion = Host(result.material_stage->ConjugateImpulse());
    auto const carry = Host(result.material_stage->ConjugateMomentumCarryChange());
    auto const work = Host(result.material_stage->ActualWork());
    auto const energy_carry = Host(result.material_stage->EnergyCarryChange());
    auto const heat = Host(result.actual_heat);
    auto const storage_bound = Host(result.storage_roundoff_bound);
    auto const stage_bound = Host(result.material_stage->
                                      RZConjugateMomentumRoundoffBound());
    auto const metric = RZMomentMetric(s.geometry);
    std::array<amrex::Real, 11> totals{};
    amrex::Long local_cells = 0, local_faces = 0;
    for (amrex::MFIter it(old); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            ++local_cells;
            auto const index = cell();
            auto const radius = metric.Angular(index[0]).mean_radius;
            auto const radiation_energy = packet[it](index, 0) +
                old[it](index, 0) - next[it](index, 0);
            auto const material_energy = work[it](index) +
                energy_carry[it](index) + heat[it](index);
            totals[0] += radiation_energy;
            totals[1] += material_energy;
            totals[2] += std::abs(packet[it](index, 0)) +
                std::abs(old[it](index, 0)) + std::abs(next[it](index, 0)) +
                std::abs(work[it](index)) + std::abs(energy_carry[it](index)) +
                std::abs(heat[it](index));
            auto const radiation_pz = packet[it](index, 3) +
                old[it](index, 3) - next[it](index, 3);
            auto const material_pz = PhysConst::c *
                (ion[it](index, 2) + carry[it](index, 2));
            totals[3] += radiation_pz;
            totals[4] += material_pz;
            totals[5] += std::abs(packet[it](index, 3)) +
                std::abs(old[it](index, 3)) + std::abs(next[it](index, 3)) +
                PhysConst::c * (std::abs(ion[it](index, 2)) +
                                std::abs(carry[it](index, 2)));
            auto const radiation_lz = radius * (packet[it](index, 2) +
                old[it](index, 2) - next[it](index, 2));
            auto const material_lz = radius * PhysConst::c *
                (ion[it](index, 1) + carry[it](index, 1));
            totals[6] += radiation_lz;
            totals[7] += material_lz;
            totals[8] += radius * (std::abs(packet[it](index, 2)) +
                std::abs(old[it](index, 2)) + std::abs(next[it](index, 2)) +
                PhysConst::c * (std::abs(ion[it](index, 1)) +
                                std::abs(carry[it](index, 1))));
            totals[9] += storage_bound[it](index, 3) +
                PhysConst::c * stage_bound[it](index, 2);
            totals[10] += radius * (storage_bound[it](index, 2) +
                PhysConst::c * stage_bound[it](index, 1));
        }
    }
    for (auto const& face : result.rz_leakage->face_accounts)
    {
        ++local_faces;
        totals[0] -= face.actual[0];
        totals[2] += std::abs(face.actual[0]);
        totals[3] -= face.actual[3];
        totals[5] += std::abs(face.actual[3]);
        totals[6] -= face.actual[4];
        totals[8] += std::abs(face.actual[4]);
    }
    amrex::ParallelDescriptor::ReduceLongSum(local_cells);
    amrex::ParallelDescriptor::ReduceLongSum(local_faces);
    amrex::ParallelDescriptor::ReduceRealSum(totals.data(), 11);
    if (s.options.rz_closed_reciprocal_spatial_endpoint_trial)
    {
        // Radiation's old-minus-new transfer feeds material AND stationary
        // external walls, not another material kick/work. Reports are already
        // global. Never add them on each rank
        // before the MPI sum (including an empty rank).
        totals[4] += PhysConst::c * result.rz_axial_wall_momentum;
        totals[5] += PhysConst::c *
            (std::abs(result.rz_axial_wall_momentum_by_side[0]) +
             std::abs(result.rz_axial_wall_momentum_by_side[1]));
    }
    std::array<amrex::Real, 3> allowance{};
    bool valid = detail::AcceptedRZGlobalNativeInventory(
        totals, local_cells, local_faces, amrex::ParallelDescriptor::NProcs(),
        result.energy_roundoff_bound, s.options.energy_tolerance, allowance);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}
#endif

// All device kernels in this translation unit are namespace functions; no
// extended lambda may depend on a private nested implementation type.
Trial
Evaluate (SourceContext& s, Vector const& nodes, amrex::MultiFab const& velocity)
{
    Trial trial;
    trial.reason = "invalid native thermal temperature trial";
    bool valid = nodes.size() == s.old_nodes.size();
    for (std::size_t n = 0; n < nodes.size(); ++n)
    {
        auto const temperature = PhysicalNodeTemperature(s, nodes, n);
        bool const vacant = s.options.rz_single_selected_table_vacuum &&
                            s.composite_table && s.active_nodes[n] == 0;
        valid = valid && std::isfinite(temperature) &&
            (vacant ? temperature >= 0 && temperature == s.old_nodes[n]
                    : (temperature >= s.eos.minimumTemperature() &&
                       temperature <= s.eos.maximumTemperature()));
        if (!s.active_nodes[n])
        {
            valid = valid && temperature == s.old_nodes[n];
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        return trial;
    }
    auto temperature = Copy(s.old_temperature);
    amrex::MultiFab increment;
    if (s.options.rz_centered_ideal_temperature_trial)
    {
        Vector physical(nodes.size());
        for (std::size_t n = 0; n < nodes.size(); ++n)
        { physical[n] = PhysicalNodeTemperature(s, nodes, n); }
        s.node_layout.Scatter(physical, s.geometry, temperature);
        increment = Copy(s.old_temperature);
        increment.setVal(0);
        s.node_layout.Scatter(nodes, s.geometry, increment);
    }
    else
    {
        s.node_layout.Scatter(nodes, s.geometry, temperature);
    }
    auto const host_temperature = Host(temperature), rho = Host(s.density), frame = Host(velocity);
    auto const coefficients = Host(s.opacity);
    amrex::MultiFab rz_weights;
    if (s.rz_reciprocal)
    {
        rz_weights = Host(s.corner_weights);
    }
    amrex::MultiFab cell_temperature(s.old_diffusion.boxArray(), s.old_diffusion.DistributionMap(),
                                     1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    for (amrex::MFIter it(cell_temperature); it.isValid(); ++it)
    {
        auto const t = host_temperature.const_array(it), r = rho.const_array(it);
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            if (s.rz_reciprocal)
            {
                amrex::Real reference = 0, weighted_difference = 0;
                bool occupied = false;
                for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                {
                    auto node = cell();
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        node[d] += (corner >> d) & 1;
                    }
                    if (rz_weights[it](cell(), corner) > 0)
                    {
                        reference = occupied ? std::min(reference, t(node)) : t(node);
                        occupied = true;
                    }
                }
                if (occupied)
                {
                    for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                    {
                        auto node = cell();
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            node[d] += (corner >> d) & 1;
                        }
                        weighted_difference += rz_weights[it](cell(), corner) *
                                               (t(node) - reference);
                    }
                }
                else
                {
                    valid = valid && coefficients[it](cell()) == 0 &&
                            s.diffusion_flags[it](cell()) == 0;
                }
                cell_temperature[it](cell()) = reference + weighted_difference;
                valid = valid && std::isfinite(cell_temperature[it](cell()));
                continue;
            }
            auto const index = amrex::lbound(amrex::Box(cell(), cell()));
            auto const state = detail::GatherHybridCellMaterialState(
                index.x, index.y, index.z, s.options.minimum_electron_density, s.floor, s.eos, r, t,
                {}, s.geometry.ProbLoArray(), s.geometry.CellSizeArray(),
                amrex::lbound(s.geometry.Domain()));
            valid = valid && state.valid;
            if (state.heat_capacity == 0)
            {
                valid = valid && s.options.allow_vacuum_noop && state.electron_density == 0 &&
                        coefficients[it](cell()) == 0 && s.diffusion_flags[it](cell()) == 0;
            }
            else
            {
                valid = valid && state.electron_density > s.options.minimum_electron_density;
            }
            amrex::Real reference_temperature = 0, centered_temperature = 0;
            bool have_reference = false;
            if (state.heat_capacity > 0)
            {
                // The absolute weighted-sum/division mean need not preserve
                // even four or eight identical nodal temperatures bitwise.
                // Center the SAME physical Cv-weighted mean at the coldest real
                // active corner. Nonnegative differences avoid cancellation at
                // arbitrarily large hot/cold capacity/temperature contrasts;
                // homogeneous LTE is exact without a zero-source/state repair.
                // Frozen old-state caloric allocation weights remain unchanged.
                for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                {
                    auto node = cell();
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        node[d] += (corner >> d) & 1;
                    }
                    if (rho[it](node, 0) <= PhysConst::q_e * s.options.minimum_electron_density)
                    {
                        continue;
                    }
                    auto const node_temperature = host_temperature[it](node, 0);
                    if (!have_reference)
                    {
                        reference_temperature = node_temperature;
                        have_reference = true;
                    }
                    else
                    {
                        reference_temperature = std::min(reference_temperature, node_temperature);
                    }
                }
                for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                {
                    auto node = cell();
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        node[d] += (corner >> d) & 1;
                    }
                    if (rho[it](node, 0) <= PhysConst::q_e * s.options.minimum_electron_density)
                    {
                        continue;
                    }
                    auto const node_temperature = host_temperature[it](node, 0);
                    auto const caloric = s.eos.stateFromChargeDensityTemperature(
                        std::max(rho[it](node, 0), PhysConst::q_e * s.floor), node_temperature);
                    auto const corner_volume = detail::HybridCellCornerVolume(
                        index.x, index.y, index.z, corner & 1, (corner >> 1) & 1, (corner >> 2) & 1,
                        s.geometry.ProbLoArray(), s.geometry.CellSizeArray(),
                        amrex::lbound(s.geometry.Domain()));
                    centered_temperature += caloric.heat_capacity_density * corner_volume *
                                            (node_temperature - reference_temperature);
                }
                valid = valid && have_reference && std::isfinite(centered_temperature);
                reference_temperature += centered_temperature / state.heat_capacity;
            }
            cell_temperature[it](cell(), 0) = reference_temperature;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        return trial;
    }
    auto& candidate = trial.candidate;
    bool const reciprocal =
        s.options.caloric_backend == NativeThermalCaloricBackend::ReciprocalNodalEmission;
    NativeThermalCornerEmission const emission{&s.corner_weights, &temperature,
        s.options.reciprocal_emission_rays_per_corner,
        s.options.rz_centered_ideal_temperature_trial ? &s.old_temperature : nullptr,
        s.options.rz_centered_ideal_temperature_trial ? &increment : nullptr};
    amrex::MultiFab debit, scale;
    amrex::MultiFab projected(s.old_diffusion.boxArray(), s.old_diffusion.DistributionMap(), 1, 0,
                              amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    if (s.backend)
    {
        if (!(*s.backend)(temperature, cell_temperature, velocity, s.trial_work,
                          reciprocal ? &emission : nullptr, candidate, projected, trial.reason))
        {
            return trial;
        }
        valid = candidate.photons && candidate.diffusion_radiation.ok() &&
                candidate.diffusion_radiation.nComp() == 4 && candidate.physical_debit.ok() &&
                candidate.raw_debit.ok() && candidate.operand_scale.ok() &&
                candidate.storage_roundoff_bound.ok() && projected.ok() && projected.nComp() == 1;
        for (auto const* field :
             {&candidate.diffusion_radiation, &candidate.physical_debit, &candidate.raw_debit,
              &candidate.operand_scale, &candidate.storage_roundoff_bound})
        {
            valid = valid && field->nComp() == 4 &&
                    field->boxArray() == s.old_diffusion.boxArray() &&
                    field->DistributionMap() == s.old_diffusion.DistributionMap();
        }
        valid = valid && projected.boxArray() == s.old_diffusion.boxArray() &&
                projected.DistributionMap() == s.old_diffusion.DistributionMap();
        if (reciprocal)
        {
            auto const& offset = candidate.reciprocal_corner_heat_offset;
            valid = valid && offset.ok() && offset.nComp() == (1 << AMREX_SPACEDIM) &&
                    offset.boxArray() == s.old_diffusion.boxArray() &&
                    offset.DistributionMap() == s.old_diffusion.DistributionMap();
        }
        if (s.rz_reciprocal)
        {
            valid = valid && candidate.rz_packet_debit.ok() &&
                candidate.rz_packet_debit.nComp() == 4 &&
                candidate.rz_packet_debit.boxArray() == s.old_diffusion.boxArray() &&
                candidate.rz_packet_debit.DistributionMap() ==
                    s.old_diffusion.DistributionMap() &&
                candidate.rz_work_excluded_packet_debit.ok() &&
                candidate.rz_work_excluded_packet_debit.nComp() == 4 &&
                candidate.rz_work_excluded_packet_debit.boxArray() ==
                    s.old_diffusion.boxArray() &&
                candidate.rz_work_excluded_packet_debit.DistributionMap() ==
                    s.old_diffusion.DistributionMap() &&
                candidate.rz_diffusion_work_input.ok() &&
                candidate.rz_diffusion_work_input.nComp() == 1 &&
                candidate.rz_diffusion_work_input.boxArray() ==
                    s.old_diffusion.boxArray() &&
                candidate.rz_diffusion_work_input.DistributionMap() ==
                    s.old_diffusion.DistributionMap() &&
                candidate.rz_leakage && candidate.rz_leakage->valid &&
                candidate.rz_leakage->donor_radiation_debit_correction.ok() &&
                candidate.rz_leakage->donor_radiation_debit_correction.nComp() == 4 &&
                candidate.rz_leakage->donor_radiation_debit_correction.boxArray() ==
                    s.old_diffusion.boxArray() &&
                candidate.rz_leakage->donor_radiation_debit_correction.DistributionMap() ==
                    s.old_diffusion.DistributionMap() &&
                candidate.rz_outward_pressure_stress.ok() &&
                candidate.rz_outward_pressure_stress.nComp() == 3 &&
                candidate.rz_theta_connection.ok() &&
                candidate.rz_theta_connection.nComp() == 1;
            if (candidate.rz_leakage)
            {
                for (auto const* field :
                     {&candidate.rz_leakage->donor_operand_scale,
                      &candidate.rz_leakage->donor_roundoff_bound})
                {
                    valid = valid && field->ok() && field->nComp() == 4 &&
                        field->boxArray() == s.old_diffusion.boxArray() &&
                        field->DistributionMap() == s.old_diffusion.DistributionMap();
                }
            }
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
            {
                auto const face_boxes = amrex::convert(
                    s.old_diffusion.boxArray(),
                    amrex::IntVect::TheDimensionVector(direction));
                auto const* energy = candidate.rz_face_lab_energy_current[direction].get();
                auto const* stress = candidate.rz_face_lab_stress[direction].get();
                auto const* leakage = candidate.rz_face_leaked_energy[direction].get();
                valid = valid && energy && stress && leakage &&
                    energy->nComp() == 1 && stress->nComp() == 3 &&
                    leakage->nComp() == 1 &&
                    energy->boxArray() == face_boxes && stress->boxArray() == face_boxes &&
                    leakage->boxArray() == face_boxes &&
                    energy->DistributionMap() == s.old_diffusion.DistributionMap() &&
                    stress->DistributionMap() == s.old_diffusion.DistributionMap() &&
                    leakage->DistributionMap() == s.old_diffusion.DistributionMap();
            }
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            trial.reason = "invalid caller-private thermal radiation accounts";
            return trial;
        }
        if (s.options.rz_closed_reciprocal_endpoint_trial)
        {
            bool zero = candidate.rz_leakage->staged_births == 0 &&
                        candidate.rz_leakage->face_accounts.empty() &&
                        std::isfinite(candidate.rz_axial_wall_momentum) &&
                        candidate.rz_axial_wall_momentum == 0;
            for (auto const wall : candidate.rz_axial_wall_momentum_by_side)
            {
                zero = std::isfinite(wall) && wall == 0 && zero;
            }
            for (auto const* f : {&candidate.diffusion_radiation,
                 &candidate.rz_diffusion_work_input, &candidate.rz_outward_pressure_stress,
                 &candidate.rz_theta_connection,
                 &candidate.rz_leakage->donor_radiation_debit_correction,
                 &candidate.rz_leakage->donor_operand_scale,
                 &candidate.rz_leakage->donor_roundoff_bound})
            {
                auto const finite = f->is_finite();
                auto const norm = f->norm0(0, f->nComp(), f->nGrowVect());
                zero = finite && norm == 0 && zero;
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                for (auto const* f : {candidate.rz_face_lab_energy_current[d].get(),
                     candidate.rz_face_lab_stress[d].get(), candidate.rz_face_leaked_energy[d].get()})
                {
                    auto const finite = f->is_finite();
                    auto const norm = f->norm0(0, f->nComp(), f->nGrowVect());
                    zero = finite && norm == 0 && zero;
                }
            }
            amrex::ParallelDescriptor::ReduceBoolAnd(zero);
            if (!zero) { trial.reason = "nonzero D in closed reciprocal endpoint"; return trial; }
        }
        if (!candidate.photons->ValidateSnapshotFor(*s.live))
        {
            trial.reason = "invalid caller-private radiation trial provenance";
            return trial;
        }
        for (auto const* field :
             {&candidate.diffusion_radiation, &candidate.physical_debit, &candidate.raw_debit,
              &candidate.operand_scale, &candidate.storage_roundoff_bound, &projected})
        {
            bool const finite = field->is_finite(0, field->nComp(), 0);
            valid = finite && valid;
        }
        if (s.rz_reciprocal)
        {
            for (auto const* field : {&candidate.rz_packet_debit,
                                      &candidate.rz_work_excluded_packet_debit,
                                      &candidate.rz_diffusion_work_input,
                                      &candidate.rz_leakage->donor_radiation_debit_correction,
                                      &candidate.rz_leakage->donor_operand_scale,
                                      &candidate.rz_leakage->donor_roundoff_bound,
                                      &candidate.rz_outward_pressure_stress,
                                      &candidate.rz_theta_connection})
            {
                valid = field->is_finite(0, field->nComp(), 0) && valid;
            }
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
            {
                for (auto const* field :
                     {candidate.rz_face_lab_energy_current[direction].get(),
                      candidate.rz_face_lab_stress[direction].get(),
                      candidate.rz_face_leaked_energy[direction].get()})
                {
                    valid = field->is_finite(0, field->nComp(), 0) && valid;
                }
            }
        }
        for (int d = 0; d < 4; ++d)
        {
            auto const scale_minimum = candidate.operand_scale.min(d);
            auto const bound_minimum = candidate.storage_roundoff_bound.min(d);
            valid = scale_minimum >= 0 && bound_minimum >= 0 && valid;
        }
        auto const minimum_energy = candidate.diffusion_radiation.min(0);
        valid = minimum_energy >= 0 && valid;
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            trial.reason = "nonphysical caller-private radiation accounts";
            return trial;
        }
        debit = Host(candidate.physical_debit);
        scale = Host(candidate.operand_scale);
        projected = Host(projected);
    }
    else
    {
        trial.candidate.photons = std::make_unique<PhotonPacketTransaction>(*s.live, *s.endpoint);
        auto packet = s.packets.Replay(cell_temperature, velocity, *trial.candidate.photons);
        if (!packet.valid)
        {
            trial.reason = packet.reason;
            return trial;
        }
        debit = Host(packet.radiation_debit);
        scale = Host(packet.operand_scale);
        auto raw = Host(packet.radiation_debit);
        for (amrex::MFIter it(projected); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                auto energy = debit[it](cell(), 0);
                for (int d = 0; d < 3; ++d)
                {
                    energy -= frame[it](cell(), d) / PhysConst::c * debit[it](cell(), d + 1);
                }
                projected[it](cell(), 0) = energy;
            }
        }
        auto diffusion = Host(s.old_diffusion);
        amrex::MultiFab storage(s.old_diffusion.boxArray(), s.old_diffusion.DistributionMap(), 4, 0,
                                amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        storage.setVal(0);
        for (amrex::MFIter it(diffusion); it.isValid(); ++it)
        {
            auto const flags = s.diffusion_flags.const_array(it), v = frame.const_array(it);
            auto const chi = coefficients.const_array(it);
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                if (flags(cell(), 0) == 0)
                {
                    continue;
                }
                FourVector old{};
                amrex::GpuArray<amrex::Real, 3> beta{};
                for (int d = 0; d < 4; ++d)
                {
                    old[d] = diffusion[it](cell(), d);
                }
                for (int d = 0; d < 3; ++d)
                {
                    beta[d] = v(cell(), d) / PhysConst::c;
                }
                auto const t = cell_temperature[it](cell(), 0), t2 = t * t;
                auto const bath = s.volume * s.options.packets.radiation_constant * t2 * t2;
                auto const source =
                    TryPureGrayAbsorptionSource(old, beta, PhysConst::c * s.dt * chi(cell()), bath);
                valid = valid && source.valid;
                if (!source.valid)
                {
                    continue;
                }
                projected[it](cell(), 0) += source.physical_energy_minus_work;
                for (int d = 0; d < 4; ++d)
                {
                    diffusion[it](cell(), d) = source.radiation[d];
                    debit[it](cell(), d) += source.physical_transfer[d];
                    raw[it](cell(), d) += source.raw_transfer[d];
                    scale[it](cell(), d) += std::abs(old[d]) + std::abs(source.radiation[d]) +
                                            std::abs(source.physical_transfer[d]);
                    storage[it](cell(), d) = source.storage_bound[d];
                }
            }
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            trial.reason = "invalid D comoving dipole or pure absorption source";
            return trial;
        }
        candidate.diffusion_radiation.define(diffusion.boxArray(), diffusion.DistributionMap(), 4,
                                             0);
        candidate.physical_debit.define(debit.boxArray(), debit.DistributionMap(), 4, 0);
        candidate.raw_debit.define(raw.boxArray(), raw.DistributionMap(), 4, 0);
        candidate.operand_scale.define(scale.boxArray(), scale.DistributionMap(), 4, 0);
        candidate.storage_roundoff_bound.define(storage.boxArray(), storage.DistributionMap(), 4,
                                                0);
        amrex::htod_memcpy(candidate.diffusion_radiation, diffusion);
        amrex::htod_memcpy(candidate.physical_debit, debit);
        amrex::htod_memcpy(candidate.raw_debit, raw);
        amrex::htod_memcpy(candidate.operand_scale, scale);
        amrex::htod_memcpy(candidate.storage_roundoff_bound, storage);
    }
    amrex::MultiFab impulse(debit.boxArray(), debit.DistributionMap(), 3, 0,
                            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    for (amrex::MFIter it(impulse); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            for (int d = 0; d < 3; ++d)
            {
                impulse[it](cell(), d) = debit[it](cell(), d + 1) / PhysConst::c;
            }
        }
    }
    candidate.material_stage = std::make_unique<ParticleImpulseTransaction>();
    auto const assignment = s.options.cartesian_native_nodal_force
                                ? ParticleImpulseAssignment::ConservativeNodalCellAverage
                            : s.options.UsesRZNativeNodalForce()
                                ? s.options.RZForceAssignment()
                            : s.rz_reciprocal
                                ? ParticleImpulseAssignment::RZAngularConservative
                                : ParticleImpulseAssignment::NearestCell;
    bool const staged = candidate.material_stage->Stage(*s.material, s.path, impulse, true, assignment);
    if (!staged ||
        candidate.material_stage->MaximumBetaSquared() >
            (s.rz_reciprocal ? 1.e-6_rt : 1.e-4_rt))
    {
        trial.reason = s.options.UsesRZNativeNodalForce()
            ? "invalid or high-beta native nodal-support thermal kick"
            : "invalid or high-beta total native thermal kick";
        return trial;
    }
    bool const overlapping = s.options.UsesOverlappingMaterialForce();
    auto const work = Host(overlapping ? candidate.material_stage->RequestedWork()
                                      : candidate.material_stage->ActualWork());
    auto const carry = Host(candidate.material_stage->EnergyCarryChange());
    amrex::MultiFab heat(debit.boxArray(), debit.DistributionMap(), 1, 1,
                         amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    heat.setVal(0);
    amrex::MultiFab nodal_allocation(s.old_temperature.boxArray(),
                                     s.old_temperature.DistributionMap(), 2, 0);
    nodal_allocation.setVal(0);
    auto allocation = Host(nodal_allocation);
    amrex::MultiFab weights, offset;
    if (reciprocal)
    {
        weights = Host(s.corner_weights);
        offset = Host(candidate.reciprocal_corner_heat_offset);
        valid = offset.is_finite();
    }
    for (amrex::MFIter it(heat); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            amrex::Real secant_projection = 0;
            for (int d = 0; d < 3; ++d)
            {
                secant_projection += frame[it](cell(), d) / PhysConst::c * debit[it](cell(), d + 1);
            }
            // Algebraically D-(actual_work+delta_carry), evaluated in the
            // retained energy-minus-work basis to protect weak thermal
            // exchange. The independent raw E/P ledger below remains
            // authoritative.
            // An overlapping force has source-origin work but center-owned
            // actual kinetic work/carry. Subtract the former here; independent
            // GLOBAL finite-work/carry accounting remains authoritative below.
            // Subtracting a different cell's kinetic work would create heat
            // in a true vacuum and remove work from the occupied edge support.
            auto const request = projected[it](cell(), 0) +
                (overlapping ? secant_projection - work[it](cell(), 0)
                             : (secant_projection - work[it](cell(), 0)) - carry[it](cell(), 0));
            heat[it](cell(), 0) = request;
            if (request == 0 && !reciprocal)
            {
                continue;
            }
            amrex::Real capacity = 0;
            std::array<amrex::IntVect, 1 << AMREX_SPACEDIM> corners{};
            for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
            {
                auto node = cell();
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    node[d] += (corner >> d) & 1;
                }
                corners[corner] = node;
                capacity += reciprocal ? weights[it](cell(), corner)
                            : request < 0 && s.eos.isIdealGas() &&
                                    s.model->transportsElectronInternalEnergyAtRawDensity()
                                ? s.node_state[it](node, 3)
                                : s.node_state[it](node, 1);
            }
            valid = valid && (capacity > 0 || (reciprocal && request == 0)) &&
                    std::isfinite(request + capacity);
            auto const arithmetic = 128 * std::numeric_limits<amrex::Real>::epsilon() *
                                    (scale[it](cell(), 0) + std::abs(work[it](cell(), 0)) +
                                     std::abs(carry[it](cell(), 0)));
            if (!valid)
            {
                continue;
            }
            for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
            {
                auto const& node = corners[corner];
                auto const weight = reciprocal ? weights[it](cell(), corner)
                                    : request < 0 && s.eos.isIdealGas() &&
                                            s.model->transportsElectronInternalEnergyAtRawDensity()
                                        ? s.node_state[it](node, 3)
                                        : s.node_state[it](node, 1);
                if (reciprocal)
                {
                    auto const origin = offset[it](cell(), corner);
                    auto const measure = s.rz_reciprocal ? 1._rt : s.volume;
                    allocation[it](node, 0) += (request * weight + origin) / measure;
                    allocation[it](node, 1) +=
                        (arithmetic * weight +
                         128 * std::numeric_limits<amrex::Real>::epsilon() * std::abs(origin)) /
                        measure;
                }
                else
                {
                    allocation[it](node, 0) += request * weight / capacity / s.volume;
                    allocation[it](node, 1) += arithmetic * weight / capacity / s.volume;
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        trial.reason = "invalid native caloric source allocation";
        return trial;
    }
    amrex::htod_memcpy(nodal_allocation, allocation);
    amrex::Gpu::streamSynchronize();
    // Sum only contributions from this rank's disjoint valid cells. Canonical
    // duplicate-node synchronization follows the sum; no global nodal array
    // or vector-valued collective is allocated, including on empty ranks.
    nodal_allocation.SumBoundary(s.geometry.periodicity(), true);
    nodal_allocation.OverrideSync(s.geometry.periodicity());
    auto const accumulated = Host(nodal_allocation);
    trial.heat = s.node_layout.GatherHost(accumulated, 0);
    trial.roundoff = s.node_layout.GatherHost(accumulated, 1);
    trial.residual.resize(nodes.size());
    trial.norm = 0;
    auto const rz_measures = s.rz_reciprocal && !s.composite_table
                                 ? s.node_layout.GatherHost(s.node_state, 4)
                                 : Vector{};
    auto const* table_owners =
        s.composite_table ? &s.table_caloric->Owners() : nullptr;
    for (std::size_t n = 0; n < nodes.size(); ++n)
    {
        if (!s.active_nodes[n])
        {
            trial.residual[n] = 0;
            valid = valid && trial.heat[n] == 0 &&
                    PhysicalNodeTemperature(s, nodes, n) == s.old_nodes[n];
            continue;
        }
        amrex::Real change_joules = 0, source_scale = 0;
        if (s.composite_table)
        {
            auto const response = detail::EvaluateCompositeTableCaloricNode(
                s.eos, (*table_owners)[n].caloric, trial.heat[n], &nodes[n]);
            valid = valid && response.Valid();
            change_joules = response.realized_joules;
            trial.residual[n] = response.residual_joules;
            auto certified_bound = response.arithmetic_bound_joules;
            auto const inverse = detail::EvaluateCompositeTableCaloricNode(
                s.eos, (*table_owners)[n].caloric, trial.heat[n]);
            if (inverse.Valid() && inverse.converged && inverse.temperature == nodes[n])
            {
                // At a finite table inverse image, the exact prescribed
                // temperature can be unchanged while Q lies between adjacent
                // representable caloric values. Use only the provider's
                // certified image width for that identical bitwise T.
                certified_bound = std::max(certified_bound,
                                           inverse.arithmetic_bound_joules);
            }
            trial.roundoff[n] += certified_bound;
            source_scale = std::abs(trial.heat[n]);
        }
        else if (s.options.rz_centered_ideal_temperature_trial)
        {
            // The frozen ideal owned-node capacity is constant. Multiplying
            // the increment directly avoids subtracting two hot native T
            // values before the coupled source root has converged.
            change_joules = s.old_capacity[n] * nodes[n];
            trial.residual[n] =
                change_joules - s.old_corner_energy_offset[n] - trial.heat[n];
            auto const state = s.eos.stateFromChargeDensityTemperature(
                s.charge[n], PhysicalNodeTemperature(s, nodes, n));
            trial.roundoff[n] += 64 * std::numeric_limits<amrex::Real>::epsilon() *
                (std::abs(state.internal_energy_density * rz_measures[n]) +
                 std::abs(s.old_energy[n]));
            source_scale = std::abs(trial.heat[n] + s.old_corner_energy_offset[n]);
        }
        else
        {
            auto const state = s.eos.stateFromChargeDensityTemperature(s.charge[n], nodes[n]);
            auto const change = s.options.reciprocal_composite_old_temperature
                                    ? warpx::hybrid::CenteredOwnedNodalCaloricDifference(
                                          s.eos, s.charge[n], s.old_nodes[n], nodes[n])
                                    : s.eos.internalEnergyDensityDifference(
                                          s.charge[n], {}, s.old_nodes[n], nodes[n]);
            auto const measure = s.rz_reciprocal ? rz_measures[n] : 1._rt;
            change_joules = change * measure;
            trial.residual[n] =
                change_joules - s.old_corner_energy_offset[n] - trial.heat[n];
            trial.roundoff[n] += 64 * std::numeric_limits<amrex::Real>::epsilon() *
                                 (std::abs(state.internal_energy_density * measure) +
                                  std::abs(s.old_energy[n]));
            source_scale = std::abs(trial.heat[n] + s.old_corner_energy_offset[n]);
        }
        // The accepted trial is followed by an independent caloric inverse.
        // Reciprocal emission evaluates a nodal T^4 source, so even an allowed
        // inverse-image shift can amplify its final source-equation defect.
        // Solve this NEW model more accurately internally, without weakening
        // the physical gates or shrinking the independently derived arithmetic
        // floor. The legacy trial-Cv source keeps its established criterion.
        auto const internal_tolerance =
            s.options.material_tolerance * (reciprocal ? .01_rt : 1._rt);
        auto const bound = internal_tolerance * std::max(std::abs(change_joules), source_scale) +
                           trial.roundoff[n];
        auto const defect = std::abs(trial.residual[n]);
        trial.norm =
            std::max(trial.norm, bound > 0     ? defect / bound
                                 : defect == 0 ? 0
                                               : std::numeric_limits<amrex::Real>::infinity());
        valid = valid && std::isfinite(change_joules + trial.residual[n] + bound);
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        trial.reason = "nonfinite native caloric residual";
        return trial;
    }
    amrex::ParallelDescriptor::ReduceRealMax(trial.norm);
    candidate.requested_heat.define(heat.boxArray(), heat.DistributionMap(), 1, 1);
    amrex::htod_memcpy(candidate.requested_heat, heat);
    candidate.requested_heat.FillBoundary(s.geometry.periodicity());
    if (reciprocal)
    {
        candidate.nodal_requested_heat_density.define(nodal_allocation.boxArray(),
                                                      nodal_allocation.DistributionMap(), 1, 1);
        candidate.nodal_requested_heat_density.setVal(0);
        if (s.rz_reciprocal)
        {
            auto owner_heat = Host(candidate.nodal_requested_heat_density);
            for (std::size_t n = 0; n < s.node_layout.owned.size(); ++n)
            {
                auto const& owner = s.node_layout.owned[n];
                owner_heat[owner.fab](owner.index) = trial.heat[n];
            }
            amrex::htod_memcpy(candidate.nodal_requested_heat_density, owner_heat);
        }
        else
        {
            amrex::MultiFab::Copy(candidate.nodal_requested_heat_density, nodal_allocation,
                                  0, 0, 1, 0);
            candidate.nodal_requested_heat_density.FillBoundary(s.geometry.periodicity());
        }
    }
    candidate.nodal_temperature = std::move(temperature);
    candidate.maximum_caloric_residual = trial.norm;
    amrex::Gpu::streamSynchronize();
    trial.valid = true;
    return trial;
}

// Matrix-free, locally scaled Newton/GMRES. The 1 K scaling used only for an
// exactly cold zero-residual coordinate changes no physical acceptance bound.
bool
NewtonDirection (SourceContext& s, Vector const& nodes, Trial const& base,
                 amrex::MultiFab const& frame, Vector& direction, std::string& failure_reason)
{
    auto const size = nodes.size();
    Vector scale(size), energy_scale(size), rhs(size);
    for (std::size_t n = 0; n < size; ++n)
    {
        if (!s.active_nodes[n])
        {
            scale[n] = 1;
            energy_scale[n] = 1;
            rhs[n] = 0;
            continue;
        }
        scale[n] = CaloricTemperatureScale(s, nodes, base, n);
        energy_scale[n] = s.old_capacity[n] * scale[n];
        rhs[n] = -base.residual[n] / energy_scale[n];
    }
    auto const norm = std::sqrt(Dot(rhs, rhs));
    if (!(norm > 0) || !std::isfinite(norm))
    {
        failure_reason = "zero/nonfinite scaled Newton right-hand side";
        return false;
    }
    int const limit = static_cast<int>(std::min(
        s.node_layout.global_size, static_cast<amrex::Long>(s.options.max_linear_iterations)));
    std::vector<Vector> basis;
    basis.push_back(rhs);
    s.peak_basis_vectors = std::max(s.peak_basis_vectors, 1);
    for (auto& value : basis[0])
    {
        value /= norm;
    }
    std::vector<Vector> matrix(static_cast<std::size_t>(limit + 1), Vector(limit, 0));
    Vector cosine(limit), sine(limit), projection(static_cast<std::size_t>(limit + 1), 0);
    projection[0] = norm;
    int used = 0;
    for (int column = 0; column < limit; ++column)
    {
        // A centered derivative has quadratic truncation error. Its larger
        // cbrt(epsilon) displacement resolves stiff source differences more
        // accurately than a one-sided sqrt(epsilon) cancellation. Domain
        // boundaries retain a one-sided fallback, not an extrapolated EOS.
        auto const fraction = s.options.rz_centered_ideal_temperature_trial
            ? std::cbrt(std::numeric_limits<amrex::Real>::epsilon())
            : std::sqrt(std::numeric_limits<amrex::Real>::epsilon());
        Trial positive, negative;
        for (int sign : {1, -1})
        {
            Vector perturbed = nodes;
            for (std::size_t n = 0; n < size; ++n)
            {
                perturbed[n] += sign * fraction * scale[n] * basis[column][n];
            }
            auto probe = Evaluate(s, perturbed, frame);
            if (sign > 0) { positive = std::move(probe); }
            else { negative = std::move(probe); }
            if (!s.options.rz_centered_ideal_temperature_trial && positive.valid)
            { break; }
        }
        if (!positive.valid && !negative.valid)
        {
            failure_reason = "directional thermal residual rejected: " + negative.reason;
            return false;
        }
        Vector value(size);
        for (std::size_t n = 0; n < size; ++n)
        {
            auto const difference = positive.valid && negative.valid
                ? (positive.residual[n] - negative.residual[n]) / (2 * fraction)
                : positive.valid
                    ? (positive.residual[n] - base.residual[n]) / fraction
                    : (base.residual[n] - negative.residual[n]) / fraction;
            value[n] = difference / energy_scale[n];
        }
        // Reorthogonalize the same Arnoldi column to avoid losing the small
        // caloric modes next to a very stiff exchange eigenmode.
        int const orthogonalization_passes =
            s.options.rz_centered_ideal_temperature_trial ? 2 : 1;
        for (int pass = 0; pass < orthogonalization_passes; ++pass)
        {
            for (int row = 0; row <= column; ++row)
            {
                auto const overlap = Dot(value, basis[row]);
                matrix[row][column] += overlap;
                for (std::size_t n = 0; n < size; ++n)
                {
                    value[n] -= overlap * basis[row][n];
                }
            }
        }
        matrix[column + 1][column] = std::sqrt(Dot(value, value));
        auto const length = matrix[column + 1][column];
        if (length > 0)
        {
            for (auto& component : value)
            {
                component /= length;
            }
            basis.push_back(std::move(value));
            s.peak_basis_vectors = std::max(s.peak_basis_vectors, static_cast<int>(basis.size()));
        }
        for (int row = 0; row < column; ++row)
        {
            auto const upper = matrix[row][column], lower = matrix[row + 1][column];
            matrix[row][column] = cosine[row] * upper + sine[row] * lower;
            matrix[row + 1][column] = -sine[row] * upper + cosine[row] * lower;
        }
        auto const radius = std::hypot(matrix[column][column], matrix[column + 1][column]);
        if (!(radius > 0) || !std::isfinite(radius))
        {
            failure_reason = "zero/nonfinite Arnoldi column " + std::to_string(column);
            return false;
        }
        cosine[column] = matrix[column][column] / radius;
        sine[column] = matrix[column + 1][column] / radius;
        matrix[column][column] = radius;
        matrix[column + 1][column] = 0;
        projection[column + 1] = -sine[column] * projection[column];
        projection[column] *= cosine[column];
        used = column + 1;
        if (std::abs(projection[column + 1]) <= 0.01_rt * norm || length == 0)
        {
            break;
        }
    }
    Vector coefficients(static_cast<std::size_t>(used), 0);
    for (int row = used - 1; row >= 0; --row)
    {
        auto value = projection[row];
        for (int column = row + 1; column < used; ++column)
        {
            value -= matrix[row][column] * coefficients[column];
        }
        coefficients[row] = value / matrix[row][row];
    }
    direction.assign(size, 0);
    for (int column = 0; column < used; ++column)
    {
        for (std::size_t n = 0; n < size; ++n)
        {
            direction[n] += scale[n] * coefficients[column] * basis[column][n];
        }
    }
    return true;
}

bool
CompositeTableInverseDirection (SourceContext const& s, Vector const& nodes,
                                Trial const& base, Vector& direction,
                                bool& unaffordable)
{
    direction.assign(nodes.size(), 0);
    unaffordable = false;
    bool valid = s.composite_table && s.table_caloric &&
                 base.heat.size() == nodes.size() &&
                 s.table_caloric->Owners().size() == nodes.size();
    if (valid)
    {
        auto const& owners = s.table_caloric->Owners();
        for (std::size_t n = 0; n < nodes.size(); ++n)
        {
            if (!s.active_nodes[n]) { continue; }
            auto const response = detail::EvaluateCompositeTableCaloricNode(
                s.eos, owners[n].caloric, base.heat[n]);
            unaffordable = unaffordable ||
                           response.failure ==
                               detail::CompositeTableCaloricFailure::Unaffordable;
            valid = valid && response.Valid() && response.converged &&
                    std::isfinite(response.temperature);
            direction[n] = response.temperature - nodes[n];
        }
    }
    amrex::ParallelDescriptor::ReduceBoolOr(unaffordable);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    auto const magnitude = valid ? Dot(direction, direction) : 0;
    return valid && std::isfinite(magnitude) && magnitude > 0;
}

} // namespace

NativeResolvedThermalSourceResult
TryAdvanceNativeResolvedThermalSource (
    WarpXParticleContainer& live, PhotonPacketTransaction const& endpoint,
    ParticleImpulseMaterial& material, std::string const& path, amrex::MultiFab const& diffusion,
    amrex::iMultiFab const& mask, amrex::MultiFab const& temperature,
    amrex::MultiFab const& opacity, amrex::MultiFab const& rate, HybridPICModel const& model,
    amrex::Geometry const& geometry, amrex::Real dt,
    NativeResolvedThermalSourceOptions const& options,
    NativeResolvedThermalTrialBackend const* backend)
{
    NativeResolvedThermalSourceResult rejected;
    rejected.reason = "invalid or unsupported native resolved thermal source";
    int backend_minimum = backend ? 1 : 0, backend_maximum = backend_minimum;
    amrex::ParallelDescriptor::ReduceIntMin(backend_minimum);
    amrex::ParallelDescriptor::ReduceIntMax(backend_maximum);
    if (backend_minimum != backend_maximum)
    {
        return rejected;
    }
    bool callable = !backend || static_cast<bool>(*backend);
    amrex::ParallelDescriptor::ReduceBoolAnd(callable);
    if (!callable)
    {
        return rejected;
    }
    if (!endpoint.ValidateSnapshotFor(live))
    {
        rejected.reason = "invalid native photon endpoint provenance";
        return rejected;
    }
    auto const eos = model.electronThermodynamicsExecutor();
    bool const builtin_caloric =
        (eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) &&
        eos.m_num_materials == 0 && std::isfinite(eos.m_gamma) && eos.m_gamma > 1;
    bool composite_table_caloric = false;
    bool const same_controls = SameSourceControls(options);
    if (!same_controls) { return rejected; }
    bool selected_table = false;
    if (options.rz_single_selected_table_trial)
    {
        selected_table = options.rz_single_selected_table_vacuum
            ? detail::HasNativeSingleSelectedTableMaterial0OrExactVacuum(model, geometry)
            : detail::HasNativeSingleSelectedTableMaterial0(model, geometry);
    }
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
    composite_table_caloric =
        eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
        ((!eos.m_use_resolved_material_selection && !options.rz_single_selected_table_trial) ||
         selected_table) &&
        model.transportsElectronInternalEnergyAtRawDensity() &&
        options.rz_reciprocal_implicit &&
        options.caloric_backend == NativeThermalCaloricBackend::ReciprocalNodalEmission &&
        options.reciprocal_composite_old_temperature && options.allow_vacuum_noop;
#endif
    bool valid =
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
        (!options.rz_centered_ideal_temperature_trial ||
         (eos.isIdealGas() && builtin_caloric && options.rz_reciprocal_implicit &&
          backend && !options.rz_independent_gray_opacity &&
          !options.rz_closed_reciprocal_endpoint_trial &&
          !options.rz_closed_reciprocal_spatial_endpoint_trial)) &&
#else
        !options.rz_centered_ideal_temperature_trial &&
#endif
        (!options.rz_single_selected_table_vacuum ||
         (options.rz_single_selected_table_trial && options.rz_independent_gray_opacity &&
          options.rz_independent_gray_opacity->IsMixed() &&
          options.rz_independent_gray_opacity->Options().opacity.allow_exact_vacuum)) &&
        (!options.rz_independent_gray_opacity ||
         (backend && selected_table && composite_table_caloric)) &&
        (!options.rz_single_selected_table_trial || selected_table) &&
        same_controls && !EB::enabled() &&
        !WarpX::GetInstance().GetPartContainer().hasHybridIonization() &&
        WarpX::GetInstance().finestLevel() == 0 &&
#if defined(WARPX_DIM_RZ)
        !options.cartesian_native_nodal_force && options.rz_reciprocal_implicit &&
        (!options.rz_native_nodal_force ||
         (builtin_caloric &&
          (geometry.isPeriodic(1) || options.rz_closed_reciprocal_spatial_endpoint_trial) &&
          WarpX::nox >= 1 && WarpX::nox <= 3 &&
          !options.rz_single_selected_table_trial && !options.rz_independent_gray_opacity)) &&
        (geometry.Coord() == 0 || geometry.Coord() == 1) &&
#else
        !options.rz_native_nodal_force && !options.rz_reciprocal_implicit && geometry.Coord() == 0 &&
        (!options.cartesian_native_nodal_force ||
         (geometry.isAllPeriodic() && WarpX::nox >= 1 && WarpX::nox <= 3)) &&
#endif
        model.solvesElectronEnergyEquation() &&
        (builtin_caloric || composite_table_caloric) &&
        options.packets.survival_weighted_emission && options.packets.emission &&
        (!options.reciprocal_composite_old_temperature ||
         options.caloric_backend == NativeThermalCaloricBackend::ReciprocalNodalEmission) &&
        (!options.rz_reciprocal_implicit ||
         (options.caloric_backend == NativeThermalCaloricBackend::ReciprocalNodalEmission &&
          options.reciprocal_composite_old_temperature && options.allow_vacuum_noop &&
          options.packets.rz_angular_conjugate &&
          options.packets.rz_reciprocal_corner_emission && backend != nullptr &&
          model.transportsElectronInternalEnergyAtRawDensity())) &&
        (options.caloric_backend == NativeThermalCaloricBackend::TrialCvCellMean ||
         (options.caloric_backend == NativeThermalCaloricBackend::ReciprocalNodalEmission &&
          backend != nullptr && options.reciprocal_emission_rays_per_corner > 0 &&
          options.reciprocal_emission_rays_per_corner <= 1024)) &&
        std::isfinite(dt) && dt > 0 && !path.empty() && options.max_temperature_iterations > 0 &&
        options.max_linear_iterations > 0 && options.max_material_iterations > 0 &&
        options.max_line_search > 0 && options.material_tolerance > 0 &&
        options.material_tolerance < 1 && options.work_tolerance > 0 &&
        options.work_tolerance < 1 && options.energy_tolerance > 0 &&
        options.energy_tolerance < 1 && std::isfinite(options.minimum_electron_density) &&
        options.minimum_electron_density >= 0 && diffusion.nComp() == 4 && mask.nComp() == 1 &&
        opacity.nComp() == 1 && rate.nComp() == 3 &&
        diffusion.boxArray() == endpoint.Particles().ParticleBoxArray(0) &&
        diffusion.DistributionMap() == endpoint.Particles().ParticleDistributionMap(0) &&
        mask.boxArray() == diffusion.boxArray() &&
        mask.DistributionMap() == diffusion.DistributionMap() &&
        opacity.boxArray() == diffusion.boxArray() &&
        opacity.DistributionMap() == diffusion.DistributionMap() &&
        rate.boxArray() == diffusion.boxArray() &&
        rate.DistributionMap() == diffusion.DistributionMap() &&
        amrex::convert(diffusion.boxArray(), amrex::IntVect(1)) == temperature.boxArray() &&
        temperature.DistributionMap() == diffusion.DistributionMap() && temperature.nComp() == 1 &&
        temperature.nGrowVect().allGE(1);
    auto const& native_geometry = WarpX::GetInstance().Geom(0);
    valid = valid && geometry.Domain() == native_geometry.Domain();
    bool closed_endpoint = false;
    bool closed_table = false;
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
#if defined(WARPX_USE_SINGULARITY_EOS)
    closed_table = options.rz_closed_reciprocal_table_trial && composite_table_caloric &&
        eos.tableStorage() != nullptr && options.rz_closed_reciprocal_spatial_endpoint_trial &&
        !options.rz_closed_reciprocal_endpoint_trial && material.CandidateSpecies().size() == 1 &&
        material.CandidateSpecies().front().name == model.electronThermodynamicsMaterialSpeciesName(0);
    if (closed_table)
    {
        auto const& species = WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(
            model.electronThermodynamicsMaterialSpeciesName(0));
        closed_table = !species.HasEvolvingChargeState();
    }
#endif
    closed_endpoint = (options.rz_closed_reciprocal_endpoint_trial ||
        options.rz_closed_reciprocal_spatial_endpoint_trial) && (builtin_caloric || closed_table) &&
        model.transportsElectronInternalEnergyAtRawDensity() && options.rz_reciprocal_implicit &&
        options.packets.rz_reflecting_axial_reciprocal_endpoint &&
        !options.packets.rz_reflecting_axial_frozen_bath &&
        !geometry.isPeriodic(1) && !native_geometry.isPeriodic(1) &&
        WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
        WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
        WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting &&
        PhysConst::c * dt <= .5_rt * std::min(geometry.CellSize(0), geometry.CellSize(1));
#endif
    valid = valid && (!options.rz_closed_reciprocal_table_trial || closed_table) &&
        (!(options.rz_closed_reciprocal_endpoint_trial &&
                        options.rz_closed_reciprocal_spatial_endpoint_trial)) &&
        (!(options.rz_closed_reciprocal_endpoint_trial ||
           options.rz_closed_reciprocal_spatial_endpoint_trial) || closed_endpoint);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        valid &= (closed_endpoint && d == 1 ? true : options.rz_reciprocal_implicit && d == 0
                      ? !geometry.isPeriodic(d) && !native_geometry.isPeriodic(d)
                      : geometry.isPeriodic(d) && native_geometry.isPeriodic(d)) &&
                 geometry.ProbLo(d) == native_geometry.ProbLo(d) &&
                 geometry.ProbHi(d) == native_geometry.ProbHi(d);
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        return rejected;
    }
    if (options.rz_centered_ideal_temperature_trial && (mask.min(0) != 1 || mask.max(0) != 1))
    {
        rejected.reason = "increment thermal coordinates require every cell in D";
        return rejected;
    }
    if (options.rz_closed_reciprocal_endpoint_trial)
    {
        // This exact zero-D endpoint is not permission to use a spatial D
        // backend. Layout has already been checked collectively on all ranks.
        auto const low = mask.min(0), high = mask.max(0);
        auto const finite = diffusion.is_finite();
        auto const zero = diffusion.norm0(0, 4, diffusion.nGrowVect());
        if (low != 0 || high != 0 || !finite || zero != 0) { return rejected; }
    }
    SourceContext s;
    s.live = &live;
    s.endpoint = &endpoint;
    s.material = &material;
    s.model = &model;
    s.path = path;
    s.geometry = geometry;
    s.options = options;
    s.backend = backend;
    s.rz_reciprocal = options.rz_reciprocal_implicit;
    s.composite_table = composite_table_caloric;
    s.dt = dt;
    s.eos = eos;
    auto const dx = geometry.CellSizeArray();
    s.volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    s.floor =
        model.transportsElectronInternalEnergyAtRawDensity() ? 0 : model.electronDensityFloor();
    s.old_temperature = Copy(temperature);
    s.old_temperature.FillBoundary(geometry.periodicity());
    s.density = Copy(*WarpX::GetInstance().m_fields.get(warpx::fields::FieldType::rho_fp, 0));
    s.old_diffusion = Copy(diffusion);
    s.opacity = Copy(opacity);
    s.kinetic_mask.define(mask.boxArray(), mask.DistributionMap(), 1, 0);
    s.capture_mask.define(mask.boxArray(), mask.DistributionMap(), 1, 0);
    s.capture_mask.setVal(1);
    s.diffusion_flags.define(mask.boxArray(), mask.DistributionMap(), 1, 0,
                             amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    auto const old_diffusion_host = Host(diffusion);
    for (amrex::MFIter it(mask); it.isValid(); ++it)
    {
        std::vector<int> flags(static_cast<std::size_t>(mask[it].box().numPts()));
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, mask[it].dataPtr(),
                         mask[it].dataPtr() + flags.size(), flags.begin());
        std::vector<int> kinetic(static_cast<std::size_t>(it.validbox().numPts()));
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const flag = flags[static_cast<std::size_t>(mask[it].box().index(cell()))];
            valid = valid && (flag == 0 || flag == 1);
            s.diffusion_flags[it](cell(), 0) = flag;
            kinetic[static_cast<std::size_t>(it.validbox().index(cell()))] = 1 - flag;
            if (flag == 0)
            {
                for (int d = 0; d < 4; ++d)
                {
                    valid = valid && old_diffusion_host[it](cell(), d) == 0;
                }
            }
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, kinetic.begin(), kinetic.end(),
                         s.kinetic_mask[it].dataPtr());
    }
    s.node_layout.Define(s.old_temperature, geometry);
    auto expected_nodes = geometry.Domain().numPts();
    if (s.rz_reciprocal)
    {
        expected_nodes = static_cast<amrex::Long>(geometry.Domain().length(0) + 1) *
                         (geometry.Domain().length(1) + (closed_endpoint ? 1 : 0));
    }
    valid = valid && s.node_layout.global_size == expected_nodes;
    s.old_nodes = s.node_layout.Gather(s.old_temperature);
    bool const reciprocal =
        options.caloric_backend == NativeThermalCaloricBackend::ReciprocalNodalEmission;
    if (s.composite_table)
    {
        detail::NativeTableCornerCaptureOptions capture_options;
        capture_options.minimum_electron_density = options.minimum_electron_density;
        capture_options.thermodynamic_electron_density_floor = 0;
        capture_options.rz_native_fv = true;
        capture_options.rz_reflecting_axial_trial = options.rz_closed_reciprocal_table_trial;
        capture_options.require_single_selected_material0 = options.rz_single_selected_table_trial;
        capture_options.allow_single_selected_vacuum = options.rz_single_selected_table_vacuum;
        capture_options.rz_verboncoeur_axis_correction =
            WarpX::GetInstance().verboncoeurAxisCorrection();
        auto capture = detail::CaptureNativeCompositeTableCornersFromModel(
            model, s.density, s.old_temperature, geometry, capture_options);
        if (!capture.Valid())
        {
            rejected.reason = capture.Reason();
            return rejected;
        }
        s.table_caloric =
            std::make_unique<detail::NativeCompositeTableCornerCapture>(std::move(capture));
        if (options.rz_independent_gray_opacity &&
            !options.rz_independent_gray_opacity->ValidateCaloricOwners(
                *s.table_caloric, rejected.reason))
        { return rejected; }
        auto const& owners = s.table_caloric->Owners();
        bool capture_layout_valid = owners.size() == s.node_layout.owned.size() &&
                                    s.table_caloric->GlobalOwnerCount() ==
                                        s.node_layout.global_size;
        amrex::ParallelDescriptor::ReduceBoolAnd(capture_layout_valid);
        if (!capture_layout_valid)
        {
            rejected.reason = "table capture and source owner layouts disagree";
            return rejected;
        }
        s.old_nodes.resize(owners.size());
        s.charge.assign(owners.size(), 0);
        bool capture_order_valid = true;
        for (std::size_t n = 0; n < owners.size(); ++n)
        {
            capture_order_valid = capture_order_valid &&
                                  owners[n].fab == s.node_layout.owned[n].fab &&
                                  owners[n].index == s.node_layout.owned[n].index;
            s.old_nodes[n] = owners[n].caloric.ReferenceTemperature();
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(capture_order_valid);
        if (!capture_order_valid)
        {
            rejected.reason = "table capture and source owner ordering disagree";
            return rejected;
        }
    }
    else if (s.rz_reciprocal)
    {
        s.rz_caloric = std::make_unique<NativeRZOwnedNodalCaloricState>(
            PrepareNativeRZOwnedNodalCaloric(model, s.density, s.old_temperature,
                options.minimum_electron_density,
                NativeRZCaloricOptions{closed_endpoint}));
        if (!s.rz_caloric->Valid())
        {
            rejected.reason = s.rz_caloric->Reason();
            return rejected;
        }
        auto state = Host(s.rz_caloric->Nodes());
        s.old_nodes = s.node_layout.GatherHost(state, 6);
        s.charge = s.node_layout.GatherHost(state, 0);
        s.node_state = std::move(state);
    }
    else if (reciprocal)
    {
        if (!model.PrepareOwnedNodalElectronCaloricState(
                0, s.density, temperature, s.composite_state, options.minimum_electron_density,
                options.allow_vacuum_noop, options.reciprocal_composite_old_temperature))
        {
            rejected.reason = "reciprocal native caloric measure or canonical old T rejected";
            return rejected;
        }
        s.charge = s.node_layout.Gather(s.composite_state);
    }
    else
    {
        s.charge = s.node_layout.Gather(s.density);
    }
    s.old_capacity.resize(s.old_nodes.size());
    s.old_energy.resize(s.old_nodes.size());
    s.available.resize(s.old_nodes.size());
    s.old_corner_energy_offset.assign(s.old_nodes.size(), 0);
    Vector rz_measures;
    if (s.rz_reciprocal && !s.composite_table)
    {
        s.old_corner_energy_offset = s.node_layout.GatherHost(s.node_state, 5);
        rz_measures = s.node_layout.GatherHost(s.node_state, 4);
        for (std::size_t n = 0; n < rz_measures.size(); ++n)
        {
            s.old_corner_energy_offset[n] *= rz_measures[n];
        }
    }
    else if (!s.composite_table && reciprocal &&
             options.reciprocal_composite_old_temperature)
    {
        auto const composite_host = Host(s.composite_state);
        s.old_corner_energy_offset = s.node_layout.GatherHost(composite_host, 5);
    }
    s.active_nodes.assign(s.old_nodes.size(), 1);
    if (s.composite_table)
    {
        auto const& owners = s.table_caloric->Owners();
        for (std::size_t n = 0; n < owners.size(); ++n)
        {
            auto const& owner = owners[n];
            amrex::Real capacity = 0;
            bool occupied = false;
            for (int corner_index = 0; corner_index < owner.corner_count; ++corner_index)
            {
                auto const& corner = owner.corners[corner_index];
                if (corner.raw_charge == 0) { continue; }
                occupied = true;
                auto const state = eos.stateFromMaterialMassDensitiesTemperature(
                    corner.effective_charge, corner.mass_density,
                    owner.caloric.ReferenceTemperature());
                capacity += state.heat_capacity_density * corner.measure;
                valid = valid && std::isfinite(state.pressure +
                                               state.internal_energy_density +
                                               state.heat_capacity_density) &&
                        state.pressure >= 0 && state.heat_capacity_density > 0;
            }
            s.active_nodes[n] = occupied ? 1 : 0;
            s.old_capacity[n] = capacity;
            s.old_energy[n] = 0;
            s.available[n] = owner.caloric.AvailableJoules();
            s.old_corner_energy_offset[n] = owner.caloric.OldOffsetJoules();
            valid = valid && std::isfinite(s.old_nodes[n] + s.old_capacity[n] +
                                           s.available[n] +
                                           s.old_corner_energy_offset[n]) &&
                    ((!occupied && options.rz_single_selected_table_vacuum)
                        ? s.old_nodes[n] >= 0
                        : (s.old_nodes[n] >= eos.minimumTemperature() &&
                           s.old_nodes[n] <= eos.maximumTemperature())) &&
                    (occupied ? s.old_capacity[n] > 0 && s.available[n] >= 0
                              : options.allow_vacuum_noop && s.old_capacity[n] == 0 &&
                                    s.available[n] == 0 &&
                                    s.old_corner_energy_offset[n] == 0);
        }
    }
    else
    {
        for (std::size_t n = 0; n < s.old_nodes.size(); ++n)
        {
            if (options.allow_vacuum_noop && s.charge[n] == 0)
            {
                s.active_nodes[n] = 0;
                s.old_capacity[n] = 0;
                s.old_energy[n] = 0;
                s.available[n] = 0;
                valid = valid && std::isfinite(s.old_nodes[n]) &&
                        s.old_nodes[n] >= eos.minimumTemperature() &&
                        s.old_nodes[n] <= eos.maximumTemperature();
                continue;
            }
            valid = valid &&
                    s.charge[n] >
                        (reciprocal ? 0 : PhysConst::q_e * options.minimum_electron_density);
            s.charge[n] = std::max(s.charge[n], PhysConst::q_e * s.floor);
            auto const state = eos.stateFromChargeDensityTemperature(s.charge[n], s.old_nodes[n]);
            auto const minimum =
                eos.stateFromChargeDensityTemperature(s.charge[n], eos.minimumTemperature());
            auto const measure = s.rz_reciprocal ? rz_measures[n] : 1._rt;
            s.old_capacity[n] = state.heat_capacity_density * measure;
            s.old_energy[n] = state.internal_energy_density * measure;
            s.available[n] = options.reciprocal_composite_old_temperature
                                 ? warpx::hybrid::CenteredOwnedNodalCaloricDifference(
                                       eos, s.charge[n], eos.minimumTemperature(), s.old_nodes[n]) *
                                       measure +
                                       s.old_corner_energy_offset[n]
                                 : (state.internal_energy_density -
                                    minimum.internal_energy_density) * measure;
            valid = valid && state.heat_capacity_density > 0 &&
                    std::isfinite(state.internal_energy_density + state.heat_capacity_density +
                                  s.available[n]) &&
                    s.available[n] >= 0;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        return rejected;
    }
    if (!s.rz_reciprocal)
    {
        amrex::MultiFab node_state(s.old_temperature.boxArray(),
                                   s.old_temperature.DistributionMap(), 4, 1);
        node_state.setVal(0);
        auto state_host = Host(node_state);
        for (std::size_t n = 0; n < s.node_layout.owned.size(); ++n)
        {
            auto const& owner = s.node_layout.owned[n];
            state_host[owner.fab](owner.index, 0) = s.charge[n];
            state_host[owner.fab](owner.index, 1) = s.old_capacity[n];
            state_host[owner.fab](owner.index, 2) = s.old_energy[n];
            state_host[owner.fab](owner.index, 3) = s.available[n];
        }
        amrex::htod_memcpy(node_state, state_host);
        amrex::Gpu::streamSynchronize();
        node_state.OverrideSync(geometry.periodicity());
        node_state.FillBoundary(geometry.periodicity());
        s.node_state = Host(node_state);
    }
    if (reciprocal)
    {
        auto const rho = Host(s.composite_table ? s.table_caloric->FrozenCharge() : s.density);
        auto const old = Host(s.composite_table ? s.table_caloric->FrozenTemperature()
                                                : s.old_temperature);
        std::array<std::unique_ptr<amrex::MultiFab>,
                   ElectronThermodynamicsExecutor::max_materials> table_material;
        if (s.composite_table)
        {
            for (int material_index = 0; material_index < eos.m_num_materials; ++material_index)
            {
                table_material[material_index] = std::make_unique<amrex::MultiFab>(
                    Host(s.table_caloric->FrozenUnitIonCharge(material_index)));
            }
        }
        amrex::MultiFab weights(diffusion.boxArray(), diffusion.DistributionMap(),
                                1 << AMREX_SPACEDIM, 0,
                                amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        weights.setVal(0);
        amrex::Gpu::streamSynchronize();
        for (amrex::MFIter it(weights); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                amrex::Real total_capacity = 0;
                int last = -1;
                for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                {
                    auto node = cell();
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        node[d] += (corner >> d) & 1;
                    }
                    if (rho[it](node) == 0)
                    {
                        continue;
                    }
                    ElectronThermodynamicState state;
                    if (s.composite_table)
                    {
                        ElectronThermodynamicsExecutor::MaterialMassDensities mass_density{};
                        for (int material_index = 0; material_index < eos.m_num_materials;
                             ++material_index)
                        {
                            mass_density[material_index] =
                                (*table_material[material_index])[it](node) *
                                s.table_caloric->FrozenMassPerUnitCharge(material_index);
                        }
                        state = eos.stateFromMaterialMassDensitiesTemperature(
                            rho[it](node), mass_density, old[it](node));
                    }
                    else
                    {
                        state = eos.stateFromChargeDensityTemperature(
                            std::max(rho[it](node), PhysConst::q_e * s.floor), old[it](node));
                    }
                    auto const corner_volume = s.rz_reciprocal
#if defined(WARPX_DIM_RZ)
                                                   ? RZNativeCornerVolume(cell(), corner, geometry)
#else
                                                   ? 0._rt
#endif
                                                   : 1._rt;
                    auto const incidence = state.heat_capacity_density * corner_volume;
                    weights[it](cell(), corner) = incidence;
                    total_capacity += incidence;
                    last = corner;
                    valid = valid && std::isfinite(state.pressure +
                                                   state.internal_energy_density +
                                                   state.heat_capacity_density + incidence) &&
                            state.pressure >= 0 && state.heat_capacity_density > 0 &&
                            incidence > 0;
                }
                if (last < 0)
                {
                    valid = valid && options.allow_vacuum_noop;
                    continue;
                }
                amrex::Real sum = 0;
                for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                {
                    auto const weight =
                        corner == last ? 1 - sum : weights[it](cell(), corner) / total_capacity;
                    weights[it](cell(), corner) = weight;
                    sum += weight;
                    valid = valid && std::isfinite(weight) && weight >= 0;
                }
                valid = valid && sum == 1;
            }
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            rejected.reason = "invalid frozen native corner capacity incidence";
            return rejected;
        }
        s.corner_weights.define(weights.boxArray(), weights.DistributionMap(), weights.nComp(), 0);
        amrex::htod_memcpy(s.corner_weights, weights);
    }
    if (!backend &&
        !s.packets.Prepare(endpoint.Particles(), live, s.kinetic_mask, opacity, rate, geometry, dt,
                           options.packets, rejected.reason, &s.capture_mask))
    {
        return rejected;
    }
    amrex::MultiFab zero(diffusion.boxArray(), diffusion.DistributionMap(), 3, 0);
    zero.setVal(0);
    ParticleImpulseTransaction initial;
    auto const assignment = s.options.cartesian_native_nodal_force
                                ? ParticleImpulseAssignment::ConservativeNodalCellAverage
                            : s.options.UsesRZNativeNodalForce()
                                ? s.options.RZForceAssignment()
                            : s.rz_reciprocal
                                ? ParticleImpulseAssignment::RZAngularConservative
                                : ParticleImpulseAssignment::NearestCell;
    if (!initial.Stage(material, path, zero, true, assignment) ||
        initial.MaximumBetaSquared() > (s.rz_reciprocal ? 1.e-6_rt : 1.e-4_rt))
    {
        rejected.reason = "invalid or high-beta native material endpoint";
        return rejected;
    }
    auto frame = Copy(initial.WorkVelocity());
    s.trial_work.define(diffusion.boxArray(), diffusion.DistributionMap(), 1, 0);
    s.trial_work.setVal(0);
    auto nodes = s.old_nodes;
    if (options.rz_centered_ideal_temperature_trial)
    { std::fill(nodes.begin(), nodes.end(), 0); }
    int temperature_iterations = 0;
    for (int work_iteration = 0; work_iteration < options.max_material_iterations; ++work_iteration)
    {
        Trial accepted;
        for (int iteration = 0; iteration < options.max_temperature_iterations; ++iteration)
        {
            auto trial = Evaluate(s, nodes, frame);
            ++temperature_iterations;
            if (!trial.valid)
            {
                rejected.reason = trial.reason;
                return rejected;
            }
            if (trial.norm <= 1)
            {
                accepted = std::move(trial);
                break;
            }
            Vector direction;
            bool have_direction = false;
            bool table_unaffordable = false;
            std::string direction_failure;
            if (s.composite_table)
            {
                have_direction = CompositeTableInverseDirection(
                    s, nodes, trial, direction, table_unaffordable);
                if (work_iteration == 0 && iteration == 0)
                {
                    s.table_initial_inverse_unaffordable = table_unaffordable;
                }
            }
            if (!have_direction && (!s.composite_table || table_unaffordable))
            {
                // A table inverse can be unaffordable at an intermediate hot
                // iterate even when the coupled T-dependent radiation source
                // has an affordable colder root. Prescribed provider trials
                // remain valid there, so retain the full nonlinear direction
                // instead of clipping Q or rejecting the physical root.
                if (s.composite_table)
                {
                    ++s.table_inverse_newton_fallbacks;
                    ++s.table_unaffordable_newton_fallbacks;
                }
                have_direction = NewtonDirection(
                    s, nodes, trial, frame, direction, direction_failure);
            }
            if (!have_direction)
            {
                rejected.reason = s.composite_table
                                      ? "certified table caloric direction failed"
                                      : "native thermal Newton linear solve failed";
                if (!direction_failure.empty()) { rejected.reason += ": " + direction_failure; }
                return rejected;
            }
            bool improved = false;
            amrex::Real fraction = 1;
            auto const base_merit = FrozenCaloricMerit(s, nodes, trial, trial);
            amrex::Real last_probe_merit = 0, last_probe_norm = 0;
            std::string last_probe_reason;
            for (int search = 0; search < options.max_line_search; ++search)
            {
                Vector next = nodes;
                for (std::size_t n = 0; n < next.size(); ++n)
                {
                    next[n] += fraction * direction[n];
                }
                auto probe = Evaluate(s, next, frame);
                // Globalize the same fixed-scale residual solved by GMRES.
                // The physical per-node acceptance ratio has trial-dependent
                // heat/change denominators and is not a descent merit. It is
                // still required unchanged at the accepted iterate above.
                // A represented table inverse image can already satisfy that
                // EXACT terminal criterion while its certified energy-width
                // floor has a larger raw merit. Such a converged table trial
                // needs no descent; all later native work/inverse/global gates
                // remain mandatory. Built-in arithmetic and order are unchanged.
                auto const probe_merit = probe.valid
                    ? FrozenCaloricMerit(s, nodes, trial, probe)
                    : std::numeric_limits<amrex::Real>::infinity();
                last_probe_merit = probe_merit;
                last_probe_norm = probe.norm;
                last_probe_reason = probe.valid ? "valid" : probe.reason;
                if (detail::AcceptedNativeThermalLineSearchProbe(
                        probe.valid, s.composite_table, probe.norm, probe_merit, base_merit))
                {
                    if (s.composite_table && probe.norm <= 1 && !(probe_merit < base_merit))
                    {
                        ++s.table_converged_nondescent_probes;
                    }
                    nodes = std::move(next);
                    improved = true;
                    break;
                }
                fraction *= 0.5_rt;
            }
            if (!improved)
            {
                amrex::Real maximum_ulp_step = 0;
                for (std::size_t n = 0; n < nodes.size(); ++n)
                {
                    if (!s.active_nodes[n]) { continue; }
                    auto const native_temperature = PhysicalNodeTemperature(s, nodes, n);
                    auto const ulp = std::nextafter(
                        native_temperature, std::numeric_limits<amrex::Real>::infinity()) -
                        native_temperature;
                    maximum_ulp_step = std::max(maximum_ulp_step,
                        ulp > 0 ? std::abs(direction[n]) / ulp :
                        std::numeric_limits<amrex::Real>::infinity());
                }
                amrex::ParallelDescriptor::ReduceRealMax(maximum_ulp_step);
                std::ostringstream message;
                message << "native thermal Newton line search failed"
                        << " (work_iteration=" << work_iteration
                        << ", temperature_iteration=" << iteration
                        << ", base_physical_norm=" << trial.norm
                        << ", base_merit=" << base_merit
                        << ", final_probe_norm=" << last_probe_norm
                        << ", final_probe_merit=" << last_probe_merit
                        << ", maximum_full_direction_ulps=" << maximum_ulp_step
                        << ", final_probe=" << last_probe_reason << ')';
                std::vector<std::size_t> local_order;
                for (std::size_t n = 0; n < nodes.size(); ++n)
                {
                    if (s.active_nodes[n]) { local_order.push_back(n); }
                }
                std::sort(local_order.begin(), local_order.end(), [&] (auto a, auto b)
                { return std::abs(trial.residual[a]) > std::abs(trial.residual[b]); });
                message << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
                for (std::size_t k = 0; k < std::min<std::size_t>(3, local_order.size()); ++k)
                {
                    auto const n = local_order[k];
                    message << " [local_node=" << n << ", T="
                            << PhysicalNodeTemperature(s, nodes, n)
                            << ", residual_J=" << trial.residual[n]
                            << ", requested_J=" << trial.heat[n]
                            << ", arithmetic_J=" << trial.roundoff[n]
                            << ", capacity=" << s.old_capacity[n]
                            << ", direction_K=" << direction[n] << ']';
                }
                rejected.reason = message.str();
                return rejected;
            }
        }
        if (!accepted.valid)
        {
            rejected.reason = "native thermal temperature iteration budget";
            return rejected;
        }
        auto const next_frame = Host(accepted.candidate.material_stage->WorkVelocity());
        auto const old_frame = Host(frame), force = Host(accepted.candidate.physical_debit);
        auto const work = Host(accepted.candidate.material_stage->RequestedWork());
        auto const previous_work = Host(s.trial_work);
        amrex::Real mismatch = 0;
        for (amrex::MFIter it(frame); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                amrex::Real predicted = 0, scale = std::abs(work[it](cell(), 0)), error = 0;
                for (int d = 0; d < 3; ++d)
                {
                    auto const impulse = force[it](cell(), d + 1) / PhysConst::c;
                    predicted += old_frame[it](cell(), d) * impulse;
                    scale += std::abs(old_frame[it](cell(), d) * impulse) +
                             std::abs(next_frame[it](cell(), d) * impulse);
                    error +=
                        std::abs((next_frame[it](cell(), d) - old_frame[it](cell(), d)) * impulse);
                }
                auto const allowance = options.work_tolerance * scale +
                                       128 * std::numeric_limits<amrex::Real>::epsilon() * scale;
                auto const defect = std::max(std::abs(predicted - work[it](cell(), 0)), error);
                mismatch = std::max(mismatch, allowance > 0 ? defect / allowance
                                              : defect == 0
                                                  ? 0
                                                  : std::numeric_limits<amrex::Real>::infinity());
                if (backend)
                {
                    auto const work_scale =
                        std::abs(previous_work[it](cell(), 0)) + std::abs(work[it](cell(), 0));
                    auto const work_bound = (options.work_tolerance +
                                             128 * std::numeric_limits<amrex::Real>::epsilon()) *
                                            work_scale;
                    auto const difference =
                        std::abs(previous_work[it](cell(), 0) - work[it](cell(), 0));
                    mismatch = std::max(
                        mismatch, work_bound > 0    ? difference / work_bound
                                  : difference == 0 ? 0
                                                    : std::numeric_limits<amrex::Real>::infinity());
                }
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(mismatch);
        if (mismatch > 1)
        {
            frame = Copy(accepted.candidate.material_stage->WorkVelocity());
            s.trial_work = Copy(accepted.candidate.material_stage->RequestedWork());
            continue;
        }
        auto& result = accepted.candidate;
        amrex::Real rz_partition_roundoff = 0;
        if (s.rz_reciprocal)
        {
            auto const prior = Host(s.trial_work);
            auto const secant = Host(frame);
            auto const excluded = Host(result.rz_work_excluded_packet_debit);
            auto const d_work = Host(result.rz_diffusion_work_input);
            for (amrex::MFIter it(prior); it.isValid(); ++it)
            {
                for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
                {
                    if (s.diffusion_flags[it](cell()) == 0)
                    {
                        valid = valid && d_work[it](cell()) == 0;
                        continue;
                    }
                    amrex::Real projection = 0, operands = std::abs(prior[it](cell()));
                    for (int d = 0; d < 3; ++d)
                    {
                        auto const term = secant[it](cell(), d) / PhysConst::c *
                                          excluded[it](cell(), d + 1);
                        projection += term;
                        operands += std::abs(term);
                    }
                    auto const expected = prior[it](cell()) - projection;
                    operands += std::abs(projection) + std::abs(expected) +
                                std::abs(d_work[it](cell()));
                    auto const allowance = 128 *
                        std::numeric_limits<amrex::Real>::epsilon() * operands;
                    valid = valid && std::isfinite(expected + allowance) &&
                        std::abs(d_work[it](cell()) - expected) <= allowance;
                    rz_partition_roundoff += allowance;
                }
            }
            amrex::ParallelDescriptor::ReduceBoolAnd(valid);
            amrex::ParallelDescriptor::ReduceRealSum(rz_partition_roundoff);
            if (!valid)
            {
                rejected.reason = "RZ D finite-work population partition failed";
                return rejected;
            }
        }
        auto check = Copy(s.old_temperature), actual = Copy(result.requested_heat);
        amrex::Real realization = 0, table_owner_realized = 0, table_owner_bound = 0;
        if (s.composite_table)
        {
            Vector inverse_nodes(nodes.size());
            auto const& owners = s.table_caloric->Owners();
            for (std::size_t n = 0; n < owners.size(); ++n)
            {
                auto const response = detail::EvaluateCompositeTableCaloricNode(
                    eos, owners[n].caloric, accepted.heat[n]);
                valid = valid && response.Valid() && response.converged;
                inverse_nodes[n] = response.temperature;
                table_owner_realized += response.realized_joules;
                table_owner_bound += response.arithmetic_bound_joules;
            }
            amrex::ParallelDescriptor::ReduceRealSum(table_owner_realized);
            amrex::ParallelDescriptor::ReduceRealSum(table_owner_bound);
            amrex::ParallelDescriptor::ReduceBoolAnd(valid);
            if (valid)
            {
                s.node_layout.Scatter(inverse_nodes, geometry, check);
            }
        }
        else if (s.rz_reciprocal)
        {
            auto response = EvaluateNativeRZOwnedNodalCaloric(
                model, *s.rz_caloric, result.nodal_requested_heat_density);
            valid = response.valid;
            if (valid)
            {
                check = std::move(response.candidate_temperature);
                amrex::MultiFab::Copy(actual, response.cell_realized_energy, 0, 0, 1, 0);
                actual.FillBoundary(geometry.periodicity());
            }
        }
        else if (reciprocal)
        {
            amrex::MultiFab realized, residual;
            valid = model.EvaluateOwnedNodalElectronEnergySource(
                0, s.density, s.old_temperature, s.composite_state,
                result.nodal_requested_heat_density, check, realized, residual, nullptr,
                options.reciprocal_composite_old_temperature);
            if (valid)
            {
                amrex::MultiFab::Copy(actual, realized, 0, 0, 1, 0);
                actual.FillBoundary(geometry.periodicity());
            }
        }
        else
        {
            realization = model.EvaluateElectronEnergySource(0, check, actual,
                                                             options.minimum_electron_density);
            valid = std::isfinite(realization);
        }
        bool finite_check = check.is_finite(), finite_actual = actual.is_finite();
        valid = valid && finite_check && finite_actual;
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            rejected.reason = "final native caloric response rejected";
            return rejected;
        }
        auto const check_nodes = s.node_layout.Gather(check);
        for (std::size_t n = 0; n < nodes.size(); ++n)
        {
            if (!s.active_nodes[n])
            {
                valid = valid && check_nodes[n] == s.old_nodes[n];
                continue;
            }
            if (s.composite_table)
            {
                auto const& caloric = s.table_caloric->Owners()[n].caloric;
                auto const trial_response = detail::EvaluateCompositeTableCaloricNode(
                    eos, caloric, accepted.heat[n], &nodes[n]);
                auto const inverse_response = detail::EvaluateCompositeTableCaloricNode(
                    eos, caloric, accepted.heat[n]);
                auto const error =
                    trial_response.realized_joules - inverse_response.realized_joules;
                auto const allowance =
                    options.material_tolerance *
                        std::max(std::abs(trial_response.realized_joules),
                                 std::abs(accepted.heat[n])) +
                    accepted.roundoff[n] + inverse_response.arithmetic_bound_joules;
                valid = valid && trial_response.Valid() && inverse_response.Valid() &&
                        inverse_response.converged &&
                        inverse_response.temperature == check_nodes[n] &&
                        std::abs(error) <= allowance;
            }
            else
            {
                auto const trial_temperature = PhysicalNodeTemperature(s, nodes, n);
                auto const error = options.reciprocal_composite_old_temperature
                                       ? warpx::hybrid::CenteredOwnedNodalCaloricDifference(
                                             eos, s.charge[n], trial_temperature, check_nodes[n])
                                       : eos.internalEnergyDensityDifference(
                                             s.charge[n], {}, trial_temperature, check_nodes[n]);
                auto const change = options.reciprocal_composite_old_temperature
                                        ? warpx::hybrid::CenteredOwnedNodalCaloricDifference(
                                              eos, s.charge[n], s.old_nodes[n], trial_temperature)
                                        : eos.internalEnergyDensityDifference(
                                              s.charge[n], {}, s.old_nodes[n], trial_temperature);
                auto const measure = s.rz_reciprocal ? rz_measures[n] : 1._rt;
                auto const allowance = options.material_tolerance *
                                           std::max(std::abs(change * measure),
                                                    std::abs(accepted.heat[n] +
                                                             s.old_corner_energy_offset[n])) +
                                       accepted.roundoff[n];
                valid = valid && std::abs(error * measure) <= allowance;
            }
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            rejected.reason = "independent native caloric inverse disagrees with trial";
            return rejected;
        }
        auto realized = Host(actual);
        auto const old_temperature_host =
            Host(s.composite_table ? s.table_caloric->FrozenTemperature() : s.old_temperature);
        auto const accepted_temperature_host = Host(check);
        auto const frozen_density_host =
            Host(s.composite_table ? s.table_caloric->FrozenCharge() : s.density);
        std::array<std::unique_ptr<amrex::MultiFab>,
                   ElectronThermodynamicsExecutor::max_materials> table_material;
        if (s.composite_table)
        {
            for (int material_index = 0; material_index < eos.m_num_materials; ++material_index)
            {
                table_material[material_index] = std::make_unique<amrex::MultiFab>(
                    Host(s.table_caloric->FrozenUnitIonCharge(material_index)));
            }
        }
        amrex::Real caloric_bound = 0;
        for (amrex::MFIter it(actual); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                amrex::Real quadrature = 0, allowance = 0;
                for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner)
                {
                    auto node = cell();
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        node[d] += (corner >> d) & 1;
                    }
                    auto const charge =
                        s.composite_table
                            ? frozen_density_host[it](node)
                            : reciprocal ? (frozen_density_host[it](node) == 0
                                                ? 0
                                                : std::max(frozen_density_host[it](node),
                                                           PhysConst::q_e * s.floor))
                                         : s.node_state[it](node, 0);
                    if (charge == 0)
                    {
                        valid = valid && accepted_temperature_host[it](node, 0) ==
                                             old_temperature_host[it](node, 0);
                        continue;
                    }
                    auto const before = old_temperature_host[it](node, 0);
                    auto const after = accepted_temperature_host[it](node, 0);
                    auto const corner_measure = s.rz_reciprocal
#if defined(WARPX_DIM_RZ)
                                                    ? RZNativeCornerVolume(cell(), corner, geometry)
#else
                                                    ? 0._rt
#endif
                                                    : s.volume / (1 << AMREX_SPACEDIM);
                    ElectronThermodynamicsExecutor::MaterialMassDensities mass_density{};
                    if (s.composite_table)
                    {
                        for (int material_index = 0; material_index < eos.m_num_materials;
                             ++material_index)
                        {
                            mass_density[material_index] =
                                (*table_material[material_index])[it](node) *
                                s.table_caloric->FrozenMassPerUnitCharge(material_index);
                        }
                    }
                    auto const increment =
                        s.composite_table
                            ? eos.internalEnergyDensityDifference(
                                  charge, mass_density, before, after)
                            : options.reciprocal_composite_old_temperature
                                  ? warpx::hybrid::CenteredOwnedNodalCaloricDifference(
                                        eos, charge, before, after)
                                  : eos.internalEnergyDensityDifference(charge, {}, before, after);
                    quadrature += corner_measure * increment;
                    if (before != after)
                    {
                        auto const state = s.composite_table
                                               ? eos.stateFromMaterialMassDensitiesTemperature(
                                                     charge, mass_density, after)
                                               : eos.stateFromChargeDensityTemperature(charge, after);
                        auto const old_state =
                            s.composite_table
                                ? eos.stateFromMaterialMassDensitiesTemperature(
                                      charge, mass_density, before)
                                : reciprocal
                                      ? eos.stateFromChargeDensityTemperature(charge, before)
                                      : ElectronThermodynamicState{
                                            0, s.node_state[it](node, 2), 0, 0, 0, 0};
                        allowance +=
                            corner_measure * 128 *
                            std::numeric_limits<amrex::Real>::epsilon() *
                            (std::abs(state.internal_energy_density) +
                             std::abs(old_state.internal_energy_density));
                    }
                }
                valid = valid && std::isfinite(quadrature + allowance);
                if (s.composite_table)
                {
                    realized[it](cell(), 0) = quadrature;
                }
                else
                {
                    valid = valid &&
                            std::abs(quadrature - realized[it](cell(), 0)) <= allowance;
                }
                caloric_bound += allowance;
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum(caloric_bound);
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            rejected.reason = "independent native EOS corner quadrature failed";
            return rejected;
        }
        if (s.composite_table)
        {
            amrex::htod_memcpy(actual, realized);
            amrex::Gpu::streamSynchronize();
            actual.FillBoundary(geometry.periodicity());
            auto const cell_realized = actual.sum(0);
            auto const scale = std::abs(table_owner_realized) + std::abs(cell_realized);
            valid = std::isfinite(table_owner_realized + table_owner_bound + cell_realized) &&
                    std::abs(table_owner_realized - cell_realized) <=
                        table_owner_bound + caloric_bound +
                            128 * std::numeric_limits<amrex::Real>::epsilon() * scale;
            amrex::ParallelDescriptor::ReduceBoolAnd(valid);
            if (!valid)
            {
                rejected.reason = "table owner and physical-cell caloric inventories disagree";
                return rejected;
            }
        }
        // Adopt the independently inverted physical source image after checking
        // its trial equation and reconstructing every original physical corner.
        result.nodal_temperature = std::move(check);
        result.actual_heat = std::move(actual);
        result.thermal_redistribution = Copy(result.actual_heat);
        amrex::MultiFab::Subtract(result.thermal_redistribution, result.requested_heat, 0, 0, 1, 0);
        result.raw_energy_residual =
            result.raw_debit.sum(0) - result.material_stage->ActualWork().sum(0) -
            result.material_stage->EnergyCarryChange().sum(0) - result.actual_heat.sum(0);
        auto const operations = 128 + 8 * static_cast<amrex::Real>(geometry.Domain().numPts()) +
                                4 * amrex::ParallelDescriptor::NProcs();
        auto const neps = operations * std::numeric_limits<amrex::Real>::epsilon();
        result.energy_roundoff_bound =
            result.storage_roundoff_bound.sum(0) + caloric_bound +
            result.material_stage->WorkAccountingRoundoffBound().sum(0) +
            rz_partition_roundoff +
            neps / (1 - neps) *
                (result.operand_scale.norm1(0) + result.requested_heat.norm1(0) +
                 result.actual_heat.norm1(0));
        auto const energy_scale = result.requested_heat.norm1(0) + result.actual_heat.norm1(0);
        valid = std::isfinite(result.raw_energy_residual + result.energy_roundoff_bound) &&
                std::abs(result.raw_energy_residual) <=
                    options.energy_tolerance * energy_scale + result.energy_roundoff_bound;
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            rejected.reason = "raw native thermal energy ledger failed";
            return rejected;
        }
        auto const raw_momentum = Host(result.raw_debit), scales = Host(result.operand_scale);
        auto const stored_bound = Host(result.storage_roundoff_bound);
        auto const actual_impulse = Host(s.rz_reciprocal
                                             ? result.material_stage->ConjugateImpulse()
                                             : result.material_stage->ActualImpulse());
        auto const momentum_carry = Host(
            s.rz_reciprocal ? result.material_stage->ConjugateMomentumCarryChange()
                            : result.material_stage->MomentumCarryChange());
        auto const angular_bound = s.rz_reciprocal || options.cartesian_native_nodal_force
                                       ? Host(result.material_stage->
                                                  MomentumAccountingRoundoffBound())
                                       : amrex::MultiFab{};
        auto const mass = Host(result.material_stage->MaterialMass());
        bool const overlapping = s.options.UsesOverlappingMaterialForce();
        // For the new overlapping assignment ONLY, source and native readback
        // have different cell ownership. Check integrated requested radial
        // impulse, true axial torque and axial momentum, never false per-cell
        // equality. Existing nearest-cell physics checks remain unchanged.
        std::array<amrex::Real, 12> overlap_ledger{};
        for (amrex::MFIter it(result.raw_debit); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                for (int d = 0; d < 3; ++d)
                {
                    auto const error = raw_momentum[it](cell(), d + 1) / PhysConst::c -
                                       actual_impulse[it](cell(), d) -
                                       momentum_carry[it](cell(), d);
                    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
                    if (overlapping)
                    {
                        amrex::Real measure = 1;
#if defined(WARPX_DIM_RZ)
                        if (d == 1) {
                            measure = RZMomentMetric(geometry).Angular(cell()[0]).mean_radius;
                        }
#endif
                        auto const source =
                            measure * raw_momentum[it](cell(), d + 1) / PhysConst::c;
                        auto const applied = measure * actual_impulse[it](cell(), d);
                        auto const carried = measure * momentum_carry[it](cell(), d);
                        auto const bound = measure *
                            (stored_bound[it](cell(), d + 1) / PhysConst::c +
                             angular_bound[it](cell(), d));
                        auto const operands = measure * scales[it](cell(), d + 1) / PhysConst::c +
                            std::abs(applied) + std::abs(carried) + std::abs(source);
                        valid = valid &&
                            std::isfinite(source + applied + carried + bound + operands) &&
                            bound >= 0 && operands >= 0;
                        overlap_ledger[d] += source;
                        overlap_ledger[3 + d] += applied + carried;
                        overlap_ledger[6 + d] += bound;
                        overlap_ledger[9 + d] += operands;
                        continue;
                    }
                    if (s.rz_reciprocal)
                    {
                        amrex::Real allowance = 0;
                        valid = detail::AcceptedRZConjugateMomentumLedger(
                                    raw_momentum[it](cell(), d + 1),
                                    actual_impulse[it](cell(), d),
                                    momentum_carry[it](cell(), d),
                                    stored_bound[it](cell(), d + 1),
                                    scales[it](cell(), d + 1), mass[it](cell()),
                                    angular_bound[it](cell(), d), allowance) && valid;
                        continue;
                    }
                    auto const allowance =
                        stored_bound[it](cell(), d + 1) / PhysConst::c +
                        128 * epsilon *
                            (scales[it](cell(), d + 1) / PhysConst::c +
                             std::abs(actual_impulse[it](cell(), d)) +
                             std::abs(momentum_carry[it](cell(), d))) +
                        4096 * epsilon * epsilon * mass[it](cell(), 0) * PhysConst::c;
                    valid =
                        valid && std::isfinite(error + allowance) && std::abs(error) <= allowance;
                }
            }
        }
        if (overlapping)
        {
            amrex::ParallelDescriptor::ReduceRealSum(overlap_ledger.data(), 12);
            auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
            auto const overlap_neps = (16 * geometry.Domain().numPts() +
                16 * amrex::ParallelDescriptor::NProcs() + 128) * epsilon;
            valid = valid && std::isfinite(overlap_neps) &&
                overlap_neps >= 0 && overlap_neps < .125;
            for (int d = 0; d < 3; ++d)
            {
                auto const allowance = overlap_ledger[6 + d] +
                    overlap_neps / (1 - overlap_neps) * overlap_ledger[9 + d];
                valid = valid && std::isfinite(allowance) && allowance >= 0 &&
                    std::abs(overlap_ledger[d] - overlap_ledger[3 + d]) <= allowance;
            }
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid)
        {
            rejected.reason = "raw native thermal momentum/carry ledger failed";
            return rejected;
        }
#if defined(WARPX_DIM_RZ)
        if (s.rz_reciprocal && !CheckRZFaceAndGlobalAccounts(s, result))
        {
            rejected.reason = "RZ native field face, interface, or angular account failed";
            return rejected;
        }
        if (s.rz_reciprocal && !CheckRZGlobalNativeInventory(s, result))
        {
            rejected.reason = "RZ native global E/Pz/Lz inventory failed";
            return rejected;
        }
#endif
        result.temperature_iterations = temperature_iterations;
        result.material_iterations = work_iteration + 1;
        result.caloric_owned_nodes = static_cast<amrex::Long>(s.node_layout.owned.size());
        result.caloric_global_nodes = s.node_layout.global_size;
        result.caloric_peak_basis_vectors = s.peak_basis_vectors;
        result.table_inverse_newton_fallbacks =
            s.table_inverse_newton_fallbacks;
        result.table_unaffordable_newton_fallbacks =
            s.table_unaffordable_newton_fallbacks;
        result.table_converged_nondescent_probes = s.table_converged_nondescent_probes;
        result.table_initial_inverse_unaffordable =
            s.table_initial_inverse_unaffordable;
        if (!UniqueLogicalKeys(result.photons->Particles()))
        {
            rejected.reason = "duplicate or malformed accepted thermal birth identity";
            return rejected;
        }
        result.valid = true;
        return std::move(result);
    }
    rejected.reason = "native thermal material iteration budget";
    return rejected;
}
} // namespace warpx::radiation
