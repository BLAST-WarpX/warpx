/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "AdaptivePacketDiffusionTransition.H"
#include "CollidedDiffusionAdmission.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "FluxRelaxedPackets.H"
#include "NativeResolvedRZAdaptiveThermalHistory.H"
#include "NativeResolvedRZThermalHistory.H"
#include "NativeCompositeTableCornerCapture.H"
#include "NativeRZHistoryPhotonWeightProposal.H"
#include "NativeResolvedRZThermalInterval.H"
#include "NativeResolvedThermalInterval.H"
#include "OpacityEvaluator.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhotonParticleContainer.H"
#include "RZAngularGeometry.H"
#include "ResolvedMaterialOpacity.H"
#include "ResolvedGrayMaterialTableOpacity.H"
#include "Utils/MaterialTableIdentity.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_BoxIterator.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

using warpx::fields::FieldType;
using namespace amrex::literals;
using namespace warpx::radiation;

namespace
{
constexpr char manifest_name[] = "/RadiationResolvedThermal_data.txt";

// Global integer counts are already reduced; predraw records remain local.
// This opt-in report never supplies an acceptance predicate or counter input.
#if defined(WARPX_DIM_RZ)
void
PrintHistoryWeightDiagnostics (NativePhotonWeightProposalDiagnostics const& value, int step)
{
    if (!value.collected) { return; }
    std::ostringstream scope;
    scope << "RZ_RUNTIME_NATIVE_WEIGHT_SCOPE step=" << step
          << " input_records=" << value.input_records
          << " positive_records=" << value.positive_records
          << " total_cohorts=" << value.cohorts
          << " visited_cohorts=" << value.visited_cohorts
          << " scheduled_blocks=" << value.scheduled_blocks
          << " eligible_positive_rays=" << value.eligible_positive_rays
          << " native_headroom_refused_blocks=" << value.native_headroom_refused_blocks
          << " history_lower_bound_refused=" << static_cast<int>(value.history_lower_bound_refused);
    for (std::size_t n = 0; n < value.active_size.size(); ++n)
    { scope << " size" << n << '=' << value.active_size[n]; }
    amrex::Print() << scope.str() << '\n';
    for (std::size_t n = 0; n < value.stages.size(); ++n)
    {
        amrex::Print() << "RZ_RUNTIME_NATIVE_WEIGHT_REASON step=" << step
                       << " reason=" << NativePhotonWeightDiagnosticStageNames[n]
                       << " count=" << value.stages[n] << '\n';
    }
    auto const rank = amrex::ParallelDescriptor::MyProc();
    for (std::size_t sample = 0; sample < value.captures.size(); ++sample)
    {
        auto const& capture = value.captures[sample];
        auto prefix = [&] (char const* kind)
        {
            std::ostringstream output;
            output << std::setprecision(std::numeric_limits<long double>::max_digits10)
                   << kind << " step=" << step << " rank=" << rank << " sample=" << sample;
            return output;
        };
        auto meta = prefix("RZ_RUNTIME_NATIVE_WEIGHT_CAPTURE");
        meta << " block=" << capture.block
             << " stage=" << NativePhotonWeightDiagnosticStageNames[
                    static_cast<std::size_t>(capture.stage)]
             << " records=" << capture.rows.size()
             << " certificate_observed=" << static_cast<int>(capture.certificate_observed)
             << " certificate_valid=" << static_cast<int>(capture.certificate.valid)
             << " certificate_admitted=" << static_cast<int>(capture.certificate.admitted)
             << " inverse_residual=" << capture.certificate.inverse_residual_upper
             << " inverse_norm=" << capture.certificate.inverse_norm_upper
             << " condition=" << capture.certificate.condition_upper
             << " gamma=" << capture.certificate.gamma_upper
             << " half_resolution=" << capture.certificate.half_resolution_lower
             << " certificate_kind=" << NativePhotonWeightCertificateKindNames[
                    static_cast<std::size_t>(capture.certificate_kind)]
             << " formation_observed=" << static_cast<int>(capture.formation_observed)
             << " formation_valid=" << static_cast<int>(capture.formation_valid)
             << " original_certificate_observed=" << static_cast<int>(capture.original_certificate_observed)
             << " original_certificate_valid=" << static_cast<int>(capture.original_certificate.valid)
             << " original_certificate_admitted=" << static_cast<int>(capture.original_certificate.admitted)
             << " original_normalized_error=" << capture.original_certificate.normalized_total_error_upper
             << " original_half_resolution=" << capture.original_certificate.half_resolution_lower
             << " plus=" << capture.plus << " minus=" << capture.minus
             << " probability=" << capture.probability;
        amrex::AllPrint() << meta.str() << '\n';
        auto vector = [&] (char const* name, auto const& items)
        {
            auto output = prefix("RZ_RUNTIME_NATIVE_WEIGHT_CAPTURE_VECTOR");
            output << " kind=" << name << " count=" << items.size();
            for (auto item : items) { output << ' ' << item; }
            amrex::AllPrint() << output.str() << '\n';
        };
        vector("cohort", capture.cohort);
        vector("active", capture.active);
        vector("selected_rows", capture.selected_rows);
        vector("retained_rows", capture.retained_rows);
        vector("retained_columns", capture.retained_columns);
        vector("direction", capture.direction);
        for (std::size_t record = 0; record < capture.rows.size(); ++record)
        {
            auto const& row = capture.rows[record];
            auto output = prefix("RZ_RUNTIME_NATIVE_WEIGHT_CAPTURE_ROW");
            output << " record=" << record << " key0=" << row.key.first
                   << " key1=" << row.key.second << " weight=" << row.weight;
            for (std::size_t feature = 0; feature < row.features.size(); ++feature)
            { output << " f" << feature << '=' << row.features[feature]; }
            amrex::AllPrint() << output.str() << '\n';
        }
        for (std::size_t record = 0; record < capture.unit_poses.size(); ++record)
        {
            auto output = prefix("RZ_RUNTIME_NATIVE_WEIGHT_CAPTURE_POSE");
            auto const& pose = capture.unit_poses[record];
            output << " record=" << record << " r=" << pose[0] << " theta=" << pose[1]
                   << " ux=" << pose[2] << " uy=" << pose[3] << " uz=" << pose[4];
            amrex::AllPrint() << output.str() << '\n';
        }
        for (std::size_t record = 0; record < capture.feature_formation.size(); ++record)
        {
            auto output = prefix("RZ_RUNTIME_NATIVE_WEIGHT_CAPTURE_FORMATION");
            auto const& formation = capture.feature_formation[record];
            output << " record=" << record << " valid=" << static_cast<int>(formation.valid);
            for (std::size_t feature = 0; feature < formation.absolute_radius.size(); ++feature)
            { output << " radius" << feature << '=' << formation.absolute_radius[feature]; }
            amrex::AllPrint() << output.str() << '\n';
        }
    }
}
#endif

bool
ValidClosedWallAccounts (ResolvedThermalAxialWallAccounts const& value)
{
    bool valid = true;
    for (auto scalar : {value.kinetic_net, value.diffusion_net, value.cumulative_kinetic_net,
        value.cumulative_diffusion_net, value.last_arithmetic_bound, value.cumulative_arithmetic_bound,
        value.last_operand, value.cumulative_operand})
    {
        valid = valid && std::isfinite(scalar);
    }
    valid = valid && value.last_arithmetic_bound >= 0 && value.cumulative_arithmetic_bound >= 0 &&
        value.last_operand >= 0 && value.cumulative_operand >= 0;
    for (int side = 0; side < 2; ++side)
    {
        for (auto scalar : {value.kinetic[side], value.diffusion[side],
                            value.cumulative_kinetic[side], value.cumulative_diffusion[side]})
        {
            valid = valid && std::isfinite(scalar) && (side == 0 ? scalar <= 0 : scalar >= 0);
        }
    }
    auto const consistent = [] (auto const& sides, amrex::Real net, amrex::Real bound)
    {
        auto const residual = static_cast<long double>(net) - sides[0] - sides[1];
        return std::abs(residual) <= bound;
    };
    return valid && consistent(value.kinetic, value.kinetic_net, value.last_arithmetic_bound) &&
        consistent(value.diffusion, value.diffusion_net, value.last_arithmetic_bound) &&
        consistent(value.cumulative_kinetic, value.cumulative_kinetic_net, value.cumulative_arithmetic_bound) &&
        consistent(value.cumulative_diffusion, value.cumulative_diffusion_net, value.cumulative_arithmetic_bound);
}

#if defined(WARPX_DIM_RZ)
bool
PrepareClosedWallAccounts (ResolvedThermalAxialWallAccounts const& before,
    std::array<amrex::Real, 2> const& kinetic, amrex::Real kinetic_net,
    std::array<amrex::Real, 2> const& diffusion, amrex::Real diffusion_net,
    amrex::Real operand, amrex::Real accepted_bound, std::uint64_t next_interval,
    ResolvedThermalAxialWallAccounts& after)
{
    if (!ValidClosedWallAccounts(before)) { return false; }
    // Only new CLOSED state uses these wall-accumulator operations. Each
    // accepted outer interval adds four sides, two nets and two scalar bounds.
    // No MPI sum: every input is already global in the accepted solver result.
    auto const neps = (64.L + 16.L * next_interval) *
        std::numeric_limits<amrex::Real>::epsilon();
    if (!std::isfinite(neps) || neps < 0 || neps >= .5L) { return false; }
    auto const gamma = neps / (1 - neps);
    after = before;
    after.kinetic = kinetic;
    after.diffusion = diffusion;
    after.kinetic_net = kinetic_net;
    after.diffusion_net = diffusion_net;
    after.last_operand = operand;
    after.last_arithmetic_bound = accepted_bound + gamma * operand;
    for (int side = 0; side < 2; ++side)
    {
        after.cumulative_kinetic[side] += kinetic[side];
        after.cumulative_diffusion[side] += diffusion[side];
    }
    after.cumulative_kinetic_net += kinetic_net;
    after.cumulative_diffusion_net += diffusion_net;
    after.cumulative_operand += operand;
    after.cumulative_arithmetic_bound += after.last_arithmetic_bound +
        gamma * (before.cumulative_operand + operand);
    return ValidClosedWallAccounts(after);
}

ResolvedThermalAxialWallSurface
ClosedWallSurface (NativeResolvedThermalSourceResult const& source,
    NativePhotonPathReport const& path, amrex::Geometry const& geometry, amrex::Real dt)
{
    auto const& sigma = *source.rz_face_lab_stress[1];
    amrex::Vector<amrex::Box> slabs;
    amrex::Vector<int> ranks;
    for (int grid = 0; grid < sigma.boxArray().size(); ++grid)
    {
        auto const box = sigma.boxArray()[grid];
        for (int side = 0; side < 2; ++side)
        {
            auto const j = side == 0 ? geometry.Domain().smallEnd(1) : geometry.Domain().bigEnd(1) + 1;
            if (j < box.smallEnd(1) || j > box.bigEnd(1)) { continue; }
            auto slab = box;
            slab.setSmall(1, j); slab.setBig(1, j);
            slabs.push_back(slab);
            ranks.push_back(sigma.DistributionMap()[grid]);
        }
    }
    amrex::BoxArray const boxes(slabs.data(), static_cast<int>(slabs.size()));
    amrex::DistributionMapping const mapping(ranks);
    auto const copy = [&] (amrex::MultiFab const& field)
    {
        auto result = std::make_shared<amrex::MultiFab>(boxes, mapping, field.nComp(), 0);
        result->setVal(0);
        result->ParallelCopy(field, 0, 0, field.nComp(), 0, 0, geometry.periodicity());
        return std::shared_ptr<amrex::MultiFab const>(std::move(result));
    };
    ResolvedThermalAxialWallSurface result;
    result.dt = dt;
    result.energy_current = copy(*source.rz_face_lab_energy_current[1]);
    result.stress = copy(sigma);
    result.leakage = copy(*source.rz_face_leaked_energy[1]);
    result.kinetic_mirror_keys = path.axial_wall_draw_keys;
    result.birth_keys = source.rz_history_birth_keys;
    if (source.rz_leakage)
    {
        for (auto const& face : source.rz_leakage->face_accounts)
        {
            result.leakage_birth_keys.insert(result.leakage_birth_keys.end(),
                face.keys.begin(), face.keys.end());
        }
    }
    for (auto const& carry : path.photon_carries)
    {
        if (carry.provisional_draw_key != 0 && carry.completed_events > 0)
        {
            result.collided_same_outer_birth_keys.push_back(carry.provisional_draw_key);
        }
    }
    return result;
}
#endif

std::string
ParserIdentity (amrex::Parser const& parser)
{
    std::set<std::string> const variables{"x", "y", "z", "t", "photon_energy", "ne", "Te"};
    amrex::Parser const raw(parser.expr());
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
           << std::quoted(parser.expr());
    for (auto const& symbol : raw.symbols())
    {
        if (!variables.contains(symbol))
        {
            auto const value = utils::parser::makeParser(symbol, {});
            stream << ' ' << std::quoted(symbol) << ' ' << value.compileHost<0>()();
        }
    }
    return stream.str();
}

warpx::radiation::detail::ResolvedMaterialChargeFields
ResolvedThermalMaterialChargeFields (ablastr::fields::MultiFabRegister& fields,
                                     HybridPICModel const& model,
                                     ElectronThermodynamicsExecutor const& eos)
{
    warpx::radiation::detail::ResolvedMaterialChargeFields result{};
    for (int material = 0; material < eos.m_num_materials; ++material)
    {
        result[material] = fields.get(
            "ni_charge_fp_" + model.electronThermodynamicsMaterialSpeciesName(material), 0);
    }
    return result;
}

void
CopyMask (amrex::MultiFab const& source, amrex::iMultiFab& target)
{
    int invalid = 0;
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(target); it.isValid(); ++it)
    {
        auto const old = source.const_array(it);
        auto const next = target.array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     if (old(i, j, k) != 0 && old(i, j, k) != 1)
                     {
                         return {1};
                     }
                     next(i, j, k) = static_cast<int>(old(i, j, k));
                     return {0};
                 });
    }
    invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!invalid, "Invalid thermal resolved binary mask.");
}

bool
FillProperCoefficients (amrex::MultiFab& scattering, amrex::MultiFab& absorption,
                        amrex::MultiFab const& density, amrex::MultiFab const& temperature,
                        amrex::Geometry const& geometry, ResolvedOpacityExecutor const transport,
                        amrex::ParserExecutor<7> const planck, amrex::Real const energy,
                        amrex::Real const time)
{
    // Exterior ghosts are persisted with the accepted sample. Initialize them
    // before valid-cell evaluation; FillBoundary supplies interbox/periodic data.
    scattering.setVal(0);
    absorption.setVal(0);
    auto const low = geometry.ProbLoArray(), dx = geometry.CellSizeArray();
    auto const origin = amrex::lbound(geometry.Domain());
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(scattering); it.isValid(); ++it)
    {
        auto const ne = density.const_array(it), te = temperature.const_array(it);
        auto const chi = scattering.array(it), alpha = absorption.array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     amrex::Real x = 0, y = 0, z = 0;
#if defined(WARPX_DIM_3D)
                     x = low[0] + (i - origin.x + .5_rt) * dx[0];
                     y = low[1] + (j - origin.y + .5_rt) * dx[1];
                     z = low[2] + (k - origin.z + .5_rt) * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                x = low[0] + (i - origin.x + .5_rt) * dx[0];
                z = low[1] + (j - origin.y + .5_rt) * dx[1];
#elif defined(WARPX_DIM_1D_Z)
                z = low[0] + (i - origin.x + .5_rt) * dx[0];
#endif
                     chi(i, j, k) = transport(x, y, z, time, ne(i, j, k), te(i, j, k));
                     alpha(i, j, k) = planck(x, y, z, time, energy, ne(i, j, k), te(i, j, k));
                     return {!std::isfinite(chi(i, j, k) + alpha(i, j, k)) || chi(i, j, k) < 0 ||
                             alpha(i, j, k) < 0};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid)
    {
        return false;
    }
    scattering.FillBoundary(geometry.periodicity());
    absorption.FillBoundary(geometry.periodicity());
    return true;
}

bool
CoefficientsMatch (amrex::MultiFab const& stored, amrex::MultiFab const& expected)
{
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
    for (amrex::MFIter it(stored); it.isValid(); ++it)
    {
        auto const a = stored.const_array(it), b = expected.const_array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     auto const scale = amrex::max(std::abs(a(i, j, k)), std::abs(b(i, j, k)));
                     return {std::abs(a(i, j, k) - b(i, j, k)) > 64 * epsilon * scale};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid == 0;
}

#if defined(WARPX_DIM_RZ) && (!defined(AMREX_USE_GPU) || defined(AMREX_USE_CUDA))
amrex::MultiFab
CopyField (amrex::MultiFab const& source)
{
    amrex::MultiFab result(source.boxArray(), source.DistributionMap(), source.nComp(),
                           source.nGrowVect());
    amrex::MultiFab::Copy(result, source, 0, 0, source.nComp(), source.nGrowVect());
    return result;
}

bool
FieldsExactlyMatch (amrex::MultiFab const& first, amrex::MultiFab const& second)
{
    bool valid = first.boxArray() == second.boxArray() &&
                 first.DistributionMap() == second.DistributionMap() &&
                 first.ixType() == second.ixType() && first.nComp() == second.nComp() &&
                 first.nGrowVect() == second.nGrowVect();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        return false;
    }
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    int const components = first.nComp();
    for (amrex::MFIter it(first); it.isValid(); ++it)
    {
        auto const before = first.const_array(it);
        auto const after = second.const_array(it);
        ops.eval(it.fabbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     int different = 0;
                     for (int n = 0; n < components; ++n)
                     {
                         different |= before(i, j, k, n) != after(i, j, k, n);
                     }
                     return {different};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid == 0;
}

bool
HistoryOuterOpacityAdmissible (
    amrex::MultiFab const& scattering, amrex::MultiFab const& absorption,
    amrex::MultiFab const& rate, amrex::Geometry const& geometry, amrex::Real outer_dt,
    NativeResolvedRZThermalHistoryOptions const& options, bool enforce_absorption_cap = true)
{
    bool valid = scattering.nComp() == 1 && absorption.nComp() == 1 && rate.nComp() == 3 &&
                 scattering.boxArray() == absorption.boxArray() &&
                 scattering.boxArray() == rate.boxArray() &&
                 scattering.DistributionMap() == absorption.DistributionMap() &&
                 scattering.DistributionMap() == rate.DistributionMap() &&
                 std::isfinite(outer_dt) && outer_dt > 0;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid)
    {
        return false;
    }
    auto const radial_lo = geometry.ProbLo(0);
    auto const dr = geometry.CellSize(0);
    int const radial_first = geometry.Domain().smallEnd(0);
    auto const scattering_limit = options.interval.collisions.max_optical_depth;
    auto const absorption_limit = options.interval.thermal.packets.max_absorption_optical_depth;
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(scattering); it.isValid(); ++it)
    {
        auto const chi = scattering.const_array(it);
        auto const alpha = absorption.const_array(it);
        auto const velocity = rate.const_array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     amrex::Real beta_squared = 0;
                     for (int d = 0; d < 3; ++d)
                     {
                         auto value = velocity(i, j, k, d);
                         if (d == 1)
                         {
                             auto const lower = radial_lo + (i - radial_first) * dr;
                             value *= (lower + dr) /
                                      RZAngularGeometry::MeanRadius(lower, lower + dr);
                         }
                         beta_squared += value * value / PhysConst::c2;
                     }
                     if (!std::isfinite(beta_squared) || beta_squared < 0 || beta_squared >= 1)
                     {
                         return {1};
                     }
                     auto const beta = std::sqrt(beta_squared);
                     auto const maximum_doppler = (1 + beta) / std::sqrt(1 - beta_squared);
                     auto const scattering_depth =
                         chi(i, j, k) * PhysConst::c * outer_dt * maximum_doppler;
                     auto const absorption_depth =
                         alpha(i, j, k) * PhysConst::c * outer_dt * maximum_doppler;
                     return {!std::isfinite(scattering_depth + absorption_depth) ||
                             scattering_depth < 0 || absorption_depth < 0 ||
                             scattering_depth > scattering_limit ||
                             (enforce_absorption_cap && absorption_depth > absorption_limit)};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid == 0;
}
#endif

void
PublishThermalMask (amrex::iMultiFab const& mask, amrex::MultiFab& accepted,
                    amrex::Geometry const& geometry)
{
    // CUDA extended lambdas require an addressable enclosing function.
    for (amrex::MFIter it(mask); it.isValid(); ++it)
    {
        auto const source = mask.const_array(it);
        auto const target = accepted.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           { target(i, j, k) = source(i, j, k); });
    }
    accepted.FillBoundary(geometry.periodicity());
}

void
ConstrainThermalMask (amrex::iMultiFab const& old_mask, amrex::iMultiFab& desired,
                      amrex::MultiFab const& opacity, amrex::MultiFab const& velocity,
                      amrex::Geometry const& geometry, amrex::Real dt, bool demotion_only)
{
    amrex::Real spacing = geometry.CellSize(0), length = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        spacing = std::min(spacing, geometry.CellSize(d));
        length = std::max(length, geometry.ProbHi(d) - geometry.ProbLo(d));
    }
#if defined(WARPX_DIM_RZ)
    auto const radial_lo = geometry.ProbLo(0), dr = geometry.CellSize(0);
    int const radial_first = geometry.Domain().smallEnd(0);
#endif
    for (amrex::MFIter it(desired); it.isValid(); ++it)
    {
        auto const old = old_mask.const_array(it);
        auto const next = desired.array(it);
        auto const chi = opacity.const_array(it), v = velocity.const_array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               amrex::Real squared = 0;
                               for (int d = 0; d < 3; ++d)
                               {
                                   auto value = v(i, j, k, d);
#if defined(WARPX_DIM_RZ)
                                   if (d == 1)
                                   {
                                       auto const lower =
                                           radial_lo + (i - radial_first) * dr;
                                       value *= (lower + dr) /
                                                RZAngularGeometry::MeanRadius(
                                                    lower, lower + dr);
                                   }
#endif
                                   squared += value * value / PhysConst::c2;
                               }
                               auto const beta = std::sqrt(squared);
                               if (chi(i, j, k) * spacing < 8 ||
                                   chi(i, j, k) * PhysConst::c * dt < 8 || beta > .001_rt ||
                                   beta * chi(i, j, k) * length > .01_rt ||
                                   (demotion_only && old(i, j, k) == 0))
                               {
                                   next(i, j, k) = 0;
                               }
                           });
    }
    desired.FillBoundary(geometry.periodicity());
}

#if !defined(WARPX_DIM_RZ)
bool
BookThermalCensusRepresentation (NativeResolvedThermalIntervalResult const& interval,
                                 amrex::iMultiFab const& desired, amrex::iMultiFab& accepted,
                                 amrex::MultiFab& carry)
{
    auto const gamma = 128 * std::numeric_limits<amrex::Real>::epsilon();
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(carry); it.isValid(); ++it)
    {
        auto const next = interval.candidate.diffusion_radiation.const_array(it);
        auto const old_e = interval.field.energy.const_array(it);
        auto const old_q = interval.field.lab_momentum.const_array(it);
        auto const census = interval.census.admitted_four_inventory.const_array(it);
        auto const proposed = desired.const_array(it);
        auto const mask = accepted.array(it);
        auto const remainder = carry.array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int>
                 {
                     int invalid = 0;
                     mask(i, j, k) = next(i, j, k, 0) > 0 ? 1 : 0;
                     invalid |= mask(i, j, k) && proposed(i, j, k) != 1;
                     for (int d = 0; d < 4; ++d)
                     {
                         auto const before = d == 0 ? old_e(i, j, k) : old_q(i, j, k, d - 1);
                         auto const actual = next(i, j, k, d);
                         // ONLY actual census addition roundoff belongs to this account.
                         // No transport, recoil, heat or measured global defect enters.
                         auto const difference = census(i, j, k, d) - (actual - before);
                         auto const operands =
                             std::abs(census(i, j, k, d)) + std::abs(actual) + std::abs(before);
                         invalid |=
                             !std::isfinite(difference) || std::abs(difference) > gamma * operands;
                         remainder(i, j, k, d) += difference;
                         invalid |= !std::isfinite(remainder(i, j, k, d));
                         invalid |= mask(i, j, k) == 0 && actual != 0;
                     }
                     return {invalid};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return invalid == 0;
}
#endif
} // namespace

void
RadiationTransport::ConfigureResolvedThermalMaterialOpacity ()
{
    bool supported = false;
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU) && \
    defined(WARPX_USE_SINGULARITY_EOS) && defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ParmParse pp("radiation_transport");
    auto const& geometry = WarpX::GetInstance().Geom(0);
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    int intervals = 0;
    bool adaptive = false;
    std::string backend = "reciprocal_nodal_emission";
    pp.query("resolved_thermal_history_subintervals", intervals);
    pp.query("resolved_adaptive_representation", adaptive);
    pp.query("resolved_thermal_caloric_backend", backend);
    supported = m_use_resolved_packet_fld && m_enable_lte_exchange &&
        m_use_registered_material_opacity_tables && m_num_groups == 1 &&
        m_opacity_species.size() == 1 && m_momentum_species.size() == 1 &&
        m_opacity_species.front() == m_momentum_species.front() &&
        m_opacity_species_material_indices[0] == 0 &&
        eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
        eos.m_use_resolved_material_selection && eos.tableStorage() &&
        m_hybrid_model->electronThermodynamicsMaterialSpeciesName(0) == m_momentum_species.front() &&
        m_hybrid_model->solvesElectronEnergyEquation() &&
        m_hybrid_model->transportsElectronInternalEnergyAtRawDensity() &&
        geometry.isPeriodic(1) && intervals > 0 && intervals <= 32 &&
        !adaptive && !m_resolved_packet.thermal_adaptive_history &&
        !m_resolved_packet.thermal_reflecting_z &&
        !m_resolved_packet.thermal_history_photon_weights &&
        backend == "reciprocal_nodal_emission";
    auto const& table = m_registered_material_opacity_tables[0];
    supported = supported && table.initialized() && table.groupEdges().size() == 2 &&
        table.groupEdges()[0] == 0 && table.groupEdges()[1] ==
            std::numeric_limits<amrex::Real>::infinity() &&
        table.hostExecutor().m_interpolation_mode ==
            static_cast<int>(m_resolved_packet.thermal_material_opacity_zero_scattering
                ? MaterialOpacityInterpolation::ExactZeroLinear
                : MaterialOpacityInterpolation::PositiveLog);
    for (auto const* key : {"resolved_transport_coefficient(x,y,z)",
                           "resolved_transport_coefficient(x,y,z,t)",
                           "resolved_transport_coefficient(x,y,z,t,ne,Te)",
                           "planck_absorption_coefficient",
                           "planck_absorption_coefficient(x,y,z,t,photon_energy,ne,Te)"})
    {
        bool absent = !pp.contains(key);
        amrex::ParallelDescriptor::ReduceBoolAnd(absent);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(absent,
            "Registered thermal table opacity excludes legacy resolved/Planck parsers.");
    }
#endif
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported,
        "Registered thermal table opacity requires one matched registered fixed-charge "
        "TABLE carrier, CPU periodic gray fixed-mask reciprocal N1..32; "
        "weights, adaptive, closed z, N0 and GPU are unsupported.");
}

bool
RadiationTransport::FillResolvedThermalOpacityFromSamples (
    amrex::MultiFab& scattering, amrex::MultiFab& absorption,
    amrex::MultiFab const& ne, amrex::MultiFab const& te,
    amrex::Geometry const& geometry, amrex::Real time, std::string* reason) const
{
    if (!m_resolved_packet.thermal_material_opacity_table)
    {
        return FillProperCoefficients(scattering, absorption, ne, te, geometry,
            m_resolved_packet.opacityExecutor(), m_planck_absorption_coefficient,
            m_resolved_packet.photon_energy, time);
    }
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    detail::ResolvedGrayMaterialTableOpacityOptions options;
    options.enabled = true;
    options.axis_policy = m_resolved_packet.thermal_material_opacity_exact_axis_nodes
        ? detail::ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes
        : detail::ResolvedGrayMaterialOpacityAxisPolicy::StrictInterior;
    options.channel_policy = m_resolved_packet.thermal_material_opacity_zero_scattering
        ? detail::ResolvedGrayMaterialOpacityChannelPolicy::ExactZeroLinearAllScatteringZero
        : detail::ResolvedGrayMaterialOpacityChannelPolicy::PositiveLogMatched;
    options.photon_energy = m_resolved_packet.photon_energy;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_resolved_packet.configured_material_identity.size() == 1,
        "Registered thermal table opacity requires one configured native carrier.");
    auto const& configured = m_resolved_packet.configured_material_identity.front();
    // This coefficient seam also consumes persisted samples before hybrid
    // InitData binds the EOS. Match the later native binding's operation order
    // without depending on its initialization epoch or regathering the samples.
    options.mass_per_electron = PhysConst::q_e * (configured[0] / configured[1]);
    options.minimum_electron_density = m_minimum_electron_density;
    options.thermodynamic_floor = 0;
    std::string detail_reason;
    if (m_resolved_packet.thermal_material_opacity_independent_gray)
    {
        detail::ResolvedIndependentGrayMaterialTableOpacityOptions independent;
        independent.enabled = true;
        independent.allow_exact_vacuum =
            m_resolved_packet.thermal_material_opacity_independent_vacuum;
        independent.axis_policy = options.axis_policy;
        independent.encoding = m_resolved_packet.thermal_material_opacity_zero_scattering
            ? detail::IndependentGrayMaterialOpacityEncoding::ExactZeroLinearAllScatteringZero
            : detail::IndependentGrayMaterialOpacityEncoding::PositiveLogPositiveScattering;
        independent.photon_energy = options.photon_energy;
        independent.mass_per_electron = options.mass_per_electron;
        independent.minimum_electron_density = options.minimum_electron_density;
        independent.thermodynamic_floor = options.thermodynamic_floor;
        detail::ResolvedIndependentGrayMaterialOpacityFields channels;
        channels.rosseland_transport.define(ne.boxArray(), ne.DistributionMap(), 1, 1);
        channels.true_scattering.define(scattering.boxArray(), scattering.DistributionMap(),
                                        1, scattering.nGrowVect());
        channels.proper_absorption.define(absorption.boxArray(), absorption.DistributionMap(),
                                          1, absorption.nGrowVect());
        bool const valid = detail::TryFillResolvedIndependentGrayMaterialTableOpacity(channels,
            ne, te, geometry, m_registered_material_opacity_tables[0], independent, detail_reason);
        if (valid)
        {
            amrex::MultiFab::Copy(scattering, channels.true_scattering, 0, 0, 1,
                                 scattering.nGrowVect());
            amrex::MultiFab::Copy(absorption, channels.proper_absorption, 0, 0, 1,
                                 absorption.nGrowVect());
        }
        if (reason) { *reason = detail_reason; }
        return valid;
    }
    bool const valid = detail::TryFillResolvedGrayMaterialTableOpacity(
        scattering, absorption, ne, te, geometry, m_registered_material_opacity_tables[0],
        options, detail_reason);
    if (reason) { *reason = detail_reason; }
    return valid;
#else
    amrex::ignore_unused(reason);
    return false;
#endif
}

void
RadiationTransport::ConfigureResolvedThermalRuntime ()
{
    auto const& geometry = WarpX::GetInstance().Geom(0);
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    amrex::ParmParse pp("radiation_transport");
#if defined(WARPX_DIM_RZ)
    std::string backend = "reciprocal_nodal_emission";
#else
    std::string backend = "trial_cv_cell_mean";
#endif
    pp.query("resolved_thermal_caloric_backend", backend);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(backend == "trial_cv_cell_mean" ||
                                         backend == "reciprocal_nodal_emission",
                                     "resolved_thermal_caloric_backend must be trial_cv_cell_mean "
                                     "or reciprocal_nodal_emission.");
    m_resolved_packet.thermal_reciprocal_nodal = backend == "reciprocal_nodal_emission";
    bool const builtin_caloric =
        (eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) && eos.m_num_materials == 0;
    // Read the existing opt-in early so table/material/time opacity is widened
    // ONLY for the qualified CPU history combination, never the fixed runtime.
    pp.query("resolved_thermal_history_subintervals",
             m_resolved_packet.thermal_history_subintervals);
    bool composite_table_caloric = false;
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
    composite_table_caloric =
        eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
        (!eos.m_use_resolved_material_selection || m_resolved_packet.thermal_material_opacity_table) &&
        m_hybrid_model->transportsElectronInternalEnergyAtRawDensity() &&
        m_resolved_packet.thermal_reciprocal_nodal &&
        (!m_resolved_packet.thermal_reflecting_z ||
         m_resolved_packet.thermal_reflecting_z_table) &&
        !usesAdaptivePacketRepresentation() &&
        ((!m_resolved_packet.time_dependent_opacity &&
          !m_resolved_packet.material_dependent_opacity) ||
         m_resolved_packet.thermal_history_subintervals > 0) && m_momentum_species.size() == 1 &&
        m_hybrid_model->electronThermodynamicsMaterialSpeciesName(0) ==
            m_momentum_species.front();
    if (composite_table_caloric)
    {
        auto const& material = WarpX::GetInstance()
                                   .GetPartContainer()
                                   .GetParticleContainerFromName(m_momentum_species.front());
        composite_table_caloric = !material.HasEvolvingChargeState();
    }
#endif
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ABORT_WITH_MESSAGE("Resolved thermal packet-FLD does not support this geometry.");
#endif
#if defined(WARPX_DIM_RZ)
    bool closed_geometry = false;
    bool closed_table = false;
#if !defined(AMREX_USE_GPU)
#if defined(WARPX_USE_SINGULARITY_EOS)
    closed_table = m_resolved_packet.thermal_reflecting_z_table && composite_table_caloric &&
        eos.tableStorage() != nullptr && m_resolved_packet.thermal_history_subintervals > 0 &&
        !m_resolved_packet.thermal_history_photon_weights && m_num_groups == 1;
#endif
    closed_geometry = m_resolved_packet.thermal_reflecting_z &&
        !geometry.isPeriodic(1) && (builtin_caloric || closed_table) &&
        !usesAdaptivePacketRepresentation() &&
        m_resolved_packet.thermal_reciprocal_nodal &&
        WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
        WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
        WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting;
#endif
    bool const supported_geometry =
        geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) &&
        (geometry.isPeriodic(1) || closed_geometry) && WarpX::n_rz_azimuthal_modes == 1 &&
        WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
        WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting &&
        m_hybrid_model->transportsElectronInternalEnergyAtRawDensity();
#else
    bool const closed_geometry = false;
    bool const closed_table = false;
    bool const supported_geometry = geometry.isAllPeriodic();
#endif
    bool nodal_force_supported = false;
#if defined(WARPX_DIM_RZ)
    nodal_force_supported = supported_geometry &&
        (geometry.isPeriodic(1) || closed_geometry) && builtin_caloric &&
        m_resolved_packet.thermal_reciprocal_nodal &&
        !m_resolved_packet.thermal_material_opacity_table &&
        WarpX::nox >= 1 && WarpX::nox <= 3 && WarpX::GetInstance().maxLevel() == 0;
#elif !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    nodal_force_supported = geometry.isAllPeriodic() && builtin_caloric &&
        WarpX::nox >= 1 && WarpX::nox <= 3 && WarpX::GetInstance().maxLevel() == 0;
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resolved_packet.thermal_native_nodal_force ||
        nodal_force_supported,
        "resolved_thermal_native_nodal_force requires level zero, native shapes 1-3 and "
        "built-in ideal/latent calorics: periodic Cartesian, or axis/reflecting-radial/"
        "periodic-z m0 RZ (or the separately admitted CPU reflecting-z selector) "
        "with reciprocal nodal calorics and raw-density FV.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_resolved_packet.thermal_material_opacity_independent_vacuum ||
            (WarpX::nox >= 1 && WarpX::nox <= 3 && geometry.isPeriodic(1)),
        "Exact-vacuum recoil requires native particle shapes 1-3 and periodic axial geometry.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resolved_packet.thermal_reflecting_z_table ||
        (closed_table && closed_geometry),
        "resolved_thermal_reflecting_z_table requires CPU fixed-mask reciprocal gray RZ "
        "positive material history, one fixed-charge unresolved table with storage, raw-density "
        "FV and actual PEC/reflecting z; weights, adaptive, N0 and GPU are unsupported.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resolved_packet.thermal_reflecting_z || closed_geometry,
        "resolved_thermal_reflecting_z requires CPU fixed-mask built-in ideal/latent "
        "reciprocal m0 RZ with actual PEC/reflecting nonperiodic z and raw-density FV; "
        "periodic z, GPU, tables and adaptive representation are unsupported.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        supported_geometry && m_resolved_packet.moving_diffusion &&
            (builtin_caloric || composite_table_caloric) &&
            m_lte_exchange_solver == LteExchangeSolver::ImplicitTemperature &&
            m_lte_exchange_tolerance <= 1.e-10_rt,
        "Resolved thermal packet-FLD requires periodic Cartesian or guarded m0 "
        "RZ "
        "(axis/reflecting radial wall, periodic z, raw-density FV) native "
        "ideal/latent electrons, or CPU-only fixed-mask reciprocal RZ with one "
        "fixed-charge unresolved-composition table material and spatial opacity "
        "(or declared-time/material opacity only with explicit fixed-mask history); "
        "moving diffusion and implicit LTE tolerance<=1.e-10 are required.");
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_resolved_packet.thermal_reciprocal_nodal,
        "RZ thermal packet-FLD requires the reciprocal_nodal_emission caloric backend.");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_resolved_packet.thermal_reciprocal_nodal ||
            !pp.contains("resolved_thermal_reciprocal_rays_per_corner"),
        "resolved_thermal_reciprocal_rays_per_corner requires reciprocal_nodal_emission.");
    pp.query("resolved_thermal_reciprocal_rays_per_corner",
             m_resolved_packet.thermal_reciprocal_rays_per_corner);
    pp.query("resolved_thermal_emission_rays", m_resolved_packet.thermal_emission_rays);
    pp.query("resolved_thermal_linear_iterations", m_resolved_packet.thermal_linear_iterations);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_resolved_packet.thermal_emission_rays > 0 &&
                                         m_resolved_packet.thermal_linear_iterations > 0 &&
                                         m_resolved_packet.thermal_reciprocal_rays_per_corner > 0 &&
                                         m_resolved_packet.thermal_reciprocal_rays_per_corner <=
                                             1024 &&
                                         m_resolved_packet.thermal_history_subintervals >= 0 &&
                                         m_resolved_packet.thermal_history_subintervals <= 32,
                                     "Invalid resolved thermal replay/linear budgets.");
    bool history_supported = false;
    bool adaptive_history_supported = false;
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
    adaptive_history_supported = m_resolved_packet.thermal_adaptive_history &&
        builtin_caloric && usesAdaptivePacketRepresentation() && geometry.isPeriodic(1) &&
        !m_resolved_packet.thermal_reflecting_z &&
        !m_resolved_packet.thermal_history_photon_weights && m_num_groups == 1 &&
        m_resolved_packet.thermal_reciprocal_nodal &&
        m_resolved_packet.thermal_history_subintervals > 0 &&
        std::numeric_limits<long double>::min_exponent <= -8192 &&
        std::numeric_limits<long double>::max_exponent >= 8192;
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resolved_packet.thermal_adaptive_history ||
        adaptive_history_supported,
        "resolved_thermal_adaptive_history requires CPU built-in periodic adaptive "
        "reciprocal gray RZ with positive history subintervals and extended long double.");
#if defined(WARPX_DIM_RZ) && (!defined(AMREX_USE_GPU) || defined(AMREX_USE_CUDA))
    history_supported = builtin_caloric && !usesAdaptivePacketRepresentation() &&
                        m_resolved_packet.thermal_reciprocal_nodal;
#if !defined(AMREX_USE_GPU)
    history_supported = history_supported || composite_table_caloric;
#endif
#endif
    history_supported = history_supported || adaptive_history_supported;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_resolved_packet.thermal_history_subintervals == 0 || history_supported,
        "resolved_thermal_history_subintervals is supported only for CPU/CUDA fixed-mask "
        "mode-zero RZ with built-in ideal/latent electrons and reciprocal nodal calorics, "
        "or CPU periodic-z one fixed-charge table material with frozen raw-density corners.");
    bool all_d_history_supported = false;
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
    all_d_history_supported = history_supported &&
        m_use_resolved_packet_fld && m_enable_lte_exchange && m_num_groups == 1 &&
        m_resolved_packet.thermal_history_subintervals > 0 &&
        m_resolved_packet.thermal_reciprocal_nodal && geometry.isPeriodic(1) &&
        m_hybrid_model->transportsElectronInternalEnergyAtRawDensity() &&
        !usesAdaptivePacketRepresentation() && !m_resolved_packet.thermal_adaptive_history &&
        !m_resolved_packet.thermal_history_photon_weights &&
        !m_resolved_packet.thermal_reflecting_z &&
        !m_resolved_packet.thermal_history_pure_absorption_step_depth;
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resolved_packet.thermal_history_all_d_implicit_absorption ||
        all_d_history_supported,
        "All-D implicit absorption requires CPU gray fixed-mask periodic RZ material history.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resolved_packet.thermal_history_all_d_implicit_timestep ||
        (m_resolved_packet.thermal_history_all_d_implicit_absorption && all_d_history_supported),
        "All-D implicit timestep requires witnessed CPU all-D periodic material history.");
    bool history_weights_supported = false;
    bool closed_history_weights_supported = false;
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
    // The physical history finishes its immutable K/D wall accounts first.
    // Endpoint weights use the same wall-aware native witness and do not
    // replay a reflection or add a second material impulse/work update.
    closed_history_weights_supported = m_resolved_packet.thermal_reflecting_z &&
        m_resolved_packet.thermal_history_photon_weights_original_certificate;
#endif
#if defined(WARPX_DIM_RZ) && (!defined(AMREX_USE_GPU) || defined(AMREX_USE_CUDA))
    // Table history admission above does NOT admit its endpoint to the
    // separately qualified built-in photon-weight population proposal.
    history_weights_supported = builtin_caloric && !usesAdaptivePacketRepresentation() &&
        m_resolved_packet.thermal_reciprocal_nodal &&
        (!m_resolved_packet.thermal_reflecting_z || closed_history_weights_supported) &&
        m_resolved_packet.thermal_history_subintervals > 0;
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_resolved_packet.thermal_history_photon_weights || history_weights_supported,
        "resolved_thermal_history_photon_weights supports only CPU/CUDA fixed-mask gray "
        "mode-zero RZ built-in ideal/latent accepted material histories; CPU reflecting-z "
        "additionally requires the original-direction certificate.");
    std::ostringstream identity;
    identity << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
             << "proper_gray_lte_drift_endpoints_halfcell_v1 "
                "leakage_salt_54484d4c45414b31 "
             << "cdt/"
                "dx<=0.5;chi_t*cdt>=8;chi_t*dx>=8;betaD<=0.001;betaD*chi_t*L<=0.01 "
             << std::quoted(m_resolved_packet.thermal_material_opacity_table
                    ? "registered_gray_table_planck_channel" :
                    ParserIdentity(m_planck_absorption_coefficient_parser)) << ' '
             << std::quoted(ParserIdentity(m_resolved_packet.opacity_parser)) << ' '
             << std::quoted(ResolvedMaterialOpacityIdentity()) << ' '
             << m_resolved_packet.thermal_emission_rays << ' '
             << m_resolved_packet.thermal_linear_iterations << ' ' << m_lte_exchange_max_iterations
             << ' ' << m_lte_exchange_tolerance << ' '
             << "survival_weighted=1;max_absorption_depth=16;a_R=7.565733250280007e-"
                "16";
    if (composite_table_caloric)
    {
        auto const& material = WarpX::GetInstance()
                                   .GetPartContainer()
                                   .GetParticleContainerFromName(m_momentum_species.front());
        identity << " composite_table_caloric_rz_fixed_mask_v1 "
                 << std::quoted(m_momentum_species.front()) << ' '
                 << static_cast<amrex::Real>(material.getMass()) << ' '
                 << static_cast<amrex::Real>(material.getCharge()) << ' '
                 << eos.minimumTemperature() << ' ' << eos.maximumTemperature() << ' '
                 << eos.m_atomic_mass[0] << ' ' << eos.m_atomic_number[0] << ' '
                 << eos.m_mass_per_charge[0] << ' ' << eos.m_mass_per_physical_charge[0];
    }
    m_resolved_packet.thermal_identity = identity.str();
    if (m_resolved_packet.thermal_native_nodal_force)
    {
        m_resolved_packet.thermal_identity +=
#if defined(WARPX_DIM_RZ)
            (m_resolved_packet.thermal_reflecting_z
                ? " rz_conservative_native_reflected_closed_nodal_source_work_"
                  "center_carry_v3_shape="
                : " rz_conservative_native_reflected_radial_nodal_source_work_"
                  "center_carry_v2_shape=") +
#else
            " conservative_native_nodal_source_work_center_carry_v1_shape=" +
#endif
            std::to_string(WarpX::nox);
    }
    if (m_resolved_packet.thermal_material_opacity_table)
    {
        std::ostringstream table_identity;
        table_identity << (m_resolved_packet.thermal_material_opacity_independent_gray
            ? (m_resolved_packet.thermal_material_opacity_independent_mixed
                ? " cpu_registered_independent_gray_fixed_mask_packet_fld_frozen_left_v1"
                  "_outer_optical16"
                  "_conditional_libm_2m26"
                : " cpu_registered_independent_gray_R_P_true_S_frozen_left_slice_v1"
                  "_conditional_libm_2m26")
            : " cpu_registered_gray_table_material_history_v1_true_selector_fixed_left_SI"
              "_matched_channels_conditional_libm_2m26");
        if (m_resolved_packet.thermal_material_opacity_exact_axis_nodes)
        {
            table_identity << " gray_table_exact_raw_axis_nodes_v1_owner_binary64_log_correlated_full_affine";
        }
        if (m_resolved_packet.thermal_material_opacity_zero_scattering)
        {
            table_identity << (m_resolved_packet.thermal_history_all_d_implicit_absorption
                ? " gray_table_all_S_positive_zero_mask8_linear_v1_owner_binary64_full_affine_all_d_no_photon_implicit_absorption_guard"
                : m_resolved_packet.thermal_history_pure_absorption_step_depth
                ? " gray_table_all_S_positive_zero_mask8_linear_v1_owner_binary64_full_affine_authenticated_step_optical_guard"
                : " gray_table_all_S_positive_zero_mask8_linear_v1_owner_binary64_full_affine_outer_optical_guard");
        }
        auto const contract = warpx::materials::MaterialTableIdentity::Current();
        if (m_resolved_packet.thermal_material_opacity_independent_vacuum)
        {
            table_identity << " exact_vacuum_K_raw_unit_zero_no_EOS_opacity_query_v1";
            if (WarpX::nox == 1) {
                table_identity << " rz_native_linear_corner_support_source_work_center_carry_v1";
            }
            else {
                table_identity << " rz_native_corner_support_source_work_center_carry_v1_shape="
                               << WarpX::nox;
            }
        }
        table_identity << ' ' << std::quoted(contract.selection_contract);
        for (auto const& table : contract.tables)
        {
            table_identity << ' ' << std::quoted(table.role) << ' ' << std::quoted(table.binding)
                << ' ' << table.material_id << ' ' << std::quoted(table.material_key)
                << ' ' << table.sha256 << ' ' << table.size << ' '
                << std::quoted(table.interpretation);
        }
        m_resolved_packet.thermal_identity += table_identity.str();
    }
    if (m_resolved_packet.thermal_reciprocal_nodal)
    {
        m_resolved_packet.thermal_identity +=
            " reciprocal_nodal_frozen_old_cv_composite_charge_direct_bath_origin_birth_v2"
            "_centered_actual_old_corner_energy " +
            std::to_string(m_resolved_packet.thermal_reciprocal_rays_per_corner);
    }
#if defined(WARPX_DIM_RZ)
    m_resolved_packet.thermal_identity +=
        " rz_m0_native_fv_full_annular_corner_births_angular_conjugate_v1"
        " actual_keyed_dk_leakage_same_total_finite_work_fixed_mask";
    m_resolved_packet.thermal_identity += m_resolved_packet.thermal_reflecting_z
        ? "_reflecting_z_cpu_v1_native_pic_arithmetic_ne_old_cv_T_external_K_D_wall"
        : "_periodic_z";
    m_resolved_packet.thermal_identity +=
        " necessary_annular_causal_moment_cone";
#endif
    if (m_resolved_packet.thermal_history_subintervals > 0)
    {
        m_resolved_packet.thermal_identity +=
            " rz_material_history_v1_partition_uniform_left_fresh_outer_clocks"
            "_frozen_density_positions";
        m_resolved_packet.thermal_identity += m_resolved_packet.thermal_history_all_d_implicit_timestep
            ? "_outer_all_d_implicit_timestep_no_photon_scattering16"
            : m_resolved_packet.thermal_history_all_d_implicit_absorption
            ? "_outer_halfcell_all_d_implicit_absorption_no_photon_scattering16"
            : m_resolved_packet.thermal_history_pure_absorption_step_depth
            ? "_outer_halfcell_step_absorption16_event64"
            : "_outer_halfcell_optical16_event64";
        m_resolved_packet.thermal_identity += "_carry_diffusion_0_n=" +
            std::to_string(m_resolved_packet.thermal_history_subintervals);
        if (m_resolved_packet.thermal_history_pure_absorption_step_depth)
        {
            m_resolved_packet.thermal_identity +=
                " pure_absorption_actual_history_step_dt_depth16_v1_allcell_every_trial";
        }
        if (m_resolved_packet.thermal_history_all_d_implicit_absorption)
        {
            m_resolved_packet.thermal_identity +=
                " all_d_no_photon_implicit_absorption_v1_actual_step_every_trial_scattering16";
        }
        if (m_resolved_packet.thermal_history_all_d_implicit_timestep)
        {
            m_resolved_packet.thermal_identity +=
                " all_d_no_photon_implicit_step_without_photon_halfcell_v1";
        }
        if (composite_table_caloric)
        {
            m_resolved_packet.thermal_identity +=
                " cpu_table_material_history_original_corner_certified_inverse_v1"
                "_frozen_raw_density_unit_ion_charge_provider";
        }
        if (m_resolved_packet.thermal_reflecting_z_table)
        {
            m_resolved_packet.thermal_identity +=
                " cpu_closed_table_material_history_actual_corner_Kwall_Dwall_v1";
        }
    }
    if (m_resolved_packet.thermal_history_photon_weights)
    {
        m_resolved_packet.thermal_identity +=
            " rz_history_native_photon_weights_v1_original_native_law_floor"
            "_all_cells_one_attempt_per_14_sorted_rays"
            "_well_resolved_half_operation_precision_fresh_outer_no_target";
        if (m_resolved_packet.thermal_history_photon_weights_direction_method ==
            PhotonWeightDirectionMethod::RankRevealingOriginalRows)
        {
            m_resolved_packet.thermal_identity +=
                " direction_original_row_rrqr_v1_original_scaled_rows_gaussian_full_13_predraw";
        }
        if (m_resolved_packet.thermal_history_photon_weights_original_certificate)
        {
#if defined(AMREX_USE_CUDA)
            m_resolved_packet.thermal_identity +=
                " cuda_host_original_direction_certificate_v1_fixed_pose_conditional_libm_full_13_predraw";
#else
            m_resolved_packet.thermal_identity +=
                " cpu_original_direction_certificate_v1_fixed_pose_conditional_libm_full_13_predraw";
#endif
        }
        if (closed_history_weights_supported)
        {
            m_resolved_packet.thermal_identity +=
                " cpu_closed_history_original_certificate_v1_immutable_Kwall_Dwall_no_second_kick";
        }
    }
    if (usesAdaptivePacketRepresentation())
    {
        m_resolved_packet.thermal_identity +=
            " adaptive_thermal_total_proper_opacity_census_carry_v1";
#if defined(WARPX_DIM_RZ)
        m_resolved_packet.thermal_identity +=
            " rz_adaptive_thermal_physical_path_census_post_same_native_work"
            "_annular_direct_bath_admission_v1";
#endif
    }
    if (m_resolved_packet.thermal_adaptive_history)
    {
        m_resolved_packet.thermal_identity +=
            " adaptive_fresh_outer_material_history_v1_entry_demote_frozen_N_mask"
            "_actual_current_outer_collision_endpoint_promote_joint_radiation_carry"
            "_original_inner_law_derived_outer_arithmetic_extended_LD_nearest_libm_eps";
    }
    auto const& local = m_resolved_packet.thermal_identity;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(local.size() <=
                                         static_cast<std::size_t>(std::numeric_limits<int>::max()),
                                     "Thermal model identity exceeds the MPI count limit.");
    int count = static_cast<int>(local.size());
    int const root = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(&count, 1, root);
    std::string reference(static_cast<std::size_t>(count), '\0');
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        reference = local;
    }
    amrex::ParallelDescriptor::Bcast(reference.data(), count, root);
    bool same = reference == local;
    amrex::ParallelDescriptor::ReduceBoolAnd(same);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(same, "Resolved thermal model must agree across MPI ranks.");
}

void
RadiationTransport::InitializeResolvedThermalOpacity (
    ablastr::fields::MultiFabRegister& fields) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(usesResolvedThermalExchange() && !m_is_restart &&
                                         !m_resolved_packet.material_opacity_initialized,
                                     "Thermal opacity initialization is fresh and unique.");
    auto const& geometry = WarpX::GetInstance().Geom(0);
    auto& ne = *fields.get(ResolvedElectronDensityFieldName, 0);
    auto& te = *fields.get(ResolvedElectronTemperatureFieldName, 0);
    auto const floor = m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
                           ? 0._rt
                           : m_hybrid_model->electronDensityFloor();
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    auto const material_fields =
        ResolvedThermalMaterialChargeFields(fields, *m_hybrid_model, eos);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resolved_packet.thermal_material_opacity_table ||
        (m_resolved_packet.thermal_material_opacity_independent_vacuum
            ? detail::HasNativeSingleSelectedTableMaterial0OrExactVacuum(*m_hybrid_model, geometry)
            : detail::HasNativeSingleSelectedTableMaterial0(*m_hybrid_model, geometry)),
        "Registered thermal table opacity requires actual positive material0 corners.");
    auto const gather = m_resolved_packet.thermal_material_opacity_independent_vacuum
        ? detail::SnapshotResolvedMaterialOpacityWithExactVacuum
        : detail::SnapshotResolvedMaterialOpacity;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        gather(
            *fields.get(FieldType::rho_fp, 0),
            *fields.get(FieldType::hybrid_electron_temperature_fp, 0), ne, te, geometry,
            eos, m_minimum_electron_density, floor, material_fields),
        "Invalid native thermal opacity samples.");
    auto const time = WarpX::GetInstance().gett_new(0);
    std::string opacity_reason;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        FillResolvedThermalOpacityFromSamples(*fields.get(ResolvedOpacityFieldName, 0),
                               *fields.get(ResolvedAbsorptionFieldName, 0), ne, te, geometry,
                               time, &opacity_reason),
        "Invalid proper gray scattering/absorption coefficients. " + opacity_reason);
    if (m_resolved_packet.thermal_material_opacity_independent_vacuum)
    {
        auto const& mask = *fields.get(ResolvedMaskFieldName, 0);
        bool kinetic_only = true;
        for (amrex::MFIter it(ne); it.isValid(); ++it)
        {
            for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
            {
                kinetic_only = kinetic_only && (ne[it](cell()) != 0 || mask[it](cell()) == 0);
            }
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(kinetic_only);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(kinetic_only,
            "Exact vacuum must be kinetic; an empty diffusion cell is unsupported.");
    }
    m_resolved_packet.thermal_opacity_time = time;
    m_resolved_packet.opacity_time = m_resolved_packet.time_dependent_opacity ? time : 0;
    m_resolved_packet.material_opacity_initialized = true;
}

void
RadiationTransport::WriteResolvedThermalCheckpoint (std::string const& directory) const
{
    if (!usesResolvedThermalExchange())
    {
        return;
    }
    std::ofstream output{directory + manifest_name};
    output << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
           << "resolved_thermal_static_v1\n"
           << std::quoted(m_resolved_packet.thermal_identity) << '\n'
           << m_resolved_packet.thermal_opacity_time << '\n';
    if (m_resolved_packet.thermal_reflecting_z)
    {
        auto const& wall = m_resolved_packet.thermal_axial_wall;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ValidClosedWallAccounts(wall),
            "Cannot checkpoint invalid accepted closed radiation wall state.");
        output << "closed_radiation_axial_walls_v1 kg_m_per_s " << m_resolved_packet.interval << '\n'
            << wall.kinetic[0] << ' ' << wall.kinetic[1] << ' ' << wall.kinetic_net << '\n'
            << wall.diffusion[0] << ' ' << wall.diffusion[1] << ' ' << wall.diffusion_net << '\n'
            << wall.cumulative_kinetic[0] << ' ' << wall.cumulative_kinetic[1] << ' '
            << wall.cumulative_kinetic_net << '\n'
            << wall.cumulative_diffusion[0] << ' ' << wall.cumulative_diffusion[1] << ' '
            << wall.cumulative_diffusion_net << '\n'
            << wall.last_operand << ' ' << wall.cumulative_operand << ' '
            << wall.last_arithmetic_bound << ' ' << wall.cumulative_arithmetic_bound << '\n';
    }
    output.flush();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(output.good(), "Could not checkpoint resolved thermal model.");
}

void
RadiationTransport::ReadResolvedThermalCheckpoint (std::string const& directory)
{
    if (!usesResolvedThermalExchange())
    {
        return;
    }
    std::ifstream input{directory + manifest_name};
    std::string version, identity, trailing;
    amrex::Real time = -1;
    bool metadata = static_cast<bool>(input >> version >> std::quoted(identity) >> time) &&
            version == "resolved_thermal_static_v1" &&
            identity == m_resolved_packet.thermal_identity && std::isfinite(time);
    ResolvedThermalAxialWallAccounts accepted_walls;
    if (m_resolved_packet.thermal_reflecting_z)
    {
        std::string tag, units;
        std::uint64_t interval = 0;
        metadata = metadata && static_cast<bool>(input >> tag >> units >> interval >>
            accepted_walls.kinetic[0] >> accepted_walls.kinetic[1] >> accepted_walls.kinetic_net >>
            accepted_walls.diffusion[0] >> accepted_walls.diffusion[1] >> accepted_walls.diffusion_net >>
            accepted_walls.cumulative_kinetic[0] >> accepted_walls.cumulative_kinetic[1] >>
            accepted_walls.cumulative_kinetic_net >>
            accepted_walls.cumulative_diffusion[0] >> accepted_walls.cumulative_diffusion[1] >>
            accepted_walls.cumulative_diffusion_net >> accepted_walls.last_operand >>
            accepted_walls.cumulative_operand >> accepted_walls.last_arithmetic_bound >>
            accepted_walls.cumulative_arithmetic_bound) &&
            tag == "closed_radiation_axial_walls_v1" && units == "kg_m_per_s" &&
            interval == m_resolved_packet.interval && ValidClosedWallAccounts(accepted_walls);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(metadata && !(input >> trailing),
        "Resolved thermal restart changed its proper-opacity/caloric/replay "
        "model or has incomplete metadata.");
    auto const actual_time = WarpX::GetInstance().gett_new(0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::abs(time - actual_time) <= 64 * std::numeric_limits<amrex::Real>::epsilon() *
                                            std::max(std::abs(time), std::abs(actual_time)),
        "Resolved thermal coefficient epoch differs from checkpoint time.");
    auto& fields = WarpX::GetInstance().m_fields;
    for (auto const* name : {ResolvedAbsorptionFieldName, ResolvedElectronDensityFieldName,
                             ResolvedElectronTemperatureFieldName})
    {
        std::ifstream const header{directory + "/Level_0/" + name + "[level=0]_H"};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(header.good() && fields.get(name, 0)->is_finite() &&
                                             fields.get(name, 0)->min(0) >= 0,
                                         "Resolved thermal restart requires every finite persisted "
                                         "coefficient/sample.");
    }
    auto const& stored = *fields.get(ResolvedAbsorptionFieldName, 0);
    amrex::MultiFab expected_scattering(stored.boxArray(), stored.DistributionMap(), 1, 1);
    amrex::MultiFab expected_absorption(stored.boxArray(), stored.DistributionMap(), 1, 1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        FillResolvedThermalOpacityFromSamples(expected_scattering, expected_absorption,
                               *fields.get(ResolvedElectronDensityFieldName, 0),
                               *fields.get(ResolvedElectronTemperatureFieldName, 0),
                               WarpX::GetInstance().Geom(0), time) &&
            CoefficientsMatch(stored, expected_absorption) &&
            CoefficientsMatch(*fields.get(ResolvedOpacityFieldName, 0), expected_scattering),
        "Resolved thermal restart changed evaluated proper opacity or its "
        "accepted coefficient samples.");
    if (m_resolved_packet.thermal_reflecting_z)
    {
        // Only after every boundary/model/epoch/coefficient/wall gate. No
        // malformed file can publish even a partial last/cumulative wall state.
        m_resolved_packet.thermal_axial_wall = accepted_walls;
    }
    m_resolved_packet.thermal_opacity_time = time;
    m_resolved_packet.material_opacity_initialized = true;
}

void
RadiationTransport::AdvanceResolvedRZThermalHistoryRuntime (
    MultiParticleContainer& particles, ablastr::fields::MultiFabRegister& fields,
    amrex::Real current_time, amrex::Real dt) const
{
#if !defined(WARPX_DIM_RZ) || (defined(AMREX_USE_GPU) && !defined(AMREX_USE_CUDA))
    amrex::ignore_unused(particles, fields, current_time, dt);
    WARPX_ABORT_WITH_MESSAGE(
        "Resolved thermal material history supports only CPU/CUDA RZ.");
#else
    // The ordinary GPU source, private material clones and history audits use
    // the same native owners and explicit device/pinned-host transfers. Only
    // the runtime's field preflights below execute additional CUDA reductions.
    auto const& geometry = WarpX::GetInstance().Geom(0);
    auto& live = particles.GetParticleContainerFromName(m_photon_species);
    auto const initial_allocator = m_resolved_packet.thermal_history_all_d_implicit_absorption
        ? WarpXParticleContainer::ParticleType::the_next_id : amrex::Long(0);
    auto& energy = *fields.get(FieldType::radiation_diffusion_energy, 0);
    auto const& live_temperature =
        *fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& live_density = *fields.get(FieldType::rho_fp, 0);
    auto const endpoint = current_time + dt;
    auto const spacing = std::min(geometry.CellSize(0), geometry.CellSize(1));
    bool const ordinary_outer_admissible =
        m_resolved_packet.thermal_history_subintervals > 0 &&
            m_resolved_packet.thermal_history_subintervals <= 32 &&
            m_resolved_packet.material_opacity_initialized &&
            m_resolved_packet.interval < std::numeric_limits<std::uint64_t>::max() &&
            std::isfinite(current_time) && current_time >= 0 &&
            std::isfinite(endpoint) && endpoint > current_time &&
            std::isfinite(PhysConst::c * dt) &&
            PhysConst::c * dt <= .5_rt * spacing &&
            (!usesAdaptivePacketRepresentation() || m_resolved_packet.thermal_adaptive_history);
    bool implicit_outer_admissible = false;
    if (m_resolved_packet.thermal_history_all_d_implicit_timestep)
    {
        // A structural preflight only admits the private History call. Actual
        // duration/path authority is subsequently issued by that engine and
        // revalidated throughout every leaf, before any live publication.
        bool const scalar_scope =
            m_resolved_packet.thermal_history_all_d_implicit_absorption &&
            m_resolved_packet.thermal_history_subintervals > 0 &&
            m_resolved_packet.thermal_history_subintervals <= 32 &&
            m_resolved_packet.material_opacity_initialized &&
            m_resolved_packet.interval < std::numeric_limits<std::uint64_t>::max() &&
            std::isfinite(current_time) && current_time >= 0 &&
            std::isfinite(dt) && dt > 0 && std::isfinite(endpoint) && endpoint > current_time &&
            std::isfinite(PhysConst::c * dt) && !usesAdaptivePacketRepresentation();
        bool collective_scope = scalar_scope;
        amrex::ParallelDescriptor::ReduceBoolAnd(collective_scope);
        if (collective_scope)
        {
            amrex::iMultiFab admission_mask(energy.boxArray(), energy.DistributionMap(), 1, 1);
            admission_mask.setVal(0);
            CopyMask(*fields.get(ResolvedMaskFieldName, 0), admission_mask);
            admission_mask.FillBoundary(geometry.periodicity());
            PhotonPacketTransaction admission_start(live);
            std::string reason;
            implicit_outer_admissible = ValidateNativeRZAllDNoPhotonState(
                live, admission_start, admission_mask, geometry, reason);
        }
    }
    bool const outer_admissible = m_resolved_packet.thermal_history_all_d_implicit_timestep
        ? implicit_outer_admissible : ordinary_outer_admissible;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        outer_admissible,
        m_resolved_packet.thermal_history_all_d_implicit_timestep
            ? "All-D implicit timestep requires a finite actual all-D/zero-stored-photon outer state."
            : "RZ thermal material history requires one finite fixed-mask outer interval "
              "within the original half-cell guard.");

    auto frozen_density = CopyField(live_density);
    amrex::iMultiFab mask(energy.boxArray(), energy.DistributionMap(), 1, 1);
    if (m_resolved_packet.thermal_history_all_d_implicit_absorption) { mask.setVal(0); }
    CopyMask(*fields.get(ResolvedMaskFieldName, 0), mask);
    mask.FillBoundary(geometry.periodicity());
    amrex::MultiFab radiation(energy.boxArray(), energy.DistributionMap(), 4, 1);
    radiation.setVal(0);
    amrex::MultiFab::Copy(radiation, energy, 0, 0, 1, 0);
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(radiation, *fields.get(ResolvedMomentumFieldName(d), 0), 0,
                              d + 1, 1, 0);
    }
    radiation.FillBoundary(geometry.periodicity());
    PhotonPacketTransaction start(live);
    if (m_resolved_packet.thermal_history_all_d_implicit_absorption)
    {
        std::string state_reason;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ValidateNativeRZAllDNoPhotonState(live, start, mask, geometry, state_reason),
            "All-D implicit absorption initial native state rejected: " + state_reason);
    }
    ParticleImpulseMaterial material(particles, m_momentum_species);

    NativeResolvedRZThermalHistoryOptions history_options;
    history_options.subintervals = m_resolved_packet.thermal_history_subintervals;
    history_options.maximum_subintervals = 32;
    history_options.photon_time_policy = m_resolved_packet.thermal_history_all_d_implicit_timestep
        ? NativeRZThermalPhotonTimePolicy::AllDNoPhotonImplicitStep
        : NativeRZThermalPhotonTimePolicy::PhotonHalfCell;
    history_options.absorption_depth_policy = m_resolved_packet.thermal_history_all_d_implicit_absorption
        ? NativeRZThermalAbsorptionDepthPolicy::AllDNoPhotonImplicitAbsorption
        : m_resolved_packet.thermal_history_pure_absorption_step_depth
        ? NativeRZThermalAbsorptionDepthPolicy::PureAbsorptionActualStep16
        : NativeRZThermalAbsorptionDepthPolicy::WholeOuter16;
    history_options.opacity_model_identity =
        "normal_resolved_runtime_bound_proper_gray_v1";
    auto& options = history_options.interval;
    options.reflecting_axial_fixed_mask_trial = m_resolved_packet.thermal_reflecting_z;
    options.reflecting_axial_table_trial = m_resolved_packet.thermal_reflecting_z_table;
    options.single_selected_table_trial = m_resolved_packet.thermal_material_opacity_table;
    options.thermal.rz_closed_reciprocal_spatial_endpoint_trial = m_resolved_packet.thermal_reflecting_z;
    options.thermal.rz_closed_reciprocal_table_trial = m_resolved_packet.thermal_reflecting_z_table;
    options.thermal.rz_single_selected_table_trial = m_resolved_packet.thermal_material_opacity_table;
    options.thermal.rz_single_selected_table_vacuum =
        m_resolved_packet.thermal_material_opacity_independent_vacuum;
    options.thermal.rz_native_nodal_force = m_resolved_packet.thermal_native_nodal_force;
    options.thermal.packets.rz_reflecting_axial_reciprocal_endpoint = m_resolved_packet.thermal_reflecting_z;
    options.field.reflecting_axial_field_trial = m_resolved_packet.thermal_reflecting_z;
    options.field.reflecting_axial_table_trial = m_resolved_packet.thermal_reflecting_z_table;
    options.field.single_selected_table_trial = m_resolved_packet.thermal_material_opacity_table;
    options.field.single_selected_table_vacuum =
        m_resolved_packet.thermal_material_opacity_independent_vacuum;
    options.thermal.packets.seed = m_particle_conversion_seed;
    options.thermal.packets.interval = m_resolved_packet.interval;
    options.thermal.packets.photon_energy = m_resolved_packet.photon_energy;
    options.thermal.packets.emission_rays_per_cell =
        m_resolved_packet.thermal_emission_rays;
    options.thermal.caloric_backend = NativeThermalCaloricBackend::ReciprocalNodalEmission;
    options.thermal.reciprocal_emission_rays_per_corner =
        m_resolved_packet.thermal_reciprocal_rays_per_corner;
    options.thermal.reciprocal_composite_old_temperature = true;
    options.thermal.max_temperature_iterations = m_lte_exchange_max_iterations;
    options.thermal.max_linear_iterations = m_resolved_packet.thermal_linear_iterations;
    options.thermal.max_material_iterations = m_resolved_packet.max_iterations;
    options.thermal.material_tolerance = m_lte_exchange_tolerance;
    options.thermal.work_tolerance = m_lte_exchange_tolerance;
    options.thermal.energy_tolerance = m_lte_exchange_tolerance;
    options.thermal.minimum_electron_density = m_minimum_electron_density;
    options.collisions.seed = m_particle_conversion_seed;
    options.collisions.interval = m_resolved_packet.interval;
    options.collisions.moving_cross_sections = true;
    options.packet_subsegment_limit = m_resolved_packet.packet_subsegment_limit;
    if (m_resolved_packet.thermal_material_opacity_independent_gray)
    {
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
        // The registered owner is absent in HDF5-disabled CPU/CUDA builds.
        // This compile guard does not activate unsupported TABLE histories.
        history_options.independent_gray_table = &m_registered_material_opacity_tables[0];
        history_options.independent_gray_mixed =
            m_resolved_packet.thermal_material_opacity_independent_mixed;
        auto& independent = history_options.independent_gray_opacity;
        independent.enabled = true;
        independent.allow_exact_vacuum =
            m_resolved_packet.thermal_material_opacity_independent_vacuum;
        independent.photon_energy = m_resolved_packet.photon_energy;
        independent.mass_per_electron = PhysConst::q_e *
            (m_resolved_packet.configured_material_identity.front()[0] /
             m_resolved_packet.configured_material_identity.front()[1]);
        independent.minimum_electron_density = m_minimum_electron_density;
        independent.axis_policy = m_resolved_packet.thermal_material_opacity_exact_axis_nodes
            ? detail::ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes
            : detail::ResolvedGrayMaterialOpacityAxisPolicy::StrictInterior;
        independent.encoding = m_resolved_packet.thermal_material_opacity_zero_scattering
            ? detail::IndependentGrayMaterialOpacityEncoding::ExactZeroLinearAllScatteringZero
            : detail::IndependentGrayMaterialOpacityEncoding::PositiveLogPositiveScattering;
#else
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
            "Independent gray TABLE history requires material-opacity HDF5 support.");
#endif
    }

    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    std::array<std::unique_ptr<amrex::MultiFab>, ElectronThermodynamicsExecutor::max_materials>
        frozen_material_charge;
    detail::ResolvedMaterialChargeFields material_fields{};
    auto const live_material_fields =
        ResolvedThermalMaterialChargeFields(fields, *m_hybrid_model, eos);
    for (int material_index = 0; material_index < eos.m_num_materials; ++material_index)
    {
        frozen_material_charge[material_index] = std::make_unique<amrex::MultiFab>(
            CopyField(*live_material_fields[material_index]));
        material_fields[material_index] = frozen_material_charge[material_index].get();
    }
    auto const floor = m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
                           ? 0._rt
                           : m_hybrid_model->electronDensityFloor();
    NativeRZThermalHistoryOpacityEvaluator evaluate =
        [&] (amrex::Real declared_time, amrex::MultiFab const& private_temperature,
             amrex::MultiFab const& private_rate, amrex::MultiFab& scattering,
             amrex::MultiFab& absorption, std::string& reason)
        {
            amrex::MultiFab ne(scattering.boxArray(), scattering.DistributionMap(), 1, 1);
            amrex::MultiFab te(scattering.boxArray(), scattering.DistributionMap(), 1, 1);
            auto const gather = m_resolved_packet.thermal_material_opacity_independent_vacuum
                ? detail::SnapshotResolvedMaterialOpacityWithExactVacuum
                : detail::SnapshotResolvedMaterialOpacity;
            if (!gather(
                    frozen_density, private_temperature, ne, te, geometry, eos,
                    m_minimum_electron_density, floor, material_fields))
            {
                reason = "invalid frozen-density private-temperature sample";
                return false;
            }
            if (!FillResolvedThermalOpacityFromSamples(scattering, absorption, ne, te, geometry,
                                                       declared_time, &reason))
            {
                if (reason.empty()) { reason = "invalid declared-time proper gray coefficients"; }
                return false;
            }
            // The opt-in engine audits these authentic loaded-owner samples
            // against its actual stored step.dt before Prepare and every
            // frozen/secant/accepted frame. This callback has no duration
            // authority and must not substitute dt/N or enlarge the old cap.
            bool const whole_outer = history_options.absorption_depth_policy ==
                NativeRZThermalAbsorptionDepthPolicy::WholeOuter16;
            bool const all_d_implicit = history_options.absorption_depth_policy ==
                NativeRZThermalAbsorptionDepthPolicy::AllDNoPhotonImplicitAbsorption;
            // The all-D request only delegates absorption admission to the
            // engine witness. This provider still audits the original finite
            // whole-outer depth graph and scattering16; it grants no cap authority.
            if ((whole_outer || all_d_implicit) &&
                !HistoryOuterOpacityAdmissible(scattering, absorption, private_rate, geometry, dt,
                                               history_options, whole_outer))
            {
                reason = whole_outer ? "ordinary outer optical-depth-16 guard rejected"
                                     : "all-D outer scattering or finite-depth guard rejected";
                return false;
            }
            return true;
        };
    NativeResolvedRZThermalHistoryResult fixed_history;
    NativeResolvedRZAdaptiveThermalHistoryResult adaptive_history;
    amrex::MultiFab frozen_representation_carry;
    if (m_resolved_packet.thermal_adaptive_history)
    {
        frozen_representation_carry = CopyField(*fields.get(ResolvedRepresentationCarryFieldName, 0));
        NativeResolvedRZAdaptiveThermalHistoryOptions adaptive_options;
        adaptive_options.fresh_outer_endpoint = true;
        adaptive_options.history = history_options;
        adaptive_options.mask_policy = m_resolved_packet.adaptive_policy;
        for (int d = 0; d < 3; ++d)
        {
            adaptive_options.census.initial_u[d] = live.GetRealCompIndex(
                std::string("radiation_diffusion_census_u") + "xyz"[d] + "0");
        }
        adaptive_history = TryAdvanceNativeResolvedRZAdaptiveThermalHistory(
            live, start, material, "diffusion_0", radiation, mask, frozen_representation_carry,
            live_temperature, *m_hybrid_model, geometry, current_time, dt, evaluate, adaptive_options);
    }
    else
    {
        fixed_history = TryAdvanceNativeResolvedRZThermalHistory(
            live, start, material, "diffusion_0", radiation, mask, live_temperature, *m_hybrid_model,
            geometry, current_time, dt, evaluate, history_options);
    }
    // Nonowning views share all ordinary reporting/publication below. The
    // adaptive inner history remains PRE-promotion evidence; never overwrite
    // its photons/D/clocks with final publication owners.
    auto& result = m_resolved_packet.thermal_adaptive_history ? adaptive_history.history : fixed_history;
    auto& publication_photons = m_resolved_packet.thermal_adaptive_history
        ? adaptive_history.photons : result.photons;
    auto const& publication_radiation = m_resolved_packet.thermal_adaptive_history
        ? adaptive_history.diffusion_radiation : result.diffusion_radiation;
    auto const completed = m_resolved_packet.thermal_adaptive_history
        ? adaptive_history.completed_private_subintervals : result.completed_private_subintervals;
    auto const& reason = m_resolved_packet.thermal_adaptive_history ? adaptive_history.reason : result.reason;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_resolved_packet.thermal_adaptive_history ? adaptive_history.valid : result.valid,
        "Resolved RZ thermal material history rejected before joint native publication after " +
            std::to_string(completed) + " private subintervals: " + reason);

    std::ostringstream expected_policy;
    expected_policy
        << "rz_private_material_history_v1_uniform_left_actual_T_rate_"
           "frozen_density_positions"
        << "_same_outer_clock_distinct_birth_seed_n=" << history_options.subintervals
        << "_cap=" << history_options.maximum_subintervals
        << "_clock_seed=" << options.collisions.seed
        << "_clock_interval=" << options.collisions.interval
        << "_event_cap=" << options.collisions.max_events_per_photon
        << "_carry_path_len=11:diffusion_0"
        << "_opacity_len=" << history_options.opacity_model_identity.size() << ':'
        << history_options.opacity_model_identity;
    if (m_resolved_packet.thermal_native_nodal_force)
    {
        expected_policy << (m_resolved_packet.thermal_reflecting_z
            ? "_builtin_native_reflected_closed_nodal_source_work_center_carry_v3_shape="
            : "_builtin_native_reflected_radial_nodal_source_work_center_carry_v2_shape=")
                        << WarpX::nox;
    }
    if (m_resolved_packet.thermal_reflecting_z)
    {
        expected_policy << "_private_fixed_mask_closed_z_Kwall_Dwall_v1";
    }
    if (m_resolved_packet.thermal_material_opacity_independent_gray)
    {
        expected_policy << (m_resolved_packet.thermal_material_opacity_independent_mixed
            ? "_independent_fixed_mask_packet_fld_R_P_true_S_frozen_left_outer_optical16_v1"
            : "_independent_R_P_true_S_frozen_accepted_left_slice_v1");
    }
    if (m_resolved_packet.thermal_material_opacity_independent_vacuum)
    { expected_policy << "_exact_vacuum_K_raw_unit_zero_no_table_query_v1"
                         "_rz_native_linear_source_work_center_carry_v1"; }
    if (m_resolved_packet.thermal_material_opacity_independent_vacuum && WarpX::nox > 1)
    { expected_policy << "_native_shape=" << WarpX::nox; }
    if (m_resolved_packet.thermal_material_opacity_table)
    {
        expected_policy << "_private_single_registered_table_material0_true_selector_v1";
    }
    if (m_resolved_packet.thermal_reflecting_z_table)
    {
        expected_policy << "_private_closed_table_actual_corner_caloric_v1";
    }
    if (m_resolved_packet.thermal_history_pure_absorption_step_depth)
    {
        expected_policy << "_pure_absorption_actual_history_step_dt_depth16_v1_allcell_every_trial";
    }
    if (m_resolved_packet.thermal_history_all_d_implicit_absorption)
    {
        expected_policy << "_all_d_no_photon_implicit_absorption_v1_actual_step_every_trial_scattering16";
    }
    if (m_resolved_packet.thermal_history_all_d_implicit_timestep)
    {
        expected_policy << "_all_d_no_photon_implicit_step_without_photon_halfcell_v1";
    }
    if (m_resolved_packet.thermal_adaptive_history)
    {
        expected_policy << "_private_current_outer_collided_census_v1"
                           "_private_fresh_outer_adaptive_entry_frozen_history_actual_census_endpoint_v1";
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            adaptive_history.acceptance_witness.valid &&
                result.nodal_temperature.is_finite() && publication_radiation.is_finite() &&
                adaptive_history.radiation_representation_carry.is_finite() &&
                adaptive_history.accepted_diffusion_mask.min(0) >= 0 &&
                adaptive_history.accepted_diffusion_mask.max(0) <= 1 &&
                FieldsExactlyMatch(frozen_representation_carry,
                    *fields.get(ResolvedRepresentationCarryFieldName, 0)),
            "Adaptive RZ history final fields/carry preflight rejected before joint publication.");
    }
    auto const& accepted_policy = m_resolved_packet.thermal_adaptive_history
        ? adaptive_history.policy_identity : result.policy_identity;
    amrex::MultiFab accepted_admission;
    if (m_resolved_packet.thermal_adaptive_history)
    {
        accepted_admission = CopyField(*fields.get(ResolvedAdmissionFieldName, 0));
        auto const& admitted = adaptive_history.endpoint_transition.realized_promotion_inventory;
        if (admitted.ok()) { amrex::MultiFab::Add(accepted_admission, admitted, 0, 0, 1, 0); }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(accepted_admission.is_finite(),
            "Adaptive RZ history admitted-energy diagnostic overflowed before joint publication.");
    }

    amrex::MultiFab ne(energy.boxArray(), energy.DistributionMap(), 1, 1);
    amrex::MultiFab te(energy.boxArray(), energy.DistributionMap(), 1, 1);
    amrex::MultiFab scattering(energy.boxArray(), energy.DistributionMap(), 1, 1);
    amrex::MultiFab absorption(energy.boxArray(), energy.DistributionMap(), 1, 1);
    bool material_density_unchanged = true;
    for (int material_index = 0; material_index < eos.m_num_materials; ++material_index)
    {
        material_density_unchanged = FieldsExactlyMatch(
            *material_fields[material_index], *live_material_fields[material_index]) &&
            material_density_unchanged;
    }
    auto const endpoint_gather = m_resolved_packet.thermal_material_opacity_independent_vacuum
        ? detail::SnapshotResolvedMaterialOpacityWithExactVacuum
        : detail::SnapshotResolvedMaterialOpacity;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        accepted_policy == expected_policy.str() &&
            endpoint_gather(
                frozen_density, result.nodal_temperature, ne, te, geometry, eos,
                m_minimum_electron_density, floor, material_fields) &&
            FillResolvedThermalOpacityFromSamples(scattering, absorption, ne, te, geometry,
                                                 endpoint) &&
            FieldsExactlyMatch(frozen_density, live_density) && material_density_unchanged &&
            result.actual_material_impulse.is_finite(0, 3, 0) &&
            result.material->ValidateLive() && publication_photons->ValidateLive(),
        "Resolved RZ thermal history endpoint/provider/live-owner preflight rejected.");

    auto const work = result.actual_material_work.sum(0);
    auto const carry = result.material_energy_carry_change.sum(0);
    auto const heat = result.actual_heat.sum(0);
    auto defect = -result.endpoint_residual[0];
    bool balanced = std::isfinite(work + carry + heat + defect) &&
                    std::abs(result.endpoint_residual[0]) <=
                        result.endpoint_roundoff_bound[0] &&
                    std::abs(result.endpoint_residual[1]) <=
                        result.endpoint_roundoff_bound[1] &&
                    std::abs(result.endpoint_residual[2]) <=
                        result.endpoint_roundoff_bound[2];
    if (m_resolved_packet.thermal_adaptive_history)
    {
        auto const& witness = adaptive_history.acceptance_witness;
        defect = -static_cast<amrex::Real>(witness.actual_residual[0]);
        balanced = balanced && witness.valid && witness.limits.valid && std::isfinite(defect);
        for (int d = 0; d < 3; ++d)
        {
            balanced = balanced && std::isfinite(witness.actual_residual[d]) &&
                std::abs(witness.actual_residual[d]) <= witness.limits.endpoint[d] &&
                std::abs(witness.composed_residual[d]) <= witness.limits.endpoint[d];
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(balanced);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        balanced, "Resolved RZ thermal history cumulative endpoint ledger rejected.");

    int material_iterations = 0;
    int temperature_iterations = 0;
    int linear_iterations = 0;
    amrex::Real maximum_caloric_residual = 0;
    amrex::Real maximum_linear_residual = 0;
    amrex::Long collisions = 0;
    amrex::Long leakage_births = 0;
    for (auto const& step : result.steps)
    {
        auto const& candidate = step.accepted.candidate;
        material_iterations += candidate.material_iterations;
        temperature_iterations += candidate.temperature_iterations;
        linear_iterations += candidate.rz_field_linear_iterations;
        maximum_caloric_residual =
            std::max(maximum_caloric_residual, candidate.maximum_caloric_residual);
        maximum_linear_residual =
            std::max(maximum_linear_residual, candidate.rz_field_linear_residual);
        collisions += step.accepted.path.path.collision_events;
        if (candidate.rz_leakage)
        {
            leakage_births += candidate.rz_leakage->staged_births;
        }
    }

    std::array<amrex::Long, 3> weight_diagnostics{};
    NativePhotonWeightProposalDiagnostics weight_predraw_diagnostics;
    if (m_resolved_packet.thermal_history_photon_weights)
    {
        // All original physical/endpoint/provider/reporting/live conditions
        // above have passed. The genuine solver-owned witness remains immutable.
        // This boundary is explicitly fresh; the completed history clocks below
        // are never passed as in-progress input, repaired or selectively retained.
        NativeRZPhotonWeightProposalOptions weights;
        weights.fresh_outer_endpoint = true;
        weights.fixed_gray_cohort = true;
        weights.require_well_resolved_direction = true;
        weights.direction_method = m_resolved_packet.thermal_history_photon_weights_direction_method;
        weights.use_original_direction_certificate =
            m_resolved_packet.thermal_history_photon_weights_original_certificate;
        weights.seed = m_particle_conversion_seed;
        weights.outer_interval = m_resolved_packet.interval;
        int diagnostic_control = 0, capture_control = 0;
        amrex::ParmParse weight_pp("radiation_transport");
        weight_pp.query("resolved_thermal_history_photon_weights_diagnostics", diagnostic_control);
        weight_pp.query("resolved_thermal_history_photon_weights_capture_blocks_per_rank", capture_control);
        bool diagnostic_controls_valid = (diagnostic_control == 0 || diagnostic_control == 1) &&
            capture_control >= 0 && capture_control <= 64 &&
            (diagnostic_control == 1 || capture_control == 0);
        amrex::ParallelDescriptor::ReduceBoolAnd(diagnostic_controls_valid);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(diagnostic_controls_valid,
            "History photon-weight diagnostics require a 0/1 flag and 0..64 captured blocks; "
            "capture requires enabled diagnostics.");
        weights.collect_diagnostics = diagnostic_control == 1;
        weights.maximum_captured_blocks_per_rank = static_cast<std::uint64_t>(capture_control);
        auto proposed = TryProposeNativeRZHistoryPhotonWeights(live, result, geometry, weights);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            proposed.valid && proposed.photons,
            "Accepted native history weight proposal violated its preflight contract.");
        // The adapter audits identities/provenance and certifies BOTH represented
        // endpoints before any draw. This required transaction capture is solely
        // a mechanical invariant, NOT recoverable selected-outcome acceptance.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(proposed.photons->ValidateLive(),
            "Certified native history weight copy violated transaction invariants.");
        weight_diagnostics = {proposed.preflight_pairs, proposed.counter_choices,
                              proposed.removed};
        weight_predraw_diagnostics = std::move(proposed.diagnostics);
        if (proposed.applied)
        {
            // Native history residual is BEFORE-minus-AFTER; packet report is
            // new-minus-old. This is diagnostic only, never a post-draw predicate.
            auto const post_residual = result.acceptance_witness.residual[0] -
                                       proposed.actual_change[1];
            defect = -static_cast<amrex::Real>(post_residual);
        }
        result.photons = std::move(proposed.photons);
    }

    ResolvedThermalAxialWallAccounts accepted_walls;
    std::vector<ResolvedThermalAxialWallSurface> accepted_surfaces;
    if (m_resolved_packet.thermal_reflecting_z)
    {
        amrex::Real wall_bound = 0;
        for (auto const& step : result.steps) { wall_bound += step.accepted.axial_wall_roundoff_bound; }
        bool valid_walls = PrepareClosedWallAccounts(m_resolved_packet.thermal_axial_wall,
            {result.kinetic_wall_momentum_by_side[0], result.kinetic_wall_momentum_by_side[1]},
            result.wall_momentum[2],
            {result.diffusion_wall_momentum_by_side[0], result.diffusion_wall_momentum_by_side[1]},
            result.diffusion_wall_momentum, result.axial_wall_momentum_operand,
            wall_bound / PhysConst::c, m_resolved_packet.interval + 1, accepted_walls);
        amrex::ParallelDescriptor::ReduceBoolAnd(valid_walls);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(valid_walls,
            "Closed thermal history wall accounts rejected before joint publication.");
        if (m_resolved_packet.thermal_axial_wall_witness)
        {
            accepted_surfaces.reserve(result.steps.size());
            for (auto const& step : result.steps)
            {
                accepted_surfaces.push_back(ClosedWallSurface(step.accepted.candidate,
                    step.accepted.path.path, geometry, step.dt));
            }
        }
    }
    if (m_resolved_packet.thermal_history_all_d_implicit_absorption)
    {
        // Re-earn the structural state check on both original and final owners
        // immediately before joint publication. No reflected wall/adaptive/
        // weight operation is permitted to introduce a K owner in this mode.
        amrex::iMultiFab current_mask(energy.boxArray(), energy.DistributionMap(), 1, 1);
        current_mask.setVal(0);
        CopyMask(*fields.get(ResolvedMaskFieldName, 0), current_mask);
        current_mask.FillBoundary(geometry.periodicity());
        std::string state_reason;
        bool const initial_empty = ValidateNativeRZAllDNoPhotonState(
            live, start, current_mask, geometry, state_reason);
        bool const final_empty = ValidateNativeRZAllDNoPhotonState(
            live, *publication_photons, current_mask, geometry, state_reason);
        bool state_valid = initial_empty && final_empty &&
            initial_allocator == WarpXParticleContainer::ParticleType::the_next_id &&
            result.photon_clocks.empty() && result.census_receipt.actual_birth_keys.empty() &&
            result.census_receipt.collided_clocks.empty();
        // History/Interval certify the exposed zero-K path/packet/leakage
        // accounts at every trial and final candidate. Fresh-outer cleanup
        // below must not conceal an unexpected surviving clock or birth.
        for (auto const& step : result.steps)
        {
            state_valid = state_valid && step.accepted.path.photon_carries.empty() &&
                step.accepted.candidate.rz_history_birth_keys.empty();
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(state_valid);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(state_valid,
            "All-D implicit absorption publication state rejected: " + state_reason);
    }
    // All numerical, physical, policy and live-provenance gates precede these
    // publications. No numerical rejection is permitted below this point.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        result.material->Commit(),
        "Live material changed before RZ thermal history joint commit.");
    publication_photons->Commit();
    m_hybrid_model->CommitElectronTemperature(0, result.nodal_temperature);
    amrex::MultiFab::Copy(energy, publication_radiation, 0, 0, 1, 0);
    energy.FillBoundary(geometry.periodicity());
    for (int d = 0; d < 3; ++d)
    {
        auto& q = *fields.get(ResolvedMomentumFieldName(d), 0);
        amrex::MultiFab::Copy(q, publication_radiation, d + 1, 0, 1, 0);
        q.FillBoundary(geometry.periodicity());
    }
    if (m_resolved_packet.thermal_adaptive_history)
    {
        PublishThermalMask(adaptive_history.accepted_diffusion_mask,
                           *fields.get(ResolvedMaskFieldName, 0), geometry);
        amrex::MultiFab::Copy(*fields.get(ResolvedRepresentationCarryFieldName, 0),
                              adaptive_history.radiation_representation_carry, 0, 0, 4, 1);
        amrex::MultiFab::Copy(*fields.get(ResolvedAdmissionFieldName, 0), accepted_admission,
                              0, 0, 1, 1);
    }
    amrex::MultiFab::Copy(*fields.get(FieldType::radiation_material_energy, 0), result.actual_heat,
                          0, 0, 1, 0);
    amrex::MultiFab::Copy(*fields.get(FieldType::radiation_material_momentum, 0),
                          result.actual_material_impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(*fields.get(FieldType::radiation_material_kinetic_energy, 0),
                          result.actual_material_work, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*fields.get(ResolvedOpacityFieldName, 0), scattering, 0, 0, 1, 1);
    amrex::MultiFab::Copy(*fields.get(ResolvedAbsorptionFieldName, 0), absorption, 0, 0, 1, 1);
    amrex::MultiFab::Copy(*fields.get(ResolvedElectronDensityFieldName, 0), ne, 0, 0, 1, 1);
    amrex::MultiFab::Copy(*fields.get(ResolvedElectronTemperatureFieldName, 0), te, 0, 0, 1, 1);
    m_resolved_packet.opacity_time = m_resolved_packet.time_dependent_opacity ? endpoint : 0;
    m_resolved_packet.thermal_opacity_time = endpoint;
    m_last_numerical_energy_residual = defect;
    m_cumulative_numerical_energy_residual += defect;
    m_last_material_carry_energy_change = carry;
    m_last_boundary_energy_loss = m_last_diffusion_boundary_energy_loss =
        m_last_streaming_boundary_energy_loss = 0;
    m_last_boundary_energy_injection = 0;
    m_last_diffusion_boundary_momentum_loss = m_last_streaming_boundary_momentum_loss = {0, 0, 0};
    if (m_resolved_packet.thermal_reflecting_z)
    {
        m_resolved_packet.thermal_axial_wall = accepted_walls;
        m_resolved_packet.thermal_axial_wall_surfaces.swap(accepted_surfaces); // no-throw publication
        m_last_streaming_boundary_momentum_loss[2] = accepted_walls.kinetic_net;
        m_last_diffusion_boundary_momentum_loss[2] = accepted_walls.diffusion_net;
        m_last_particle_wall_momentum = {0, 0, 0}; // material walls accumulate later, separately
    }
    else
    {
        m_last_particle_wall_momentum = result.wall_momentum;
    }
    m_last_coupled_solve = {material_iterations, history_options.subintervals, 0, 0,
                            maximum_caloric_residual, defect};
    m_last_implicit_diffusion = {};
    m_last_implicit_diffusion.linear_iterations = linear_iterations;
    m_last_implicit_diffusion.maximum_relative_residual = maximum_linear_residual;
    result.photon_clocks.clear(); // declared fresh at the next accepted outer interval
    if (m_resolved_packet.thermal_history_photon_weights)
    {
        m_resolved_packet.last_history_photon_weights = weight_diagnostics;
        m_resolved_packet.last_history_weight_witness = result.acceptance_witness;
    }
    // Do not emit attempted proposals as accepted-step resource evidence.
    PrintHistoryWeightDiagnostics(weight_predraw_diagnostics, WarpX::GetInstance().getistep(0));
    ++m_resolved_packet.interval;
    if (m_resolved_packet.verbosity > 0)
    {
        amrex::Print() << "Resolved RZ thermal history accepted interval="
                       << m_resolved_packet.interval
                       << " private_subintervals=" << history_options.subintervals
                       << " heat=" << heat << " finite_work=" << work
                       << " signed_energy_defect_J=" << defect
                       << " temperature_iterations=" << temperature_iterations
                       << " collisions=" << collisions << " leakage_births=" << leakage_births;
        if (m_resolved_packet.thermal_history_photon_weights)
        {
            amrex::Print() << " weight_pairs=" << weight_diagnostics[0]
                           << " weight_choices=" << weight_diagnostics[1]
                           << " weight_removed=" << weight_diagnostics[2];
        }
        amrex::Print() << '\n';
    }
#endif
}

void
RadiationTransport::AdvanceResolvedThermalRuntime (MultiParticleContainer& particles,
                                                   ablastr::fields::MultiFabRegister& fields,
                                                   amrex::Real current_time, amrex::Real dt) const
{
    if (m_resolved_packet.thermal_history_subintervals > 0)
    {
        AdvanceResolvedRZThermalHistoryRuntime(particles, fields, current_time, dt);
        return;
    }
    auto const& geometry = WarpX::GetInstance().Geom(0);
    auto& live = particles.GetParticleContainerFromName(m_photon_species);
    auto& energy = *fields.get(FieldType::radiation_diffusion_energy, 0);
    auto const& live_temperature = *fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_resolved_packet.material_opacity_initialized &&
            m_resolved_packet.interval < std::numeric_limits<std::uint64_t>::max(),
        "Thermal resolved interval requires initialized coefficients and an "
        "available logical counter.");
    auto const endpoint = current_time + dt;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(current_time) && std::isfinite(endpoint) && endpoint > current_time,
        "Thermal resolved runtime requires a finite advancing interval.");
    amrex::MultiFab ne(energy.boxArray(), energy.DistributionMap(), 1, 1);
    amrex::MultiFab te(energy.boxArray(), energy.DistributionMap(), 1, 1);
    auto const floor = m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
                           ? 0._rt
                           : m_hybrid_model->electronDensityFloor();
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    auto const material_fields =
        ResolvedThermalMaterialChargeFields(fields, *m_hybrid_model, eos);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        detail::SnapshotResolvedMaterialOpacity(
            *fields.get(FieldType::rho_fp, 0), live_temperature, ne, te, geometry,
            eos, m_minimum_electron_density, floor, material_fields),
        "Thermal resolved coefficient snapshot rejected before live "
        "publication.");
    amrex::MultiFab scattering(energy.boxArray(), energy.DistributionMap(), 1, 1);
    amrex::MultiFab absorption(energy.boxArray(), energy.DistributionMap(), 1, 1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        FillProperCoefficients(scattering, absorption, ne, te, geometry,
                               m_resolved_packet.opacityExecutor(), m_planck_absorption_coefficient,
                               m_resolved_packet.photon_energy, endpoint),
        "Thermal proper opacity rejected before live publication.");
    amrex::iMultiFab mask(energy.boxArray(), energy.DistributionMap(), 1, 1);
    CopyMask(*fields.get(ResolvedMaskFieldName, 0), mask);
    mask.FillBoundary(geometry.periodicity());
    amrex::MultiFab radiation(energy.boxArray(), energy.DistributionMap(), 4, 1);
    radiation.setVal(0);
    amrex::MultiFab::Copy(radiation, energy, 0, 0, 1, 0);
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(radiation, *fields.get(ResolvedMomentumFieldName(d), 0), 0, d + 1, 1,
                              0);
    }
    radiation.FillBoundary(geometry.periodicity());
    PhotonPacketTransaction start(live);
    ParticleImpulseMaterial material(particles, m_momentum_species);
    CollidedDiffusionAdmissionComponents census;
    for (int d = 0; d < 3; ++d)
    {
        census.initial_u[d] =
            live.GetRealCompIndex(std::string("radiation_diffusion_census_u") + "xyz"[d] + "0");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        SnapshotCollidedDiffusionAdmission(start.Particles(), census).valid,
        "Thermal source could not stage the physical collision census.");
    amrex::MultiFab zero(energy.boxArray(), energy.DistributionMap(), 3, 0);
    zero.setVal(0);
    ParticleImpulseTransaction frame;
#if defined(WARPX_DIM_RZ)
    auto const assignment = m_resolved_packet.thermal_native_nodal_force
        ? (m_resolved_packet.thermal_reflecting_z
               ? ParticleImpulseAssignment::RZAngularReflectingClosedNodalCellAverage
               : ParticleImpulseAssignment::RZAngularReflectingRadialNodalCellAverage)
        : ParticleImpulseAssignment::RZAngularConservative;
#else
    auto const assignment = m_resolved_packet.thermal_native_nodal_force
        ? ParticleImpulseAssignment::ConservativeNodalCellAverage
        : ParticleImpulseAssignment::NearestCell;
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        frame.Stage(material, "diffusion_0", zero, true, assignment),
        "Thermal source could not sample the private native rate frame.");
    amrex::GpuArray<amrex::Real, 4> original_fields{};
    for (int d = 0; d < 4; ++d)
    {
        original_fields[d] = radiation.sum(d);
    }
    amrex::MultiFab representation_carry;
    amrex::iMultiFab desired;
    amrex::Long demoted = 0;
    if (usesAdaptivePacketRepresentation())
    {
        auto const& old_carry = *fields.get(ResolvedRepresentationCarryFieldName, 0);
        representation_carry.define(old_carry.boxArray(), old_carry.DistributionMap(), 4, 1);
        amrex::MultiFab::Copy(representation_carry, old_carry, 0, 0, 4, 1);
        amrex::MultiFab total(scattering.boxArray(), scattering.DistributionMap(), 1, 1);
        amrex::MultiFab::Copy(total, scattering, 0, 0, 1, 1);
        amrex::MultiFab::Add(total, absorption, 0, 0, 1, 1);
        desired.define(mask.boxArray(), mask.DistributionMap(), 1, 1);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            TryProposeAdaptiveDiffusionMask(mask, total, geometry, desired,
                                            m_resolved_packet.adaptive_policy),
            "Thermal adaptive representation rejected its total-opacity proposal.");
        ConstrainThermalMask(mask, desired, total, frame.WorkVelocity(), geometry, dt, false);
        amrex::MultiFab old_q(energy.boxArray(), energy.DistributionMap(), 3,
                              1);
        amrex::MultiFab::Copy(old_q, radiation, 1, 0, 3, 1);
#if defined(WARPX_DIM_RZ)
        // Current physical T/Cv/density/rate can change during ordinary hybrid
        // evolution. An admitted D population must never be trapped by a bad
        // next-entry proper source state: conservatively demote before paths.
        // This is a scope upper bound, NOT a replacement for direct W*a*T^4.
        amrex::MultiFab bath_bound(energy.boxArray(), energy.DistributionMap(),
                                   1, 0);
        bath_bound.setVal(0);
        ResolvedRZThermalDiffusionFieldOptions preflight_options;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            TryBuildResolvedRZThermalBathUpperBound(live_temperature,
                                                    bath_bound, geometry) &&
                TryConstrainResolvedRZThermalDiffusionMask(
                    energy, old_q, bath_bound, total, absorption, mask, desired,
                    frame.WorkVelocity(), geometry, dt, preflight_options),
            "Adaptive RZ thermal retained-D constitutive preflight rejected "
            "before paths.");
#endif
        amrex::iMultiFab early(mask.boxArray(), mask.DistributionMap(), 1, 1);
        amrex::iMultiFab::Copy(early, desired, 0, 0, 1, 1);
        ConstrainThermalMask(mask, early, total, frame.WorkVelocity(), geometry,
                             dt, true);
        auto converted = TryConvertAdaptivePacketDiffusionRepresentation(
            start, energy, old_q, mask, early, old_carry, geometry, census,
            LogicalPhotonBirthKeyContext{m_particle_conversion_seed,
                                         m_resolved_packet.interval},
            m_resolved_packet.photon_energy);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            converted.valid,
            "Thermal adaptive demotion rejected before native publication: " + converted.reason);
        amrex::MultiFab::Copy(radiation, converted.energy, 0, 0, 1, 0);
        amrex::MultiFab::Copy(radiation, converted.momentum, 0, 1, 3, 0);
        radiation.FillBoundary(geometry.periodicity());
        mask = std::move(converted.mask);
        representation_carry = std::move(converted.representation_carry);
        demoted = converted.demoted_cells;
    }
#if defined(WARPX_DIM_RZ)
    NativeResolvedRZThermalIntervalOptions options;
    options.reflecting_axial_fixed_mask_trial = m_resolved_packet.thermal_reflecting_z;
    options.thermal.rz_closed_reciprocal_spatial_endpoint_trial = m_resolved_packet.thermal_reflecting_z;
    options.thermal.packets.rz_reflecting_axial_reciprocal_endpoint = m_resolved_packet.thermal_reflecting_z;
    options.field.reflecting_axial_field_trial = m_resolved_packet.thermal_reflecting_z;
    if (usesAdaptivePacketRepresentation())
    {
        options.desired_diffusion_mask = &desired;
        options.radiation_representation_carry = &representation_carry;
        options.census = census;
    }
#else
    NativeResolvedThermalIntervalOptions options;
    options.census = census;
#endif
    options.thermal.packets.seed = m_particle_conversion_seed;
    options.thermal.packets.interval = m_resolved_packet.interval;
    options.thermal.packets.photon_energy = m_resolved_packet.photon_energy;
    options.thermal.packets.emission_rays_per_cell = m_resolved_packet.thermal_emission_rays;
    options.thermal.caloric_backend = m_resolved_packet.thermal_reciprocal_nodal
                                          ? NativeThermalCaloricBackend::ReciprocalNodalEmission
                                          : NativeThermalCaloricBackend::TrialCvCellMean;
    options.thermal.reciprocal_emission_rays_per_corner =
        m_resolved_packet.thermal_reciprocal_rays_per_corner;
    // Actual hybrid macrosteps can leave shared valid temperature copies a
    // few ULPs apart. Preserve their original corner energy in the opt-in
    // reciprocal model; never repair the live temperature before acceptance.
    options.thermal.reciprocal_composite_old_temperature =
        m_resolved_packet.thermal_reciprocal_nodal;
    options.thermal.max_temperature_iterations = m_lte_exchange_max_iterations;
    options.thermal.max_linear_iterations = m_resolved_packet.thermal_linear_iterations;
    options.thermal.max_material_iterations = m_resolved_packet.max_iterations;
    options.thermal.material_tolerance = m_lte_exchange_tolerance;
    options.thermal.work_tolerance = m_lte_exchange_tolerance;
    options.thermal.energy_tolerance = m_lte_exchange_tolerance;
    options.thermal.minimum_electron_density = m_minimum_electron_density;
#if defined(WARPX_DIM_RZ)
    options.thermal.rz_native_nodal_force = m_resolved_packet.thermal_native_nodal_force;
#else
    options.thermal.cartesian_native_nodal_force = m_resolved_packet.thermal_native_nodal_force;
#endif
    options.collisions.seed = m_particle_conversion_seed;
    options.collisions.interval = m_resolved_packet.interval;
    options.collisions.moving_cross_sections = true;
    options.packet_subsegment_limit = m_resolved_packet.packet_subsegment_limit;
#if defined(WARPX_DIM_RZ)
    auto result = TryAdvanceNativeResolvedRZThermalInterval(
        live, start, material, "diffusion_0", radiation, mask, live_temperature, scattering,
        absorption, frame.WorkVelocity(), *m_hybrid_model, geometry, dt, options);
#else
    auto result = TryAdvanceNativeResolvedThermalInterval(
        live, start, material, "diffusion_0", radiation,
        usesAdaptivePacketRepresentation() ? desired : mask, live_temperature, scattering,
        absorption, frame.WorkVelocity(), *m_hybrid_model, geometry, dt, options);
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(result.valid,
                                     "Resolved thermal packet-FLD interval rejected before "
                                     "joint native publication: " +
                                         result.reason);
    auto& candidate = result.candidate;
    if (usesAdaptivePacketRepresentation())
    {
#if !defined(WARPX_DIM_RZ)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            BookThermalCensusRepresentation(result, desired, mask, representation_carry),
            "Thermal adaptive census exceeded its actual addition-roundoff account.");
        mask.FillBoundary(geometry.periodicity());
        representation_carry.FillBoundary(geometry.periodicity());
#else
        mask = std::move(result.accepted_diffusion_mask);
        representation_carry = std::move(result.radiation_representation_carry);
#endif
    }
    amrex::GpuArray<amrex::Real, 4> before{}, after{};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        detail::PhotonInventory(live, before) &&
            detail::PhotonInventory(candidate.photons->Particles(), after),
        "Invalid actual photon inventory in thermal runtime acceptance.");
    for (int d = 0; d < 4; ++d)
    {
        before[d] += original_fields[d];
        after[d] += candidate.diffusion_radiation.sum(d);
    }
    amrex::GpuArray<amrex::Real, 4> representation_change{};
    if (usesAdaptivePacketRepresentation())
    {
        auto const& old_carry = *fields.get(ResolvedRepresentationCarryFieldName, 0);
        amrex::MultiFab change(old_carry.boxArray(), old_carry.DistributionMap(), 4, 0);
        amrex::MultiFab::Copy(change, representation_carry, 0, 0, 4, 0);
        amrex::MultiFab::Subtract(change, old_carry, 0, 0, 4, 0);
        for (int d = 0; d < 4; ++d)
        {
            representation_change[d] = change.sum(d);
        }
    }
    auto const work = candidate.material_stage->ActualWork().sum(0);
    auto const carry = candidate.material_stage->EnergyCarryChange().sum(0);
    auto const heat = candidate.actual_heat.sum(0);
    auto const defect = after[0] - before[0] + representation_change[0] + work + carry + heat;
    auto const scale = std::abs(before[0]) + std::abs(after[0]) + std::abs(work) + std::abs(carry) +
                       candidate.actual_heat.norm1(0);
    bool balanced =
        std::isfinite(defect) && std::isfinite(scale) &&
        std::abs(defect) <= m_lte_exchange_tolerance * scale + candidate.energy_roundoff_bound;
#if defined(WARPX_DIM_RZ)
    // The RZ interval independently remeasures actual photons/D/native ions
    // and carries in E/Pz/Lz, including its wall account. Summed radial or
    // unweighted theta momentum is not a global cylindrical invariant.
    for (auto const residual : result.raw_inventory_residual)
    {
        balanced = balanced && std::isfinite(residual);
    }
#else
    for (int d = 0; d < 3; ++d)
    {
        auto const residual =
            (after[d + 1] - before[d + 1] + representation_change[d + 1]) / PhysConst::c +
            candidate.material_stage->ActualImpulse().sum(d) +
            candidate.material_stage->MomentumCarryChange().sum(d);
        balanced = balanced && std::isfinite(residual) &&
                   std::abs(residual) <= m_lte_exchange_tolerance * scale / PhysConst::c;
    }
#endif
    amrex::ParallelDescriptor::ReduceBoolAnd(balanced);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        balanced, "Combined thermal packet/D/EOS/native work/carry ledger rejected.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(candidate.photons->ValidateLive(),
                                     "Live photons changed before thermal joint commit.");
#if defined(WARPX_DIM_RZ)
    ResolvedThermalAxialWallAccounts accepted_walls;
    std::vector<ResolvedThermalAxialWallSurface> accepted_surfaces;
    if (m_resolved_packet.thermal_reflecting_z)
    {
        bool valid_walls = PrepareClosedWallAccounts(m_resolved_packet.thermal_axial_wall,
            {result.path.path.axial_wall_momentum_by_side[0], result.path.path.axial_wall_momentum_by_side[1]},
            result.path.path.wall_momentum[2],
            {result.diffusion_wall_momentum_by_side[0], result.diffusion_wall_momentum_by_side[1]},
            result.diffusion_wall_momentum, result.axial_wall_momentum_operand,
            result.axial_wall_roundoff_bound / PhysConst::c, m_resolved_packet.interval + 1,
            accepted_walls);
        amrex::ParallelDescriptor::ReduceBoolAnd(valid_walls);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(valid_walls,
            "Closed thermal interval wall accounts rejected before joint publication.");
        if (m_resolved_packet.thermal_axial_wall_witness)
        {
            accepted_surfaces.push_back(ClosedWallSurface(candidate, result.path.path, geometry, dt));
        }
    }
#endif
    // All numerical/physical rejection gates precede these publications.
    candidate.material_stage->Commit();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(material.Commit(),
                                     "Live material changed before thermal joint commit.");
    candidate.photons->Commit();
    m_hybrid_model->CommitElectronTemperature(0, candidate.nodal_temperature);
    amrex::MultiFab::Copy(energy, candidate.diffusion_radiation, 0, 0, 1, 0);
    energy.FillBoundary(geometry.periodicity());
    for (int d = 0; d < 3; ++d)
    {
        auto& q = *fields.get(ResolvedMomentumFieldName(d), 0);
        amrex::MultiFab::Copy(q, candidate.diffusion_radiation, d + 1, 0, 1, 0);
        q.FillBoundary(geometry.periodicity());
    }
    if (usesAdaptivePacketRepresentation())
    {
        auto& accepted_mask = *fields.get(ResolvedMaskFieldName, 0);
        PublishThermalMask(mask, accepted_mask, geometry);
        amrex::MultiFab::Copy(*fields.get(ResolvedRepresentationCarryFieldName, 0),
                              representation_carry, 0, 0, 4, 1);
    }
    amrex::MultiFab::Copy(*fields.get(FieldType::radiation_material_energy, 0),
                          candidate.actual_heat, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*fields.get(FieldType::radiation_material_momentum, 0),
                          candidate.material_stage->ActualImpulse(), 0, 0, 3, 0);
    amrex::MultiFab::Copy(*fields.get(FieldType::radiation_material_kinetic_energy, 0),
                          candidate.material_stage->ActualWork(), 0, 0, 1, 0);
#if !defined(WARPX_DIM_RZ)
    if (result.census.admitted_four_inventory.ok())
    {
        amrex::MultiFab::Add(*fields.get(ResolvedAdmissionFieldName, 0),
                             result.census.admitted_four_inventory, 0, 0, 1, 0);
    }
#else
    if (result.admitted_four_inventory.ok())
    {
        amrex::MultiFab::Add(*fields.get(ResolvedAdmissionFieldName, 0),
                             result.admitted_four_inventory, 0, 0, 1, 0);
    }
#endif
    amrex::MultiFab::Copy(*fields.get(ResolvedOpacityFieldName, 0), scattering, 0, 0, 1, 1);
    amrex::MultiFab::Copy(*fields.get(ResolvedAbsorptionFieldName, 0), absorption, 0, 0, 1, 1);
    amrex::MultiFab::Copy(*fields.get(ResolvedElectronDensityFieldName, 0), ne, 0, 0, 1, 1);
    amrex::MultiFab::Copy(*fields.get(ResolvedElectronTemperatureFieldName, 0), te, 0, 0, 1, 1);
    m_resolved_packet.opacity_time = m_resolved_packet.time_dependent_opacity ? endpoint : 0;
    m_resolved_packet.thermal_opacity_time = endpoint;
    m_last_numerical_energy_residual = defect;
    m_cumulative_numerical_energy_residual += defect;
    m_last_material_carry_energy_change = carry;
    m_last_boundary_energy_loss = m_last_diffusion_boundary_energy_loss =
        m_last_streaming_boundary_energy_loss = 0;
    m_last_boundary_energy_injection = 0;
    m_last_diffusion_boundary_momentum_loss = m_last_streaming_boundary_momentum_loss =
        m_last_particle_wall_momentum = {0, 0, 0};
#if defined(WARPX_DIM_RZ)
    if (m_resolved_packet.thermal_reflecting_z)
    {
        m_resolved_packet.thermal_axial_wall = accepted_walls;
        m_resolved_packet.thermal_axial_wall_surfaces.swap(accepted_surfaces);
        m_last_streaming_boundary_momentum_loss[2] = accepted_walls.kinetic_net;
        m_last_diffusion_boundary_momentum_loss[2] = accepted_walls.diffusion_net;
    }
    else
    {
        m_last_particle_wall_momentum = result.path.path.wall_momentum;
    }
#endif
    m_last_coupled_solve = {candidate.material_iterations,      1,     0, 0,
                            candidate.maximum_caloric_residual, defect};
    m_last_implicit_diffusion = {};
#if defined(WARPX_DIM_RZ)
    m_last_implicit_diffusion.linear_iterations = candidate.rz_field_linear_iterations;
    m_last_implicit_diffusion.maximum_relative_residual = candidate.rz_field_linear_residual;
#else
    m_last_implicit_diffusion.linear_iterations = result.field.linear_iterations;
    m_last_implicit_diffusion.maximum_relative_residual = result.field.linear_residual;
#endif
    ++m_resolved_packet.interval;
    if (m_resolved_packet.verbosity > 0)
    {
        amrex::Print()
            << "Resolved thermal accepted interval="
            << m_resolved_packet.interval << " heat=" << heat
            << " finite_work=" << work << " signed_energy_defect_J=" << defect
            << " caloric_iterations=" << candidate.temperature_iterations
            << " demoted=" << demoted
#if defined(WARPX_DIM_RZ)
            << " leakage_births=" << candidate.rz_leakage->staged_births
            << " wall_reflections=" << result.path.path.wall_reflections
            << " packet_subsegments=" << result.path.subsegments
            << " promoted=" << result.promoted_cells
            << " admitted=" << result.admitted_photons
            << " deferred=" << result.deferred_cells
#else
                       << " deferred=" << result.deferred_admissions
#endif
            << '\n';
    }
}
