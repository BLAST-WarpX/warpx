/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeRZIndependentGrayOpacitySlice.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "NativeCompositeTableCornerCapture.H"
#include "NativePhotonReplayInternals.H"
#include "PhotonPacketTransaction.H"
#include "WarpX.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace warpx::radiation
{
struct NativeRZIndependentGrayOpacitySlice::FrozenPhotonOwners
{
    photon_replay_detail::CapturedParticles particles;
    std::vector<std::string> real_names, int_names;
};

NativeRZIndependentGrayOpacitySlice::~NativeRZIndependentGrayOpacitySlice () = default;

namespace
{
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU) && \
    defined(WARPX_USE_SINGULARITY_EOS) && defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
bool Collective (bool valid)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}

std::unique_ptr<amrex::MultiFab const> Copy (amrex::MultiFab const& source)
{
    auto copy = std::make_unique<amrex::MultiFab>(source.boxArray(), source.DistributionMap(),
                                                source.nComp(), source.nGrowVect());
    amrex::MultiFab::Copy(*copy, source, 0, 0, source.nComp(), source.nGrowVect());
    return copy;
}

bool SameFields (amrex::MultiFab const& left, amrex::MultiFab const& right)
{
    bool valid = left.ok() && right.ok() && left.boxArray() == right.boxArray() &&
        left.DistributionMap() == right.DistributionMap() && left.nComp() == right.nComp() &&
        left.nGrowVect() == right.nGrowVect() && left.ixType() == right.ixType();
    if (!Collective(valid)) { return false; }
    for (amrex::MFIter it(left); it.isValid(); ++it)
    {
        auto const count = static_cast<std::size_t>(left[it].box().numPts()) * left.nComp();
        valid = valid && std::memcmp(left[it].dataPtr(), right[it].dataPtr(),
                                     count * sizeof(amrex::Real)) == 0;
    }
    return Collective(valid);
}

bool SameMasks (amrex::iMultiFab const& left, amrex::iMultiFab const& right)
{
    bool valid = left.ok() && right.ok() && left.boxArray() == right.boxArray() &&
        left.DistributionMap() == right.DistributionMap() && left.nComp() == 1 &&
        right.nComp() == 1 && left.nGrowVect() == right.nGrowVect();
    if (!Collective(valid)) { return false; }
    for (amrex::MFIter it(left); it.isValid(); ++it)
    {
        auto const count = static_cast<std::size_t>(left[it].box().numPts());
        valid = valid && std::memcmp(left[it].dataPtr(), right[it].dataPtr(),
                                     count * sizeof(int)) == 0;
    }
    return Collective(valid);
}

bool SameConsumedCorners (amrex::MultiFab const& left, amrex::MultiFab const& right,
                         amrex::BoxArray const& cells, amrex::DistributionMapping const& mapping)
{
    auto const nodes = amrex::convert(cells, amrex::IntVect::TheNodeVector());
    bool valid = left.ok() && right.ok() && left.nComp() >= 1 && right.nComp() >= 1 &&
        left.boxArray() == nodes && right.boxArray() == nodes &&
        left.DistributionMap() == mapping && right.DistributionMap() == mapping;
    if (!Collective(valid)) { return false; }
    for (amrex::MFIter it(left); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(cells[it.index()]); cell.ok(); ++cell)
        {
            for (int corner = 0; corner < 4; ++corner)
            {
                auto node = cell(); node[0] += corner & 1; node[1] += (corner >> 1) & 1;
                auto const a = left[it](node), b = right[it](node);
                valid = valid && std::memcmp(&a, &b, sizeof(a)) == 0;
            }
        }
    }
    return Collective(valid);
}

// Exact loaded-owner contents/interpretation, not a path name or executor
// address. The initialized host accessor is never called on an absent owner.
std::vector<std::uint8_t> TableIdentity (MaterialOpacityTable const& table)
{
    if (!table.initialized()) { return {}; }
    auto const view = table.hostExecutor();
    if (view.m_num_density < 1 || view.m_num_temperature < 1 || view.m_num_groups != 1 ||
        !view.m_log_density || !view.m_log_temperature || !view.m_log_planck_absorption ||
        !view.m_log_planck_emission || !view.m_log_rosseland_transport || !view.m_log_scattering)
    { return {}; }
    std::vector<std::uint8_t> bytes;
    auto const append = [&] (void const* input, std::size_t count)
    {
        auto const* data = static_cast<std::uint8_t const*>(input);
        if (count != 0) { bytes.insert(bytes.end(), data, data + count); }
    };
    auto const scalar = [&] (auto value) { append(&value, sizeof(value)); };
    auto const values = [&] (auto const& input)
    {
        scalar(input.size());
        append(input.data(), input.size() * sizeof(*input.data()));
    };
    scalar(table.materialId()); values(table.materialKey());
    scalar(table.meanAtomicMassKg()); scalar(table.meanAtomicNumber());
    values(table.groupEdges()); values(table.representativeEnergies());
    values(table.rawDensityAxis()); values(table.rawTemperatureAxis());
    scalar(view.m_num_density); scalar(view.m_num_temperature); scalar(view.m_num_groups);
    scalar(view.m_interpolation_mode);
    append(view.m_log_density, static_cast<std::size_t>(view.m_num_density) * sizeof(amrex::Real));
    append(view.m_log_temperature, static_cast<std::size_t>(view.m_num_temperature) * sizeof(amrex::Real));
    auto const count = static_cast<std::size_t>(view.m_num_density) * view.m_num_temperature;
    for (auto const* channel : {view.m_log_planck_absorption, view.m_log_planck_emission,
                                view.m_log_rosseland_transport, view.m_log_scattering})
    { append(channel, count * sizeof(amrex::Real)); }
    scalar(view.m_zero_mask != nullptr);
    if (view.m_zero_mask) { append(view.m_zero_mask, count * sizeof(std::uint8_t)); }
    return bytes;
}
#endif
} // namespace

NativeRZIndependentGrayOpacitySlice::NativeRZIndependentGrayOpacitySlice (
    NativeRZThermalHistoryStepContext const& context,
    NativeRZThermalAllDNoKEventWitness const* witness,
    PhotonPacketTransaction const& source_photons,
    amrex::MultiFab const& raw_charge, amrex::MultiFab const& left_temperature,
    amrex::MultiFab const& unit_ion_charge, MaterialOpacityTable const& table,
    HybridPICModel const& native_model, ElectronThermodynamicsExecutor const& native_eos,
    amrex::Geometry const& geometry, IndependentGrayMovingAllDFieldOptions const& options,
    std::unique_ptr<amrex::MultiFab const> charge,
    std::unique_ptr<amrex::MultiFab const> temperature,
    std::unique_ptr<amrex::MultiFab const> unit,
    std::unique_ptr<IndependentGrayMovingFieldOpacity const> opacity,
    std::vector<std::uint8_t>&& table_identity, WarpXParticleContainer const* live_photons,
    amrex::iMultiFab const* original_mask, std::unique_ptr<amrex::iMultiFab const> frozen_mask,
    std::unique_ptr<FrozenPhotonOwners const> source_photons_capture)
    : m_context(&context), m_witness(witness), m_source_photons(&source_photons),
      m_original_charge(&raw_charge), m_original_temperature(&left_temperature),
      m_original_unit(&unit_ion_charge), m_table(&table), m_model(&native_model),
      m_eos(native_eos), m_geometry(geometry), m_options(options),
      m_charge(std::move(charge)), m_temperature(std::move(temperature)), m_unit(std::move(unit)),
      m_opacity(std::move(opacity)), m_table_identity(std::move(table_identity)),
      m_live_photons(live_photons), m_original_mask(original_mask),
      m_frozen_mask(std::move(frozen_mask)),
      m_source_photons_capture(std::move(source_photons_capture))
{}

std::unique_ptr<NativeRZIndependentGrayOpacitySlice const>
NativeRZIndependentGrayOpacitySlice::Issue (
    NativeRZThermalHistoryStepContext const& context,
    NativeRZThermalAllDNoKEventWitness const& witness,
    PhotonPacketTransaction const& source_photons,
    amrex::MultiFab const& raw_charge, amrex::MultiFab const& left_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table, amrex::Geometry const& geometry,
    NativeRZThermalPhotonTimePolicy photon_time_policy,
    IndependentGrayMovingAllDFieldOptions const& options, std::string& reason)
{
    return IssueEpoch(context, &witness, nullptr, source_photons, nullptr,
        raw_charge, left_temperature, unit_ion_charge, table, geometry,
        photon_time_policy, options, reason);
}

std::unique_ptr<NativeRZIndependentGrayOpacitySlice const>
NativeRZIndependentGrayOpacitySlice::IssueMixed (
    NativeRZThermalHistoryStepContext const& context, WarpXParticleContainer const& live,
    PhotonPacketTransaction const& source_photons, amrex::iMultiFab const& fixed_mask,
    amrex::MultiFab const& raw_charge, amrex::MultiFab const& left_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table, amrex::Geometry const& geometry,
    NativeRZThermalPhotonTimePolicy photon_time_policy,
    IndependentGrayMovingAllDFieldOptions const& options, std::string& reason)
{
    return IssueEpoch(context, nullptr, &live, source_photons, &fixed_mask,
        raw_charge, left_temperature, unit_ion_charge, table, geometry,
        photon_time_policy, options, reason);
}

std::unique_ptr<NativeRZIndependentGrayOpacitySlice const>
NativeRZIndependentGrayOpacitySlice::IssueEpoch (
    NativeRZThermalHistoryStepContext const& context,
    NativeRZThermalAllDNoKEventWitness const* witness, WarpXParticleContainer const* live,
    PhotonPacketTransaction const& source_photons, amrex::iMultiFab const* fixed_mask,
    amrex::MultiFab const& raw_charge, amrex::MultiFab const& left_temperature,
    detail::ResolvedMaterialChargeFields const& unit_ion_charge,
    MaterialOpacityTable const& table, amrex::Geometry const& geometry,
    NativeRZThermalPhotonTimePolicy photon_time_policy,
    IndependentGrayMovingAllDFieldOptions const& options, std::string& reason)
{
    reason = "independent-gray slice owner requires witnessed CPU periodic true-TABLE all-D half-cell history";
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(context, witness, live, fixed_mask, source_photons,
                         raw_charge, left_temperature,
                        unit_ion_charge, table, geometry, photon_time_policy, options);
    return nullptr;
#else
    auto const* model = WarpX::GetInstance().get_pointer_HybridPICModel();
    int presence[]{witness != nullptr, live != nullptr, fixed_mask != nullptr};
    int maximum[]{presence[0], presence[1], presence[2]};
    amrex::ParallelDescriptor::ReduceIntMin(presence, 3);
    amrex::ParallelDescriptor::ReduceIntMax(maximum, 3);
    if (!Collective(std::equal(std::begin(presence), std::end(presence), maximum) &&
            ((presence[0] && !presence[1] && !presence[2]) ||
             (!presence[0] && presence[1] && presence[2])))) { return nullptr; }
    bool valid = model && photon_time_policy == NativeRZThermalPhotonTimePolicy::PhotonHalfCell &&
        geometry.isPeriodic(1) && !geometry.isPeriodic(0) && options.enabled &&
        raw_charge.ok() && left_temperature.ok() && unit_ion_charge[0] &&
        PhysConst::c * context.outerDt() <=
            .5 * std::min(geometry.CellSize(0), geometry.CellSize(1));
    if (!Collective(valid)) { return nullptr; }
    valid = unit_ion_charge[0]->ok() && raw_charge.nComp() >= 1 &&
        raw_charge.ixType().nodeCentered() && raw_charge.nGrowVect().allGE(1) &&
        left_temperature.nComp() == 1 && left_temperature.nGrowVect().allGE(1) &&
        left_temperature.boxArray() == raw_charge.boxArray() &&
        left_temperature.DistributionMap() == raw_charge.DistributionMap() &&
        unit_ion_charge[0]->nComp() == 1 && unit_ion_charge[0]->nGrowVect().allGE(1) &&
        unit_ion_charge[0]->boxArray() == raw_charge.boxArray() &&
        unit_ion_charge[0]->DistributionMap() == raw_charge.DistributionMap();
    if (!Collective(valid)) { return nullptr; }
    if (!ValidateNativeRZThermalHistoryStepContext(context, context.dt(), context.seed(),
            context.interval(), reason)) { return nullptr; }
    if (witness && !witness->ValidateStep(context, context.dt(), context.seed(), context.interval(),
            source_photons, witness->CanonicalMask(), reason)) { return nullptr; }
    if (live && !source_photons.ValidateSnapshotFor(*live)) { return nullptr; }
    auto charge = Copy(raw_charge), temperature = Copy(left_temperature), unit = Copy(*unit_ion_charge[0]);
    detail::ResolvedMaterialChargeFields copied_units{}; copied_units[0] = unit.get();
    auto opacity = fixed_mask
        ? IndependentGrayMovingFieldOpacity::IssueMixed(*charge, *temperature,
              copied_units, table, *fixed_mask, geometry, context.dt(), options, reason)
        : IndependentGrayMovingFieldOpacity::Issue(*charge, *temperature,
              copied_units, table, geometry, context.dt(), options, reason);
    if (!opacity) { return nullptr; }
    auto identity = TableIdentity(table);
    if (!Collective(!identity.empty())) { return nullptr; }
    auto const eos = model->electronThermodynamicsExecutor();
    std::unique_ptr<amrex::iMultiFab> mask;
    std::unique_ptr<FrozenPhotonOwners> photons;
    if (fixed_mask)
    {
        mask = std::make_unique<amrex::iMultiFab>(fixed_mask->boxArray(),
            fixed_mask->DistributionMap(), 1, fixed_mask->nGrowVect());
        amrex::iMultiFab::Copy(*mask, *fixed_mask, 0, 0, 1, fixed_mask->nGrowVect());
        photons = std::make_unique<FrozenPhotonOwners>();
        photons->particles = photon_replay_detail::CaptureParticles(source_photons.Particles());
        photons->real_names = source_photons.Particles().GetRealSoANames();
        photons->int_names = source_photons.Particles().GetIntSoANames();
    }
    auto result = std::unique_ptr<NativeRZIndependentGrayOpacitySlice const>(
        new NativeRZIndependentGrayOpacitySlice(context, witness, source_photons,
            raw_charge, left_temperature, *unit_ion_charge[0], table, *model, eos,
            geometry, options, std::move(charge), std::move(temperature), std::move(unit),
            std::move(opacity), std::move(identity), live, fixed_mask,
            std::move(mask), std::move(photons)));
    if (!result->Validate(context, source_photons,
            fixed_mask ? *fixed_mask : witness->CanonicalMask(), reason)) { return nullptr; }
    return result;
#endif
}

bool NativeRZIndependentGrayOpacitySlice::Validate (
    NativeRZThermalHistoryStepContext const& context,
    PhotonPacketTransaction const& current_photons,
    amrex::iMultiFab const& consumed_mask, std::string& reason) const
{
    reason = "independent-gray slice opacity/source/context owner changed";
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(context, current_photons, consumed_mask);
    return false;
#else
    bool valid = &context == m_context && m_model == WarpX::GetInstance().get_pointer_HybridPICModel();
    if (!Collective(valid)) { return false; }
    if (m_witness)
    {
        if (!m_witness->ValidateStep(context, context.dt(), context.seed(), context.interval(),
                current_photons, consumed_mask, reason)) { return false; }
        if (!m_witness->ValidateOwners(*m_source_photons, consumed_mask, reason)) { return false; }
    }
    else
    {
        if (!ValidateNativeRZThermalHistoryStepContext(context, context.dt(), context.seed(),
                context.interval(), reason)) { return false; }
        reason = "independent-gray slice opacity/source/context owner changed";
        bool const original_mask = SameMasks(*m_original_mask, *m_frozen_mask);
        bool const mask = SameMasks(consumed_mask, *m_frozen_mask);
        if (!original_mask || !mask) { return false; }
        if (!m_source_photons->ValidateSnapshotFor(*m_live_photons) ||
            !current_photons.ValidateSnapshotFor(*m_live_photons)) { return false; }
        auto const now = photon_replay_detail::CaptureParticles(m_source_photons->Particles());
        auto const& before = *m_source_photons_capture;
        valid = now.tiles.size() == before.particles.tiles.size() &&
            before.real_names == m_source_photons->Particles().GetRealSoANames() &&
            before.int_names == m_source_photons->Particles().GetIntSoANames();
        if (valid)
        {
            for (std::size_t n = 0; n < now.tiles.size(); ++n)
            {
                auto const& a = now.tiles[n]; auto const& b = before.particles.tiles[n];
                valid = valid && a.key == b.key && a.count == b.count &&
                    photon_replay_detail::SameBits(a.idcpu, b.idcpu) &&
                    photon_replay_detail::SameBits(a.real_data, b.real_data) &&
                    photon_replay_detail::SameBits(a.int_data, b.int_data);
            }
        }
        if (!Collective(valid)) { return false; }
    }
    auto const eos = m_model->electronThermodynamicsExecutor();
    auto const& native_geometry = WarpX::GetInstance().Geom(0);
    valid = eos.isSingularitySpiner() && eos.m_num_materials == 1 && eos.m_use_resolved_material_selection &&
        eos.tableStorage() == m_eos.tableStorage() &&
        eos.m_mass_per_charge[0] == m_eos.m_mass_per_charge[0] &&
        eos.m_mass_per_physical_charge[0] == m_eos.m_mass_per_physical_charge[0] &&
        eos.m_atomic_mass[0] == m_eos.m_atomic_mass[0] && eos.m_atomic_number[0] == m_eos.m_atomic_number[0] &&
        eos.minimumTemperature() == m_eos.minimumTemperature() &&
        eos.maximumTemperature() == m_eos.maximumTemperature() && TableIdentity(*m_table) == m_table_identity &&
        native_geometry.Domain() == m_geometry.Domain() && native_geometry.Coord() == m_geometry.Coord();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        valid = valid && native_geometry.ProbLo(d) == m_geometry.ProbLo(d) &&
            native_geometry.ProbHi(d) == m_geometry.ProbHi(d) &&
            native_geometry.isPeriodic(d) == m_geometry.isPeriodic(d);
    }
    // Each exact comparison is collective: do not place any of them behind a
    // rank-local && shortcut. Empty ranks still join every ownership check.
    bool const charge = SameFields(*m_original_charge, *m_charge);
    bool const temperature = SameFields(*m_original_temperature, *m_temperature);
    bool const unit = SameFields(*m_original_unit, *m_unit);
    valid = valid && charge && temperature && unit;
    if (!Collective(valid)) { return false; }
    reason.clear();
    return true;
#endif
}

bool NativeRZIndependentGrayOpacitySlice::ValidatePacketFrame (
    amrex::MultiFab const& velocity, amrex::Real scattering_limit,
    amrex::Real absorption_limit, std::string& reason) const
{
    reason = "independent mixed-opacity original outer photon optical/frame guard rejected";
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(velocity, scattering_limit, absorption_limit);
    return false;
#else
    if (!IsMixed() || !Validate(*m_context, *m_source_photons, *m_frozen_mask, reason))
    { return false; }
    reason = "independent mixed-opacity original outer photon optical/frame guard rejected";
    auto const& channels = m_opacity->Snapshot().coefficients;
    auto const& scattering = channels.true_scattering;
    bool valid = velocity.ok() && velocity.nComp() == 3 &&
        velocity.boxArray() == scattering.boxArray() &&
        velocity.DistributionMap() == scattering.DistributionMap() &&
        std::isfinite(scattering_limit) && scattering_limit > 0 && scattering_limit <= 16 &&
        std::isfinite(absorption_limit) && absorption_limit > 0 && absorption_limit <= 16;
    amrex::Real minimum[]{scattering_limit, absorption_limit};
    amrex::Real maximum[]{scattering_limit, absorption_limit};
    amrex::ParallelDescriptor::ReduceRealMin(minimum, 2);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 2);
    if (!Collective(valid && std::equal(std::begin(minimum), std::end(minimum), maximum)))
    { return false; }
    for (amrex::MFIter it(scattering); it.isValid(); ++it)
    {
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const index = cell();
            amrex::Real beta2 = 0;
            auto const lower = m_geometry.ProbLo(0) +
                (index[0] - m_geometry.Domain().smallEnd(0)) * m_geometry.CellSize(0);
            for (int d = 0; d < 3; ++d)
            {
                auto value = velocity[it](index, d);
                if (d == 1)
                { value *= (lower + m_geometry.CellSize(0)) /
                    RZAngularGeometry::MeanRadius(lower, lower + m_geometry.CellSize(0)); }
                beta2 += value * value / PhysConst::c2;
            }
            bool const frame = std::isfinite(beta2) && beta2 >= 0 && beta2 <= 1.e-6;
            valid = valid && frame;
            if (!frame) { continue; }
            auto const doppler = (1 + std::sqrt(beta2)) / std::sqrt(1 - beta2);
            auto const s = scattering[it](index) * PhysConst::c * m_context->outerDt() * doppler;
            auto const p = channels.proper_absorption[it](index) *
                PhysConst::c * m_context->outerDt() * doppler;
            valid = valid && std::isfinite(s + p) && s >= 0 && p >= 0 &&
                s <= scattering_limit && p <= absorption_limit;
        }
    }
    if (!Collective(valid))
    {
        reason = "independent mixed-opacity original outer photon optical/frame guard rejected";
        return false;
    }
    reason.clear();
    return true;
#endif
}

bool NativeRZIndependentGrayOpacitySlice::ValidateCaloricOwners (
    detail::NativeCompositeTableCornerCapture const& caloric, std::string& reason) const
{
    reason = "independent-gray opacity and native caloric consumed corners differ";
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(caloric);
    return false;
#else
    if (!Validate(*m_context, *m_source_photons,
            m_witness ? m_witness->CanonicalMask() : *m_frozen_mask, reason))
    { return false; }
    reason = "independent-gray opacity and native caloric consumed corners differ";
    if (!Collective(caloric.Valid())) { return false; }
    auto const& r = m_opacity->Snapshot().coefficients.rosseland_transport;
    bool const charge = SameConsumedCorners(*m_charge, caloric.FrozenCharge(), r.boxArray(), r.DistributionMap());
    bool const temperature = SameConsumedCorners(*m_temperature, caloric.FrozenTemperature(),
                                                r.boxArray(), r.DistributionMap());
    bool const unit = SameConsumedCorners(*m_unit, caloric.FrozenUnitIonCharge(0),
                                         r.boxArray(), r.DistributionMap());
    bool const mass = caloric.FrozenMassPerUnitCharge(0) == m_eos.m_mass_per_charge[0];
    if (!Collective(charge && temperature && unit && mass)) { return false; }
    reason.clear();
    return true;
#endif
}

ResolvedRZThermalDiffusionFieldResult NativeRZIndependentGrayOpacitySlice::Advance (
    NativeRZThermalHistoryStepContext const& context,
    PhotonPacketTransaction const& current_photons,
    amrex::MultiFab const& lab_energy, amrex::MultiFab const& lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::iMultiFab const& consumed_mask,
    amrex::MultiFab const& velocity, amrex::MultiFab const& actual_work) const
{
    ResolvedRZThermalDiffusionFieldResult rejected;
    if (!Validate(context, current_photons, consumed_mask, rejected.reason)) { return rejected; }
    auto result = m_opacity->Advance(lab_energy, lab_momentum, proper_bath,
                                     consumed_mask, velocity, actual_work);
    if (!Validate(context, current_photons, consumed_mask, rejected.reason)) { return rejected; }
    return result;
}

bool NativeRZIndependentGrayOpacitySlice::Preflight (
    NativeRZThermalHistoryStepContext const& context,
    PhotonPacketTransaction const& current_photons,
    amrex::MultiFab const& lab_energy, amrex::MultiFab const& lab_momentum,
    amrex::MultiFab const& proper_bath, amrex::iMultiFab const& consumed_mask,
    amrex::MultiFab const& velocity, amrex::MultiFab const& actual_work,
    std::string& reason) const
{
    if (!Validate(context, current_photons, consumed_mask, reason)) { return false; }
    bool const numerical = m_opacity->Preflight(lab_energy, lab_momentum, proper_bath,
                                               consumed_mask, velocity, actual_work, reason);
    auto numerical_reason = reason;
    bool const owners = Validate(context, current_photons, consumed_mask, reason);
    if (!numerical) { reason = std::move(numerical_reason); }
    return numerical && owners;
}
} // namespace warpx::radiation
