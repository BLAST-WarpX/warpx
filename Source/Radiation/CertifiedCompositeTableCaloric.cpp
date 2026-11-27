/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "CertifiedCompositeTableCaloric.H"

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::radiation::detail
{
namespace
{
using Corners = std::array<CompositeTableCorner, FrozenCompositeTableCaloricNode::max_corners>;

struct Difference
{
    amrex::Real joules = 0;
    amrex::Real bound = 0;
    bool valid = false;
};

bool
ValidState (ElectronThermodynamicState const& state)
{
    return std::isfinite(state.pressure) && state.pressure >= 0 &&
        std::isfinite(state.internal_energy_density) &&
        std::isfinite(state.heat_capacity_density) && state.heat_capacity_density > 0;
}

// Sum each physical corner's *centered* energy before rounding the composite
// result. The two Boolean selectors let the same independent evaluator form
// old-copy offset, trial-centered energy and actual old-to-new realization.
Difference
CornerDifference (ElectronThermodynamicsExecutor const& eos, Corners const& corners,
                  int count, amrex::Real reference, amrex::Real trial,
                  bool original_old, bool original_new)
{
    long double signed_joules = 0, unsigned_terms = 0;
    int active = 0;
    for (int c = 0; c < count; ++c)
    {
        auto const& corner = corners[c];
        if (corner.raw_charge == 0) { continue; }
        ++active;
        auto const first_temperature = original_old ? corner.old_temperature : reference;
        auto const last_temperature = original_new ? corner.old_temperature : trial;
        auto const first = eos.stateFromMaterialMassDensitiesTemperature(
            corner.effective_charge, corner.mass_density, first_temperature);
        auto const last = eos.stateFromMaterialMassDensitiesTemperature(
            corner.effective_charge, corner.mass_density, last_temperature);
        if (!ValidState(first) || !ValidState(last)) { return {}; }
        auto const measure = static_cast<long double>(corner.measure);
        auto const term = measure *
            (static_cast<long double>(last.internal_energy_density) -
             static_cast<long double>(first.internal_energy_density));
        signed_joules += term;
        unsigned_terms += std::abs(term);
    }
    // The table returns represented Real U values. Subtract before weighting
    // and sum in extended precision; the operation budget also covers hosts
    // where long double has no extra mantissa bits. Do not inflate this
    // *arithmetic* error by a signed table's arbitrary energy reference.
    // Independent EOS/physics accuracy remains the caller's responsibility.
    auto const eps = std::numeric_limits<long double>::epsilon();
    auto const operations = 8 * active + 16;
    auto const product = operations * eps;
    if (!(product < 1) || !std::isfinite(signed_joules) ||
        !std::isfinite(unsigned_terms))
    {
        return {};
    }
    auto const joules = static_cast<amrex::Real>(signed_joules);
    auto const gamma = product / (1 - product);
    auto const inf = std::numeric_limits<amrex::Real>::infinity();
    auto const rounding = std::max(std::abs(std::nextafter(joules, inf) - joules),
                                   std::abs(joules - std::nextafter(joules, -inf)));
    auto const bound = static_cast<amrex::Real>(gamma * unsigned_terms) + rounding;
    return {joules, bound, std::isfinite(joules) && std::isfinite(bound)};
}

[[maybe_unused]] CompositeTableInverseImage
InverseImage (ElectronThermodynamicsExecutor const& eos, Corners const& corners,
              int count, amrex::Real reference, amrex::Real temperature)
{
    auto left_lo = eos.minimumTemperature(), left_hi = eos.maximumTemperature();
    auto right_lo = left_lo, right_hi = left_hi;
    for (int iteration = 0; iteration < eos.table_inverse_iterations; ++iteration)
    {
        auto const left_trial = amrex::Real(0.5) * (left_lo + left_hi);
        auto const right_trial = amrex::Real(0.5) * (right_lo + right_hi);
        if (left_trial < temperature) { left_lo = left_trial; }
        else { left_hi = left_trial; }
        if (right_trial <= temperature) { right_lo = right_trial; }
        else { right_hi = right_trial; }
    }
    if (temperature == eos.minimumTemperature()) { left_lo = temperature; }
    if (temperature == eos.maximumTemperature()) { right_hi = temperature; }
    auto const lower = CornerDifference(eos, corners, count, reference, left_lo, false, false);
    auto const upper = CornerDifference(eos, corners, count, reference, right_hi, false, false);
    if (!lower.valid || !upper.valid || upper.joules < lower.joules) { return {}; }
    auto const inf = std::numeric_limits<amrex::Real>::infinity();
    return {left_lo, right_hi,
            std::nextafter(lower.joules - lower.bound, -inf),
            std::nextafter(upper.joules + upper.bound, inf), true};
}
} // namespace

FrozenCompositeTableCaloricNode
PrepareCompositeTableCaloricNode (ElectronThermodynamicsExecutor const& eos,
    std::array<CompositeTableCorner, FrozenCompositeTableCaloricNode::max_corners> const& corners,
    int count, amrex::Real reference_temperature)
{
    FrozenCompositeTableCaloricNode result;
#ifndef WARPX_USE_SINGULARITY_EOS
    amrex::ignore_unused(eos, corners, count, reference_temperature);
    result.m_failure = CompositeTableCaloricFailure::UnsupportedExecutor;
    return result;
#else
    if (!eos.isSingularitySpiner() || eos.tableStorage() == nullptr ||
        eos.m_num_materials <= 0 || eos.m_num_materials > eos.max_materials ||
        !std::isfinite(eos.minimumTemperature()) ||
        !std::isfinite(eos.maximumTemperature()) ||
        !(eos.maximumTemperature() > eos.minimumTemperature()))
    {
        result.m_failure = CompositeTableCaloricFailure::UnsupportedExecutor;
        return result;
    }
    if (count <= 0 || count > result.max_corners || !std::isfinite(reference_temperature))
    {
        result.m_failure = CompositeTableCaloricFailure::InvalidCorner;
        return result;
    }
    result.m_corners = corners;
    result.m_count = count;
    result.m_materials = eos.m_num_materials;
    result.m_resolved = eos.m_use_resolved_material_selection;
    result.m_tables = static_cast<void const*>(eos.tableStorage());
    result.m_selector = eos.m_resolved_material_selector;
    result.m_minimum_mass_density = eos.m_minimum_mass_density;
    result.m_maximum_mass_density = eos.m_maximum_mass_density;
    result.m_atomic_mass = eos.m_atomic_mass;
    result.m_atomic_number = eos.m_atomic_number;
    result.m_floor = eos.minimumTemperature();
    result.m_ceiling = eos.maximumTemperature();
    result.m_reference = reference_temperature;
    bool occupied = false;
    for (int c = 0; c < count; ++c)
    {
        auto const& corner = corners[c];
        bool any_material = false;
        for (int material = 0; material < eos.m_num_materials; ++material)
        {
            auto const mass = corner.mass_density[material];
            if (!std::isfinite(mass) || mass < 0)
            {
                result.m_failure = CompositeTableCaloricFailure::InvalidCorner;
                return result;
            }
            any_material = any_material || mass > 0;
        }
        if (!std::isfinite(corner.measure) || !(corner.measure > 0) ||
            !std::isfinite(corner.raw_charge + corner.effective_charge +
                           corner.old_temperature) ||
            corner.raw_charge < 0 || corner.effective_charge < corner.raw_charge ||
            (corner.raw_charge == 0 ?
                 (corner.effective_charge != 0 || any_material ||
                  corner.old_temperature != reference_temperature) : !any_material))
        {
            result.m_failure = CompositeTableCaloricFailure::InvalidCorner;
            return result;
        }
        if (corner.raw_charge == 0) { continue; }
        occupied = true;
        auto const copy_bound = 8 * std::numeric_limits<amrex::Real>::epsilon() *
            std::max({amrex::Real(1), std::abs(reference_temperature),
                      std::abs(corner.old_temperature)});
        if (std::abs(corner.old_temperature - reference_temperature) > copy_bound)
        {
            result.m_failure = CompositeTableCaloricFailure::InconsistentOldCopies;
            return result;
        }
        if (reference_temperature < result.m_floor ||
            reference_temperature > result.m_ceiling ||
            corner.old_temperature < result.m_floor ||
            corner.old_temperature > result.m_ceiling)
        {
            result.m_failure = CompositeTableCaloricFailure::InvalidEOS;
            return result;
        }
    }
    if (!occupied)
    {
        result.m_failure = CompositeTableCaloricFailure::None;
        return result;
    }
    auto const offset = CornerDifference(eos, corners, count, reference_temperature,
                                         reference_temperature, false, true);
    auto const minimum = CornerDifference(eos, corners, count, reference_temperature,
                                          result.m_floor, false, false);
    auto const maximum = CornerDifference(eos, corners, count, reference_temperature,
                                          result.m_ceiling, false, false);
    if (!offset.valid || !minimum.valid || !maximum.valid ||
        maximum.joules < minimum.joules ||
        offset.joules < minimum.joules || offset.joules > maximum.joules)
    {
        result.m_failure = CompositeTableCaloricFailure::InvalidEOS;
        return result;
    }
    result.m_old_offset = offset.joules;
    result.m_old_offset_bound = offset.bound;
    result.m_available = offset.joules - minimum.joules;
    if (!std::isfinite(result.m_available) || result.m_available < 0)
    {
        result.m_failure = CompositeTableCaloricFailure::InvalidEOS;
        return result;
    }
    result.m_failure = CompositeTableCaloricFailure::None;
    return result;
#endif
}

CompositeTableCaloricResult
EvaluateCompositeTableCaloricNode (ElectronThermodynamicsExecutor const& eos,
    FrozenCompositeTableCaloricNode const& old, amrex::Real requested_joules,
    amrex::Real const* prescribed_temperature)
{
    CompositeTableCaloricResult result;
    if (!old.Valid() || !std::isfinite(requested_joules))
    {
        result.failure = CompositeTableCaloricFailure::InvalidCorner;
        return result;
    }
#ifndef WARPX_USE_SINGULARITY_EOS
    amrex::ignore_unused(eos, prescribed_temperature);
    result.failure = CompositeTableCaloricFailure::UnsupportedExecutor;
    return result;
#else
    if (!eos.isSingularitySpiner() || eos.m_num_materials != old.m_materials ||
        eos.m_use_resolved_material_selection != old.m_resolved ||
        static_cast<void const*>(eos.tableStorage()) != old.m_tables ||
        eos.minimumTemperature() != old.m_floor ||
        eos.maximumTemperature() != old.m_ceiling)
    {
        result.failure = CompositeTableCaloricFailure::ChangedExecutor;
        return result;
    }
    auto const& selector = eos.m_resolved_material_selector;
    auto const& frozen_selector = old.m_selector;
    if (old.m_resolved &&
        (selector.num_materials != frozen_selector.num_materials ||
         selector.vacuum_mass_density != frozen_selector.vacuum_mass_density ||
         selector.mixed_cell_absolute_tolerance !=
             frozen_selector.mixed_cell_absolute_tolerance ||
         selector.mixed_cell_relative_tolerance !=
             frozen_selector.mixed_cell_relative_tolerance))
    {
        result.failure = CompositeTableCaloricFailure::ChangedExecutor;
        return result;
    }
    for (int material = 0; material < old.m_materials; ++material)
    {
        if (eos.m_minimum_mass_density[material] !=
                old.m_minimum_mass_density[material] ||
            eos.m_maximum_mass_density[material] !=
                old.m_maximum_mass_density[material] ||
            eos.m_atomic_mass[material] != old.m_atomic_mass[material] ||
            eos.m_atomic_number[material] != old.m_atomic_number[material])
        {
            result.failure = CompositeTableCaloricFailure::ChangedExecutor;
            return result;
        }
    }
    bool occupied = false;
    for (int c = 0; c < old.m_count; ++c)
    {
        occupied = occupied || old.m_corners[c].raw_charge > 0;
    }
    if (!occupied)
    {
        if (requested_joules != 0 ||
            (prescribed_temperature && *prescribed_temperature != old.m_reference))
        {
            result.failure = CompositeTableCaloricFailure::Unaffordable;
            return result;
        }
        result.failure = CompositeTableCaloricFailure::None;
        result.temperature = old.m_reference;
        result.converged = true;
        result.inverse_image = {old.m_reference, old.m_reference, 0, 0, true};
        return result;
    }
    auto const target = requested_joules + old.m_old_offset;
    auto const minimum = CornerDifference(eos, old.m_corners, old.m_count, old.m_reference,
                                          old.m_floor, false, false);
    auto const maximum = CornerDifference(eos, old.m_corners, old.m_count, old.m_reference,
                                          old.m_ceiling, false, false);
    if (!minimum.valid || !maximum.valid || !std::isfinite(target))
    {
        result.failure = CompositeTableCaloricFailure::InvalidEOS;
        return result;
    }
    if (!prescribed_temperature &&
        (target < minimum.joules || target > maximum.joules))
    {
        result.failure = CompositeTableCaloricFailure::Unaffordable;
        return result;
    }
    amrex::Real next_temperature = old.m_reference;
    if (prescribed_temperature)
    {
        next_temperature = *prescribed_temperature;
    }
    else if (target == 0 && old.m_old_offset == 0)
    {
        // Exact no-source roundtrip: do not manufacture inverse heat.
        next_temperature = old.m_reference;
    }
    else if (target == minimum.joules) { next_temperature = old.m_floor; }
    else if (target == maximum.joules) { next_temperature = old.m_ceiling; }
    else
    {
        auto lo = old.m_floor, hi = old.m_ceiling;
        for (int iteration = 0; iteration < eos.table_inverse_iterations; ++iteration)
        {
            auto const midpoint = amrex::Real(0.5) * (lo + hi);
            auto const centered = CornerDifference(eos, old.m_corners, old.m_count,
                                                   old.m_reference, midpoint, false, false);
            if (!centered.valid)
            {
                result.failure = CompositeTableCaloricFailure::InvalidEOS;
                return result;
            }
            if (centered.joules < target) { lo = midpoint; }
            else { hi = midpoint; }
        }
        next_temperature = amrex::Real(0.5) * (lo + hi);
    }
    if (!std::isfinite(next_temperature) || next_temperature < old.m_floor ||
        next_temperature > old.m_ceiling)
    {
        result.failure = CompositeTableCaloricFailure::InvalidEOS;
        return result;
    }
    auto const realized = CornerDifference(eos, old.m_corners, old.m_count,
                                           old.m_reference, next_temperature, true, false);
    auto const centered = CornerDifference(eos, old.m_corners, old.m_count,
                                           old.m_reference, next_temperature, false, false);
    if (!realized.valid || !centered.valid)
    {
        result.failure = CompositeTableCaloricFailure::InvalidEOS;
        return result;
    }
    result.temperature = next_temperature;
    result.realized_joules = realized.joules;
    result.residual_joules = realized.joules - requested_joules;
    result.arithmetic_bound_joules = realized.bound + centered.bound +
        old.m_old_offset_bound +
        std::abs(std::nextafter(target, std::numeric_limits<amrex::Real>::infinity()) - target);
    if (!prescribed_temperature)
    {
        result.inverse_image = InverseImage(eos, old.m_corners, old.m_count,
                                            old.m_reference, next_temperature);
        if (!result.inverse_image.valid ||
            target < result.inverse_image.lower_centered_joules ||
            target > result.inverse_image.upper_centered_joules)
        {
            result.failure = CompositeTableCaloricFailure::UncertifiedInverse;
            return result;
        }
        result.arithmetic_bound_joules +=
            result.inverse_image.upper_centered_joules -
            result.inverse_image.lower_centered_joules;
    }
    if (!std::isfinite(result.residual_joules + result.arithmetic_bound_joules))
    {
        result.failure = CompositeTableCaloricFailure::InvalidEOS;
        return result;
    }
    result.converged = std::abs(result.residual_joules) <= result.arithmetic_bound_joules;
    result.failure = CompositeTableCaloricFailure::None;
    return result;
#endif
}
} // namespace warpx::radiation::detail
