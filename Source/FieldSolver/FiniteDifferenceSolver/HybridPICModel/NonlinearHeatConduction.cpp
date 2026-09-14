/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NonlinearHeatConduction.H"

#include "Utils/TextMsg.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <limits>

namespace warpx::hybrid
{
using namespace amrex::literals;

namespace
{
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ElectronThermodynamicState
caloricState (ElectronThermodynamicsExecutor const& eos, amrex::Real charge, amrex::Real mass,
              amrex::Real temperature)
{
    ElectronThermodynamicsExecutor::MaterialMassDensities materials{};
    materials[0] = mass;
    return eos.stateFromMaterialMassDensitiesTemperature(charge, materials, temperature);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
validState (ElectronThermodynamicState const& state)
{
    return std::isfinite(state.internal_energy_density) &&
           std::isfinite(state.heat_capacity_density) && state.heat_capacity_density > 0;
}
} // namespace

NonlinearHeatSolveResult
tryNonlinearHeatConduction (amrex::MultiFab& temperature, amrex::MultiFab& energy,
                            amrex::MultiFab const& old_energy, amrex::MultiFab const& charge,
                            amrex::MultiFab const& outgoing, amrex::MultiFab const& incoming,
                            amrex::MultiFab const& volume, amrex::Geometry const& geometry,
                            ElectronThermodynamicsExecutor const& eos,
                            amrex::MultiFab const* material_mass_density)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        temperature.nComp() == 1 && temperature.ixType().nodeCentered() && energy.nComp() == 1 &&
            incoming.nComp() == 2 * AMREX_SPACEDIM &&
            (!eos.isSingularitySpiner() || (eos.m_num_materials == 1 && material_mass_density)),
        "Nonlinear heat conduction requires scalar nodal fields and explicit single-material table "
        "density.");
    for (auto const* field : {static_cast<amrex::MultiFab const*>(&energy), &old_energy, &charge,
                              &outgoing, &incoming, &volume})
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            field->boxArray() == temperature.boxArray() &&
                field->DistributionMap() == temperature.DistributionMap(),
            "Nonlinear heat-conduction fields must have matching layouts.");
    }
    if (material_mass_density)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            material_mass_density->nComp() == 1 &&
                material_mass_density->boxArray() == temperature.boxArray() &&
                material_mass_density->DistributionMap() == temperature.DistributionMap(),
            "Nonlinear heat-conduction material density must match temperature.");
    }
    NonlinearHeatSolveResult result;
    if (!temperature.is_finite() || temperature.min(0) < 0 || !volume.is_finite() ||
        volume.min(0) <= 0)
    {
        return result;
    }
    auto const lower = amrex::max(temperature.min(0), eos.minimumTemperature());
    auto const upper = amrex::min(temperature.max(0), eos.maximumTemperature());
    if (!(lower <= upper))
    {
        return result;
    }
    auto const& ba = temperature.boxArray();
    auto const& dm = temperature.DistributionMap();
    amrex::MultiFab current(ba, dm, 1, 1), next(ba, dm, 1, 1);
    amrex::MultiFab errors(ba, dm, 1, 0), trial_energy(ba, dm, 1, 0);
    amrex::MultiFab::Copy(current, temperature, 0, 0, 1, 0);
    constexpr auto epsilon = std::numeric_limits<amrex::Real>::epsilon();
    constexpr auto tolerance = 128 * epsilon;
    bool const tabulated = eos.isSingularitySpiner();
    bool converged = false;
    for (int iteration = 0; iteration < 512; ++iteration)
    {
        ++result.iterations;
        current.FillBoundary(geometry.periodicity());
        for (amrex::MFIter it(current); it.isValid(); ++it)
        {
            auto const t = current.const_array(it), old = old_energy.const_array(it);
            auto const rho = charge.const_array(it), out = outgoing.const_array(it);
            auto const in = incoming.const_array(it);
            auto const mass = material_mass_density ? material_mass_density->const_array(it)
                                                    : amrex::Array4<amrex::Real const>{};
            auto const update = next.array(it), error = errors.array(it);
            auto const u = trial_energy.array(it);
            amrex::ParallelFor(
                it.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    auto const old_t = t(i, j, k), density = rho(i, j, k);
                    auto const diagonal = out(i, j, k);
                    auto const material = tabulated ? mass(i, j, k) : 0;
                    auto rhs = old(i, j, k);
                    bool valid = std::isfinite(rhs) && std::isfinite(density) && density >= 0 &&
                                 std::isfinite(diagonal) && diagonal >= 0 && std::isfinite(old_t) &&
                                 std::isfinite(material) && material >= 0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        for (int side = 0; side < 2; ++side)
                        {
                            auto const coefficient = in(i, j, k, 2 * d + side);
                            valid = valid && std::isfinite(coefficient) && coefficient >= 0;
                            if (coefficient > 0)
                            {
                                auto const offset = side == 0 ? -1 : 1;
                                rhs += coefficient * t(i + (d == 0 ? offset : 0),
                                                       j + (d == 1 ? offset : 0),
                                                       k + (d == 2 ? offset : 0));
                            }
                        }
                    }
                    if (density == 0)
                    {
                        valid = valid && rhs == 0 && diagonal == 0 && material == 0;
                        update(i, j, k) = old_t;
                        u(i, j, k) = 0;
                        error(i, j, k) = valid ? 0 : std::numeric_limits<amrex::Real>::infinity();
                        return;
                    }
                    auto const before = caloricState(eos, density, material, old_t);
                    valid = valid && std::isfinite(rhs) && validState(before);
                    auto const lhs = before.internal_energy_density + diagonal * old_t;
                    auto const scale = std::abs(before.internal_energy_density) +
                                       std::abs(diagonal * old_t) + std::abs(rhs);
                    auto const residual = std::abs(lhs - rhs);
                    u(i, j, k) = before.internal_energy_density;
                    auto lo = lower, hi = upper, value = old_t;
                    bool solved = false;
                    for (int solve = 0; solve < 80 && valid; ++solve)
                    {
                        auto const state = caloricState(eos, density, material, value);
                        valid = valid && validState(state);
                        auto const f = state.internal_energy_density + diagonal * value - rhs;
                        auto const bound = 16 * epsilon *
                                           (std::abs(state.internal_energy_density) +
                                            std::abs(diagonal * value) + std::abs(rhs));
                        if (valid && std::abs(f) <= bound)
                        {
                            solved = true;
                            break;
                        }
                        if (f < 0)
                        {
                            lo = value;
                        }
                        else
                        {
                            hi = value;
                        }
                        auto const proposal = value - f / (state.heat_capacity_density + diagonal);
                        value = std::isfinite(proposal) && proposal > lo && proposal < hi
                                    ? proposal
                                    : lo + (hi - lo) / 2;
                    }
                    update(i, j, k) = value;
                    error(i, j, k) =
                        valid && solved && std::isfinite(residual) && std::isfinite(scale)
                            ? (scale > 0 ? residual / scale : residual)
                            : std::numeric_limits<amrex::Real>::infinity();
                });
        }
        auto const residual = errors.norm0(0);
        if (!std::isfinite(residual))
        {
            return result;
        }
        if (residual <= tolerance)
        {
            converged = true;
            break;
        }
        amrex::MultiFab::Swap(current, next, 0, 0, 1, 1);
    }
    if (!converged)
    {
        return result;
    }
    auto const owner = temperature.OwnerMask(geometry.periodicity());
    amrex::MultiFab inventory(ba, dm, 3, 0);
    for (amrex::MFIter it(inventory); it.isValid(); ++it)
    {
        auto const old = old_energy.const_array(it), u = trial_energy.const_array(it);
        auto const t = current.const_array(it), out = outgoing.const_array(it);
        auto const measure = volume.const_array(it);
        auto const owned = owner->const_array(it);
        auto const sums = inventory.array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               auto const v = owned(i, j, k) ? measure(i, j, k) : 0;
                               sums(i, j, k, 0) = v * old(i, j, k);
                               sums(i, j, k, 1) = v * u(i, j, k);
                               sums(i, j, k, 2) =
                                   v * (std::abs(old(i, j, k)) + std::abs(u(i, j, k)) +
                                        std::abs(out(i, j, k) * t(i, j, k)));
                           });
    }
    auto const before = inventory.sum(0), after = inventory.sum(1), scale = inventory.sum(2);
    if (!std::isfinite(before) || !std::isfinite(after) || !std::isfinite(scale) || scale < 0 ||
        std::abs(after - before) > 4096 * epsilon * scale)
    {
        return result;
    }
    amrex::MultiFab::Copy(temperature, current, 0, 0, 1, 0);
    amrex::MultiFab::Copy(energy, trial_energy, 0, 0, 1, 0);
    result.valid = true;
    return result;
}
} // namespace warpx::hybrid
