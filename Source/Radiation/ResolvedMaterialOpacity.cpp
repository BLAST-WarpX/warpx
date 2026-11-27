/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ResolvedMaterialOpacity.H"

#include "HybridCellMaterialState.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>

namespace warpx::radiation::detail
{
namespace
{
bool
SnapshotResolvedMaterialOpacityImpl (amrex::MultiFab const& density,
                                     amrex::MultiFab const& temperature,
                                 amrex::MultiFab& electron_density,
                                 amrex::MultiFab& electron_temperature,
                                 amrex::Geometry const& geometry,
                                 ElectronThermodynamicsExecutor eos, amrex::Real minimum_density,
                                 amrex::Real thermodynamic_floor,
                                 ResolvedMaterialChargeFields const& material_fields,
                                 bool allow_exact_vacuum)
{
    int const material_count = eos.m_num_materials;
    int minimum_material_count = material_count;
    int maximum_material_count = material_count;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_material_count);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_material_count);
    if (minimum_material_count != maximum_material_count || material_count < 0 ||
        material_count > eos.max_materials)
    {
        return false;
    }
    bool const builtin =
        (eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) && std::isfinite(eos.m_gamma) &&
        eos.m_gamma > 1 && eos.m_num_latent_transitions >= 0 &&
        eos.m_num_latent_transitions <= eos.max_latent_transitions && material_count == 0;
    bool tabulated = false;
#ifdef WARPX_USE_SINGULARITY_EOS
    tabulated = eos.isSingularitySpiner() && material_count > 0 &&
                eos.tableStorage() != nullptr;
#endif
    bool valid =
        (builtin || tabulated) &&
        (!allow_exact_vacuum ||
         (tabulated && material_count == 1 && eos.m_use_resolved_material_selection &&
          thermodynamic_floor == 0)) &&
        std::isfinite(minimum_density) && minimum_density >= 0 &&
        std::isfinite(thermodynamic_floor) && thermodynamic_floor >= 0 &&
        electron_density.nComp() == 1 && electron_temperature.nComp() == 1 &&
        electron_density.ixType().cellCentered() && density.ixType().nodeCentered() &&
        density.nComp() >= 1 && temperature.nComp() == 1 &&
        density.boxArray() == temperature.boxArray() &&
        amrex::convert(electron_density.boxArray(), amrex::IntVect(1)) == density.boxArray() &&
        electron_density.boxArray() == electron_temperature.boxArray() &&
        electron_density.DistributionMap() == density.DistributionMap() &&
        density.DistributionMap() == temperature.DistributionMap() &&
        electron_density.DistributionMap() == electron_temperature.DistributionMap();
    for (int material = 0; material < material_count; ++material)
    {
        auto const* field = material_fields[material];
        valid = valid && field && field->nComp() == 1 &&
                field->ixType() == density.ixType() &&
                field->boxArray() == density.boxArray() &&
                field->DistributionMap() == density.DistributionMap() &&
                field->nGrowVect().allGE(1);
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) {
        return false;
    }
    auto const low = geometry.ProbLoArray(), spacing = geometry.CellSizeArray();
    auto const origin = amrex::lbound(geometry.Domain());
    amrex::MultiFab candidate_density(electron_density.boxArray(),
                                      electron_density.DistributionMap(), 1,
                                      electron_density.nGrowVect());
    amrex::MultiFab candidate_temperature(electron_temperature.boxArray(),
                                          electron_temperature.DistributionMap(), 1,
                                          electron_temperature.nGrowVect());
    candidate_density.setVal(0);
    candidate_temperature.setVal(0);
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(candidate_density); it.isValid(); ++it) {
        auto const rho = density.const_array(it), temperature_in = temperature.const_array(it);
        auto const ne = candidate_density.array(it), te = candidate_temperature.array(it);
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays material_arrays{};
        for (int material = 0; material < material_count; ++material)
        {
            material_arrays[material] = material_fields[material]->const_array(it);
        }
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                     for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner) {
                         int node[3] = {i, j, k};
                         for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                             node[d] += (corner >> d) & 1;
                         }
                         auto const charge = rho(node[0], node[1], node[2]);
                         auto const value = temperature_in(node[0], node[1], node[2]);
                         bool const vacant = allow_exact_vacuum && charge == 0;
                         if (!amrex::Math::isfinite(charge) || charge < 0 ||
                             !amrex::Math::isfinite(value) ||
                             (vacant ? (value < 0 ||
                                        material_arrays[0](node[0], node[1], node[2]) != 0)
                                     : (value < eos.minimumTemperature() ||
                                        value > eos.maximumTemperature()))) {
                             return {1};
                         }
                     }
                     auto const state = GatherHybridCellMaterialState(
                         i, j, k, minimum_density, thermodynamic_floor, eos, rho, temperature_in,
                         material_arrays, low, spacing, origin);
                     if (!state.valid || !amrex::Math::isfinite(state.electron_density) ||
                         state.electron_density < 0 ||
                         !amrex::Math::isfinite(state.electron_temperature) ||
                         state.electron_temperature < 0) {
                         return {1};
                     }
                     ne(i, j, k) = state.electron_density;
                     te(i, j, k) = state.electron_temperature;
                     return {0};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid) {
        return false;
    }
    // Publish the complete initialized sample only after valid-cell acceptance.
    // Interbox/periodic ghosts are filled below; exterior ghosts remain zero.
    amrex::MultiFab::Copy(electron_density, candidate_density, 0, 0, 1,
                          electron_density.nGrowVect());
    amrex::MultiFab::Copy(electron_temperature, candidate_temperature, 0, 0, 1,
                          electron_temperature.nGrowVect());
    electron_density.FillBoundary(geometry.periodicity());
    electron_temperature.FillBoundary(geometry.periodicity());
    return true;
}
} // namespace

bool SnapshotResolvedMaterialOpacity (
    amrex::MultiFab const& density, amrex::MultiFab const& temperature,
    amrex::MultiFab& electron_density, amrex::MultiFab& electron_temperature,
    amrex::Geometry const& geometry, ElectronThermodynamicsExecutor eos,
    amrex::Real minimum_density, amrex::Real thermodynamic_floor,
    ResolvedMaterialChargeFields const& material_fields)
{
    return SnapshotResolvedMaterialOpacityImpl(density, temperature, electron_density,
        electron_temperature, geometry, eos, minimum_density, thermodynamic_floor,
        material_fields, false);
}

bool SnapshotResolvedMaterialOpacityWithExactVacuum (
    amrex::MultiFab const& density, amrex::MultiFab const& temperature,
    amrex::MultiFab& electron_density, amrex::MultiFab& electron_temperature,
    amrex::Geometry const& geometry, ElectronThermodynamicsExecutor eos,
    amrex::Real minimum_density, amrex::Real thermodynamic_floor,
    ResolvedMaterialChargeFields const& material_fields)
{
    // This entry has a separate identity; callers establish collective mode
    // selection before entering either gather, not by a per-cell shortcut.
    return SnapshotResolvedMaterialOpacityImpl(density, temperature, electron_density,
        electron_temperature, geometry, eos, minimum_density, thermodynamic_floor,
        material_fields, true);
}
} // namespace warpx::radiation::detail
