/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ImplicitMaterialTableCoefficients.H"

#include "HybridCellMaterialState.H"
#include "MaterialOpacityTableResponse.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

namespace warpx::radiation::detail {
using namespace amrex::literals;

void
FillCoupledMaterialTableCoefficients (
    amrex::MultiFab const& density, amrex::MultiFab const& trial,
    amrex::MultiFab& opacity, amrex::MultiFab& absorption,
    amrex::MultiFab& emission, amrex::Geometry const& geometry,
    ElectronThermodynamicsExecutor eos, amrex::Real minimum_density,
    amrex::Real thermodynamic_floor, MaterialOpacityTableExecutor table,
    amrex::Real mass_per_electron, EnergyGroupsExecutor groups,
    amrex::GpuArray<amrex::MultiFab const*,
                    ElectronThermodynamicsExecutor::max_materials> const&
        material_fields) {
    // Negative coefficients are rejected by the existing implicit solver before
    // a radiation/material candidate can be committed. Do not turn errors into
    // transparent cells, including a missing composition snapshot.
    opacity.setVal(-1);
    absorption.setVal(-1);
    emission.setVal(-1);
    if (!amrex::Math::isfinite(mass_per_electron) || mass_per_electron <= 0 ||
        groups.m_num_groups != table.m_num_groups || eos.m_num_materials < 0 ||
        eos.m_num_materials > ElectronThermodynamicsExecutor::max_materials) {
        return;
    }
    for (int material = 0; material < eos.m_num_materials; ++material) {
        if (material_fields[material] == nullptr) {
            return;
        }
    }
    auto const lower = geometry.ProbLoArray();
    auto const dx = geometry.CellSizeArray();
    auto const domain_lo = amrex::lbound(geometry.Domain());
    for (amrex::MFIter mfi(opacity); mfi.isValid(); ++mfi) {
        auto const rho = density.const_array(mfi);
        auto const te = trial.const_array(mfi);
        auto const kr = opacity.array(mfi);
        auto const ap = absorption.array(mfi);
        auto const emit = emission.array(mfi);
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays materials{};
        for (int material = 0; material < eos.m_num_materials; ++material) {
            materials[material] = material_fields[material]->const_array(mfi);
        }
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto const state = GatherHybridCellMaterialState(
                    i, j, k, minimum_density, thermodynamic_floor, eos, rho, te,
                    materials, lower, dx, domain_lo);
                if (!state.valid || state.electron_density <= minimum_density) {
                    return;
                }
                for (int group = 0; group < groups.m_num_groups; ++group) {
                    auto const response = EvaluateMaterialOpacityTableResponse(
                        table, state.electron_density * mass_per_electron,
                        state.electron_temperature, group,
                        groups.planckFraction(
                            group, PhysConst::kb * state.electron_temperature));
                    if (response.valid) {
                        kr(i, j, k, group) = response.transport_opacity;
                        ap(i, j, k, group) = response.absorption_rate;
                        emit(i, j, k, group) = response.emission_rate_density;
                    }
                }
            });
    }
}
} // namespace warpx::radiation::detail
