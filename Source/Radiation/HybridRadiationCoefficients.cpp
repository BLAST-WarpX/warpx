/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HybridRadiationCoefficients.H"

#include "HybridCellMaterialState.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array4.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

namespace warpx::radiation::detail
{
using namespace amrex::literals;

void
FillCoupledHybridCoefficients (
    amrex::MultiFab const& density, amrex::MultiFab const& trial, amrex::MultiFab& opacity,
    amrex::MultiFab& absorption, amrex::MultiFab& emission, amrex::Geometry const& geometry,
    ElectronThermodynamicsExecutor const eos, amrex::Real const minimum_density,
    amrex::Real const thermodynamic_floor, OpacityEvaluator<1> const planck,
    OpacityEvaluator<1> const rosseland, warpx::radiation::EnergyGroupsExecutor const groups,
    amrex::Real const evaluation_time,
    amrex::GpuArray<amrex::MultiFab const*, ElectronThermodynamicsExecutor::max_materials> const&
        material_fields)
{
    auto const lower = geometry.ProbLoArray();
    auto const dx = geometry.CellSizeArray();
    auto const domain_lo = amrex::lbound(geometry.Domain());
    int const num_groups = groups.m_num_groups;
    opacity.setVal(0);
    for (amrex::MFIter mfi(opacity); mfi.isValid(); ++mfi) {
        auto const rho = density.const_array(mfi);
        auto const te = trial.const_array(mfi);
        auto const ar = opacity.array(mfi);
        auto const ap = absorption.array(mfi);
        auto const emit = emission.array(mfi);
        ElectronThermodynamicsExecutor::MaterialChargeDensityArrays materials{};
        for (int material = 0; material < eos.m_num_materials; ++material)
        {
            materials[material] = material_fields[material]->const_array(mfi);
        }
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                auto const state =
                    GatherHybridCellMaterialState(i, j, k, minimum_density, thermodynamic_floor,
                                                  eos, rho, te, materials, lower, dx, domain_lo);
                amrex::Real const t2 = state.electron_temperature * state.electron_temperature;
                amrex::Real const blackbody = 7.565733250280007e-16_rt * t2 * t2;
                amrex::GpuArray<int, AMREX_SPACEDIM> const index{AMREX_D_DECL(i, j, k)};
                amrex::GpuArray<int, AMREX_SPACEDIM> const lo{
                    AMREX_D_DECL(domain_lo.x, domain_lo.y, domain_lo.z)};
                amrex::GpuArray<amrex::Real, 3> position{0, 0, 0};
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
#if defined(WARPX_DIM_1D_Z)
                    int const component = 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    int const component = d == 1 ? 2 : 0;
#else
                    int const component = d;
#endif
                    position[component] = lower[d] + (index[d] - lo[d] + 0.5_rt) * dx[d];
                }
                for (int g = 0; g < num_groups; ++g)
                {
                    auto const photon_energy = groups.representativeEnergy(g);
                    auto const p =
                        planck({}, position[0], position[1], position[2], evaluation_time,
                               photon_energy, state.electron_density, state.electron_temperature);
                    ar(i, j, k, g) =
                        state.valid && state.electron_density > minimum_density
                            ? rosseland({}, position[0], position[1], position[2], evaluation_time,
                                        photon_energy, state.electron_density,
                                        state.electron_temperature)
                            : -1.0_rt;
                    ap(i, j, k, g) = PhysConst::c * p;
                    emit(i, j, k, g) =
                        ap(i, j, k, g) * blackbody *
                        groups.planckFraction(g, PhysConst::kb * state.electron_temperature);
                }
            });
    }
}

} // namespace warpx::radiation::detail
