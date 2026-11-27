/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_HYBRIDRADIATIONCOEFFICIENTS_H_
#define WARPX_RADIATION_HYBRIDRADIATIONCOEFFICIENTS_H_

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"
#include "OpacityEvaluator.H"

#include <AMReX_Array.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_REAL.H>

namespace warpx::radiation::detail
{
/** Evaluate shared native material coefficients at frozen charge density.
 * Density/temperature inputs retain the native nodal layout; outputs are
 * cell-centered group coefficients. Charge density is in C/m^3, temperature
 * in K, opacity in 1/m, absorption in 1/s, and emission in W/m^3. Backend
 * adapters retain model-specific conversions and trial-state commit ownership.
 * The free-function launch also avoids NVCC private-member lambda restrictions.
 */
void
FillCoupledHybridCoefficients (
    amrex::MultiFab const& density, amrex::MultiFab const& trial, amrex::MultiFab& opacity,
    amrex::MultiFab& absorption, amrex::MultiFab& emission, amrex::Geometry const& geometry,
    ElectronThermodynamicsExecutor const eos, amrex::Real const minimum_density,
    amrex::Real const thermodynamic_floor, OpacityEvaluator<1> const planck,
    OpacityEvaluator<1> const rosseland, warpx::radiation::EnergyGroupsExecutor const groups,
    amrex::Real const evaluation_time,
    amrex::GpuArray<amrex::MultiFab const*, ElectronThermodynamicsExecutor::max_materials> const&
        material_fields);
} // namespace warpx::radiation::detail

#endif
