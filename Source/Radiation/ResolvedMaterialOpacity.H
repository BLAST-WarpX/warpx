/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RESOLVEDMATERIALOPACITY_H_
#define WARPX_RADIATION_RESOLVEDMATERIALOPACITY_H_

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"

#include <AMReX_Array.H>
#include <AMReX_BaseFwd.H>

namespace warpx::radiation::detail
{
using ResolvedMaterialChargeFields =
    amrex::GpuArray<amrex::MultiFab const*, ElectronThermodynamicsExecutor::max_materials>;

/** Snapshot the native nodal material gather in cell ne [m^-3] and Te [K].
 * Outputs are scratch; invalid nodal input returns false collectively. The
 * density threshold/floor and heat-capacity weighting match the existing
 * hybrid radiation adapter. A tabulated EOS additionally requires every frozen
 * deposited unit-ion charge field; no effective composition is manufactured.
 * No field supplied as an input is modified. */
bool SnapshotResolvedMaterialOpacity (amrex::MultiFab const& density,
                                      amrex::MultiFab const& temperature,
                                      amrex::MultiFab& electron_density,
                                      amrex::MultiFab& electron_temperature,
                                      amrex::Geometry const& geometry,
                                      ElectronThermodynamicsExecutor eos,
                                      amrex::Real minimum_density, amrex::Real thermodynamic_floor,
                                      ResolvedMaterialChargeFields const& material_fields = {});

/** Separate raw single-selected TABLE gather with exact vacant corners.
 * Empty raw/unit-ion charge is not an EOS sample, so finite nonnegative T is
 * preserved outside the table temperature domain. Active corners retain the
 * original EOS-domain checks; zero capacity yields canonical cell ne=Te=0. */
[[nodiscard]] bool SnapshotResolvedMaterialOpacityWithExactVacuum (
    amrex::MultiFab const& density, amrex::MultiFab const& temperature,
    amrex::MultiFab& electron_density, amrex::MultiFab& electron_temperature,
    amrex::Geometry const& geometry, ElectronThermodynamicsExecutor eos,
    amrex::Real minimum_density, amrex::Real thermodynamic_floor,
    ResolvedMaterialChargeFields const& material_fields);
} // namespace warpx::radiation::detail

#endif
