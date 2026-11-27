/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_HYBRIDCALORICRESIDUAL_H_
#define WARPX_RADIATION_HYBRIDCALORICRESIDUAL_H_

#include "CaloricConvergence.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"

#include <AMReX_BaseFwd.H>

namespace warpx::radiation::detail
{
using CaloricMaterialFields =
    amrex::GpuArray<amrex::MultiFab const*, ElectronThermodynamicsExecutor::max_materials>;

/** Original raw global caloric mismatch and heating scale [J/m^3].
 * Native nodal layouts and frozen density/composition are caller contracts. */
amrex::GpuArray<amrex::Real, 2> CoupledCaloricResidual (
    amrex::MultiFab const& density, amrex::MultiFab const& old, amrex::MultiFab const& candidate,
    amrex::MultiFab const& check, ElectronThermodynamicsExecutor eos, amrex::Real minimum_density,
    amrex::Real thermodynamic_floor, CaloricMaterialFields const& material_fields);

/** Local EOS-energy acceptance using independently perturbed source images.
 * The check must itself remain inside that bracket. Representation allowance
 * uses local energy/temperature ULPs and local heating, never a domain maximum.
 * Supports builtin ideal/latent and composition-aware table caloric inverses.
 * The table allowance certifies its actual finite bisection partition and
 * endpoint clipping at each node's deposited material densities; signed
 * caloric energy references require no change of energy zero.
 * This adds no source correction and never changes any supplied state. */
LocalCaloricConvergence CoupledLocalCaloricResidual (
    amrex::MultiFab const& density, amrex::MultiFab const& old, amrex::MultiFab const& candidate,
    amrex::MultiFab const& check, amrex::MultiFab const& lower_image,
    amrex::MultiFab const& upper_image, ElectronThermodynamicsExecutor eos,
    amrex::Real minimum_density, amrex::Real thermodynamic_floor,
    CaloricMaterialFields const& material_fields, amrex::Real material_tolerance);
} // namespace warpx::radiation::detail

#endif
