/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_IMPLICIT_MATERIAL_TABLE_COEFFICIENTS_H_
#define WARPX_IMPLICIT_MATERIAL_TABLE_COEFFICIENTS_H_

#include "EnergyGroups.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"
#include "MaterialOpacityTable.H"

#include <AMReX_Array.H>
#include <AMReX_BaseFwd.H>

namespace warpx::radiation::detail {
/** Frozen native nodal material state to cell/group coefficients for implicit
 * LTE/FLD. mass_per_electron is m_ion*q_e/q_ion [kg] for one fixed-charge
 * carrier. Invalid coefficients reject the nonlinear attempt; this function
 * never commits electron, photon or particle state. The caller owns
 * single-material admission.
 */
void FillCoupledMaterialTableCoefficients (
    amrex::MultiFab const& density, amrex::MultiFab const& trial,
    amrex::MultiFab& opacity, amrex::MultiFab& absorption,
    amrex::MultiFab& emission, amrex::Geometry const& geometry,
    ElectronThermodynamicsExecutor eos, amrex::Real minimum_density,
    amrex::Real thermodynamic_floor, MaterialOpacityTableExecutor table,
    amrex::Real mass_per_electron, EnergyGroupsExecutor groups,
    amrex::GpuArray<amrex::MultiFab const*,
                    ElectronThermodynamicsExecutor::max_materials> const&
        material_fields);
} // namespace warpx::radiation::detail
#endif
