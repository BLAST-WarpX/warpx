/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "ElectronThermodynamics.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>

#include <utility>
#include <vector>

namespace warpx::hybrid
{
struct HeatConductionResult
{
    int substeps = 0;
    int iterations = 0;
    int rejected_steps = 0;
};

/** Fixed-charge ion moments on matching physical nodal charge deposits.
 * Zbar = sum(Z_s n_s)/sum(n_s), Zeff = sum(Z_s^2 n_s)/sum(Z_s n_s).
 * Each pair supplies a species charge-density field [C/m3] and Z_s > 0.
 * Empty support has zero moments. No density floor supplies missing ions.
 */
void computeElectronChargeMoments (
    amrex::MultiFab& mean_charge, amrex::MultiFab& effective_charge,
    amrex::MultiFab const& charge,
    std::vector<std::pair<amrex::MultiFab const*, amrex::Real>> const& species);

/** Isotropic, insulating/periodic electron heat transport.
 * Conductivity receives physical charge density [C/m3] and Te [eV], and
 * returns W/(m K). Temperature is stored in K; realized_energy is the
 * accepted local redistribution [J/m3]. Live outputs are committed only
 * after the full interval succeeds. No vacuum density floor is inserted.
 * Optional charge moments select conductivity(rho,Te,Zbar,Zeff) instead of
 * the two-argument parser; both moment fields must be provided together.
 * Nonideal EOSs use a nonlinear caloric graph solve in Cartesian geometry;
 * a single table material requires explicit mass density [kg/m3]. Conductivity
 * and its limiter remain lagged per substep. The ideal RZ path is unchanged.
 */
HeatConductionResult advanceElectronHeatConduction (
    amrex::MultiFab& temperature, amrex::MultiFab& realized_energy, amrex::MultiFab const& charge,
    amrex::Geometry const& geometry, amrex::Real gamma, amrex::ParserExecutor<2> conductivity,
    amrex::Real flux_limiter, amrex::Real dt, int maximum_substeps, amrex::Real axis_volume_factor,
    amrex::MultiFab const* mean_charge = nullptr,
    amrex::MultiFab const* effective_charge = nullptr,
    amrex::ParserExecutor<4> composition_conductivity = {},
    ElectronThermodynamicsExecutor const* thermodynamics = nullptr,
    amrex::MultiFab const* material_mass_density = nullptr);
} // namespace warpx::hybrid
