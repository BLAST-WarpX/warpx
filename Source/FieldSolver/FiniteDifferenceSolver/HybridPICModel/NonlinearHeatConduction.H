/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "ElectronThermodynamics.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace warpx::hybrid
{
enum class NonlinearHeatFailure { None, Input, LocalSolve, Iterations, Energy };
struct NonlinearHeatSolveResult
{
    int iterations = 0;
    NonlinearHeatFailure failure = NonlinearHeatFailure::Input;
    amrex::Real residual = 0;
    amrex::Real energy_residual = 0;
    bool valid = false;
};

/** Monotone nonlinear backward-Euler conductive graph at fixed density.
 * U(T_i)-U_old_i + out_i*T_i - sum(in_ij*T_j) = 0.
 * Coefficients already include dt and the recipient's inverse node volume.
 * Each nonlinear Jacobi update solves the monotone local EOS equation using
 * safeguarded Newton/bisection within the old global temperature range. No
 * frozen-Cv energy approximation, energy clipping or table extrapolation.
 * Optional material_mass_density is kg/m3 for one table-EOS material.
 * Both outputs remain unchanged unless local and global energy gates pass.
 */
NonlinearHeatSolveResult
tryNonlinearHeatConduction (amrex::MultiFab& temperature, amrex::MultiFab& energy,
                            amrex::MultiFab const& old_energy, amrex::MultiFab const& charge,
                            amrex::MultiFab const& outgoing, amrex::MultiFab const& incoming,
                            amrex::MultiFab const& volume, amrex::Geometry const& geometry,
                            ElectronThermodynamicsExecutor const& eos,
                            amrex::MultiFab const* material_mass_density = nullptr);
} // namespace warpx::hybrid
