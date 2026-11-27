/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RESOLVED_DIFFUSION_FRONT_H_
#define WARPX_RADIATION_RESOLVED_DIFFUSION_FRONT_H_

#include "AsymptoticDiffusionField.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

namespace warpx::radiation
{
struct ResolvedDiffusionFront
{
    amrex::iMultiFab mask;
    amrex::MultiFab opacity;
    amrex::Long kinetic_cells = 0;
};

/** Private scalar partition for validated, matching native field layouts.
 * Exactly zero E/q eligible cells remain kinetic. Positive or invalid states
 * are never erased; downstream validation still owns their acceptance.
 * Inputs, including all caller ghosts, are read-only. */
ResolvedDiffusionFront BuildResolvedDiffusionFront (
    amrex::MultiFab const& energy, amrex::MultiFab const& momentum,
    amrex::MultiFab const& opacity, amrex::iMultiFab const& eligible,
    amrex::Geometry const& geometry);

/** Read-only conversion screen, not an accepted future material evolution.
 * Predict one frozen-velocity, zero-work scalar step from a proposed census
 * state. Require the existing constitutive E/q realizability checks. If it
 * fails, the caller retains the physical photons instead of admitting them;
 * no future impulse/work or predictor fields may be committed.
 * This extra screen does not guarantee arbitrary moving/steep states remain
 * admissible after a later physical material update. */
bool AdmissibleResolvedDiffusionCensus (
    amrex::MultiFab const& energy, amrex::MultiFab const& momentum,
    amrex::MultiFab const& opacity, amrex::iMultiFab const& eligible,
    amrex::MultiFab const& velocity, amrex::Geometry const& geometry, amrex::Real dt,
    AsymptoticDiffusionFieldOptions const& options, bool moving);
}
#endif
