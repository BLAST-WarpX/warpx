/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVETHERMALCORNEREMISSION_H_
#define WARPX_RADIATION_NATIVETHERMALCORNEREMISSION_H_

#include <AMReX_MultiFab.H>

namespace warpx::radiation
{
/** Immutable per-cell corner incidence and canonical trial nodal T. Weights
 * are frozen from the actual OLD corner capacity and sum to one in occupied
 * cells, zero in vacuum. This is a separately selected caloric/emission model,
 * never a rescaling of the legacy trial-Cv cell-temperature source. */
struct NativeThermalCornerEmission
{
    amrex::MultiFab const* weights = nullptr;
    amrex::MultiFab const* nodal_temperature = nullptr;
    int rays_per_corner = 8;
    /** Synchronous reference-plus-increment coordinates for a separately
     * admitted ideal all-D trial. Never retained by replay or checkpoint. */
    amrex::MultiFab const* nodal_reference = nullptr;
    amrex::MultiFab const* nodal_increment = nullptr;
};

/** Difference of fourth powers without subtracting two large baths. */
inline amrex::Real
ThermalFourthPowerDifference (amrex::Real temperature, amrex::Real reference)
{
    return (temperature - reference) * (temperature + reference) *
           (temperature * temperature + reference * reference);
}
} // namespace warpx::radiation
#endif
