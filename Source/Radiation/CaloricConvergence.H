/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_CALORICCONVERGENCE_H_
#define WARPX_RADIATION_CALORICCONVERGENCE_H_

#include <AMReX_REAL.H>

namespace warpx::radiation
{
/** Material-owned, node-local convergence contract, independent of a
 * particular radiation backend. A raw caloric residual is reported separately.
 */
struct LocalCaloricConvergence
{
    bool valid = false;
    bool converged = false;
    amrex::Real maximum_relative_residual = 0;
    /** Caloric-energy units chosen by the material adapter, not kelvin. */
    amrex::Real maximum_roundoff_bound = 0;
};
} // namespace warpx::radiation

#endif
