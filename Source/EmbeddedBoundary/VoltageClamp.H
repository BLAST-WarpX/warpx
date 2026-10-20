/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EB_VOLTAGE_CLAMP_H_
#define WARPX_EB_VOLTAGE_CLAMP_H_

#include <AMReX_REAL.H>

#include <string>
#include <utility>
#include <vector>

/* Voltage clamp for embedded-boundary conductors in electromagnetic runs
 * (RZ m=0, explicit staggered Yee, one level). Defined in RZ builds with embedded
 * boundaries only.
 *
 * The conductors are the components of WarpX's frozen-edge staircase. The fixed
 * nodes are the endpoints of frozen E edges. Conductor k has a unit potential
 * psi_k (1 on its fixed nodes, 0 on the others and on the PEC walls, harmonic
 * elsewhere) and a unit
 * field E_k = -grad psi_k. The Python driver (pywarpx.staircase_bias_corrector)
 * measures the conductor voltages and adds sum_k dV_k E_k to E.
 * See Docs/source/usage/workflows/electrode_voltage_clamp.rst.
 *
 * The functions register their output fields as voltage_clamp_{phi,weight,E}_<k>,
 * or with suffix "grounded", on level 0. All functions are collective over MPI.
 */
namespace warpx::embedded_boundary::voltage_clamp
{
    /** Name of the unit field E_k (k >= 0) or of the grounded field (k < 0).
     *
     * \param[in] k conductor index, or negative for the grounded field
     */
    std::string EFieldName (int k);

    /** Solve for the unit potential and field of conductor k.
     *
     * \param[in] k        conductor index, starting at 0
     * \param[in] selector parser expression in (x, y, z), 1 on the fixed nodes of
     *                     conductor k and 0 on all other fixed nodes
     * \param[in] rtol     relative tolerance of the MLMG solve
     * \param[in] max_iter maximum number of MLMG iterations
     * \return final MLMG residual
     */
    amrex::Real SolveUnitPotential (int k, std::string const& selector,
                                    amrex::Real rtol, int max_iter);

    /** Abort unless every fixed node belongs to exactly one of the n conductors, or
     *  to a conductor that frozen edges join to a PEC wall (part of the ground, not
     *  listed). Call once after the unit solves.
     *
     * \param[in] n number of conductors
     */
    void CheckSelection (int n);
}

#endif // WARPX_EB_VOLTAGE_CLAMP_H_
