/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDENERGY_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDENERGY_H_

#include "ReducedDiags.H"

#include <AMReX_MultiFab.H>

#include <string>

/**
 *  This class mainly contains a function that
 *  computes the field energy.
 */
class FieldEnergy : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    FieldEnergy(const std::string& rd_name);

    /**
     * This function computes the field energy (EF):
     * EF = sum( 1/2 * (|E|^2 * eps0 + |B|^2 / mu0) * dV ),
     * where E is the electric field, B is the magnetic field,
     * eps0 is the vacuum permittivity, mu0 is the vacuum permeability,
     * dV is the cell volume (area, in 2D) and the sum is over all cells.
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step) final;

    /**
     * \brief Calculate the integral of the field squared, taking into
     *        account the fraction of the cell volume within the domain.
     *
     * \param field The MultiFab to be integrated
     * \param lev   The refinement level
     * \return The integral
     */
    amrex::Real ComputeNorm2(const amrex::MultiFab& field, int lev);

};

#endif
