/* Copyright 2023 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_CHARGEONEB_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_CHARGEONEB_H_

#include "ReducedDiags.H"

#include <AMReX_Parser.H>

#include <string>

/**
 *  This class mainly contains a function that
 *  computes the total charge at the surface
 *  of the embedded boundary, by using the formula
 *  $Q_{tot} = \epsilon_0 \iint dS \cdot E$
 *  where the integral is performed over the
 *  surface of the embedded boundary.
 *
 *  If a weighting function is provided, this computes
 *  $Q_{tot} = \epsilon_0 \iint dS \cdot E \times weighting(x, y, z)
 */
class ChargeOnEB : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ChargeOnEB (const std::string& rd_name);

    /**
     * This function computes the charge at the surface of the EB:
     * $Q_{tot} = \epsilon_0 \iint dS \cdot E$
     *  where the integral is performed over the EB surface
     *
     * @param[in] step current time step
     */
    void ComputeDiags (int step) final;

private:
    /// Optional parser to add weight inside the integral
    std::unique_ptr<amrex::Parser> m_parser_weighting;
    /// Whether the weighting is activated
    bool m_do_parser_weighting = false;

};

#endif
