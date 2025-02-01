/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDPOYTINGFLUX_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDPOYTINGFLUX_H_

#include "ReducedDiags.H"

#include <string>

/**
 * \brief This class mainly contains a function that computes the field Poynting flux,
 *        S = E cross B, integrated over each face of the domain.
 */
class FieldPoyntingFlux : public ReducedDiags
{
public:

    /**
     * \brief Constructor
     *
     * \param[in] rd_name reduced diags names
     */
    FieldPoyntingFlux (const std::string& rd_name);

    /**
     * \brief Call the routine to compute the Poynting flux if needed
     *
     * \param[in] step current time step
     */
    void ComputeDiags (int step) final;

    /**
     * \brief Call the routine to compute the Poynting flux at the mid step time level
     *
     * \param[in] step current time step
     */
    void ComputeDiagsMidStep (int step) final;

    /**
     * \brief This function computes the electromagnetic Poynting flux,
     * obtained by integrating the electromagnetic Poynting flux density g = eps0 * (E x B)
     * on the surface of the domain.
     *
     * \param[in] step current time step
     */
    void ComputePoyntingFlux ();

    void WriteCheckpointData (std::string const & dir) final;

    void ReadCheckpointData (std::string const & dir) final;

private:

    bool use_mid_step_value = false;

};

#endif
