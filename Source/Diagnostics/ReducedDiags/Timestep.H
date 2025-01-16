/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Thomas Marks
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_TIMESTEP_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_TIMESTEP_H_

#include "ReducedDiags.H"
#include <string>

/**
 * This class contains a function for retrieving the current simulation timestep as a diagnostic.
 * Useful mainly for simulations using adaptive timestepping.
 */
class Timestep: public ReducedDiags {
public:
    /**
     * constructor
     * @param[in] rd_name reduced diags name
     */
    Timestep (const std::string& rd_name);

    /**
     * This function gets the current physical timestep of the simulation at all refinement levels.
     * @param[in] step current time step
     */
    void ComputeDiags (int step) final;
};

#endif //WARPX_DIAGNOSTICS_REDUCEDDIAGS_TIMESTEP_H_
