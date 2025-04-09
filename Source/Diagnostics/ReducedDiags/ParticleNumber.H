/* Copyright 2019-2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLENUMBER_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLENUMBER_H_

#include "ReducedDiags.H"

#include <string>

/**
 *  This class mainly contains a function that computes the total number of macroparticles and of
 *  physical particles (i.e. the sum of the weights) of each species.
 */
class ParticleNumber : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ParticleNumber(const std::string& rd_name);

    /**
     * This function computes the total number of macroparticles and physical particles of each
     * species.
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step) final;

};

#endif // WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLENUMBER_H_
