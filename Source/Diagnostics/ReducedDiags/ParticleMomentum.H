/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEMOMENTUM_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEMOMENTUM_H_

#include "ReducedDiags.H"

#include <string>

/**
 * \brief This class mainly contains a function that computes
 * the particle relativistic momentum of each species.
 */
class ParticleMomentum : public ReducedDiags
{
public:

    /**
     * \brief Constructor
     *
     * \param[in] rd_name reduced diags names
     */
    ParticleMomentum(const std::string& rd_name);

    /**
     * \brief This function computes the particle relativistic momentum,
     * obtained by summing over all particles the product p * w,
     * where p is the relativistic momentum and w is the individual particle weight.
     *
     * \param [in] step current time step
     */
    void ComputeDiags(int step) final;
};

#endif
