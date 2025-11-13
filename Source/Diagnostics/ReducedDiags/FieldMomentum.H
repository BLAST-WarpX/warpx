/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDMOMENTUM_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDMOMENTUM_H_

#include "ReducedDiags.H"

#include <string>

/**
 * \brief This class mainly contains a function that computes the field momentum.
 */
class FieldMomentum : public ReducedDiags
{
public:

    /**
     * \brief Constructor
     *
     * \param[in] rd_name reduced diags names
     */
    FieldMomentum(const std::string& rd_name);

    /**
     * \brief This function computes the electromagnetic momentum,
     * obtained by integrating the electromagnetic momentum density g = eps0 * (E x B).
     *
     * \param[in] step current time step
     */
    void ComputeDiags(int step) final;
};

#endif
