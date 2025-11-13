/* Copyright 2020-2021 Michael Rowan, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_LOADBALANCEEFFICIENCY_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_LOADBALANCEEFFICIENCY_H_

#include "ReducedDiags.H"

#include <string>

/**
 *  This class mainly contains a function that gets the
 *  current load balance efficiency for writing to output.
 */
class LoadBalanceEfficiency : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    LoadBalanceEfficiency(const std::string& rd_name);

    /**
     * This function gets the current load balance efficiency
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step) final;
};

#endif
