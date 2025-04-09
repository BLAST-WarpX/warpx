/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_BEAMRELEVANT_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_BEAMRELEVANT_H_

#include "ReducedDiags.H"

#include <string>

/**
 *  This class contains diagnostics that are relevant to beam.
 */
class BeamRelevant : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    BeamRelevant(const std::string& rd_name);

    /// name of beam species
    std::string m_beam_name;

    /**
     * This function computes beam relevant quantities.
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step) final;

};

#endif
