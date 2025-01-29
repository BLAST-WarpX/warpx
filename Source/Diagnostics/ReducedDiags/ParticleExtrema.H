/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEEXTREMA_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEEXTREMA_H_

#include "ReducedDiags.H"

#include <map>
#include <string>

/**
 *  This class mainly contains a function that
 *  computes the particle extrema of each species.
 */
class ParticleExtrema : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ParticleExtrema(const std::string& rd_name);

    /// name of species
    std::string m_species_name;

    /**
     * This function computes the particle extrema
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step) final;

private:
    /// auxiliary structure to store headers and indices of the reduced diagnostics
    struct aux_header_index
    {
        std::string header;
        int idx;
    };

    /// map to store header texts and indices of the reduced diagnostics
    std::map<std::string, aux_header_index> m_headers_indices;
};

#endif
