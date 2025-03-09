/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_VARIANCEACCUMULATIONBUFFER_H_
#define WARPX_VARIANCEACCUMULATIONBUFFER_H_

#include <ablastr/fields/MultiFabRegister.H>

namespace warpx::particles::deposition {

class VarianceAccumulationBuffer {
    public:

    // Accumulation vector fields, aliased to fields in register
    ablastr::fields::MultiLevelVectorField w;
    ablastr::fields::MultiLevelVectorField w2;
    ablastr::fields::MultiLevelVectorField vbar;

    // Constructor for accumulation buffer
    // This is built after the T vector fields are created, so use it to allocate the
    VarianceAccumulationBuffer (ablastr::fields::MultiLevelVectorField const& T_vf, std::string const& species_name );

    // This public member functions
    void setAllValues (amrex::Real val);
    void SynchronizeBoundaryAndNormalizeVariance (ablastr::fields::MultiLevelVectorField const& var_vf);
};

} // namespace warpx::particles::deposition
#endif // WARPX_VARIANCEACCUMULATIONBUFFER_H_
