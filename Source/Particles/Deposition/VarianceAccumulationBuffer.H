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
    protected:
    // Species name for array lookup from register
    std::string m_species_name;

    amrex::Vector<std::array<std::unique_ptr<amrex::iMultiFab>, 3> > m_nsamples;

    public:
    // Constructor for accumulation buffer
    // This is built after the T vector fields are created, so use it to allocate the
    VarianceAccumulationBuffer (ablastr::fields::MultiLevelVectorField const& T_vf, std::string const& species_name );

    // This public member functions
    void reset ();
    void ConvertVarianceToTemperatureAndFilter (ablastr::fields::MultiLevelVectorField const& var_vf,
                                                amrex::Real normalization_factor,
                                                bool apply_filter);

    // Data MF Accessor
    amrex::MultiFab* get(std::string arr, ablastr::fields::Direction dir, int lev);
    amrex::iMultiFab* get_n(ablastr::fields::Direction dir, int lev);
};

} // namespace warpx::particles::deposition
#endif // WARPX_VARIANCEACCUMULATIONBUFFER_H_
