/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_FLUX_RELAXED_FIELDS_H_
#define WARPX_RADIATION_FLUX_RELAXED_FIELDS_H_

#include <AMReX_Config.H>

#include <string>

namespace warpx::radiation::detail
{
/** Cell-integrated flux-relaxed radiation momentum, c * p [J], per component. */
inline std::string
FluxRelaxedFieldName (int component)
{
    return std::string("radiation_flux_relaxed_q") + "xyz"[component];
}

/** Exact persistent-state marker written beside the field-registry data. */
inline std::string const&
FluxRelaxedStateMarker ()
{
    static std::string const marker =
        "flux_relaxed_state_v1 cartesian_gray_level0_particle_carry dims=" +
        std::to_string(AMREX_SPACEDIM);
    return marker;
}

inline std::string const&
LimitedM1PacketStateMarker ()
{
    static std::string const marker =
        "limited_m1_packet_state_v1 cartesian_gray_level0_particle_carry dims=" +
        std::to_string(AMREX_SPACEDIM);
    return marker;
}
} // namespace warpx::radiation::detail

#endif // WARPX_RADIATION_FLUX_RELAXED_FIELDS_H_
