/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FIXED_SHAPE_GATHER_H_
#define WARPX_FIXED_SHAPE_GATHER_H_

#include "WarpX.H"

#include <array>

struct FixedShapeGather
{
    std::array<amrex::FArrayBox const*, 6> fields{};
    amrex::Box box;
    [[nodiscard]] explicit operator bool () const { return fields[0] != nullptr; }
};

// Only replace fields belonging to the normal kinetic push. A Python caller
// may supply independent field arrays to PushP.
inline FixedShapeGather
get_fixed_shape_gather (WarpX& warpx, int level,
                        std::array<amrex::MultiFab const*, 6> const* requested = nullptr)
{
    FixedShapeGather result;
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    if (!warpx.m_fields.has_vector("composite_fixed_E", 1)) { return result; }

    using warpx::fields::FieldType;
    for (int family = 0; family < 2; ++family) {
        auto const aux = family == 0 ? FieldType::Efield_aux : FieldType::Bfield_aux;
        auto const* name = family == 0 ? "composite_fixed_E" : "composite_fixed_B";
        for (int d = 0; d < 3; ++d) {
            auto const dir = ablastr::fields::Direction{d};
            int const component = 3 * family + d;
            if (requested && (*requested)[component] != warpx.m_fields.get(aux, dir, level)) {
                return {};
            }
            result.fields[component] = &(*warpx.m_fields.get(name, dir, 1))[0];
        }
    }
    // Shape kernels work in a local coordinate system. Include the low guards
    // for particles at a physical or periodic boundary. RZ retains the axis
    // as its radial origin.
    result.box = amrex::enclosedCells(result.fields[0]->box());
#ifdef WARPX_DIM_RZ
    result.box.setSmall(0, warpx.Geom(1).Domain().smallEnd(0));
#endif
#else
    amrex::ignore_unused(warpx, level, requested);
#endif
    return result;
}
#endif
