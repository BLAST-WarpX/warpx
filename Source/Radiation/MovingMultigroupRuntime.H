/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOVINGMULTIGROUPRUNTIME_H_
#define WARPX_RADIATION_MOVINGMULTIGROUPRUNTIME_H_

#include <AMReX_REAL.H>

#include <array>
#include <string>
#include <vector>

namespace warpx::radiation {
/** Native runtime identity, separate from solver scratch and material data.
 * Stored moments use fixed LAB energy bins; opacity and the positive angular
 * Planck-bin dipole reconstruction use proper bins and exact Lorentz overlap.
 * This is the explicitly selected isotropic-rest-pressure/AP moment backend,
 * not limited M1, and it does not change the established gray FLD backend.
 */
struct MovingMultigroupRuntime {
    std::array<amrex::Real, 3> initial_lab_flux_ratio{};
    std::vector<std::array<amrex::Real, 2>> configured_material_identity;
    std::string model_identity;
    // Explicit RZ P1 optical mirror, not a kinetic half-space wall model.
    bool bounded_rz_p1_mirror = false;
    amrex::Real relaxation = 0.5;
    amrex::Real linear_tolerance = 1.e-13;
    int max_linear_iterations = 500;
    int verbosity = 0;
};

inline std::string
MovingGroupMomentumFieldName (int group, int direction) {
    return std::string("radiation_moving_group_q") + "xyz"[direction] + "_g" +
           std::to_string(group);
}
} // namespace warpx::radiation

#endif
