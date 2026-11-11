/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FIXED_SHAPE_CURRENT_H_
#define WARPX_FIXED_SHAPE_CURRENT_H_
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <array>
#include <memory>

class WarpX;

struct FixedShapeGather {
    std::array<amrex::FArrayBox const*, 6> fields{};
    [[nodiscard]] explicit
    operator bool () const {
        return fields[0] != nullptr;
    }
};
// Returns the reference force field only when the given fields belong to the
// normal kinetic push. Python callers may supply independent field arrays.
FixedShapeGather get_fixed_shape_gather (
    WarpX&, int level,
    std::array<amrex::MultiFab const*, 6> const* requested = nullptr);

#ifdef WARPX_DIM_3D

// Single-rank research reference. Only the particle current occupies the
// finest-spacing lattice. The electromagnetic solve remains on the two leaves.
class FixedShapeCurrent {
  public:
    explicit FixedShapeCurrent (WarpX& warpx);
    void Assemble (WarpX& warpx, amrex::Real dt);
    void UpdateGather (WarpX& warpx);

  private:
    std::array<std::unique_ptr<amrex::MultiFab>, 3> raw, filtered;
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> fine_owns;
    amrex::iMultiFab valid_centers;
    std::array<std::array<std::unique_ptr<amrex::MultiFab>, 3>, 2>
        coarse_gather;
    std::array<std::array<amrex::MultiFab*, 3>, 2> force{};
};
#endif
#endif
