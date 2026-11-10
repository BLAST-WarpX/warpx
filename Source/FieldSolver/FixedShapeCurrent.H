/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FIXED_SHAPE_CURRENT_H_
#define WARPX_FIXED_SHAPE_CURRENT_H_
#ifdef WARPX_DIM_3D
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <array>
#include <memory>

class WarpX;

// Single-rank research reference. Only the particle current occupies the
// finest-spacing lattice. The electromagnetic solve remains on the two leaves.
class FixedShapeCurrent {
  public:
    explicit FixedShapeCurrent (WarpX& warpx);
    void Assemble (amrex::Real dt);

  private:
    WarpX& w;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> raw, filtered;
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> fine_owns;
    amrex::iMultiFab valid_centers;
};
#endif
#endif
