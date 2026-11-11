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

#ifdef WARPX_DIM_3D

// Single-rank research reference. Only the particle current occupies the
// finest-spacing lattice. The electromagnetic solve remains on the two leaves.
class FixedShapeCurrent {
  public:
    static constexpr char const* raw_name = "composite_yee_fixed_J";
    explicit FixedShapeCurrent (WarpX& warpx);
    void Assemble (WarpX& warpx, amrex::Real dt);
    void UpdateGather (WarpX& warpx);

  private:
    std::array<amrex::MultiFab*, 3> raw{};
    std::array<std::unique_ptr<amrex::MultiFab>, 3> filtered;
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> fine_owns;
    amrex::iMultiFab valid_centers;
    std::array<std::array<std::unique_ptr<amrex::MultiFab>, 3>, 2>
        coarse_gather;
    std::array<std::array<amrex::MultiFab*, 3>, 2> force{};
};
#endif
#endif
