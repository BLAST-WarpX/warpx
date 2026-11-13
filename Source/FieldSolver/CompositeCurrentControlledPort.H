/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_COMPOSITECURRENTCONTROLLEDPORT_H_
#define WARPX_COMPOSITECURRENTCONTROLLEDPORT_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <array>
#include <vector>

/** Project a physical Ampere contour using the finest available grid at each
 * point.
 *
 * The correction is a discrete curl on every fine patch. Coarse patches receive
 * their parent patch's correction, so reconstruction of the auxiliary fields
 * retains that curl. PML split fields receive the same continuation.
 */
class CompositeCurrentControlledPort
{
  public:
    CompositeCurrentControlledPort (std::array<amrex::Real, 3> lower,
                                    std::array<amrex::Real, 3> upper, int direction,
                                    amrex::Real terminal_0, amrex::Real terminal_1);

    void Apply (amrex::Real target_axis_current);
    [[nodiscard]] std::array<amrex::Real, 2> Currents (amrex::Real target_axis_current);

    // Public enclosing functions and sample types are required by NVCC's
    // extended device lambdas. This class is an internal implementation detail.
    struct Sample {
        amrex::GpuArray<int, 3> index{};
        amrex::GpuArray<amrex::Real, 3> fraction{};
        amrex::Real weight = 0;
        int row = 0;
    };

    struct Level {
        int first = 0;
        int last = 0;
        int offset = 0;
        std::array<std::vector<amrex::Gpu::DeviceVector<Sample>>, 3> samples;
    };

    struct Measurement {
        std::vector<amrex::Real> circulation;
        std::vector<amrex::Real> absolute_circulation;
    };

    [[nodiscard]] Measurement Measure ();
    void AddCorrection (int level, PatchType patch_type,
                        std::vector<amrex::Real> const& coefficients);

  private:
    void BuildSamples ();
    [[nodiscard]] std::vector<amrex::Real> Solve (Measurement const& measured,
                                                  amrex::Real desired_circulation) const;

    std::array<amrex::Real, 3> m_lower;
    std::array<amrex::Real, 3> m_upper;
    int m_direction;
    int m_mesh_direction;
    int m_unknowns = 0;
    std::array<int, 2> m_terminal_rows{};
    std::vector<amrex::Real> m_planes;
    std::vector<Level> m_levels;
    std::vector<int> m_columns;
    std::vector<double> m_response;
};

#endif
