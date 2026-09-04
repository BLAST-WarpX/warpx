/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CURRENTCONTROLLEDPORT_H_
#define WARPX_CURRENTCONTROLLEDPORT_H_

#include "CurrentControlledPort_fwd.H"

#include "Fields.H"
#include "Utils/CurrentWaveform.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_BaseFwd.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <memory>

/** Enforce one current waveform between two internal terminal surfaces.
 *
 * The two terminals are congruent, axis-aligned cross-sections.  On every
 * cross-section between them, this class projects the magnetic circulation to
 *
 *     integral B.dl = mu0 I(t).
 *
 * Thus the constrained quantity is the total Ampere current.  In a Maxwell
 * solve it includes particle, macroscopic, and displacement current.  In the
 * magnetoquasistatic Hybrid-PIC solve it is the ion plus electron-fluid
 * current.  The contour can enclose plasma, a macroscopic material, or an EB;
 * the contour itself must remain in field-carrying cells.  Supported
 * geometries are 3D, 2D XZ (in-plane current per unit invariant length), and
 * axisymmetric RZ.
 */
class CurrentControlledPort {
  public:
    CurrentControlledPort ();
    ~CurrentControlledPort () = default;

    CurrentControlledPort (CurrentControlledPort const&) = delete;
    CurrentControlledPort& operator=(CurrentControlledPort const&) = delete;
    CurrentControlledPort (CurrentControlledPort&&) = delete;
    CurrentControlledPort& operator=(CurrentControlledPort&&) = delete;

    [[nodiscard]] static bool is_enabled ();

    void InitData (
        ablastr::fields::VectorField const& Bfield,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
        amrex::Real time);

    void ApplyBfield (
        ablastr::fields::VectorField const& Bfield,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
        int lev, PatchType patch_type, amrex::Real time);

    /** {target, terminal-0 current, terminal-1 current}; A, or A/m in 2D XZ. */
    [[nodiscard]] std::array<amrex::Real, 3> status () const;

  private:
    struct Terminal {
        std::array<amrex::Real, 3> lo{};
        std::array<amrex::Real, 3> hi{};
    };

    int m_direction = 2;
    Terminal m_terminal_0;
    Terminal m_terminal_1;
    amrex::Real m_axis_sign{1};
    warpx::utils::CurrentWaveform m_waveform;
    bool m_initialized = false;
    amrex::Real m_last_target{};
    std::array<amrex::Real, 2> m_last_terminal_current{};
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> m_owner_mask;

    void ValidateGeometry (ablastr::fields::VectorField const& Bfield) const;

#ifdef WARPX_DIM_3D
    void Apply3D (
        ablastr::fields::VectorField const& Bfield,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
        amrex::Real target_axis_current);
#endif
#ifdef WARPX_DIM_XZ
    void ApplyXZ (
        ablastr::fields::VectorField const& Bfield,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
        amrex::Real target_axis_current);
#endif
#ifdef WARPX_DIM_RZ
    void ApplyRZ (
        ablastr::fields::VectorField const& Bfield,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
        amrex::Real target_axis_current);
#endif

#ifdef WARPX_DIM_3D
    // A solver-consistent curl of an axial top-hat vector potential.  Scaling
    // this basis on each axial plane changes the Ampere circulation without
    // changing the discrete divergence of B.
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_curl_basis;
#endif

  public:
    // The implementation helpers below contain extended __device__ lambdas.
    // NVCC requires their enclosing member functions and any nested types used
    // by the generated host stubs to be public.
#ifdef WARPX_DIM_3D
    struct Edge {
        int component = 0;
        int tangent_direction = 0;
        int fixed_direction = 1;
        amrex::Real fixed_coordinate{};
        amrex::Real tangent_lo{};
        amrex::Real tangent_hi{};
        amrex::Real orientation{1};
    };

    [[nodiscard]] std::array<Edge, 4> Edges () const;
    void BuildCurlBasis (
        ablastr::fields::VectorField const& Bfield,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B);
    [[nodiscard]] amrex::Real
    MeasureEdgeLocal (amrex::MultiFab const& field,
                      amrex::iMultiFab const& owner_mask,
                      amrex::iMultiFab const* eb_flag, Edge const& edge,
                      int plane_index) const;
    [[nodiscard]] amrex::Real MeasureContourLocal (
        ablastr::fields::VectorField const& Bfield,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
        int plane_index) const;
    void AddScaledCurlBasis (ablastr::fields::VectorField const& Bfield,
                             int plane_index, amrex::Real scale) const;
#endif

#ifdef WARPX_DIM_XZ
    [[nodiscard]] std::array<amrex::Real, 3> MeasureXZContourLocal (
        amrex::MultiFab const& by, amrex::iMultiFab const& owner_mask,
        amrex::iMultiFab const* eb_flag, int plane_index) const;
    void CorrectXZContour (amrex::MultiFab& by, amrex::iMultiFab const* eb_flag,
                           int plane_index, amrex::Real current_error,
                           std::array<amrex::Real, 3> const& contour) const;
#endif

#ifdef WARPX_DIM_RZ
    [[nodiscard]] std::array<amrex::Real, 3> MeasureRZContourLocal (
        amrex::MultiFab const& btheta, amrex::iMultiFab const& owner_mask,
        amrex::iMultiFab const* eb_flag, int plane_index) const;
    void CorrectRZContour (amrex::MultiFab& btheta,
                           amrex::iMultiFab const* eb_flag, int plane_index,
                           amrex::Real current_error,
                           std::array<amrex::Real, 3> const& contour) const;
#endif
};

#endif // WARPX_CURRENTCONTROLLEDPORT_H_
