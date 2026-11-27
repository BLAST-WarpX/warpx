/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_CYLINDRICAL_NODAL_ALGORITHM_H_
#define WARPX_CYLINDRICAL_NODAL_ALGORITHM_H_

#include "CylindricalYeeAlgorithm.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

/**
 * \brief Pointwise centered mode-0 derivatives for collocated nodal fields in
 * RZ.
 *
 * Upward/DownwardDr and Upward/DownwardDz provide centered second-order
 * derivatives. Upward/DownwardDrr_over_r computes (1/r) d/dr (r F) for regular
 * odd-parity fields. Off axis this is the centered difference of r*F divided
 * by r. The axis limit is 2*F(i+1)/dr, using F(0)=0 and F(-dr)=-F(dr).
 *
 * This helper supplies pointwise derivatives, not the matching energy-weighted
 * transpose needed for magnetic work. Callers must not infer a discrete
 * Poynting identity near axis/PEC boundaries from the pointwise Ampere current.
 */
struct CylindricalNodalAlgorithm : CylindricalYeeAlgorithm {
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    UpwardDr (amrex::Array4<amrex::Real const> const& F,
              amrex::Real const* coefs_r, int n_coefs_r, int i, int j, int k,
              int comp) {
        amrex::ignore_unused(n_coefs_r);
        return 0.5 * coefs_r[0] * (F(i + 1, j, k, comp) - F(i - 1, j, k, comp));
    }

    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    DownwardDr (amrex::Array4<amrex::Real const> const& F,
                amrex::Real const* coefs_r, int n_coefs_r, int i, int j, int k,
                int comp) {
        return UpwardDr(F, coefs_r, n_coefs_r, i, j, k, comp);
    }

    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    UpwardDz (amrex::Array4<amrex::Real const> const& F,
              amrex::Real const* coefs_z, int n_coefs_z, int i, int j, int k,
              int comp) {
        amrex::ignore_unused(n_coefs_z);
        return 0.5 * coefs_z[0] * (F(i, j + 1, k, comp) - F(i, j - 1, k, comp));
    }

    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    DownwardDz (amrex::Array4<amrex::Real const> const& F,
                amrex::Real const* coefs_z, int n_coefs_z, int i, int j, int k,
                int comp) {
        return UpwardDz(F, coefs_z, n_coefs_z, i, j, k, comp);
    }

    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    UpwardDrr_over_r (amrex::Array4<amrex::Real const> const& F, amrex::Real r,
                      amrex::Real dr, amrex::Real const* coefs_r, int n_coefs_r,
                      int i, int j, int k, int comp) {
        amrex::ignore_unused(n_coefs_r);
        if (r == 0.0) {
            return 2.0 * coefs_r[0] * F(i + 1, j, k, comp);
        }
        return 0.5 * coefs_r[0] *
               ((r + dr) * F(i + 1, j, k, comp) -
                (r - dr) * F(i - 1, j, k, comp)) /
               r;
    }

    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    DownwardDrr_over_r (amrex::Array4<amrex::Real const> const& F,
                        amrex::Real r, amrex::Real dr,
                        amrex::Real const* coefs_r, int n_coefs_r, int i, int j,
                        int k, int comp) {
        return UpwardDrr_over_r(F, r, dr, coefs_r, n_coefs_r, i, j, k, comp);
    }
};

#endif // WARPX_CYLINDRICAL_NODAL_ALGORITHM_H_
