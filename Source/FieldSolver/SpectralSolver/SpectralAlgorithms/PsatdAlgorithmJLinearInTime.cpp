/* Copyright 2019
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PsatdAlgorithmJLinearInTime.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpX_Complex.H"

#include <AMReX_Array4.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_Math.H>
#include <AMReX_MFIter.H>
#include <AMReX_PODVector.H>

#include <cmath>

#if WARPX_USE_FFT

using namespace amrex::literals;

PsatdAlgorithmJLinearInTime::PsatdAlgorithmJLinearInTime(
    const SpectralKSpace& spectral_kspace,
    const amrex::DistributionMapping& dm,
    const SpectralFieldIndex& spectral_index,
    const int norder_x,
    const int norder_y,
    const int norder_z,
    ablastr::utils::enums::GridType grid_type,
    const amrex::Real dt,
    const bool time_averaging,
    const bool dive_cleaning,
    const bool divb_cleaning)
    // Initializer list
    : SpectralBaseAlgorithm(spectral_kspace, dm, spectral_index, norder_x, norder_y, norder_z, grid_type),
    m_dt(dt),
    m_time_averaging(time_averaging),
    m_dive_cleaning(dive_cleaning),
    m_divb_cleaning(divb_cleaning)
{
    const amrex::BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Always allocate these coefficients
    C_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    S_ck_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X1_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X2_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X3_coef = SpectralRealCoefficients(ba, dm, 1, 0);

    InitializeSpectralCoefficients(spectral_kspace, dm, dt);

    // Allocate these coefficients only with time averaging
    if (time_averaging)
    {
        X5_coef = SpectralRealCoefficients(ba, dm, 1, 0);
        X6_coef = SpectralRealCoefficients(ba, dm, 1, 0);
        InitializeSpectralCoefficientsAveraging(spectral_kspace, dm, dt);
    }
}

void
PsatdAlgorithmJLinearInTime::pushSpectralFields (SpectralFieldData& f) const
{
    const bool time_averaging = m_time_averaging;
    const bool dive_cleaning = m_dive_cleaning;
    const bool divb_cleaning = m_divb_cleaning;

    const amrex::Real dt = m_dt;

    const SpectralFieldIndex& Idx = m_spectral_index;

    // Loop over boxes
    for (amrex::MFIter mfi(f.fields); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = f.fields[mfi].box();

        // Extract arrays for the fields to be updated
        const amrex::Array4<Complex> fields = f.fields[mfi].array();

        // These coefficients are always allocated
        const amrex::Array4<const amrex::Real> C_arr = C_coef[mfi].array();
        const amrex::Array4<const amrex::Real> S_ck_arr = S_ck_coef[mfi].array();
        const amrex::Array4<const amrex::Real> X1_arr = X1_coef[mfi].array();
        const amrex::Array4<const amrex::Real> X2_arr = X2_coef[mfi].array();
        const amrex::Array4<const amrex::Real> X3_arr = X3_coef[mfi].array();

        amrex::Array4<const amrex::Real> X5_arr;
        amrex::Array4<const amrex::Real> X6_arr;
        if (time_averaging)
        {
            X5_arr = X5_coef[mfi].array();
            X6_arr = X6_coef[mfi].array();
        }

        // Extract pointers for the k vectors
        const amrex::Real* modified_kx_arr = modified_kx_vec[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* modified_ky_arr = modified_ky_vec[mfi].dataPtr();
#endif
        const amrex::Real* modified_kz_arr = modified_kz_vec[mfi].dataPtr();

        // Loop over indices within one box
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Record old values of the fields to be updated
            const Complex Ex_old = fields(i,j,k,Idx.Ex);
            const Complex Ey_old = fields(i,j,k,Idx.Ey);
            const Complex Ez_old = fields(i,j,k,Idx.Ez);
            const Complex Bx_old = fields(i,j,k,Idx.Bx);
            const Complex By_old = fields(i,j,k,Idx.By);
            const Complex Bz_old = fields(i,j,k,Idx.Bz);

            // Shortcuts for the values of J and rho
            const Complex Jx_old = fields(i,j,k,Idx.Jx_old);
            const Complex Jy_old = fields(i,j,k,Idx.Jy_old);
            const Complex Jz_old = fields(i,j,k,Idx.Jz_old);
            const Complex Jx_new = fields(i,j,k,Idx.Jx_new);
            const Complex Jy_new = fields(i,j,k,Idx.Jy_new);
            const Complex Jz_new = fields(i,j,k,Idx.Jz_new);
            const Complex rho_old = fields(i,j,k,Idx.rho_old);
            const Complex rho_new = fields(i,j,k,Idx.rho_new);

            Complex F_old, G_old;
            if (dive_cleaning) { F_old = fields(i,j,k,Idx.F); }
            if (divb_cleaning) { G_old = fields(i,j,k,Idx.G); }

            // k vector values
            const amrex::Real kx = modified_kx_arr[i];
#if defined(WARPX_DIM_3D)
            const amrex::Real ky = modified_ky_arr[j];
            const amrex::Real kz = modified_kz_arr[k];
#else
            constexpr amrex::Real ky = 0._rt;
            const     amrex::Real kz = modified_kz_arr[j];
#endif
            // Physical constants and imaginary unit
            constexpr amrex::Real c2 = PhysConst::c * PhysConst::c;
            constexpr amrex::Real ep0 = PhysConst::ep0;
            constexpr amrex::Real inv_ep0 = 1._rt / PhysConst::ep0;
            constexpr Complex I = Complex{0._rt, 1._rt};

            // These coefficients are initialized in the function InitializeSpectralCoefficients
            const amrex::Real C = C_arr(i,j,k);
            const amrex::Real S_ck = S_ck_arr(i,j,k);
            const amrex::Real X1 = X1_arr(i,j,k);
            const amrex::Real X2 = X2_arr(i,j,k);
            const amrex::Real X3 = X3_arr(i,j,k);
            const amrex::Real X4 = - S_ck / PhysConst::ep0;

            // Update equations for E in the formulation with rho

            fields(i,j,k,Idx.Ex) = C * Ex_old
                + I * c2 * S_ck * (ky * Bz_old - kz * By_old)
                + X4 * Jx_old - I * (X2 * rho_new - X3 * rho_old) * kx - X1 * (Jx_new - Jx_old) / dt;

            fields(i,j,k,Idx.Ey) = C * Ey_old
                + I * c2 * S_ck * (kz * Bx_old - kx * Bz_old)
                + X4 * Jy_old - I * (X2 * rho_new - X3 * rho_old) * ky - X1 * (Jy_new - Jy_old) / dt;

            fields(i,j,k,Idx.Ez) = C * Ez_old
                + I * c2 * S_ck * (kx * By_old - ky * Bx_old)
                + X4 * Jz_old - I * (X2 * rho_new - X3 * rho_old) * kz - X1 * (Jz_new - Jz_old) / dt;

            // Update equations for B

            fields(i,j,k,Idx.Bx) = C * Bx_old
                - I * S_ck * (ky * Ez_old - kz * Ey_old) + I * X1 * (ky * Jz_old - kz * Jy_old)
                + I * X2/c2 * (ky * (Jz_new - Jz_old) - kz * (Jy_new - Jy_old));

            fields(i,j,k,Idx.By) = C * By_old
                - I * S_ck * (kz * Ex_old - kx * Ez_old) + I * X1 * (kz * Jx_old - kx * Jz_old)
                + I * X2/c2 * (kz * (Jx_new - Jx_old) - kx * (Jz_new - Jz_old));

            fields(i,j,k,Idx.Bz) = C * Bz_old
                - I * S_ck * (kx * Ey_old - ky * Ex_old) + I * X1 * (kx * Jy_old - ky * Jx_old)
                + I * X2/c2 * (kx * (Jy_new - Jy_old) - ky * (Jx_new - Jx_old));

            if (dive_cleaning)
            {
                const Complex k_dot_E = kx * Ex_old + ky * Ey_old + kz * Ez_old;
                const Complex k_dot_J  = kx * Jx_old + ky * Jy_old + kz * Jz_old;
                const Complex k_dot_dJ = kx * (Jx_new - Jx_old) + ky * (Jy_new - Jy_old) + kz * (Jz_new - Jz_old);

                fields(i,j,k,Idx.Ex) += I * c2 * S_ck * F_old * kx;
                fields(i,j,k,Idx.Ey) += I * c2 * S_ck * F_old * ky;
                fields(i,j,k,Idx.Ez) += I * c2 * S_ck * F_old * kz;

                fields(i,j,k,Idx.F) = C * F_old + S_ck * (I * k_dot_E - rho_old * inv_ep0)
                    - X1 * ((rho_new - rho_old) / dt + I * k_dot_J) - I * X2/c2 * k_dot_dJ;
            }

            if (divb_cleaning)
            {
                const Complex k_dot_B = kx * Bx_old + ky * By_old + kz * Bz_old;

                fields(i,j,k,Idx.Bx) += I * S_ck * G_old * kx;
                fields(i,j,k,Idx.By) += I * S_ck * G_old * ky;
                fields(i,j,k,Idx.Bz) += I * S_ck * G_old * kz;

                fields(i,j,k,Idx.G) = C * G_old + I * c2 * S_ck * k_dot_B;
            }

            if (time_averaging)
            {
                const amrex::Real X5 = X5_arr(i,j,k);
                const amrex::Real X6 = X6_arr(i,j,k);

                // TODO: Here the code is *accumulating* the average,
                // because it is meant to be used with sub-cycling
                // maybe this should be made more generic

                fields(i,j,k,Idx.Ex_avg) += S_ck * Ex_old
                    + I * c2 * ep0 * X1 * (ky * Bz_old - kz * By_old)
                    + I * X5 * rho_old * kx + I * X6 * rho_new * kx + X3/c2 * Jx_old - X2/c2 * Jx_new;

                fields(i,j,k,Idx.Ey_avg) += S_ck * Ey_old
                    + I * c2 * ep0 * X1 * (kz * Bx_old - kx * Bz_old)
                    + I * X5 * rho_old * ky + I * X6 * rho_new * ky + X3/c2 * Jy_old - X2/c2 * Jy_new;

                fields(i,j,k,Idx.Ez_avg) += S_ck * Ez_old
                    + I * c2 * ep0 * X1 * (kx * By_old - ky * Bx_old)
                    + I * X5 * rho_old * kz + I * X6 * rho_new * kz + X3/c2 * Jz_old - X2/c2 * Jz_new;

                fields(i,j,k,Idx.Bx_avg) += S_ck * Bx_old
                    - I * ep0 * X1 * (ky * Ez_old - kz * Ey_old)
                    - I * X5/c2 * (ky * Jz_old - kz * Jy_old) - I * X6/c2 * (ky * Jz_new - kz * Jy_new);

                fields(i,j,k,Idx.By_avg) += S_ck * By_old
                    - I * ep0 * X1 * (kz * Ex_old - kx * Ez_old)
                    - I * X5/c2 * (kz * Jx_old - kx * Jz_old) - I * X6/c2 * (kz * Jx_new - kx * Jz_new);

                fields(i,j,k,Idx.Bz_avg) += S_ck * Bz_old
                    - I * ep0 * X1 * (kx * Ey_old - ky * Ex_old)
                    - I * X5/c2 * (kx * Jy_old - ky * Jx_old) - I * X6/c2 * (kx * Jy_new - ky * Jx_new);

                if (dive_cleaning)
                {
                    fields(i,j,k,Idx.Ex_avg) += I * c2 * ep0 * X1 * F_old * kx;
                    fields(i,j,k,Idx.Ey_avg) += I * c2 * ep0 * X1 * F_old * ky;
                    fields(i,j,k,Idx.Ez_avg) += I * c2 * ep0 * X1 * F_old * kz;
                }

                if (divb_cleaning)
                {
                    fields(i,j,k,Idx.Bx_avg) += I * ep0 * X1 * G_old * kx;
                    fields(i,j,k,Idx.By_avg) += I * ep0 * X1 * G_old * ky;
                    fields(i,j,k,Idx.Bz_avg) += I * ep0 * X1 * G_old * kz;
                }
            }
        });
    }
}

void PsatdAlgorithmJLinearInTime::InitializeSpectralCoefficients (
    const SpectralKSpace& spectral_kspace,
    const amrex::DistributionMapping& dm,
    const amrex::Real dt)
{
    const amrex::BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Loop over boxes and allocate the corresponding coefficients for each box
    for (amrex::MFIter mfi(ba, dm); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = ba[mfi];

        // Extract pointers for the k vectors
        const amrex::Real* kx_s = modified_kx_vec[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* ky_s = modified_ky_vec[mfi].dataPtr();
#endif
        const amrex::Real* kz_s = modified_kz_vec[mfi].dataPtr();

        // Coefficients always allocated
        const amrex::Array4<amrex::Real> C = C_coef[mfi].array();
        const amrex::Array4<amrex::Real> S_ck = S_ck_coef[mfi].array();
        const amrex::Array4<amrex::Real> X1 = X1_coef[mfi].array();
        const amrex::Array4<amrex::Real> X2 = X2_coef[mfi].array();
        const amrex::Array4<amrex::Real> X3 = X3_coef[mfi].array();

        // Loop over indices within one box
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Calculate norm of k vector
            const amrex::Real knorm_s = std::sqrt(
                amrex::Math::powi<2>(kx_s[i]) +
#if defined(WARPX_DIM_3D)
                amrex::Math::powi<2>(ky_s[j]) + amrex::Math::powi<2>(kz_s[k]));
#else
                amrex::Math::powi<2>(kz_s[j]));
#endif
            // Physical constants and imaginary unit
            constexpr amrex::Real c = PhysConst::c;
            constexpr amrex::Real ep0 = PhysConst::ep0;

            const amrex::Real c2 = amrex::Math::powi<2>(c);
            const amrex::Real dt2 = amrex::Math::powi<2>(dt);

            const amrex::Real om_s = c * knorm_s;
            const amrex::Real om2_s = amrex::Math::powi<2>(om_s);

            // C
            C(i,j,k) = std::cos(om_s * dt);

            // S_ck
            if (om_s != 0.)
            {
                S_ck(i,j,k) = std::sin(om_s * dt) / om_s;
            }
            else // om_s = 0
            {
                S_ck(i,j,k) = dt;
            }

            // X1 (multiplies i*([k] \times J) in the update equation for update B)
            if (om_s != 0.)
            {
                X1(i,j,k) = (1._rt - C(i,j,k)) / (ep0 * om2_s);
            }
            else // om_s = 0
            {
                X1(i,j,k) = 0.5_rt * dt2 / ep0;
            }

            // X2 (multiplies rho_new in the update equation for E)
            if (om_s != 0.)
            {
                X2(i,j,k) = c2 * (dt - S_ck(i,j,k)) / (ep0 * dt * om2_s);
            }
            else // om_s = 0
            {
                X2(i,j,k) = c2 * dt2 / (6._rt * ep0);
            }

            // X3 (multiplies rho_old in the update equation for E)
            if (om_s != 0.)
            {
                X3(i,j,k) = c2 * (dt * C(i,j,k) - S_ck(i,j,k)) / (ep0 * dt * om2_s);
            }
            else // om_s = 0
            {
                X3(i,j,k) = - c2 * dt2 / (3._rt * ep0);
            }
        });
    }
}

void PsatdAlgorithmJLinearInTime::InitializeSpectralCoefficientsAveraging (
    const SpectralKSpace& spectral_kspace,
    const amrex::DistributionMapping& dm,
    const amrex::Real dt)
{
    const amrex::BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Loop over boxes and allocate the corresponding coefficients for each box
    for (amrex::MFIter mfi(ba, dm); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = ba[mfi];

        // Extract pointers for the k vectors
        const amrex::Real* kx_s = modified_kx_vec[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* ky_s = modified_ky_vec[mfi].dataPtr();
#endif
        const amrex::Real* kz_s = modified_kz_vec[mfi].dataPtr();

        const amrex::Array4<amrex::Real const> C = C_coef[mfi].array();
        const amrex::Array4<amrex::Real const> S_ck = S_ck_coef[mfi].array();

        const amrex::Array4<amrex::Real> X5 = X5_coef[mfi].array();
        const amrex::Array4<amrex::Real> X6 = X6_coef[mfi].array();

        // Loop over indices within one box
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Calculate norm of k vector
            const amrex::Real knorm_s = std::sqrt(
                amrex::Math::powi<2>(kx_s[i]) +
#if defined(WARPX_DIM_3D)
                amrex::Math::powi<2>(ky_s[j]) + amrex::Math::powi<2>(kz_s[k]));
#else
                amrex::Math::powi<2>(kz_s[j]));
#endif
            // Physical constants and imaginary unit
            constexpr amrex::Real c = PhysConst::c;
            constexpr amrex::Real c2 = c*c;
            constexpr amrex::Real ep0 = PhysConst::ep0;

            // Auxiliary coefficients
            const amrex::Real dt3 = dt * dt * dt;

            const amrex::Real om_s  = c * knorm_s;
            const amrex::Real om2_s = om_s * om_s;
            const amrex::Real om4_s = om2_s * om2_s;

            if (om_s != 0.)
            {
                X5(i,j,k) = c2 / ep0 * (S_ck(i,j,k) / om2_s - (1._rt - C(i,j,k)) / (om4_s * dt)
                                        - 0.5_rt * dt / om2_s);
            }
            else
            {
                X5(i,j,k) = - c2 * dt3 / (8._rt * ep0);
            }

            if (om_s != 0.)
            {
                X6(i,j,k) = c2 / ep0 * ((1._rt - C(i,j,k)) / (om4_s * dt) - 0.5_rt * dt / om2_s);
            }
            else
            {
                X6(i,j,k) = - c2 * dt3 / (24._rt * ep0);
            }
        });
    }
}

void PsatdAlgorithmJLinearInTime::CurrentCorrection (SpectralFieldData& field_data)
{
    // Profiling
    BL_PROFILE("PsatdAlgorithmJLinearInTime::CurrentCorrection");

    amrex::ignore_unused(field_data);
    WARPX_ABORT_WITH_MESSAGE(
        "Current correction not implemented for multi-J PSATD algorithm");
}

void
PsatdAlgorithmJLinearInTime::VayDeposition (SpectralFieldData& field_data)
{
    // Profiling
    BL_PROFILE("PsatdAlgorithmJLinearInTime::VayDeposition()");

    amrex::ignore_unused(field_data);
    WARPX_ABORT_WITH_MESSAGE(
        "Vay deposition not implemented for multi-J PSATD algorithm");
}

#endif // WARPX_USE_FFT
