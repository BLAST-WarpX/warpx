/* Copyright 2019
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PsatdAlgorithmJConstantInTime.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
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

using namespace amrex;

PsatdAlgorithmJConstantInTime::PsatdAlgorithmJConstantInTime(
    const SpectralKSpace& spectral_kspace,
    const DistributionMapping& dm,
    const SpectralFieldIndex& spectral_index,
    const int norder_x,
    const int norder_y,
    const int norder_z,
    ablastr::utils::enums::GridType grid_type,
    const amrex::Vector<amrex::Real>& v_galilean,
    const amrex::Real dt,
    const bool update_with_rho,
    const bool time_averaging,
    const bool dive_cleaning,
    const bool divb_cleaning)
    // Initializer list
    : SpectralBaseAlgorithm(spectral_kspace, dm, spectral_index, norder_x, norder_y, norder_z, grid_type),
    // Initialize the centered finite-order modified k vectors:
    // these are computed always with the assumption of centered grids
    // (argument grid_type=GridType::Collocated), for both collocated and staggered grids
    modified_kx_vec_centered(spectral_kspace.getModifiedKComponent(dm, 0, norder_x, GridType::Collocated)),
#if defined(WARPX_DIM_3D)
    modified_ky_vec_centered(spectral_kspace.getModifiedKComponent(dm, 1, norder_y, GridType::Collocated)),
    modified_kz_vec_centered(spectral_kspace.getModifiedKComponent(dm, 2, norder_z, GridType::Collocated)),
#else
    modified_kz_vec_centered(spectral_kspace.getModifiedKComponent(dm, 1, norder_z, GridType::Collocated)),
#endif
    m_v_galilean(v_galilean),
    m_dt(dt),
    m_update_with_rho(update_with_rho),
    m_time_averaging(time_averaging),
    m_dive_cleaning(dive_cleaning),
    m_divb_cleaning(divb_cleaning),
    m_is_galilean{
        (v_galilean[0] != 0.) || (v_galilean[1] != 0.) || (v_galilean[2] != 0.)}
{
    const amrex::BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Always allocate these coefficients
    C_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    S_ck_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X1_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    X2_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    X3_coef = SpectralComplexCoefficients(ba, dm, 1, 0);

    // Allocate these coefficients only with Galilean PSATD
    if (m_is_galilean)
    {
        X4_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
        T2_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    }

    InitializeSpectralCoefficients(spectral_kspace, dm, dt);

    // Allocate these coefficients only with time averaging
    if (time_averaging)
    {
        Psi1_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
        Psi2_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
        Y1_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
        Y3_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
        Y2_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
        Y4_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
        InitializeSpectralCoefficientsAveraging(spectral_kspace, dm, dt);
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !dive_cleaning || !m_is_galilean,
        "warpx.do_dive_cleaning = 1 not implemented for Galilean PSATD algorithms"
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !divb_cleaning || !m_is_galilean,
        "warpx.do_divb_cleaning = 1 not implemented for Galilean PSATD algorithms"
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !time_averaging || update_with_rho,
        "PSATD: psatd.time_averaging = 1 implemented only with psatd.update_with_rho = 1"
    );
}

void
PsatdAlgorithmJConstantInTime::pushSpectralFields (SpectralFieldData& f) const
{
    const bool update_with_rho = m_update_with_rho;
    const bool time_averaging  = m_time_averaging;
    const bool dive_cleaning   = m_dive_cleaning;
    const bool divb_cleaning   = m_divb_cleaning;
    const bool is_galilean     = m_is_galilean;

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
        const amrex::Array4<const Complex> X1_arr = X1_coef[mfi].array();
        const amrex::Array4<const Complex> X2_arr = X2_coef[mfi].array();
        const amrex::Array4<const Complex> X3_arr = X3_coef[mfi].array();

        amrex::Array4<const Complex> X4_arr;
        amrex::Array4<const Complex> T2_arr;
        if (is_galilean)
        {
            X4_arr = X4_coef[mfi].array();
            T2_arr = T2_coef[mfi].array();
        }

        // These coefficients are allocated only with averaged Galilean PSATD
        amrex::Array4<const Complex> Psi1_arr;
        amrex::Array4<const Complex> Psi2_arr;
        amrex::Array4<const Complex> Y1_arr;
        amrex::Array4<const Complex> Y2_arr;
        amrex::Array4<const Complex> Y3_arr;
        amrex::Array4<const Complex> Y4_arr;

        if (time_averaging)
        {
            Psi1_arr = Psi1_coef[mfi].array();
            Psi2_arr = Psi2_coef[mfi].array();
            Y1_arr = Y1_coef[mfi].array();
            Y2_arr = Y2_coef[mfi].array();
            Y3_arr = Y3_coef[mfi].array();
            Y4_arr = Y4_coef[mfi].array();
        }

        // Extract pointers for the k vectors
        const amrex::Real* modified_kx_arr = modified_kx_vec[mfi].dataPtr();
        const amrex::Real* modified_kx_arr_c = modified_kx_vec_centered[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* modified_ky_arr = modified_ky_vec[mfi].dataPtr();
        const amrex::Real* modified_ky_arr_c = modified_ky_vec_centered[mfi].dataPtr();
#endif
        const amrex::Real* modified_kz_arr = modified_kz_vec[mfi].dataPtr();
        const amrex::Real* modified_kz_arr_c = modified_kz_vec_centered[mfi].dataPtr();

        // Galilean velocity
        const amrex::Real vgx = m_v_galilean[0];
        const amrex::Real vgy = m_v_galilean[1];
        const amrex::Real vgz = m_v_galilean[2];

        // Loop over indices within one box
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Record old values of the fields to be updated
            const Complex Ex_old = fields(i,j,k,Idx.Ex);
            const Complex Ey_old = fields(i,j,k,Idx.Ey);
            const Complex Ez_old = fields(i,j,k,Idx.Ez);
            const Complex Bx_old = fields(i,j,k,Idx.Bx);
            const Complex By_old = fields(i,j,k,Idx.By);
            const Complex Bz_old = fields(i,j,k,Idx.Bz);

            // Shortcuts for the values of J
            const Complex Jx = fields(i,j,k,Idx.Jx_mid);
            const Complex Jy = fields(i,j,k,Idx.Jy_mid);
            const Complex Jz = fields(i,j,k,Idx.Jz_mid);

            Complex F_old;
            if (dive_cleaning)
            {
                F_old = fields(i,j,k,Idx.F);
            }

            Complex G_old;
            if (divb_cleaning)
            {
                G_old = fields(i,j,k,Idx.G);
            }

            // k vector values
            const amrex::Real kx = modified_kx_arr[i];
            const amrex::Real kx_c = modified_kx_arr_c[i];
#if defined(WARPX_DIM_3D)
            const amrex::Real ky = modified_ky_arr[j];
            const amrex::Real ky_c = modified_ky_arr_c[j];
            const amrex::Real kz = modified_kz_arr[k];
            const amrex::Real kz_c = modified_kz_arr_c[k];
#else
            constexpr amrex::Real ky = 0._rt;
            constexpr amrex::Real ky_c = 0._rt;
            const     amrex::Real kz = modified_kz_arr[j];
            const     amrex::Real kz_c = modified_kz_arr_c[j];
#endif
            // Physical constants and imaginary unit
            constexpr Real c2 = PhysConst::c * PhysConst::c;
            constexpr Real ep0 = PhysConst::ep0;
            constexpr Real inv_ep0 = 1._rt / PhysConst::ep0;
            constexpr Complex I = Complex{0._rt, 1._rt};

            // These coefficients are initialized in the function InitializeSpectralCoefficients
            const amrex::Real C = C_arr(i,j,k);
            const amrex::Real S_ck = S_ck_arr(i,j,k);
            const Complex X1 = X1_arr(i,j,k);
            const Complex X2 = X2_arr(i,j,k);
            const Complex X3 = X3_arr(i,j,k);
            const Complex X4 = (is_galilean) ? X4_arr(i,j,k) : - S_ck / PhysConst::ep0;
            const Complex T2 = (is_galilean) ? T2_arr(i,j,k) : 1.0_rt;

            // Shortcuts for the values of rho
            Complex rho_old, rho_new;
            if (update_with_rho)
            {
                rho_old = fields(i,j,k,Idx.rho_old);
                rho_new = fields(i,j,k,Idx.rho_new);
            }
            else // update_with_rho = 0
            {
                const amrex::Real kc_dot_vg = kx_c*vgx + ky_c*vgy + kz_c*vgz;
                const Complex k_dot_E = kx*Ex_old + ky*Ey_old + kz*Ez_old;
                const Complex k_dot_J = kx*Jx + ky*Jy + kz*Jz;

                rho_old = I*ep0*k_dot_E;

                if (kc_dot_vg == 0._rt)
                {
                    rho_new = rho_old - I*k_dot_J*dt;
                }
                else // Galilean PSATD
                {
                    rho_new = T2*rho_old + (1._rt-T2)*k_dot_J/kc_dot_vg;
                }
            }

            // Update equations for E
            // T2 = 1 always with standard PSATD (zero Galilean velocity)

            fields(i,j,k,Idx.Ex) = T2 * C * Ex_old
                                   + I * c2 * T2 * S_ck * (ky * Bz_old - kz * By_old)
                                   + X4 * Jx - I * (X2 * rho_new - T2 * X3 * rho_old) * kx;

            fields(i,j,k,Idx.Ey) = T2 * C * Ey_old
                                   + I * c2 * T2 * S_ck * (kz * Bx_old - kx * Bz_old)
                                   + X4 * Jy - I * (X2 * rho_new - T2 * X3 * rho_old) * ky;

            fields(i,j,k,Idx.Ez) = T2 * C * Ez_old
                                   + I * c2 * T2 * S_ck * (kx * By_old - ky * Bx_old)
                                   + X4 * Jz - I * (X2 * rho_new - T2 * X3 * rho_old) * kz;

            // Update equations for B
            // T2 = 1 always with standard PSATD (zero Galilean velocity)

            fields(i,j,k,Idx.Bx) = T2 * C * Bx_old
                                   - I * T2 * S_ck * (ky * Ez_old - kz * Ey_old)
                                   + I * X1 * (ky * Jz - kz * Jy);

            fields(i,j,k,Idx.By) = T2 * C * By_old
                                   - I * T2 * S_ck * (kz * Ex_old - kx * Ez_old)
                                   + I * X1 * (kz * Jx - kx * Jz);

            fields(i,j,k,Idx.Bz) = T2 * C * Bz_old
                                   - I * T2 * S_ck * (kx * Ey_old - ky * Ex_old)
                                   + I * X1 * (kx * Jy - ky * Jx);

            if (dive_cleaning)
            {
                const Complex k_dot_J  = kx * Jx + ky * Jy + kz * Jz;
                const Complex k_dot_E = kx * Ex_old + ky * Ey_old + kz * Ez_old;

                fields(i,j,k,Idx.Ex) += I * c2 * S_ck * F_old * kx;
                fields(i,j,k,Idx.Ey) += I * c2 * S_ck * F_old * ky;
                fields(i,j,k,Idx.Ez) += I * c2 * S_ck * F_old * kz;

                fields(i,j,k,Idx.F) = C * F_old + S_ck * (I * k_dot_E - rho_old * inv_ep0)
                    - X1 * ((rho_new - rho_old) / dt + I * k_dot_J);
            }

            if (divb_cleaning)
            {
                const Complex k_dot_B = kx * Bx_old + ky * By_old + kz * Bz_old;

                fields(i,j,k,Idx.Bx) += I * S_ck * G_old * kx;
                fields(i,j,k,Idx.By) += I * S_ck * G_old * ky;
                fields(i,j,k,Idx.Bz) += I * S_ck * G_old * kz;

                fields(i,j,k,Idx.G) = C * G_old + I * c2 * S_ck * k_dot_B;
            }

            // Additional update equations for averaged Galilean algorithm
            if (time_averaging)
            {
                // These coefficients are initialized in the function InitializeSpectralCoefficients below
                const Complex Psi1 = Psi1_arr(i,j,k);
                const Complex Psi2 = Psi2_arr(i,j,k);
                const Complex Y1 = Y1_arr(i,j,k);
                const Complex Y3 = Y3_arr(i,j,k);
                const Complex Y2 = Y2_arr(i,j,k);
                const Complex Y4 = Y4_arr(i,j,k);

                fields(i,j,k,Idx.Ex_avg) = Psi1 * Ex_old
                                           - I * c2 * Psi2 * (ky * Bz_old - kz * By_old)
                                           + Y4 * Jx + (Y2 * rho_new + Y3 * rho_old) * kx;

                fields(i,j,k,Idx.Ey_avg) = Psi1 * Ey_old
                                           - I * c2 * Psi2 * (kz * Bx_old - kx * Bz_old)
                                           + Y4 * Jy + (Y2 * rho_new + Y3 * rho_old) * ky;

                fields(i,j,k,Idx.Ez_avg) = Psi1 * Ez_old
                                           - I * c2 * Psi2 * (kx * By_old - ky * Bx_old)
                                           + Y4 * Jz + (Y2 * rho_new + Y3 * rho_old) * kz;

                fields(i,j,k,Idx.Bx_avg) = Psi1 * Bx_old
                                           + I * Psi2 * (ky * Ez_old - kz * Ey_old)
                                           + I * Y1 * (ky * Jz - kz * Jy);

                fields(i,j,k,Idx.By_avg) = Psi1 * By_old
                                           + I * Psi2 * (kz * Ex_old - kx * Ez_old)
                                           + I * Y1 * (kz * Jx - kx * Jz);

                fields(i,j,k,Idx.Bz_avg) = Psi1 * Bz_old
                                           + I * Psi2 * (kx * Ey_old - ky * Ex_old)
                                           + I * Y1 * (kx * Jy - ky * Jx);
            }
        });
    }
}

void PsatdAlgorithmJConstantInTime::InitializeSpectralCoefficients (
    const SpectralKSpace& spectral_kspace,
    const amrex::DistributionMapping& dm,
    const amrex::Real dt)
{
    const bool is_galilean     = m_is_galilean;

    const amrex::BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Loop over boxes and allocate the corresponding coefficients for each box
    for (amrex::MFIter mfi(ba, dm); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = ba[mfi];

        // Extract pointers for the k vectors
        const amrex::Real* kx_s = modified_kx_vec[mfi].dataPtr();
        const amrex::Real* kx_c = modified_kx_vec_centered[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* ky_s = modified_ky_vec[mfi].dataPtr();
        const amrex::Real* ky_c = modified_ky_vec_centered[mfi].dataPtr();
#endif
        const amrex::Real* kz_s = modified_kz_vec[mfi].dataPtr();
        const amrex::Real* kz_c = modified_kz_vec_centered[mfi].dataPtr();

        // Coefficients always allocated
        const amrex::Array4<amrex::Real> C = C_coef[mfi].array();
        const amrex::Array4<amrex::Real> S_ck = S_ck_coef[mfi].array();
        const amrex::Array4<Complex> X1 = X1_coef[mfi].array();
        const amrex::Array4<Complex> X2 = X2_coef[mfi].array();
        const amrex::Array4<Complex> X3 = X3_coef[mfi].array();

        amrex::Array4<Complex> X4;
        amrex::Array4<Complex> T2;
        if (is_galilean)
        {
            X4 = X4_coef[mfi].array();
            T2 = T2_coef[mfi].array();
        }

        // Extract Galilean velocity
        const amrex::Real vg_x = m_v_galilean[0];
#if defined(WARPX_DIM_3D)
        const amrex::Real vg_y = m_v_galilean[1];
#endif
        const amrex::Real vg_z = m_v_galilean[2];

        // Loop over indices within one box
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
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
            constexpr Complex I = Complex{0._rt, 1._rt};

            const amrex::Real c2 = amrex::Math::powi<2>(c);
            const amrex::Real dt2 = amrex::Math::powi<2>(dt);

            // Calculate the dot product of the k vector with the Galilean velocity.
            // This has to be computed always with the centered (collocated) finite-order
            // modified k vectors, to work correctly for both collocated and staggered grids.
            // w_c = 0 always with standard PSATD (zero Galilean velocity).
            const amrex::Real w_c = kx_c[i]*vg_x +
#if defined(WARPX_DIM_3D)
                ky_c[j]*vg_y + kz_c[k]*vg_z;
#else
                kz_c[j]*vg_z;
#endif
            const amrex::Real w2_c = amrex::Math::powi<2>(w_c);

            const amrex::Real om_s = c * knorm_s;
            const amrex::Real om2_s = amrex::Math::powi<2>(om_s);

            const Complex theta_c      = amrex::exp( I * w_c * dt * 0.5_rt);
            const Complex theta2_c     = amrex::exp( I * w_c * dt);
            const Complex theta_c_star = amrex::exp(-I * w_c * dt * 0.5_rt);

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

            // Auxiliary variable
            const amrex::Real tmp = (om_s != 0.)?
                ((1._rt - C(i,j,k)) / (ep0 * om2_s)):(0.5_rt * dt2 / ep0);

            // T2
            if (is_galilean)
            {
                T2(i,j,k) = theta_c * theta_c;
            }

            // X1 (multiplies i*([k] \times J) in the update equation for update B)
            if ((om_s != 0.) || (w_c != 0.))
            {
                X1(i,j,k) = (1._rt - theta2_c * C(i,j,k) + I * w_c * theta2_c * S_ck(i,j,k))
                            / (ep0 * (om2_s - w2_c));
            }
            else // om_s = 0 and w_c = 0
            {
                X1(i,j,k) = 0.5_rt * dt2 / ep0;
            }

            // X2 (multiplies rho_new in the update equation for E)
            if (w_c != 0.)
            {
                X2(i,j,k) = c2 * (theta_c_star * X1(i,j,k) - theta_c * tmp)
                            / (theta_c_star - theta_c);
            }
            else // w_c = 0
            {
                if (om_s != 0.)
                {
                    X2(i,j,k) = c2 * (dt - S_ck(i,j,k)) / (ep0 * dt * om2_s);
                }
                else // om_s = 0 and w_c = 0
                {
                    X2(i,j,k) = c2 * dt2 / (6._rt * ep0);
                }
            }

            // X3 (multiplies rho_old in the update equation for E)
            if (w_c != 0.)
            {
                X3(i,j,k) = c2 * (theta_c_star * X1(i,j,k) - theta_c_star * tmp)
                            / (theta_c_star - theta_c);
            }
            else // w_c = 0
            {
                if (om_s != 0.)
                {
                    X3(i,j,k) = c2 * (dt * C(i,j,k) - S_ck(i,j,k)) / (ep0 * dt * om2_s);
                }
                else // om_s = 0 and w_c = 0
                {
                    X3(i,j,k) = - c2 * dt2 / (3._rt * ep0);
                }
            }

            // X4 (multiplies J in the update equation for E)
            if (is_galilean)
            {
                X4(i,j,k) = I * w_c * X1(i,j,k) - theta2_c * S_ck(i,j,k) / ep0;
            }
        });
    }
}

void PsatdAlgorithmJConstantInTime::InitializeSpectralCoefficientsAveraging (
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
        const amrex::Real* kx_c = modified_kx_vec_centered[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* ky_s = modified_ky_vec[mfi].dataPtr();
        const amrex::Real* ky_c = modified_ky_vec_centered[mfi].dataPtr();
#endif
        const amrex::Real* kz_s = modified_kz_vec[mfi].dataPtr();
        const amrex::Real* kz_c = modified_kz_vec_centered[mfi].dataPtr();

        // Coefficients allocated only with averaged Galilean PSATD
        const amrex::Array4<Complex> Psi1 = Psi1_coef[mfi].array();
        const amrex::Array4<Complex> Psi2 = Psi2_coef[mfi].array();
        const amrex::Array4<Complex> Y1 = Y1_coef[mfi].array();
        const amrex::Array4<Complex> Y3 = Y3_coef[mfi].array();
        const amrex::Array4<Complex> Y2 = Y2_coef[mfi].array();
        const amrex::Array4<Complex> Y4 = Y4_coef[mfi].array();

        // Extract Galilean velocity
        const amrex::Real vg_x = m_v_galilean[0];
#if defined(WARPX_DIM_3D)
        const amrex::Real vg_y = m_v_galilean[1];
#endif
        const amrex::Real vg_z = m_v_galilean[2];

        // Loop over indices within one box
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
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
            constexpr Complex I = Complex{0._rt, 1._rt};

            const amrex::Real c2 = amrex::Math::powi<2>(c);
            const amrex::Real dt2 = amrex::Math::powi<2>(dt);

            // Calculate the dot product of the k vector with the Galilean velocity.
            // This has to be computed always with the centered (collocated) finite-order
            // modified k vectors, to work correctly for both collocated and staggered grids.
            // w_c = 0 always with standard PSATD (zero Galilean velocity).
            const amrex::Real w_c = kx_c[i]*vg_x +
#if defined(WARPX_DIM_3D)
                ky_c[j]*vg_y + kz_c[k]*vg_z;
#else
                kz_c[j]*vg_z;
#endif
            const amrex::Real w2_c = amrex::Math::powi<2>(w_c);
            const amrex::Real w3_c = amrex::Math::powi<3>(w_c);

            const amrex::Real om_s = c * knorm_s;
            const amrex::Real om2_s = amrex::Math::powi<2>(om_s);
            const amrex::Real om4_s = amrex::Math::powi<4>(om_s);

            const Complex theta_c  = amrex::exp(I * w_c * dt * 0.5_rt);
            const Complex theta2_c = amrex::exp(I * w_c * dt);
            const Complex theta3_c = amrex::exp(I * w_c * dt * 1.5_rt);
            const Complex theta5_c = amrex::exp(I * w_c * dt * 2.5_rt);

            // C1,C3
            const amrex::Real C1 = std::cos(0.5_rt * om_s * dt);
            const amrex::Real C3 = std::cos(1.5_rt * om_s * dt);

            const amrex::Real S1_om = (om_s != 0.)?
                (std::sin(0.5_rt * om_s * dt) / om_s) : (0.5_rt * dt);
            const amrex::Real S3_om = (om_s != 0.)?
                 (std::sin(1.5_rt * om_s * dt) / om_s) : (1.5_rt * dt);

            // Psi1 (multiplies E in the update equation for <E>)
            // Psi1 (multiplies B in the update equation for <B>)
            if ((om_s != 0.) || (w_c != 0.))
            {
                Psi1(i,j,k) = (theta3_c * (om2_s * S3_om + I * w_c * C3)
                              - theta_c * (om2_s * S1_om + I * w_c * C1)) / (dt * (om2_s - w2_c));
            }
            else // om_s = 0 and w_c = 0
            {
                Psi1(i,j,k) = 1._rt;
            }

            // Psi2 (multiplies i*([k] \times B) in the update equation for <E>)
            // Psi2 (multiplies i*([k] \times E) in the update equation for <B>)
            if ((om_s != 0.) || (w_c != 0.))
            {
                Psi2(i,j,k) = (theta3_c * (C3 - I * w_c * S3_om)
                              - theta_c * (C1 - I * w_c * S1_om)) / (dt * (om2_s - w2_c));
            }
            else // om_s = 0 and w_c = 0
            {
                Psi2(i,j,k) = - dt;
            }

            // Psi3
            Complex Psi3;
            if (w_c != 0.)
            {
                Psi3 = - I * (theta3_c - theta_c) / (dt * w_c);
            }
            else // w_c = 0
            {
                Psi3 = 1._rt;
            }

            // Y1 (multiplies i*([k] \times J) in the update equation for <B>)
            if ((om_s != 0.) || (w_c != 0.))
            {
                Y1(i,j,k) = (1._rt - Psi1(i,j,k) - I * w_c * Psi2(i,j,k)) / (ep0 * (om2_s - w2_c));
            }
            else // om_s = 0 and w_c = 0
            {
                Y1(i,j,k) = 13._rt * dt2 / (24._rt * ep0);
            }

            // Y2 (multiplies rho_new in the update equation for <E>)
            if ((om_s != 0.) && (w_c != 0.))
            {
                Y2(i,j,k) = I * c2 * (ep0 * om2_s * Y1(i,j,k) - Psi3 + Psi1(i,j,k))
                            / (ep0 * om2_s * (theta2_c - 1._rt));
            }
            else if ((om_s != 0.) && (w_c == 0.))
            {
                Y2(i,j,k) = I * c2 * (C1 - C3 - dt2 * om2_s) / (ep0 * dt2 * om4_s);
            }
            else if ((om_s == 0.) && (w_c != 0.))
            {
                Y2(i,j,k) = c2 * (9._rt * dt2 * w2_c * theta3_c - dt2 * w2_c * theta_c
                            - 24._rt * theta3_c + 24._rt * theta_c + I * 8._rt * dt * w_c
                            + I * 24._rt * dt * w_c * theta3_c - I * 8._rt * dt * w_c * theta_c)
                            / (8._rt * ep0 * dt * w3_c * (1._rt - theta2_c));
            }
            else // om_s = 0 and w_c = 0
            {
                Y2(i,j,k) = - I * 5._rt * c2 * dt2 / (24._rt * ep0);
            }

            // Y3 (multiplies rho_old in the update equation for <E>)
            if ((om_s != 0.) && (w_c != 0.))
            {
                Y3(i,j,k) = I * c2 * (Psi3 - Psi1(i,j,k) - ep0 * theta2_c * om2_s * Y1(i,j,k))
                            / (ep0 * om2_s * (theta2_c - 1._rt));
            }
            else if ((om_s != 0.) && (w_c == 0.))
            {
                Y3(i,j,k) = I * c2 * (C3 - C1 + dt * om2_s * (S3_om - S1_om)) / (ep0 * dt2 * om4_s);
            }
            else if ((om_s == 0.) && (w_c != 0.))
            {
                Y3(i,j,k) = c2 * (9._rt * dt2 * w2_c * theta3_c - dt2 * w2_c * theta_c
                            - 16._rt * theta5_c + 8._rt * theta3_c + 8._rt * theta_c
                            + I * 12._rt * dt * w_c * theta5_c + I * 8._rt * dt * w_c * theta3_c
                            - I * 4._rt * dt * w_c * theta_c + I * 8._rt * dt * w_c * theta2_c)
                            / (8._rt * ep0 * dt * w3_c * (theta2_c - 1._rt));
            }
            else // om_s = 0 and w_c = 0
            {
                Y3(i,j,k) = - I * c2 * dt2 / (3._rt * ep0);
            }

            // Y4 (multiplies J in the update equation for <E>)
            Y4(i,j,k) = (Psi2(i,j,k) + I * ep0 * w_c * Y1(i,j,k)) / ep0;
        });
    }
}

void PsatdAlgorithmJConstantInTime::CurrentCorrection (SpectralFieldData& field_data)
{
    // Profiling
    BL_PROFILE("PsatdAlgorithmJConstantInTime::CurrentCorrection");

    const SpectralFieldIndex& Idx = m_spectral_index;

    // Loop over boxes
    for (amrex::MFIter mfi(field_data.fields); mfi.isValid(); ++mfi){

        const amrex::Box& bx = field_data.fields[mfi].box();

        // Extract arrays for the fields to be updated
        const amrex::Array4<Complex> fields = field_data.fields[mfi].array();

        // Extract pointers for the k vectors
        const amrex::Real* const modified_kx_arr = modified_kx_vec[mfi].dataPtr();
        const amrex::Real* const modified_kx_arr_c = modified_kx_vec_centered[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* const modified_ky_arr = modified_ky_vec[mfi].dataPtr();
        const amrex::Real* const modified_ky_arr_c = modified_ky_vec_centered[mfi].dataPtr();
#endif
        const amrex::Real* const modified_kz_arr = modified_kz_vec[mfi].dataPtr();
        const amrex::Real* const modified_kz_arr_c = modified_kz_vec_centered[mfi].dataPtr();

        // Local copy of member variables before GPU loop
        const amrex::Real dt = m_dt;

        // Galilean velocity
        const amrex::Real vgx = m_v_galilean[0];
        const amrex::Real vgy = m_v_galilean[1];
        const amrex::Real vgz = m_v_galilean[2];

        // Loop over indices within one box
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Shortcuts for the values of J and rho
            const Complex Jx = fields(i,j,k,Idx.Jx_mid);
            const Complex Jy = fields(i,j,k,Idx.Jy_mid);
            const Complex Jz = fields(i,j,k,Idx.Jz_mid);
            const Complex rho_old = fields(i,j,k,Idx.rho_old);
            const Complex rho_new = fields(i,j,k,Idx.rho_new);

            // k vector values, and coefficients
            const amrex::Real kx = modified_kx_arr[i];
            const amrex::Real kx_c = modified_kx_arr_c[i];
#if defined(WARPX_DIM_3D)
            const amrex::Real ky = modified_ky_arr[j];
            const amrex::Real kz = modified_kz_arr[k];
            const amrex::Real ky_c = modified_ky_arr_c[j];
            const amrex::Real kz_c = modified_kz_arr_c[k];
#else
            constexpr amrex::Real ky = 0._rt;
            const     amrex::Real kz = modified_kz_arr[j];
            constexpr amrex::Real ky_c = 0._rt;
            const     amrex::Real kz_c = modified_kz_arr_c[j];
#endif
            constexpr Complex I = Complex{0._rt, 1._rt};

            const amrex::Real k_norm = std::sqrt(kx * kx + ky * ky + kz * kz);

            // Correct J
            if (k_norm != 0._rt)
            {
                const Complex k_dot_J = kx * Jx + ky * Jy + kz * Jz;
                const amrex::Real k_dot_vg = kx_c * vgx + ky_c * vgy + kz_c * vgz;

                // k_dot_vg = 0 always with standard PSATD (zero Galilean velocity)
                if ( k_dot_vg != 0._rt )
                {
                    const Complex rho_old_mod = rho_old * amrex::exp(I * k_dot_vg * dt);
                    const Complex den = 1._rt - amrex::exp(I * k_dot_vg * dt);

                    fields(i,j,k,Idx.Jx_mid) = Jx - (k_dot_J - k_dot_vg * (rho_new - rho_old_mod) / den)
                        * kx / (k_norm * k_norm);

                    fields(i,j,k,Idx.Jy_mid) = Jy - (k_dot_J - k_dot_vg * (rho_new - rho_old_mod) / den)
                        * ky / (k_norm * k_norm);

                    fields(i,j,k,Idx.Jz_mid) = Jz - (k_dot_J - k_dot_vg * (rho_new - rho_old_mod) / den)
                        * kz / (k_norm * k_norm);
                }

                else
                {
                    fields(i,j,k,Idx.Jx_mid) = Jx - (k_dot_J - I * (rho_new - rho_old) / dt)
                        * kx / (k_norm * k_norm);

                    fields(i,j,k,Idx.Jy_mid) = Jy - (k_dot_J - I * (rho_new - rho_old) / dt)
                        * ky / (k_norm * k_norm);

                    fields(i,j,k,Idx.Jz_mid) = Jz - (k_dot_J - I * (rho_new - rho_old) / dt)
                        * kz / (k_norm * k_norm);
                }
            }
        });
    }
}

void
PsatdAlgorithmJConstantInTime::VayDeposition (SpectralFieldData& field_data)
{
    // Profiling
    BL_PROFILE("PsatdAlgorithmJConstantInTime::VayDeposition()");

    const SpectralFieldIndex& Idx = m_spectral_index;

    // Loop over boxes
    for (amrex::MFIter mfi(field_data.fields); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = field_data.fields[mfi].box();

        // Extract arrays for the fields to be updated
        const amrex::Array4<Complex> fields = field_data.fields[mfi].array();

        // Extract pointers for the modified k vectors
        const amrex::Real* const modified_kx_arr = modified_kx_vec[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* const modified_ky_arr = modified_ky_vec[mfi].dataPtr();
#endif
        const amrex::Real* const modified_kz_arr = modified_kz_vec[mfi].dataPtr();

        // Loop over indices within one box
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Shortcuts for the values of D
            const Complex Dx = fields(i,j,k,Idx.Jx_mid);
#if defined(WARPX_DIM_3D)
            const Complex Dy = fields(i,j,k,Idx.Jy_mid);
#endif
            const Complex Dz = fields(i,j,k,Idx.Jz_mid);

            // Imaginary unit
            constexpr Complex I = Complex{0._rt, 1._rt};

            // Modified k vector values
            const amrex::Real kx_mod = modified_kx_arr[i];
#if defined(WARPX_DIM_3D)
            const amrex::Real ky_mod = modified_ky_arr[j];
            const amrex::Real kz_mod = modified_kz_arr[k];
#else
            const amrex::Real kz_mod = modified_kz_arr[j];
#endif

            // Compute Jx
            if (kx_mod != 0._rt) { fields(i,j,k,Idx.Jx_mid) = I * Dx / kx_mod; }
            else                 { fields(i,j,k,Idx.Jx_mid) = 0._rt; }

#if defined(WARPX_DIM_3D)
            // Compute Jy
            if (ky_mod != 0._rt) { fields(i,j,k,Idx.Jy_mid) = I * Dy / ky_mod; }
            else                 { fields(i,j,k,Idx.Jy_mid) = 0._rt; }
#endif

            // Compute Jz
            if (kz_mod != 0._rt) { fields(i,j,k,Idx.Jz_mid) = I * Dz / kz_mod; }
            else                 { fields(i,j,k,Idx.Jz_mid) = 0._rt; }
        });
    }
}

#endif // WARPX_USE_FFT
