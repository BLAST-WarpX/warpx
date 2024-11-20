/* Copyright 2019 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PsatdAlgorithmPml.H"

#include "FieldSolver/SpectralSolver/SpectralFieldData.H"
#include "FieldSolver/SpectralSolver/SpectralKSpace.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpX_Complex.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_PODVector.H>

#include <cmath>

#if WARPX_USE_FFT

using namespace amrex;

PsatdAlgorithmPml::PsatdAlgorithmPml(
        const SpectralKSpace& spectral_kspace,
        const DistributionMapping& dm,
        const SpectralFieldIndex& spectral_index,
        int norder_x,
        int norder_y,
        int norder_z,
        ablastr::utils::enums::GridType grid_type,
        const amrex::Vector<amrex::Real>& v_galilean,
        Real dt,
        bool dive_cleaning,
        bool divb_cleaning) :
    SpectralBaseAlgorithm(spectral_kspace, dm, spectral_index, norder_x, norder_y, norder_z, grid_type),
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
    m_dive_cleaning(dive_cleaning),
    m_divb_cleaning(divb_cleaning),
    m_is_galilean{(v_galilean[0] != 0.) || (v_galilean[1] != 0.) || (v_galilean[2] != 0.)}
{
    const BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Allocate arrays of coefficients
    C_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    S_ck_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    inv_k2_coef = SpectralRealCoefficients(ba, dm, 1, 0);

    // Allocate this coefficient only with Galilean PSATD
    if (m_is_galilean)
    {
        T2_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    }

    InitializeSpectralCoefficients(spectral_kspace, dm);
}

void PsatdAlgorithmPml::pushSpectralFields(SpectralFieldData& f) const
{
    const bool dive_cleaning = m_dive_cleaning;
    const bool divb_cleaning = m_divb_cleaning;
    const bool is_galilean   = m_is_galilean;

    const SpectralFieldIndex& Idx = m_spectral_index;

    // Loop over boxes
    for (amrex::MFIter mfi(f.fields); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = f.fields[mfi].box();

        // Extract arrays for the fields to be updated
        const amrex::Array4<Complex> fields = f.fields[mfi].array();

        // Extract arrays for the coefficients
        const amrex::Array4<const amrex::Real> C_arr = C_coef[mfi].array();
        const amrex::Array4<const amrex::Real> S_ck_arr = S_ck_coef[mfi].array();
        const amrex::Array4<const amrex::Real> inv_k2_arr = inv_k2_coef[mfi].array();

        amrex::Array4<const Complex> T2_arr;
        if (is_galilean)
        {
            T2_arr = T2_coef[mfi].array();
        }

        // Extract pointers for the k vectors
        const amrex::Real* modified_kx_arr = modified_kx_vec[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* modified_ky_arr = modified_ky_vec[mfi].dataPtr();
#endif
        const amrex::Real* modified_kz_arr = modified_kz_vec[mfi].dataPtr();

        const amrex::Real dt = m_dt;

        // Loop over indices within one box
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Record old values of the fields to be updated
            const Complex Exy = fields(i,j,k,Idx.Exy);
            const Complex Exz = fields(i,j,k,Idx.Exz);
            const Complex Eyx = fields(i,j,k,Idx.Eyx);
            const Complex Eyz = fields(i,j,k,Idx.Eyz);
            const Complex Ezx = fields(i,j,k,Idx.Ezx);
            const Complex Ezy = fields(i,j,k,Idx.Ezy);

            const Complex Bxy = fields(i,j,k,Idx.Bxy);
            const Complex Bxz = fields(i,j,k,Idx.Bxz);
            const Complex Byx = fields(i,j,k,Idx.Byx);
            const Complex Byz = fields(i,j,k,Idx.Byz);
            const Complex Bzx = fields(i,j,k,Idx.Bzx);
            const Complex Bzy = fields(i,j,k,Idx.Bzy);

            Complex Ex, Ey, Ez;
            Complex Bx, By, Bz;

            // Used only if dive_cleaning = true and divb_cleaning = true
            Complex Exx, Eyy, Ezz;
            Complex Bxx, Byy, Bzz;
            Complex Fx, Fy, Fz;
            Complex Gx, Gy, Gz;
            Complex F, G;

            if (!dive_cleaning && !divb_cleaning)
            {
                Ex = Exy + Exz;
                Ey = Eyx + Eyz;
                Ez = Ezx + Ezy;

                Bx = Bxy + Bxz;
                By = Byx + Byz;
                Bz = Bzx + Bzy;
            }
            else if (dive_cleaning && divb_cleaning)
            {
                Exx = fields(i,j,k,Idx.Exx);
                Eyy = fields(i,j,k,Idx.Eyy);
                Ezz = fields(i,j,k,Idx.Ezz);

                Bxx = fields(i,j,k,Idx.Bxx);
                Byy = fields(i,j,k,Idx.Byy);
                Bzz = fields(i,j,k,Idx.Bzz);

                Fx = fields(i,j,k,Idx.Fx);
                Fy = fields(i,j,k,Idx.Fy);
                Fz = fields(i,j,k,Idx.Fz);

                Gx = fields(i,j,k,Idx.Gx);
                Gy = fields(i,j,k,Idx.Gy);
                Gz = fields(i,j,k,Idx.Gz);

                Ex = Exx + Exy + Exz;
                Ey = Eyx + Eyy + Eyz;
                Ez = Ezx + Ezy + Ezz;

                Bx = Bxx + Bxy + Bxz;
                By = Byx + Byy + Byz;
                Bz = Bzx + Bzy + Bzz;

                F = Fx + Fy + Fz;
                G = Gx + Gy + Gz;
            }

            // k vector values, and coefficients
            const amrex::Real kx = modified_kx_arr[i];
#if defined(WARPX_DIM_3D)
            const amrex::Real ky = modified_ky_arr[j];
            const amrex::Real kz = modified_kz_arr[k];
#else
            constexpr amrex::Real ky = 0._rt;
            const amrex::Real kz = modified_kz_arr[j];
#endif
            constexpr amrex::Real c2 = PhysConst::c*PhysConst::c;

            const Complex I = Complex{0._rt, 1._rt};

            const amrex::Real kx2 = kx*kx;
            const amrex::Real ky2 = ky*ky;
            const amrex::Real kz2 = kz*kz;
            const amrex::Real knorm = std::sqrt(kx2 + ky2 + kz2);

            if (knorm != 0._rt)
            {
                const amrex::Real C = C_arr(i,j,k);
                const amrex::Real S_ck = S_ck_arr(i,j,k);
                const amrex::Real inv_k2 = inv_k2_arr(i,j,k);
                const Complex T2 = (is_galilean) ? T2_arr(i,j,k) : 1.0_rt;

                const amrex::Real C1 = (kx2 * C + ky2 + kz2) * inv_k2;
                const amrex::Real C2 = (kx2 + ky2 * C + kz2) * inv_k2;
                const amrex::Real C3 = (kx2 + ky2 + kz2 * C) * inv_k2;
                const amrex::Real C4 = kx2 * (C - 1._rt) * inv_k2;
                const amrex::Real C5 = ky2 * (C - 1._rt) * inv_k2;
                const amrex::Real C6 = kz2 * (C - 1._rt) * inv_k2;
                const amrex::Real C7 = ky * kz * (1._rt - C) * inv_k2;
                const amrex::Real C8 = kx * kz * (1._rt - C) * inv_k2;
                const amrex::Real C9 = kx * ky * (1._rt - C) * inv_k2;

                if (!dive_cleaning && !divb_cleaning)
                {
                    const Complex C10 = I * c2 * kx * ky * kz * (dt - S_ck) * inv_k2;
                    const Complex C11 = I * c2 * ky2 * kz * (dt - S_ck) * inv_k2;
                    const Complex C12 = I * c2 * kz2 * ky * (dt - S_ck) * inv_k2;
                    const Complex C13 = I * c2 * kz2 * kx * (dt - S_ck) * inv_k2;
                    const Complex C14 = I * c2 * kx2 * kz * (dt - S_ck) * inv_k2;
                    const Complex C15 = I * c2 * kx2 * ky * (dt - S_ck) * inv_k2;
                    const Complex C16 = I * c2 * ky2 * kx * (dt - S_ck) * inv_k2;
                    const Complex C17 = I * c2 * kx * (ky2 * dt + (kz2 + kx2) * S_ck) * inv_k2;
                    const Complex C18 = I * c2 * kx * (kz2 * dt + (ky2 + kx2) * S_ck) * inv_k2;
                    const Complex C19 = I * c2 * ky * (kz2 * dt + (kx2 + ky2) * S_ck) * inv_k2;
                    const Complex C20 = I * c2 * ky * (kx2 * dt + (kz2 + ky2) * S_ck) * inv_k2;
                    const Complex C21 = I * c2 * kz * (kx2 * dt + (ky2 + kz2) * S_ck) * inv_k2;
                    const Complex C22 = I * c2 * kz * (ky2 * dt + (kx2 + kz2) * S_ck) * inv_k2;

                    const Complex C10_c2 = C10 / c2;
                    const Complex C11_c2 = C11 / c2;
                    const Complex C12_c2 = C12 / c2;
                    const Complex C13_c2 = C13 / c2;
                    const Complex C14_c2 = C14 / c2;
                    const Complex C15_c2 = C15 / c2;
                    const Complex C16_c2 = C16 / c2;
                    const Complex C17_c2 = C17 / c2;
                    const Complex C18_c2 = C18 / c2;
                    const Complex C19_c2 = C19 / c2;
                    const Complex C20_c2 = C20 / c2;
                    const Complex C21_c2 = C21 / c2;
                    const Complex C22_c2 = C22 / c2;

                    // Update E
                    fields(i,j,k,Idx.Exy) = T2 * (C2  * Exy + C5  * Exz + C9  * Ey
                                                + C10 * Bx  + C11 * By  + C19 * Bz);

                    fields(i,j,k,Idx.Exz) = T2 * (C6  * Exy + C3  * Exz + C8  * Ez
                                                - C10 * Bx  - C22 * By  - C12 * Bz);

                    fields(i,j,k,Idx.Eyz) = T2 * (C3  * Eyz + C6  * Eyx + C7  * Ez
                                                + C21 * Bx  + C10 * By  + C13 * Bz);

                    fields(i,j,k,Idx.Eyx) = T2 * (C9  * Ex + C4  * Eyz + C1  * Eyx
                                                - C14 * Bx - C10 * By  - C18 * Bz);

                    fields(i,j,k,Idx.Ezx) = T2 * (C8  * Ex + C1  * Ezx + C4  * Ezy
                                                + C15 * Bx + C17 * By  + C10 * Bz);

                    fields(i,j,k,Idx.Ezy) = T2 * (C7  * Ey + C5  * Ezx + C2  * Ezy
                                                - C20 * Bx - C16 * By  - C10 * Bz);

                    // Update B
                    fields(i,j,k,Idx.Bxy) = T2 * (C2     * Bxy + C5     * Bxz + C9     * By
                                                - C10_c2 * Ex  - C11_c2 * Ey  - C19_c2 * Ez);

                    fields(i,j,k,Idx.Bxz) = T2 * (C6     * Bxy + C3     * Bxz + C8     * Bz
                                                + C10_c2 * Ex  + C22_c2 * Ey  + C12_c2 * Ez);

                    fields(i,j,k,Idx.Byz) = T2 * (C3     * Byz + C6     * Byx + C7     * Bz
                                                - C21_c2 * Ex  - C10_c2 * Ey  - C13_c2 * Ez);

                    fields(i,j,k,Idx.Byx) = T2 * (C9     * Bx + C4     * Byz + C1     * Byx
                                                + C14_c2 * Ex + C10_c2 * Ey  + C18_c2 * Ez);

                    fields(i,j,k,Idx.Bzx) = T2 * (C8     * Bx + C1     * Bzx + C4     * Bzy
                                                - C15_c2 * Ex - C17_c2 * Ey  - C10_c2 * Ez);

                    fields(i,j,k,Idx.Bzy) = T2 * (C7     * By + C5     * Bzx + C2     * Bzy
                                                + C20_c2 * Ex + C16_c2 * Ey  + C10_c2 * Ez);
                }
                else if (dive_cleaning && divb_cleaning)
                {
                    const Complex C23 = I * c2 * kx * S_ck;
                    const Complex C24 = I * c2 * ky * S_ck;
                    const Complex C25 = I * c2 * kz * S_ck;

                    const Complex C23_c2 = C23 / c2;
                    const Complex C24_c2 = C24 / c2;
                    const Complex C25_c2 = C25 / c2;

                    // Update E
                    fields(i,j,k,Idx.Exx) = T2 * (C1 * Exx + C4 * Exy + C4 * Exz
                                                - C9 * Ey  - C8 * Ez  + C23 * F);

                    fields(i,j,k,Idx.Exy) = T2 * (C5 * Exx + C2  * Exy + C5 * Exz
                                                + C9 * Ey  + C24 * Bz  - C7 * G);

                    fields(i,j,k,Idx.Exz) = T2 * (C6 * Exx + C6  * Exy + C3 * Exz
                                                + C8 * Ez  - C25 * By  + C7 * G);

                    fields(i,j,k,Idx.Eyx) = T2 * (C9 * Ex  + C1  * Eyx + C4 * Eyy
                                                + C4 * Eyz - C23 * Bz  + C8 * G);

                    fields(i,j,k,Idx.Eyy) = T2 * (- C9 * Ex  + C5 * Eyx + C2  * Eyy
                                                  + C5 * Eyz - C7 * Ez  + C24 * F);

                    fields(i,j,k,Idx.Eyz) = T2 * (C6 * Eyx + C6  * Eyy + C3 * Eyz
                                                + C7 * Ez  + C25 * Bx  - C8 * G);

                    fields(i,j,k,Idx.Ezx) = T2 * (C8 * Ex  + C1  * Ezx + C4 * Ezy
                                                + C4 * Ezz + C23 * By  - C9 * G);

                    fields(i,j,k,Idx.Ezy) = T2 * (C7 * Ey  + C5  * Ezx + C2 * Ezy
                                                + C5 * Ezz - C24 * Bx  + C9 * G);

                    fields(i,j,k,Idx.Ezz) = T2 * (- C8 * Ex  - C7 * Ey  + C6  * Ezx
                                                  + C6 * Ezy + C3 * Ezz + C25 * F);

                    // Update B
                    fields(i,j,k,Idx.Bxx) = T2 * (C1 * Bxx + C4 * Bxy + C4     * Bxz
                                                - C9 * By  - C8 * Bz  + C23_c2 * G);

                    fields(i,j,k,Idx.Bxy) = T2 * (- C24_c2 * Ez  + C5 * Bxx + C2 * Bxy
                                                  + C5     * Bxz + C9 * By  + C7 * F);

                    fields(i,j,k,Idx.Bxz) = T2 * (C25_c2 * Ey  + C6 * Bxx + C6 * Bxy
                                                + C3     * Bxz + C8 * Bz  - C7 * F);

                    fields(i,j,k,Idx.Byx) = T2 * (C23_c2 * Ez  + C9 * Bx  + C1 * Byx
                                                + C4     * Byy + C4 * Byz - C8 * F);

                    fields(i,j,k,Idx.Byy) = T2 * (- C9 * Bx  + C5 * Byx + C2     * Byy
                                                  + C5 * Byz - C7 * Bz  + C24_c2 * G);

                    fields(i,j,k,Idx.Byz) = T2 * (- C25_c2 * Ex  + C6 * Byx + C6 * Byy
                                                  + C3     * Byz + C7 * Bz  + C8 * F);

                    fields(i,j,k,Idx.Bzx) = T2 * (- C23_c2 * Ey  + C8 * Bx  + C1 * Bzx
                                                  + C4     * Bzy + C4 * Bzz + C9 * F);

                    fields(i,j,k,Idx.Bzy) = T2 * (C24_c2 * Ex  + C7 * By  + C5 * Bzx
                                                + C2     * Bzy + C5 * Bzz - C9 * F);

                    fields(i,j,k,Idx.Bzz) = T2 * (- C8 * Bx  - C7 * By  + C6     * Bzx
                                                  + C6 * Bzy + C3 * Bzz + C25_c2 * G);

                    // Update F
                    fields(i,j,k,Idx.Fx) = T2 * (C23_c2 * Ex + C8 * By - C9 * Bz
                                               + C1     * Fx + C4 * Fy + C4 * Fz);

                    fields(i,j,k,Idx.Fy) = T2 * (C24_c2 * Ey - C7 * Bx + C9 * Bz
                                               + C5     * Fx + C2 * Fy + C5 * Fz);

                    fields(i,j,k,Idx.Fz) = T2 * (C25_c2 * Ez + C7 * Bx - C8 * By
                                               + C6     * Fx + C6 * Fy + C3 * Fz);

                    // Update G
                    fields(i,j,k,Idx.Gx) = T2 * (- C8 * Ey + C9 * Ez + C23 * Bx
                                                 + C1 * Gx + C4 * Gy + C4  * Gz);

                    fields(i,j,k,Idx.Gy) = T2 * (C7 * Ex - C9 * Ez + C24 * By
                                               + C5 * Gx + C2 * Gy + C5  * Gz);

                    fields(i,j,k,Idx.Gz) = T2 * (- C7 * Ex + C8 * Ey + C25 * Bz
                                                 + C6 * Gx + C6 * Gy + C3  * Gz);
                }
            }
        });
    }
}

void PsatdAlgorithmPml::InitializeSpectralCoefficients (
    const SpectralKSpace& spectral_kspace,
    const amrex::DistributionMapping& dm)
{
    const amrex::Real dt = m_dt;
    const bool is_galilean = m_is_galilean;

    const amrex::BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Loop over boxes and allocate the corresponding coefficients
    // for each box owned by the local MPI process
    for (amrex::MFIter mfi(ba, dm); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = ba[mfi];

        // Extract pointers for the k vectors
        const amrex::Real* kx   = modified_kx_vec[mfi].dataPtr();
        const amrex::Real* kx_c = modified_kx_vec_centered[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* ky   = modified_ky_vec[mfi].dataPtr();
        const amrex::Real* ky_c = modified_ky_vec_centered[mfi].dataPtr();
#endif
        const amrex::Real* kz   = modified_kz_vec[mfi].dataPtr();
        const amrex::Real* kz_c = modified_kz_vec_centered[mfi].dataPtr();

        // Extract arrays for the coefficients
        const amrex::Array4<amrex::Real> C = C_coef[mfi].array();
        const amrex::Array4<amrex::Real> S_ck = S_ck_coef[mfi].array();
        const amrex::Array4<amrex::Real> inv_k2 = inv_k2_coef[mfi].array();

        amrex::Array4<Complex> T2;
        if (is_galilean)
        {
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
            // Calculate norm of vector
            const amrex::Real knorm = std::sqrt(
                amrex::Math::powi<2>(kx[i]) +
#if (AMREX_SPACEDIM==3)
                amrex::Math::powi<2>(ky[j]) + amrex::Math::powi<2>(kz[k]));
#else
                amrex::Math::powi<2>(kz[j]));
#endif
            // Calculate the dot product of the k vector with the Galilean velocity.
            // This has to be computed always with the centered (collocated) finite-order
            // modified k vectors, to work correctly for both collocated and staggered grids.
            // w_c = 0 always with standard PSATD (zero Galilean velocity).
            const amrex::Real w_c = kx_c[i]*vg_x +
#if (AMREX_SPACEDIM==3)
                ky_c[j]*vg_y + kz_c[k]*vg_z;
#else
                kz_c[j]*vg_z;
#endif
            constexpr amrex::Real c = PhysConst::c;
            constexpr Complex I{0._rt,1._rt};

            // Coefficients for knorm = 0 do not need to be set
            if (knorm != 0._rt)
            {
                C(i,j,k) = std::cos(c*knorm*dt);
                S_ck(i,j,k) = std::sin(c*knorm*dt) / (c*knorm);
                inv_k2(i,j,k) = 1._rt / (knorm*knorm);

                if (is_galilean)
                {
                    T2(i,j,k) = amrex::exp(I*w_c*dt);
                }
            }
        });
    }
}

void PsatdAlgorithmPml::CurrentCorrection (SpectralFieldData& /*field_data*/)
{
    WARPX_ABORT_WITH_MESSAGE("Current correction not implemented for PML PSATD");
}

void PsatdAlgorithmPml::VayDeposition (SpectralFieldData& /*field_data*/)
{
    WARPX_ABORT_WITH_MESSAGE("Vay deposition not implemented for PML PSATD");
}

#endif // WARPX_USE_FFT
