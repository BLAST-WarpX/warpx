/* Copyright 2021 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PsatdAlgorithmPmlRZ.H"
#include "FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <cmath>

using namespace amrex::literals;

/* \brief Initialize coefficients for the update equation */
PsatdAlgorithmPmlRZ::PsatdAlgorithmPmlRZ (SpectralKSpaceRZ const & spectral_kspace,
                                          amrex::DistributionMapping const & dm,
                                          const SpectralFieldIndex& spectral_index,
                                          int const n_rz_azimuthal_modes,
                                          int const norder_z,
                                          ablastr::utils::enums::GridType grid_type,
                                          amrex::Real const dt):
    // Initialize members of base class and member variables
    SpectralBaseAlgorithmRZ{spectral_kspace, dm, spectral_index, norder_z, grid_type},
    m_dt{dt}
{
    // Allocate the arrays of coefficients
    amrex::BoxArray const & ba = spectral_kspace.spectralspace_ba;
    C_coef = SpectralRealCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
    S_ck_coef = SpectralRealCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
}

/* Advance the E and B field in spectral space (stored in `f`)
 * over one time step */
void
PsatdAlgorithmPmlRZ::pushSpectralFields (SpectralFieldDataRZ & f)
{

    if (not coefficients_initialized) {
        // This is called from here since it needs the kr values
        // which can be obtained from the SpectralFieldDataRZ
        InitializeSpectralCoefficients(f);
        coefficients_initialized = true;
    }

    const SpectralFieldIndex& Idx = m_spectral_index;

    // Loop over boxes
    for (amrex::MFIter mfi(f.fields); mfi.isValid(); ++mfi){

        amrex::Box const & bx = f.fields[mfi].box();

        // Extract arrays for the fields to be updated
        amrex::Array4<Complex> const& fields = f.fields[mfi].array();
        // Extract arrays for the coefficients
        amrex::Array4<const amrex::Real> const& C_arr = C_coef[mfi].array();
        amrex::Array4<const amrex::Real> const& S_ck_arr = S_ck_coef[mfi].array();

        // Extract pointers for the k vectors
        HankelTransform::RealVector const & kr_modes = f.getKrArray(mfi);
        amrex::Real const* kr_arr = kr_modes.dataPtr();
        int const nr = bx.length(0);

        // Loop over indices within one box
        // Note that k = 0
        int const modes = f.n_rz_azimuthal_modes;
        amrex::ParallelFor(bx, modes,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
        {

            // All of the fields of each mode are grouped together
            int const Ep_m_pml = Idx.Er_pml + Idx.n_fields*mode;
            int const Em_m_pml = Idx.Et_pml + Idx.n_fields*mode;
            int const Bp_m_pml = Idx.Br_pml + Idx.n_fields*mode;
            int const Bm_m_pml = Idx.Bt_pml + Idx.n_fields*mode;
            int const Ez_m = Idx.Ez + Idx.n_fields*mode;
            int const Bz_m = Idx.Bz + Idx.n_fields*mode;

            // Record old values of the fields to be updated
            Complex const Ep_old_pml = fields(i,j,k,Ep_m_pml);
            Complex const Em_old_pml = fields(i,j,k,Em_m_pml);
            Complex const Bp_old_pml = fields(i,j,k,Bp_m_pml);
            Complex const Bm_old_pml = fields(i,j,k,Bm_m_pml);
            Complex const Ez_old = fields(i,j,k,Ez_m);
            Complex const Bz_old = fields(i,j,k,Bz_m);

            // k vector values, and coefficients
            // The k values for each mode are grouped together
            int const ir = i + nr*mode;
            amrex::Real const kr = kr_arr[ir];

            constexpr amrex::Real c2 = PhysConst::c*PhysConst::c;
            Complex const I = Complex{0._rt,1._rt};
            amrex::Real const C = C_arr(i,j,k,mode);
            amrex::Real const S_ck = S_ck_arr(i,j,k,mode);

            // Update E (see WarpX online documentation: theory section)
            fields(i,j,k,Ep_m_pml) = C*Ep_old_pml
                        + S_ck*(-c2*I*kr/2._rt*Bz_old);
            fields(i,j,k,Em_m_pml) = C*Em_old_pml
                        + S_ck*(-c2*I*kr/2._rt*Bz_old);
            // Update B (see WarpX online documentation: theory section)
            fields(i,j,k,Bp_m_pml) = C*Bp_old_pml
                        - S_ck*(-I*kr/2._rt*Ez_old);
            fields(i,j,k,Bm_m_pml) = C*Bm_old_pml
                        - S_ck*(-I*kr/2._rt*Ez_old);

        });
    }
}

void PsatdAlgorithmPmlRZ::InitializeSpectralCoefficients (SpectralFieldDataRZ const & f)
{

    // Fill them with the right values:
    // Loop over boxes and allocate the corresponding coefficients
    // for each box owned by the local MPI proc
    for (amrex::MFIter mfi(f.fields); mfi.isValid(); ++mfi){

        amrex::Box const & bx = f.fields[mfi].box();

        // Extract pointers for the k vectors
        amrex::Real const* const modified_kz = modified_kz_vec[mfi].dataPtr();

        // Extract arrays for the coefficients
        amrex::Array4<amrex::Real> const& C = C_coef[mfi].array();
        amrex::Array4<amrex::Real> const& S_ck = S_ck_coef[mfi].array();

        HankelTransform::RealVector const & kr_modes = f.getKrArray(mfi);
        amrex::Real const* kr_arr = kr_modes.dataPtr();
        int const nr = bx.length(0);
        amrex::Real const dt = m_dt;

        // Loop over indices within one box
        int const modes = f.n_rz_azimuthal_modes;
        amrex::ParallelFor(bx, modes,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
        {
            // Calculate norm of vector
            int const ir = i + nr*mode;
            amrex::Real const kr = kr_arr[ir];
            amrex::Real const kz = modified_kz[j];
            amrex::Real const k_norm = std::sqrt(kr*kr + kz*kz);

            // Calculate coefficients
            constexpr amrex::Real c = PhysConst::c;
            if (k_norm != 0._rt){
                C(i,j,k,mode) = std::cos(c*k_norm*dt);
                S_ck(i,j,k,mode) = std::sin(c*k_norm*dt)/(c*k_norm);
            } else { // Handle k_norm = 0, by using the analytical limit
                C(i,j,k,mode) = 1._rt;
                S_ck(i,j,k,mode) = dt;
            }
        });
     }
}

void
PsatdAlgorithmPmlRZ::CurrentCorrection (SpectralFieldDataRZ& /* field_data */)
{
    WARPX_ABORT_WITH_MESSAGE(
        "Current correction not implemented in RZ geometry PML");
}

void
PsatdAlgorithmPmlRZ::VayDeposition (SpectralFieldDataRZ& /*field_data*/)
{
    WARPX_ABORT_WITH_MESSAGE(
        "Vay deposition not implemented in RZ geometry PML");
}
