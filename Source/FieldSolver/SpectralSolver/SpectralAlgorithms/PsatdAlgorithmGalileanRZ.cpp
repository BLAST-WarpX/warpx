/* Copyright 2019-2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PsatdAlgorithmGalileanRZ.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <cmath>

using namespace amrex::literals;

/* \brief Initialize coefficients for the update equation */
PsatdAlgorithmGalileanRZ::PsatdAlgorithmGalileanRZ (SpectralKSpaceRZ const & spectral_kspace,
                                                    amrex::DistributionMapping const & dm,
                                                    const SpectralFieldIndex& spectral_index,
                                                    int const n_rz_azimuthal_modes, int const norder_z,
                                                    ablastr::utils::enums::GridType grid_type,
                                                    const amrex::Vector<amrex::Real>& v_galilean,
                                                    amrex::Real const dt,
                                                    bool const update_with_rho):
    // Initialize members of base class
    SpectralBaseAlgorithmRZ{spectral_kspace, dm, spectral_index, norder_z, grid_type},
    m_dt{dt},
    m_v_galilean{v_galilean},
    m_update_with_rho{update_with_rho}
{
    // Allocate the arrays of coefficients
    amrex::BoxArray const & ba = spectral_kspace.spectralspace_ba;
    C_coef = SpectralRealCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
    S_ck_coef = SpectralRealCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
    X1_coef = SpectralComplexCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
    X2_coef = SpectralComplexCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
    X3_coef = SpectralComplexCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
    X4_coef = SpectralComplexCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
    Theta2_coef = SpectralComplexCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
    T_rho_coef = SpectralComplexCoefficients(ba, dm, n_rz_azimuthal_modes, 0);
}

/* Advance the E and B field in spectral space (stored in `f`)
 * over one time step
 * The algorithm is described in https://doi.org/10.1103/PhysRevE.94.053305
 * */
void
PsatdAlgorithmGalileanRZ::pushSpectralFields (SpectralFieldDataRZ & f)
{

    bool const update_with_rho = m_update_with_rho;

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
        amrex::Array4<const Complex> const& X1_arr = X1_coef[mfi].array();
        amrex::Array4<const Complex> const& X2_arr = X2_coef[mfi].array();
        amrex::Array4<const Complex> const& X3_arr = X3_coef[mfi].array();
        amrex::Array4<const Complex> const& X4_arr = X4_coef[mfi].array();
        amrex::Array4<const Complex> const& Theta2_arr = Theta2_coef[mfi].array();
        amrex::Array4<const Complex> const& T_rho_arr = T_rho_coef[mfi].array();

        // Extract pointers for the k vectors
        auto const & kr_modes = f.getKrArray(mfi);
        amrex::Real const* kr_arr = kr_modes.dataPtr();
        amrex::Real const* modified_kz_arr = modified_kz_vec[mfi].dataPtr();
        int const nr = bx.length(0);

        // Loop over indices within one box
        // Note that k = 0
        int const modes = f.n_rz_azimuthal_modes;
        amrex::ParallelFor(bx, modes,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
        {

            // All of the fields of each mode are grouped together
            auto const Ep_m = Idx.Ex + Idx.n_fields*mode;
            auto const Em_m = Idx.Ey + Idx.n_fields*mode;
            auto const Ez_m = Idx.Ez + Idx.n_fields*mode;
            auto const Bp_m = Idx.Bx + Idx.n_fields*mode;
            auto const Bm_m = Idx.By + Idx.n_fields*mode;
            auto const Bz_m = Idx.Bz + Idx.n_fields*mode;
            auto const Jp_m = Idx.Jx_mid + Idx.n_fields*mode;
            auto const Jm_m = Idx.Jy_mid + Idx.n_fields*mode;
            auto const Jz_m = Idx.Jz_mid + Idx.n_fields*mode;
            auto const rho_old_m = Idx.rho_old + Idx.n_fields*mode;
            auto const rho_new_m = Idx.rho_new + Idx.n_fields*mode;

            // Record old values of the fields to be updated
            Complex const Ep_old = fields(i,j,k,Ep_m);
            Complex const Em_old = fields(i,j,k,Em_m);
            Complex const Ez_old = fields(i,j,k,Ez_m);
            Complex const Bp_old = fields(i,j,k,Bp_m);
            Complex const Bm_old = fields(i,j,k,Bm_m);
            Complex const Bz_old = fields(i,j,k,Bz_m);
            // Shortcut for the values of J and rho
            Complex const Jp = fields(i,j,k,Jp_m);
            Complex const Jm = fields(i,j,k,Jm_m);
            Complex const Jz = fields(i,j,k,Jz_m);
            Complex const rho_old = fields(i,j,k,rho_old_m);
            Complex const rho_new = fields(i,j,k,rho_new_m);

            // k vector values, and coefficients
            // The k values for each mode are grouped together
            int const ir = i + nr*mode;
            amrex::Real const kr = kr_arr[ir];
            amrex::Real const kz = modified_kz_arr[j];

            constexpr amrex::Real c2 = PhysConst::c*PhysConst::c;
            Complex const I = Complex{0._rt,1._rt};
            amrex::Real const C = C_arr(i,j,k,mode);
            amrex::Real const S_ck = S_ck_arr(i,j,k,mode);
            Complex const X1 = X1_arr(i,j,k,mode);
            Complex const X2 = X2_arr(i,j,k,mode);
            Complex const X3 = X3_arr(i,j,k,mode);
            Complex const X4 = X4_arr(i,j,k,mode);
            Complex const T2 = Theta2_arr(i,j,k,mode);
            Complex const T_rho = T_rho_arr(i,j,k,mode);

            Complex rho_diff;
            if (update_with_rho) {
                rho_diff = X2*rho_new - T2*X3*rho_old;
            } else {
                Complex const divE = kr*(Ep_old - Em_old) + I*kz*Ez_old;
                Complex const divJ = kr*(Jp - Jm) + I*kz*Jz;

                auto const myeps0 = PhysConst::ep0;  // temporary for NVCC
                rho_diff = T2*(X2 - X3)*myeps0*divE + T_rho*X2*divJ;
            }

            // Update E (see WarpX online documentation: theory section)
            fields(i,j,k,Ep_m) = T2*C*Ep_old
                        + T2*S_ck*(-c2*I*kr/2._rt*Bz_old + c2*kz*Bp_old)
                        + X4*Jp + 0.5_rt*kr*rho_diff;
            fields(i,j,k,Em_m) = T2*C*Em_old
                        + T2*S_ck*(-c2*I*kr/2._rt*Bz_old - c2*kz*Bm_old)
                        + X4*Jm - 0.5_rt*kr*rho_diff;
            fields(i,j,k,Ez_m) = T2*C*Ez_old
                        + T2*S_ck*(c2*I*kr*Bp_old + c2*I*kr*Bm_old)
                        + X4*Jz - I*kz*rho_diff;
            // Update B (see WarpX online documentation: theory section)
            // Note: here X1 is T2*x1/(ep0*c*c*k_norm*k_norm), where
            // x1 has the same definition as in the original paper
            fields(i,j,k,Bp_m) = T2*C*Bp_old
                        - T2*S_ck*(-I*kr/2._rt*Ez_old + kz*Ep_old)
                        + X1*(-I*kr/2._rt*Jz + kz*Jp);
            fields(i,j,k,Bm_m) = T2*C*Bm_old
                        - T2*S_ck*(-I*kr/2._rt*Ez_old - kz*Em_old)
                        + X1*(-I*kr/2._rt*Jz - kz*Jm);
            fields(i,j,k,Bz_m) = T2*C*Bz_old
                        - T2*S_ck*I*(kr*Ep_old + kr*Em_old)
                        + X1*I*(kr*Jp + kr*Jm);
        });
    }
}

void PsatdAlgorithmGalileanRZ::InitializeSpectralCoefficients (SpectralFieldDataRZ const & f)
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
        amrex::Array4<Complex> const& X1 = X1_coef[mfi].array();
        amrex::Array4<Complex> const& X2 = X2_coef[mfi].array();
        amrex::Array4<Complex> const& X3 = X3_coef[mfi].array();
        amrex::Array4<Complex> const& X4 = X4_coef[mfi].array();
        amrex::Array4<Complex> const& Theta2 = Theta2_coef[mfi].array();
        amrex::Array4<Complex> const& T_rho = T_rho_coef[mfi].array();

        // Extract real (for portability on GPU)
        const amrex::Real vz = m_v_galilean[2];

        auto const & kr_modes = f.getKrArray(mfi);
        amrex::Real const* kr_arr = kr_modes.dataPtr();
        int const nr = bx.length(0);
        amrex::Real const dt = m_dt;

        // Loop over indices within one box
        int const modes = f.n_rz_azimuthal_modes;
        amrex::ParallelFor(bx, modes,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
        {
            constexpr amrex::Real c = PhysConst::c;
            constexpr amrex::Real ep0 = PhysConst::ep0;
            Complex const I = Complex{0._rt,1._rt};

            // Calculate norm of vector
            int const ir = i + nr*mode;
            amrex::Real const kr = kr_arr[ir];
            amrex::Real const kz = modified_kz[j];
            amrex::Real const k_norm = std::sqrt(kr*kr + kz*kz);

            // Calculate coefficients
            if (k_norm != 0._rt){

                C(i,j,k,mode) = std::cos(c*k_norm*dt);
                S_ck(i,j,k,mode) = std::sin(c*k_norm*dt)/(c*k_norm);

                // Calculate dot product with galilean velocity
                amrex::Real const kv = kz*vz;

                amrex::Real const nu = kv/(k_norm*c);
                Complex const theta = amrex::exp( 0.5_rt*I*kv*dt );
                Complex const theta_star = amrex::exp( -0.5_rt*I*kv*dt );
                Complex const e_theta = amrex::exp( I*c*k_norm*dt );

                Theta2(i,j,k,mode) = theta*theta;

                if (kz == 0._rt) {
                    T_rho(i,j,k,mode) = -dt;
                } else {
                    T_rho(i,j,k,mode) = (1._rt - theta*theta)/(I*kz*vz);
                }

                if ( (nu != 1._rt) && (nu != 0._rt) ) {

                    // Note: the coefficients X1, X2, X do not correspond
                    // exactly to the original Galilean paper, but the
                    // update equation have been modified accordingly so that
                    // the expressions below (with the update equations)
                    // are mathematically equivalent to those of the paper.
                    const Complex x1 = 1._rt/(1._rt-nu*nu) *
                        (theta_star - C(i,j,k,mode)*theta + I*kv*S_ck(i,j,k,mode)*theta);
                    // x1, above, is identical to the original paper
                    X1(i,j,k,mode) = theta*x1/(ep0*c*c*k_norm*k_norm);
                    // The difference betwen X2 and X3 below, and those
                    // from the original paper is the factor ep0*k_norm*k_norm
                    X2(i,j,k,mode) = (x1 - theta*(1._rt - C(i,j,k,mode)))
                                     /(theta_star-theta)/(ep0*k_norm*k_norm);
                    X3(i,j,k,mode) = (x1 - theta_star*(1._rt - C(i,j,k,mode)))
                                     /(theta_star-theta)/(ep0*k_norm*k_norm);
                    X4(i,j,k,mode) = I*kv*X1(i,j,k,mode) - theta*theta*S_ck(i,j,k,mode)/ep0;

                } else if (nu == 0._rt) {

                    X1(i,j,k,mode) = (1._rt - C(i,j,k,mode))/(ep0 * c*c * k_norm*k_norm);
                    X2(i,j,k,mode) = (1._rt - S_ck(i,j,k,mode)/dt)/(ep0 * k_norm*k_norm);
                    X3(i,j,k,mode) = (C(i,j,k,mode) - S_ck(i,j,k,mode)/dt)/(ep0 * k_norm*k_norm);
                    X4(i,j,k,mode) = -S_ck(i,j,k,mode)/ep0;

                } else if ( nu == 1._rt) {
                    X1(i,j,k,mode) = (1._rt - e_theta*e_theta + 2._rt*I*c*k_norm*dt) / (4._rt*c*c*ep0*k_norm*k_norm);
                    X2(i,j,k,mode) = (3._rt - 4._rt*e_theta + e_theta*e_theta + 2._rt*I*c*k_norm*dt) / (4._rt*ep0*k_norm*k_norm*(1._rt - e_theta));
                    X3(i,j,k,mode) = (3._rt - 2._rt/e_theta - 2._rt*e_theta + e_theta*e_theta - 2._rt*I*c*k_norm*dt) / (4._rt*ep0*(e_theta - 1._rt)*k_norm*k_norm);
                    X4(i,j,k,mode) = I*(-1._rt + e_theta*e_theta + 2._rt*I*c*k_norm*dt) / (4._rt*ep0*c*k_norm);
                }

            } else { // Handle k_norm = 0, by using the analytical limit
                C(i,j,k,mode) = 1._rt;
                S_ck(i,j,k,mode) = dt;
                X1(i,j,k,mode) = 0.5_rt * dt*dt / ep0;
                X2(i,j,k,mode) = c*c * dt*dt / (6._rt*ep0);
                X3(i,j,k,mode) = - c*c * dt*dt / (3._rt*ep0);
                X4(i,j,k,mode) = -dt/ep0;
                Theta2(i,j,k,mode) = 1._rt;
            }
        });
     }
}

void
PsatdAlgorithmGalileanRZ::CurrentCorrection (SpectralFieldDataRZ& field_data)
{
    // Profiling
    WARPX_PROFILE( "PsatdAlgorithmGalileanRZ::CurrentCorrection" );

    const SpectralFieldIndex& Idx = m_spectral_index;

    // Loop over boxes
    for (amrex::MFIter mfi(field_data.fields); mfi.isValid(); ++mfi){

        amrex::Box const & bx = field_data.fields[mfi].box();

        // Extract arrays for the fields to be updated
        const amrex::Array4<Complex> fields = field_data.fields[mfi].array();

        // Extract pointers for the k vectors
        auto const & kr_modes = field_data.getKrArray(mfi);
        amrex::Real const* kr_arr = kr_modes.dataPtr();
        amrex::Real const* modified_kz_arr = modified_kz_vec[mfi].dataPtr();
        int const nr = bx.length(0);

        // Local copy of member variables before GPU loop
        const amrex::Real vz = m_v_galilean[2];
        const amrex::Real dt = m_dt;

        // Loop over indices within one box
        int const modes = field_data.n_rz_azimuthal_modes;
        ParallelFor(bx, modes,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
        {
            // All of the fields of each mode are grouped together
            auto const Jp_m = Idx.Jx_mid + Idx.n_fields*mode;
            auto const Jm_m = Idx.Jy_mid + Idx.n_fields*mode;
            auto const Jz_m = Idx.Jz_mid + Idx.n_fields*mode;
            auto const rho_old_m = Idx.rho_old + Idx.n_fields*mode;
            auto const rho_new_m = Idx.rho_new + Idx.n_fields*mode;

            // Shortcuts for the values of J and rho
            Complex const Jp = fields(i,j,k,Jp_m);
            Complex const Jm = fields(i,j,k,Jm_m);
            Complex const Jz = fields(i,j,k,Jz_m);
            Complex const rho_old = fields(i,j,k,rho_old_m);
            Complex const rho_new = fields(i,j,k,rho_new_m);

            // k vector values, and coefficients
            // The k values for each mode are grouped together
            int const ir = i + nr*mode;
            amrex::Real const kr = kr_arr[ir];
            amrex::Real const kz = modified_kz_arr[j];
            amrex::Real const k_norm2 = kr*kr + kz*kz;

            constexpr Complex I = Complex{0._rt,1._rt};

            // Correct J
            if ( k_norm2 != 0._rt )
            {
                Complex const theta2 = amrex::exp(I*kz*vz*dt);
                Complex const inv_1_T2 = 1._rt/(kz*vz == 0._rt ?  1._rt : 1._rt - theta2);
                Complex const j_corr_coef = (kz == 0._rt ? 1._rt/dt : -I*kz*vz*inv_1_T2);
                Complex const F = - (j_corr_coef*(rho_new - rho_old*theta2) + I*kz*Jz + kr*(Jp - Jm))/k_norm2;

                fields(i,j,k,Jp_m) += +0.5_rt*kr*F;
                fields(i,j,k,Jm_m) += -0.5_rt*kr*F;
                fields(i,j,k,Jz_m) += -I*kz*F;
            }
        });
    }
}

void
PsatdAlgorithmGalileanRZ::VayDeposition (SpectralFieldDataRZ& /*field_data*/)
{
    WARPX_ABORT_WITH_MESSAGE(
        "Vay deposition not implemented in RZ geometry");
}
