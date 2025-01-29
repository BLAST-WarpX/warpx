#ifndef WARPX_PSATD_ALGORITHM_COMOVING_H_
#define WARPX_PSATD_ALGORITHM_COMOVING_H_

#include "FieldSolver/SpectralSolver/SpectralFieldData.H"
#include "FieldSolver/SpectralSolver/SpectralKSpace.H"
#include "SpectralBaseAlgorithm.H"

#include <ablastr/utils/Enums.H>

#include <AMReX_Array.H>
#include <AMReX_Config.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

#if WARPX_USE_FFT

/* \brief Class that updates the field in spectral space and stores the coefficients
 * of the corresponding update equation, according to the comoving spectral scheme.
 */
class PsatdAlgorithmComoving : public SpectralBaseAlgorithm
{
    public:

        /**
         * \brief Class constructor
         */
        PsatdAlgorithmComoving (const SpectralKSpace& spectral_kspace,
                                const amrex::DistributionMapping& dm,
                                const SpectralFieldIndex& spectral_index,
                                int norder_x,
                                int norder_y,
                                int norder_z,
                                ablastr::utils::enums::GridType grid_type,
                                const amrex::Vector<amrex::Real>& v_comoving,
                                amrex::Real dt,
                                bool update_with_rho);

        /**
         * \brief Override the update equations in Fourier space
         */
        void pushSpectralFields (SpectralFieldData& f) const final;

        /* \brief Initialize the coefficients needed in the update equations
         */
        void InitializeSpectralCoefficients (const SpectralKSpace& spectral_kspace,
                                             const amrex::DistributionMapping& dm,
                                             amrex::Real dt);

        /**
         * \brief Virtual function for current correction in Fourier space.
         * This function overrides the virtual function \c CurrentCorrection in the
         * base class \c SpectralBaseAlgorithm and cannot be overridden by further
         * derived classes.
         *
         * \param[in,out] field_data All fields in Fourier space
         */
        void CurrentCorrection (SpectralFieldData& field_data) final;

        /**
         * \brief Virtual function for Vay current deposition in Fourier space.
         * This function overrides the virtual function \c VayDeposition in the
         * base class \c SpectralBaseAlgorithm and cannot be overridden by further
         * derived classes.
         *
         * \param[in,out] field_data All fields in Fourier space
         */
        void VayDeposition (SpectralFieldData& field_data) final;

    private:

        // Real and complex spectral coefficients
        SpectralRealCoefficients C_coef, S_ck_coef;
        SpectralComplexCoefficients Theta2_coef, X1_coef, X2_coef, X3_coef, X4_coef;

        // k vectors
        KVectorComponent kx_vec;
#if defined(WARPX_DIM_3D)
        KVectorComponent ky_vec;
#endif
        KVectorComponent kz_vec;

        // Additional member variables
        amrex::Vector<amrex::Real> m_v_comoving;
        amrex::Real m_dt;
};

#endif // WARPX_USE_FFT
#endif // WARPX_PSATD_ALGORITHM_COMOVING_H_
