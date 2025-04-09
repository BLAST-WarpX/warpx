/* Copyright 2019
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PSATD_ALGORITHM_J_CONSTANT_IN_TIME_H_
#define WARPX_PSATD_ALGORITHM_J_CONSTANT_IN_TIME_H_

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
/* \brief Class that updates the field in spectral space
 * and stores the coefficients of the corresponding update equation.
 */
class PsatdAlgorithmJConstantInTime : public SpectralBaseAlgorithm
{
    public:

        /**
         * \brief Constructor of the class PsatdAlgorithmJConstantInTime
         *
         * \param[in] spectral_kspace spectral space
         * \param[in] dm distribution mapping
         * \param[in] spectral_index object containing indices to access data in spectral space
         * \param[in] norder_x order of the spectral solver along x
         * \param[in] norder_y order of the spectral solver along y
         * \param[in] norder_z order of the spectral solver along z
         * \param[in] grid_type type of grid (collocated or not)
         * \param[in] v_galilean Galilean velocity (three-dimensional array)
         * \param[in] dt time step of the simulation
         * \param[in] update_with_rho whether the update equation for E uses rho or not
         * \param[in] time_averaging whether to use time averaging for large time steps
         * \param[in] dive_cleaning Update F as part of the field update, so that errors in divE=rho propagate away at the speed of light
         * \param[in] divb_cleaning Update G as part of the field update, so that errors in divB=0 propagate away at the speed of light
         */
        PsatdAlgorithmJConstantInTime (
            const SpectralKSpace& spectral_kspace,
            const amrex::DistributionMapping& dm,
            const SpectralFieldIndex& spectral_index,
            int norder_x,
            int norder_y,
            int norder_z,
            ablastr::utils::enums::GridType grid_type,
            const amrex::Vector<amrex::Real>& v_galilean,
            amrex::Real dt,
            bool update_with_rho,
            bool time_averaging,
            bool dive_cleaning,
            bool divb_cleaning);

        /**
         * \brief Updates the E and B fields in spectral space, according to the relevant PSATD equations
         *
         * \param[in,out] f all the fields in spectral space
         */
        void pushSpectralFields (SpectralFieldData& f) const final;

        /**
         * \brief Initializes the coefficients used in \c pushSpectralFields to update the E and B fields
         *
         * \param[in] spectral_kspace spectral space
         * \param[in] dm distribution mapping
         * \param[in] dt time step of the simulation
         */
        void InitializeSpectralCoefficients (
            const SpectralKSpace& spectral_kspace,
            const amrex::DistributionMapping& dm,
            amrex::Real dt);

        /**
         * \brief Initializes additional coefficients used in \c pushSpectralFields to update the E and B fields,
         *        required only when using time averaging with large time steps
         *
         * \param[in] spectral_kspace spectral space
         * \param[in] dm distribution mapping
         * \param[in] dt time step of the simulation
         */
        void InitializeSpectralCoefficientsAveraging (
            const SpectralKSpace& spectral_kspace,
            const amrex::DistributionMapping& dm,
            amrex::Real dt);

        /**
         * \brief Virtual function for current correction in Fourier space
         * (<a href="https://doi.org/10.1016/j.jcp.2013.03.010"> Vay et al, 2013</a>).
         * This function overrides the virtual function \c CurrentCorrection in the
         * base class \c SpectralBaseAlgorithm and cannot be overridden by further
         * derived classes.
         *
         * \param[in,out] field_data All fields in Fourier space
         */
        void CurrentCorrection (SpectralFieldData& field_data) final;

        /**
         * \brief Virtual function for Vay current deposition in Fourier space
         * (<a href="https://doi.org/10.1016/j.jcp.2013.03.010"> Vay et al, 2013</a>).
         * This function overrides the virtual function \c VayDeposition in the
         * base class \c SpectralBaseAlgorithm and cannot be overridden by further
         * derived classes.
         *
         * \param[in,out] field_data All fields in Fourier space
         */
        void VayDeposition (SpectralFieldData& field_data) final;

    private:

        // These real and complex coefficients are always allocated
        SpectralRealCoefficients C_coef, S_ck_coef;
        SpectralComplexCoefficients T2_coef, X1_coef, X2_coef, X3_coef, X4_coef;

        // These real and complex coefficients are allocated only with averaged Galilean PSATD
        SpectralComplexCoefficients Psi1_coef, Psi2_coef, Y1_coef, Y2_coef, Y3_coef, Y4_coef;

        // Centered modified finite-order k vectors
        KVectorComponent modified_kx_vec_centered;
#if defined(WARPX_DIM_3D)
        KVectorComponent modified_ky_vec_centered;
#endif
        KVectorComponent modified_kz_vec_centered;

        // Other member variables
        amrex::Vector<amrex::Real> m_v_galilean;
        amrex::Real m_dt;
        bool m_update_with_rho;
        bool m_time_averaging;
        bool m_dive_cleaning;
        bool m_divb_cleaning;
        bool m_is_galilean;
};
#endif // WARPX_USE_FFT
#endif // WARPX_PSATD_ALGORITHM_J_CONSTANT_IN_TIME_H_
