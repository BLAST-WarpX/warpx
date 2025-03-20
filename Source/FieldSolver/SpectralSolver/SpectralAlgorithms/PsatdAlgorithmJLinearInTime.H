/* Copyright 2019
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PSATD_ALGORITHM_J_LINEAR_IN_TIME_H_
#define WARPX_PSATD_ALGORITHM_J_LINEAR_IN_TIME_H_

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
/*
 * \brief Class that updates the fields in spectral space according to the multi-J algorithm
 *        and stores the coefficients of the corresponding update equations. J is assumed to be
 *        linear in time and two currents, deposited at the beginning and the end of the time step,
 *        are used for the PSATD update equations, instead of only one current deposited at half time.
 */
class PsatdAlgorithmJLinearInTime : public SpectralBaseAlgorithm
{
    public:

        /**
         * \brief Constructor of the class PsatdAlgorithmJLinearInTime
         *
         * \param[in] spectral_kspace spectral space
         * \param[in] dm distribution mapping
         * \param[in] spectral_index object containing indices to access data in spectral space
         * \param[in] norder_x order of the spectral solver along x
         * \param[in] norder_y order of the spectral solver along y
         * \param[in] norder_z order of the spectral solver along z
         * \param[in] grid_type type of grid (collocated or not)
         * \param[in] dt time step of the simulation
         * \param[in] time_averaging whether to use time averaging for large time steps
         * \param[in] dive_cleaning Update F as part of the field update, so that errors in divE=rho propagate away at the speed of light
         * \param[in] divb_cleaning Update G as part of the field update, so that errors in divB=0 propagate away at the speed of light
         */
        PsatdAlgorithmJLinearInTime (
            const SpectralKSpace& spectral_kspace,
            const amrex::DistributionMapping& dm,
            const SpectralFieldIndex& spectral_index,
            int norder_x,
            int norder_y,
            int norder_z,
            ablastr::utils::enums::GridType grid_type,
            amrex::Real dt,
            bool time_averaging,
            bool dive_cleaning,
            bool divb_cleaning
        );

        /**
         * \brief Updates the E and B fields in spectral space, according to the multi-J PSATD equations
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
         * \brief Initialize additional coefficients used in \c pushSpectralFields to update E,B,
         *        required only when using time averaging with the assumption that J is linear in time
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
        SpectralRealCoefficients X1_coef, X2_coef, X3_coef, X5_coef, X6_coef;

        // Other member variables
        amrex::Real m_dt;
        bool m_time_averaging;
        bool m_dive_cleaning;
        bool m_divb_cleaning;
};
#endif // WARPX_USE_FFT
#endif // WARPX_PSATD_ALGORITHM_J_LINEAR_IN_TIME_H_
