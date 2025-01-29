/* Copyright 2019 Axel Huebl, Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PSATD_ALGORITHM_PML_H_
#define WARPX_PSATD_ALGORITHM_PML_H_

#include "SpectralBaseAlgorithm.H"

#include "FieldSolver/SpectralSolver/SpectralFieldData_fwd.H"
#include "FieldSolver/SpectralSolver/SpectralKSpace_fwd.H"

#include <ablastr/utils/Enums.H>

#include <AMReX_REAL.H>
#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

#if WARPX_USE_FFT

/*
 * \brief Class that updates the field in spectral space
 * and stores the coefficients of the corresponding update equation.
 */
class PsatdAlgorithmPml : public SpectralBaseAlgorithm
{
    public:

        /**
         * \brief Constructor of the class PsatdAlgorithmPml
         *
         * \param[in] spectral_kspace Spectral space
         * \param[in] dm Distribution mapping
         * \param[in] spectral_index Object containing indices to access data in spectral space
         * \param[in] norder_x Order of the spectral solver along x
         * \param[in] norder_y Order of the spectral solver along y
         * \param[in] norder_z Order of the spectral solver along z
         * \param[in] grid_type Type of grid (collocated or not)
         * \param[in] v_galilean Galilean velocity
         * \param[in] dt Time step of the simulation
         * \param[in] dive_cleaning Whether to use divergence correction for E (F term)
         * \param[in] divb_cleaning Whether to use divergence correction for B (G term)
         */
        PsatdAlgorithmPml(
            const SpectralKSpace& spectral_kspace,
            const amrex::DistributionMapping& dm,
            const SpectralFieldIndex& spectral_index,
            int norder_x,
            int norder_y,
            int norder_z,
            ablastr::utils::enums::GridType grid_type,
            const amrex::Vector<amrex::Real>& v_galilean,
            amrex::Real dt,
            bool dive_cleaning,
            bool divb_cleaning);

        /**
         * \brief Initializes the coefficients used in \c pushSpectralFields
         *        to update the E and B fields
         *
         * \param[in] spectral_kspace Spectral space
         * \param[in] dm Distribution mapping
         */
        void InitializeSpectralCoefficients(
            const SpectralKSpace& spectral_kspace,
            const amrex::DistributionMapping& dm);

        /**
         * \brief Updates the E and B fields in spectral space,
         *        according to the relevant PSATD equations
         *
         * \param[in,out] f All fields in spectral space
         */
        void pushSpectralFields(SpectralFieldData& f) const final;

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

        SpectralRealCoefficients C_coef, S_ck_coef, inv_k2_coef;
        SpectralComplexCoefficients T2_coef;
        // Centered modified finite-order k vectors
        KVectorComponent modified_kx_vec_centered;
#if defined(WARPX_DIM_3D)
        KVectorComponent modified_ky_vec_centered;
#endif
        KVectorComponent modified_kz_vec_centered;
        amrex::Vector<amrex::Real> m_v_galilean;
        amrex::Real m_dt;
        bool m_dive_cleaning;
        bool m_divb_cleaning;
        bool m_is_galilean;
};

#endif // WARPX_USE_FFT
#endif // WARPX_PSATD_ALGORITHM_PML_H_
