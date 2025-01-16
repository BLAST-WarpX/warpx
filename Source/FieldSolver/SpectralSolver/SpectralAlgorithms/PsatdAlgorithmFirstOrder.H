/* Copyright 2019
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PSATD_ALGORITHM_FIRST_ORDER_H_
#define WARPX_PSATD_ALGORITHM_FIRST_ORDER_H_

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
 * \brief Class that updates the fields in spectral space according to the first-order PSATD equations.
 */
class PsatdAlgorithmFirstOrder : public SpectralBaseAlgorithm
{
    public:

        /**
         * \brief Constructor of the class PsatdAlgorithmFirstOrder
         *
         * \param[in] spectral_kspace spectral space
         * \param[in] dm distribution mapping
         * \param[in] spectral_index object containing indices to access data in spectral space
         * \param[in] norder_x order of the spectral solver along x
         * \param[in] norder_y order of the spectral solver along y
         * \param[in] norder_z order of the spectral solver along z
         * \param[in] grid_type type of grid (collocated or not)
         * \param[in] dt time step of the simulation
         * \param[in] div_cleaning whether to use divergence correction for both E and B (thus, F and G)
         * \param[in] J_in_time time dependency of J (currently supported: constant, linear)
         * \param[in] rho_in_time time dependency of rho (currently supported: constant, linear)
         */
        PsatdAlgorithmFirstOrder (
            const SpectralKSpace& spectral_kspace,
            const amrex::DistributionMapping& dm,
            const SpectralFieldIndex& spectral_index,
            int norder_x,
            int norder_y,
            int norder_z,
            ablastr::utils::enums::GridType grid_type,
            amrex::Real dt,
            bool div_cleaning,
            JInTime J_in_time,
            RhoInTime rho_in_time);

        /**
         * \brief Updates E, B, F, and G fields in spectral space,
         *        according to the first-order PSATD equations
         *
         * \param[in,out] f all the fields in spectral space
         */
        void pushSpectralFields (SpectralFieldData& f) const final;

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

        // Other member variables
        amrex::Real m_dt;
        bool m_div_cleaning;
        JInTime m_J_in_time;
        RhoInTime m_rho_in_time;
};
#endif // WARPX_USE_FFT
#endif // WARPX_PSATD_ALGORITHM_FIRST_ORDER_H_
