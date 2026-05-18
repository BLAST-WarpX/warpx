/* Copyright 2019 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPECTRAL_BASE_ALGORITHM_RZ_H_
#define WARPX_SPECTRAL_BASE_ALGORITHM_RZ_H_

#include "FieldSolver/SpectralSolver/SpectralKSpaceRZ.H"
#include "FieldSolver/SpectralSolver/SpectralFieldDataRZ.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/utils/Enums.H>


/* \brief Class that updates the field in spectral space
 * and stores the coefficients of the corresponding update equation.
 *
 * `SpectralBaseAlgorithmRZ` is only a base class and cannot be used directly.
 * Instead use its subclasses, which implement the specific field update
 * equations for a given spectral algorithm.
 */
class SpectralBaseAlgorithmRZ
{
    public:
        // Virtual member function ; meant to be overridden in subclasses
        virtual void pushSpectralFields(SpectralFieldDataRZ & f) = 0;

        // The destructor should also be a virtual function, so that
        // a pointer to subclass of `SpectraBaseAlgorithm` actually
        // calls the subclass's destructor.
        virtual ~SpectralBaseAlgorithmRZ() = default;

        /**
         * \brief Default Copy constructor
         */
        SpectralBaseAlgorithmRZ ( SpectralBaseAlgorithmRZ const &) = default;

        /**
         * \brief Default Copy operator
         */
        SpectralBaseAlgorithmRZ& operator= ( SpectralBaseAlgorithmRZ const & ) = default;


        /**
         * \brief Default Move constructor
         */
        SpectralBaseAlgorithmRZ ( SpectralBaseAlgorithmRZ&& ) = default;

        /**
         * \brief Default Move operator
         */
        SpectralBaseAlgorithmRZ& operator= ( SpectralBaseAlgorithmRZ&& ) = default;

        /**
         * \brief Virtual function for current correction in Fourier space
         * (<a href="https://doi.org/10.1016/j.jcp.2013.03.010"> Vay et al, 2013</a>).
         * This virtual function is pure and must be defined in derived classes.
         *
         * \param[in,out] field_data All fields in Fourier space
         */
        virtual void CurrentCorrection (SpectralFieldDataRZ& field_data) = 0;

        /**
         * \brief Compute spectral divergence of E
         */
        void ComputeSpectralDivE ( int lev,
                                   SpectralFieldDataRZ& field_data,
                                   ablastr::fields::VectorField const & Efield,
                                   amrex::MultiFab& divE );

        /**
         * \brief Virtual function for Vay current deposition in Fourier space
         * (<a href="https://doi.org/10.1016/j.jcp.2013.03.010"> Vay et al, 2013</a>).
         * This virtual function is pure and must be defined in derived classes.
         *
         * \param[in,out] field_data All fields in Fourier space
         */
        virtual void VayDeposition (SpectralFieldDataRZ& field_data) = 0;

    protected: // Meant to be used in the subclasses

        using SpectralRealCoefficients = amrex::FabArray< amrex::BaseFab <amrex::Real> >;
        using SpectralComplexCoefficients = amrex::FabArray< amrex::BaseFab <Complex> >;

        // Constructor
        SpectralBaseAlgorithmRZ(SpectralKSpaceRZ const & spectral_kspace,
                                amrex::DistributionMapping const & dm,
                                const SpectralFieldIndex& spectral_index,
                                int const norder_z,
                                ablastr::utils::enums::GridType grid_type)
          // Compute and assign the modified k vectors
          : m_spectral_index(spectral_index),
            modified_kz_vec(spectral_kspace.getModifiedKComponent(dm, 1, norder_z, grid_type))
        {}

        SpectralFieldIndex m_spectral_index;

        // Modified finite-order vectors
        KVectorComponent modified_kz_vec;
};

#endif // WARPX_SPECTRAL_BASE_ALGORITHM_RZ_H_
