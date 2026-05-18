/* Copyright 2019-2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPECTRAL_SOLVER_RZ_H_
#define WARPX_SPECTRAL_SOLVER_RZ_H_

#include "SpectralSolverRZ_fwd.H"

#include "SpectralAlgorithms/SpectralBaseAlgorithmRZ.H"
#include "SpectralFieldDataRZ.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/utils/Enums.H>


/* \brief Top-level class for the electromagnetic spectral solver
 *
 * Stores the field in spectral space, and has member functions
 * to Fourier-transform the fields between real space and spectral space
 * and to update fields in spectral space over one time step.
 */
class SpectralSolverRZ
{
    public:
        // Inline definition of the member functions of `SpectralSolverRZ`,
        // except the constructor (see `SpectralSolverRZ.cpp`)
        // The body of these functions is short, since the work is done in the
        // underlying classes `SpectralFieldData` and `PsatdAlgorithm`

        // Constructor
        SpectralSolverRZ (int lev,
                          amrex::BoxArray const & realspace_ba,
                          amrex::DistributionMapping const & dm,
                          int n_rz_azimuthal_modes,
                          int norder_z,
                          ablastr::utils::enums::GridType grid_type,
                          const amrex::Vector<amrex::Real>& v_galilean,
                          amrex::RealVect dx, amrex::Real dt,
                          bool with_pml,
                          bool update_with_rho,
                          bool fft_do_time_averaging,
                          TimeDependencyJ time_dependency_J,
                          TimeDependencyRho time_dependency_rho,
                          bool dive_cleaning,
                          bool divb_cleaning);

        /* \brief Transform the component `i_comp` of MultiFab `field_mf`
         *  to spectral space, and store the corresponding result internally
         *  (in the spectral field specified by `field_index`) */
        void ForwardTransform (int lev, amrex::MultiFab const & field_mf, int field_index,
                               int i_comp=0);

        /* \brief Transform the two MultiFabs `field_mf1` and `field_mf2`
         *  to spectral space, and store the corresponding results internally
         *  (in the spectral field specified by `field_index1` and `field_index2`) */
        void ForwardTransform (int lev, amrex::MultiFab const & field_mf1, int field_index1,
                               amrex::MultiFab const & field_mf2, int field_index2);

        /* \brief Transform spectral field specified by `field_index` back to
         * real space, and store it in the component `i_comp` of `field_mf` */
        void BackwardTransform (int lev, amrex::MultiFab& field_mf, int field_index,
                                int i_comp=0);

        /* \brief Transform spectral fields specified by `field_index1` and `field_index2`
         * back to real space, and store it in `field_mf1` and `field_mf2`*/
        void BackwardTransform (int lev, amrex::MultiFab& field_mf1, int field_index1,
                                amrex::MultiFab& field_mf2, int field_index2);

        /* \brief Update the fields in spectral space, over one timestep */
        void pushSpectralFields (bool doing_pml=false);

        /* \brief Initialize K space filtering arrays */
        void InitFilter (amrex::IntVect const & filter_npass_each_dir,
                         bool compensation);

        /* \brief Apply K space filtering for a scalar */
        void ApplyFilter (int lev, int field_index);

        /* \brief Apply K space filtering for a vector */
        void ApplyFilter (int lev, int field_index1,
                          int field_index2, int field_index3);

        /**
          * \brief Public interface to call the member function ComputeSpectralDivE
          * of the base class SpectralBaseAlgorithmRZ from objects of class SpectralSolverRZ
          */
        void ComputeSpectralDivE (int lev,
                                  ablastr::fields::VectorField const & Efield,
                                  amrex::MultiFab& divE);

        /**
         * \brief Public interface to call the virtual function \c CurrentCorrection,
         * defined in the base class SpectralBaseAlgorithmRZ and possibly overridden
         * by its derived classes (e.g. PsatdAlgorithmRZ), from
         * objects of class SpectralSolverRZ through the private unique pointer \c algorithm
         */
        void CurrentCorrection ();

        /**
         * \brief Public interface to call the virtual function \c VayDeposition,
         * declared in the base class SpectralBaseAlgorithmRZ and defined in its
         * derived classes, from objects of class SpectralSolverRZ through the private
         * unique pointer \c algorithm.
         */
        void VayDeposition ();

        /**
         * \brief Copy spectral data from component \c src_comp to component \c dest_comp
         *        of \c field_data.fields
         *
         * \param[in] src_comp  component of the source FabArray from which the data are copied
         * \param[in] dest_comp component of the destination FabArray where the data are copied
         */
        void CopySpectralDataComp (int src_comp, int dest_comp);

        /**
         * \brief Set to zero the data on component \c icomp of \c field_data.fields
         *
         * \param[in] icomp component of the FabArray where the data are set to zero
         */
        void ZeroOutDataComp (int icomp);

        /**
         * \brief Scale the data on component \c icomp of \c field_data.fields
         *        by a given scale factor
         *
         * \param[in] icomp component of the FabArray where the data are scaled
         * \param[in] scale_factor scale factor to use for scaling
         */
        void ScaleDataComp (int icomp, amrex::Real scale_factor);

        SpectralFieldIndex m_spectral_index;

        // Solve time step size
        amrex::Real m_dt;

    private:

        SpectralKSpaceRZ k_space; // Save the instance to initialize filtering
        SpectralFieldDataRZ field_data; // Store field in spectral space
                                        // and perform the Fourier transforms
        std::unique_ptr<SpectralBaseAlgorithmRZ> algorithm;
        std::unique_ptr<SpectralBaseAlgorithmRZ> PML_algorithm;
        // Defines field update equation in spectral space,
        // and the associated coefficients.
        // SpectralBaseAlgorithmRZ is a base class ; this pointer is meant
        // to point an instance of a *sub-class* defining a specific algorithm

};

#endif // WARPX_SPECTRAL_SOLVER_RZ_H_
