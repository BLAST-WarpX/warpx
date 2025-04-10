/* Copyright 2019-2020 Maxence Thevenet, Remi Lehe, Edoardo Zoni
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPECTRAL_SOLVER_H_
#define WARPX_SPECTRAL_SOLVER_H_

#include "SpectralSolver_fwd.H"

#include "SpectralAlgorithms/SpectralBaseAlgorithm.H"
#include "SpectralFieldData.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/utils/Enums.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

#ifdef WARPX_USE_FFT
/**
 * \brief Top-level class for the electromagnetic spectral solver
 *
 * Stores the field in spectral space, and has member functions
 * to Fourier-transform the fields between real space and spectral space
 * and to update fields in spectral space over one time step.
 */
class SpectralSolver
{
    public:

        /**
         * \brief Constructor of the class SpectralSolver
         *
         * Select the spectral algorithm to be used, allocate the corresponding coefficients
         * for the discrete field update equations, and prepare the structures that store
         * the fields in spectral space.
         *
         * \param[in] lev mesh refinement level
         * \param[in] realspace_ba BoxArray in real space
         * \param[in] dm DistributionMapping for the given BoxArray
         * \param[in] norder_x spectral order along x
         * \param[in] norder_y spectral order along y
         * \param[in] norder_z spectral order along z
         * \param[in] grid_type type of grid (collocated or not)
         * \param[in] v_galilean three-dimensional vector containing the components of the Galilean
         *                       velocity for the standard or averaged Galilean PSATD solvers
         * \param[in] v_comoving three-dimensional vector containing the components of the comoving
         *                       velocity for the comoving PSATD solver
         * \param[in] dx AMREX_SPACEDIM-dimensional vector containing the cell sizes along each direction
         * \param[in] dt time step for the analytical integration of Maxwell's equations
         * \param[in] pml whether the boxes in the given BoxArray are PML boxes
         * \param[in] periodic_single_box whether there is only one periodic single box
         *                                (no domain decomposition)
         * \param[in] update_with_rho whether rho is used in the field update equations
         * \param[in] fft_do_time_averaging whether the time averaging algorithm is used
         * \param[in] psatd_solution_type whether the PSATD equations are derived
         *                                from a first-order or second-order model
         * \param[in] J_in_time integer that corresponds to the time dependency of J
         *                      (constant, linear) for the PSATD algorithm
         * \param[in] rho_in_time integer that corresponds to the time dependency of rho
         *                        (linear, quadratic) for the PSATD algorithm
         * \param[in] dive_cleaning whether to use div(E) cleaning to account for errors in
         *                          Gauss law (new field F in the update equations)
         * \param[in] divb_cleaning whether to use div(B) cleaning to account for errors in
         *                          div(B) = 0 law (new field G in the update equations)
         */
        SpectralSolver (int lev,
                        const amrex::BoxArray& realspace_ba,
                        const amrex::DistributionMapping& dm,
                        int norder_x,
                        int norder_y,
                        int norder_z,
                        ablastr::utils::enums::GridType grid_type,
                        const amrex::Vector<amrex::Real>& v_galilean,
                        const amrex::Vector<amrex::Real>& v_comoving,
                        amrex::RealVect dx,
                        amrex::Real dt,
                        bool pml,
                        bool periodic_single_box,
                        bool update_with_rho,
                        bool fft_do_time_averaging,
                        PSATDSolutionType psatd_solution_type,
                        JInTime J_in_time,
                        RhoInTime rho_in_time,
                        bool dive_cleaning,
                        bool divb_cleaning);

        /**
         * \brief Transform the component i_comp of the MultiFab mf to Fourier space,
         * and store the result internally (in the spectral field specified by field_index)
         *
         * \param[in] lev mesh refinement level
         * \param[in] mf MultiFab that is transformed to Fourier space (component i_comp)
         * \param[in] field_index index of the spectral field that stores the FFT result
         * \param[in] i_comp component of the MultiFab mf that is transformed to Fourier space
         */
        void ForwardTransform (int lev,
                               const amrex::MultiFab& mf,
                               int field_index,
                               int i_comp = 0);

        /**
         * \brief Transform spectral field specified by `field_index` back to
         * real space, and store it in the component `i_comp` of `mf`
         */
        void BackwardTransform( int lev,
                                amrex::MultiFab& mf,
                                int field_index,
                                const amrex::IntVect& fill_guards,
                                int i_comp=0 );

        /**
         * \brief Update the fields in spectral space, over one timestep
         */
        void pushSpectralFields();

        /**
          * \brief Public interface to call the member function ComputeSpectralDivE
          * of the base class SpectralBaseAlgorithm from objects of class SpectralSolver
          */
        void ComputeSpectralDivE (
            int lev,
            ablastr::fields::VectorField const & Efield,
            amrex::MultiFab& divE
        )
        {
            algorithm->ComputeSpectralDivE( lev, field_data, Efield, divE );
        }

        /**
         * \brief Public interface to call the virtual function \c CurrentCorrection,
         * defined in the base class SpectralBaseAlgorithm and possibly overridden
         * by its derived classes (e.g. PsatdAlgorithm, PsatdAlgorithmComoving, etc.), from
         * objects of class SpectralSolver through the private unique pointer \c algorithm
         */
        void CurrentCorrection ()
        {
            algorithm->CurrentCorrection(field_data);
        }

        /**
         * \brief Public interface to call the virtual function \c VayDeposition,
         * declared in the base class SpectralBaseAlgorithm and defined in its
         * derived classes, from objects of class SpectralSolver through the private
         * unique pointer \c algorithm.
         */
        void VayDeposition ()
        {
            algorithm->VayDeposition(field_data);
        }

        /**
         * \brief Copy spectral data from component \c src_comp to component \c dest_comp
         *        of \c field_data.fields.
         *
         * \param[in] src_comp  component of the source FabArray from which the data are copied
         * \param[in] dest_comp component of the destination FabArray where the data are copied
         */
        void CopySpectralDataComp (const int src_comp, const int dest_comp)
        {
            // The last two arguments represent the number of components and
            // the number of ghost cells to perform this operation
            Copy(field_data.fields, field_data.fields, src_comp, dest_comp, 1, 0);
        }

        /**
         * \brief Set to zero the data on component \c icomp of \c field_data.fields
         *
         * \param[in] icomp component of the FabArray where the data are set to zero
         */
        void ZeroOutDataComp (const int icomp)
        {
            // The last argument represents the number of components to perform this operation
            field_data.fields.setVal(0., icomp, 1);
        }

        /**
         * \brief Scale the data on component \c icomp of \c field_data.fields
         *        by a given scale factor
         *
         * \param[in] icomp component of the FabArray where the data are scaled
         * \param[in] scale_factor scale factor to use for scaling
         */
        void ScaleDataComp (const int icomp, const amrex::Real scale_factor)
        {
            // The last argument represents the number of components to perform this operation
            field_data.fields.mult(scale_factor, icomp, 1);
        }

        SpectralFieldIndex m_spectral_index;

        // Solve time step size
        amrex::Real m_dt;

    protected:

        amrex::IntVect m_fill_guards;

    private:

        void ReadParameters ();

        // Store field in spectral space and perform the Fourier transforms
        SpectralFieldData field_data;

        // Defines field update equation in spectral space and the associated coefficients.
        // SpectralBaseAlgorithm is a base class; this pointer is meant to point
        // to an instance of a sub-class defining a specific algorithm
        std::unique_ptr<SpectralBaseAlgorithm> algorithm;
};
#endif // WARPX_USE_FFT
#endif // WARPX_SPECTRAL_SOLVER_H_
