/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPECTRAL_FIELD_DATA_H_
#define WARPX_SPECTRAL_FIELD_DATA_H_

#include "SpectralFieldData_fwd.H"

#include "SpectralKSpace.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpX_Complex.H"

#include <ablastr/math/fft/AnyFFT.H>

#include <AMReX_BaseFab.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_FabArray.H>
#include <AMReX_IndexType.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <vector>

// Declare type for spectral fields
using SpectralField = amrex::FabArray< amrex::BaseFab <Complex> >;

class SpectralFieldIndex
{
    public:

        /**
         * \brief Constructor of the class SpectralFieldIndex
         *
         * Set integer indices to access data in spectral space
         * and total number of fields to be stored.
         *
         * \param[in] update_with_rho whether rho is used in the field update equations
         * \param[in] time_averaging  whether the time averaging algorithm is used
         * \param[in] J_in_time       the multi-J algorithm used (hence two currents
         *                            computed at the beginning and the end of the time interval
         *                            instead of one current computed at half time)
         * \param[in] rho_in_time     the multi-rho algorithm used (hence two densities
         *                            computed at the beginning and the end of the time interval
         *                            instead of one density computed at half time)
         * \param[in] dive_cleaning   whether to use div(E) cleaning to account for errors in
         *                            Gauss law (new field F in the update equations)
         * \param[in] divb_cleaning   whether to use div(B) cleaning to account for errors in
         *                            div(B) = 0 law (new field G in the update equations)
         * \param[in] pml             whether the indices are used to access spectral data
         *                            for the PML spectral solver
         * \param[in] pml_rz          whether the indices are used to access spectral data
         *                            for the RZ PML spectral solver
         */
        SpectralFieldIndex (bool update_with_rho,
                            bool time_averaging,
                            JInTime J_in_time,
                            RhoInTime rho_in_time,
                            bool dive_cleaning,
                            bool divb_cleaning,
                            bool pml,
                            bool pml_rz = false);

        /**
         * \brief Default constructor
         */
        SpectralFieldIndex () = default;

        /**
         * \brief Default destructor
         */
        ~SpectralFieldIndex () = default;

        /**
         * \brief Default Copy constructor
         */
        SpectralFieldIndex ( SpectralFieldIndex const &) = default;

        /**
         * \brief Default Copy operator
         */
        SpectralFieldIndex& operator= ( SpectralFieldIndex const & ) = default;

        /**
         * \brief Default Move constructor
         */
        SpectralFieldIndex ( SpectralFieldIndex&& ) = default;

        /**
         * \brief Default Move operator
         */
        SpectralFieldIndex& operator= ( SpectralFieldIndex&& ) = default;

    // Total number of fields that are actually allocated
    int n_fields;

    // Indices overwritten in the constructor, for the fields that are actually allocated
    // (index -1 will never be used, unless there is some bug in the code implementation,
    // which would result in a runtime crash due to out-of-bound accesses that can be detected
    // by running the code in DEBUG mode)

    // Always
    int Ex = -1, Ey = -1, Ez = -1;
    int Bx = -1, By = -1, Bz = -1;
    int divE = -1;

    // Time averaging
    int Ex_avg = -1, Ey_avg = -1, Ez_avg = -1;
    int Bx_avg = -1, By_avg = -1, Bz_avg = -1;

    // J
    int Jx_old = -1, Jy_old = -1, Jz_old = -1;
    int Jx_mid = -1, Jy_mid = -1, Jz_mid = -1;
    int Jx_new = -1, Jy_new = -1, Jz_new = -1;

    // rho
    int rho_old = -1, rho_mid = -1, rho_new = -1;

    // div(E) and div(B) cleaning
    int F = -1, G = -1;

    // PML
    int Exy = -1, Exz = -1, Eyx = -1, Eyz = -1, Ezx = -1, Ezy = -1;
    int Bxy = -1, Bxz = -1, Byx = -1, Byz = -1, Bzx = -1, Bzy = -1;

    // PML with div(E) and/or div(B) cleaning
    int Exx = -1, Eyy = -1, Ezz = -1, Bxx = -1, Byy = -1, Bzz = -1;
    int Fx  = -1, Fy  = -1, Fz  = -1, Gx  = -1, Gy  = -1, Gz  = -1;

    // PML RZ
    int Er_pml = -1, Et_pml = -1, Br_pml = -1, Bt_pml = -1;
};

/** \brief Class that stores the fields in spectral space, and performs the
 *  Fourier transforms between real space and spectral space
 */
class SpectralFieldData
{

    public:
        SpectralFieldData( int lev,
                           const amrex::BoxArray& realspace_ba,
                           const SpectralKSpace& k_space,
                           const amrex::DistributionMapping& dm,
                           int n_field_required,
                           bool periodic_single_box);
        SpectralFieldData() = default; // Default constructor
        ~SpectralFieldData();

        // default move and copy operations
        SpectralFieldData(const SpectralFieldData&) = delete;
        SpectralFieldData& operator=(const SpectralFieldData&) = delete;
        SpectralFieldData(SpectralFieldData&&) = default;
        SpectralFieldData& operator=(SpectralFieldData&& field_data) = default;

        void ForwardTransform (int lev,
                               const amrex::MultiFab& mf, int field_index,
                               int i_comp);

        void BackwardTransform (int lev, amrex::MultiFab& mf, int field_index,
                                const amrex::IntVect& fill_guards, int i_comp);

        // `fields` stores fields in spectral space, as multicomponent FabArray
        SpectralField fields;

    private:
        // tmpRealField and tmpSpectralField store fields
        // right before/after the Fourier transform
        SpectralField tmpSpectralField; // contains Complexs
        amrex::MultiFab tmpRealField; // contains Reals
        ablastr::math::anyfft::FFTplans forward_plan, backward_plan;
        // Correcting "shift" factors when performing FFT from/to
        // a cell-centered grid in real space, instead of a nodal grid
        // (0,1,2) is the dimension number
        SpectralShiftFactor shift0_FFTfromCell, shift0_FFTtoCell,
                            shift1_FFTfromCell, shift1_FFTtoCell,
                            shift2_FFTfromCell, shift2_FFTtoCell;

        bool m_periodic_single_box;
};

#endif // WARPX_SPECTRAL_FIELD_DATA_H_
