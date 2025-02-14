/* Copyright 2019-2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SpectralAlgorithms/PsatdAlgorithmGalileanRZ.H"
#include "SpectralAlgorithms/PsatdAlgorithmRZ.H"
#include "SpectralAlgorithms/PsatdAlgorithmPmlRZ.H"
#include "SpectralKSpaceRZ.H"
#include "SpectralSolverRZ.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

/* \brief Initialize the spectral Maxwell solver
 *
 * This function selects the spectral algorithm to be used, allocates the
 * corresponding coefficients for the discretized field update equation,
 * and prepares the structures that store the fields in spectral space.
 *
 * \param n_rz_azimuthal_modes Number of azimuthal modes
 * \param norder_z Order of accuracy of the spatial derivatives along z
 * \param grid_type Type of grid (collocated or not)
 * \param dx       Cell size along each dimension
 * \param dt       Time step
 * \param with_pml Whether PML boundary will be used
 */
SpectralSolverRZ::SpectralSolverRZ (const int lev,
                                    amrex::BoxArray const & realspace_ba,
                                    amrex::DistributionMapping const & dm,
                                    int const n_rz_azimuthal_modes,
                                    int const norder_z,
                                    ablastr::utils::enums::GridType grid_type,
                                    const amrex::Vector<amrex::Real>& v_galilean,
                                    amrex::RealVect const dx, amrex::Real const dt,
                                    bool const with_pml,
                                    bool const update_with_rho,
                                    const bool fft_do_time_averaging,
                                    const JInTime J_in_time,
                                    const RhoInTime rho_in_time,
                                    const bool dive_cleaning,
                                    const bool divb_cleaning)
    : m_dt(dt), k_space(realspace_ba, dm, dx)
{
    // Initialize all structures using the same distribution mapping dm

    // - The k space object contains info about the size of
    //   the spectral space corresponding to each box in `realspace_ba`,
    //   as well as the value of the corresponding k coordinates.

    const bool is_pml = false;
    m_spectral_index = SpectralFieldIndex(
        update_with_rho, fft_do_time_averaging, J_in_time, rho_in_time,
        dive_cleaning, divb_cleaning, is_pml, with_pml);

    // - Select the algorithm depending on the input parameters
    //   Initialize the corresponding coefficients over k space
    if (with_pml) {
            PML_algorithm = std::make_unique<PsatdAlgorithmPmlRZ>(
                k_space, dm, m_spectral_index, n_rz_azimuthal_modes, norder_z, grid_type, dt);
    }
    if (v_galilean[2] == 0) {
         // v_galilean is 0: use standard PSATD algorithm
        algorithm = std::make_unique<PsatdAlgorithmRZ>(
            k_space, dm, m_spectral_index, n_rz_azimuthal_modes, norder_z, grid_type, dt,
            update_with_rho, fft_do_time_averaging, J_in_time, rho_in_time, dive_cleaning, divb_cleaning);
    } else {
        // Otherwise: use the Galilean algorithm
        algorithm = std::make_unique<PsatdAlgorithmGalileanRZ>(
            k_space, dm, m_spectral_index, n_rz_azimuthal_modes, norder_z, grid_type, v_galilean, dt, update_with_rho);
    }

    // - Initialize arrays for fields in spectral space + FFT plans
    field_data = SpectralFieldDataRZ(lev, realspace_ba, k_space, dm,
                                     m_spectral_index.n_fields,
                                     n_rz_azimuthal_modes);
}

/* \brief Transform the component `i_comp` of MultiFab `field_mf`
 *  to spectral space, and store the corresponding result internally
 *  (in the spectral field specified by `field_index`) */
void
SpectralSolverRZ::ForwardTransform (const int lev,
                                    amrex::MultiFab const & field_mf, int const field_index,
                                    int const i_comp) {
    WARPX_PROFILE("SpectralSolverRZ::ForwardTransform");
    field_data.ForwardTransform(lev, field_mf, field_index, i_comp);
}

/* \brief Transform the two MultiFabs `field_mf1` and `field_mf2`
 *  to spectral space, and store the corresponding results internally
 *  (in the spectral field specified by `field_index1` and `field_index2`) */
void
SpectralSolverRZ::ForwardTransform (const int lev,
                                    amrex::MultiFab const & field_mf1, int const field_index1,
                                    amrex::MultiFab const & field_mf2, int const field_index2) {
    WARPX_PROFILE("SpectralSolverRZ::ForwardTransform");
    field_data.ForwardTransform(lev,
                                field_mf1, field_index1,
                                field_mf2, field_index2);
}

/* \brief Transform spectral field specified by `field_index` back to
 * real space, and store it in the component `i_comp` of `field_mf` */
void
SpectralSolverRZ::BackwardTransform (const int lev,
                                     amrex::MultiFab& field_mf, int const field_index,
                                     int const i_comp) {
    WARPX_PROFILE("SpectralSolverRZ::BackwardTransform");
    field_data.BackwardTransform(lev, field_mf, field_index, i_comp);
}

/* \brief Transform spectral fields specified by `field_index1` and `field_index2`
 * back to real space, and store it in `field_mf1` and `field_mf2`*/
void
SpectralSolverRZ::BackwardTransform (const int lev,
                                     amrex::MultiFab& field_mf1, int const field_index1,
                                     amrex::MultiFab& field_mf2, int const field_index2) {
    WARPX_PROFILE("SpectralSolverRZ::BackwardTransform");
    field_data.BackwardTransform(lev,
                                 field_mf1, field_index1,
                                 field_mf2, field_index2);
}

/* \brief Update the fields in spectral space, over one timestep */
void
SpectralSolverRZ::pushSpectralFields (const bool doing_pml) {
    WARPX_PROFILE("SpectralSolverRZ::pushSpectralFields");
    // Virtual function: the actual function used here depends
    // on the sub-class of `SpectralBaseAlgorithm` that was
    // initialized in the constructor of `SpectralSolverRZ`
    if (doing_pml) {
        PML_algorithm->pushSpectralFields(field_data);
    } else {
        algorithm->pushSpectralFields(field_data);
    }
}

/**
  * \brief Public interface to call the member function ComputeSpectralDivE
  * of the base class SpectralBaseAlgorithmRZ from objects of class SpectralSolverRZ
  */
void
SpectralSolverRZ::ComputeSpectralDivE (const int lev,
                                       ablastr::fields::VectorField const & Efield,
                                       amrex::MultiFab& divE
)
{
    algorithm->ComputeSpectralDivE(lev, field_data, Efield, divE);
}

/**
 * \brief Public interface to call the virtual function \c CurrentCorrection,
 * defined in the base class SpectralBaseAlgorithmRZ and possibly overridden
 * by its derived classes (e.g. PsatdAlgorithmRZ), from
 * objects of class SpectralSolverRZ through the private unique pointer \c algorithm
 */
void
SpectralSolverRZ::CurrentCorrection ()
{
    algorithm->CurrentCorrection(field_data);
}

void
SpectralSolverRZ::VayDeposition ()
{
    algorithm->VayDeposition(field_data);
}
