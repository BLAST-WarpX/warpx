/* Copyright 2019 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/SpectralSolver/SpectralAlgorithms/SpectralBaseAlgorithm.H"
#include "FieldSolver/SpectralSolver/SpectralFieldData.H"
#include "SpectralAlgorithms/PsatdAlgorithmComoving.H"
#include "SpectralAlgorithms/PsatdAlgorithmPml.H"
#include "SpectralAlgorithms/PsatdAlgorithmGalilean.H"
#include "SpectralAlgorithms/PsatdAlgorithmJRhomFirstOrder.H"
#include "SpectralAlgorithms/PsatdAlgorithmJRhomSecondOrder.H"
#include "SpectralKSpace.H"
#include "SpectralSolver.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/utils/Enums.H>

#include <memory>

#if WARPX_USE_FFT

SpectralSolver::SpectralSolver (
                const amrex::BoxArray& realspace_ba,
                const amrex::DistributionMapping& dm,
                const int norder_x,
                const int norder_y,
                const int norder_z,
                ablastr::utils::enums::GridType grid_type,
                const amrex::Vector<amrex::Real>& v_galilean,
                const amrex::Vector<amrex::Real>& v_comoving,
                const amrex::RealVect dx, const amrex::Real dt,
                const bool pml, const bool periodic_single_box,
                const bool update_with_rho,
                const bool fft_do_time_averaging,
                const PSATDSolutionType psatd_solution_type,
                const TimeDependencyJ time_dependency_J,
                const TimeDependencyRho time_dependency_rho,
                const bool dive_cleaning,
                const bool divb_cleaning)
    : m_dt(dt)
{
    // Initialize all structures using the same distribution mapping dm

    // - Initialize k space object (Contains info about the size of
    // the spectral space corresponding to each box in `realspace_ba`,
    // as well as the value of the corresponding k coordinates)
    const SpectralKSpace k_space= SpectralKSpace(realspace_ba, dm, dx);

    m_spectral_index = SpectralFieldIndex(
        update_with_rho, fft_do_time_averaging, time_dependency_J, time_dependency_rho,
        dive_cleaning, divb_cleaning, pml);

    // - Select the algorithm depending on the input parameters
    //   Initialize the corresponding coefficients over k space

    if (pml) // PSATD or Galilean PSATD equations in the PML region
    {
        algorithm = std::make_unique<PsatdAlgorithmPml>(
            k_space, dm, m_spectral_index, norder_x, norder_y, norder_z, grid_type,
            v_galilean, dt, dive_cleaning, divb_cleaning);
    }
    else // PSATD equations in the regular domain
    {
        // Comoving PSATD algorithm
        if (v_comoving[0] != 0. || v_comoving[1] != 0. || v_comoving[2] != 0.)
        {
            algorithm = std::make_unique<PsatdAlgorithmComoving>(
                k_space, dm, m_spectral_index, norder_x, norder_y, norder_z, grid_type,
                v_comoving, dt, update_with_rho);
        }
        // Galilean PSATD algorithm (only J constant in time)
        else if (v_galilean[0] != 0. || v_galilean[1] != 0. || v_galilean[2] != 0.)
        {
            algorithm = std::make_unique<PsatdAlgorithmGalilean>(
                k_space, dm, m_spectral_index, norder_x, norder_y, norder_z, grid_type,
                v_galilean, dt, update_with_rho, fft_do_time_averaging,
                dive_cleaning, divb_cleaning);
        }
        else if (psatd_solution_type == PSATDSolutionType::FirstOrder)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !fft_do_time_averaging,
                "psatd.do_time_averaging=1 not supported when psatd.solution_type=first-order");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (!dive_cleaning && !divb_cleaning) || (dive_cleaning && divb_cleaning),
                "warpx.do_dive_cleaning and warpx.do_divb_cleaning must be equal when psatd.solution_type=first-order");

            const bool div_cleaning = (dive_cleaning && divb_cleaning);

            // First-order PSATD equations with variable time dependency of J and rho
            // (valid also for standard PSATD, where J is constant and rho is linear)
            algorithm = std::make_unique<PsatdAlgorithmJRhomFirstOrder>(
                k_space, dm, m_spectral_index, norder_x, norder_y, norder_z, grid_type,
                dt, div_cleaning, time_dependency_J, time_dependency_rho);
        }
        else if (psatd_solution_type == PSATDSolutionType::SecondOrder)
        {
            // Second-order PSATD equations with variable time dependency of J and rho
            // (valid also for standard PSATD, where J is constant and rho is linear)
            algorithm = std::make_unique<PsatdAlgorithmJRhomSecondOrder>(
              k_space, dm, m_spectral_index, norder_x, norder_y, norder_z, grid_type,
              dt, update_with_rho, fft_do_time_averaging, dive_cleaning, divb_cleaning, time_dependency_J, time_dependency_rho);
        }
    }

    // - Initialize arrays for fields in spectral space + FFT plans
    field_data = SpectralFieldData(realspace_ba, k_space, dm,
                                   m_spectral_index.n_fields, periodic_single_box);
}

void
SpectralSolver::ForwardTransform (const int lev,
                                  const amrex::MultiFab& mf,
                                  const int field_index,
                                  const int i_comp)
{
    WARPX_PROFILE("SpectralSolver::ForwardTransform");
    field_data.ForwardTransform(lev, mf, field_index, i_comp);
}

void
SpectralSolver::BackwardTransform( const int lev,
                                   amrex::MultiFab& mf,
                                   const int field_index,
                                   const amrex::IntVect& fill_guards,
                                   const int i_comp )
{
    WARPX_PROFILE("SpectralSolver::BackwardTransform");
    field_data.BackwardTransform(lev, mf, field_index, fill_guards, i_comp);
}

void
SpectralSolver::pushSpectralFields(){
    WARPX_PROFILE("SpectralSolver::pushSpectralFields");
    // Virtual function: the actual function used here depends
    // on the sub-class of `SpectralBaseAlgorithm` that was
    // initialized in the constructor of `SpectralSolver`
    algorithm->pushSpectralFields( field_data );
}

void SpectralSolver::ComputeSpectralDivE (
        const int lev,
        ablastr::fields::VectorField const & Efield,
        amrex::MultiFab& divE)
{
    algorithm->ComputeSpectralDivE(lev, field_data, Efield, divE );
}

void SpectralSolver::CurrentCorrection ()
{
    algorithm->CurrentCorrection(field_data);
}

void SpectralSolver::VayDeposition ()
{
    algorithm->VayDeposition(field_data);
}

void SpectralSolver::CopySpectralDataComp (const int src_comp, const int dest_comp)
{
    // The last two arguments represent the number of components and
    // the number of ghost cells to perform this operation
    Copy(field_data.fields, field_data.fields, src_comp, dest_comp, 1, 0);
}

void SpectralSolver::ZeroOutDataComp (const int icomp)
{
    // The last argument represents the number of components to perform this operation
    field_data.fields.setVal(0., icomp, 1);
}

void SpectralSolver::ScaleDataComp (const int icomp, const amrex::Real scale_factor)
{
    // The last argument represents the number of components to perform this operation
    field_data.fields.mult(scale_factor, icomp, 1);
}

#endif // WARPX_USE_FFT
