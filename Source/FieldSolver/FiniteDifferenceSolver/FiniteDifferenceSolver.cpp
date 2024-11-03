/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#ifndef WARPX_DIM_RZ
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#ifdef WARPX_DIM_RZ
#   include "WarpX.H"
#endif

#include <AMReX.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_PODVector.H>
#include <AMReX_Vector.H>

#include <vector>

/* This function initializes the stencil coefficients for the chosen finite-difference algorithm */
FiniteDifferenceSolver::FiniteDifferenceSolver (
    ElectromagneticSolverAlgo const fdtd_algo,
    std::array<amrex::Real,3> cell_size,
    ablastr::utils::enums::GridType grid_type):
    // Register the type of finite-difference algorithm
    m_fdtd_algo{fdtd_algo},
    m_grid_type{grid_type}
{
    // return if not FDTD
    if (fdtd_algo == ElectromagneticSolverAlgo::None || fdtd_algo == ElectromagneticSolverAlgo::PSATD) {
        return;
    }

    // Calculate coefficients of finite-difference stencil
#ifdef WARPX_DIM_RZ
    m_dr = cell_size[0];
    m_nmodes = WarpX::n_rz_azimuthal_modes;
    m_rmin = WarpX::GetInstance().Geom(0).ProbLo(0);
    if (fdtd_algo == ElectromagneticSolverAlgo::Yee ||
        fdtd_algo == ElectromagneticSolverAlgo::HybridPIC ) {
        CylindricalYeeAlgorithm::InitializeStencilCoefficients( cell_size,
            m_h_stencil_coefs_r, m_h_stencil_coefs_z );
        m_stencil_coefs_r.resize(m_h_stencil_coefs_r.size());
        m_stencil_coefs_z.resize(m_h_stencil_coefs_z.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              m_h_stencil_coefs_r.begin(), m_h_stencil_coefs_r.end(),
                              m_stencil_coefs_r.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              m_h_stencil_coefs_z.begin(), m_h_stencil_coefs_z.end(),
                              m_stencil_coefs_z.begin());
        amrex::Gpu::synchronize();
    } else {
        WARPX_ABORT_WITH_MESSAGE(
            "FiniteDifferenceSolver: Unknown algorithm");
    }
#else
    if (grid_type == ablastr::utils::enums::GridType::Collocated) {

        CartesianNodalAlgorithm::InitializeStencilCoefficients( cell_size,
            m_h_stencil_coefs_x, m_h_stencil_coefs_y, m_h_stencil_coefs_z );

    } else if (fdtd_algo == ElectromagneticSolverAlgo::Yee ||
               fdtd_algo == ElectromagneticSolverAlgo::ECT ||
               fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {

        CartesianYeeAlgorithm::InitializeStencilCoefficients( cell_size,
            m_h_stencil_coefs_x, m_h_stencil_coefs_y, m_h_stencil_coefs_z );

    } else if (fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        CartesianCKCAlgorithm::InitializeStencilCoefficients( cell_size,
            m_h_stencil_coefs_x, m_h_stencil_coefs_y, m_h_stencil_coefs_z );

    } else {
        WARPX_ABORT_WITH_MESSAGE(
            "FiniteDifferenceSolver: Unknown algorithm");
    }

    m_stencil_coefs_x.resize(m_h_stencil_coefs_x.size());
    m_stencil_coefs_y.resize(m_h_stencil_coefs_y.size());
    m_stencil_coefs_z.resize(m_h_stencil_coefs_z.size());

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          m_h_stencil_coefs_x.begin(), m_h_stencil_coefs_x.end(),
                          m_stencil_coefs_x.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          m_h_stencil_coefs_y.begin(), m_h_stencil_coefs_y.end(),
                          m_stencil_coefs_y.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          m_h_stencil_coefs_z.begin(), m_h_stencil_coefs_z.end(),
                          m_stencil_coefs_z.begin());
    amrex::Gpu::synchronize();
#endif
}
