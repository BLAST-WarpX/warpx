/* Copyright 2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#ifndef WARPX_DIM_RZ
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <memory>

/**
 * Compute the minimum of array x, where x has dimension AMREX_SPACEDIM
 */
AMREX_FORCE_INLINE amrex::Real
minDim (const amrex::Real* x)
{
    return std::min({AMREX_D_DECL(x[0], x[1], x[2])});
}

/**
 * Determine the timestep of the simulation. */
void
WarpX::ComputeDt ()
{
    // Handle cases where the timestep is not limited by the speed of light
    // and no constant timestep is provided
    if (electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_const_dt.has_value(), "warpx.const_dt must be specified with the hybrid-PIC solver.");
    } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::None) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_const_dt.has_value() || m_dt_update_interval.isActivated(),
            "warpx.const_dt must be specified with the electrostatic solver, or warpx.dt_update_interval must be > 0."
        );
    }

    // Determine the appropriate timestep as limited by the speed of light
    const amrex::Real* dx = geom[max_level].CellSize();
    amrex::Real deltat = 0.;

    if (m_const_dt.has_value()) {
        deltat = m_const_dt.value();
    } else if (electrostatic_solver_id  != ElectrostaticSolverAlgo::None) {
        // Set dt for electrostatic algorithm
        if (m_max_dt.has_value()) {
            deltat = m_max_dt.value();
        } else {
            deltat = cfl * minDim(dx) / PhysConst::c;
        }
    } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        // Computation of dt for spectral algorithm
        // (determined by the minimum cell size in all directions)
        deltat = cfl * minDim(dx) / PhysConst::c;
    } else {
        // Computation of dt for FDTD algorithm
#ifdef WARPX_DIM_RZ
        // - In RZ geometry
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee) {
            deltat = cfl * CylindricalYeeAlgorithm::ComputeMaxDt(dx,  n_rz_azimuthal_modes);
#else
        // - In Cartesian geometry
        if (grid_type == GridType::Collocated) {
            deltat = cfl * CartesianNodalAlgorithm::ComputeMaxDt(dx);
        } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee
                    || electromagnetic_solver_id == ElectromagneticSolverAlgo::ECT) {
            deltat = cfl * CartesianYeeAlgorithm::ComputeMaxDt(dx);
        } else if (electromagnetic_solver_id == ElectromagneticSolverAlgo::CKC) {
            deltat = cfl * CartesianCKCAlgorithm::ComputeMaxDt(dx);
#endif
        } else {
            WARPX_ABORT_WITH_MESSAGE("ComputeDt: Unknown algorithm");
        }
    }

    dt.resize(0);
    dt.resize(max_level+1,deltat);

    if (m_do_subcycling) {
        for (int lev = max_level-1; lev >= 0; --lev) {
            dt[lev] = dt[lev+1] * refRatio(lev)[0];
        }
    }
}

/**
 * Determine the simulation timestep from the maximum speed of all particles
 * Sets timestep so that a particle can only cross cfl*dx cells per timestep.
 */
void
WarpX::UpdateDtFromParticleSpeeds ()
{
    const amrex::Real* dx = geom[max_level].CellSize();
    const amrex::Real dx_min = minDim(dx);

    const amrex::ParticleReal max_v = mypc->maxParticleVelocity();
    amrex::Real deltat_new = 0.;

    // Protections from overly-large timesteps
    if (max_v == 0) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_max_dt.has_value(), "Particles at rest and no constant or maximum timestep specified. Aborting.");
        deltat_new = m_max_dt.value();
    } else {
        deltat_new = cfl * dx_min / max_v;
    }

    // Restrict to be less than user-specified maximum timestep, if present
    if (m_max_dt.has_value()) {
        deltat_new = std::min(deltat_new, m_max_dt.value());
    }

    // Update dt
    dt[max_level] = deltat_new;

    for (int lev = max_level-1; lev >= 0; --lev) {
        dt[lev] = dt[lev+1] * refRatio(lev)[0];
    }
}

void
WarpX::PrintDtDxDyDz ()
{
    for (int lev=0; lev <= max_level; lev++) {
        const amrex::Real* dx_lev = geom[lev].CellSize();
        amrex::Print() << "Level " << lev << ": dt = " << dt[lev]
#if defined(WARPX_DIM_1D_Z)
                       << " ; dz = " << dx_lev[0] << '\n';
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                       << " ; dx = " << dx_lev[0]
                       << " ; dz = " << dx_lev[1] << '\n';
#elif defined(WARPX_DIM_3D)
                       << " ; dx = " << dx_lev[0]
                       << " ; dy = " << dx_lev[1]
                       << " ; dz = " << dx_lev[2] << '\n';
#endif
    }
}
