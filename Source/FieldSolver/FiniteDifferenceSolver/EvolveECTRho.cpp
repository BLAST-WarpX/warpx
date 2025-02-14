/* Copyright 2020 Remi Lehe, Lorenzo Giacomel
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#ifndef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

using namespace amrex;
using namespace ablastr::fields;

/**
 * \brief Update the B field, over one timestep
 */
void FiniteDifferenceSolver::EvolveECTRho (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& edge_lengths,
    ablastr::fields::VectorField const& face_areas,
    ablastr::fields::VectorField const& ECTRhofield,
    const int lev) {

#if !defined(WARPX_DIM_RZ) and defined(AMREX_USE_EB)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::ECT) {

        EvolveRhoCartesianECT(Efield, edge_lengths, face_areas, ECTRhofield, lev);

    }
#else
    amrex::ignore_unused(Efield, edge_lengths, face_areas, ECTRhofield, lev);
#endif
}

// If we implement ECT in 1D we will need to take care of this #ifndef differently
#ifndef WARPX_DIM_RZ
void FiniteDifferenceSolver::EvolveRhoCartesianECT (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& edge_lengths,
    ablastr::fields::VectorField const& face_areas,
    ablastr::fields::VectorField const& ECTRhofield, const int lev ) {
#ifdef AMREX_USE_EB

#if !(defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ))
    WARPX_ABORT_WITH_MESSAGE(
        "EvolveRhoCartesianECT: Embedded Boundaries are only implemented in 3D and XZ");
#endif

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*ECTRhofield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        amrex::Array4<amrex::Real> const &Ex = Efield[0]->array(mfi);
        amrex::Array4<amrex::Real> const &Ey = Efield[1]->array(mfi);
        amrex::Array4<amrex::Real> const &Ez = Efield[2]->array(mfi);
        amrex::Array4<amrex::Real> const &Rhox = ECTRhofield[0]->array(mfi);
        amrex::Array4<amrex::Real> const &Rhoy = ECTRhofield[1]->array(mfi);
        amrex::Array4<amrex::Real> const &Rhoz = ECTRhofield[2]->array(mfi);
        amrex::Array4<amrex::Real> const &lx = edge_lengths[0]->array(mfi);
        amrex::Array4<amrex::Real> const &ly = edge_lengths[1]->array(mfi);
        amrex::Array4<amrex::Real> const &lz = edge_lengths[2]->array(mfi);
        amrex::Array4<amrex::Real> const &Sx = face_areas[0]->array(mfi);
        amrex::Array4<amrex::Real> const &Sy = face_areas[1]->array(mfi);
        amrex::Array4<amrex::Real> const &Sz = face_areas[2]->array(mfi);

        // Extract tileboxes for which to loop
        amrex::Box const &trhox = mfi.tilebox(ECTRhofield[0]->ixType().toIntVect());
        amrex::Box const &trhoy = mfi.tilebox(ECTRhofield[1]->ixType().toIntVect());
        amrex::Box const &trhoz = mfi.tilebox(ECTRhofield[2]->ixType().toIntVect());

        amrex::ParallelFor(trhox, trhoy, trhoz,

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (Sx(i, j, k) <= 0) { return; }

// If we implement ECT in 1D we will need to take care of this #ifndef differently
#ifndef WARPX_DIM_XZ
                Rhox(i, j, k) = (Ey(i, j, k) * ly(i, j, k) - Ey(i, j, k + 1) * ly(i, j, k + 1) +
                    Ez(i, j + 1, k) * lz(i, j + 1, k) - Ez(i, j, k) * lz(i, j, k)) / Sx(i, j, k);
#endif
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (Sy(i, j, k) <= 0) { return; }

#ifdef WARPX_DIM_XZ
                Rhoy(i, j, k) = (Ez(i, j, k) * lz(i, j, k) - Ez(i + 1, j, k) * lz(i + 1, j, k) +
                    Ex(i, j + 1, k) * lx(i, j + 1, k) - Ex(i, j, k) * lx(i, j, k)) / Sy(i, j, k);
#elif defined(WARPX_DIM_3D)
                Rhoy(i, j, k) = (Ez(i, j, k) * lz(i, j, k) - Ez(i + 1, j, k) * lz(i + 1, j, k) +
                    Ex(i, j, k + 1) * lx(i, j, k + 1) - Ex(i, j, k) * lx(i, j, k)) / Sy(i, j, k);
#else
                WARPX_ABORT_WITH_MESSAGE("EvolveRhoCartesianECT: Embedded Boundaries are only implemented in 3D and XZ");
#endif
            },

            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (Sz(i, j, k) <= 0) { return; }

// If we implement ECT in 1D we will need to take care of this #ifndef differently
#ifndef WARPX_DIM_XZ
                Rhoz(i, j, k) =  (Ex(i, j, k) * lx(i, j, k) - Ex(i, j + 1, k) * lx(i, j + 1, k) +
                    Ey(i + 1, j, k) * ly(i + 1, j, k) - Ey(i, j, k) * ly(i, j, k)) / Sz(i, j, k);
#endif
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
#ifdef WARPX_DIM_XZ
        amrex::ignore_unused(Ey, Rhox, Rhoz, ly);
#endif
    }
#else
    amrex::ignore_unused(Efield, edge_lengths, face_areas, ECTRhofield, lev);
#endif
}
#endif
