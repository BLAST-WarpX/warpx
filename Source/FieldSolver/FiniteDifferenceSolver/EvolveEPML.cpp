/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"

#include "BoundaryConditions/PML.H"
#include "BoundaryConditions/PMLComponent.H"
#include "BoundaryConditions/PML_current.H"
#ifndef WARPX_DIM_RZ
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <array>

using namespace amrex;

/**
 * \brief Update the E field, over one timestep
 */
void FiniteDifferenceSolver::EvolveEPML (
    ablastr::fields::MultiFabRegister& fields,
    PatchType patch_type,
    int level,
    MultiSigmaBox const& sigba,
    amrex::Real const dt, bool pml_has_particles ) {

    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(fields, patch_type, level, sigba, dt, pml_has_particles);
    WARPX_ABORT_WITH_MESSAGE(
        "PML are not implemented in cylindrical geometry.");
#else
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const ablastr::fields::VectorField Efield = (patch_type == PatchType::fine) ?
        fields.get_alldirs(FieldType::pml_E_fp, level) : fields.get_alldirs(FieldType::pml_E_cp, level);
    const ablastr::fields::VectorField Bfield = (patch_type == PatchType::fine) ?
        fields.get_alldirs(FieldType::pml_B_fp, level) : fields.get_alldirs(FieldType::pml_B_cp, level);
    const ablastr::fields::VectorField Jfield = (patch_type == PatchType::fine) ?
        fields.get_alldirs(FieldType::pml_j_fp, level) : fields.get_alldirs(FieldType::pml_j_cp, level);
    ablastr::fields::VectorField edge_lengths;
    if (fields.has_vector(FieldType::pml_edge_lengths, level)) {
        edge_lengths = fields.get_alldirs(FieldType::pml_edge_lengths, level);
    }
    amrex::MultiFab * Ffield = nullptr;
    if (fields.has(FieldType::pml_F_fp, level)) {
        Ffield = (patch_type == PatchType::fine) ?
            fields.get(FieldType::pml_F_fp, level) : fields.get(FieldType::pml_F_cp, level);
    }

    if (m_grid_type == GridType::Collocated) {

        EvolveEPMLCartesian <CartesianNodalAlgorithm> (
            Efield, Bfield, Jfield, edge_lengths, Ffield, sigba, dt, pml_has_particles );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee || m_fdtd_algo == ElectromagneticSolverAlgo::ECT) {

        EvolveEPMLCartesian <CartesianYeeAlgorithm> (
            Efield, Bfield, Jfield,  edge_lengths, Ffield, sigba, dt, pml_has_particles );

    } else if (m_fdtd_algo == ElectromagneticSolverAlgo::CKC) {

        EvolveEPMLCartesian <CartesianCKCAlgorithm> (
            Efield, Bfield, Jfield,  edge_lengths, Ffield, sigba, dt, pml_has_particles );

    } else {
        WARPX_ABORT_WITH_MESSAGE("EvolveEPML: Unknown algorithm");
    }
#endif
}


#ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveEPMLCartesian (
    std::array< amrex::MultiFab*, 3 > Efield,
    std::array< amrex::MultiFab*, 3 > const Bfield,
    std::array< amrex::MultiFab*, 3 > const Jfield,
    std::array< amrex::MultiFab*, 3 > const edge_lengths,
    amrex::MultiFab* const Ffield,
    MultiSigmaBox const& sigba,
    amrex::Real const dt, bool pml_has_particles ) {

    Real constexpr c2 = PhysConst::c * PhysConst::c;

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        amrex::Array4<amrex::Real> lx, ly, lz;
        if (EB::enabled()) {
            lx = edge_lengths[0]->array(mfi);
            ly = edge_lengths[1]->array(mfi);
            lz = edge_lengths[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tex  = mfi.tilebox(Efield[0]->ixType().ixType());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().ixType());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().ixType());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tex, tey, tez,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (lx && lx(i, j, k) <= 0) { return; }

                Ex(i, j, k, PMLComp::xz) -= c2 * dt * (
                    T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k, PMLComp::yx)
                  + T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k, PMLComp::yz) );
                Ex(i, j, k, PMLComp::xy) += c2 * dt * (
                    T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k, PMLComp::zx)
                  + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k, PMLComp::zy) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field push if this cell is fully covered by embedded boundaries
#ifdef WARPX_DIM_3D
                if (ly && ly(i,j,k) <= 0) { return; }
#elif defined(WARPX_DIM_XZ)
                //In XZ Ey is associated with a mesh node, so we need to check if the mesh node is covered
                amrex::ignore_unused(ly);
                if (lx && (lx(i, j, k)<=0 || lx(i-1, j, k)<=0 || lz(i, j-1, k)<=0 || lz(i, j, k)<=0)) { return; }
#endif

                Ey(i, j, k, PMLComp::yx) -= c2 * dt * (
                    T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k, PMLComp::zx)
                  + T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k, PMLComp::zy) );
                Ey(i, j, k, PMLComp::yz) += c2 * dt * (
                    T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k, PMLComp::xy)
                  + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k, PMLComp::xz) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (lz && lz(i, j, k) <= 0) { return; }

                Ez(i, j, k, PMLComp::zy) -= c2 * dt * (
                    T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k, PMLComp::xy)
                  + T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k, PMLComp::xz) );
                Ez(i, j, k, PMLComp::zx) += c2 * dt * (
                    T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k, PMLComp::yx)
                  + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k, PMLComp::yz) );
            }

        );

        // If F is not a null pointer, further update E using the grad(F) term
        // (hyperbolic correction for errors in charge conservation)
        if (Ffield) {
            // Extract field data for this grid/tile
            Array4<Real> const& F = Ffield->array(mfi);

            // Loop over the cells and update the fields
            amrex::ParallelFor(tex, tey, tez,

                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Ex(i, j, k, PMLComp::xx) += c2 * dt * (
                        T_Algo::UpwardDx(F, coefs_x, n_coefs_x, i, j, k, PMLComp::x)
                      + T_Algo::UpwardDx(F, coefs_x, n_coefs_x, i, j, k, PMLComp::y)
                      + T_Algo::UpwardDx(F, coefs_x, n_coefs_x, i, j, k, PMLComp::z) );
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Ey(i, j, k, PMLComp::yy) += c2 * dt * (
                        T_Algo::UpwardDy(F, coefs_y, n_coefs_y, i, j, k, PMLComp::x)
                      + T_Algo::UpwardDy(F, coefs_y, n_coefs_y, i, j, k, PMLComp::y)
                      + T_Algo::UpwardDy(F, coefs_y, n_coefs_y, i, j, k, PMLComp::z) );
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Ez(i, j, k, PMLComp::zz) += c2 * dt * (
                        T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, k, PMLComp::x)
                      + T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, k, PMLComp::y)
                      + T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, k, PMLComp::z) );
                }
            );
        }

        // Update the E field in the PML, using the current
        // deposited by the particles in the PML
        if (pml_has_particles) {

            // Extract field data for this grid/tile
            Array4<Real> const& Jx = Jfield[0]->array(mfi);
            Array4<Real> const& Jy = Jfield[1]->array(mfi);
            Array4<Real> const& Jz = Jfield[2]->array(mfi);
            const Real* sigmaj_x = sigba[mfi].sigma[0].data();
            const Real* sigmaj_y = sigba[mfi].sigma[1].data();
            const Real* sigmaj_z = sigba[mfi].sigma[2].data();
            int const x_lo = sigba[mfi].sigma[0].lo();
#if defined(WARPX_DIM_3D)
            int const y_lo = sigba[mfi].sigma[1].lo();
            int const z_lo = sigba[mfi].sigma[2].lo();
#else
            int const y_lo = 0;
            int const z_lo = sigba[mfi].sigma[1].lo();
#endif
            const Real mu_c2_dt = (PhysConst::mu0*PhysConst::c*PhysConst::c) * dt;

            amrex::ParallelFor( tex, tey, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    push_ex_pml_current(i, j, k, Ex, Jx,
                        sigmaj_y, sigmaj_z, y_lo, z_lo, mu_c2_dt);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    push_ey_pml_current(i, j, k, Ey, Jy,
                        sigmaj_x, sigmaj_z, x_lo, z_lo, mu_c2_dt);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    push_ez_pml_current(i, j, k, Ez, Jz,
                        sigmaj_x, sigmaj_y, x_lo, y_lo, mu_c2_dt);
                }
            );
        }
    } // MFIter
}

#endif // ifndef WARPX_DIM_RZ
