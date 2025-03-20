/* Copyright 2019 Aurore Blelly, Axel Huebl, Maxence Thevenet
 * Remi Lehe, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "BoundaryConditions/PML.H"
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
#   include "BoundaryConditions/PML_RZ.H"
#endif
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "PML_current.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX_PML_kernels.H"

#include <ablastr/fields/MultiFabRegister.H>

#ifdef AMREX_USE_SENSEI_INSITU
#   include <AMReX_AmrMeshInSituBridge.H>
#endif
#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

using namespace amrex;

void
WarpX::DampPML ()
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        DampPML(lev);
    }
}

void
WarpX::DampPML (const int lev)
{
    DampPML(lev, PatchType::fine);
    if (lev > 0) { DampPML(lev, PatchType::coarse); }
}

void
WarpX::DampPML (const int lev, PatchType patch_type)
{
    if (!do_pml) { return; }

    WARPX_PROFILE("WarpX::DampPML()");
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    if (pml_rz[lev]) {
        using ablastr::fields::Direction;
        using warpx::fields::FieldType;
        pml_rz[lev]->ApplyDamping( m_fields.get(FieldType::Efield_fp, Direction{1}, lev),
                                   m_fields.get(FieldType::Efield_fp, Direction{2}, lev),
                                   m_fields.get(FieldType::Bfield_fp, Direction{1}, lev),
                                   m_fields.get(FieldType::Bfield_fp, Direction{2}, lev),
                                   dt[lev], m_fields);
    }
#endif
    if (pml[lev]) {
        DampPML_Cartesian (lev, patch_type);
    }
}

void
WarpX::DampPML_Cartesian (const int lev, PatchType patch_type)
{
    const bool dive_cleaning = WarpX::do_pml_dive_cleaning;
    const bool divb_cleaning = WarpX::do_pml_divb_cleaning;

    if (pml[lev]->ok())
    {
        using warpx::fields::FieldType;

        const auto& pml_E = (patch_type == PatchType::fine) ? m_fields.get_alldirs(FieldType::pml_E_fp, lev) : m_fields.get_alldirs(FieldType::pml_E_cp, lev);
        const auto& pml_B = (patch_type == PatchType::fine) ? m_fields.get_alldirs(FieldType::pml_B_fp, lev) : m_fields.get_alldirs(FieldType::pml_B_cp, lev);
        const auto& sigba = (patch_type == PatchType::fine) ? pml[lev]->GetMultiSigmaBox_fp() : pml[lev]->GetMultiSigmaBox_cp();

        const amrex::IntVect Ex_stag = pml_E[0]->ixType().toIntVect();
        const amrex::IntVect Ey_stag = pml_E[1]->ixType().toIntVect();
        const amrex::IntVect Ez_stag = pml_E[2]->ixType().toIntVect();

        const amrex::IntVect Bx_stag = pml_B[0]->ixType().toIntVect();
        const amrex::IntVect By_stag = pml_B[1]->ixType().toIntVect();
        const amrex::IntVect Bz_stag = pml_B[2]->ixType().toIntVect();

        amrex::IntVect F_stag;
        if (m_fields.has(FieldType::pml_F_fp, lev)) {
            amrex::MultiFab* pml_F = (patch_type == PatchType::fine) ?
                m_fields.get(FieldType::pml_F_fp, lev) : m_fields.get(FieldType::pml_F_cp, lev);
            F_stag = pml_F->ixType().toIntVect();
        }

        amrex::IntVect G_stag;
        if (m_fields.has(FieldType::pml_G_fp, lev)) {
            amrex::MultiFab* pml_G = (patch_type == PatchType::fine) ?
                m_fields.get(FieldType::pml_G_fp, lev) : m_fields.get(FieldType::pml_G_cp, lev);
            G_stag = pml_G->ixType().toIntVect();
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*pml_E[0], TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            const Box& tex  = mfi.tilebox( pml_E[0]->ixType().toIntVect() );
            const Box& tey  = mfi.tilebox( pml_E[1]->ixType().toIntVect() );
            const Box& tez  = mfi.tilebox( pml_E[2]->ixType().toIntVect() );
            const Box& tbx  = mfi.tilebox( pml_B[0]->ixType().toIntVect() );
            const Box& tby  = mfi.tilebox( pml_B[1]->ixType().toIntVect() );
            const Box& tbz  = mfi.tilebox( pml_B[2]->ixType().toIntVect() );

            auto const& pml_Exfab = pml_E[0]->array(mfi);
            auto const& pml_Eyfab = pml_E[1]->array(mfi);
            auto const& pml_Ezfab = pml_E[2]->array(mfi);
            auto const& pml_Bxfab = pml_B[0]->array(mfi);
            auto const& pml_Byfab = pml_B[1]->array(mfi);
            auto const& pml_Bzfab = pml_B[2]->array(mfi);

            amrex::Real const * AMREX_RESTRICT sigma_fac_x = sigba[mfi].sigma_fac[0].data();
#if defined(WARPX_DIM_3D)
            amrex::Real const * AMREX_RESTRICT sigma_fac_y = sigba[mfi].sigma_fac[1].data();
            amrex::Real const * AMREX_RESTRICT sigma_fac_z = sigba[mfi].sigma_fac[2].data();
#else
            amrex::Real const * AMREX_RESTRICT sigma_fac_y = nullptr;
            amrex::Real const * AMREX_RESTRICT sigma_fac_z = sigba[mfi].sigma_fac[1].data();
#endif
            amrex::Real const * AMREX_RESTRICT sigma_star_fac_x = sigba[mfi].sigma_star_fac[0].data();
#if defined(WARPX_DIM_3D)
            amrex::Real const * AMREX_RESTRICT sigma_star_fac_y = sigba[mfi].sigma_star_fac[1].data();
            amrex::Real const * AMREX_RESTRICT sigma_star_fac_z = sigba[mfi].sigma_star_fac[2].data();
#else
            amrex::Real const * AMREX_RESTRICT sigma_star_fac_y = nullptr;
            amrex::Real const * AMREX_RESTRICT sigma_star_fac_z = sigba[mfi].sigma_star_fac[1].data();
#endif
            int const x_lo = sigba[mfi].sigma_fac[0].lo();
#if defined(WARPX_DIM_3D)
            int const y_lo = sigba[mfi].sigma_fac[1].lo();
            int const z_lo = sigba[mfi].sigma_fac[2].lo();
#else
            int const y_lo = 0;
            int const z_lo = sigba[mfi].sigma_fac[1].lo();
#endif

            amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                warpx_damp_pml_ex(i, j, k, pml_Exfab, Ex_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
                                  sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, x_lo, y_lo, z_lo,
                                  dive_cleaning);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                warpx_damp_pml_ey(i, j, k, pml_Eyfab, Ey_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
                                  sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, x_lo, y_lo, z_lo,
                                  dive_cleaning);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                warpx_damp_pml_ez(i, j, k, pml_Ezfab, Ez_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
                                  sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, x_lo, y_lo, z_lo,
                                  dive_cleaning);
            });

            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                warpx_damp_pml_bx(i, j, k, pml_Bxfab, Bx_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
                                  sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, x_lo, y_lo, z_lo,
                                  divb_cleaning);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                warpx_damp_pml_by(i, j, k, pml_Byfab, By_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
                                  sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, x_lo, y_lo, z_lo,
                                  divb_cleaning);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                warpx_damp_pml_bz(i, j, k, pml_Bzfab, Bz_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
                                  sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, x_lo, y_lo, z_lo,
                                  divb_cleaning);
            });

            // For warpx_damp_pml_F(), mfi.nodaltilebox is used in the ParallelFor loop and here we
            // use mfi.tilebox. However, it does not matter because in damp_pml, where nodaltilebox
            // is used, only a simple multiplication is performed.
            if (m_fields.has(FieldType::pml_F_fp, lev)) {
                amrex::MultiFab* pml_F = (patch_type == PatchType::fine) ?
                    m_fields.get(FieldType::pml_F_fp, lev) : m_fields.get(FieldType::pml_F_cp, lev);
                const Box& tnd = mfi.nodaltilebox();
                auto const& pml_F_fab = pml_F->array(mfi);
                amrex::ParallelFor(tnd, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    warpx_damp_pml_scalar(i, j, k, pml_F_fab, F_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
                                          sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, x_lo, y_lo, z_lo);
                });
            }

            // Damp G when WarpX::do_divb_cleaning = true
            if (m_fields.has(FieldType::pml_G_fp, lev)) {
                amrex::MultiFab* pml_G = (patch_type == PatchType::fine) ?
                    m_fields.get(FieldType::pml_G_fp, lev) : m_fields.get(FieldType::pml_G_cp, lev);

                const Box& tb = mfi.tilebox(G_stag);
                auto const& pml_G_fab = pml_G->array(mfi);
                amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    warpx_damp_pml_scalar(i, j, k, pml_G_fab, G_stag, sigma_fac_x, sigma_fac_y, sigma_fac_z,
                                          sigma_star_fac_x, sigma_star_fac_y, sigma_star_fac_z, x_lo, y_lo, z_lo);
                });
            }
        }
    }
}

void
WarpX::DampJPML ()
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        DampJPML(lev);
    }
}

void
WarpX::DampJPML (int lev)
{
    DampJPML(lev, PatchType::fine);
    if (lev > 0) { DampJPML(lev, PatchType::coarse); }
}

void
WarpX::DampJPML (int lev, PatchType patch_type)
{
    if (!do_pml) { return; }
    if (!do_pml_j_damping) { return; }
    if (!pml[lev]) { return; }

    WARPX_PROFILE("WarpX::DampJPML()");

    if (pml[lev]->ok())
    {
        using warpx::fields::FieldType;

        const auto& pml_j = (patch_type == PatchType::fine) ? m_fields.get_alldirs(FieldType::pml_j_fp, lev) : m_fields.get_alldirs(FieldType::pml_j_cp, lev);
        const auto& sigba = (patch_type == PatchType::fine) ? pml[lev]->GetMultiSigmaBox_fp()
                                                            : pml[lev]->GetMultiSigmaBox_cp();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*pml_j[0], TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            auto const& pml_jxfab = pml_j[0]->array(mfi);
            auto const& pml_jyfab = pml_j[1]->array(mfi);
            auto const& pml_jzfab = pml_j[2]->array(mfi);
            const Real* sigma_cumsum_fac_j_x = sigba[mfi].sigma_cumsum_fac[0].data();
            const Real* sigma_star_cumsum_fac_j_x = sigba[mfi].sigma_star_cumsum_fac[0].data();
#if defined(WARPX_DIM_3D)
            const Real* sigma_cumsum_fac_j_y = sigba[mfi].sigma_cumsum_fac[1].data();
            const Real* sigma_star_cumsum_fac_j_y = sigba[mfi].sigma_star_cumsum_fac[1].data();
            const Real* sigma_cumsum_fac_j_z = sigba[mfi].sigma_cumsum_fac[2].data();
            const Real* sigma_star_cumsum_fac_j_z = sigba[mfi].sigma_star_cumsum_fac[2].data();
#else
            const Real* sigma_cumsum_fac_j_y = nullptr;
            const Real* sigma_star_cumsum_fac_j_y = nullptr;
            const Real* sigma_cumsum_fac_j_z = sigba[mfi].sigma_cumsum_fac[1].data();
            const Real* sigma_star_cumsum_fac_j_z = sigba[mfi].sigma_star_cumsum_fac[1].data();
#endif

            // Skip the field update if this gridpoint is inside the embedded boundary
            amrex::Array4<amrex::Real> eb_lxfab, eb_lyfab, eb_lzfab;
            if (EB::enabled()) {
                const auto &pml_edge_lenghts = m_fields.get_alldirs(FieldType::pml_edge_lengths, lev);

                eb_lxfab = pml_edge_lenghts[0]->array(mfi);
                eb_lyfab = pml_edge_lenghts[1]->array(mfi);
                eb_lzfab = pml_edge_lenghts[2]->array(mfi);
            } else {
                amrex::ignore_unused(eb_lxfab, eb_lyfab, eb_lzfab);
            }

            const Box& tjx  = mfi.tilebox( pml_j[0]->ixType().toIntVect() );
            const Box& tjy  = mfi.tilebox( pml_j[1]->ixType().toIntVect() );
            const Box& tjz  = mfi.tilebox( pml_j[2]->ixType().toIntVect() );

            int const x_lo = sigba[mfi].sigma_cumsum_fac[0].lo();
#if defined(WARPX_DIM_3D)
            int const y_lo = sigba[mfi].sigma_cumsum_fac[1].lo();
            int const z_lo = sigba[mfi].sigma_cumsum_fac[2].lo();
#else
            int const y_lo = 0;
            int const z_lo = sigba[mfi].sigma_cumsum_fac[1].lo();
#endif

            int const xs_lo = sigba[mfi].sigma_star_cumsum_fac[0].lo();
#if defined(WARPX_DIM_3D)
            int const ys_lo = sigba[mfi].sigma_star_cumsum_fac[1].lo();
            int const zs_lo = sigba[mfi].sigma_star_cumsum_fac[2].lo();
#else
            int const ys_lo = 0;
            int const zs_lo = sigba[mfi].sigma_star_cumsum_fac[1].lo();
#endif

            amrex::ParallelFor( tjx, tjy, tjz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (eb_lxfab && eb_lxfab(i, j, k) <= 0) { return; }

                    damp_jx_pml(i, j, k, pml_jxfab, sigma_star_cumsum_fac_j_x,
                                sigma_cumsum_fac_j_y, sigma_cumsum_fac_j_z,
                                xs_lo,y_lo, z_lo);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (eb_lyfab && eb_lyfab(i, j, k) <= 0) { return; }

                    damp_jy_pml(i, j, k, pml_jyfab, sigma_cumsum_fac_j_x,
                                sigma_star_cumsum_fac_j_y, sigma_cumsum_fac_j_z,
                                x_lo,ys_lo, z_lo);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (eb_lzfab && eb_lzfab(i, j, k) <= 0) { return; }

                    damp_jz_pml(i, j, k, pml_jzfab, sigma_cumsum_fac_j_x,
                                sigma_cumsum_fac_j_y, sigma_star_cumsum_fac_j_z,
                                x_lo,y_lo, zs_lo);
                }
            );
        }

    }
}

/**
 * \brief Copy the current J from the regular grid to the PML
 */
void
WarpX::CopyJPML ()
{
    using ablastr::fields::Direction;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (pml[lev] && pml[lev]->ok()){
            pml[lev]->CopyJtoPMLs(m_fields, lev);
        }
    }
}
