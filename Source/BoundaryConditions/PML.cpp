/* Copyright 2019-2022 Andrew Myers, Aurore Blelly, Axel Huebl,
 * Luca Fedeli, Maxence Thevenet, Remi Lehe, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PML.H"

#include "BoundaryConditions/PML.H"
#include "BoundaryConditions/PMLComponent.H"
#include "Fields.H"
#ifdef WARPX_USE_FFT
#   include "FieldSolver/SpectralSolver/SpectralFieldData.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>
#include <ablastr/utils/Enums.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FBI.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealVect.H>
#include <AMReX_SPACE.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#ifdef AMREX_USE_EB
#   include "AMReX_EBFabFactory.H"
#endif

using namespace amrex;
using warpx::fields::FieldType;

namespace
{
    void FillLo (Sigma& sigma, Sigma& sigma_cumsum,
                        Sigma& sigma_star, Sigma& sigma_star_cumsum,
                        const int olo, const int ohi, const int glo, Real fac,
                        const amrex::Real v_sigma)
    {
        const int slo = sigma.m_lo;
        const int sslo = sigma_star.m_lo;

        const int N = ohi+1-olo+1;
        Real* p_sigma = sigma.data();
        Real* p_sigma_cumsum = sigma_cumsum.data();
        Real* p_sigma_star = sigma_star.data();
        Real* p_sigma_star_cumsum = sigma_star_cumsum.data();
        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            i += olo;
            Real offset = static_cast<Real>(glo-i);
            p_sigma[i-slo] = fac*(offset*offset);
            // sigma_cumsum is the analytical integral of sigma function at same points than sigma
            p_sigma_cumsum[i-slo] = (fac*(offset*offset*offset)/3._rt)/v_sigma;
            if (i <= ohi+1) {
                offset = static_cast<Real>(glo-i) - 0.5_rt;
                p_sigma_star[i-sslo] = fac*(offset*offset);
                // sigma_star_cumsum is the analytical integral of sigma function at same points than sigma_star
                p_sigma_star_cumsum[i-sslo] = (fac*(offset*offset*offset)/3._rt)/v_sigma;
            }
        });
    }

    void FillHi (Sigma& sigma, Sigma& sigma_cumsum,
                        Sigma& sigma_star, Sigma& sigma_star_cumsum,
                        const int olo, const int ohi, const int ghi, Real fac,
                        const amrex::Real v_sigma)
    {
        const int slo = sigma.m_lo;
        const int sslo = sigma_star.m_lo;

        const int N = ohi+1-olo+1;
        Real* p_sigma = sigma.data();
        Real* p_sigma_cumsum = sigma_cumsum.data();
        Real* p_sigma_star = sigma_star.data();
        Real* p_sigma_star_cumsum = sigma_star_cumsum.data();
        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            i += olo;
            Real offset = static_cast<Real>(i-ghi-1);
            p_sigma[i-slo] = fac*(offset*offset);
            p_sigma_cumsum[i-slo] = (fac*(offset*offset*offset)/3._rt)/v_sigma;
            if (i <= ohi+1) {
                offset = static_cast<Real>(i-ghi) - 0.5_rt;
                p_sigma_star[i-sslo] = fac*(offset*offset);
                p_sigma_star_cumsum[i-sslo] = (fac*(offset*offset*offset)/3._rt)/v_sigma;
            }
        });
    }

#if (AMREX_SPACEDIM != 1)
    void FillZero (Sigma& sigma, Sigma& sigma_cumsum,
                          Sigma& sigma_star, Sigma& sigma_star_cumsum,
                          const int olo, const int ohi)
    {
        const int slo = sigma.m_lo;
        const int sslo = sigma_star.m_lo;

        const int N = ohi+1-olo+1;
        Real* p_sigma = sigma.data();
        Real* p_sigma_cumsum = sigma_cumsum.data();
        Real* p_sigma_star = sigma_star.data();
        Real* p_sigma_star_cumsum = sigma_star_cumsum.data();
        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            i += olo;
            p_sigma[i-slo] = Real(0.0);
            p_sigma_cumsum[i-slo] = Real(0.0);
            if (i <= ohi+1) {
                p_sigma_star[i-sslo] = Real(0.0);
                p_sigma_star_cumsum[i-sslo] = Real(0.0);
            }
        });
    }
#endif
}


SigmaBox::SigmaBox (const Box& box, const BoxArray& grids, const Real* dx, const IntVect& ncell,
                    const IntVect& delta, const amrex::Box& regdomain, const amrex::Real v_sigma_sb)
{
    BL_ASSERT(box.cellCentered());

    const IntVect& sz = box.size();
    const int*     lo = box.loVect();
    const int*     hi = box.hiVect();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        sigma                [idim].resize(sz[idim]+1,std::numeric_limits<Real>::quiet_NaN());
        sigma_cumsum         [idim].resize(sz[idim]+1,std::numeric_limits<Real>::quiet_NaN());
        sigma_star           [idim].resize(sz[idim]+1,std::numeric_limits<Real>::quiet_NaN());
        sigma_star_cumsum    [idim].resize(sz[idim]+1,std::numeric_limits<Real>::quiet_NaN());
        sigma_fac            [idim].resize(sz[idim]+1,std::numeric_limits<Real>::quiet_NaN());
        sigma_cumsum_fac     [idim].resize(sz[idim]+1,std::numeric_limits<Real>::quiet_NaN());
        sigma_star_fac       [idim].resize(sz[idim]+1,std::numeric_limits<Real>::quiet_NaN());
        sigma_star_cumsum_fac[idim].resize(sz[idim]+1,std::numeric_limits<Real>::quiet_NaN());

        sigma                [idim].m_lo = lo[idim];
        sigma                [idim].m_hi = hi[idim]+1;
        sigma_cumsum         [idim].m_lo = lo[idim];
        sigma_cumsum         [idim].m_hi = hi[idim]+1;
        sigma_star           [idim].m_lo = lo[idim];
        sigma_star           [idim].m_hi = hi[idim]+1;
        sigma_star_cumsum    [idim].m_lo = lo[idim];
        sigma_star_cumsum    [idim].m_hi = hi[idim]+1;
        sigma_fac            [idim].m_lo = lo[idim];
        sigma_fac            [idim].m_hi = hi[idim]+1;
        sigma_cumsum_fac     [idim].m_lo = lo[idim];
        sigma_cumsum_fac     [idim].m_hi = hi[idim]+1;
        sigma_star_fac       [idim].m_lo = lo[idim];
        sigma_star_fac       [idim].m_hi = hi[idim]+1;
        sigma_star_cumsum_fac[idim].m_lo = lo[idim];
        sigma_star_cumsum_fac[idim].m_hi = hi[idim]+1;
    }

    Array<Real,AMREX_SPACEDIM> fac;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        fac[idim] = 4.0_rt*PhysConst::c/(dx[idim]*static_cast<Real>(delta[idim]*delta[idim]));
    }

    if (regdomain.ok()) { // The union of the regular grids is a single box
        define_single(regdomain, ncell, fac, v_sigma_sb);
    } else {
        define_multiple(box, grids, ncell, fac, v_sigma_sb);
    }
}

void SigmaBox::define_single (const Box& regdomain, const IntVect& ncell,
                              const Array<Real,AMREX_SPACEDIM>& fac,
                              const amrex::Real v_sigma_sb)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const int slo = sigma[idim].lo();
        const int shi = sigma[idim].hi()-1;
        const int dlo = regdomain.smallEnd(idim);
        const int dhi = regdomain.bigEnd(idim);

        // Lo
        int olo = std::max(slo, dlo-ncell[idim]);
        int ohi = std::min(shi, dlo-1);
        if (ohi >= olo) {
            FillLo(sigma[idim], sigma_cumsum[idim],
                   sigma_star[idim], sigma_star_cumsum[idim],
                   olo, ohi, dlo, fac[idim], v_sigma_sb);
        }

#if (AMREX_SPACEDIM != 1)
        // Mid
        olo = std::max(slo, dlo);
        ohi = std::min(shi, dhi);
        if (ohi >= olo) {
            FillZero(sigma[idim], sigma_cumsum[idim],
                     sigma_star[idim], sigma_star_cumsum[idim],
                     olo, ohi);
        }
#endif

        // Hi
        olo = std::max(slo, dhi+1);
        ohi = std::min(shi, dhi+ncell[idim]);
        if (ohi >= olo) {
            FillHi(sigma[idim], sigma_cumsum[idim],
                   sigma_star[idim], sigma_star_cumsum[idim],
                   olo, ohi, dhi, fac[idim], v_sigma_sb);
        }
    }

    amrex::Gpu::streamSynchronize();
}

void SigmaBox::define_multiple (const Box& box, const BoxArray& grids, const IntVect& ncell,
                                const Array<Real,AMREX_SPACEDIM>& fac, const amrex::Real v_sigma_sb)
{
    const std::vector<std::pair<int,Box> >& isects = grids.intersections(box, false, ncell);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
#if (AMREX_SPACEDIM >= 2)
        const int jdim = (idim+1) % AMREX_SPACEDIM;
#endif
#if defined(WARPX_DIM_3D)
        const int kdim = (idim+2) % AMREX_SPACEDIM;
#endif

        Vector<int> direct_faces, side_faces, direct_side_edges, side_side_edges, corners;
        for (const auto& kv : isects)
        {
            const Box& grid_box = grids[kv.first];

            if (amrex::grow(grid_box, idim, ncell[idim]).intersects(box))
            {
                direct_faces.push_back(kv.first);
            }
#if (AMREX_SPACEDIM == 2)
            else if (amrex::grow(grid_box, jdim, ncell[jdim]).intersects(box))
            {
                side_faces.push_back(kv.first);
            }
            else
            {
                corners.push_back(kv.first);
            }
#elif defined(WARPX_DIM_3D)
            else if ((amrex::grow(grid_box, jdim, ncell[jdim]).intersects(box)) ||
                amrex::grow(grid_box, kdim, ncell[kdim]).intersects(box))
            {
                side_faces.push_back(kv.first);
            }
            else if (amrex::grow(amrex::grow(grid_box,idim,ncell[idim]),jdim,ncell[jdim]).intersects(box) ||
                     amrex::grow(amrex::grow(grid_box,idim,ncell[idim]),kdim,ncell[kdim]).intersects(box) )
            {
                direct_side_edges.push_back(kv.first);
            }
            else if (amrex::grow(amrex::grow(grid_box,jdim,ncell[jdim]),
                                 kdim,ncell[kdim]).intersects(box))
            {
                side_side_edges.push_back(kv.first);
            }
            else
            {
                corners.push_back(kv.first);
            }
#endif
        }

#if (AMREX_SPACEDIM >= 2)
        for (auto gid : corners)
        {
            const Box& grid_box = grids[gid];

            Box lobox = amrex::adjCellLo(grid_box, idim, ncell[idim]);
            lobox.grow(jdim,ncell[jdim]);
#if defined(WARPX_DIM_3D)
            lobox.grow(kdim,ncell[kdim]);
#endif
            const Box looverlap = lobox & box;

            if (looverlap.ok()) {
                FillLo(sigma[idim], sigma_cumsum[idim],
                       sigma_star[idim], sigma_star_cumsum[idim],
                       looverlap.smallEnd(idim), looverlap.bigEnd(idim),
                       grid_box.smallEnd(idim), fac[idim], v_sigma_sb);
            }

            Box hibox = amrex::adjCellHi(grid_box, idim, ncell[idim]);
            hibox.grow(jdim,ncell[jdim]);
#if defined(WARPX_DIM_3D)
            hibox.grow(kdim,ncell[kdim]);
#endif
            const Box hioverlap = hibox & box;
            if (hioverlap.ok()) {
                FillHi(sigma[idim], sigma_cumsum[idim],
                       sigma_star[idim],  sigma_star_cumsum[idim],
                       hioverlap.smallEnd(idim), hioverlap.bigEnd(idim),
                       grid_box.bigEnd(idim), fac[idim], v_sigma_sb);
            }

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                looverlap.ok() || hioverlap.ok(),
                "SigmaBox::SigmaBox(): corners, how did this happen?"
            );
        }
#endif

#if defined(WARPX_DIM_3D)
        for (auto gid : side_side_edges)
        {
            const Box& grid_box = grids[gid];
            const Box& overlap = amrex::grow(amrex::grow(grid_box,jdim,ncell[jdim]),kdim,ncell[kdim]) & box;

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                overlap.ok(),
                "SigmaBox::SigmaBox(): side_side_edges, how did this happen?"
            );

            FillZero(sigma[idim], sigma_cumsum[idim],
                sigma_star[idim], sigma_star_cumsum[idim],
                overlap.smallEnd(idim), overlap.bigEnd(idim));
        }

        for (auto gid : direct_side_edges)
        {
            const Box& grid_box = grids[gid];

            Box lobox = amrex::adjCellLo(grid_box, idim, ncell[idim]);
            const Box looverlap = lobox.grow(jdim,ncell[jdim]).grow(kdim,ncell[kdim]) & box;
            if (looverlap.ok()) {
                FillLo(sigma[idim], sigma_cumsum[idim],
                       sigma_star[idim],  sigma_star_cumsum[idim],
                       looverlap.smallEnd(idim), looverlap.bigEnd(idim),
                       grid_box.smallEnd(idim), fac[idim], v_sigma_sb);
            }

            Box hibox = amrex::adjCellHi(grid_box, idim, ncell[idim]);
            const Box hioverlap = hibox.grow(jdim,ncell[jdim]).grow(kdim,ncell[kdim]) & box;
            if (hioverlap.ok()) {
                FillHi(sigma[idim], sigma_cumsum[idim],
                       sigma_star[idim], sigma_star_cumsum[idim],
                       hioverlap.smallEnd(idim), hioverlap.bigEnd(idim),
                       grid_box.bigEnd(idim), fac[idim], v_sigma_sb);
            }

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                looverlap.ok() || hioverlap.ok(),
                "SigmaBox::SigmaBox(): direct_side_edges, how did this happen?"
            );
        }
#endif

#if (AMREX_SPACEDIM >= 2)
        for (auto gid : side_faces)
        {
            const Box& grid_box = grids[gid];
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            const Box& overlap = amrex::grow(grid_box,jdim,ncell[jdim]) & box;
#else
            const Box& overlap = amrex::grow(amrex::grow(grid_box,jdim,ncell[jdim]),kdim,ncell[kdim]) & box;
#endif

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                overlap.ok(),
                "SigmaBox::SigmaBox(): side_faces, how did this happen?"
            );

            FillZero(sigma[idim], sigma_cumsum[idim],
                sigma_star[idim], sigma_star_cumsum[idim],
                overlap.smallEnd(idim), overlap.bigEnd(idim));
        }
#endif

        for (auto gid : direct_faces)
        {
            const Box& grid_box = grids[gid];

            const Box& lobox = amrex::adjCellLo(grid_box, idim, ncell[idim]);
            const Box looverlap = lobox & box;
            if (looverlap.ok()) {
                FillLo(sigma[idim], sigma_cumsum[idim],
                       sigma_star[idim], sigma_star_cumsum[idim],
                       looverlap.smallEnd(idim), looverlap.bigEnd(idim),
                       grid_box.smallEnd(idim), fac[idim], v_sigma_sb);
            }

            const Box& hibox = amrex::adjCellHi(grid_box, idim, ncell[idim]);
            const Box hioverlap = hibox & box;
            if (hioverlap.ok()) {
                FillHi(sigma[idim], sigma_cumsum[idim],
                       sigma_star[idim], sigma_star_cumsum[idim],
                       hioverlap.smallEnd(idim), hioverlap.bigEnd(idim),
                       grid_box.bigEnd(idim), fac[idim], v_sigma_sb);
            }

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                looverlap.ok() || hioverlap.ok(),
                "SigmaBox::SigmaBox(): direct faces, how did this happen?"
            );

        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            direct_faces.size() <= 1,
            "SigmaBox::SigmaBox(): direct_faces.size() > 1, Box gaps not wide enough?"
        );
    }

    amrex::Gpu::streamSynchronize();
}


void
SigmaBox::ComputePMLFactorsB (const Real* a_dx, Real dt)
{
    GpuArray<Real*,AMREX_SPACEDIM> p_sigma_star_fac;
    GpuArray<Real*,AMREX_SPACEDIM> p_sigma_star_cumsum_fac;
    GpuArray<Real const*,AMREX_SPACEDIM> p_sigma_star;
    GpuArray<Real const*,AMREX_SPACEDIM> p_sigma_star_cumsum;
    GpuArray<int, AMREX_SPACEDIM> N;
    GpuArray<Real, AMREX_SPACEDIM> dx;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        p_sigma_star_fac[idim] = sigma_star_fac[idim].data();
        p_sigma_star_cumsum_fac[idim] = sigma_star_cumsum_fac[idim].data();
        p_sigma_star[idim] = sigma_star[idim].data();
        p_sigma_star_cumsum[idim] = sigma_star_cumsum[idim].data();
        N[idim] = static_cast<int>(sigma_star[idim].size());
        dx[idim] = a_dx[idim];
    }
    amrex::ParallelFor(
#if (AMREX_SPACEDIM >= 2)
        amrex::max(AMREX_D_DECL(N[0],N[1],N[2])),
#else
        N[0],
#endif
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (i < N[idim]) {
                p_sigma_star_fac[idim][i] = std::exp(-p_sigma_star[idim][i]*dt);
                p_sigma_star_cumsum_fac[idim][i] = std::exp(-p_sigma_star_cumsum[idim][i]*dx[idim]);
            }
        }
    });
}

void
SigmaBox::ComputePMLFactorsE (const Real* a_dx, Real dt)
{
    GpuArray<Real*,AMREX_SPACEDIM> p_sigma_fac;
    GpuArray<Real*,AMREX_SPACEDIM> p_sigma_cumsum_fac;
    GpuArray<Real const*,AMREX_SPACEDIM> p_sigma;
    GpuArray<Real const*,AMREX_SPACEDIM> p_sigma_cumsum;
    GpuArray<int, AMREX_SPACEDIM> N;
    GpuArray<Real, AMREX_SPACEDIM> dx;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        p_sigma_fac[idim] = sigma_fac[idim].data();
        p_sigma_cumsum_fac[idim] = sigma_cumsum_fac[idim].data();
        p_sigma[idim] = sigma[idim].data();
        p_sigma_cumsum[idim] = sigma_cumsum[idim].data();
        N[idim] = static_cast<int>(sigma[idim].size());
        dx[idim] = a_dx[idim];
    }
    amrex::ParallelFor(
#if (AMREX_SPACEDIM >= 2)
        amrex::max(AMREX_D_DECL(N[0],N[1],N[2])),
#else
        N[0],
#endif
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (i < N[idim]) {
                p_sigma_fac[idim][i] = std::exp(-p_sigma[idim][i]*dt);
                p_sigma_cumsum_fac[idim][i] = std::exp(-p_sigma_cumsum[idim][i]*dx[idim]);
            }
        }
    });
}

MultiSigmaBox::MultiSigmaBox (const BoxArray& ba, const DistributionMapping& dm,
                              const BoxArray& grid_ba, const Real* dx,
                              const IntVect& ncell, const IntVect& delta,
                              const amrex::Box& regular_domain, const amrex::Real v_sigma_sb)
    : FabArray<SigmaBox>(ba,dm,1,0,MFInfo(),
                         SigmaBoxFactory(grid_ba,dx,ncell,delta, regular_domain, v_sigma_sb))
{}

void
MultiSigmaBox::ComputePMLFactorsB (const Real* dx, Real dt)
{
    if (dt == dt_B) { return; }

    dt_B = dt;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*this); mfi.isValid(); ++mfi)
    {
        (*this)[mfi].ComputePMLFactorsB(dx, dt);
    }
}

void
MultiSigmaBox::ComputePMLFactorsE (const Real* dx, Real dt)
{
    if (dt == dt_E) { return; }

    dt_E = dt;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*this); mfi.isValid(); ++mfi)
    {
        (*this)[mfi].ComputePMLFactorsE(dx, dt);
    }
}

PML::PML (const int lev, const BoxArray& grid_ba,
          const DistributionMapping& grid_dm, const bool do_similar_dm_pml,
          const Geometry* geom, const Geometry* cgeom,
          int ncell, int delta, amrex::IntVect ref_ratio,
          Real dt, int nox_fft, int noy_fft, int noz_fft,
          ablastr::utils::enums::GridType grid_type,
          int do_moving_window, int /*pml_has_particles*/, int do_pml_in_domain,
          const PSATDSolutionType psatd_solution_type,
          const JInTime J_in_time, const RhoInTime rho_in_time,
          const bool do_pml_dive_cleaning, const bool do_pml_divb_cleaning,
          const amrex::IntVect& fill_guards_fields,
          const amrex::IntVect& fill_guards_current,
          bool eb_enabled,
          int max_guard_EB, const amrex::Real v_sigma_sb,
          const amrex::IntVect do_pml_Lo, const amrex::IntVect do_pml_Hi)
    : m_dive_cleaning(do_pml_dive_cleaning),
      m_divb_cleaning(do_pml_divb_cleaning),
      m_fill_guards_fields(fill_guards_fields),
      m_fill_guards_current(fill_guards_current),
      m_geom(geom),
      m_cgeom(cgeom)
{
#ifndef AMREX_USE_EB
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!eb_enabled, "PML: eb_enabled is true but was not compiled in.");
#endif

    using ablastr::fields::Direction;

    // When `do_pml_in_domain` is true, the PML overlap with the last `ncell` of the physical domain or fine patch(es)
    // (instead of extending `ncell` outside of the physical domain or fine patch(es))
    // In order to implement this, we define a new reduced Box Array ensuring that it does not
    // include ncells from the edges of the physical domain or fine patch.
    // (thus creating the PML boxes at the right position, where they overlap with the original domain or fine patch(es))

    BoxArray grid_ba_reduced = grid_ba;
    if (do_pml_in_domain) {
        BoxList bl = grid_ba.boxList();
        // Here we loop over all the boxes in the original grid_ba BoxArray
        // For each box, we find if its in the edge (or boundary), and the size of those boxes are decreased by ncell
        for (auto& b : bl) {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (do_pml_Lo[idim]) {
                    // Get neighboring box on lower side in direction idim and check if it intersects with any of the boxes
                    // in grid_ba. If no intersection, then the box, b, in the boxlist, is in the edge and we decrase
                    // the size by ncells using growLo(idim,-ncell)
                    Box const& bb = amrex::adjCellLo(b, idim);
                    if ( ! grid_ba.intersects(bb) ) {
                        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(b.length(idim) > ncell, " box length must be greater that pml size");
                        b.growLo(idim, -ncell);
                    }
                }
                if (do_pml_Hi[idim]) {
                    // Get neighboring box on higher side in direction idim and check if it intersects with any of the boxes
                    // in grid_ba. If no intersection, then the box, b, in the boxlist, is in the edge and we decrase
                    // the size by ncells using growHi(idim,-ncell)
                    Box const& bb = amrex::adjCellHi(b, idim);
                    if ( ! grid_ba.intersects(bb) ) {
                        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(b.length(idim) > ncell, " box length must be greater that pml size");
                        b.growHi(idim, -ncell);
                    }
                }
            }
        }
        grid_ba_reduced = BoxArray(std::move(bl));
    }
    Box const domain0 = grid_ba_reduced.minimalBox();
    const bool is_single_box_domain = domain0.numPts() == grid_ba_reduced.numPts();
    const BoxArray& ba = MakeBoxArray(is_single_box_domain, domain0, *geom, grid_ba_reduced,
                                      IntVect(ncell), do_pml_in_domain, do_pml_Lo, do_pml_Hi);


    if (ba.empty()) {
        m_ok = false;
        return;
    } else {
        m_ok = true;
    }
    // Define the number of guard cells in each di;rection, for E, B, and F
    auto nge = IntVect(AMREX_D_DECL(2, 2, 2));
    auto ngb = IntVect(AMREX_D_DECL(2, 2, 2));
    int ngf_int = 0;
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::CKC) {
        ngf_int = std::max( ngf_int, 1 );
    }
    auto ngf = IntVect(AMREX_D_DECL(ngf_int, ngf_int, ngf_int));

    if (do_moving_window) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev <= 1,
            "The number of grow cells for the moving window currently assumes 2 levels max.");
        const int rr = ref_ratio[WarpX::moving_window_dir];
        nge[WarpX::moving_window_dir] = std::max(nge[WarpX::moving_window_dir], rr);
        ngb[WarpX::moving_window_dir] = std::max(ngb[WarpX::moving_window_dir], rr);
        ngf[WarpX::moving_window_dir] = std::max(ngf[WarpX::moving_window_dir], rr);
    }

    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
        using namespace ablastr::utils::enums;

        // Increase the number of guard cells, in order to fit the extent
        // of the stencil for the spectral solver
        int ngFFt_x = (grid_type == GridType::Collocated) ? nox_fft : nox_fft/2;
        int ngFFt_y = (grid_type == GridType::Collocated) ? noy_fft : noy_fft/2;
        int ngFFt_z = (grid_type == GridType::Collocated) ? noz_fft : noz_fft/2;

        const ParmParse pp_psatd("psatd");
        utils::parser::queryWithParser(pp_psatd, "nx_guard", ngFFt_x);
        utils::parser::queryWithParser(pp_psatd, "ny_guard", ngFFt_y);
        utils::parser::queryWithParser(pp_psatd, "nz_guard", ngFFt_z);

#if defined(WARPX_DIM_3D)
        auto ngFFT = IntVect(ngFFt_x, ngFFt_y, ngFFt_z);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        auto ngFFT = IntVect(ngFFt_x, ngFFt_z);
#elif defined(WARPX_DIM_1D_Z)
        auto ngFFT = IntVect(ngFFt_z);
#endif

        // Set the number of guard cells to the maximum of each field
        // (all fields should have the same number of guard cells)
        ngFFT = ngFFT.max(nge);
        ngFFT = ngFFT.max(ngb);
        ngFFT = ngFFT.max(ngf);
        nge = ngFFT;
        ngb = ngFFT;
        ngf = ngFFT;
    }

    DistributionMapping dm;
    if (do_similar_dm_pml) {
        auto ng_sim = amrex::elemwiseMax(amrex::elemwiseMax(nge, ngb), ngf);
        dm = amrex::MakeSimilarDM(ba, grid_ba, grid_dm, ng_sim);
    } else {
        dm.define(ba);
    }

#ifdef AMREX_USE_EB
    if (eb_enabled) {
        pml_field_factory = amrex::makeEBFabFactory(
            *geom,
            ba,
            dm,
            {max_guard_EB, max_guard_EB, max_guard_EB},
            amrex::EBSupport::full
        );
    } else
#endif
    {
        amrex::ignore_unused(max_guard_EB);
        pml_field_factory = std::make_unique<FArrayBoxFactory>();
    }

    // Allocate diagonal components (xx,yy,zz) only with divergence cleaning
    const int ncompe = (m_dive_cleaning) ? 3 : 2;
    const int ncompb = (m_divb_cleaning) ? 3 : 2;

    auto& warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    const amrex::BoxArray ba_Ex = amrex::convert(ba, warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, 0)->ixType().toIntVect());
    const amrex::BoxArray ba_Ey = amrex::convert(ba, warpx.m_fields.get(FieldType::Efield_fp, Direction{1}, 0)->ixType().toIntVect());
    const amrex::BoxArray ba_Ez = amrex::convert(ba, warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, 0)->ixType().toIntVect());
    warpx.m_fields.alloc_init(FieldType::pml_E_fp, Direction{0}, lev, ba_Ex, dm, ncompe, nge, 0.0_rt, false, false);
    warpx.m_fields.alloc_init(FieldType::pml_E_fp, Direction{1}, lev, ba_Ey, dm, ncompe, nge, 0.0_rt, false, false);
    warpx.m_fields.alloc_init(FieldType::pml_E_fp, Direction{2}, lev, ba_Ez, dm, ncompe, nge, 0.0_rt, false, false);

    const amrex::BoxArray ba_Bx = amrex::convert(ba, warpx.m_fields.get(FieldType::Bfield_fp, Direction{0}, 0)->ixType().toIntVect());
    const amrex::BoxArray ba_By = amrex::convert(ba, warpx.m_fields.get(FieldType::Bfield_fp, Direction{1}, 0)->ixType().toIntVect());
    const amrex::BoxArray ba_Bz = amrex::convert(ba, warpx.m_fields.get(FieldType::Bfield_fp, Direction{2}, 0)->ixType().toIntVect());
    warpx.m_fields.alloc_init(FieldType::pml_B_fp, Direction{0}, lev, ba_Bx, dm, ncompb, ngb, 0.0_rt, false, false);
    warpx.m_fields.alloc_init(FieldType::pml_B_fp, Direction{1}, lev, ba_By, dm, ncompb, ngb, 0.0_rt, false, false);
    warpx.m_fields.alloc_init(FieldType::pml_B_fp, Direction{2}, lev, ba_Bz, dm, ncompb, ngb, 0.0_rt, false, false);

    const amrex::BoxArray ba_jx = amrex::convert(ba, WarpX::GetInstance().m_fields.get(FieldType::current_fp, Direction{0}, 0)->ixType().toIntVect());
    const amrex::BoxArray ba_jy = amrex::convert(ba, WarpX::GetInstance().m_fields.get(FieldType::current_fp, Direction{1}, 0)->ixType().toIntVect());
    const amrex::BoxArray ba_jz = amrex::convert(ba, WarpX::GetInstance().m_fields.get(FieldType::current_fp, Direction{2}, 0)->ixType().toIntVect());
    warpx.m_fields.alloc_init(FieldType::pml_j_fp, Direction{0}, lev, ba_jx, dm, 1, ngb, 0.0_rt, false, false);
    warpx.m_fields.alloc_init(FieldType::pml_j_fp, Direction{1}, lev, ba_jy, dm, 1, ngb, 0.0_rt, false, false);
    warpx.m_fields.alloc_init(FieldType::pml_j_fp, Direction{2}, lev, ba_jz, dm, 1, ngb, 0.0_rt, false, false);

#ifdef AMREX_USE_EB
    if (eb_enabled) {
        const amrex::IntVect max_guard_EB_vect = amrex::IntVect(max_guard_EB);
        warpx.m_fields.alloc_init(FieldType::pml_edge_lengths, Direction{0}, lev, ba_Ex, dm, WarpX::ncomps, max_guard_EB_vect, 0.0_rt, false, false);
        warpx.m_fields.alloc_init(FieldType::pml_edge_lengths, Direction{1}, lev, ba_Ey, dm, WarpX::ncomps, max_guard_EB_vect, 0.0_rt, false, false);
        warpx.m_fields.alloc_init(FieldType::pml_edge_lengths, Direction{2}, lev, ba_Ez, dm, WarpX::ncomps, max_guard_EB_vect, 0.0_rt, false, false);

        if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee ||
            WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::CKC ||
            WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::ECT) {

            auto const eb_fact = fieldEBFactory();

            ablastr::fields::VectorField t_pml_edge_lengths = warpx.m_fields.get_alldirs(FieldType::pml_edge_lengths, lev);
            WarpX::ComputeEdgeLengths(t_pml_edge_lengths, eb_fact);
            WarpX::ScaleEdges(t_pml_edge_lengths, WarpX::CellSize(lev));

        }
    }
#endif


    if (m_dive_cleaning)
    {
        const amrex::BoxArray ba_F_nodal = amrex::convert(ba, amrex::IntVect::TheNodeVector());
        warpx.m_fields.alloc_init(FieldType::pml_F_fp, lev, ba_F_nodal, dm, 3, ngf, 0.0_rt, false, false);
    }

    if (m_divb_cleaning)
    {
        // TODO Shall we define a separate guard cells parameter ngG?
        const amrex::IntVect& G_nodal_flag =
            (grid_type == GridType::Collocated) ? amrex::IntVect::TheNodeVector()
                                                : amrex::IntVect::TheCellVector();
        const amrex::BoxArray ba_G_nodal = amrex::convert(ba, G_nodal_flag);
        warpx.m_fields.alloc_init(FieldType::pml_G_fp, lev, ba_G_nodal, dm, 3, ngf, 0.0_rt, false, false);
    }

    Box single_domain_box = is_single_box_domain ? domain0 : Box();
    // Empty box (i.e., Box()) means it's not a single box domain.
    sigba_fp = std::make_unique<MultiSigmaBox>(ba, dm, grid_ba_reduced, geom->CellSize(),
                                               IntVect(ncell), IntVect(delta), single_domain_box, v_sigma_sb);

    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
#ifndef WARPX_USE_FFT
        amrex::ignore_unused(lev, dt, psatd_solution_type, J_in_time, rho_in_time);
#   if(AMREX_SPACEDIM!=3)
        amrex::ignore_unused(noy_fft);
#   endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
            "PML: PSATD solver selected but not built.");
#else
        // Flags passed to the spectral solver constructor
        const bool in_pml = true;
        const bool periodic_single_box = false;
        const bool update_with_rho = false;
        const bool fft_do_time_averaging = false;
        const RealVect dx{AMREX_D_DECL(geom->CellSize(0), geom->CellSize(1), geom->CellSize(2))};
        // Get the cell-centered box, with guard cells
        BoxArray realspace_ba = ba; // Copy box
        amrex::Vector<amrex::Real> const v_galilean = WarpX::GetInstance().m_v_galilean;
        amrex::Vector<amrex::Real> const v_comoving_zero = {0., 0., 0.};
        realspace_ba.enclosedCells().grow(nge); // cell-centered + guard cells
        spectral_solver_fp = std::make_unique<SpectralSolver>(lev, realspace_ba, dm,
            nox_fft, noy_fft, noz_fft, grid_type, v_galilean,
            v_comoving_zero, dx, dt, in_pml, periodic_single_box, update_with_rho,
            fft_do_time_averaging, psatd_solution_type, J_in_time, rho_in_time, m_dive_cleaning, m_divb_cleaning);
#endif
    }

    if (cgeom)
    {
        if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) {
            nge = IntVect(AMREX_D_DECL(1, 1, 1));
            ngb = IntVect(AMREX_D_DECL(1, 1, 1));
        }

        BoxArray grid_cba = grid_ba;
        grid_cba.coarsen(ref_ratio);

        BoxArray grid_cba_reduced = grid_cba;
        if (do_pml_in_domain) {
            BoxList bl = grid_cba.boxList();
            for (auto& b : bl) {
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    if (do_pml_Lo[idim]) {
                        Box const& bb = amrex::adjCellLo(b, idim);
                        if ( ! grid_cba.intersects(bb) ) {
                            b.growLo(idim, -ncell/ref_ratio[idim]);
                        }
                    }
                    if (do_pml_Hi[idim]) {
                        Box const& bb = amrex::adjCellHi(b, idim);
                        if ( ! grid_cba.intersects(bb) ) {
                            b.growHi(idim, -ncell/ref_ratio[idim]);
                        }
                    }
                }
            }
            grid_cba_reduced = BoxArray(std::move(bl));
        }
        Box const cdomain = grid_cba_reduced.minimalBox();

        const IntVect cncells = IntVect(ncell)/ref_ratio;
        const IntVect cdelta = IntVect(delta)/ref_ratio;

        // Assuming that refinement ratio is equal in all dimensions
        const BoxArray& cba = MakeBoxArray(is_single_box_domain, cdomain, *cgeom, grid_cba_reduced,
                                           cncells, do_pml_in_domain, do_pml_Lo, do_pml_Hi);
        DistributionMapping cdm;
        if (do_similar_dm_pml) {
            auto ng_sim = amrex::elemwiseMax(amrex::elemwiseMax(nge, ngb), ngf);
            cdm = amrex::MakeSimilarDM(cba, grid_cba_reduced, grid_dm, ng_sim);
        } else {
            cdm.define(cba);
        }

        const amrex::BoxArray cba_Ex = amrex::convert(cba, WarpX::GetInstance().m_fields.get(FieldType::Efield_cp, Direction{0}, 1)->ixType().toIntVect());
        const amrex::BoxArray cba_Ey = amrex::convert(cba, WarpX::GetInstance().m_fields.get(FieldType::Efield_cp, Direction{1}, 1)->ixType().toIntVect());
        const amrex::BoxArray cba_Ez = amrex::convert(cba, WarpX::GetInstance().m_fields.get(FieldType::Efield_cp, Direction{2}, 1)->ixType().toIntVect());
        warpx.m_fields.alloc_init(FieldType::pml_E_cp, Direction{0}, lev, cba_Ex, cdm, ncompe, nge, 0.0_rt, false, false);
        warpx.m_fields.alloc_init(FieldType::pml_E_cp, Direction{1}, lev, cba_Ey, cdm, ncompe, nge, 0.0_rt, false, false);
        warpx.m_fields.alloc_init(FieldType::pml_E_cp, Direction{2}, lev, cba_Ez, cdm, ncompe, nge, 0.0_rt, false, false);

        const amrex::BoxArray cba_Bx = amrex::convert(cba, WarpX::GetInstance().m_fields.get(FieldType::Bfield_cp, Direction{0}, 1)->ixType().toIntVect());
        const amrex::BoxArray cba_By = amrex::convert(cba, WarpX::GetInstance().m_fields.get(FieldType::Bfield_cp, Direction{1}, 1)->ixType().toIntVect());
        const amrex::BoxArray cba_Bz = amrex::convert(cba, WarpX::GetInstance().m_fields.get(FieldType::Bfield_cp, Direction{2}, 1)->ixType().toIntVect());
        warpx.m_fields.alloc_init(FieldType::pml_B_cp, Direction{0}, lev, cba_Bx, cdm, ncompb, ngb, 0.0_rt, false, false);
        warpx.m_fields.alloc_init(FieldType::pml_B_cp, Direction{1}, lev, cba_By, cdm, ncompb, ngb, 0.0_rt, false, false);
        warpx.m_fields.alloc_init(FieldType::pml_B_cp, Direction{2}, lev, cba_Bz, cdm, ncompb, ngb, 0.0_rt, false, false);

        if (m_dive_cleaning)
        {
            const amrex::BoxArray cba_F_nodal = amrex::convert(cba, amrex::IntVect::TheNodeVector());
            warpx.m_fields.alloc_init(FieldType::pml_F_cp, lev, cba_F_nodal, cdm, 3, ngf, 0.0_rt, false, false);
        }

        if (m_divb_cleaning)
        {
            // TODO Shall we define a separate guard cells parameter ngG?
            const amrex::IntVect& G_nodal_flag =
                (grid_type == GridType::Collocated) ? amrex::IntVect::TheNodeVector()
                                                    : amrex::IntVect::TheCellVector();
            const amrex::BoxArray cba_G_nodal = amrex::convert(cba, G_nodal_flag);
            warpx.m_fields.alloc_init(FieldType::pml_G_cp, lev, cba_G_nodal, cdm, 3, ngf, 0.0_rt, false, false);
        }

        const amrex::BoxArray cba_jx = amrex::convert(cba, WarpX::GetInstance().m_fields.get(FieldType::current_cp, Direction{0}, 1)->ixType().toIntVect());
        const amrex::BoxArray cba_jy = amrex::convert(cba, WarpX::GetInstance().m_fields.get(FieldType::current_cp, Direction{1}, 1)->ixType().toIntVect());
        const amrex::BoxArray cba_jz = amrex::convert(cba, WarpX::GetInstance().m_fields.get(FieldType::current_cp, Direction{2}, 1)->ixType().toIntVect());
        warpx.m_fields.alloc_init(FieldType::pml_j_cp, Direction{0}, lev, cba_jx, cdm, 1, ngb, 0.0_rt, false, false);
        warpx.m_fields.alloc_init(FieldType::pml_j_cp, Direction{1}, lev, cba_jy, cdm, 1, ngb, 0.0_rt, false, false);
        warpx.m_fields.alloc_init(FieldType::pml_j_cp, Direction{2}, lev, cba_jz, cdm, 1, ngb, 0.0_rt, false, false);

        single_domain_box = is_single_box_domain ? cdomain : Box();
        sigba_cp = std::make_unique<MultiSigmaBox>(cba, cdm, grid_cba_reduced, cgeom->CellSize(),
                                                   cncells, cdelta, single_domain_box, v_sigma_sb);

        if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
#ifndef WARPX_USE_FFT
            amrex::ignore_unused(dt);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                "PML: PSATD solver selected but not built.");
#else
            // Flags passed to the spectral solver constructor
            const bool in_pml = true;
            const bool periodic_single_box = false;
            const bool update_with_rho = false;
            const bool fft_do_time_averaging = false;
            const RealVect cdx{AMREX_D_DECL(cgeom->CellSize(0), cgeom->CellSize(1), cgeom->CellSize(2))};
            // Get the cell-centered box, with guard cells
            BoxArray realspace_cba = cba; // Copy box
            amrex::Vector<amrex::Real> const v_galilean = WarpX::GetInstance().m_v_galilean;
            amrex::Vector<amrex::Real> const v_comoving_zero = {0., 0., 0.};
            realspace_cba.enclosedCells().grow(nge); // cell-centered + guard cells
            spectral_solver_cp = std::make_unique<SpectralSolver>(lev, realspace_cba, cdm,
                nox_fft, noy_fft, noz_fft, grid_type, v_galilean,
                v_comoving_zero, cdx, dt, in_pml, periodic_single_box, update_with_rho,
                fft_do_time_averaging, psatd_solution_type, J_in_time, rho_in_time, m_dive_cleaning, m_divb_cleaning);
#endif
        }
    }
}

BoxArray
PML::MakeBoxArray (bool is_single_box_domain, const amrex::Box& regular_domain,
                   const amrex::Geometry& geom, const amrex::BoxArray& grid_ba,
                   const amrex::IntVect& ncell, int do_pml_in_domain,
                   const amrex::IntVect& do_pml_Lo, const amrex::IntVect& do_pml_Hi)
{
    if (is_single_box_domain) {
        return MakeBoxArray_single(regular_domain, grid_ba, ncell, do_pml_Lo, do_pml_Hi);
    } else { // the union of the regular grids is *not* a single rectangular domain
        return MakeBoxArray_multiple(geom, grid_ba, ncell, do_pml_in_domain, do_pml_Lo, do_pml_Hi);
    }
}

BoxArray
PML::MakeBoxArray_single (const amrex::Box& regular_domain, const amrex::BoxArray& grid_ba,
                          const amrex::IntVect& ncell, const amrex::IntVect& do_pml_Lo,
                          const amrex::IntVect& do_pml_Hi)
{
    BoxList bl;
    const auto grid_ba_size = static_cast<int>(grid_ba.size());
    for (int i = 0; i < grid_ba_size; ++i) {
        Box const& b = grid_ba[i];
        for (OrientationIter oit; oit.isValid(); ++oit) {
            // In 3d, a Box has 6 faces.  This iterates over the 6 faces.
            // 3 of them are on the lower side and the others are on the
            // higher side.
            const Orientation ori = oit();
            const int idim = ori.coordDir(); // either 0 or 1 or 2 (i.e., x, y, z-direction)
            bool pml_bndry = false;
            if (ori.isLow() && do_pml_Lo[idim]) {  // This is one of the lower side faces.
                pml_bndry = b.smallEnd(idim) == regular_domain.smallEnd(idim);
            } else if (ori.isHigh() && do_pml_Hi[idim]) { // This is one of the higher side faces.
                pml_bndry = b.bigEnd(idim) == regular_domain.bigEnd(idim);
            }
            if (pml_bndry) {
                Box bbox = amrex::adjCell(b, ori, ncell[idim]);
                for (int jdim = 0; jdim < idim; ++jdim) {
                    if (do_pml_Lo[jdim] &&
                        bbox.smallEnd(jdim) == regular_domain.smallEnd(jdim)) {
                        bbox.growLo(jdim, ncell[jdim]);
                    }
                    if (do_pml_Hi[jdim] &&
                        bbox.bigEnd(jdim) == regular_domain.bigEnd(jdim)) {
                        bbox.growHi(jdim, ncell[jdim]);
                    }
                }
                bl.push_back(bbox);
            }
        }
    }

    return BoxArray(std::move(bl));
}

BoxArray
PML::MakeBoxArray_multiple (const amrex::Geometry& geom, const amrex::BoxArray& grid_ba,
                            const amrex::IntVect& ncell, int do_pml_in_domain,
                            const amrex::IntVect& do_pml_Lo, const amrex::IntVect& do_pml_Hi)
{
    Box domain = geom.Domain();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (do_pml_Lo[idim]){
            domain.growLo(idim, ncell[idim]);
        }
        if (do_pml_Hi[idim]){
            domain.growHi(idim, ncell[idim]);
        }
    }
    BoxList bl;
    const auto grid_ba_size = static_cast<int>(grid_ba.size());
    for (int i = 0; i < grid_ba_size; ++i)
    {
        const Box& grid_bx = grid_ba[i];
        const IntVect& grid_bx_sz = grid_bx.size();

        if (do_pml_in_domain == 0) {
            // Make sure that, in the case of several distinct refinement patches,
            //  the PML cells surrounding these patches cannot overlap
            // The check is only needed along the axis where PMLs are being used.
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (do_pml_Lo[idim] || do_pml_Hi[idim]) {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        grid_bx.length(idim) > ncell[idim],
                        "Consider using larger amr.blocking_factor with PMLs");
                }
            }
        }

        Box bx = grid_bx;
        bx.grow(ncell);
        bx &= domain;

        Vector<Box> bndryboxes;
#if defined(WARPX_DIM_3D)
        const int kbegin = -1, kend = 1;
#else
        const int kbegin =  0, kend = 0;
#endif
        for (int kk = kbegin; kk <= kend; ++kk) {
            for (int jj = -1; jj <= 1; ++jj) {
                for (int ii = -1; ii <= 1; ++ii) {
                    if (ii != 0 || jj != 0 || kk != 0) {
                        Box b = grid_bx;
                        b.shift(grid_bx_sz * IntVect{AMREX_D_DECL(ii,jj,kk)});
                        b &= bx;
                        if (b.ok()) {
                            bndryboxes.push_back(b);
                        }
                    }
                }
            }
        }

        const BoxList& noncovered = grid_ba.complementIn(bx);
        for (const Box& b : noncovered) {
            for (const auto& bb : bndryboxes) {
                const Box ib = b & bb;
                if (ib.ok()) {
                    bl.push_back(ib);
                }
            }
        }
    }

    BoxArray ba(bl);
    ba.removeOverlap(false);

    return ba;
}

void
PML::ComputePMLFactors (amrex::Real dt)
{
    if (sigba_fp) {
        sigba_fp->ComputePMLFactorsB(m_geom->CellSize(), dt);
        sigba_fp->ComputePMLFactorsE(m_geom->CellSize(), dt);
    }
    if (sigba_cp) {
        sigba_cp->ComputePMLFactorsB(m_cgeom->CellSize(), dt);
        sigba_cp->ComputePMLFactorsE(m_cgeom->CellSize(), dt);
    }
}

void
PML::CopyJtoPMLs (
    ablastr::fields::MultiFabRegister& fields,
    PatchType patch_type,
    int lev
)
{
    using ablastr::fields::Direction;

    bool const has_j_fp = fields.has_vector(FieldType::current_fp, lev);
    bool const has_pml_j_fp = fields.has_vector(FieldType::pml_j_fp, lev);
    bool const has_j_cp = fields.has_vector(FieldType::current_cp, lev);
    bool const has_pml_j_cp = fields.has_vector(FieldType::pml_j_cp, lev);

    if (patch_type == PatchType::fine && has_pml_j_fp && has_j_fp)
    {
        ablastr::fields::VectorField pml_j_fp = fields.get_alldirs(FieldType::pml_j_fp, lev);
        ablastr::fields::VectorField jp = fields.get_alldirs(FieldType::current_fp, lev);
        CopyToPML(*pml_j_fp[0], *jp[0], *m_geom);
        CopyToPML(*pml_j_fp[1], *jp[1], *m_geom);
        CopyToPML(*pml_j_fp[2], *jp[2], *m_geom);
    }
    else if (patch_type == PatchType::coarse && has_j_cp && has_pml_j_cp)
    {
        ablastr::fields::VectorField pml_j_cp = fields.get_alldirs(FieldType::pml_j_cp, lev);
        ablastr::fields::VectorField jp = fields.get_alldirs(FieldType::current_cp, lev);
        CopyToPML(*pml_j_cp[0], *jp[0], *m_cgeom);
        CopyToPML(*pml_j_cp[1], *jp[1], *m_cgeom);
        CopyToPML(*pml_j_cp[2], *jp[2], *m_cgeom);
    }
}

void
PML::CopyJtoPMLs (
    ablastr::fields::MultiFabRegister& fields,
    int lev
)
{
    CopyJtoPMLs(fields, PatchType::fine, lev);
    CopyJtoPMLs(fields, PatchType::coarse, lev);
}

void PML::Exchange (ablastr::fields::VectorField mf_pml,
                    ablastr::fields::VectorField mf,
                    const PatchType& patch_type,
                    const int do_pml_in_domain)
{
    const amrex::Geometry& geom = (patch_type == PatchType::fine) ? *m_geom : *m_cgeom;
    if (mf_pml[0] && mf[0]) { Exchange(*mf_pml[0], *mf[0], geom, do_pml_in_domain); }
    if (mf_pml[1] && mf[1]) { Exchange(*mf_pml[1], *mf[1], geom, do_pml_in_domain); }
    if (mf_pml[2] && mf[2]) { Exchange(*mf_pml[2], *mf[2], geom, do_pml_in_domain); }
}

void PML::Exchange (amrex::MultiFab* mf_pml,
                    amrex::MultiFab* mf,
                    const PatchType& patch_type,
                    const int do_pml_in_domain)
{
    const amrex::Geometry& geom = (patch_type == PatchType::fine) ? *m_geom : *m_cgeom;
    if (mf_pml && mf) { Exchange(*mf_pml, *mf, geom, do_pml_in_domain); }
}

void
PML::Exchange (MultiFab& pml, MultiFab& reg, const Geometry& geom,
                int do_pml_in_domain)
{
    WARPX_PROFILE("PML::Exchange");

    const IntVect& ngr = reg.nGrowVect();
    const IntVect& ngp = pml.nGrowVect();
    const int ncp = pml.nComp();
    const auto& period = geom.periodicity();

    // Create temporary MultiFab to copy to and from the PML
    MultiFab tmpregmf(reg.boxArray(), reg.DistributionMap(), ncp, ngr);
    tmpregmf.setVal(0.0);

    // Create the sum of the split fields, in the PML
    MultiFab totpmlmf(pml.boxArray(), pml.DistributionMap(), 1, 0); // Allocate
    MultiFab::LinComb(totpmlmf, 1.0, pml, 0, 1.0, pml, 1, 0, 1, 0); // Sum
    if (ncp == 3) {
        MultiFab::Add(totpmlmf,pml,2,0,1,0); // Sum the third split component
    }

    // Copy from the sum of PML split field to valid cells of regular grid
    if (do_pml_in_domain){
        // Valid cells of the PML and of the regular grid overlap
        // Copy from valid cells of the PML to valid cells of the regular grid
        ablastr::utils::communication::ParallelCopy(reg, totpmlmf, 0, 0, 1, IntVect(0), IntVect(0),
                                                    WarpX::do_single_precision_comms,
                                                    period);
    } else {
        // Valid cells of the PML only overlap with guard cells of regular grid
        // (and outermost valid cell of the regular grid, for nodal direction)
        // Copy from valid cells of PML to ghost cells of regular grid
        // but avoid updating the outermost valid cell
        if (ngr.max() > 0) {
            MultiFab::Copy(tmpregmf, reg, 0, 0, 1, ngr);
            ablastr::utils::communication::ParallelCopy(tmpregmf, totpmlmf, 0, 0, 1, IntVect(0), ngr,
                                   WarpX::do_single_precision_comms,
                                                        period);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(reg); mfi.isValid(); ++mfi)
            {
                const FArrayBox& src = tmpregmf[mfi];
                FArrayBox& dst = reg[mfi];
                const auto srcarr = src.array();
                auto dstarr = dst.array();
                const BoxList& bl = amrex::boxDiff(dst.box(), mfi.validbox());
                // boxDiff avoids the outermost valid cell
                for (const Box& bx : bl) {
                    amrex::ParallelFor(bx,
                                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                                       {
                                           dstarr(i,j,k,0) = srcarr(i,j,k,0);
                                       });
                }
            }
        }
    }

    // Copy from valid cells of the regular grid to guard cells of the PML
    // (and outermost valid cell in the nodal direction)
    // More specifically, copy from regular data to PML's first component
    // Zero out the second (and third) component
    MultiFab::Copy(tmpregmf,reg,0,0,1,0); // Fill first component of tmpregmf
    tmpregmf.setVal(0.0, 1, ncp-1, 0); // Zero out the second (and third) component
    if (do_pml_in_domain){
        // Where valid cells of tmpregmf overlap with PML valid cells,
        // copy the PML (this is order to avoid overwriting PML valid cells,
        // in the next `ParallelCopy`)
        ablastr::utils::communication::ParallelCopy(tmpregmf, pml, 0, 0, ncp, IntVect(0), IntVect(0),
                                                    WarpX::do_single_precision_comms,
                                                    period);
    }
    ablastr::utils::communication::ParallelCopy(pml, tmpregmf, 0, 0, ncp, IntVect(0), ngp,
                                                WarpX::do_single_precision_comms, period);
}


void
PML::CopyToPML (MultiFab& pml, MultiFab& reg, const Geometry& geom)
{
  const IntVect& ngp = pml.nGrowVect();
  const auto& period = geom.periodicity();

    ablastr::utils::communication::ParallelCopy(pml, reg, 0, 0, 1, IntVect(0), ngp,
                                                WarpX::do_single_precision_comms, period);
}

void
PML::FillBoundary (ablastr::fields::VectorField mf_pml, PatchType patch_type, std::optional<bool> nodal_sync)
{
    const auto& period =
        (patch_type == PatchType::fine) ?
        m_geom->periodicity() :
        m_cgeom->periodicity();

    const Vector<MultiFab*> mf{mf_pml[0], mf_pml[1], mf_pml[2]};
    ablastr::utils::communication::FillBoundary(mf, WarpX::do_single_precision_comms, period, nodal_sync);
}

void
PML::FillBoundary (amrex::MultiFab & mf_pml, PatchType patch_type, std::optional<bool> nodal_sync)
{
    const auto& period =
        (patch_type == PatchType::fine) ?
        m_geom->periodicity() :
        m_cgeom->periodicity();

    ablastr::utils::communication::FillBoundary(mf_pml, WarpX::do_single_precision_comms, period, nodal_sync);
}

void
PML::CheckPoint (
    ablastr::fields::MultiFabRegister& fields,
    const std::string& dir
) const
{
    using ablastr::fields::Direction;

    if (fields.has_vector(FieldType::pml_E_fp, 0))
    {
        ablastr::fields::VectorField pml_E_fp = fields.get_alldirs(FieldType::pml_E_fp, 0);
        ablastr::fields::VectorField pml_B_fp = fields.get_alldirs(FieldType::pml_B_fp, 0);
        VisMF::AsyncWrite(*pml_E_fp[0], dir+"_Ex_fp");
        VisMF::AsyncWrite(*pml_E_fp[1], dir+"_Ey_fp");
        VisMF::AsyncWrite(*pml_E_fp[2], dir+"_Ez_fp");
        VisMF::AsyncWrite(*pml_B_fp[0], dir+"_Bx_fp");
        VisMF::AsyncWrite(*pml_B_fp[1], dir+"_By_fp");
        VisMF::AsyncWrite(*pml_B_fp[2], dir+"_Bz_fp");
    }

    if (fields.has_vector(FieldType::pml_E_cp, 0))
    {
        ablastr::fields::VectorField pml_E_cp = fields.get_alldirs(FieldType::pml_E_cp, 0);
        ablastr::fields::VectorField pml_B_cp = fields.get_alldirs(FieldType::pml_B_cp, 0);
        VisMF::AsyncWrite(*pml_E_cp[0], dir+"_Ex_cp");
        VisMF::AsyncWrite(*pml_E_cp[1], dir+"_Ey_cp");
        VisMF::AsyncWrite(*pml_E_cp[2], dir+"_Ez_cp");
        VisMF::AsyncWrite(*pml_B_cp[0], dir+"_Bx_cp");
        VisMF::AsyncWrite(*pml_B_cp[1], dir+"_By_cp");
        VisMF::AsyncWrite(*pml_B_cp[2], dir+"_Bz_cp");
    }
}

void
PML::Restart (
    ablastr::fields::MultiFabRegister& fields,
    const std::string& dir
)
{
    using ablastr::fields::Direction;

    if (fields.has_vector(FieldType::pml_E_fp, 0))
    {
        ablastr::fields::VectorField pml_E_fp = fields.get_alldirs(FieldType::pml_E_fp, 0);
        ablastr::fields::VectorField pml_B_fp = fields.get_alldirs(FieldType::pml_B_fp, 0);
        VisMF::Read(*pml_E_fp[0], dir+"_Ex_fp");
        VisMF::Read(*pml_E_fp[1], dir+"_Ey_fp");
        VisMF::Read(*pml_E_fp[2], dir+"_Ez_fp");
        VisMF::Read(*pml_B_fp[0], dir+"_Bx_fp");
        VisMF::Read(*pml_B_fp[1], dir+"_By_fp");
        VisMF::Read(*pml_B_fp[2], dir+"_Bz_fp");
    }

    if (fields.has_vector(FieldType::pml_E_cp, 0))
    {
        ablastr::fields::VectorField pml_E_cp = fields.get_alldirs(FieldType::pml_E_cp, 0);
        ablastr::fields::VectorField pml_B_cp = fields.get_alldirs(FieldType::pml_B_cp, 0);
        VisMF::Read(*pml_E_cp[0], dir+"_Ex_cp");
        VisMF::Read(*pml_E_cp[1], dir+"_Ey_cp");
        VisMF::Read(*pml_E_cp[2], dir+"_Ez_cp");
        VisMF::Read(*pml_B_cp[0], dir+"_Bx_cp");
        VisMF::Read(*pml_B_cp[1], dir+"_By_cp");
        VisMF::Read(*pml_B_cp[2], dir+"_Bz_cp");
    }
}

#ifdef WARPX_USE_FFT
void
PML::PushPSATD (ablastr::fields::MultiFabRegister& fields, const int lev)
{
    ablastr::fields::VectorField pml_E_fp = fields.get_alldirs(FieldType::pml_E_fp, lev);
    ablastr::fields::VectorField pml_B_fp = fields.get_alldirs(FieldType::pml_B_fp, lev);
    ablastr::fields::ScalarField pml_F_fp = fields.get(FieldType::pml_F_fp, lev);
    ablastr::fields::ScalarField pml_G_fp = fields.get(FieldType::pml_G_fp, lev);

    // Update the fields on the fine and coarse patch
    PushPMLPSATDSinglePatch(lev, *spectral_solver_fp, pml_E_fp, pml_B_fp, pml_F_fp, pml_G_fp, m_fill_guards_fields);
    if (spectral_solver_cp) {
        ablastr::fields::VectorField pml_E_cp = fields.get_alldirs(FieldType::pml_E_cp, lev);
        ablastr::fields::VectorField pml_B_cp = fields.get_alldirs(FieldType::pml_B_cp, lev);
        ablastr::fields::ScalarField pml_F_cp = fields.get(FieldType::pml_F_cp, lev);
        ablastr::fields::ScalarField pml_G_cp = fields.get(FieldType::pml_G_cp, lev);
        PushPMLPSATDSinglePatch(lev, *spectral_solver_cp, pml_E_cp, pml_B_cp, pml_F_cp, pml_G_cp, m_fill_guards_fields);
    }
}

void
PushPMLPSATDSinglePatch (
    const int lev,
    SpectralSolver& solver,
    ablastr::fields::VectorField& pml_E,
    ablastr::fields::VectorField& pml_B,
    ablastr::fields::ScalarField pml_F,
    ablastr::fields::ScalarField pml_G,
    const amrex::IntVect& fill_guards)
{
    const SpectralFieldIndex& Idx = solver.m_spectral_index;

    // Perform forward Fourier transforms
    solver.ForwardTransform(lev, *pml_E[0], Idx.Exy, PMLComp::xy);
    solver.ForwardTransform(lev, *pml_E[0], Idx.Exz, PMLComp::xz);
    solver.ForwardTransform(lev, *pml_E[1], Idx.Eyx, PMLComp::yx);
    solver.ForwardTransform(lev, *pml_E[1], Idx.Eyz, PMLComp::yz);
    solver.ForwardTransform(lev, *pml_E[2], Idx.Ezx, PMLComp::zx);
    solver.ForwardTransform(lev, *pml_E[2], Idx.Ezy, PMLComp::zy);
    solver.ForwardTransform(lev, *pml_B[0], Idx.Bxy, PMLComp::xy);
    solver.ForwardTransform(lev, *pml_B[0], Idx.Bxz, PMLComp::xz);
    solver.ForwardTransform(lev, *pml_B[1], Idx.Byx, PMLComp::yx);
    solver.ForwardTransform(lev, *pml_B[1], Idx.Byz, PMLComp::yz);
    solver.ForwardTransform(lev, *pml_B[2], Idx.Bzx, PMLComp::zx);
    solver.ForwardTransform(lev, *pml_B[2], Idx.Bzy, PMLComp::zy);

    // WarpX::do_pml_dive_cleaning = true
    if (pml_F)
    {
        solver.ForwardTransform(lev, *pml_E[0], Idx.Exx, PMLComp::xx);
        solver.ForwardTransform(lev, *pml_E[1], Idx.Eyy, PMLComp::yy);
        solver.ForwardTransform(lev, *pml_E[2], Idx.Ezz, PMLComp::zz);
        solver.ForwardTransform(lev, *pml_F, Idx.Fx, PMLComp::x);
        solver.ForwardTransform(lev, *pml_F, Idx.Fy, PMLComp::y);
        solver.ForwardTransform(lev, *pml_F, Idx.Fz, PMLComp::z);
    }

    // WarpX::do_pml_divb_cleaning = true
    if (pml_G)
    {
        solver.ForwardTransform(lev, *pml_B[0], Idx.Bxx, PMLComp::xx);
        solver.ForwardTransform(lev, *pml_B[1], Idx.Byy, PMLComp::yy);
        solver.ForwardTransform(lev, *pml_B[2], Idx.Bzz, PMLComp::zz);
        solver.ForwardTransform(lev, *pml_G, Idx.Gx, PMLComp::x);
        solver.ForwardTransform(lev, *pml_G, Idx.Gy, PMLComp::y);
        solver.ForwardTransform(lev, *pml_G, Idx.Gz, PMLComp::z);
    }

    // Advance fields in spectral space
    solver.pushSpectralFields();

    // Perform backward Fourier transforms
    solver.BackwardTransform(lev, *pml_E[0], Idx.Exy, fill_guards, PMLComp::xy);
    solver.BackwardTransform(lev, *pml_E[0], Idx.Exz, fill_guards, PMLComp::xz);
    solver.BackwardTransform(lev, *pml_E[1], Idx.Eyx, fill_guards, PMLComp::yx);
    solver.BackwardTransform(lev, *pml_E[1], Idx.Eyz, fill_guards, PMLComp::yz);
    solver.BackwardTransform(lev, *pml_E[2], Idx.Ezx, fill_guards, PMLComp::zx);
    solver.BackwardTransform(lev, *pml_E[2], Idx.Ezy, fill_guards, PMLComp::zy);
    solver.BackwardTransform(lev, *pml_B[0], Idx.Bxy, fill_guards, PMLComp::xy);
    solver.BackwardTransform(lev, *pml_B[0], Idx.Bxz, fill_guards, PMLComp::xz);
    solver.BackwardTransform(lev, *pml_B[1], Idx.Byx, fill_guards, PMLComp::yx);
    solver.BackwardTransform(lev, *pml_B[1], Idx.Byz, fill_guards, PMLComp::yz);
    solver.BackwardTransform(lev, *pml_B[2], Idx.Bzx, fill_guards, PMLComp::zx);
    solver.BackwardTransform(lev, *pml_B[2], Idx.Bzy, fill_guards, PMLComp::zy);

    // WarpX::do_pml_dive_cleaning = true
    if (pml_F)
    {
        solver.BackwardTransform(lev, *pml_E[0], Idx.Exx, fill_guards, PMLComp::xx);
        solver.BackwardTransform(lev, *pml_E[1], Idx.Eyy, fill_guards, PMLComp::yy);
        solver.BackwardTransform(lev, *pml_E[2], Idx.Ezz, fill_guards, PMLComp::zz);
        solver.BackwardTransform(lev, *pml_F, Idx.Fx, fill_guards, PMLComp::x);
        solver.BackwardTransform(lev, *pml_F, Idx.Fy, fill_guards, PMLComp::y);
        solver.BackwardTransform(lev, *pml_F, Idx.Fz, fill_guards, PMLComp::z);
    }

    // WarpX::do_pml_divb_cleaning = true
    if (pml_G)
    {
        solver.BackwardTransform(lev, *pml_B[0], Idx.Bxx, fill_guards, PMLComp::xx);
        solver.BackwardTransform(lev, *pml_B[1], Idx.Byy, fill_guards, PMLComp::yy);
        solver.BackwardTransform(lev, *pml_B[2], Idx.Bzz, fill_guards, PMLComp::zz);
        solver.BackwardTransform(lev, *pml_G, Idx.Gx, fill_guards, PMLComp::x);
        solver.BackwardTransform(lev, *pml_G, Idx.Gy, fill_guards, PMLComp::y);
        solver.BackwardTransform(lev, *pml_G, Idx.Gz, fill_guards, PMLComp::z);
    }
}
#endif
