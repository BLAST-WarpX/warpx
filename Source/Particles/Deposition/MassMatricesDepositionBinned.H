/* Copyright 2026 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_MASS_MATRICES_DEPOSITION_BINNED_H_
#define WARPX_MASS_MATRICES_DEPOSITION_BINNED_H_

#include "Particles/Deposition/MassMatricesDeposition.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/ShapeFactors.H"
#include "Utils/ParticleUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_DenseBins.H>
#include <AMReX_Dim3.H>
#include <AMReX_GpuBuffer.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cstddef>
#include <type_traits>

#if defined(WARPX_DIM_XZ) && (defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP))

/**
 * Binned ("gather") deposition of the full Villasenor mass matrices in 2D (XZ) on GPU.
 *
 * The standard kernel (doVillasenorSigmaDeposition) loops over particles and scatter-adds
 * ~400 (node, component) entries per particle with global atomics. With many particles per
 * cell, this is limited by atomic throughput, and by the FP64 work of forming
 * fp*weight_J*weight_E for each entry.
 *
 * Here, particles are binned by the cell in which their orbit segment lies (cells being
 * shifted by half a cell for even shape orders, matching the definition of the segments in
 * doVillasenorJandSigmaDepositionKernel). All particles of a bin whose orbit does not cross
 * a cell boundary (one segment) deposit to exactly the same set of (node, component) entries
 * of the mass matrices. One thread block is launched per bin and each thread owns one entry:
 * it loops over the particles of the bin and accumulates its entry in a register, from
 * per-particle products fp*weight_J and weights weight_E staged in shared memory. Only one
 * atomic add per entry and bin is needed at the end.
 *
 * Particles whose orbit crosses a cell boundary (typically a small fraction) are collected
 * in a list and deposited afterwards with the standard kernel, so that they do not cause
 * warp divergence in the binned kernel.
 *
 * The weights are computed with the same expressions as in the standard kernel, so that
 * the deposited values only differ by the order of the floating-point summations.
 */
namespace MassMatricesBinned
{
    /** One (node, component) entry of the mass matrices for a single-segment particle */
    struct Entry
    {
        int m;      //!< mass matrix index: 0 xx, 1 xy, 2 xz, 3 yx, 4 yy, 5 yz, 6 zx, 7 zy, 8 zz
        int i;      //!< node offset along x, relative to the bin cell
        int k;      //!< node offset along z, relative to the bin cell
        int pj_off; //!< offset of fp*weight_J in the per-particle record
        int we_off; //!< offset of weight_E in the per-particle record
        int comp;   //!< component of the mass matrix
    };

    /** Layout of the per-particle record staged in shared memory (in units of amrex::Real) */
    template <int depos_order>
    struct RecordLayout
    {
        //! number of cellX_nodeZ weights (rows of Jx / columns of Ex)
        static constexpr int NCN = depos_order*(depos_order+1);
        //! number of nodeX_nodeZ weights (rows of Jy / columns of Ey)
        static constexpr int NNN = (depos_order+1)*(depos_order+1);
        //! number of nodeX_cellZ weights (rows of Jz / columns of Ez)
        static constexpr int NNC = (depos_order+1)*depos_order;
        static constexpr int NW = NCN + NNN + NNC;
        // fp*weight_J for the 9 mass matrices, then the 3 sets of weight_E
        static constexpr int pj_xx = 0;
        static constexpr int pj_xy = NCN;
        static constexpr int pj_xz = 2*NCN;
        static constexpr int pj_yx = 3*NCN;
        static constexpr int pj_yy = 3*NCN + NNN;
        static constexpr int pj_yz = 3*NCN + 2*NNN;
        static constexpr int pj_zx = 3*NCN + 3*NNN;
        static constexpr int pj_zy = 3*NCN + 3*NNN + NNC;
        static constexpr int pj_zz = 3*NCN + 3*NNN + 2*NNC;
        static constexpr int we_cn = 3*NW;
        static constexpr int we_nn = 3*NW + NCN;
        static constexpr int we_nc = 3*NW + NCN + NNN;
        static constexpr int size = 4*NW;
        //! offset of the first node of the stencil relative to the bin cell
        static constexpr int node_offset = -((depos_order-1)/2);
        //! number of (node, component) entries deposited by a single-segment particle
        static constexpr int n_entries = (depos_order == 1) ? 56 : (depos_order == 2) ? 375 : 1348;
        //! threads per block of the deposition kernel
        static constexpr int threads_per_block = (depos_order == 1) ? 64 : 256;
        //! minimum number of blocks per SM (launch bound, which limits the number of
        //! registers per thread so that the staging of one block can overlap with
        //! the accumulation of another block on the same SM)
        static constexpr int min_blocks_per_sm = 2;
        //! number of entries accumulated by each thread
        static constexpr int entries_per_thread =
            (n_entries + threads_per_block - 1) / threads_per_block;
    };

    /**
     * \brief Build the list of (node, component) entries deposited by a single-segment
     * particle, replicating the loop structure of doVillasenorJandSigmaDepositionKernel
     * with num_segments = 1.
     */
    template <int depos_order>
    amrex::Vector<Entry> makeEntries (const int max_grid_crossings)
    {
        using L = RecordLayout<depos_order>;
        amrex::Vector<Entry> entries;

        const int Ncomp_base = 2*depos_order + 2*max_grid_crossings;
        const int Ncomp_xx0 = Ncomp_base - 1;
        const int Ncomp_xy0 = Ncomp_base;
        const int Ncomp_xz0 = Ncomp_base;
        const int Ncomp_yx0 = Ncomp_base;
        const int Ncomp_yy0 = 1 + Ncomp_base;
        const int Ncomp_yz0 = 1 + Ncomp_base;
        const int Ncomp_zx0 = Ncomp_base;
        const int Ncomp_zy0 = 1 + Ncomp_base;
        const int Ncomp_zz0 = 1 + Ncomp_base;

        const int width_xx1 = depos_order + max_grid_crossings;
        const int width_yy1 = width_xx1;
        const int width_zz1 = width_xx1 - 1;

        // With a single segment, the segment shifts are equal to max_grid_crossings
        const int SegShiftX = max_grid_crossings;
        const int SegShiftZ = max_grid_crossings;

        auto cn = [](int i, int k) { return i*(depos_order+1) + k; };
        auto nn = [](int i, int k) { return i*(depos_order+1) + k; };
        auto nc = [](int i, int k) { return i*depos_order + k; };

        // Sxx, Sxz, Sxy
        for (int i = 0; i <= depos_order - 1; i++) {
            for (int k = 0; k <= depos_order; k++) {
                for (int kE = 0; kE <= depos_order; kE++) {
                    const int row_xx = depos_order - k + kE + SegShiftZ;
                    const int above_diag = (row_xx > width_xx1) ? 1 : 0;
                    for (int iE = 0; iE <= depos_order - 1; iE++) {
                        const int col_xx = depos_order - 1 - i + iE + SegShiftX;
                        if (col_xx > Ncomp_xx0 - row_xx - above_diag) { break; }
                        entries.push_back({0, i, k, L::pj_xx + cn(i,k), L::we_cn + cn(iE,kE),
                                           col_xx + Ncomp_xx0*row_xx});
                    }
                }
                for (int iE = 0; iE <= depos_order; iE++) {
                    for (int kE = 0; kE <= depos_order - 1; kE++) {
                        entries.push_back({2, i, k, L::pj_xz + cn(i,k), L::we_nc + nc(iE,kE),
                                           depos_order - 1 - i + iE + SegShiftX
                                           + Ncomp_xz0*(depos_order - k + kE + SegShiftZ)});
                    }
                }
                for (int iE = 0; iE <= depos_order; iE++) {
                    for (int kE = 0; kE <= depos_order; kE++) {
                        entries.push_back({1, i, k, L::pj_xy + cn(i,k), L::we_nn + nn(iE,kE),
                                           depos_order - 1 - i + iE + SegShiftX
                                           + Ncomp_xy0*(depos_order - k + kE + SegShiftZ)});
                    }
                }
            }
        }

        // Szx, Szz, Szy
        for (int i = 0; i <= depos_order; i++) {
            for (int k = 0; k <= depos_order - 1; k++) {
                for (int iE = 0; iE <= depos_order - 1; iE++) {
                    for (int kE = 0; kE <= depos_order; kE++) {
                        entries.push_back({6, i, k, L::pj_zx + nc(i,k), L::we_cn + cn(iE,kE),
                                           depos_order - i + iE + SegShiftX
                                           + Ncomp_zx0*(depos_order - 1 - k + kE + SegShiftZ)});
                    }
                }
                for (int kE = 0; kE <= depos_order - 1; kE++) {
                    const int row_zz = depos_order - 1 - k + kE + SegShiftZ;
                    const int above_diag = (row_zz > width_zz1) ? 1 : 0;
                    for (int iE = 0; iE <= depos_order; iE++) {
                        const int col_zz = depos_order - i + iE + SegShiftX;
                        if (col_zz > Ncomp_zz0 - 2 - row_zz - above_diag) { break; }
                        entries.push_back({8, i, k, L::pj_zz + nc(i,k), L::we_nc + nc(iE,kE),
                                           col_zz + Ncomp_zz0*row_zz});
                    }
                }
                for (int iE = 0; iE <= depos_order; iE++) {
                    for (int kE = 0; kE <= depos_order; kE++) {
                        entries.push_back({7, i, k, L::pj_zy + nc(i,k), L::we_nn + nn(iE,kE),
                                           depos_order - i + iE + SegShiftX
                                           + Ncomp_zy0*(depos_order - 1 - k + kE + SegShiftZ)});
                    }
                }
            }
        }

        // Syx, Syz, Syy
        for (int i = 0; i <= depos_order; i++) {
            for (int k = 0; k <= depos_order; k++) {
                for (int iE = 0; iE <= depos_order - 1; iE++) {
                    for (int kE = 0; kE <= depos_order; kE++) {
                        entries.push_back({3, i, k, L::pj_yx + nn(i,k), L::we_cn + cn(iE,kE),
                                           depos_order - i + iE + SegShiftX
                                           + Ncomp_yx0*(depos_order - k + kE + SegShiftZ)});
                    }
                }
                for (int iE = 0; iE <= depos_order; iE++) {
                    for (int kE = 0; kE <= depos_order - 1; kE++) {
                        entries.push_back({5, i, k, L::pj_yz + nn(i,k), L::we_nc + nc(iE,kE),
                                           depos_order - i + iE + SegShiftX
                                           + Ncomp_yz0*(depos_order - k + kE + SegShiftZ)});
                    }
                }
                for (int kE = 0; kE <= depos_order; kE++) {
                    const int row_yy = depos_order - k + kE + SegShiftZ;
                    const int above_diag = (row_yy > width_yy1) ? 1 : 0;
                    for (int iE = 0; iE <= depos_order; iE++) {
                        const int col_yy = depos_order - i + iE + SegShiftX;
                        if (col_yy > Ncomp_yy0 - 1 - row_yy - above_diag) { break; }
                        entries.push_back({4, i, k, L::pj_yy + nn(i,k), L::we_nn + nn(iE,kE),
                                           col_yy + Ncomp_yy0*row_yy});
                    }
                }
            }
        }

        return entries;
    }

    /**
     * \brief Index of the bin of a particle: the cell containing the particle, with cells
     * shifted by half a cell for even shape orders (matching the definition of the orbit
     * segments in doVillasenorJandSigmaDepositionKernel). Bins are numbered with x fastest.
     */
    template <int depos_order>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int binIndex (const amrex::ParticleReal xp, const amrex::ParticleReal zp,
                  const amrex::XDim3& dinv, const amrex::XDim3& xyzmin,
                  const int nbins_x, const int nbins_z)
    {
        constexpr double shift = ((depos_order % 2) == 0) ? 0.5 : 0.0;
        const double x = (xp - xyzmin.x)*dinv.x;
        const double z = (zp - xyzmin.z)*dinv.z;
        int ib = static_cast<int>(x - shift);
        int kb = static_cast<int>(z - shift);
        ib = amrex::max(0, amrex::min(ib, nbins_x-1));
        kb = amrex::max(0, amrex::min(kb, nbins_z-1));
        return ib + kb*nbins_x;
    }

    enum StageResult : int { staged = 0, skipped = 1, general = 2 };

    /**
     * \brief Compute the per-particle record (fp*weight_J and weight_E) of a particle,
     * if its orbit lies in a single segment inside the bin cell (ib, kb).
     *
     * \return staged if the record was filled, skipped if the particle does not deposit
     *         (suborbits), general if the particle must be deposited with the standard kernel.
     */
    template <int depos_order, bool has_exteb>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    int stageParticle (const long ip, const int ib, const int kb,
                       const amrex::ParticleReal* xp_n_data,
                       const amrex::ParticleReal* zp_n_data,
                       const GetParticlePosition<PIdx>& GetPosition,
                       const int* nsuborbits,
                       const amrex::ParticleReal* wp,
                       const amrex::ParticleReal* uxp_n,
                       const amrex::ParticleReal* uyp_n,
                       const amrex::ParticleReal* uzp_n,
                       const amrex::ParticleReal* uxp_nph,
                       const amrex::ParticleReal* uyp_nph,
                       const amrex::ParticleReal* uzp_nph,
                       GetExternalEBField const & getExternalEB,
                       const amrex::ParticleReal Bx_ext,
                       const amrex::ParticleReal By_ext,
                       const amrex::ParticleReal Bz_ext,
                       const amrex::Array4<amrex::Real const>& Bx_arr,
                       const amrex::Array4<amrex::Real const>& By_arr,
                       const amrex::Array4<amrex::Real const>& Bz_arr,
                       const amrex::IndexType Bx_type,
                       const amrex::IndexType By_type,
                       const amrex::IndexType Bz_type,
                       const amrex::Real dt,
                       const amrex::XDim3& dinv,
                       const amrex::XDim3& xyzmin,
                       const amrex::GpuArray<amrex::GpuArray<double,2>,
                                             AMREX_SPACEDIM>& domain_double,
                       const amrex::GpuArray<amrex::GpuArray<bool,2>, AMREX_SPACEDIM> & do_cropping,
                       const amrex::Dim3 lo,
                       const amrex::Real qs,
                       const amrex::Real ms,
                       const amrex::Real invvol,
                       amrex::Real* rec)
    {
        using namespace amrex::literals;
        using L = RecordLayout<depos_order>;

        amrex::Real constexpr one_third = 1.0_rt / 3.0_rt;
        amrex::Real constexpr one_sixth = 1.0_rt / 6.0_rt;

        // Skip mass matrix deposition for particles with suborbits.
        if (nsuborbits && nsuborbits[ip] > 1) { return skipped; }

        amrex::ParticleReal xp_nph, yp_nph, zp_nph;
        GetPosition(ip, xp_nph, yp_nph, zp_nph);

        amrex::ParticleReal const xp_n = xp_n_data[ip];
        amrex::ParticleReal const zp_n = zp_n_data[ip];

        // Compute position at time n + 1
        amrex::ParticleReal const xp_np1 = 2._prt*xp_nph - xp_n;
        amrex::ParticleReal const zp_np1 = 2._prt*zp_nph - zp_n;

        // Positions in grid units, and cell crossings, as in the standard kernel
        double x_new = (xp_np1 - xyzmin.x)*dinv.x;
        double x_old = (xp_n - xyzmin.x)*dinv.x;
        double z_new = (zp_np1 - xyzmin.z)*dinv.z;
        double z_old = (zp_n - xyzmin.z)*dinv.z;

        // Total change in particle position (before cropping)
        const double dxp = x_new - x_old;
        const double dzp = z_new - z_old;

        // Crop particle orbits at absorbing domain boundaries
        ParticleUtils::crop_at_boundary(x_old, z_old, x_new, z_new,
                                        domain_double[0][0], domain_double[0][1], do_cropping[0]);
        ParticleUtils::crop_at_boundary(z_old, x_old, z_new, x_new,
                                        domain_double[1][0], domain_double[1][1], do_cropping[1]);

        // cell crossings are defined at cell edges if depos_order is odd
        // cell crossings are defined at cell centers if depos_order is even
        double shift = 0.0;
        if ( (depos_order % 2) == 0 ) { shift = 0.5; }

        const auto i_old = static_cast<int>(x_old-shift);
        const auto i_new = static_cast<int>(x_new-shift);
        const auto k_old = static_cast<int>(z_old-shift);
        const auto k_new = static_cast<int>(z_new-shift);

        // Particles crossing a cell boundary, or (after cropping) not in the bin cell,
        // are deposited with the standard kernel.
        if (i_old != i_new || k_old != k_new || i_old != ib || k_old != kb) { return general; }

        // Segment factors (equal to 1 unless the orbit was cropped)
        const double dxp_seg = x_new - x_old;
        const double dzp_seg = z_new - z_old;
        const auto seg_factor_x = static_cast<amrex::Real>(dxp == 0. ? 1._rt : dxp_seg/dxp);
        const auto seg_factor_z = static_cast<amrex::Real>(dzp == 0. ? 1._rt : dzp_seg/dzp);
        const auto seg_factor_y = std::min(seg_factor_x,seg_factor_z);

        // Cell-based weights using the average segment position
        const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
        const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
        double sx_cell[depos_order] = {0.};
        double sz_cell[depos_order] = {0.};
        double const x0_bar = (x_new + x_old)/2.0;
        double const z0_bar = (z_new + z_old)/2.0;
        const int i0_cell = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );
        const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

        if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
            const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
            double sx_old_cell[depos_order] = {0.};
            double sx_new_cell[depos_order] = {0.};
            double sz_old_cell[depos_order] = {0.};
            double sz_new_cell[depos_order] = {0.};
            const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell,
                                                              x_old-0.5, x_new-0.5 );
            const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell,
                                                              z_old-0.5, z_new-0.5 );
            amrex::ignore_unused(i0_cell_2, k0_cell_2);
            for (int m = 0; m < depos_order; m++) {
                sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
            }
        }

        // Node-based weights using the old and new segment positions
        double sx_old_node[depos_order+1] = {0.};
        double sx_new_node[depos_order+1] = {0.};
        double sz_old_node[depos_order+1] = {0.};
        double sz_new_node[depos_order+1] = {0.};
        const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x_old, x_new );
        const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z_old, z_new );

        // The entries of the mass matrices assume that the stencil starts at
        // the bin cell plus a fixed offset; use the standard kernel otherwise.
        constexpr int off = L::node_offset;
        if (i0_cell != ib + off || i0_node != ib + off ||
            k0_cell != kb + off || k0_node != kb + off) { return general; }

        // Magnetic field on the particle
        amrex::ParticleReal Bxp = Bx_ext;
        amrex::ParticleReal Byp = By_ext;
        amrex::ParticleReal Bzp = Bz_ext;
        if constexpr (has_exteb) {
            amrex::ParticleReal Exp = 0._prt;
            amrex::ParticleReal Eyp = 0._prt;
            amrex::ParticleReal Ezp = 0._prt;
            getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
        } else {
            amrex::ignore_unused(getExternalEB);
        }
        {
            amrex::ParticleReal B1p = 0._prt;
            amrex::ParticleReal B2p = 0._prt;
            amrex::ParticleReal B3p = 0._prt;
            doDirectGatherVectorField<1,1>(xp_nph, yp_nph, zp_nph,
                                           B1p, B2p, B3p,
                                           Bx_arr, By_arr, Bz_arr,
                                           Bx_type, By_type, Bz_type,
                                           dinv, xyzmin, lo, /*n_rz_azimuthal_modes=*/0 );
            Bxp += B1p;
            Byp += B2p;
            Bzp += B3p;
        }

        // Mass matrices kernels
        const amrex::ParticleReal gaminv = GetImplicitGammaInverse(
            uxp_n[ip], uyp_n[ip], uzp_n[ip],
            uxp_nph[ip], uyp_nph[ip], uzp_nph[ip]);
        const amrex::Real wq_invvol = qs*wp[ip]*invvol;
        const amrex::Real rhop = wq_invvol*gaminv;
        amrex::ParticleReal fpxx, fpxy, fpxz;
        amrex::ParticleReal fpyx, fpyy, fpyz;
        amrex::ParticleReal fpzx, fpzy, fpzz;
        setMassMatricesKernels(qs, ms, dt, rhop,
                               uxp_nph[ip], uyp_nph[ip], uzp_nph[ip],
                               Bxp, Byp, Bzp,
                               fpxx, fpxy, fpxz,
                               fpyx, fpyy, fpyz,
                               fpzx, fpzy, fpzz);

        // Fill the record: the weights are computed exactly as in the standard kernel
        for (int i = 0; i <= depos_order - 1; i++) {
            for (int k = 0; k <= depos_order; k++) {
                const amrex::Real weight =
                    sx_cell[i]*(sz_old_node[k] + sz_new_node[k])/2.0_rt*seg_factor_x;
                const int j = i*(depos_order+1) + k;
                rec[L::we_cn + j] = weight;
                rec[L::pj_xx + j] = fpxx*weight;
                rec[L::pj_xy + j] = fpxy*weight;
                rec[L::pj_xz + j] = fpxz*weight;
            }
        }
        for (int i = 0; i <= depos_order; i++) {
            for (int k = 0; k <= depos_order; k++) {
                const amrex::Real weight = ( sx_old_node[i]*sz_old_node[k]*one_third
                                         +   sx_old_node[i]*sz_new_node[k]*one_sixth
                                         +   sx_new_node[i]*sz_old_node[k]*one_sixth
                                         +   sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                const int j = i*(depos_order+1) + k;
                rec[L::we_nn + j] = weight;
                rec[L::pj_yx + j] = fpyx*weight;
                rec[L::pj_yy + j] = fpyy*weight;
                rec[L::pj_yz + j] = fpyz*weight;
            }
        }
        for (int i = 0; i <= depos_order; i++) {
            for (int k = 0; k <= depos_order - 1; k++) {
                const amrex::Real weight =
                    sz_cell[k]*(sx_old_node[i] + sx_new_node[i])/2.0_rt*seg_factor_z;
                const int j = i*depos_order + k;
                rec[L::we_nc + j] = weight;
                rec[L::pj_zx + j] = fpzx*weight;
                rec[L::pj_zy + j] = fpzy*weight;
                rec[L::pj_zz + j] = fpzz*weight;
            }
        }

        return staged;
    }

    /** Arguments of the binned deposition kernel */
    struct KernelArgs
    {
        const amrex::ParticleReal* xp_n_data;
        const amrex::ParticleReal* zp_n_data;
        GetParticlePosition<PIdx> GetPosition;
        const int* nsuborbits;
        const amrex::ParticleReal* wp;
        const amrex::ParticleReal* uxp_n;
        const amrex::ParticleReal* uyp_n;
        const amrex::ParticleReal* uzp_n;
        const amrex::ParticleReal* uxp_nph;
        const amrex::ParticleReal* uyp_nph;
        const amrex::ParticleReal* uzp_nph;
        amrex::GpuArray<amrex::Array4<amrex::Real>,9> S_arrs;
        GetExternalEBField getExternalEB;
        amrex::ParticleReal Bx_ext, By_ext, Bz_ext;
        amrex::Array4<amrex::Real const> Bx_arr, By_arr, Bz_arr;
        amrex::IndexType Bx_type, By_type, Bz_type;
        amrex::Real dt;
        amrex::XDim3 dinv;
        amrex::XDim3 xyzmin;
        amrex::GpuArray<amrex::GpuArray<double,2>, AMREX_SPACEDIM> domain_double;
        amrex::GpuArray<amrex::GpuArray<bool,2>, AMREX_SPACEDIM> do_cropping;
        amrex::Dim3 lo;
        amrex::Real qs, ms, invvol;
        const int* permutation;
        const int* offsets;
        int nbins_x;
        const Entry* entries;
        int n_entries;
        int np_stage;  //!< number of particles staged in shared memory at a time
        int* general_list;
        int* general_count;
    };

    /**
     * \brief Accumulate NQ entries of a thread over the np particles staged in rec.
     * NQ is a compile-time constant so that the arrays stay in registers.
     */
    template <int NQ, int SIZE>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void accumulateEntries (amrex::Real* acc, const int* pj_off, const int* we_off,
                            const amrex::Real* rec, const int np)
    {
        for (int j = 0; j < np; ++j) {
            const amrex::Real* const r = rec + j*SIZE;
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                acc[q] += r[pj_off[q]] * r[we_off[q]];
            }
        }
    }

    /** \brief Call accumulateEntries<n> for the run-time number of entries n (1 <= n <= NQ) */
    template <int NQ, int SIZE>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void accumulateDispatch (const int n, amrex::Real* acc, const int* pj_off, const int* we_off,
                             const amrex::Real* rec, const int np)
    {
        if constexpr (NQ > 1) {
            if (n < NQ) {
                accumulateDispatch<NQ-1,SIZE>(n, acc, pj_off, we_off, rec, np);
                return;
            }
        }
        accumulateEntries<NQ,SIZE>(acc, pj_off, we_off, rec, np);
    }

    /**
     * \brief The binned deposition kernel: one thread block per bin, each thread
     * accumulating entries_per_thread (node, component) entries of the mass matrices.
     */
    template <int depos_order, bool has_exteb>
    __global__ void
    __launch_bounds__(RecordLayout<depos_order>::threads_per_block,
                      RecordLayout<depos_order>::min_blocks_per_sm)
    binnedDepositionKernel (const KernelArgs a)
    {
        using namespace amrex::literals;
        using L = RecordLayout<depos_order>;
        constexpr int EPT = L::entries_per_thread;
        constexpr int nthreads = L::threads_per_block;

        const int bin = static_cast<int>(blockIdx.x);
        const int bin_start = a.offsets[bin];
        const int bin_stop = a.offsets[bin+1];
        if (bin_start == bin_stop) { return; /* this bin has no particles */ }

        const int ib = bin % a.nbins_x;
        const int kb = bin / a.nbins_x;

        amrex::Gpu::SharedMemory<amrex::Real> gsm;
        amrex::Real* const rec = gsm.dataPtr();

        // The entries of this thread. All loops over the entries have compile-time
        // bounds so that the arrays below stay in registers.
        const int tid = static_cast<int>(threadIdx.x);
        int pj_off[EPT];
        int we_off[EPT];
        amrex::Real acc[EPT];
        int n_my_entries = 0;
#pragma unroll
        for (int q = 0; q < EPT; ++q) {
            const int e = tid + q*nthreads;
            pj_off[q] = (e < a.n_entries) ? a.entries[e].pj_off : 0;
            we_off[q] = (e < a.n_entries) ? a.entries[e].we_off : 0;
            acc[q] = 0._rt;
            if (e < a.n_entries) { ++n_my_entries; }
        }

        for (int p0 = bin_start; p0 < bin_stop; p0 += a.np_stage) {
            const int np_here = amrex::min(a.np_stage, bin_stop - p0);

            // Stage the records of np_here particles in shared memory
            if (tid < np_here) {
                const long ip = a.permutation[p0 + tid];
                amrex::Real* const r = rec + tid*L::size;
                const int result = stageParticle<depos_order, has_exteb>(
                    ip, ib, kb, a.xp_n_data, a.zp_n_data, a.GetPosition, a.nsuborbits, a.wp,
                    a.uxp_n, a.uyp_n, a.uzp_n, a.uxp_nph, a.uyp_nph, a.uzp_nph,
                    a.getExternalEB, a.Bx_ext, a.By_ext, a.Bz_ext,
                    a.Bx_arr, a.By_arr, a.Bz_arr, a.Bx_type, a.By_type, a.Bz_type,
                    a.dt, a.dinv, a.xyzmin, a.domain_double, a.do_cropping, a.lo,
                    a.qs, a.ms, a.invvol, r);
                if (result != staged) {
                    for (int j = 0; j < L::size; ++j) { r[j] = 0._rt; }
                    if (result == general) {
                        const int slot = amrex::Gpu::Atomic::Add(a.general_count, 1);
                        a.general_list[slot] = static_cast<int>(ip);
                    }
                }
            }
            __syncthreads();

            // Accumulate this thread's entries over the staged particles
            if (n_my_entries > 0) {
                accumulateDispatch<EPT,L::size>(n_my_entries, acc, pj_off, we_off, rec, np_here);
            }
            __syncthreads();
        }

        constexpr int off = L::node_offset;
#pragma unroll
        for (int q = 0; q < EPT; ++q) {
            const int e = tid + q*nthreads;
            if (e < a.n_entries) {
                const Entry ent = a.entries[e];
                amrex::Gpu::Atomic::AddNoRet(
                    &a.S_arrs[ent.m](a.lo.x + ib + off + ent.i,
                                     a.lo.y + kb + off + ent.k, 0, ent.comp),
                    acc[q]);
            }
        }
    }

    /**
     * \brief Launch the binned deposition kernel, with one thread block per bin.
     */
    template <int depos_order, bool has_exteb>
    void launchBinnedKernel (const KernelArgs& a, const int nbins,
                             const std::size_t shared_mem_bytes)
    {
        using L = RecordLayout<depos_order>;
#if defined(AMREX_USE_CUDA)
        binnedDepositionKernel<depos_order,has_exteb>
            <<<nbins, L::threads_per_block, shared_mem_bytes, amrex::Gpu::gpuStream()>>>(a);
#elif defined(AMREX_USE_HIP)
        hipLaunchKernelGGL((binnedDepositionKernel<depos_order,has_exteb>),
                           dim3(nbins), dim3(L::threads_per_block), shared_mem_bytes,
                           amrex::Gpu::gpuStream(), a);
#endif
        AMREX_GPU_ERROR_CHECK();
    }
}

/**
 * \brief Binned deposition of the full Villasenor mass matrices in 2D on GPU.
 *
 * The arguments are the same as doVillasenorSigmaDeposition, plus:
 *
 * \param ptd              Particle tile data (used to bin the particles)
 * \param nbins_x          Number of bins (cells of the tile box, including guard cells) along x
 * \param nbins_z          Number of bins along z
 */
template <int depos_order, int max_supported_grid_crossings, typename PTD>
void doVillasenorSigmaDepositionBinned (const PTD& ptd,
                                        const int nbins_x,
                                        const int nbins_z,
                                        const amrex::ParticleReal* xp_n_data,
                                        const amrex::ParticleReal* yp_n_data,
                                        const amrex::ParticleReal* zp_n_data,
                                        const GetParticlePosition<PIdx>& GetPosition,
                                        const int* nsuborbits,
                                        const amrex::ParticleReal* wp,
                                        const amrex::ParticleReal* uxp_n,
                                        const amrex::ParticleReal* uyp_n,
                                        const amrex::ParticleReal* uzp_n,
                                        const amrex::ParticleReal* uxp_nph,
                                        const amrex::ParticleReal* uyp_nph,
                                        const amrex::ParticleReal* uzp_nph,
                                        const int max_grid_crossings,
                                        amrex::Array4<amrex::Real> const& Sxx_arr,
                                        amrex::Array4<amrex::Real> const& Sxy_arr,
                                        amrex::Array4<amrex::Real> const& Sxz_arr,
                                        amrex::Array4<amrex::Real> const& Syx_arr,
                                        amrex::Array4<amrex::Real> const& Syy_arr,
                                        amrex::Array4<amrex::Real> const& Syz_arr,
                                        amrex::Array4<amrex::Real> const& Szx_arr,
                                        amrex::Array4<amrex::Real> const& Szy_arr,
                                        amrex::Array4<amrex::Real> const& Szz_arr,
                                        GetExternalEBField const & getExternalEB,
                                        const amrex::ParticleReal Bx_ext,
                                        const amrex::ParticleReal By_ext,
                                        const amrex::ParticleReal Bz_ext,
                                        const amrex::Array4<amrex::Real const>& Bx_arr,
                                        const amrex::Array4<amrex::Real const>& By_arr,
                                        const amrex::Array4<amrex::Real const>& Bz_arr,
                                        const amrex::IndexType Bx_type,
                                        const amrex::IndexType By_type,
                                        const amrex::IndexType Bz_type,
                                        const long np_to_deposit,
                                        const amrex::Real dt,
                                        const amrex::XDim3& dinv,
                                        const amrex::XDim3& xyzmin,
                                        const amrex::GpuArray<amrex::GpuArray<double,2>,
                                                              AMREX_SPACEDIM>& domain_double,
                                        const amrex::GpuArray<amrex::GpuArray<bool,2>,
                                                              AMREX_SPACEDIM>& do_cropping,
                                        const amrex::Dim3 lo,
                                        const amrex::Real qs,
                                        const amrex::Real ms)
{
    using namespace MassMatricesBinned;
    using L = RecordLayout<depos_order>;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    // Bin the particles by (shifted) cell
    const int nbins = nbins_x*nbins_z;
    amrex::DenseBins<PTD> bins;
    bins.build(np_to_deposit, ptd, nbins,
        [=] AMREX_GPU_HOST_DEVICE (const PTD& /*ptd*/, long ip) -> unsigned int
        {
            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);
            return static_cast<unsigned int>(
                binIndex<depos_order>(xp, zp, dinv, xyzmin, nbins_x, nbins_z));
        });
    const int* permutation = bins.permutationPtr();
    const int* offsets = bins.offsetsPtr();

    // Table of the entries deposited by each single-segment particle
    const amrex::Vector<Entry> h_entries = makeEntries<depos_order>(max_grid_crossings);
    const int n_entries = static_cast<int>(h_entries.size());
    amrex::Gpu::DeviceVector<Entry> d_entries(n_entries);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          h_entries.begin(), h_entries.end(), d_entries.begin());
    const Entry* entries = d_entries.data();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(n_entries == L::n_entries,
        "Unexpected number of mass matrices entries in the binned deposition");
    constexpr int nthreads = L::threads_per_block;

    // Number of particles staged in shared memory at a time
    const std::size_t max_shared_mem_bytes = amrex::Gpu::Device::sharedMemPerBlock();
    int np_stage = std::min(64, nthreads);
    auto shared_mem_needed = [](int n) {
        return static_cast<std::size_t>(n)*L::size*sizeof(amrex::Real);
    };
    while (np_stage > 16 && shared_mem_needed(np_stage) > max_shared_mem_bytes) {
        np_stage /= 2;
    }
    const std::size_t shared_mem_bytes = shared_mem_needed(np_stage);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(shared_mem_bytes <= max_shared_mem_bytes,
        "Not enough GPU shared memory for the binned mass matrices deposition");

    // List of the particles that must be deposited with the standard kernel
    amrex::Gpu::DeviceVector<int> general_list(std::max(np_to_deposit, 1L));
    int* general_list_ptr = general_list.data();
    amrex::Gpu::Buffer<int> general_count({0});
    int* general_count_ptr = general_count.data();

    // Arguments of the kernel
    KernelArgs args;
    args.xp_n_data = xp_n_data; args.zp_n_data = zp_n_data;
    args.GetPosition = GetPosition; args.nsuborbits = nsuborbits; args.wp = wp;
    args.uxp_n = uxp_n; args.uyp_n = uyp_n; args.uzp_n = uzp_n;
    args.uxp_nph = uxp_nph; args.uyp_nph = uyp_nph; args.uzp_nph = uzp_nph;
    args.S_arrs = {Sxx_arr, Sxy_arr, Sxz_arr, Syx_arr, Syy_arr, Syz_arr, Szx_arr, Szy_arr, Szz_arr};
    args.getExternalEB = getExternalEB;
    args.Bx_ext = Bx_ext; args.By_ext = By_ext; args.Bz_ext = Bz_ext;
    args.Bx_arr = Bx_arr; args.By_arr = By_arr; args.Bz_arr = Bz_arr;
    args.Bx_type = Bx_type; args.By_type = By_type; args.Bz_type = Bz_type;
    args.dt = dt; args.dinv = dinv; args.xyzmin = xyzmin;
    args.domain_double = domain_double; args.do_cropping = do_cropping; args.lo = lo;
    args.qs = qs; args.ms = ms; args.invvol = invvol;
    args.permutation = permutation; args.offsets = offsets; args.nbins_x = nbins_x;
    args.entries = entries; args.n_entries = n_entries;
    args.np_stage = np_stage;
    args.general_list = general_list_ptr; args.general_count = general_count_ptr;

    if (getExternalEB.isNoOp()) {
        launchBinnedKernel<depos_order,false>(args, nbins, shared_mem_bytes);
    } else {
        launchBinnedKernel<depos_order,true>(args, nbins, shared_mem_bytes);
    }

    // Deposit the particles that crossed a cell boundary with the standard kernel
    const int n_general = *general_count.copyToHost();
    if (n_general > 0) {
        doVillasenorSigmaDeposition<depos_order,true,max_supported_grid_crossings>(
            xp_n_data, yp_n_data, zp_n_data,
            GetPosition, nsuborbits, wp,
            uxp_n, uyp_n, uzp_n, uxp_nph, uyp_nph, uzp_nph,
            max_grid_crossings,
            Sxx_arr, Sxy_arr, Sxz_arr,
            Syx_arr, Syy_arr, Syz_arr,
            Szx_arr, Szy_arr, Szz_arr,
            getExternalEB, Bx_ext, By_ext, Bz_ext,
            Bx_arr, By_arr, Bz_arr, Bx_type, By_type, Bz_type,
            n_general, dt, dinv, xyzmin, domain_double, do_cropping, lo, qs, ms,
            general_list_ptr);
    }
    amrex::Gpu::streamSynchronize();
}

#endif // WARPX_DIM_XZ && (CUDA || HIP)

#endif // WARPX_MASS_MATRICES_DEPOSITION_BINNED_H_
