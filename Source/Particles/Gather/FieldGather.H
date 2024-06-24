/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FIELDGATHER_H_
#define WARPX_FIELDGATHER_H_

#include "Particles/Gather/GetExternalFields.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ShapeFactors.H"
#include "Utils/WarpX_Complex.H"

#include <AMReX.H>

/**
 * \brief Field gather for a single particle
 *
 * \tparam depos_order              Particle shape order
 * \tparam galerkin_interpolation   Lower the order of the particle shape by
 *                                  this value (0/1) for the parallel field component
 * \param xp,yp,zp                        Particle position coordinates
 * \param Exp,Eyp,Ezp                     Electric field on particles.
 * \param Bxp,Byp,Bzp                     Magnetic field on particles.
 * \param ex_arr,ey_arr,ez_arr            Array4 of the electric field, either full array or tile.
 * \param bx_arr,by_arr,bz_arr            Array4 of the magnetic field, either full array or tile.
 * \param ex_type,ey_type,ez_type         IndexType of the electric field
 * \param bx_type,by_type,bz_type         IndexType of the magnetic field
 * \param dinv                      3D cell size inverse
 * \param xyzmin                    The lower bounds of the domain
 * \param lo                        Index lower bounds of domain.
 * \param n_rz_azimuthal_modes       Number of azimuthal modes when using RZ geometry
 */
template <int depos_order, int galerkin_interpolation>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherShapeN ([[maybe_unused]] const amrex::ParticleReal xp,
                     [[maybe_unused]] const amrex::ParticleReal yp,
                     [[maybe_unused]] const amrex::ParticleReal zp,
                     amrex::ParticleReal& Exp,
                     amrex::ParticleReal& Eyp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& ex_arr,
                     amrex::Array4<amrex::Real const> const& ey_arr,
                     amrex::Array4<amrex::Real const> const& ez_arr,
                     amrex::Array4<amrex::Real const> const& bx_arr,
                     amrex::Array4<amrex::Real const> const& by_arr,
                     amrex::Array4<amrex::Real const> const& bz_arr,
                     const amrex::IndexType ex_type,
                     const amrex::IndexType ey_type,
                     const amrex::IndexType ez_type,
                     const amrex::IndexType bx_type,
                     const amrex::IndexType by_type,
                     const amrex::IndexType bz_type,
                     const amrex::XDim3 & dinv,
                     const amrex::XDim3 & xyzmin,
                     const amrex::Dim3& lo,
                     [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex;

    constexpr int zdir = WARPX_ZINDEX;
    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // --- Compute shape factors

    Compute_shape_factor< depos_order > const compute_shape_factor;
    Compute_shape_factor<depos_order - galerkin_interpolation > const compute_shape_factor_galerkin;

#if (AMREX_SPACEDIM >= 2)
    // x direction
    // Get particle position
#ifdef WARPX_DIM_RZ
    const amrex::Real rp = std::sqrt(xp*xp + yp*yp);
    const amrex::Real x = (rp - xyzmin.x)*dinv.x;
#else
    const amrex::Real x = (xp-xyzmin.x)*dinv.x;
#endif

    // j_[eb][xyz] leftmost grid point in x that the particle touches for the centering of each current
    // sx_[eb][xyz] shape factor along x for the centering of each current
    // There are only two possible centerings, node or cell centered, so at most only two shape factor
    // arrays will be needed.
    amrex::Real sx_node[depos_order + 1];
    amrex::Real sx_cell[depos_order + 1];
    amrex::Real sx_node_galerkin[depos_order + 1 - galerkin_interpolation] = {0._rt};
    amrex::Real sx_cell_galerkin[depos_order + 1 - galerkin_interpolation] = {0._rt};

    int j_node = 0;
    int j_cell = 0;
    int j_node_v = 0;
    int j_cell_v = 0;
    if ((ey_type[0] == NODE) || (ez_type[0] == NODE) || (bx_type[0] == NODE)) {
        j_node = compute_shape_factor(sx_node, x);
    }
    if ((ey_type[0] == CELL) || (ez_type[0] == CELL) || (bx_type[0] == CELL)) {
        j_cell = compute_shape_factor(sx_cell, x - 0.5_rt);
    }
    if ((ex_type[0] == NODE) || (by_type[0] == NODE) || (bz_type[0] == NODE)) {
        j_node_v = compute_shape_factor_galerkin(sx_node_galerkin, x);
    }
    if ((ex_type[0] == CELL) || (by_type[0] == CELL) || (bz_type[0] == CELL)) {
        j_cell_v = compute_shape_factor_galerkin(sx_cell_galerkin, x - 0.5_rt);
    }
    const amrex::Real (&sx_ex)[depos_order + 1 - galerkin_interpolation] = ((ex_type[0] == NODE) ? sx_node_galerkin : sx_cell_galerkin);
    const amrex::Real (&sx_ey)[depos_order + 1             ] = ((ey_type[0] == NODE) ? sx_node   : sx_cell  );
    const amrex::Real (&sx_ez)[depos_order + 1             ] = ((ez_type[0] == NODE) ? sx_node   : sx_cell  );
    const amrex::Real (&sx_bx)[depos_order + 1             ] = ((bx_type[0] == NODE) ? sx_node   : sx_cell  );
    const amrex::Real (&sx_by)[depos_order + 1 - galerkin_interpolation] = ((by_type[0] == NODE) ? sx_node_galerkin : sx_cell_galerkin);
    const amrex::Real (&sx_bz)[depos_order + 1 - galerkin_interpolation] = ((bz_type[0] == NODE) ? sx_node_galerkin : sx_cell_galerkin);
    int const j_ex = ((ex_type[0] == NODE) ? j_node_v : j_cell_v);
    int const j_ey = ((ey_type[0] == NODE) ? j_node   : j_cell  );
    int const j_ez = ((ez_type[0] == NODE) ? j_node   : j_cell  );
    int const j_bx = ((bx_type[0] == NODE) ? j_node   : j_cell  );
    int const j_by = ((by_type[0] == NODE) ? j_node_v : j_cell_v);
    int const j_bz = ((bz_type[0] == NODE) ? j_node_v : j_cell_v);
#endif

#if defined(WARPX_DIM_3D)
    // y direction
    const amrex::Real y = (yp-xyzmin.y)*dinv.y;
    amrex::Real sy_node[depos_order + 1];
    amrex::Real sy_cell[depos_order + 1];
    amrex::Real sy_node_v[depos_order + 1 - galerkin_interpolation];
    amrex::Real sy_cell_v[depos_order + 1 - galerkin_interpolation];
    int k_node = 0;
    int k_cell = 0;
    int k_node_v = 0;
    int k_cell_v = 0;
    if ((ex_type[1] == NODE) || (ez_type[1] == NODE) || (by_type[1] == NODE)) {
        k_node = compute_shape_factor(sy_node, y);
    }
    if ((ex_type[1] == CELL) || (ez_type[1] == CELL) || (by_type[1] == CELL)) {
        k_cell = compute_shape_factor(sy_cell, y - 0.5_rt);
    }
    if ((ey_type[1] == NODE) || (bx_type[1] == NODE) || (bz_type[1] == NODE)) {
        k_node_v = compute_shape_factor_galerkin(sy_node_v, y);
    }
    if ((ey_type[1] == CELL) || (bx_type[1] == CELL) || (bz_type[1] == CELL)) {
        k_cell_v = compute_shape_factor_galerkin(sy_cell_v, y - 0.5_rt);
    }
    const amrex::Real (&sy_ex)[depos_order + 1             ] = ((ex_type[1] == NODE) ? sy_node   : sy_cell  );
    const amrex::Real (&sy_ey)[depos_order + 1 - galerkin_interpolation] = ((ey_type[1] == NODE) ? sy_node_v : sy_cell_v);
    const amrex::Real (&sy_ez)[depos_order + 1             ] = ((ez_type[1] == NODE) ? sy_node   : sy_cell  );
    const amrex::Real (&sy_bx)[depos_order + 1 - galerkin_interpolation] = ((bx_type[1] == NODE) ? sy_node_v : sy_cell_v);
    const amrex::Real (&sy_by)[depos_order + 1             ] = ((by_type[1] == NODE) ? sy_node   : sy_cell  );
    const amrex::Real (&sy_bz)[depos_order + 1 - galerkin_interpolation] = ((bz_type[1] == NODE) ? sy_node_v : sy_cell_v);
    int const k_ex = ((ex_type[1] == NODE) ? k_node   : k_cell  );
    int const k_ey = ((ey_type[1] == NODE) ? k_node_v : k_cell_v);
    int const k_ez = ((ez_type[1] == NODE) ? k_node   : k_cell  );
    int const k_bx = ((bx_type[1] == NODE) ? k_node_v : k_cell_v);
    int const k_by = ((by_type[1] == NODE) ? k_node   : k_cell  );
    int const k_bz = ((bz_type[1] == NODE) ? k_node_v : k_cell_v);

#endif
    // z direction
    const amrex::Real z = (zp-xyzmin.z)*dinv.z;
    amrex::Real sz_node[depos_order + 1];
    amrex::Real sz_cell[depos_order + 1];
    amrex::Real sz_node_v[depos_order + 1 - galerkin_interpolation];
    amrex::Real sz_cell_v[depos_order + 1 - galerkin_interpolation];
    int l_node = 0;
    int l_cell = 0;
    int l_node_v = 0;
    int l_cell_v = 0;
    if ((ex_type[zdir] == NODE) || (ey_type[zdir] == NODE) || (bz_type[zdir] == NODE)) {
        l_node = compute_shape_factor(sz_node, z);
    }
    if ((ex_type[zdir] == CELL) || (ey_type[zdir] == CELL) || (bz_type[zdir] == CELL)) {
        l_cell = compute_shape_factor(sz_cell, z - 0.5_rt);
    }
    if ((ez_type[zdir] == NODE) || (bx_type[zdir] == NODE) || (by_type[zdir] == NODE)) {
        l_node_v = compute_shape_factor_galerkin(sz_node_v, z);
    }
    if ((ez_type[zdir] == CELL) || (bx_type[zdir] == CELL) || (by_type[zdir] == CELL)) {
        l_cell_v = compute_shape_factor_galerkin(sz_cell_v, z - 0.5_rt);
    }
    const amrex::Real (&sz_ex)[depos_order + 1             ] = ((ex_type[zdir] == NODE) ? sz_node   : sz_cell  );
    const amrex::Real (&sz_ey)[depos_order + 1             ] = ((ey_type[zdir] == NODE) ? sz_node   : sz_cell  );
    const amrex::Real (&sz_ez)[depos_order + 1 - galerkin_interpolation] = ((ez_type[zdir] == NODE) ? sz_node_v : sz_cell_v);
    const amrex::Real (&sz_bx)[depos_order + 1 - galerkin_interpolation] = ((bx_type[zdir] == NODE) ? sz_node_v : sz_cell_v);
    const amrex::Real (&sz_by)[depos_order + 1 - galerkin_interpolation] = ((by_type[zdir] == NODE) ? sz_node_v : sz_cell_v);
    const amrex::Real (&sz_bz)[depos_order + 1             ] = ((bz_type[zdir] == NODE) ? sz_node   : sz_cell  );
    int const l_ex = ((ex_type[zdir] == NODE) ? l_node   : l_cell  );
    int const l_ey = ((ey_type[zdir] == NODE) ? l_node   : l_cell  );
    int const l_ez = ((ez_type[zdir] == NODE) ? l_node_v : l_cell_v);
    int const l_bx = ((bx_type[zdir] == NODE) ? l_node_v : l_cell_v);
    int const l_by = ((by_type[zdir] == NODE) ? l_node_v : l_cell_v);
    int const l_bz = ((bz_type[zdir] == NODE) ? l_node   : l_cell  );


    // Each field is gathered in a separate block of
    // AMREX_SPACEDIM nested loops because the deposition
    // order can differ for each component of each field
    // when galerkin_interpolation is set to 1

#if defined(WARPX_DIM_1D_Z)
    // Gather field on particle Eyp from field on grid ey_arr
    // Gather field on particle Exp from field on grid ex_arr
    // Gather field on particle Bzp from field on grid bz_arr
    for (int iz=0; iz<=depos_order; iz++){
        Eyp += sz_ey[iz]*
            ey_arr(lo.x+l_ey+iz, 0, 0, 0);
        Exp += sz_ex[iz]*
            ex_arr(lo.x+l_ex+iz, 0, 0, 0);
        Bzp += sz_bz[iz]*
            bz_arr(lo.x+l_bz+iz, 0, 0, 0);
    }

    // Gather field on particle Byp from field on grid by_arr
    // Gather field on particle Ezp from field on grid ez_arr
    // Gather field on particle Bxp from field on grid bx_arr
    for (int iz=0; iz<=depos_order-galerkin_interpolation; iz++){
        Ezp += sz_ez[iz]*
            ez_arr(lo.x+l_ez+iz, 0, 0, 0);
        Bxp += sz_bx[iz]*
            bx_arr(lo.x+l_bx+iz, 0, 0, 0);
        Byp += sz_by[iz]*
            by_arr(lo.x+l_by+iz, 0, 0, 0);
    }

#elif defined(WARPX_DIM_XZ)
    // Gather field on particle Eyp from field on grid ey_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            Eyp += sx_ey[ix]*sz_ey[iz]*
                ey_arr(lo.x+j_ey+ix, lo.y+l_ey+iz, 0, 0);
        }
    }
    // Gather field on particle Exp from field on grid ex_arr
    // Gather field on particle Bzp from field on grid bz_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order-galerkin_interpolation; ix++){
            Exp += sx_ex[ix]*sz_ex[iz]*
                ex_arr(lo.x+j_ex+ix, lo.y+l_ex+iz, 0, 0);
            Bzp += sx_bz[ix]*sz_bz[iz]*
                bz_arr(lo.x+j_bz+ix, lo.y+l_bz+iz, 0, 0);
        }
    }
    // Gather field on particle Ezp from field on grid ez_arr
    // Gather field on particle Bxp from field on grid bx_arr
    for (int iz=0; iz<=depos_order-galerkin_interpolation; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            Ezp += sx_ez[ix]*sz_ez[iz]*
                ez_arr(lo.x+j_ez+ix, lo.y+l_ez+iz, 0, 0);
            Bxp += sx_bx[ix]*sz_bx[iz]*
                bx_arr(lo.x+j_bx+ix, lo.y+l_bx+iz, 0, 0);
        }
    }
    // Gather field on particle Byp from field on grid by_arr
    for (int iz=0; iz<=depos_order-galerkin_interpolation; iz++){
        for (int ix=0; ix<=depos_order-galerkin_interpolation; ix++){
            Byp += sx_by[ix]*sz_by[iz]*
                by_arr(lo.x+j_by+ix, lo.y+l_by+iz, 0, 0);
        }
    }

#elif defined(WARPX_DIM_RZ)

    amrex::ParticleReal Erp = 0.;
    amrex::ParticleReal Ethetap = 0.;
    amrex::ParticleReal Brp = 0.;
    amrex::ParticleReal Bthetap = 0.;

    // Gather field on particle Ethetap from field on grid ey_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            Ethetap += sx_ey[ix]*sz_ey[iz]*
                ey_arr(lo.x+j_ey+ix, lo.y+l_ey+iz, 0, 0);
        }
    }
    // Gather field on particle Erp from field on grid ex_arr
    // Gather field on particle Bzp from field on grid bz_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order-galerkin_interpolation; ix++){
            Erp += sx_ex[ix]*sz_ex[iz]*
                ex_arr(lo.x+j_ex+ix, lo.y+l_ex+iz, 0, 0);
            Bzp += sx_bz[ix]*sz_bz[iz]*
                bz_arr(lo.x+j_bz+ix, lo.y+l_bz+iz, 0, 0);
        }
    }
    // Gather field on particle Ezp from field on grid ez_arr
    // Gather field on particle Brp from field on grid bx_arr
    for (int iz=0; iz<=depos_order-galerkin_interpolation; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            Ezp += sx_ez[ix]*sz_ez[iz]*
                ez_arr(lo.x+j_ez+ix, lo.y+l_ez+iz, 0, 0);
            Brp += sx_bx[ix]*sz_bx[iz]*
                bx_arr(lo.x+j_bx+ix, lo.y+l_bx+iz, 0, 0);
        }
    }
    // Gather field on particle Bthetap from field on grid by_arr
    for (int iz=0; iz<=depos_order-galerkin_interpolation; iz++){
        for (int ix=0; ix<=depos_order-galerkin_interpolation; ix++){
            Bthetap += sx_by[ix]*sz_by[iz]*
                by_arr(lo.x+j_by+ix, lo.y+l_by+iz, 0, 0);
        }
    }

    amrex::Real costheta;
    amrex::Real sintheta;
    if (rp > 0.) {
        costheta = xp/rp;
        sintheta = yp/rp;
    } else {
        costheta = 1.;
        sintheta = 0.;
    }
    const Complex xy0 = Complex{costheta, -sintheta};
    Complex xy = xy0;

    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {

        // Gather field on particle Ethetap from field on grid ey_arr
        for (int iz=0; iz<=depos_order; iz++){
            for (int ix=0; ix<=depos_order; ix++){
                const amrex::Real dEy = (+ ey_arr(lo.x+j_ey+ix, lo.y+l_ey+iz, 0, 2*imode-1)*xy.real()
                                         - ey_arr(lo.x+j_ey+ix, lo.y+l_ey+iz, 0, 2*imode)*xy.imag());
                Ethetap += sx_ey[ix]*sz_ey[iz]*dEy;
            }
        }
        // Gather field on particle Erp from field on grid ex_arr
        // Gather field on particle Bzp from field on grid bz_arr
        for (int iz=0; iz<=depos_order; iz++){
            for (int ix=0; ix<=depos_order-galerkin_interpolation; ix++){
                const amrex::Real dEx = (+ ex_arr(lo.x+j_ex+ix, lo.y+l_ex+iz, 0, 2*imode-1)*xy.real()
                                         - ex_arr(lo.x+j_ex+ix, lo.y+l_ex+iz, 0, 2*imode)*xy.imag());
                Erp += sx_ex[ix]*sz_ex[iz]*dEx;
                const amrex::Real dBz = (+ bz_arr(lo.x+j_bz+ix, lo.y+l_bz+iz, 0, 2*imode-1)*xy.real()
                                         - bz_arr(lo.x+j_bz+ix, lo.y+l_bz+iz, 0, 2*imode)*xy.imag());
                Bzp += sx_bz[ix]*sz_bz[iz]*dBz;
            }
        }
        // Gather field on particle Ezp from field on grid ez_arr
        // Gather field on particle Brp from field on grid bx_arr
        for (int iz=0; iz<=depos_order-galerkin_interpolation; iz++){
            for (int ix=0; ix<=depos_order; ix++){
                const amrex::Real dEz = (+ ez_arr(lo.x+j_ez+ix, lo.y+l_ez+iz, 0, 2*imode-1)*xy.real()
                                         - ez_arr(lo.x+j_ez+ix, lo.y+l_ez+iz, 0, 2*imode)*xy.imag());
                Ezp += sx_ez[ix]*sz_ez[iz]*dEz;
                const amrex::Real dBx = (+ bx_arr(lo.x+j_bx+ix, lo.y+l_bx+iz, 0, 2*imode-1)*xy.real()
                                         - bx_arr(lo.x+j_bx+ix, lo.y+l_bx+iz, 0, 2*imode)*xy.imag());
                Brp += sx_bx[ix]*sz_bx[iz]*dBx;
            }
        }
        // Gather field on particle Bthetap from field on grid by_arr
        for (int iz=0; iz<=depos_order-galerkin_interpolation; iz++){
            for (int ix=0; ix<=depos_order-galerkin_interpolation; ix++){
                const amrex::Real dBy = (+ by_arr(lo.x+j_by+ix, lo.y+l_by+iz, 0, 2*imode-1)*xy.real()
                                         - by_arr(lo.x+j_by+ix, lo.y+l_by+iz, 0, 2*imode)*xy.imag());
                Bthetap += sx_by[ix]*sz_by[iz]*dBy;
            }
        }
        xy = xy*xy0;
    }

    // Convert Erp and Ethetap to Ex and Ey
    Exp += costheta*Erp - sintheta*Ethetap;
    Eyp += costheta*Ethetap + sintheta*Erp;
    Bxp += costheta*Brp - sintheta*Bthetap;
    Byp += costheta*Bthetap + sintheta*Brp;

#else // defined(WARPX_DIM_3D)
    // Gather field on particle Exp from field on grid ex_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<= depos_order - galerkin_interpolation; ix++){
                Exp += sx_ex[ix]*sy_ex[iy]*sz_ex[iz]*
                    ex_arr(lo.x+j_ex+ix, lo.y+k_ex+iy, lo.z+l_ex+iz);
            }
        }
    }
    // Gather field on particle Eyp from field on grid ey_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<= depos_order - galerkin_interpolation; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                Eyp += sx_ey[ix]*sy_ey[iy]*sz_ey[iz]*
                    ey_arr(lo.x+j_ey+ix, lo.y+k_ey+iy, lo.z+l_ey+iz);
            }
        }
    }
    // Gather field on particle Ezp from field on grid ez_arr
    for (int iz=0; iz<= depos_order - galerkin_interpolation; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                Ezp += sx_ez[ix]*sy_ez[iy]*sz_ez[iz]*
                    ez_arr(lo.x+j_ez+ix, lo.y+k_ez+iy, lo.z+l_ez+iz);
            }
        }
    }
    // Gather field on particle Bzp from field on grid bz_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<= depos_order - galerkin_interpolation; iy++){
            for (int ix=0; ix<= depos_order - galerkin_interpolation; ix++){
                Bzp += sx_bz[ix]*sy_bz[iy]*sz_bz[iz]*
                    bz_arr(lo.x+j_bz+ix, lo.y+k_bz+iy, lo.z+l_bz+iz);
            }
        }
    }
    // Gather field on particle Byp from field on grid by_arr
    for (int iz=0; iz<= depos_order - galerkin_interpolation; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<= depos_order - galerkin_interpolation; ix++){
                Byp += sx_by[ix]*sy_by[iy]*sz_by[iz]*
                    by_arr(lo.x+j_by+ix, lo.y+k_by+iy, lo.z+l_by+iz);
            }
        }
    }
    // Gather field on particle Bxp from field on grid bx_arr
    for (int iz=0; iz<= depos_order - galerkin_interpolation; iz++){
        for (int iy=0; iy<= depos_order - galerkin_interpolation; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                Bxp += sx_bx[ix]*sy_bx[iy]*sz_bx[iz]*
                    bx_arr(lo.x+j_bx+ix, lo.y+k_bx+iy, lo.z+l_bx+iz);
            }
        }
    }
#endif
}

/**
 * \brief Energy conserving field gather for thread thread_num for the implicit scheme
 *        This uses the same stencil for the gather that is used for Esirkepov current deposition.
 *
 * \tparam depos_order              Particle shape order
 * \param xp_n,yp_n,zp_n                  Particle position coordinates at start of step
 * \param xp_nph,yp_nph,zp_nph            Particle position coordinates at half step
 * \param Exp,Eyp,Ezp                     Electric field on particles.
 * \param Bxp,Byp,Bzp                     Magnetic field on particles.
 * \param Ex_arr,Ey_arr,Ez_arr            Array4 of the electric field, either full array or tile.
 * \param Bx_arr,By_arr,Bz_arr            Array4 of the magnetic field, either full array or tile.
 * \param Ex_type,Ey_type,Ez_type         IndexType of the electric field
 * \param Bx_type,By_type,Bz_type         IndexType of the magnetic field
 * \param dinv                      3D cell size inverse
 * \param xyzmin                    The lower bounds of the domain
 * \param lo                        Index lower bounds of domain.
 * \param n_rz_azimuthal_modes      Number of azimuthal modes when using RZ geometry
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherShapeNEsirkepovStencilImplicit (
                     [[maybe_unused]] const amrex::ParticleReal xp_n,
                     [[maybe_unused]] const amrex::ParticleReal yp_n,
                     const amrex::ParticleReal zp_n,
                     [[maybe_unused]] const amrex::ParticleReal xp_nph,
                     [[maybe_unused]] const amrex::ParticleReal yp_nph,
                     const amrex::ParticleReal zp_nph,
                     amrex::ParticleReal& Exp,
                     amrex::ParticleReal& Eyp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& Ex_arr,
                     amrex::Array4<amrex::Real const> const& Ey_arr,
                     amrex::Array4<amrex::Real const> const& Ez_arr,
                     amrex::Array4<amrex::Real const> const& Bx_arr,
                     amrex::Array4<amrex::Real const> const& By_arr,
                     amrex::Array4<amrex::Real const> const& Bz_arr,
                     [[maybe_unused]] const amrex::IndexType Ex_type,
                     [[maybe_unused]] const amrex::IndexType Ey_type,
                     [[maybe_unused]] const amrex::IndexType Ez_type,
                     [[maybe_unused]] const amrex::IndexType Bx_type,
                     [[maybe_unused]] const amrex::IndexType By_type,
                     [[maybe_unused]] const amrex::IndexType Bz_type,
                     const amrex::XDim3 & dinv,
                     const amrex::XDim3 & xyzmin,
                     const amrex::Dim3& lo,
                     const int n_rz_azimuthal_modes)
{
    using namespace amrex;
#if !defined(WARPX_DIM_RZ)
    ignore_unused(n_rz_azimuthal_modes);
#endif

#if !defined(WARPX_DIM_1D_Z)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

#if !defined(WARPX_DIM_1D_Z)
    const ParticleReal xp_np1 = 2._prt*xp_nph - xp_n;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
    const ParticleReal yp_np1 = 2._prt*yp_nph - yp_n;
#endif
    const ParticleReal zp_np1 = 2._prt*zp_nph - zp_n;

    // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
    amrex::Real const xp_new = xp_np1;
    amrex::Real const yp_new = yp_np1;
    amrex::Real const xp_mid = xp_nph;
    amrex::Real const yp_mid = yp_nph;
    amrex::Real const xp_old = xp_n;
    amrex::Real const yp_old = yp_n;
    amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
    amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
    amrex::Real const rp_mid = (rp_new + rp_old)/2._rt;
    amrex::Real costheta_mid, sintheta_mid;
    if (rp_mid > 0._rt) {
        costheta_mid = xp_mid/rp_mid;
        sintheta_mid = yp_mid/rp_mid;
    } else {
        costheta_mid = 1._rt;
        sintheta_mid = 0._rt;
    }
    const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
    // Keep these double to avoid bug in single precision
    double const x_new = (rp_new - xyzmin.x)*dinv.x;
    double const x_old = (rp_old - xyzmin.x)*dinv.x;
#else
#if !defined(WARPX_DIM_1D_Z)
    // Keep these double to avoid bug in single precision
    double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
    double const x_old = (xp_n - xyzmin.x)*dinv.x;
#endif
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double const y_new = (yp_np1 - xyzmin.y)*dinv.y;
    double const y_old = (yp_n - xyzmin.y)*dinv.y;
#endif
    // Keep these double to avoid bug in single precision
    double const z_new = (zp_np1 - xyzmin.z)*dinv.z;
    double const z_old = (zp_n - xyzmin.z)*dinv.z;

    // Shape factor arrays
    // Note that there are extra values above and below
    // to possibly hold the factor for the old particle
    // which can be at a different grid location.
    // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
    double sx_E_new[depos_order + 3] = {0.};
    double sx_E_old[depos_order + 3] = {0.};
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double sy_E_new[depos_order + 3] = {0.};
    double sy_E_old[depos_order + 3] = {0.};
#endif
    // Keep these double to avoid bug in single precision
    double sz_E_new[depos_order + 3] = {0.};
    double sz_E_old[depos_order + 3] = {0.};

#if defined(WARPX_DIM_3D)
    double sx_B_new[depos_order + 3] = {0.};
    double sx_B_old[depos_order + 3] = {0.};
    double sy_B_new[depos_order + 3] = {0.};
    double sy_B_old[depos_order + 3] = {0.};
    double sz_B_new[depos_order + 3] = {0.};
    double sz_B_old[depos_order + 3] = {0.};
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    // Special shape functions are needed for By which is cell
    // centered in both x and z. One lower order shape function is used.
    double sx_By_new[depos_order + 2] = {0.};
    double sx_By_old[depos_order + 2] = {0.};
    double sz_By_new[depos_order + 2] = {0.};
    double sz_By_old[depos_order + 2] = {0.};
#endif

    // --- Compute shape factors
    // Compute shape factors for position as they are now and at old positions
    // [ijk]_new: leftmost grid point that the particle touches
    const Compute_shape_factor< depos_order > compute_shape_factor;
    const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

#if !defined(WARPX_DIM_1D_Z)
    const int i_E_new = compute_shape_factor(sx_E_new+1, x_new);
    const int i_E_old = compute_shifted_shape_factor(sx_E_old, x_old, i_E_new);
#endif
#if defined(WARPX_DIM_3D)
    const int j_E_new = compute_shape_factor(sy_E_new+1, y_new);
    const int j_E_old = compute_shifted_shape_factor(sy_E_old, y_old, j_E_new);
#endif
    const int k_E_new = compute_shape_factor(sz_E_new+1, z_new);
    const int k_E_old = compute_shifted_shape_factor(sz_E_old, z_old, k_E_new);

#if defined(WARPX_DIM_3D)
    const int i_B_new = compute_shape_factor(sx_B_new+1, x_new - 0.5_rt);
    const int i_B_old = compute_shifted_shape_factor(sx_B_old, x_old - 0.5_rt, i_B_new);
    const int j_B_new = compute_shape_factor(sy_B_new+1, y_new - 0.5_rt);
    const int j_B_old = compute_shifted_shape_factor(sy_B_old, y_old - 0.5_rt, j_B_new);
    const int k_B_new = compute_shape_factor(sz_B_new+1, z_new - 0.5_rt);
    const int k_B_old = compute_shifted_shape_factor(sz_B_old, z_old - 0.5_rt, k_B_new);
#endif

    // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
    int dil_E = 1, diu_E = 1;
    if (i_E_old < i_E_new) { dil_E = 0; }
    if (i_E_old > i_E_new) { diu_E = 0; }
#endif
#if defined(WARPX_DIM_3D)
    int djl_E = 1, dju_E = 1;
    if (j_E_old < j_E_new) { djl_E = 0; }
    if (j_E_old > j_E_new) { dju_E = 0; }
#endif
    int dkl_E = 1, dku_E = 1;
    if (k_E_old < k_E_new) { dkl_E = 0; }
    if (k_E_old > k_E_new) { dku_E = 0; }

#if defined(WARPX_DIM_3D)
    int dil_B = 1, diu_B = 1;
    if (i_B_old < i_B_new) { dil_B = 0; }
    if (i_B_old > i_B_new) { diu_B = 0; }
    int djl_B = 1, dju_B = 1;
    if (j_B_old < j_B_new) { djl_B = 0; }
    if (j_B_old > j_B_new) { dju_B = 0; }
    int dkl_B = 1, dku_B = 1;
    if (k_B_old < k_B_new) { dkl_B = 0; }
    if (k_B_old > k_B_new) { dku_B = 0; }
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const Compute_shape_factor< depos_order-1 > compute_shape_factor_By;
    const Compute_shifted_shape_factor< depos_order-1 > compute_shifted_shape_factor_By;
    const int i_By_new = compute_shape_factor_By(sx_By_new+1, x_new - 0.5_rt);
    const int i_By_old = compute_shifted_shape_factor_By(sx_By_old, x_old - 0.5_rt, i_By_new);
    const int k_By_new = compute_shape_factor_By(sz_By_new+1, z_new - 0.5_rt);
    const int k_By_old = compute_shifted_shape_factor_By(sz_By_old, z_old - 0.5_rt, k_By_new);
    int dil_By = 1, diu_By = 1;
    if (i_By_old < i_By_new) { dil_By = 0; }
    if (i_By_old > i_By_new) { diu_By = 0; }
    int dkl_By = 1, dku_By = 1;
    if (k_By_old < k_By_new) { dkl_By = 0; }
    if (k_By_old > k_By_new) { dku_By = 0; }
#endif

#if defined(WARPX_DIM_3D)

    for (int k=dkl_E; k<=depos_order+2-dku_E; k++) {
        for (int j=djl_E; j<=depos_order+2-dju_E; j++) {
            const amrex::Real sdzjk = one_third*(sy_E_new[j]*sz_E_new[k] + sy_E_old[j]*sz_E_old[k])
                               +one_sixth*(sy_E_new[j]*sz_E_old[k] + sy_E_old[j]*sz_E_new[k]);
            amrex::Real sdxi = 0._rt;
            for (int i=dil_E; i<=depos_order+1-diu_E; i++) {
                sdxi += (sx_E_old[i] - sx_E_new[i]);
                auto sdxiov = static_cast<amrex::Real>((x_new - x_old) == 0. ? 1. : sdxi/(x_new - x_old));
                Exp += Ex_arr(lo.x+i_E_new-1+i, lo.y+j_E_new-1+j, lo.z+k_E_new-1+k)*sdxiov*sdzjk;
            }
        }
    }
    for (int k=dkl_E; k<=depos_order+2-dku_E; k++) {
        for (int i=dil_E; i<=depos_order+2-diu_E; i++) {
            const amrex::Real sdyik = one_third*(sx_E_new[i]*sz_E_new[k] + sx_E_old[i]*sz_E_old[k])
                               +one_sixth*(sx_E_new[i]*sz_E_old[k] + sx_E_old[i]*sz_E_new[k]);
            amrex::Real sdyj = 0._rt;
            for (int j=djl_E; j<=depos_order+1-dju_E; j++) {
                sdyj += (sy_E_old[j] - sy_E_new[j]);
                auto sdyjov = static_cast<amrex::Real>((y_new - y_old) == 0. ? 1. : sdyj/(y_new - y_old));
                Eyp += Ey_arr(lo.x+i_E_new-1+i, lo.y+j_E_new-1+j, lo.z+k_E_new-1+k)*sdyjov*sdyik;
            }
        }
    }
    for (int j=djl_E; j<=depos_order+2-dju_E; j++) {
        for (int i=dil_E; i<=depos_order+2-diu_E; i++) {
            const amrex::Real sdzij = one_third*(sx_E_new[i]*sy_E_new[j] + sx_E_old[i]*sy_E_old[j])
                               +one_sixth*(sx_E_new[i]*sy_E_old[j] + sx_E_old[i]*sy_E_new[j]);
            amrex::Real sdzk = 0._rt;
            for (int k=dkl_E; k<=depos_order+1-dku_E; k++) {
                sdzk += (sz_E_old[k] - sz_E_new[k]);
                auto sdzkov = static_cast<amrex::Real>((z_new - z_old) == 0. ? 1. : sdzk/(z_new - z_old));
                Ezp += Ez_arr(lo.x+i_E_new-1+i, lo.y+j_E_new-1+j, lo.z+k_E_new-1+k)*sdzkov*sdzij;
            }
        }
    }
    for (int k=dkl_B; k<=depos_order+2-dku_B; k++) {
        for (int j=djl_B; j<=depos_order+2-dju_B; j++) {
            const amrex::Real sdzjk = one_third*(sy_B_new[j]*sz_B_new[k] + sy_B_old[j]*sz_B_old[k])
                               +one_sixth*(sy_B_new[j]*sz_B_old[k] + sy_B_old[j]*sz_B_new[k]);
            amrex::Real sdxi = 0._rt;
            for (int i=dil_B; i<=depos_order+1-diu_B; i++) {
                sdxi += (sx_B_old[i] - sx_B_new[i]);
                auto sdxiov = static_cast<amrex::Real>((x_new - x_old) == 0. ? 1. : sdxi/(x_new - x_old));
                Bxp += Bx_arr(lo.x+i_B_new-1+i, lo.y+j_B_new-1+j, lo.z+k_B_new-1+k)*sdxiov*sdzjk;
            }
        }
    }
    for (int k=dkl_B; k<=depos_order+2-dku_B; k++) {
        for (int i=dil_B; i<=depos_order+2-diu_B; i++) {
            const amrex::Real sdyik = one_third*(sx_B_new[i]*sz_B_new[k] + sx_B_old[i]*sz_B_old[k])
                               +one_sixth*(sx_B_new[i]*sz_B_old[k] + sx_B_old[i]*sz_B_new[k]);
            amrex::Real sdyj = 0._rt;
            for (int j=djl_B; j<=depos_order+1-dju_B; j++) {
                sdyj += (sy_B_old[j] - sy_B_new[j]);
                auto sdyjov = static_cast<amrex::Real>((y_new - y_old) == 0. ? 1. : sdyj/(y_new - y_old));
                Byp += By_arr(lo.x+i_B_new-1+i, lo.y+j_B_new-1+j, lo.z+k_B_new-1+k)*sdyjov*sdyik;
            }
        }
    }
    for (int j=djl_B; j<=depos_order+2-dju_B; j++) {
        for (int i=dil_B; i<=depos_order+2-diu_B; i++) {
            const amrex::Real sdzij = one_third*(sx_B_new[i]*sy_B_new[j] + sx_B_old[i]*sy_B_old[j])
                               +one_sixth*(sx_B_new[i]*sy_B_old[j] + sx_B_old[i]*sy_B_new[j]);
            amrex::Real sdzk = 0._rt;
            for (int k=dkl_B; k<=depos_order+1-dku_B; k++) {
                sdzk += (sz_B_old[k] - sz_B_new[k]);
                auto sdzkov = static_cast<amrex::Real>((z_new - z_old) == 0. ? 1. : sdzk/(z_new - z_old));
                Bzp += Bz_arr(lo.x+i_B_new-1+i, lo.y+j_B_new-1+j, lo.z+k_E_new-1+k)*sdzkov*sdzij;
            }
        }
    }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    for (int k=dkl_E; k<=depos_order+2-dku_E; k++) {
        const amrex::Real sdzk = 0.5_rt*(sz_E_new[k] + sz_E_old[k]);
        amrex::Real sdxi = 0._rt;
        for (int i=dil_E; i<=depos_order+1-diu_E; i++) {
            sdxi += (sx_E_old[i] - sx_E_new[i]);
            auto sdxiov = static_cast<amrex::Real>((x_new - x_old) == 0. ? 1. : sdxi/(x_new - x_old));
            Exp += Ex_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 0)*sdxiov*sdzk;
            Bzp += Bz_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 0)*sdxiov*sdzk;
        }
    }
    for (int k=dkl_E; k<=depos_order+2-dku_E; k++) {
        for (int i=dil_E; i<=depos_order+2-diu_E; i++) {
            Real const sdyj = (
                one_third*(sx_E_new[i]*sz_E_new[k] + sx_E_old[i]*sz_E_old[k])
               +one_sixth*(sx_E_new[i]*sz_E_old[k] + sx_E_old[i]*sz_E_new[k]));
            Eyp += Ey_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 0)*sdyj;
        }
    }
    for (int i=dil_E; i<=depos_order+2-diu_E; i++) {
        const amrex::Real sdxi = 0.5_rt*(sx_E_new[i] + sx_E_old[i]);
        amrex::Real sdzk = 0._rt;
        for (int k=dkl_E; k<=depos_order+1-dku_E; k++) {
            sdzk += (sz_E_old[k] - sz_E_new[k]);
            auto sdzkov = static_cast<amrex::Real>((z_new - z_old) == 0. ? 1. : sdzk/(z_new - z_old));
            Ezp += Ez_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 0)*sdzkov*sdxi;
            Bxp += Bx_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 0)*sdzkov*sdxi;
        }
    }
    for (int k=dkl_By; k<=depos_order+1-dku_By; k++) {
        for (int i=dil_By; i<=depos_order+1-diu_By; i++) {
            Real const sdyj = (
                one_third*(sx_By_new[i]*sz_By_new[k] + sx_By_old[i]*sz_By_old[k])
               +one_sixth*(sx_By_new[i]*sz_By_old[k] + sx_By_old[i]*sz_By_new[k]));
            Byp += By_arr(lo.x+i_By_new-1+i, lo.y+k_By_new-1+k, 0, 0)*sdyj;
        }
    }

#ifdef WARPX_DIM_RZ
    Complex xy_mid = xy_mid0;

    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {

        // Gather field on particle Exp from field on grid ex_arr
        // Gather field on particle Bzp from field on grid bz_arr
        for (int k=dkl_E; k<=depos_order+2-dku_E; k++) {
            const amrex::Real sdzk = 0.5_rt*(sz_E_new[k] + sz_E_old[k]);
            amrex::Real sdxi = 0._rt;
            for (int i=dil_E; i<=depos_order+1-diu_E; i++) {
                sdxi += (sx_E_old[i] - sx_E_new[i]);
                auto sdxiov = static_cast<amrex::Real>((x_new - x_old) == 0. ? 1. : sdxi/(x_new - x_old));
                const amrex::Real dEx = (+ Ex_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode-1)*xy_mid.real()
                                         - Ex_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode)*xy_mid.imag());
                const amrex::Real dBz = (+ Bz_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode-1)*xy_mid.real()
                                         - Bz_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode)*xy_mid.imag());
                Exp += dEx*sdxiov*sdzk;
                Bzp += dBz*sdxiov*sdzk;
            }
        }
        // Gather field on particle Eyp from field on grid ey_arr
        for (int k=dkl_E; k<=depos_order+2-dku_E; k++) {
            for (int i=dil_E; i<=depos_order+2-diu_E; i++) {
                Real const sdyj = (
                    one_third*(sx_E_new[i]*sz_E_new[k] + sx_E_old[i]*sz_E_old[k])
                   +one_sixth*(sx_E_new[i]*sz_E_old[k] + sx_E_old[i]*sz_E_new[k]));
                const amrex::Real dEy = (+ Ey_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode-1)*xy_mid.real()
                                         - Ey_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode)*xy_mid.imag());
                Eyp += dEy*sdyj;
            }
        }
        // Gather field on particle Ezp from field on grid ez_arr
        // Gather field on particle Bxp from field on grid bx_arr
        for (int i=dil_E; i<=depos_order+2-diu_E; i++) {
            const amrex::Real sdxi = 0.5_rt*(sx_E_new[i] + sx_E_old[i]);
            amrex::Real sdzk = 0._rt;
            for (int k=dkl_E; k<=depos_order+1-dku_E; k++) {
                sdzk += (sz_E_old[k] - sz_E_new[k]);
                auto sdzkov = static_cast<amrex::Real>((z_new - z_old) == 0. ? 1. : sdzk/(z_new - z_old));
                const amrex::Real dEz = (+ Ez_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode-1)*xy_mid.real()
                                         - Ez_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode)*xy_mid.imag());
                const amrex::Real dBx = (+ Bx_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode-1)*xy_mid.real()
                                         - Bx_arr(lo.x+i_E_new-1+i, lo.y+k_E_new-1+k, 0, 2*imode)*xy_mid.imag());
                Ezp += dEz*sdzkov*sdxi;
                Bxp += dBx*sdzkov*sdxi;
            }
        }
        // Gather field on particle Byp from field on grid by_arr
        for (int k=dkl_By; k<=depos_order+1-dku_By; k++) {
            for (int i=dil_By; i<=depos_order+1-diu_By; i++) {
                Real const sdyj = (
                    one_third*(sx_By_new[i]*sz_By_new[k] + sx_By_old[i]*sz_By_old[k])
                   +one_sixth*(sx_By_new[i]*sz_By_old[k] + sx_By_old[i]*sz_By_new[k]));
                const amrex::Real dBy = (+ By_arr(lo.x+i_By_new-1+i, lo.y+k_By_new-1+k, 0, 2*imode-1)*xy_mid.real()
                                         - By_arr(lo.x+i_By_new-1+i, lo.y+k_By_new-1+k, 0, 2*imode)*xy_mid.imag());
                Byp += dBy*sdyj;
            }
        }
        xy_mid = xy_mid*xy_mid0;
    }

    // Convert Exp and Eyp (which are actually Er and Etheta) to Ex and Ey
    const amrex::Real Exp_save = Exp;
    Exp = costheta_mid*Exp - sintheta_mid*Eyp;
    Eyp = costheta_mid*Eyp + sintheta_mid*Exp_save;
    const amrex::Real Bxp_save = Bxp;
    Bxp = costheta_mid*Bxp - sintheta_mid*Byp;
    Byp = costheta_mid*Byp + sintheta_mid*Bxp_save;

#endif

#elif defined(WARPX_DIM_1D_Z)

    for (int k=dkl_E; k<=depos_order+2-dku_E; k++) {
        amrex::Real const sdzk = 0.5_rt*(sz_E_old[k] + sz_E_new[k]);
        Exp += Ex_arr(lo.x+k_E_new-1+k, 0, 0, 0)*sdzk;
        Eyp += Ey_arr(lo.x+k_E_new-1+k, 0, 0, 0)*sdzk;
        Bzp += Bz_arr(lo.x+k_E_new-1+k, 0, 0, 0)*sdzk;
    }
    amrex::Real sdzk = 0._rt;
    for (int k=dkl_E; k<=depos_order+1-dku_E; k++) {
        sdzk += (sz_E_old[k] - sz_E_new[k]);
        auto sdzkov = static_cast<amrex::Real>((z_new - z_old) == 0. ? 1. : sdzk/(z_new - z_old));
        Bxp += Bx_arr(lo.x+k_E_new-1+k, 0, 0, 0)*sdzkov;
        Byp += By_arr(lo.x+k_E_new-1+k, 0, 0, 0)*sdzkov;
        Ezp += Ez_arr(lo.x+k_E_new-1+k, 0, 0, 0)*sdzkov;
    }
#endif
}

/**
 * \brief Energy conserving field gather for thread thread_num for the implicit scheme
 *        This uses the same stencil for the gather that is used for Villasenor current deposition.
 *        The magnetic field is deposited using direct deposition.
 *
 * \tparam depos_order              Particle shape order
 * \param xp_n,yp_n,zp_n            Particle position coordinates at start of step
 * \param xp_nph,yp_nph,zp_nph      Particle position coordinates at half step
 * \param Exp,Eyp,Ezp               Electric field on particles.
 * \param Bxp,Byp,Bzp               Magnetic field on particles.
 * \param Ex_arr,Ey_arr,Ez_arr      Array4 of the electric field, either full array or tile.
 * \param Bx_arr,By_arr,Bz_arr      Array4 of the magnetic field, either full array or tile.
 * \param Ex_type,Ey_type,Ez_type   IndexType of the electric field
 * \param Bx_type,By_type,Bz_type   IndexType of the magnetic field
 * \param dinv                      3D cell size inverse
 * \param xyzmin                    The lower bounds of the domain
 * \param lo                        Index lower bounds of domain.
 * \param n_rz_azimuthal_modes      Number of azimuthal modes when using RZ geometry
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherPicnicShapeN (
                     [[maybe_unused]] const amrex::ParticleReal xp_n,
                     [[maybe_unused]] const amrex::ParticleReal yp_n,
                     const amrex::ParticleReal zp_n,
                     [[maybe_unused]] const amrex::ParticleReal xp_nph,
                     [[maybe_unused]] const amrex::ParticleReal yp_nph,
                     const amrex::ParticleReal zp_nph,
                     amrex::ParticleReal& Exp,
                     amrex::ParticleReal& Eyp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& Ex_arr,
                     amrex::Array4<amrex::Real const> const& Ey_arr,
                     amrex::Array4<amrex::Real const> const& Ez_arr,
                     amrex::Array4<amrex::Real const> const& Bx_arr,
                     amrex::Array4<amrex::Real const> const& By_arr,
                     amrex::Array4<amrex::Real const> const& Bz_arr,
                     [[maybe_unused]] const amrex::IndexType Ex_type,
                     [[maybe_unused]] const amrex::IndexType Ey_type,
                     [[maybe_unused]] const amrex::IndexType Ez_type,
                     [[maybe_unused]] const amrex::IndexType Bx_type,
                     [[maybe_unused]] const amrex::IndexType By_type,
                     [[maybe_unused]] const amrex::IndexType Bz_type,
                     const amrex::XDim3 & dinv,
                     const amrex::XDim3 & xyzmin,
                     const amrex::Dim3& lo,
                     const int n_rz_azimuthal_modes)
{
    using namespace amrex;
#if !defined(WARPX_DIM_RZ)
    ignore_unused(n_rz_azimuthal_modes);
#endif

#if !defined(WARPX_DIM_1D_Z)
    const ParticleReal xp_np1 = 2._prt*xp_nph - xp_n;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
    const ParticleReal yp_np1 = 2._prt*yp_nph - yp_n;
#endif
    const ParticleReal zp_np1 = 2._prt*zp_nph - zp_n;

#if !defined(WARPX_DIM_1D_Z)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
    amrex::Real const xp_new = xp_np1;
    amrex::Real const yp_new = yp_np1;
    amrex::Real const xp_mid = xp_nph;
    amrex::Real const yp_mid = yp_nph;
    amrex::Real const xp_old = xp_n;
    amrex::Real const yp_old = yp_n;
    amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
    amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
    amrex::Real const rp_mid = (rp_new + rp_old)/2._rt;
    amrex::Real costheta_mid, sintheta_mid;
    if (rp_mid > 0._rt) {
        costheta_mid = xp_mid/rp_mid;
        sintheta_mid = yp_mid/rp_mid;
    } else {
        costheta_mid = 1._rt;
        sintheta_mid = 0._rt;
    }
    const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
    // Keep these double to avoid bug in single precision
    double const x_new = (rp_new - xyzmin.x)*dinv.x;
    double const x_old = (rp_old - xyzmin.x)*dinv.x;
    double const x_bar = (rp_mid - xyzmin.x)*dinv.x;
#elif !defined(WARPX_DIM_1D_Z)
    // Keep these double to avoid bug in single precision
    double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
    double const x_old = (xp_n - xyzmin.x)*dinv.x;
    double const x_bar = (xp_nph - xyzmin.x)*dinv.x;
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double const y_new = (yp_np1 - xyzmin.y)*dinv.y;
    double const y_old = (yp_n   - xyzmin.y)*dinv.y;
    double const y_bar = (yp_nph - xyzmin.y)*dinv.y;
#endif
    // Keep these double to avoid bug in single precision
    double const z_new = (zp_np1 - xyzmin.z)*dinv.z;
    double const z_old = (zp_n   - xyzmin.z)*dinv.z;
    double const z_bar = (zp_nph - xyzmin.z)*dinv.z;

    // 1) Determine the number of segments.
    // 2) Loop over segments and gather electric field.
    // 3) Gather magnetic field.

    // cell crossings are defined at cell edges if depos_order is odd
    // cell crossings are defined at cell centers if depos_order is even

    int num_segments = 1;
    double shift = 0.0;
    if ( (depos_order % 2) == 0 ) { shift = 0.5; }

#if defined(WARPX_DIM_3D)

    // compute cell crossings in X-direction
    const auto i_old = static_cast<int>(x_old-shift);
    const auto i_new = static_cast<int>(x_new-shift);
    const int cell_crossings_x = std::abs(i_new-i_old);
    num_segments += cell_crossings_x;

    // compute cell crossings in Y-direction
    const auto j_old = static_cast<int>(y_old-shift);
    const auto j_new = static_cast<int>(y_new-shift);
    const int cell_crossings_y = std::abs(j_new-j_old);
    num_segments += cell_crossings_y;

    // compute cell crossings in Z-direction
    const auto k_old = static_cast<int>(z_old-shift);
    const auto k_new = static_cast<int>(z_new-shift);
    const int cell_crossings_z = std::abs(k_new-k_old);
    num_segments += cell_crossings_z;

    // need to assert that the number of cell crossings in each direction
    // is within the range permitted by the number of guard cells
    // e.g., if (num_segments > 7) ...

    // compute total change in particle position and the initial cell
    // locations in each direction used to find the position at cell crossings.
    const double dxp = x_new - x_old;
    const double dyp = y_new - y_old;
    const double dzp = z_new - z_old;
    const auto dirX_sign = static_cast<double>(dxp < 0. ? -1. : 1.);
    const auto dirY_sign = static_cast<double>(dyp < 0. ? -1. : 1.);
    const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
    double Xcell = 0., Ycell = 0., Zcell = 0.;
    if (num_segments > 1) {
        Xcell = static_cast<double>(i_old) + shift + 0.5*(1.-dirX_sign);
        Ycell = static_cast<double>(j_old) + shift + 0.5*(1.-dirY_sign);
        Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);
    }

    // loop over the number of segments and deposit
    const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
    const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
    double dxp_seg, dyp_seg, dzp_seg;
    double x0_new, y0_new, z0_new;
    double x0_old = x_old;
    double y0_old = y_old;
    double z0_old = z_old;

    for (int ns=0; ns<num_segments; ns++) {

        if (ns == num_segments-1) { // final segment

            x0_new = x_new;
            y0_new = y_new;
            z0_new = z_new;
            dxp_seg = x0_new - x0_old;
            dyp_seg = y0_new - y0_old;
            dzp_seg = z0_new - z0_old;

        }
        else {

            x0_new = Xcell + dirX_sign;
            y0_new = Ycell + dirY_sign;
            z0_new = Zcell + dirZ_sign;
            dxp_seg = x0_new - x0_old;
            dyp_seg = y0_new - y0_old;
            dzp_seg = z0_new - z0_old;

            if ( (dyp == 0. || std::abs(dxp_seg) < std::abs(dxp/dyp*dyp_seg))
              && (dzp == 0. || std::abs(dxp_seg) < std::abs(dxp/dzp*dzp_seg)) ) {
                Xcell = x0_new;
                dyp_seg = dyp/dxp*dxp_seg;
                dzp_seg = dzp/dxp*dxp_seg;
                y0_new = y0_old + dyp_seg;
                z0_new = z0_old + dzp_seg;
            }
            else if (dzp == 0. || std::abs(dyp_seg) < std::abs(dyp/dzp*dzp_seg)) {
                Ycell = y0_new;
                dxp_seg = dxp/dyp*dyp_seg;
                dzp_seg = dzp/dyp*dyp_seg;
                x0_new = x0_old + dxp_seg;
                z0_new = z0_old + dzp_seg;
            }
            else {
                Zcell = z0_new;
                dxp_seg = dxp/dzp*dzp_seg;
                dyp_seg = dyp/dzp*dzp_seg;
                x0_new = x0_old + dxp_seg;
                y0_new = y0_old + dyp_seg;
            }

        }

        // compute the segment factors (each equal to dt_seg/dt for nonzero dxp, dyp, or dzp)
        const auto seg_factor_x = static_cast<double>(dxp == 0. ? 1. : dxp_seg/dxp);
        const auto seg_factor_y = static_cast<double>(dyp == 0. ? 1. : dyp_seg/dyp);
        const auto seg_factor_z = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

        // compute cell-based weights using the average segment position
        double sx_cell[depos_order] = {0.};
        double sy_cell[depos_order] = {0.};
        double sz_cell[depos_order] = {0.};
        double const x0_bar = (x0_new + x0_old)/2.0;
        double const y0_bar = (y0_new + y0_old)/2.0;
        double const z0_bar = (z0_new + z0_old)/2.0;
        const int i0_cell = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );
        const int j0_cell = compute_shape_factor_cell( sy_cell, y0_bar-0.5 );
        const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

        if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
            const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
            double sx_old_cell[depos_order] = {0.};
            double sx_new_cell[depos_order] = {0.};
            double sy_old_cell[depos_order] = {0.};
            double sy_new_cell[depos_order] = {0.};
            double sz_old_cell[depos_order] = {0.};
            double sz_new_cell[depos_order] = {0.};
            const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
            const int j0_cell_2 = compute_shape_factors_cell( sy_old_cell, sy_new_cell, y0_old-0.5, y0_new-0.5 );
            const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
            ignore_unused(i0_cell_2, j0_cell_2, k0_cell_2);
            for (int m=0; m<depos_order; m++) {
                sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                sy_cell[m] = (4.0*sy_cell[m] + sy_old_cell[m] + sy_new_cell[m])/6.0;
                sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
            }
        }

        // compute node-based weights using the old and new segment positions
        double sx_old_node[depos_order+1] = {0.};
        double sx_new_node[depos_order+1] = {0.};
        double sy_old_node[depos_order+1] = {0.};
        double sy_new_node[depos_order+1] = {0.};
        double sz_old_node[depos_order+1] = {0.};
        double sz_new_node[depos_order+1] = {0.};
        const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
        const int j0_node = compute_shape_factors_node( sy_old_node, sy_new_node, y0_old, y0_new );
        const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

        // gather Ex for this segment
        amrex::Real weight;
        for (int i=0; i<=depos_order-1; i++) {
            for (int j=0; j<=depos_order; j++) {
                for (int k=0; k<=depos_order; k++) {
                    weight = sx_cell[i]*( sy_old_node[j]*sz_old_node[k]*one_third
                                        + sy_old_node[j]*sz_new_node[k]*one_sixth
                                        + sy_new_node[j]*sz_old_node[k]*one_sixth
                                        + sy_new_node[j]*sz_new_node[k]*one_third )*seg_factor_x;
                    Exp += Ex_arr(lo.x+i0_cell+i, lo.y+j0_node+j, lo.z+k0_node+k)*weight;
                }
            }
        }

        // gather Ey for this segment
        for (int i=0; i<=depos_order; i++) {
            for (int j=0; j<=depos_order-1; j++) {
                for (int k=0; k<=depos_order; k++) {
                    weight = sy_cell[j]*( sx_old_node[i]*sz_old_node[k]*one_third
                                        + sx_old_node[i]*sz_new_node[k]*one_sixth
                                        + sx_new_node[i]*sz_old_node[k]*one_sixth
                                        + sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                    Eyp += Ey_arr(lo.x+i0_node+i, lo.y+j0_cell+j, lo.z+k0_node+k)*weight;
                }
            }
        }

        // gather Ez for this segment
        for (int i=0; i<=depos_order; i++) {
            for (int j=0; j<=depos_order; j++) {
                for (int k=0; k<=depos_order-1; k++) {
                    weight = sz_cell[k]*( sx_old_node[i]*sy_old_node[j]*one_third
                                        + sx_old_node[i]*sy_new_node[j]*one_sixth
                                        + sx_new_node[i]*sy_old_node[j]*one_sixth
                                        + sx_new_node[i]*sy_new_node[j]*one_third )*seg_factor_z;
                    Ezp += Ez_arr(lo.x+i0_node+i, lo.y+j0_node+j, lo.z+k0_cell+k)*weight;
                }
            }
        }

        // update old segment values
        if (ns < num_segments-1) {
            x0_old = x0_new;
            y0_old = y0_new;
            z0_old = z0_new;
        }

    } // end loop over segments

    // gather magnetic field
    const int depos_order_B = 1; // second template parameter?
    const Compute_shape_factor< depos_order_B > compute_shape_factor_B;
    double sz_bar_node[depos_order_B+1] = {0.};
    double sz_bar_cell[depos_order_B+1] = {0.};
    const int k_bar_node = compute_shape_factor_B(sz_bar_node, z_bar);
    const int k_bar_cell = compute_shape_factor_B(sz_bar_cell, z_bar-0.5);
    double sy_bar_node[depos_order_B+1] = {0.};
    double sy_bar_cell[depos_order_B+1] = {0.};
    const int j_bar_node = compute_shape_factor_B(sy_bar_node, y_bar);
    const int j_bar_cell = compute_shape_factor_B(sy_bar_cell, y_bar-0.5);
    double sx_bar_node[depos_order_B+1] = {0.};
    double sx_bar_cell[depos_order_B+1] = {0.};
    const int i_bar_node = compute_shape_factor_B(sx_bar_node, x_bar);
    const int i_bar_cell = compute_shape_factor_B(sx_bar_cell, x_bar-0.5);

    amrex::Real weight;
    for (int i=0; i<=depos_order_B; i++) {
        for (int j=0; j<=depos_order_B; j++) {
            for (int k=0; k<=depos_order_B; k++) {
                weight = static_cast<amrex::Real>(sx_bar_node[i]*sy_bar_cell[j]*sz_bar_cell[k]);
                Bxp += Bx_arr(lo.x+i_bar_node+i, lo.y+j_bar_cell+j, lo.z+k_bar_cell+k)*weight;
                //
                weight = static_cast<amrex::Real>(sx_bar_cell[i]*sy_bar_node[j]*sz_bar_cell[k]);
                Byp += By_arr(lo.x+i_bar_cell+i, lo.y+j_bar_node+j, lo.z+k_bar_cell+k)*weight;
                //
                weight = static_cast<amrex::Real>(sx_bar_cell[i]*sy_bar_cell[j]*sz_bar_node[k]);
                Bzp += Bz_arr(lo.x+i_bar_cell+i, lo.y+j_bar_cell+j, lo.z+k_bar_node+k)*weight;
            }
        }
    }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    // compute cell crossings in X-direction
    const auto i_old = static_cast<int>(x_old-shift);
    const auto i_new = static_cast<int>(x_new-shift);
    const int cell_crossings_x = std::abs(i_new-i_old);
    num_segments += cell_crossings_x;

    // compute cell crossings in Z-direction
    const auto k_old = static_cast<int>(z_old-shift);
    const auto k_new = static_cast<int>(z_new-shift);
    const int cell_crossings_z = std::abs(k_new-k_old);
    num_segments += cell_crossings_z;

    // need to assert that the number of cell crossings in each direction
    // is within the range permitted by the number of guard cells
    // e.g., if (num_segments > 5) ...

    // compute total change in particle position and the initial cell
    // locations in each direction used to find the position at cell crossings.
    const double dxp = x_new - x_old;
    const double dzp = z_new - z_old;
    const auto dirX_sign = static_cast<double>(dxp < 0. ? -1. : 1.);
    const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
    double Xcell = 0., Zcell = 0.;
    if (num_segments > 1) {
        Xcell = static_cast<double>(i_old) + shift + 0.5*(1.-dirX_sign);
        Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);
    }

    // loop over the number of segments and deposit
    const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
    const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
    double dxp_seg, dzp_seg;
    double x0_new, z0_new;
    double x0_old = x_old;
    double z0_old = z_old;

    for (int ns=0; ns<num_segments; ns++) {

        if (ns == num_segments-1) { // final segment

            x0_new = x_new;
            z0_new = z_new;
            dxp_seg = x0_new - x0_old;
            dzp_seg = z0_new - z0_old;

        }
        else {

            x0_new = Xcell + dirX_sign;
            z0_new = Zcell + dirZ_sign;
            dxp_seg = x0_new - x0_old;
            dzp_seg = z0_new - z0_old;

            if (dzp == 0. || std::abs(dxp_seg) < std::abs(dxp/dzp*dzp_seg)) {
                Xcell = x0_new;
                dzp_seg = dzp/dxp*dxp_seg;
                z0_new = z0_old + dzp_seg;
            }
            else {
                Zcell = z0_new;
                dxp_seg = dxp/dzp*dzp_seg;
                x0_new = x0_old + dxp_seg;
            }

        }

        // compute the segment factors (each equal to dt_seg/dt for nonzero dxp, or dzp)
        const auto seg_factor_x = static_cast<double>(dxp == 0. ? 1. : dxp_seg/dxp);
        const auto seg_factor_z = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

        // compute cell-based weights using the average segment position
        double sx_cell[depos_order] = {0.};
        double sz_cell[depos_order] = {0.};
        double const x0_bar = (x0_new + x0_old)/2.0;
        double const z0_bar = (z0_new + z0_old)/2.0;
        const int i0_cell = compute_shape_factor_cell(sx_cell, x0_bar-0.5);
        const int k0_cell = compute_shape_factor_cell(sz_cell, z0_bar-0.5);

        if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
            const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
            double sx_old_cell[depos_order] = {0.};
            double sx_new_cell[depos_order] = {0.};
            double sz_old_cell[depos_order] = {0.};
            double sz_new_cell[depos_order] = {0.};
            const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
            const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
            ignore_unused(i0_cell_2, k0_cell_2);
            for (int m=0; m<depos_order; m++) {
                sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
            }
        }

        // compute node-based weights using the old and new segment positions
        double sx_old_node[depos_order+1] = {0.};
        double sx_new_node[depos_order+1] = {0.};
        double sz_old_node[depos_order+1] = {0.};
        double sz_new_node[depos_order+1] = {0.};
        const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
        const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

        // gather Ex for this segment
        amrex::Real weight;
        for (int i=0; i<=depos_order-1; i++) {
            for (int k=0; k<=depos_order; k++) {
                weight = sx_cell[i]*(sz_old_node[k] + sz_new_node[k])/2.0_rt*seg_factor_x;
                Exp += Ex_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 0)*weight;
#if defined(WARPX_DIM_RZ)
                Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                    const auto dEx = (+ Ex_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode-1)*xy_mid.real()
                                      - Ex_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode)*xy_mid.imag());
                    Exp += weight*dEx;
                    xy_mid = xy_mid*xy_mid0;
                }
#endif
            }
        }

        // gather out-of-plane Ey for this segment
        const auto seg_factor_y = std::min(seg_factor_x,seg_factor_z);
        for (int i=0; i<=depos_order; i++) {
            for (int k=0; k<=depos_order; k++) {
                weight = ( sx_old_node[i]*sz_old_node[k]*one_third
                       +   sx_old_node[i]*sz_new_node[k]*one_sixth
                       +   sx_new_node[i]*sz_old_node[k]*one_sixth
                       +   sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                Eyp += Ey_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 0)*weight;
#if defined(WARPX_DIM_RZ)
                Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                    const auto dEy = (+ Ey_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode-1)*xy_mid.real()
                                      - Ey_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode)*xy_mid.imag());
                    Eyp += weight*dEy;
                    xy_mid = xy_mid*xy_mid0;
                }
#endif
            }
        }

        // gather Ez for this segment
        for (int i=0; i<=depos_order; i++) {
            for (int k=0; k<=depos_order-1; k++) {
                weight = sz_cell[k]*(sx_old_node[i] + sx_new_node[i])/2.0_rt*seg_factor_z;
                Ezp += Ez_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 0)*weight;
#if defined(WARPX_DIM_RZ)
                Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                    const auto dEz = (+ Ez_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode-1)*xy_mid.real()
                                      - Ez_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode)*xy_mid.imag());
                    Ezp += weight*dEz;
                    xy_mid = xy_mid*xy_mid0;
                }
#endif
            }
        }

        // update old segment values
        if (ns < num_segments-1) {
            x0_old = x0_new;
            z0_old = z0_new;
        }

    }

    // gather magnetic field and out-of-plane electric field
    const int depos_order_B = 1; // second template parameter?
    const Compute_shape_factor< depos_order_B > compute_shape_factor_B;
    double sz_bar_node[depos_order_B+1] = {0.};
    double sz_bar_cell[depos_order_B+1] = {0.};
    const int k_bar_node = compute_shape_factor_B(sz_bar_node, z_bar);
    const int k_bar_cell = compute_shape_factor_B(sz_bar_cell, z_bar-0.5);
    double sx_bar_node[depos_order_B+1] = {0.};
    double sx_bar_cell[depos_order_B+1] = {0.};
    const int i_bar_node = compute_shape_factor_B(sx_bar_node, x_bar);
    const int i_bar_cell = compute_shape_factor_B(sx_bar_cell, x_bar-0.5);

    for (int i=0; i<=depos_order_B; i++) {
        for (int k=0; k<=depos_order_B; k++) {
            const auto weight_Bz = static_cast<amrex::Real>(sx_bar_cell[i]*sz_bar_node[k]);
            Bzp += Bz_arr(lo.x+i_bar_cell+i, lo.y+k_bar_node+k, 0, 0)*weight_Bz;
            //
            const auto weight_Bx = static_cast<amrex::Real>(sx_bar_node[i]*sz_bar_cell[k]);
            Bxp += Bx_arr(lo.x+i_bar_node+i, lo.y+k_bar_cell+k, 0, 0)*weight_Bx;
            //
            const auto weight_By = static_cast<amrex::Real>(sx_bar_cell[i]*sz_bar_cell[k]);
            Byp += By_arr(lo.x+i_bar_cell+i, lo.y+k_bar_cell+k, 0, 0)*weight_By;
#if defined(WARPX_DIM_RZ)
            Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                const auto dBx = (+ Bx_arr(lo.x+i_bar_node+i, lo.y+k_bar_cell+k, 0, 2*imode-1)*xy_mid.real()
                                  - Bx_arr(lo.x+i_bar_node+i, lo.y+k_bar_cell+k, 0, 2*imode)*xy_mid.imag());
                const auto dBy = (+ By_arr(lo.x+i_bar_cell+i, lo.y+k_bar_cell+k, 0, 2*imode-1)*xy_mid.real()
                                  - By_arr(lo.x+i_bar_cell+i, lo.y+k_bar_cell+k, 0, 2*imode)*xy_mid.imag());
                const auto dBz = (+ Bz_arr(lo.x+i_bar_cell+i, lo.y+k_bar_node+k, 0, 2*imode-1)*xy_mid.real()
                                  - Bz_arr(lo.x+i_bar_cell+i, lo.y+k_bar_node+k, 0, 2*imode)*xy_mid.imag());
                Bxp += weight_Bx*dBx;
                Byp += weight_By*dBy;
                Bzp += weight_Bz*dBz;
                xy_mid = xy_mid*xy_mid0;
            }
#endif
        }
    }

#ifdef WARPX_DIM_RZ

    // Convert Exp and Eyp (which are actually Er and Etheta) to Ex and Ey
    const amrex::Real Exp_save = Exp;
    Exp = costheta_mid*Exp - sintheta_mid*Eyp;
    Eyp = costheta_mid*Eyp + sintheta_mid*Exp_save;
    const amrex::Real Bxp_save = Bxp;
    Bxp = costheta_mid*Bxp - sintheta_mid*Byp;
    Byp = costheta_mid*Byp + sintheta_mid*Bxp_save;

#endif

#elif defined(WARPX_DIM_1D_Z)

    // compute cell crossings in Z-direction
    const auto k_old = static_cast<int>(z_old-shift);
    const auto k_new = static_cast<int>(z_new-shift);
    const int cell_crossings_z = std::abs(k_new-k_old);
    num_segments += cell_crossings_z;

    // need to assert that the number of cell crossings in each direction
    // is within the range permitted by the number of guard cells
    // e.g., if (num_segments > 3) ...

    // compute dzp and the initial cell location used to find the cell crossings.
    double const dzp = z_new - z_old;
    const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
    double Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);

    // loop over the number of segments and deposit
    const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
    const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
    double dzp_seg;
    double z0_new;
    double z0_old = z_old;

    for (int ns=0; ns<num_segments; ns++) {

        if (ns == num_segments-1) { // final segment
            z0_new = z_new;
            dzp_seg = z0_new - z0_old;
        }
        else {
            Zcell = Zcell + dirZ_sign;
            z0_new = Zcell;
            dzp_seg = z0_new - z0_old;
        }

        // compute the segment factor (equal to dt_seg/dt for nonzero dzp)
        const auto seg_factor = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

        // compute cell-based weights using the average segment position
        double sz_cell[depos_order] = {0.};
        double const z0_bar = (z0_new + z0_old)/2.0;
        const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

        if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
            const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
            double sz_old_cell[depos_order] = {0.};
            double sz_new_cell[depos_order] = {0.};
            const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
            ignore_unused(k0_cell_2);
            for (int m=0; m<depos_order; m++) {
                sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
            }
        }

        // compute node-based weights using the old and new segment positions
        double sz_old_node[depos_order+1] = {0.};
        double sz_new_node[depos_order+1] = {0.};
        const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

        // gather out-of-plane Ex and Ey for this segment
        for (int k=0; k<=depos_order; k++) {
            auto weight = 0.5_rt*(sz_old_node[k] + sz_new_node[k])*seg_factor;
            Exp += Ex_arr(lo.x+k0_node+k, 0, 0)*weight;
            Eyp += Ey_arr(lo.x+k0_node+k, 0, 0)*weight;
        }

        // gather Ez for this segment
        for (int k=0; k<=depos_order-1; k++) {
            auto weight = sz_cell[k]*seg_factor;
            Ezp += Ez_arr(lo.x+k0_cell+k, 0, 0)*weight;
        }

        // update old segment values
        if (ns < num_segments-1) {
            z0_old = z0_new;
        }

    }

    // gather magnetic field
    const int depos_order_B = 1; // second template parameter?
    const Compute_shape_factor< depos_order_B > compute_shape_factor_B;
    double sz_bar_node[depos_order_B+1] = {0.};
    double sz_bar_cell[depos_order_B+1] = {0.};
    const int k_bar_node = compute_shape_factor_B(sz_bar_node, z_bar);
    const int k_bar_cell = compute_shape_factor_B(sz_bar_cell, z_bar-0.5_rt);

    amrex::Real weight;
    for (int k=0; k<=depos_order_B; k++) {
        weight = static_cast<amrex::Real>(sz_bar_node[k]);
        Bzp += Bz_arr(lo.x+k_bar_node+k, 0, 0)*weight;
        //
        weight = static_cast<amrex::Real>(sz_bar_cell[k]);
        Bxp += Bx_arr(lo.x+k_bar_cell+k, 0, 0)*weight;
        Byp += By_arr(lo.x+k_bar_cell+k, 0, 0)*weight;
    }

#endif
}

/**
 * \brief Field gather for particles
 *
 * \tparam depos_order         deposition order
 * \tparam lower_in_v          lower shape order in parallel direction (Galerkin)
 * \param getPosition          A functor for returning the particle position.
 * \param getExternalEB        A functor for assigning the external E and B fields.
 * \param Exp,Eyp,Ezp          Pointer to array of electric field on particles.
 * \param Bxp,Byp,Bzp          Pointer to array of magnetic field on particles.
 * \param exfab,eyfab,ezfab    Array4 of the electric field, either full array or tile.
 * \param bxfab,byfab,bzfab    Array4 of the magnetic field, either full array or tile.
 * \param np_to_gather         Number of particles for which field is gathered.
 * \param dinv                 3D cell size inverse
 * \param xyzmin               The lower bounds of the domain
 * \param lo                   Index lower bounds of domain.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry
 */
template <int depos_order, int lower_in_v>
void doGatherShapeN(const GetParticlePosition<PIdx>& getPosition,
                    const GetExternalEBField& getExternalEB,
                    amrex::ParticleReal * const Exp, amrex::ParticleReal * const Eyp,
                    amrex::ParticleReal * const Ezp, amrex::ParticleReal * const Bxp,
                    amrex::ParticleReal * const Byp, amrex::ParticleReal * const Bzp,
                    amrex::FArrayBox const * const exfab,
                    amrex::FArrayBox const * const eyfab,
                    amrex::FArrayBox const * const ezfab,
                    amrex::FArrayBox const * const bxfab,
                    amrex::FArrayBox const * const byfab,
                    amrex::FArrayBox const * const bzfab,
                    const long np_to_gather,
                    const amrex::XDim3 & dinv,
                    const amrex::XDim3 & xyzmin,
                    const amrex::Dim3 lo,
                    const int n_rz_azimuthal_modes)
{

    amrex::Array4<const amrex::Real> const& ex_arr = exfab->array();
    amrex::Array4<const amrex::Real> const& ey_arr = eyfab->array();
    amrex::Array4<const amrex::Real> const& ez_arr = ezfab->array();
    amrex::Array4<const amrex::Real> const& bx_arr = bxfab->array();
    amrex::Array4<const amrex::Real> const& by_arr = byfab->array();
    amrex::Array4<const amrex::Real> const& bz_arr = bzfab->array();

    amrex::IndexType const ex_type = exfab->box().ixType();
    amrex::IndexType const ey_type = eyfab->box().ixType();
    amrex::IndexType const ez_type = ezfab->box().ixType();
    amrex::IndexType const bx_type = bxfab->box().ixType();
    amrex::IndexType const by_type = byfab->box().ixType();
    amrex::IndexType const bz_type = bzfab->box().ixType();

    // Loop over particles and gather fields from
    // {e,b}{x,y,z}_arr to {E,B}{xyz}p.
    amrex::ParallelFor(
        np_to_gather,
        [=] AMREX_GPU_DEVICE (long ip) {

            amrex::ParticleReal xp, yp, zp;
            getPosition(ip, xp, yp, zp);
            getExternalEB(ip, Exp[ip], Eyp[ip], Ezp[ip], Bxp[ip], Byp[ip], Bzp[ip]);

            doGatherShapeN<depos_order, lower_in_v>(
                xp, yp, zp, Exp[ip], Eyp[ip], Ezp[ip], Bxp[ip], Byp[ip], Bzp[ip],
                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        }
        );
}

/**
 * \brief Field gather for a single particle
 *
 * \param xp,yp,zp                Particle position coordinates
 * \param Exp,Eyp,Ezp             Electric field on particles.
 * \param Bxp,Byp,Bzp             Magnetic field on particles.
 * \param ex_arr,ey_arr,ez_arr    Array4 of the electric field, either full array or tile.
 * \param bx_arr,by_arr,bz_arr    Array4 of the magnetic field, either full array or tile.
 * \param ex_type,ey_type,ez_type IndexType of the electric field
 * \param bx_type,by_type,bz_type IndexType of the magnetic field
 * \param dinv                    3D cell size inverse
 * \param xyzmin                  The lower bounds of the domain
 * \param lo                      Index lower bounds of domain.
 * \param n_rz_azimuthal_modes    Number of azimuthal modes when using RZ geometry
 * \param nox                     order of the particle shape function
 * \param galerkin_interpolation  whether to use lower order in v
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherShapeN (const amrex::ParticleReal xp,
                     const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal& Exp,
                     amrex::ParticleReal& Eyp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& ex_arr,
                     amrex::Array4<amrex::Real const> const& ey_arr,
                     amrex::Array4<amrex::Real const> const& ez_arr,
                     amrex::Array4<amrex::Real const> const& bx_arr,
                     amrex::Array4<amrex::Real const> const& by_arr,
                     amrex::Array4<amrex::Real const> const& bz_arr,
                     const amrex::IndexType ex_type,
                     const amrex::IndexType ey_type,
                     const amrex::IndexType ez_type,
                     const amrex::IndexType bx_type,
                     const amrex::IndexType by_type,
                     const amrex::IndexType bz_type,
                     const amrex::XDim3 & dinv,
                     const amrex::XDim3 & xyzmin,
                     const amrex::Dim3& lo,
                     const int n_rz_azimuthal_modes,
                     const int nox,
                     const bool galerkin_interpolation)
{
    if (galerkin_interpolation) {
        if (nox == 1) {
            doGatherShapeN<1,1>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 2) {
            doGatherShapeN<2,1>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 3) {
            doGatherShapeN<3,1>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 4) {
            doGatherShapeN<4,1>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        }
    } else {
        if (nox == 1) {
            doGatherShapeN<1,0>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 2) {
            doGatherShapeN<2,0>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 3) {
            doGatherShapeN<3,0>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 4) {
            doGatherShapeN<4,0>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        }
    }
}


/**
 * \brief Field gather for a single particle
 *
 * \param xp_n,yp_n,zp_n          Particle position coordinates at start of step
 * \param xp_nph,yp_nph,zp_nph    Particle position coordinates at half time level (n + half)
 * \param Exp,Eyp,Ezp             Electric field on particles.
 * \param Bxp,Byp,Bzp             Magnetic field on particles.
 * \param ex_arr,ey_arr,ez_arr    Array4 of the electric field, either full array or tile.
 * \param bx_arr,by_arr,bz_arr    Array4 of the magnetic field, either full array or tile.
 * \param ex_type,ey_type,ez_type IndexType of the electric field
 * \param bx_type,by_type,bz_type IndexType of the magnetic field
 * \param dinv                    3D cell size inverse
 * \param xyzmin                  The lower bounds of the domain
 * \param lo                      Index lower bounds of domain.
 * \param n_rz_azimuthal_modes    Number of azimuthal modes when using RZ geometry
 * \param nox                     order of the particle shape function
 * \param gather_type             integer identifier for which algorithm to use
 * \param galerkin_interpolation  whether to use lower order in v
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherShapeNImplicit (
                     const amrex::ParticleReal xp_n,
                     const amrex::ParticleReal yp_n,
                     const amrex::ParticleReal zp_n,
                     const amrex::ParticleReal xp_nph,
                     const amrex::ParticleReal yp_nph,
                     const amrex::ParticleReal zp_nph,
                     amrex::ParticleReal& Exp,
                     amrex::ParticleReal& Eyp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& ex_arr,
                     amrex::Array4<amrex::Real const> const& ey_arr,
                     amrex::Array4<amrex::Real const> const& ez_arr,
                     amrex::Array4<amrex::Real const> const& bx_arr,
                     amrex::Array4<amrex::Real const> const& by_arr,
                     amrex::Array4<amrex::Real const> const& bz_arr,
                     const amrex::IndexType ex_type,
                     const amrex::IndexType ey_type,
                     const amrex::IndexType ez_type,
                     const amrex::IndexType bx_type,
                     const amrex::IndexType by_type,
                     const amrex::IndexType bz_type,
                     const amrex::XDim3 & dinv,
                     const amrex::XDim3 & xyzmin,
                     const amrex::Dim3& lo,
                     const int n_rz_azimuthal_modes,
                     const int nox,
                     const int depos_type )
{
    if (depos_type==0) { // CurrentDepositionAlgo::Esirkepov
        if (nox == 1) {
            doGatherShapeNEsirkepovStencilImplicit<1>(xp_n, yp_n, zp_n, xp_nph, yp_nph, zp_nph,
                                                      Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                                      ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                                      ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                                      dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 2) {
            doGatherShapeNEsirkepovStencilImplicit<2>(xp_n, yp_n, zp_n, xp_nph, yp_nph, zp_nph,
                                                      Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                                      ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                                      ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                                      dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 3) {
            doGatherShapeNEsirkepovStencilImplicit<3>(xp_n, yp_n, zp_n, xp_nph, yp_nph, zp_nph,
                                                      Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                                      ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                                      ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                                      dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 4) {
            doGatherShapeNEsirkepovStencilImplicit<4>(xp_n, yp_n, zp_n, xp_nph, yp_nph, zp_nph,
                                                      Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                                      ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                                      ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                                      dinv, xyzmin, lo, n_rz_azimuthal_modes);
        }
    }
    else if (depos_type==3) { // CurrentDepositionAlgo::Villasenor
        if (nox == 1) {
            doGatherPicnicShapeN<1>(xp_n, yp_n, zp_n, xp_nph, yp_nph, zp_nph,
                                    Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                    ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                    ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                    dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 2) {
            doGatherPicnicShapeN<2>(xp_n, yp_n, zp_n, xp_nph, yp_nph, zp_nph,
                                    Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                    ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                    ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                    dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 3) {
            doGatherPicnicShapeN<3>(xp_n, yp_n, zp_n, xp_nph, yp_nph, zp_nph,
                                    Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                    ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                    ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                    dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 4) {
            doGatherPicnicShapeN<4>(xp_n, yp_n, zp_n, xp_nph, yp_nph, zp_nph,
                                    Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                    ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                    ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                    dinv, xyzmin, lo, n_rz_azimuthal_modes);
        }
    }
    else if (depos_type==1) { // CurrentDepositionAlgo::Direct
        if (nox == 1) {
            doGatherShapeN<1,0>(xp_nph, yp_nph, zp_nph, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 2) {
            doGatherShapeN<2,0>(xp_nph, yp_nph, zp_nph, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 3) {
            doGatherShapeN<3,0>(xp_nph, yp_nph, zp_nph, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        } else if (nox == 4) {
            doGatherShapeN<4,0>(xp_nph, yp_nph, zp_nph, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                dinv, xyzmin, lo, n_rz_azimuthal_modes);
        }
    }
}

#endif // WARPX_FIELDGATHER_H_
