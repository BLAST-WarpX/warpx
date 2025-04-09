/* Copyright 2019-2022 Modern Electron, Axel Huebl, Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_NODALFIELDGATHER_H_
#define ABLASTR_NODALFIELDGATHER_H_

#include "ablastr/utils/TextMsg.H"

#include <AMReX_Array.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>


namespace ablastr::particles
{

/**
 * \brief Compute weight of each surrounding node (or cell-centered nodes) in interpolating a nodal ((or a cell-centered node) field
 *        to the given coordinates. If template parameter IdxType is amrex::IndexType::CellIndex::NODE, then the calculations will be done with respect to the nodes.
 *        If template parameter IdxType is amrex::IndexType::CellIndex::CELL, then the calculations will be done with respect to the cell-centered nodal.
 *        This currently only does linear order.
 *
 * \param xp,yp,zp Particle position coordinates
 * \param plo      Index lower bounds of domain.
 * \param dxi      inverse 3D cell spacing
 * \param i,j,k    Variables to store indices of position on grid (nodal or cell-centered, depending of the value of `nodal`)
 * \param W        2D array of weights to store each neighbouring node (or cell-centered node)
 * \param nodal    Int that tells if the weights are calculated in respect to the nodes (nodal=1) of the cell-centered nodes (nodal=0)
 */
template <amrex::IndexType::CellIndex IdxType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void compute_weights (const amrex::ParticleReal xp,
                            const amrex::ParticleReal yp,
                            const amrex::ParticleReal zp,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
                            int& i, int& j, int& k, amrex::Real W[AMREX_SPACEDIM][2]) noexcept
{
    using namespace amrex::literals;

    constexpr auto half_if_cell_centered = [](){
        if constexpr (IdxType == amrex::IndexType::CellIndex::NODE){
            return 0.0_rt;
        }
        else{
            return 0.5_rt;
        }
    }();

#if (defined WARPX_DIM_3D)
    const amrex::Real x = (xp - plo[0]) * dxi[0] - half_if_cell_centered;
    const amrex::Real y = (yp - plo[1]) * dxi[1] - half_if_cell_centered;
    const amrex::Real z = (zp - plo[2]) * dxi[2] - half_if_cell_centered;

    i = static_cast<int>(amrex::Math::floor(x));
    j = static_cast<int>(amrex::Math::floor(y));
    k = static_cast<int>(amrex::Math::floor(z));

    W[0][1] = x - i;
    W[1][1] = y - j;
    W[2][1] = z - k;

    W[0][0] = 1.0_rt - W[0][1];
    W[1][0] = 1.0_rt - W[1][1];
    W[2][0] = 1.0_rt - W[2][1];

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

#   if (defined WARPX_DIM_XZ)
    const amrex::Real x = (xp - plo[0]) * dxi[0] - half_if_cell_centered;
    amrex::ignore_unused(yp);
    i = static_cast<int>(amrex::Math::floor(x));
    W[0][1] = x - i;
#   elif (defined WARPX_DIM_RZ)
    const amrex::Real r = (std::sqrt(xp*xp+yp*yp) - plo[0]) * dxi[0] - half_if_cell_centered;
    i = static_cast<int>(amrex::Math::floor(r));
    W[0][1] = r - i;
#   endif

    const amrex::Real z = (zp - plo[1]) * dxi[1] - half_if_cell_centered;
    j = static_cast<int>(amrex::Math::floor(z));
    W[1][1] = z - j;

    W[0][0] = 1.0_rt - W[0][1];
    W[1][0] = 1.0_rt - W[1][1];

    k = 0;
#else
    const amrex::Real z = (zp - plo[0]) * dxi[0] - half_if_cell_centered;
    amrex::ignore_unused(xp, yp);
    i = static_cast<int>(amrex::Math::floor(z));

    W[0][1] = z - i;
    W[0][0] = 1.0_rt - W[0][1];

    j = 0;
    k = 0;
#endif
}

/**
 * \brief Interpolate nodal field value based on surrounding indices and weights.
 *
 * \param i,j,k                Indices of position on grid
 * \param W                    2D array of weights for each neighbouring node
 * \param scalar_field         Array4 of the nodal scalar field, either full array or tile.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real interp_field_nodal (int i, int j, int k,
                                const amrex::Real W[AMREX_SPACEDIM][2],
                                amrex::Array4<const amrex::Real> const& scalar_field) noexcept
{
    amrex::Real value = 0;
#if (defined WARPX_DIM_3D)
    value += scalar_field(i,   j  , k  ) * W[0][0] * W[1][0] * W[2][0];
    value += scalar_field(i+1, j  , k  ) * W[0][1] * W[1][0] * W[2][0];
    value += scalar_field(i,   j+1, k  ) * W[0][0] * W[1][1] * W[2][0];
    value += scalar_field(i+1, j+1, k  ) * W[0][1] * W[1][1] * W[2][0];
    value += scalar_field(i,   j  , k+1) * W[0][0] * W[1][0] * W[2][1];
    value += scalar_field(i+1, j  , k+1) * W[0][1] * W[1][0] * W[2][1];
    value += scalar_field(i  , j+1, k+1) * W[0][0] * W[1][1] * W[2][1];
    value += scalar_field(i+1, j+1, k+1) * W[0][1] * W[1][1] * W[2][1];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    value += scalar_field(i,   j ,  k) * W[0][0] * W[1][0];
    value += scalar_field(i+1, j ,  k) * W[0][1] * W[1][0];
    value += scalar_field(i,   j+1, k) * W[0][0] * W[1][1];
    value += scalar_field(i+1, j+1, k) * W[0][1] * W[1][1];
#else
    value += scalar_field(i,   j ,  k) * W[0][0];
    value += scalar_field(i+1, j ,  k) * W[0][1];
#endif
    return value;
}

/**
 * \brief Scalar field gather for a single particle. The field has to be defined
 *        at the cell nodes (see https://amrex-codes.github.io/amrex/docs_html/Basics.html#id2)
 *
 * \param xp,yp,zp                Particle position coordinates
 * \param scalar_field            Array4 of the nodal scalar field, either full array or tile.
 * \param dxi                     inverse 3D cell spacing
 * \param lo                      Index lower bounds of domain.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real doGatherScalarFieldNodal (const amrex::ParticleReal xp,
                                      const amrex::ParticleReal yp,
                                      const amrex::ParticleReal zp,
                                      amrex::Array4<const amrex::Real> const& scalar_field,
                                      amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                                      amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& lo) noexcept
{
    // first find the weight of surrounding nodes to use during interpolation
    int ii, jj, kk;
    amrex::Real W[AMREX_SPACEDIM][2];
    compute_weights<amrex::IndexType::NODE>(xp, yp, zp, lo, dxi, ii, jj, kk, W);

    return interp_field_nodal(ii, jj, kk, W, scalar_field);
}

/**
 * \brief Vector field gather for a single particle. The field has to be defined
 *        at the cell nodes (see https://amrex-codes.github.io/amrex/docs_html/Basics.html#id2)
 *
 * \param xp,yp,zp Particle position coordinates
 * \param vector_field_x,vector_field_y,vector_field_z Array4 of nodal scalar fields, either full array or tile.
 * \param dxi inverse 3D cell spacing
 * \param lo  Index lower bounds of domain.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::GpuArray<amrex::Real, 3>
doGatherVectorFieldNodal (const amrex::ParticleReal xp,
                          const amrex::ParticleReal yp,
                          const amrex::ParticleReal zp,
                          amrex::Array4<const amrex::Real> const& vector_field_x,
                          amrex::Array4<const amrex::Real> const& vector_field_y,
                          amrex::Array4<const amrex::Real> const& vector_field_z,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& lo) noexcept
{
    // first find the weight of surrounding nodes to use during interpolation
    int ii, jj, kk;
    amrex::Real W[AMREX_SPACEDIM][2];
    compute_weights<amrex::IndexType::NODE>(xp, yp, zp, lo, dxi, ii, jj, kk, W);

    amrex::GpuArray<amrex::Real, 3> const field_interp = {
        interp_field_nodal(ii, jj, kk, W, vector_field_x),
        interp_field_nodal(ii, jj, kk, W, vector_field_y),
        interp_field_nodal(ii, jj, kk, W, vector_field_z)
    };

    return field_interp;
}

} // namespace ablastr::particles

#endif // ABLASTR_NODALFIELDGATHER_H_
