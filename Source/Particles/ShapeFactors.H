/* Copyright 2019-2021 Maxence Thevenet, Michael Rowan, Luca Fedeli, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SHAPEFACTORS_H_
#define WARPX_SHAPEFACTORS_H_

#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_GpuQualifiers.H>


/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specializations are defined for orders 0 to 4 (using "if constexpr").
 *  Shape factor functors may be evaluated with double arguments
 *  in current deposition to ensure that current deposited by
 *  particles that move only a small distance is still resolved.
 *  Without this safeguard, single and double precision versions
 *  can give disagreeing results in the time evolution for some
 *  problem setups.
 */
template <int depos_order>
struct Compute_shape_factor
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(
        T* const sx,
        T xmid) const
    {
        if constexpr (depos_order == 0){
            const auto j = static_cast<int>(xmid + T(0.5));
            sx[0] = T(1.0);
            return j;
        }
        else if constexpr (depos_order == 1){
            const auto j = static_cast<int>(xmid);
            const T xint = xmid - T(j);
            sx[0] = T(1.0) - xint;
            sx[1] = xint;
            return j;
        }
        else if constexpr (depos_order == 2){
            const auto j = static_cast<int>(xmid + T(0.5));
            const T xint = xmid - T(j);
            sx[0] = T(0.5)*(T(0.5) - xint)*(T(0.5) - xint);
            sx[1] = T(0.75) - xint*xint;
            sx[2] = T(0.5)*(T(0.5) + xint)*(T(0.5) + xint);
            // index of the leftmost cell where particle deposits
            return j-1;
        }
        else if constexpr (depos_order == 3){
            const auto j = static_cast<int>(xmid);
            const T xint = xmid - T(j);
            sx[0] = (T(1.0))/(T(6.0))*(T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - xint);
            sx[1] = (T(2.0))/(T(3.0)) - xint*xint*(T(1.0) - xint/(T(2.0)));
            sx[2] = (T(2.0))/(T(3.0)) - (T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - T(0.5)*(T(1.0) - xint));
            sx[3] = (T(1.0))/(T(6.0))*xint*xint*xint;
            // index of the leftmost cell where particle deposits
            return j-1;
        }
        else if constexpr (depos_order == 4){
            const auto j = static_cast<int>(xmid + T(0.5));
            const T xint = xmid - T(j);
            sx[0] = (T(1.0))/(T(24.0))*(T(0.5) - xint)*(T(0.5) - xint)*(T(0.5) - xint)*(T(0.5) - xint);
            sx[1] = (T(1.0))/(T(24.0))*(T(4.75) - T(11.0)*xint + T(4.0)*xint*xint*(T(1.5) + xint - xint*xint));
            sx[2] = (T(1.0))/(T(24.0))*(T(14.375) + T(6.0)*xint*xint*(xint*xint - T(2.5)));
            sx[3] = (T(1.0))/(T(24.0))*(T(4.75) + T(11.0)*xint + T(4.0)*xint*xint*(T(1.5) - xint - xint*xint));
            sx[4] = (T(1.0))/(T(24.0))*(T(0.5) + xint)*(T(0.5) + xint)*(T(0.5) + xint)*(T(0.5)+xint);
            // index of the leftmost cell where particle deposits
            return j-2;
        }
        else{
            WARPX_ABORT_WITH_MESSAGE("Unknown particle shape selected in Compute_shape_factor");
            amrex::ignore_unused(sx, xmid);
        }
        return 0;
    }
};



/**
 *  Compute shifted shape factor and return index of leftmost cell where
 *  particle writes, for Esirkepov algorithm.
 *  Specializations are defined below for orders 1, 2, 3, and 4 (using "if constexpr").
 */
template <int depos_order>
struct Compute_shifted_shape_factor
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(
        T* const sx,
        const T x_old,
        const int i_new) const
    {
        if constexpr (depos_order == 0){
            const auto i = static_cast<int>(std::floor(x_old + T(0.5)));
            const int i_shift = i - i_new;
            sx[1+i_shift] = T(1.0);
            return i;
        }
        else if constexpr (depos_order == 1){
            const auto i = static_cast<int>(std::floor(x_old));
            const int i_shift = i - i_new;
            const T xint = x_old - T(i);
            sx[1+i_shift] = T(1.0) - xint;
            sx[2+i_shift] = xint;
            return i;
        }
        else if constexpr (depos_order == 2){
            const auto i = static_cast<int>(x_old + T(0.5));
            const int i_shift = i - (i_new + 1);
            const T xint = x_old - T(i);
            sx[1+i_shift] = T(0.5)*(T(0.5) - xint)*(T(0.5) - xint);
            sx[2+i_shift] = T(0.75) - xint*xint;
            sx[3+i_shift] = T(0.5)*(T(0.5) + xint)*(T(0.5) + xint);
            // index of the leftmost cell where particle deposits
            return i - 1;
        }
        else if constexpr (depos_order == 3){
            const auto i = static_cast<int>(x_old);
            const int i_shift = i - (i_new + 1);
            const T xint = x_old - T(i);
            sx[1+i_shift] = (T(1.0))/(T(6.0))*(T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - xint);
            sx[2+i_shift] = (T(2.0))/(T(3.0)) - xint*xint*(T(1.0) - xint/(T(2.0)));
            sx[3+i_shift] = (T(2.0))/(T(3.0)) - (T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - T(0.5)*(T(1.0) - xint));
            sx[4+i_shift] = (T(1.0))/(T(6.0))*xint*xint*xint;
            // index of the leftmost cell where particle deposits
            return i - 1;
        }
        else if constexpr (depos_order == 4){
            const auto i = static_cast<int>(x_old + T(0.5));
            const int i_shift = i - (i_new + 2);
            const T xint = x_old - T(i);
            sx[1+i_shift] = (T(1.0))/(T(24.0))*(T(0.5) - xint)*(T(0.5) - xint)*(T(0.5) - xint)*(T(0.5) - xint);
            sx[2+i_shift] = (T(1.0))/(T(24.0))*(T(4.75) - T(11.0)*xint + T(4.0)*xint*xint*(T(1.5) + xint - xint*xint));
            sx[3+i_shift] = (T(1.0))/(T(24.0))*(T(14.375) + T(6.0)*xint*xint*(xint*xint - T(2.5)));
            sx[4+i_shift] = (T(1.0))/(T(24.0))*(T(4.75) + T(11.0)*xint + T(4.0)*xint*xint*(T(1.5) - xint - xint*xint));
            sx[5+i_shift] = (T(1.0))/(T(24.0))*(T(0.5) + xint)*(T(0.5) + xint)*(T(0.5) + xint)*(T(0.5)+xint);
            // index of the leftmost cell where particle deposits
            return i - 2;
        }
        else{
            WARPX_ABORT_WITH_MESSAGE("Unknown particle shape selected in Compute_shifted_shape_factor");
            amrex::ignore_unused(sx, x_old, i_new);
        }
        return 0;
    }
};

/**
 *  Compute shape factors for two positions that are within
 *  half a grid cell of the same cell interface and return the common
 *  index of the leftmost cell where particle writes, which is correctly
 *  determined by the average of the positions.
 *  This is used for computing the segment weights transverse to the
 *  current density direction in the Villasenor deposition algorithm.
 */
template <int depos_order>
struct Compute_shape_factor_pair
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(
        T* const sx_old,
        T* const sx_new,
        T xold,
        T xnew) const
    {
        const T xmid = T(0.5)*(xnew + xold);
        if constexpr (depos_order == 1){
            const auto j = static_cast<int>(xmid);
            const T xint_old = xold - T(j);
            sx_old[0] = T(1.0) - xint_old;
            sx_old[1] = xint_old;
            //
            const T xint_new = xnew - T(j);
            sx_new[0] = T(1.0) - xint_new;
            sx_new[1] = xint_new;
            return j;
        }
        else if constexpr (depos_order == 2){
            const auto j = static_cast<int>(xmid + T(0.5));
            const T xint_old = xold - T(j);
            sx_old[0] = T(0.5)*(T(0.5) - xint_old)*(T(0.5) - xint_old);
            sx_old[1] = T(0.75) - xint_old*xint_old;
            sx_old[2] = T(0.5)*(T(0.5) + xint_old)*(T(0.5) + xint_old);
            //
            const T xint_new = xnew - T(j);
            sx_new[0] = T(0.5)*(T(0.5) - xint_new)*(T(0.5) - xint_new);
            sx_new[1] = T(0.75) - xint_new*xint_new;
            sx_new[2] = T(0.5)*(T(0.5) + xint_new)*(T(0.5) + xint_new);
            // index of the leftmost cell where particle deposits
            return j-1;
        }
        else if constexpr (depos_order == 3){
            const auto j = static_cast<int>(xmid);
            const T xint_old = xold - T(j);
            sx_old[0] = T(1.0)/T(6.0)*(T(1.0) - xint_old)*(T(1.0) - xint_old)*(T(1.0) - xint_old);
            sx_old[1] = T(2.0)/T(3.0) - xint_old*xint_old*(T(1.0) - xint_old/(T(2.0)));
            sx_old[2] = T(2.0)/T(3.0) - (T(1.0) - xint_old)*(T(1.0) - xint_old)*(T(1.0) - T(0.5)*(T(1.0) - xint_old));
            sx_old[3] = T(1.0)/T(6.0)*xint_old*xint_old*xint_old;
            //
            const T xint_new = xnew - T(j);
            sx_new[0] = T(1.0)/T(6.0)*(T(1.0) - xint_new)*(T(1.0) - xint_new)*(T(1.0) - xint_new);
            sx_new[1] = T(2.0)/T(3.0) - xint_new*xint_new*(T(1.0) - xint_new/(T(2.0)));
            sx_new[2] = T(2.0)/T(3.0) - (T(1.0) - xint_new)*(T(1.0) - xint_new)*(T(1.0) - T(0.5)*(T(1.0) - xint_new));
            sx_new[3] = T(1.0)/T(6.0)*xint_new*xint_new*xint_new;
            // index of the leftmost cell where particle deposits
            return j-1;
        }
        else if constexpr (depos_order == 4){
            const auto j = static_cast<int>(xmid + T(0.5));
            const T xint_old = xold - T(j);
            sx_old[0] = (T(1.0))/(T(24.0))*(T(0.5) - xint_old)*(T(0.5) - xint_old)*(T(0.5) - xint_old)*(T(0.5) - xint_old);
            sx_old[1] = (T(1.0))/(T(24.0))*(T(4.75) - T(11.0)*xint_old + T(4.0)*xint_old*xint_old*(T(1.5) + xint_old - xint_old*xint_old));
            sx_old[2] = (T(1.0))/(T(24.0))*(T(14.375) + T(6.0)*xint_old*xint_old*(xint_old*xint_old - T(2.5)));
            sx_old[3] = (T(1.0))/(T(24.0))*(T(4.75) + T(11.0)*xint_old + T(4.0)*xint_old*xint_old*(T(1.5) - xint_old - xint_old*xint_old));
            sx_old[4] = (T(1.0))/(T(24.0))*(T(0.5) + xint_old)*(T(0.5) + xint_old)*(T(0.5) + xint_old)*(T(0.5)+xint_old);
            //
            const T xint_new = xnew - T(j);
            sx_new[0] = (T(1.0))/(T(24.0))*(T(0.5) - xint_new)*(T(0.5) - xint_new)*(T(0.5) - xint_new)*(T(0.5) - xint_new);
            sx_new[1] = (T(1.0))/(T(24.0))*(T(4.75) - T(11.0)*xint_new + T(4.0)*xint_new*xint_new*(T(1.5) + xint_new - xint_new*xint_new));
            sx_new[2] = (T(1.0))/(T(24.0))*(T(14.375) + T(6.0)*xint_new*xint_new*(xint_new*xint_new - T(2.5)));
            sx_new[3] = (T(1.0))/(T(24.0))*(T(4.75) + T(11.0)*xint_new + T(4.0)*xint_new*xint_new*(T(1.5) - xint_new - xint_new*xint_new));
            sx_new[4] = (T(1.0))/(T(24.0))*(T(0.5) + xint_new)*(T(0.5) + xint_new)*(T(0.5) + xint_new)*(T(0.5)+xint_new);
            //
            // index of the leftmost cell where particle deposits
            return j-2;
        }
        else{
            WARPX_ABORT_WITH_MESSAGE("Unknown particle shape selected in Compute_shape_factor_pair");
            amrex::ignore_unused(sx_old, sx_new, xold, xnew);
        }
        return 0;
    }
};

#endif // WARPX_SHAPEFACTORS_H_
