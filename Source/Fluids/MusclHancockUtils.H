/* Copyright 2023 Grant Johnson, Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_MusclHancock_H_
#define WARPX_MusclHancock_H_

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>


// Euler push for momentum source (r-direction)
// Note: assumes U normalized by c
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real F_r (amrex::Real r, amrex::Real u_r, amrex::Real u_theta, amrex::Real u_z, amrex::Real dt)
{
    using namespace amrex::literals;
    return dt*(-u_theta*u_theta/r)/std::sqrt(1.0_rt + u_r*u_r + u_theta*u_theta + u_z*u_z) + u_r;
}

// Euler push for momentum source (theta-direction)
// Note: assumes U normalized by c
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real F_theta (amrex::Real r, amrex::Real u_r, amrex::Real u_theta, amrex::Real u_z, amrex::Real dt)
{
    using namespace amrex::literals;
    return dt*(u_theta*u_r/r)/std::sqrt(1.0_rt + u_r*u_r + u_theta*u_theta + u_z*u_z) + u_theta;
}
// Velocity at the half step
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real V_calc (const amrex::Array4<amrex::Real>& U, int i, int j, int k, int comp, amrex::Real c)
{
    using namespace amrex::literals;
    // comp -> x, y, z -> 0, 1, 2, return Vx, Vy, or Vz:
    const amrex::Real gamma = std::sqrt(1.0_rt + (U(i,j,k,1)*U(i,j,k,1) + U(i,j,k,2)*U(i,j,k,2) + U(i,j,k,3)*U(i,j,k,3))/(c*c));
    return U(i,j,k,comp+1)/gamma;
}
// mindmod
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real minmod (amrex::Real a, amrex::Real b)
{
    using namespace amrex::literals;
    if (a > 0.0_rt && b > 0.0_rt) {
        return std::min(a, b);
    } else if (a < 0.0_rt && b < 0.0_rt) {
        return std::max(a, b);
    } else {
        return 0.0_rt;
    }
}
// Min of 3 inputs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real min3 (amrex::Real a, amrex::Real b, amrex::Real c)
{
    return std::min(a, std::min(b, c) );
}
// Max of 3 inputs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real max3 (amrex::Real a, amrex::Real b, amrex::Real c)
{
    return std::max(a, std::max(b, c) );
}
// mindmod
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real minmod3 (amrex::Real a, amrex::Real b , amrex::Real c)
{
    using namespace amrex::literals;
    if (a > 0.0_rt && b > 0.0_rt && c > 0.0_rt) {
        return min3(a,b,c);
    } else if (a < 0.0_rt && b < 0.0_rt && c < 0.0_rt) {
        return max3(a,b,c);
    } else {
        return 0.0;
    }
}
//maxmod
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real maxmod (amrex::Real a, amrex::Real b)
{
    using namespace amrex::literals;
    if (a > 0.0_rt && b > 0.0_rt) {
        return std::max(a, b);
    } else if (a < 0.0_rt && b < 0.0_rt) {
        return std::min(a, b);
    } else {
        return 0.0_rt;
    }
}
// Rusanov Flux (density)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real flux_N (const amrex::Array4<amrex::Real>& Um,  const amrex::Array4<amrex::Real>& Up,
int i, int j, int k, amrex::Real Vm, amrex::Real Vp)
{
    using namespace amrex::literals;
    const amrex::Real c = std::max( std::abs(Vm) , std::abs(Vp) );
    return 0.5_rt*(Vm*Um(i,j,k,0) + Vp*Up(i,j,k,0)) - (0.5_rt*c)*(Up(i,j,k,0) - Um(i,j,k,0));
}
// Rusanov Flux (Momentum density x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real flux_NUx (const amrex::Array4<amrex::Real>& Um,  const amrex::Array4<amrex::Real>& Up,
int i, int j, int k, amrex::Real Vm, amrex::Real Vp)
{
    using namespace amrex::literals;
    const amrex::Real c = std::max( std::abs(Vm) , std::abs(Vp) );
    return 0.5_rt*(Vm*Um(i,j,k,0)*Um(i,j,k,1) + Vp*Up(i,j,k,0)*Up(i,j,k,1))
        - (0.5_rt*c)*(Up(i,j,k,0)*Up(i,j,k,1) -    Um(i,j,k,0)*Um(i,j,k,1));
}
// Rusanov Flux (Momentum density y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real flux_NUy (const amrex::Array4<amrex::Real>& Um,  const amrex::Array4<amrex::Real>& Up,
int i, int j, int k, amrex::Real Vm, amrex::Real Vp)
{
    using namespace amrex::literals;
    const amrex::Real c = std::max( std::abs(Vm) , std::abs(Vp) );
    return 0.5_rt*(Vm*Um(i,j,k,0)*Um(i,j,k,2) + Vp*Up(i,j,k,0)*Up(i,j,k,2))
        - (0.5_rt*c)*(Up(i,j,k,0)*Up(i,j,k,2) -    Um(i,j,k,0)*Um(i,j,k,2));
}
// Rusanov Flux (Momentum density z)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real flux_NUz (const amrex::Array4<amrex::Real>& Um,  const amrex::Array4<amrex::Real>& Up,
int i, int j, int k, amrex::Real Vm, amrex::Real Vp)
{
    using namespace amrex::literals;
    const amrex::Real c = std::max( std::abs(Vm) , std::abs(Vp) );
    return 0.5_rt*(Vm*Um(i,j,k,0)*Um(i,j,k,3) + Vp*Up(i,j,k,0)*Up(i,j,k,3))
        - (0.5_rt*c)*(Up(i,j,k,0)*Up(i,j,k,3) -    Um(i,j,k,0)*Um(i,j,k,3));
}
// ave_minmod high diffusivity, sigma can be between [1,2]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ave_adjustable_diff (amrex::Real a, amrex::Real b)
{
    using namespace amrex::literals;
    constexpr auto sigma = static_cast<amrex::Real>(2.0*0.732050807568877);
    if (a*b > 0.0_rt) {
        return minmod3( (a+b)/2.0_rt, sigma*a, sigma*b );
    } else {
        return 0.0_rt;
    }
}
// ave_minmod Low diffusivity
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ave (amrex::Real a, amrex::Real b)
{
    using namespace amrex::literals;
    if (a*b > 0.0_rt) {
        return minmod3( (a+b)/2.0_rt, 2.0_rt*a, 2.0_rt*b );
    } else {
        return 0.0_rt;
    }
}
// ave_superbee
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ave_superbee (amrex::Real a, amrex::Real b)
{
    using namespace amrex::literals;
    if (a*b > 0.0_rt) {
        return minmod( maxmod(a,b), minmod(2.0_rt*a,2.0_rt*b));
    } else {
        return 0.0_rt;
    }
}
// stage2 slope limiting
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ave_stage2 (amrex::Real dQ, amrex::Real a, amrex::Real b, amrex::Real c)
{
    using namespace amrex::literals;
    // sigma = sqrt(3) -1
    constexpr auto sigma = 0.732050807568877_rt;
    const amrex::Real dq_min = 2.0_rt*std::min( b - min3(a,b,c), max3(a,b,c) - b);
    return ( std::abs(dQ)/dQ ) * std::min( std::abs(dQ) , sigma*std::abs(dq_min) );
}
// Returns the offset indices for the "plus" grid
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void plus_index_offsets (int i, int j, int k, int& ip, int& jp, int& kp, int comp)
{
    using namespace amrex::literals;
    // Find the correct offsets
#if defined(WARPX_DIM_3D)
    if (comp == 0) { //x
        ip =  i - 1; jp =  j; kp = k;
    } else if (comp == 1){ //y
        ip =  i; jp = j - 1; kp = k;
    } else if (comp == 2){ //z
        ip =  i; jp =  j; kp = k - 1;
    }
 #elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    if (comp == 0) { //x
        ip =  i - 1; jp =  j; kp = k;
    } else if (comp == 2){ //z
        ip =  i; jp =  j - 1; kp = k;
    }
#else
    if (comp == 2) { //z
        ip =  i - 1; jp =  j; kp = k;
    }
#endif
}
// Compute the zero edges
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void compute_U_edges (const amrex::Array4<amrex::Real>& Um, const amrex::Array4<amrex::Real>& Up, int i, int j, int k, amrex::Box box,
amrex::Real U_tilde0, amrex::Real U_tilde1, amrex::Real U_tilde2, amrex::Real U_tilde3,
amrex::Real dU0x, amrex::Real dU1x, amrex::Real dU2x, amrex::Real dU3x, int comp)
{
    using namespace amrex::literals;
    // comp -> x, y, z -> 0, 1, 2
    int ip, jp, kp;
    plus_index_offsets(i, j, k, ip, jp, kp, comp);

    if ( box.contains(i,j,k) ) {
        Um(i,j,k,0) = U_tilde0 + dU0x/2.0_rt;
        Um(i,j,k,1) = U_tilde1 + dU1x/2.0_rt;
        Um(i,j,k,2) = U_tilde2 + dU2x/2.0_rt;
        Um(i,j,k,3) = U_tilde3 + dU3x/2.0_rt;
    }

    if ( box.contains(ip,jp,kp) ) {
        Up(ip,jp,kp,0) = U_tilde0 - dU0x/2.0_rt;
        Up(ip,jp,kp,1) = U_tilde1 - dU1x/2.0_rt;
        Up(ip,jp,kp,2) = U_tilde2 - dU2x/2.0_rt;
        Up(ip,jp,kp,3) = U_tilde3 - dU3x/2.0_rt;
    }
}
// Compute the zero edges
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void set_U_edges_to_zero (const amrex::Array4<amrex::Real>& Um,
const amrex::Array4<amrex::Real>& Up, int i, int j, int k, amrex::Box box, int comp)
{
    using namespace amrex::literals;
    // comp -> x, y, z -> 0, 1, 2
    int ip, jp, kp;
    plus_index_offsets(i, j, k, ip, jp, kp, comp);

    if ( box.contains(i,j,k) ) {
        Um(i,j,k,0) = 0.0_rt;
        Um(i,j,k,1) = 0.0_rt;
        Um(i,j,k,2) = 0.0_rt;
        Um(i,j,k,3) = 0.0_rt;
    }

    if ( box.contains(ip,jp,kp) ) {
        Up(ip,jp,kp,0) = 0.0_rt;
        Up(ip,jp,kp,1) = 0.0_rt;
        Up(ip,jp,kp,2) = 0.0_rt;
        Up(ip,jp,kp,3) = 0.0_rt;
    }
}
// Positivity Limiter
// if Q_minus or Q_plus is zero for the density (i.e. component 0 of Q_minus/Q_plus), set dQ to 0 and recompute Q_minus / Q_plus
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void positivity_limiter (const amrex::Array4<amrex::Real>& U_edge_plus,
const amrex::Array4<amrex::Real>& U_edge_minus, const amrex::Array4<amrex::Real>& N_arr,
int i, int j, int k, amrex::Box box, amrex::Real Ux, amrex::Real Uy, amrex::Real Uz,
int comp)
{

    using namespace amrex::literals;
    // comp -> x, y, z -> 0, 1, 2
    int ip, jp, kp;
    plus_index_offsets(i, j, k, ip, jp, kp, comp);

    // Set the edges to zero. If one edge in a cell is zero, we must self-consistently
    // set the slope to zero (hence why we have the three cases, the first is when
    // both points exist, and the second two are are edge cases)
    if (( box.contains(i,j,k) ) && ( box.contains(ip,jp,kp) )) {
        if ((U_edge_minus(i,j,k,0) < 0.0_rt) || (U_edge_plus(ip,jp,kp,0) < 0.0_rt)) {
            U_edge_minus(i,j,k,0) = N_arr(i,j,k);
            U_edge_minus(i,j,k,1) = Ux;
            U_edge_minus(i,j,k,2) = Uy;
            U_edge_minus(i,j,k,3) = Uz;
            U_edge_plus(ip,jp,kp,0) = N_arr(i,j,k);
            U_edge_plus(ip,jp,kp,1) = Ux;
            U_edge_plus(ip,jp,kp,2) = Uy;
            U_edge_plus(ip,jp,kp,3) = Uz;
        }
    } else if (( box.contains(i,j,k) ) && ( box.contains(ip,jp,kp) != 1)) {
        if (U_edge_minus(i,j,k,0) < 0.0_rt) {
            U_edge_minus(i,j,k,0) = N_arr(i,j,k);
            U_edge_minus(i,j,k,1) = Ux;
            U_edge_minus(i,j,k,2) = Uy;
            U_edge_minus(i,j,k,3) = Uz;
        }
    } else if (( box.contains(i,j,k) != 1 ) && ( box.contains(ip,jp,kp) )) {
        if (U_edge_plus(ip,jp,kp,0) < 0.0_rt){
            U_edge_plus(ip,jp,kp,0) = N_arr(i,j,k);
            U_edge_plus(ip,jp,kp,1) = Ux;
            U_edge_plus(ip,jp,kp,2) = Uy;
            U_edge_plus(ip,jp,kp,3) = Uz;
        }
    }
}

// Compute the difference in N (down-x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real DownDx_N (const amrex::Array4<amrex::Real>& N, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    return N(i,j,k) -  N(i-1,j,k);
#else
    amrex::ignore_unused(N, i, j, k);
    return 0.0_rt;
#endif
}
// Compute the difference in N (up-x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real UpDx_N (const amrex::Array4<amrex::Real>& N, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    return N(i+1,j,k) -  N(i,j,k);
#else
    amrex::ignore_unused(N, i, j, k);
    return 0.0_rt;
#endif
}
// Compute the difference in N (down-y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real DownDy_N (const amrex::Array4<amrex::Real>& N, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D)
    return N(i,j,k) -  N(i,j-1,k);
#else
    amrex::ignore_unused(N, i, j, k);
    return 0.0_rt;
#endif
}
// Compute the difference in N (up-y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real UpDy_N (const amrex::Array4<amrex::Real>& N, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D)
    return N(i,j+1,k) -  N(i,j,k);
#else
    amrex::ignore_unused(N, i, j, k);
    return 0.0_rt;
#endif
}
// Compute the difference in N (down-z)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real DownDz_N (const amrex::Array4<amrex::Real>& N, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D)
    return N(i,j,k) -  N(i,j,k-1);
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    return N(i,j,k) -  N(i,j-1,k);
#else
    return N(i,j,k) -  N(i-1,j,k);
#endif
}
// Compute the difference in N (up-z)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real UpDz_N (const amrex::Array4<amrex::Real>& N, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D)
    return N(i,j,k+1) -  N(i,j,k);
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    return N(i,j+1,k) -  N(i,j,k);
#else
    return N(i+1,j,k) -  N(i,j,k);
#endif
}


// Compute the difference in U (down-x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real DownDx_U (const amrex::Array4<amrex::Real>& N,
const amrex::Array4<amrex::Real>& NU, amrex::Real& U, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    // U is zero if N is zero, Check positivity before dividing
    amrex::Real U_m = 0;
    if (N(i-1,j,k) > 0) { U_m = NU(i-1,j,k)/N(i-1,j,k); }
    return U - U_m;
#else
    amrex::ignore_unused(N, NU, U, i, j, k);
    return 0.0_rt;
#endif
}
// Compute the difference in U (up-x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real UpDx_U (const amrex::Array4<amrex::Real>& N,
const amrex::Array4<amrex::Real>& NU, amrex::Real& U, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    // U is zero if N is zero, Check positivity before dividing
    amrex::Real U_p = 0;
    if (N(i+1,j,k) > 0) { U_p = NU(i+1,j,k)/N(i+1,j,k); }
    return U_p - U;
#else
    amrex::ignore_unused(N, NU, U, i, j, k);
    return 0.0_rt;
#endif
}

// Compute the difference in U (down-y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real DownDy_U (const amrex::Array4<amrex::Real>& N,
const amrex::Array4<amrex::Real>& NU, amrex::Real& U, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D)
    // U is zero if N is zero, Check positivity before dividing
    amrex::Real U_m = 0;
    if (N(i,j-1,k) > 0) { U_m = NU(i,j-1,k)/N(i,j-1,k); }
    return U - U_m;
#else
    amrex::ignore_unused(N, NU, U, i, j, k);
    return 0.0_rt;
#endif
}
// Compute the difference in U (up-y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real UpDy_U (const amrex::Array4<amrex::Real>& N,
const amrex::Array4<amrex::Real>& NU, amrex::Real& U, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
#if defined(WARPX_DIM_3D)
    // U is zero if N is zero, Check positivity before dividing
    amrex::Real U_p = 0;
    if (N(i,j+1,k) > 0) { U_p = NU(i,j+1,k)/N(i,j+1,k); }
    return U_p - U;
#else
    amrex::ignore_unused(N, NU, U, i, j, k);
    return 0.0_rt;
#endif
}

// Compute the difference in U (down-z)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real DownDz_U (const amrex::Array4<amrex::Real>& N,
const amrex::Array4<amrex::Real>& NU, amrex::Real& U, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
    amrex::Real U_m = 0_rt;

    // U is zero if N is zero, Check positivity before dividing
#if defined(WARPX_DIM_3D)
    if (N(i,j,k-1) > 0) { U_m = NU(i,j,k-1)/N(i,j,k-1); }
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    if (N(i,j-1,k) > 0) { U_m = NU(i,j-1,k)/N(i,j-1,k); }
#else
    if (N(i-1,j,k) > 0) { U_m = NU(i-1,j,k)/N(i-1,j,k); }
#endif

    // Return the difference
    return U - U_m;
}
// Compute the difference in U (up-z)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real UpDz_U (const amrex::Array4<amrex::Real>& N,
const amrex::Array4<amrex::Real>& NU, amrex::Real& U, int i, int j, int k)
{
    using namespace amrex::literals;
    // Write the correct differences
    amrex::Real U_p = 0;

    // U is zero if N is zero, Check positivity before dividing
#if defined(WARPX_DIM_3D)
    if (N(i,j,k+1) > 0) { U_p = NU(i,j,k+1)/N(i,j,k+1); }
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    if (N(i,j+1,k) > 0) { U_p = NU(i,j+1,k)/N(i,j+1,k); }
#else
    if (N(i+1,j,k) > 0) { U_p = NU(i+1,j,k)/N(i+1,j,k); }
#endif

    // Return the difference
    return U_p - U;
}


// Flux difference calculation
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real dF (const amrex::Array4<amrex::Real>& U_minus,
const amrex::Array4<amrex::Real>& U_plus,int i,int j,int k,amrex::Real clight, int comp, int dir)
{
    using namespace amrex::literals;
    // dir -> x, y, z -> 0, 1, 2
    int ip, jp, kp;
    plus_index_offsets(i, j, k, ip, jp, kp, dir);

    const amrex::Real V_L_minus = V_calc(U_minus,ip,jp,kp,dir,clight);
    const amrex::Real V_I_minus = V_calc(U_minus,i,j,k,dir,clight);
    const amrex::Real V_L_plus = V_calc(U_plus,ip,jp,kp,dir,clight);
    const amrex::Real V_I_plus = V_calc(U_plus,i,j,k,dir,clight);

    // Flux differences depending on the component to compute
    if (comp == 0){
        return flux_N(  U_minus, U_plus, i, j, k, V_I_minus, V_I_plus) - flux_N(  U_minus, U_plus, ip, jp, kp, V_L_minus, V_L_plus);
    } else if (comp == 1){
        return flux_NUx(  U_minus, U_plus, i, j, k, V_I_minus, V_I_plus) - flux_NUx(  U_minus, U_plus, ip, jp, kp, V_L_minus, V_L_plus);
    } else if (comp == 2){
        return flux_NUy(  U_minus, U_plus, i, j, k, V_I_minus, V_I_plus) - flux_NUy(  U_minus, U_plus, ip, jp, kp, V_L_minus, V_L_plus);
    } else { //if (comp == 3)
        return flux_NUz(  U_minus, U_plus, i, j, k, V_I_minus, V_I_plus) - flux_NUz(  U_minus, U_plus, ip, jp, kp, V_L_minus, V_L_plus);
    }
}

#endif /*WARPX_MusclHancock_H_*/
