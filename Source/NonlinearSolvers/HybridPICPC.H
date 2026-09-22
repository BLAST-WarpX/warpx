/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_PIC_PC_H_
#define WARPX_HYBRID_PIC_PC_H_

#include "Preconditioner.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/utils/Enums.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <iomanip>

/**
 * \brief Physics-based EMHD preconditioner for the theta-implicit hybrid-PIC
 *  solver, after L. Chacon, JCP 526 (2025) 113789.
 *
 *  With particles, density and electron pressure frozen, the only dependence
 *  of E_Ohm on the Newton iterate E is through Faraday's law,
 *  dB = -theta*dt curl(dE), giving the stiff electron (EMHD) block
 *
 *    J_pc dE = dE + U (curl curl dE),
 *    U v     = c_w (v x bhat) + a_e v - a_H del2 v,
 *    c_w     = theta*dt |B0| / (mu0 rho_c),  a_e = theta*dt eta / mu0,
 *    a_H     = theta*dt eta_H / mu0,
 *  with eta frozen at the linearization state: eta(rho, |J0|),
 *  J0 = curl B0 / mu0 (freezing at |J| = 0 would underestimate anomalous,
 *  J-dependent resistivity precisely in current sheets), plus the frozen
 *  electron-convection term -theta*dt ue0 x curl(dE) of the E-row Jacobian,
 *  ue0 = (J_i - J_plasma)0 / rho_c.
 *
 *  Primal pair. The auxiliary W = curl curl dE is kept as a second unknown
 *  (the parabolization of Chacon 2025 Sec. 3 applied globally) and the pair
 *
 *    E-row: xE + (I+S)^-1 [ U xW - theta*dt ue0 x (curl xE) ] = b~
 *    W-row: xW - curlcurl xE = 0
 *
 *  is smoothed with a per-node joint 2x2 block damped-Jacobi sweep (Chacon
 *  2025 Sec. 4.1): the defect correction builds the block with curlcurl ->
 *  l2 while all residuals use the exact discrete operators, so the fixed
 *  point is exact,
 *
 *    P dxE = rE - U_loc rW / s_W,   dxW = rW + s_W l2 dxE,
 *    P     = alpha I + beta (. x bhat),
 *    alpha = 1 + theta*dt |ue0| l1 + l2 (a_e + 2 l2 a_H),  beta = wdf l2 c_w,
 *    P^-1 v = [alpha v - beta (v x bhat) + (beta^2/alpha)(v.bhat) bhat]
 *             / (alpha^2 + beta^2),
 *
 *  where l2 is the Nyquist symbol of -del2 (sum_d 2/dx_d^2 for the wide
 *  central curl composition of the collocated Cartesian grid, sum_d 4/dx_d^2
 *  for the compact composition of the staggered and RZ grids), l1 = sum_d
 *  2/dx_d that of the first derivative, the hyper-resistive defect uses the
 *  compact-Laplacian Nyquist symbol 2 l2, and wdf = pair_whistler_defect
 *  places the whistler coupling defect inside the curl-curl symbol (the wide
 *  central composition peaks at l2/2). The W-half is stored scaled by the
 *  local weight s_W = max(c_w, 1/l2) (coefficient comp 14), so the
 *  W-residual of each node is weighted by its own E-row coupling ||U_loc||:
 *  this makes the block norm E-space accurate on strong density-contrast
 *  profiles. The curl-free (electrostatic) content of b -- physical and
 *  large in hybrid PIC -- lives on the E-row identity with an O(1) diagonal,
 *  and the whistler stiffness is inverted analytically inside the block.
 *  The m_xF/bF/rF containers hold the E-half, m_xW/bW/rW the scaled W-half.
 *
 *  Multigrid: geometric V(3,3) cycles (factor-2 coarsening while the boxes
 *  allow, same DistributionMapping; coefficient injection, l2 rebuilt per
 *  level; nodal full-weighting restriction, (tri)linear prolongation; 8
 *  bottom sweeps). The V-cycle preconditions an inner right-preconditioned
 *  GMRES (CGS2 orthogonalization, <= inner_max iterations to inner_rtol in
 *  the s_W-weighted block norm, which is E-space accurate): the Krylov
 *  space absorbs the per-level Nyquist rings of the central-difference
 *  curl that smoothing alone cannot, cheaply relative to the outer
 *  iterations (each an evaluation of the particle residual). The inner
 *  solve stops at an RHS-dependent iteration count, so the preconditioner
 *  is NONLINEAR and requires a flexible outer Krylov method
 *  (newton.linear_solver = petsc_ksp with PETSC_OPTIONS="-ksp_type
 *  fgmres"). With inner_max = 0 one application is instead vcycles
 *  V-cycles: a fixed linear operator valid under any outer solver
 *  (amrex_gmres included), at the price of more outer iterations on
 *  problems where the V-cycle stalls.
 *
 *  Physical boundaries: PEC(_Insulator) faces are modeled exactly (identity
 *  tangential-E rows with dead columns and parity-mirrored delta-B/delta-J
 *  ghosts, matching the residual's boundary application; see ClosureClass),
 *  and PMC (Neumann) faces likewise with the dual parities (identity
 *  normal-E rows, tangential B zeroed on the wall); other non-periodic BCs
 *  fall back to a homogeneous zero-extension closure.
 *  With embedded boundaries, EB-covered nodes are masked off the same nodal
 *  eb_update_E flag the residual's Ohm solves consume and act as identity
 *  rows.
 *
 *  With implicit_evolve.use_mass_matrices_pc = 1 the deposited reduced
 *  mass-matrix diagonal Sigma = dJ/dE (the ion response, as consumed by the
 *  EM-PIC preconditioners) is folded through the Hall term as
 *  J_pc = (I + S) + U K, S v = (Sigma v) x B0 / rho_c, with the exact
 *  reduction preserved via the pointwise closed-form (I+S)^-1.
 *
 *  Geometries: the collocated (nodal) grid in Cartesian 1D/2D/3D and in RZ
 *  restricted to the m = 0 azimuthal mode (the residual's CylindricalYee
 *  up/down stencil compositions, on-axis branches and axis mirror parities
 *  are reproduced exactly; the r = 0 axis and the PEC outer wall are part of
 *  the boundary closure), and the staggered (Yee) Cartesian grid with
 *  periodic BCs. On the staggered grid the internal objects keep the
 *  single-MultiFab layout on a CELL-indexed container with per-component
 *  staggered interpretation (E/J components cell-centered along their own
 *  direction, B components nodal along theirs; every component's index
 *  range fits the cell box, and periodic FillBoundary produces the correct
 *  images for both interpretations). Index-space the Yee curls are the
 *  compact up/down differences, the Hall rotation carries the residual's
 *  node-interpolation chain (HybridPICPCUStag), and the smoother is the
 *  node-sandwich block of SmoothLevel (every component-coupling term goes
 *  through the operator's own down-average/cross/up-average chain).
 *
 *  Input parameters (prefix pc_hybrid_pic.):
 *    sigma                 (float, 1.3)    Jacobi damping of the E rows, in (0, 2)
 *    sigma_w               (float, 1.2)    Jacobi damping of the W rows, in (0, 2)
 *    pair_whistler_defect  (float, 0.4)    whistler defect wdf, in (0, 1]
 *    pair_sw_min           (float, 0.0)    floor of the local W-hat scale as a
 *                                          fraction of the global one, in [0, 1]
 *    pe_row_sweeps         (int, 0)        upwind/conduction sweeps on the pe
 *                                          row after its pointwise solve
 *    mg_floor              (int, 4)        coarsest multigrid box width
 *    inner_max             (int, 8)        inner-GMRES iteration cap; 0 = no
 *                                          inner Krylov (linear V-cycle PC)
 *    inner_rtol            (float, 0.3)    inner-GMRES relative tolerance
 *                                          (1.0 = one V-cycle + one Krylov
 *                                          step per application)
 *    vcycles               (int, 1)        V-cycles per application when
 *                                          inner_max = 0
 *    verbose               (bool, false)
 *  and implicit_evolve.pe_ue_cap_fac (float, 1.0): the residual's marker-CFL
 *  cap of |ue| in the pe advance, mirrored by the pe-row leg.
 */

/** Per-level grid constants of the stencil kernels: the Cartesian 1/dx_d^2
 *  and 1/(2 dx_d) in the vector-component slots (unused slots zero), and
 *  the cylindrical radius data of the RZ stencils (index dims (r, z) =
 *  (i, j)). */
struct HybridPICPCGrid
{
    amrex::GpuArray<amrex::Real,3> invdx2 = {0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real,3> inv2dx = {0.0, 0.0, 0.0};
    amrex::Real rmin = 0.0;
    amrex::Real dr = 0.0;
    amrex::Real inv_dr = 0.0;
    amrex::Real inv_dz = 0.0;
};

#if defined(WARPX_DIM_RZ)
/** RZ (m = 0) stencil helpers mirroring the residual's discrete operators
 *  on the collocated grid: the Faraday curl uses the CylindricalYee
 *  "upward" differences and the Ampere curl the "downward" ones, each with
 *  the residual's on-axis branches, and r evaluated at the
 *  staggering-assumed positions the kernels use (nodal for r/theta rows,
 *  cell-centered for the z-row of the Faraday curl and the r-row of the
 *  hyper-resistive Laplacian). Components are (r, theta, z) = (0, 1, 2);
 *  index dims are (r, z) = (i, j). */
namespace hybridpcrz {

    /** True curl of an E-like field, Faraday (upward) stencils. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void CurlF (amrex::Array4<amrex::Real const> const& a,
                amrex::Real& cr, amrex::Real& ct, amrex::Real& cz,
                int i, int j, HybridPICPCGrid const& g)
    {
        using namespace amrex::literals;
        const amrex::Real rn = g.rmin + i*g.dr;             // nodal position
        const amrex::Real rc = g.rmin + (i + 0.5_rt)*g.dr;  // cell-centered
        // (curl a)_r = -d_z a_theta; the Faraday kernel zeroes B_r on axis
        cr = (rn == 0.0_rt) ? 0.0_rt
           : -g.inv_dz*(a(i,j+1,0,1) - a(i,j,0,1));
        // (curl a)_theta = d_z a_r - d_r a_z
        ct = g.inv_dz*(a(i,j+1,0,0) - a(i,j,0,0))
           - g.inv_dr*(a(i+1,j,0,2) - a(i,j,0,2));
        // (curl a)_z = (1/r) d_r (r a_theta), r at the cell center
        cz = (1.0_rt/rc)*g.inv_dr*((rc + 0.5_rt*g.dr)*a(i+1,j,0,1)
                                 - (rc - 0.5_rt*g.dr)*a(i,j,0,1));
    }

    /** True curl of a B-like field, Ampere (downward) stencils with the
     *  residual's on-axis regularizations. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void CurlA (amrex::Array4<amrex::Real const> const& a,
                amrex::Real& cr, amrex::Real& ct, amrex::Real& cz,
                int i, int j, HybridPICPCGrid const& g)
    {
        using namespace amrex::literals;
        const amrex::Real rn = g.rmin + i*g.dr;
        // (curl a)_r = -d_z a_theta
        cr = -g.inv_dz*(a(i,j,0,1) - a(i,j-1,0,1));
        if (rn > 0.5_rt*g.dr) {
            // (curl a)_theta = d_z a_r - d_r a_z
            ct = g.inv_dz*(a(i,j,0,0) - a(i,j-1,0,0))
               - g.inv_dr*(a(i,j,0,2) - a(i-1,j,0,2));
            // (curl a)_z = (1/r) d_r (r a_theta), r at the node
            cz = (1.0_rt/rn)*g.inv_dr*((rn + 0.5_rt*g.dr)*a(i,j,0,1)
                                     - (rn - 0.5_rt*g.dr)*a(i-1,j,0,1));
        } else {
            ct = 0.0_rt;                          // J_theta = 0 on axis
            cz = 4.0_rt*a(i,j,0,1)*g.inv_dr;      // J_z axis regularization
        }
    }

    /** Component-wise hyper-resistive Laplacian of a J-like field, the
     *  residual's RZ forms (r-row at the cell-centered radius including the
     *  -1/r^2 curvature term; theta-row zero on axis; z-row switching to
     *  the control-volume axis limit 4 (a(dr) - a(0))/dr^2, which reads no
     *  r < 0 guard: the residual's guard J_z there is not the even mirror). */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void LapJ (amrex::Array4<amrex::Real const> const& a,
               amrex::Real& lr, amrex::Real& lt, amrex::Real& lz,
               int i, int j, HybridPICPCGrid const& g)
    {
        using namespace amrex::literals;
        const amrex::Real rn = g.rmin + i*g.dr;
        const amrex::Real rc = g.rmin + (i + 0.5_rt)*g.dr;
        const amrex::Real inv_dr2 = g.inv_dr*g.inv_dr;
        const amrex::Real inv_dz2 = g.inv_dz*g.inv_dz;
        auto dzz = [&] (int c) {
            return inv_dz2*(a(i,j-1,0,c) - 2.0_rt*a(i,j,0,c) + a(i,j+1,0,c));
        };
        auto dr_rdr_over_r = [&] (amrex::Real r, int c) {
            return (1.0_rt/r)*inv_dr2*((r + 0.5_rt*g.dr)*(a(i+1,j,0,c) - a(i,j,0,c))
                                     - (r - 0.5_rt*g.dr)*(a(i,j,0,c) - a(i-1,j,0,c)));
        };
        lr = dr_rdr_over_r(rc, 0) + dzz(0) - a(i,j,0,0)/(rc*rc);
        lt = (rn > 0.0_rt)
           ? dr_rdr_over_r(rn, 1) + dzz(1) - a(i,j,0,1)/(rn*rn)
           : 0.0_rt;
        lz = dzz(2);
        if (rn > 0.5_rt*g.dr) {
            lz += dr_rdr_over_r(rn, 2);
        } else {
            lz += 4.0_rt*inv_dr2*(a(i+1,j,0,2) - a(i,j,0,2));
        }
    }

} // namespace hybridpcrz
#endif // WARPX_DIM_RZ

/** Apply (I + S)^-1 to (vx, vy, vz) in place, where S u = (diag(w) u) x bhat
 *  is the deposited-mass-matrix ion response folded through the Hall term
 *  (w_c = Sigma_cc |B0| / rho_c; identity when w = 0). Closed-form 3x3
 *  adjugate inverse; reduces to the analytic rotation inverse
 *  (v - m v x bhat + m^2 (v.bhat) bhat)/(1 + m^2) when w is isotropic. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void HybridPICPCInvIpS (amrex::Real bhx, amrex::Real bhy, amrex::Real bhz,
                        amrex::Real wx, amrex::Real wy, amrex::Real wz,
                        amrex::Real& vx, amrex::Real& vy, amrex::Real& vz)
{
    using namespace amrex::literals;
    // off-diagonal entries of A = I + S: (S u)_x = wy uy bhz - wz uz bhy, etc.
    const amrex::Real M01 =  wy*bhz;
    const amrex::Real M02 = -wz*bhy;
    const amrex::Real M10 = -wx*bhz;
    const amrex::Real M12 =  wz*bhx;
    const amrex::Real M20 =  wx*bhy;
    const amrex::Real M21 = -wy*bhx;
    const amrex::Real a00 = 1.0_rt  - M12*M21;
    const amrex::Real a01 = M02*M21 - M01;
    const amrex::Real a02 = M01*M12 - M02;
    const amrex::Real a10 = M12*M20 - M10;
    const amrex::Real a11 = 1.0_rt  - M02*M20;
    const amrex::Real a12 = M02*M10 - M12;
    const amrex::Real a20 = M10*M21 - M20;
    const amrex::Real a21 = M01*M20 - M21;
    const amrex::Real a22 = 1.0_rt  - M01*M10;
    const amrex::Real invdet = 1.0_rt / (a00 + M01*a10 + M02*a20);
    const amrex::Real ux = vx, uy = vy, uz = vz;
    vx = (a00*ux + a01*uy + a02*uz)*invdet;
    vy = (a10*ux + a11*uy + a12*uz)*invdet;
    vz = (a20*ux + a21*uy + a22*uz)*invdet;
}

/** Pair couplings at one node: inline evaluation of
 *      tc = (I+S)^-1 ( U(aW)/sW - w x F )   and   kk = sW * curl(F),
 *  with F = curl xE staged in a 3-comp array, aW the scaled auxiliary
 *  W-hat, w = theta*dt ue0 (coef comps 11-13) and sW the local W-hat scale
 *  (coef comp 14). Collocated grid: Cartesian central stencils, or the
 *  cylindrical Ampere curl and vector Laplacian in RZ; one call per node
 *  inside each consumer (smoother update, residual). */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void HybridPICPCPairTcKk (amrex::Array4<const amrex::Real> const& coef,
                          amrex::Array4<const amrex::Real> const& aW,
                          amrex::Array4<const amrex::Real> const& F,
                          int i, int j, int k,
                          bool ion,
                          HybridPICPCGrid const& g,
                          amrex::Real& tcx, amrex::Real& tcy, amrex::Real& tcz,
                          amrex::Real& kkx, amrex::Real& kky, amrex::Real& kkz)
{
    using namespace amrex::literals;

    const auto& invdx2 = g.invdx2;
    const auto& inv2dx = g.inv2dx;
    const amrex::Real bhx = coef(i,j,k,0);
    const amrex::Real bhy = coef(i,j,k,1);
    const amrex::Real bhz = coef(i,j,k,2);
    const amrex::Real cw = coef(i,j,k,3);
    const amrex::Real ae = coef(i,j,k,4);
    const amrex::Real ah = coef(i,j,k,5);
    const amrex::Real sW = coef(i,j,k,14);

    const amrex::Real ax = aW(i,j,k,0);
    const amrex::Real ay = aW(i,j,k,1);
    const amrex::Real az = aW(i,j,k,2);

    amrex::Real lx = 0.0_rt, ly = 0.0_rt, lz = 0.0_rt;
#if defined(WARPX_DIM_3D)
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[0]
       + (aW(i,j+1,k,0) - 2.0_rt*ax + aW(i,j-1,k,0))*invdx2[1]
       + (aW(i,j,k+1,0) - 2.0_rt*ax + aW(i,j,k-1,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[0]
       + (aW(i,j+1,k,1) - 2.0_rt*ay + aW(i,j-1,k,1))*invdx2[1]
       + (aW(i,j,k+1,1) - 2.0_rt*ay + aW(i,j,k-1,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[0]
       + (aW(i,j+1,k,2) - 2.0_rt*az + aW(i,j-1,k,2))*invdx2[1]
       + (aW(i,j,k+1,2) - 2.0_rt*az + aW(i,j,k-1,2))*invdx2[2];
    kkx = sW*((F(i,j+1,k,2) - F(i,j-1,k,2))*inv2dx[1]
            - (F(i,j,k+1,1) - F(i,j,k-1,1))*inv2dx[2]);
    kky = sW*((F(i,j,k+1,0) - F(i,j,k-1,0))*inv2dx[2]
            - (F(i+1,j,k,2) - F(i-1,j,k,2))*inv2dx[0]);
    kkz = sW*((F(i+1,j,k,1) - F(i-1,j,k,1))*inv2dx[0]
            - (F(i,j+1,k,0) - F(i,j-1,k,0))*inv2dx[1]);
#elif defined(WARPX_DIM_XZ)
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[0]
       + (aW(i,j+1,k,0) - 2.0_rt*ax + aW(i,j-1,k,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[0]
       + (aW(i,j+1,k,1) - 2.0_rt*ay + aW(i,j-1,k,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[0]
       + (aW(i,j+1,k,2) - 2.0_rt*az + aW(i,j-1,k,2))*invdx2[2];
    kkx = sW*(-(F(i,j+1,k,1) - F(i,j-1,k,1))*inv2dx[2]);
    kky = sW*((F(i,j+1,k,0) - F(i,j-1,k,0))*inv2dx[2]
            - (F(i+1,j,k,2) - F(i-1,j,k,2))*inv2dx[0]);
    kkz = sW*((F(i+1,j,k,1) - F(i-1,j,k,1))*inv2dx[0]);
#elif defined(WARPX_DIM_RZ)
    // cylindrical m = 0: the residual's vector Laplacian of the J-like W and
    // the Ampere (downward) curl of the B-like F, axis branches included
    amrex::ignore_unused(invdx2, inv2dx, k);
    hybridpcrz::LapJ(aW, lx, ly, lz, i, j, g);
    hybridpcrz::CurlA(F, kkx, kky, kkz, i, j, g);
    kkx *= sW;
    kky *= sW;
    kkz *= sW;
#else
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[2];
    kkx = sW*(-(F(i+1,j,k,1) - F(i-1,j,k,1))*inv2dx[2]);
    kky = sW*((F(i+1,j,k,0) - F(i-1,j,k,0))*inv2dx[2]);
    kkz = 0.0_rt;
#endif
    // U(aW)/sW - w x F, then the pointwise ion rotation
    const amrex::Real udx = (ae*ax - ah*lx)/sW;   // dissipative part U_diss W
    const amrex::Real udy = (ae*ay - ah*ly)/sW;
    const amrex::Real udz = (ae*az - ah*lz)/sW;
    tcx = cw*(ay*bhz - az*bhy)/sW + udx;
    tcy = cw*(az*bhx - ax*bhz)/sW + udy;
    tcz = cw*(ax*bhy - ay*bhx)/sW + udz;
    const amrex::Real ex = coef(i,j,k,11);
    const amrex::Real ey = coef(i,j,k,12);
    const amrex::Real ez = coef(i,j,k,13);
    tcx -= ey*F(i,j,k,2) - ez*F(i,j,k,1);
    tcy -= ez*F(i,j,k,0) - ex*F(i,j,k,2);
    tcz -= ex*F(i,j,k,1) - ey*F(i,j,k,0);
    if (ion) {
        // The ions are pushed with E* = E - D, D = eta J_p - eta_H del2 J_p
        // recomputed from the iterate's B in every residual evaluation, so
        // the mass-matrix response is Sigma dE* = Sigma (dE + U_diss W):
        // the Hall-folded ion leg acts on the dissipative coupling as well,
        // S (U_diss W) = (Sigma .* U_diss W) x bhat (eta_H-scaled; large in
        // a boosted-eta_H band at the plasma edge where Sigma is not small).
        const amrex::Real swx = coef(i,j,k,6)*udx;
        const amrex::Real swy = coef(i,j,k,7)*udy;
        const amrex::Real swz = coef(i,j,k,8)*udz;
        tcx += swy*bhz - swz*bhy;
        tcy += swz*bhx - swx*bhz;
        tcz += swx*bhy - swy*bhx;
        HybridPICPCInvIpS(bhx, bhy, bhz,
                          coef(i,j,k,6), coef(i,j,k,7), coef(i,j,k,8),
                          tcx, tcy, tcz);
    }
    // EB-covered rows are identity rows (coef comp 15 = 0 there)
    const amrex::Real mk = coef(i,j,k,15);
    tcx *= mk; tcy *= mk; tcz *= mk;
    kkx *= mk; kky *= mk; kkz *= mk;
}

/** Staggered (Yee) rotation legs of U at container index (i,j,k), mirroring
 *  the residual's Ohm structure: the Hall term is computed at NODES (each
 *  J-like component down-averaged along its cell-centered dimension), then
 *  up-averaged back to each E staggering; the resistive and hyper-resistive
 *  legs act at the component's own location with the node-stored
 *  coefficients up-averaged there. The caller passes the compact
 *  (index-space) Laplacian terms, already evaluated at the component
 *  locations, and applies the ion-response rotation afterwards. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void HybridPICPCUStag (amrex::Array4<amrex::Real const> const& av,
                       amrex::Array4<amrex::Real const> const& coef,
                       int i, int j, int k,
                       amrex::Real lx, amrex::Real ly, amrex::Real lz,
                       amrex::Real& ux, amrex::Real& uy, amrex::Real& uz)
{
    using namespace amrex::literals;
    // c-component of c_w (v~ x bhat) at node (mi,mj,mk)
    auto ncross = [&] (int mi, int mj, int mk, int c) -> amrex::Real
    {
#if defined(WARPX_DIM_3D)
        const amrex::Real vx = 0.5_rt*(av(mi-1,mj,mk,0) + av(mi,mj,mk,0));
        const amrex::Real vy = 0.5_rt*(av(mi,mj-1,mk,1) + av(mi,mj,mk,1));
        const amrex::Real vz = 0.5_rt*(av(mi,mj,mk-1,2) + av(mi,mj,mk,2));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        const amrex::Real vx = 0.5_rt*(av(mi-1,mj,mk,0) + av(mi,mj,mk,0));
        const amrex::Real vy = av(mi,mj,mk,1);
        const amrex::Real vz = 0.5_rt*(av(mi,mj-1,mk,2) + av(mi,mj,mk,2));
#else // WARPX_DIM_1D_Z
        const amrex::Real vx = av(mi,mj,mk,0);
        const amrex::Real vy = av(mi,mj,mk,1);
        const amrex::Real vz = 0.5_rt*(av(mi-1,mj,mk,2) + av(mi,mj,mk,2));
#endif
        // like the residual's enE stage, the node cross carries NO
        // density: c_w rho undoes the 1/rho folded into coef(3), and the
        // caller divides by rho interpolated to the component location
        const amrex::Real cwr = coef(mi,mj,mk,3)*coef(mi,mj,mk,9);
        if (c == 0) {
            return cwr*(vy*coef(mi,mj,mk,2) - vz*coef(mi,mj,mk,1));
        } else if (c == 1) {
            return cwr*(vz*coef(mi,mj,mk,0) - vx*coef(mi,mj,mk,2));
        }
        return cwr*(vx*coef(mi,mj,mk,1) - vy*coef(mi,mj,mk,0));
    };
    // guarded divide: zero density (physical ghosts under the
    // zero-extension closure) means a zero Hall term, not NaN
    auto odiv = [] (amrex::Real num, amrex::Real den) {
        return (den > 0.0_rt) ? num/den : 0.0_rt;
    };
#if defined(WARPX_DIM_3D)
    ux = odiv(ncross(i,j,k,0) + ncross(i+1,j,k,0),
              coef(i,j,k,9) + coef(i+1,j,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i+1,j,k,4))*av(i,j,k,0)
       - 0.5_rt*(coef(i,j,k,5) + coef(i+1,j,k,5))*lx;
    uy = odiv(ncross(i,j,k,1) + ncross(i,j+1,k,1),
              coef(i,j,k,9) + coef(i,j+1,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i,j+1,k,4))*av(i,j,k,1)
       - 0.5_rt*(coef(i,j,k,5) + coef(i,j+1,k,5))*ly;
    uz = odiv(ncross(i,j,k,2) + ncross(i,j,k+1,2),
              coef(i,j,k,9) + coef(i,j,k+1,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i,j,k+1,4))*av(i,j,k,2)
       - 0.5_rt*(coef(i,j,k,5) + coef(i,j,k+1,5))*lz;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    ux = odiv(ncross(i,j,k,0) + ncross(i+1,j,k,0),
              coef(i,j,k,9) + coef(i+1,j,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i+1,j,k,4))*av(i,j,k,0)
       - 0.5_rt*(coef(i,j,k,5) + coef(i+1,j,k,5))*lx;
    uy = odiv(ncross(i,j,k,1), coef(i,j,k,9))
       + coef(i,j,k,4)*av(i,j,k,1) - coef(i,j,k,5)*ly;
    uz = odiv(ncross(i,j,k,2) + ncross(i,j+1,k,2),
              coef(i,j,k,9) + coef(i,j+1,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i,j+1,k,4))*av(i,j,k,2)
       - 0.5_rt*(coef(i,j,k,5) + coef(i,j+1,k,5))*lz;
#else // WARPX_DIM_1D_Z
    ux = odiv(ncross(i,j,k,0), coef(i,j,k,9))
       + coef(i,j,k,4)*av(i,j,k,0) - coef(i,j,k,5)*lx;
    uy = odiv(ncross(i,j,k,1), coef(i,j,k,9))
       + coef(i,j,k,4)*av(i,j,k,1) - coef(i,j,k,5)*ly;
    uz = odiv(ncross(i,j,k,2) + ncross(i+1,j,k,2),
              coef(i,j,k,9) + coef(i+1,j,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i+1,j,k,4))*av(i,j,k,2)
       - 0.5_rt*(coef(i,j,k,5) + coef(i+1,j,k,5))*lz;
#endif
}

/** Staggered (Yee) variant of HybridPICPCPairTcKk: same contract, with the
 *  three staggered replacements. (a) U(aW) goes through the
 *  HybridPICPCUStag node chain (the compact index-space Laplacian is legal
 *  at each component's own location). (b) kk = sW curl(F) uses the compact
 *  downward (Ampere) differences, landing on the E/J staggering. (c) the
 *  convection term w x F mirrors the Hall cross: F (B-staggered) is
 *  down-averaged to nodes over its cell-centered dimensions, crossed with
 *  the node-built w = theta*dt ue0 (coef comps 11-13), and up-averaged
 *  back to each E staggering. The ion rotation keeps the node-collocation
 *  shortcut: node coefficients applied to the edge-centered triple (an
 *  upper-bound treatment, safe for a PC). Cartesian only (the staggered RZ
 *  grid is rejected in Define). */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void HybridPICPCPairTcKkStag (amrex::Array4<const amrex::Real> const& coef,
                              amrex::Array4<const amrex::Real> const& aW,
                              amrex::Array4<const amrex::Real> const& F,
                              int i, int j, int k,
                              bool ion,
                              HybridPICPCGrid const& g,
                              amrex::Real& tcx, amrex::Real& tcy, amrex::Real& tcz,
                              amrex::Real& kkx, amrex::Real& kky, amrex::Real& kkz)
{
    using namespace amrex::literals;

#if defined(WARPX_DIM_RZ)
    amrex::ignore_unused(coef, aW, F, i, j, k, ion, g);
    tcx = tcy = tcz = 0.0_rt;
    kkx = kky = kkz = 0.0_rt;
#else
    const auto& invdx2 = g.invdx2;
    const auto& inv2dx = g.inv2dx;
    const amrex::Real sW = coef(i,j,k,14);
    const amrex::Real ax = aW(i,j,k,0);
    const amrex::Real ay = aW(i,j,k,1);
    const amrex::Real az = aW(i,j,k,2);

    // index-space Laplacian at each component's own staggering (all
    // samples of a component are dx-separated in every direction)
    amrex::Real lx = 0.0_rt, ly = 0.0_rt, lz = 0.0_rt;
#if defined(WARPX_DIM_3D)
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[0]
       + (aW(i,j+1,k,0) - 2.0_rt*ax + aW(i,j-1,k,0))*invdx2[1]
       + (aW(i,j,k+1,0) - 2.0_rt*ax + aW(i,j,k-1,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[0]
       + (aW(i,j+1,k,1) - 2.0_rt*ay + aW(i,j-1,k,1))*invdx2[1]
       + (aW(i,j,k+1,1) - 2.0_rt*ay + aW(i,j,k-1,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[0]
       + (aW(i,j+1,k,2) - 2.0_rt*az + aW(i,j-1,k,2))*invdx2[1]
       + (aW(i,j,k+1,2) - 2.0_rt*az + aW(i,j,k-1,2))*invdx2[2];
    {
        const amrex::Real fx = 2.0_rt*inv2dx[0];
        const amrex::Real fy = 2.0_rt*inv2dx[1];
        const amrex::Real fz = 2.0_rt*inv2dx[2];
        kkx = sW*((F(i,j,k,2) - F(i,j-1,k,2))*fy
                - (F(i,j,k,1) - F(i,j,k-1,1))*fz);
        kky = sW*((F(i,j,k,0) - F(i,j,k-1,0))*fz
                - (F(i,j,k,2) - F(i-1,j,k,2))*fx);
        kkz = sW*((F(i,j,k,1) - F(i-1,j,k,1))*fx
                - (F(i,j,k,0) - F(i,j-1,k,0))*fy);
    }
#elif defined(WARPX_DIM_XZ)
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[0]
       + (aW(i,j+1,k,0) - 2.0_rt*ax + aW(i,j-1,k,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[0]
       + (aW(i,j+1,k,1) - 2.0_rt*ay + aW(i,j-1,k,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[0]
       + (aW(i,j+1,k,2) - 2.0_rt*az + aW(i,j-1,k,2))*invdx2[2];
    {
        const amrex::Real fx = 2.0_rt*inv2dx[0];
        const amrex::Real fz = 2.0_rt*inv2dx[2];
        kkx = sW*(-(F(i,j,k,1) - F(i,j-1,k,1))*fz);
        kky = sW*((F(i,j,k,0) - F(i,j-1,k,0))*fz
                - (F(i,j,k,2) - F(i-1,j,k,2))*fx);
        kkz = sW*((F(i,j,k,1) - F(i-1,j,k,1))*fx);
    }
#else
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[2];
    {
        const amrex::Real fz = 2.0_rt*inv2dx[2];
        kkx = sW*(-(F(i,j,k,1) - F(i-1,j,k,1))*fz);
        kky = sW*((F(i,j,k,0) - F(i-1,j,k,0))*fz);
        kkz = 0.0_rt;
    }
#endif

    // U(aW) through the staggered Hall/resistive node chain
    amrex::Real ux, uy, uz;
    HybridPICPCUStag(aW, coef, i, j, k, lx, ly, lz, ux, uy, uz);

    // c-component of w x F at node (mi,mj,mk): F down-averaged to the node
    // over its cell-centered dimensions (the bn* pattern of the coefficient
    // build), w read at the node
    auto ncrossF = [&] (int mi, int mj, int mk, int c) -> amrex::Real
    {
#if defined(WARPX_DIM_3D)
        const amrex::Real fx = 0.25_rt*(F(mi,mj-1,mk-1,0) + F(mi,mj,mk-1,0)
                                      + F(mi,mj-1,mk,0)   + F(mi,mj,mk,0));
        const amrex::Real fy = 0.25_rt*(F(mi-1,mj,mk-1,1) + F(mi,mj,mk-1,1)
                                      + F(mi-1,mj,mk,1)   + F(mi,mj,mk,1));
        const amrex::Real fz = 0.25_rt*(F(mi-1,mj-1,mk,2) + F(mi,mj-1,mk,2)
                                      + F(mi-1,mj,mk,2)   + F(mi,mj,mk,2));
#elif defined(WARPX_DIM_XZ)
        const amrex::Real fx = 0.5_rt*(F(mi,mj-1,mk,0) + F(mi,mj,mk,0));
        const amrex::Real fy = 0.25_rt*(F(mi-1,mj-1,mk,1) + F(mi,mj-1,mk,1)
                                      + F(mi-1,mj,mk,1)   + F(mi,mj,mk,1));
        const amrex::Real fz = 0.5_rt*(F(mi-1,mj,mk,2) + F(mi,mj,mk,2));
#else
        const amrex::Real fx = 0.5_rt*(F(mi-1,mj,mk,0) + F(mi,mj,mk,0));
        const amrex::Real fy = 0.5_rt*(F(mi-1,mj,mk,1) + F(mi,mj,mk,1));
        const amrex::Real fz = F(mi,mj,mk,2);
#endif
        const amrex::Real wx = coef(mi,mj,mk,11);
        const amrex::Real wy = coef(mi,mj,mk,12);
        const amrex::Real wz = coef(mi,mj,mk,13);
        if (c == 0) { return wy*fz - wz*fy; }
        if (c == 1) { return wz*fx - wx*fz; }
        return wx*fy - wy*fx;
    };
    // up-average the node cross back to each E staggering
#if defined(WARPX_DIM_3D)
    const amrex::Real cvx = 0.5_rt*(ncrossF(i,j,k,0) + ncrossF(i+1,j,k,0));
    const amrex::Real cvy = 0.5_rt*(ncrossF(i,j,k,1) + ncrossF(i,j+1,k,1));
    const amrex::Real cvz = 0.5_rt*(ncrossF(i,j,k,2) + ncrossF(i,j,k+1,2));
#elif defined(WARPX_DIM_XZ)
    const amrex::Real cvx = 0.5_rt*(ncrossF(i,j,k,0) + ncrossF(i+1,j,k,0));
    const amrex::Real cvy = ncrossF(i,j,k,1);
    const amrex::Real cvz = 0.5_rt*(ncrossF(i,j,k,2) + ncrossF(i,j+1,k,2));
#else
    const amrex::Real cvx = ncrossF(i,j,k,0);
    const amrex::Real cvy = ncrossF(i,j,k,1);
    const amrex::Real cvz = 0.5_rt*(ncrossF(i,j,k,2) + ncrossF(i+1,j,k,2));
#endif

    tcx = (ux/sW - cvx)*coef(i,j,k,15);
    tcy = (uy/sW - cvy)*coef(i,j,k,15);
    tcz = (uz/sW - cvz)*coef(i,j,k,15);
    kkx *= coef(i,j,k,15);
    kky *= coef(i,j,k,15);
    kkz *= coef(i,j,k,15);
    if (ion) {
        HybridPICPCInvIpS(coef(i,j,k,0), coef(i,j,k,1), coef(i,j,k,2),
                          coef(i,j,k,6), coef(i,j,k,7), coef(i,j,k,8),
                          tcx, tcy, tcz);
    }
#endif // WARPX_DIM_RZ
}

template <class T, class Ops>
class HybridPICPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;
        using MF = amrex::MultiFab;

        /** V-cycle smoothing depths (Chacon 2025 Sec. 4: V(3,3), 8 bottom
         *  sweeps). The pre-smooth carries the x-is-zero initialization, so
         *  kNuPre >= 1 is required. A grid that cannot be coarsened (see
         *  mg_floor) is smoothed kNuSingle times per cycle instead. */
        static constexpr int kNuPre = 3;
        static constexpr int kNuPost = 3;
        static constexpr int kNuBottom = 8;
        static constexpr int kNuSingle = 48;
        /** Components of a Krylov block: the W-half (0..2) and the E-half
         *  (3..5) of the pair. */
        static constexpr int kNKB = 6;
        /** Largest number of Krylov coefficients passed by value to the
         *  block kernels (GpuArray kernel argument); bounds inner_max. */
        static constexpr int kMaxKry = 64;
        /** Thread-block size of the batched dot-product kernel. */
        static constexpr int kDotThreads = 256;
        /** Iterate ghost layers: one exchange feeds U (1) + curl (1) + curl
         *  (1) per sweep. MG coarsening keeps boxes >= mg_floor >= 4 cells
         *  so periodic exchanges never wrap more than one period. */
        static constexpr int kNGrow = 3;
        /** Number of frozen coefficient components (see m_coef). */
        static constexpr int kNCoef = 16;

        HybridPICPC () = default;
        ~HybridPICPC () override = default;

        HybridPICPC(const HybridPICPC&) = delete;
        HybridPICPC& operator=(const HybridPICPC&) = delete;
        HybridPICPC(HybridPICPC&&) noexcept = delete;
        HybridPICPC& operator=(HybridPICPC&&) noexcept = delete;

        void Define (const T&, Ops*) override;

        /** Refreeze the coefficients (bhat, c_w, eta terms) from the current
         *  Newton linearization state and inject them down the hierarchy. */
        void Update (const T& a_U) override;

        /** Approximately solve J_pc x = b on the primal pair: the inner
         *  GMRES preconditioned by multigrid V-cycles, or vcycles V-cycles
         *  when inner_max = 0. */
        void Apply (T& a_x, const T& a_b) override;

        void printParameters () const override;

        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

        /* Implementation details; public because CUDA requires the enclosing
         * function of an extended device lambda to have public access. */

        /** Pair staging: m_tmp = F = curl xE on growntilebox(2) from the
         *  ghost-filled E-half m_xF[a_lev]; tc and kk are then evaluated
         *  inline per node (HybridPICPCPairTcKk) by every consumer. */
        void StagePair (int a_lev);

        /** Damped joint block-Jacobi sweeps on level a_lev: per sweep, one
         *  ghost exchange of the two iterates, the staging, then the update. */
        void SmoothLevel (int a_lev, int a_nsweeps, bool a_x_is_zero);

        /** rE = bF - xE - tc, rW = bW - xW + kk on the valid region of a_lev. */
        void ResidualLevel (int a_lev);

        /** Inner-GMRES matvec on the fine level, in block form (see m_kryB):
         *  a_blk (ghost-filled, kNKB comps) -> a_out (kNKB comps, valid),
         *  out_W = W - kk, out_E = xE + tc. */
        void ApplyReducedOp (const MF& a_blk, MF& a_out);

        /** Block helpers of the inner GMRES (see m_kryB). SplitBlock reads
         *  the kNKB comps of a_blk starting at a_scomp into m_bW[0] (comps
         *  0..2) and m_bF[0] (comps 3..5) on the valid region; MergeBlock
         *  writes m_xW[0]/m_xF[0] into comps 0..kNKB-1 of a_blk. */
        void SplitBlock (const MF& a_blk, int a_scomp);
        void MergeBlock (MF& a_blk) const;

        /** Batched masked inner products of the block a_w with the first
         *  a_nv Arnoldi basis vectors of m_kryB: a_out[n] = <a_w, v_n> for
         *  n < a_nv and, when a_with_ww, a_out[a_nv] = <a_w, a_w>. One
         *  kernel per box (per-block partial sums, no atomics), ONE
         *  device->host copy, fixed-order host summation (deterministic),
         *  one vector MPI reduction. */
        void BatchDot (const MF& a_w, int a_nv, bool a_with_ww, RT* a_out);

        /** Fused block linear combination on the valid region:
         *  dst(comps a_dcomp..) = a_scale * ( [a_src] - sum_{n<a_nv} a_c[n] v_n ),
         *  with the a_src term present iff a_src != nullptr. One kernel. */
        void BlockAxpy (MF& a_dst, int a_dcomp, const MF* a_src, int a_nv,
                        const amrex::GpuArray<amrex::Real, kMaxKry>& a_c,
                        amrex::Real a_scale) const;

        /** Full-weighting restriction of rW[a_lev] and rF[a_lev] into
         *  bW[a_lev+1] and bF[a_lev+1]. */
        void RestrictLevel (int a_lev);

        /** (Tri)linear prolongation of xW[a_lev+1] and xF[a_lev+1] added
         *  into xW[a_lev] and xF[a_lev]. */
        void ProlongLevel (int a_lev);

        /** Coefficient injection from level a_lev-1 to a_lev. */
        void RestrictCoef (int a_lev);

        /** Which discrete curl composition stage this is: the Faraday curl
         *  (E-like -> B-like, upward stencils in RZ) or the Ampere curl
         *  (B-like -> J-like, downward stencils with the on-axis branches
         *  in RZ). Cartesian uses the same central stencil for both. */
        enum class CurlKind { EtoB, BtoJ };

        /** out = curl a on the valid region grown by a_grow (a must have
         *  ghosts), reading a's components a_scomp..a_scomp+2. */
        void ComputeCurl (const MF& a, MF& out, int a_lev, int a_grow,
                          CurlKind a_kind, int a_scomp = 0) const;

        /** Pointwise (I+S)^-1 on the valid region of the fine-level a. */
        void RotateValid (MF& a) const;

        /** Index-space copy of one solver-vector component into component
         *  a_comp of a 3-component container (MultiFab::Copy requires
         *  matching ixTypes; the cell-indexed container's valid region is a
         *  subset of every staggered component's, so the reads are safe). */
        void CopyCompIn (const MF& a_src, MF& a_dst, int a_comp, bool a_add) const;

        /** Staggered output: x_n = src_n on each component's own staggered
         *  valid box (top nodal planes read the container's ghost ring,
         *  which holds the periodic images). */
        void WriteXOut (const MF& a_src, T& a_x) const;

        /** Physical-boundary closure classes for the vector objects in the
         *  PC pipeline, mirroring the residual's PEC handling exactly
         *  (see ApplyDomainClosure): at a PEC face with normal component n,
         *   EField : linearized E-like objects (the PC input b and the
         *            E-half of the pair): tangential components zeroed ON
         *            the wall plane (ApplyEfieldBoundary is applied to the
         *            iterate AND to the Ohm output in the residual), ghosts
         *            mirror-filled tangential-odd / normal-even;
         *   BField : first-curl (delta-B-like) intermediates: normal
         *            component zeroed on the wall, ghosts tangential-even /
         *            normal-odd (ApplyBfieldBoundary after the Faraday
         *            stage);
         *   JField : delta-J-like objects (the W-half, W = curlcurl dE =
         *            -mu0 dJ_plasma / (theta dt)): ghosts tangential-odd /
         *            normal-even, NO wall write (CalculatePlasmaCurrent
         *            output receives no BC; its wall/ghost values follow
         *            from the BC-filled B);
         *   BCoef  : the coefficient-state total B (mirror ghosts with
         *            B parities, no wall write -- the copied field already
         *            satisfies its BCs);
         *   Coef   : the frozen coefficients (comps 0-2 = bhat with B-parity
         *            signs, the rest even; mirror ghosts, no wall write).
         *  At a PMC (Neumann) face the residual applies the dual rules
         *  (ApplyEfieldBoundary uses the B-parity kernel on E and
         *  ApplyBfieldBoundary the E-parity kernel on B): the roles of the
         *  normal and tangential components swap in every class above
         *  (normal E zeroed on the wall and odd, tangential E even;
         *  tangential B zeroed on the wall and odd, normal B even). */
        enum class ClosureClass { EField, BField, JField, BCoef, Coef };

        /** Apply the physical-boundary closure of class a_cls to a on level
         *  a_lev (PEC/PEC_Insulator and PMC/Neumann faces; other non-periodic
         *  BCs keep the zero-extension), then zero EB-covered nodes on the
         *  fine level when an EB mask is present. The closure indexes
         *  components RELATIVE to a_scomp (a_ncomp < 0: all remaining
         *  components). */
        void ApplyDomainClosure (MF& a, int a_lev, ClosureClass a_cls,
                                 int a_scomp = 0, int a_ncomp = -1) const;

        /** Ghost exchange of a preconditioner container on level a_lev.
         *  When the level consists of a single box spanning the domain the
         *  AMReX FillBoundary reduces to a periodic wrap of the ghost region
         *  onto the valid box, done here as ONE kernel with no host-device
         *  synchronization. Equals FillBoundary for periodically consistent
         *  data (the only case in which FillBoundary is itself well defined
         *  on a nodal periodic box): only ghost cells that overlap a
         *  periodic image of the valid box are written -- nothing at
         *  non-periodic faces (the closures handle those), the valid region
         *  including the duplicated nodal plane is untouched, and the
         *  source of a ghost is its wrap into the primary [lo, lo+N) range.
         *  Falls back to FillBoundary for multi-box layouts. */
        void FillGhosts (MF& a, int a_lev) const;

        /** Enforce the exact identity rows of the true Jacobian on a_out:
         *  wall-tangential components at PEC faces get a_out = a_in, and
         *  EB-covered nodes get a_out = a_in (the PC acts as the identity
         *  there; the covered residual rows are structurally zero). */
        void OverrideIdentityRows (T& a_out, const T& a_in) const;

        /** verbose >= 2 diagnostic after an inner GMRES solve: prints where the
         *  un-reduced inner residual r = b - M x lives (8 x 8 block shares of
         *  ||r_E||^2 and ||b_E||^2 over the nodal domain, W/E split, boundary
         *  planes). Public only because CUDA extended lambdas cannot live in
         *  private member functions. */
        void PrintInnerResidualMap ();

    private:

        /** Recursive V(kNuPre, kNuPost) cycle for the correction on level
         *  a_lev; a non-coarsenable grid smooths kNuSingle times instead. */
        void Vcycle (int a_lev, bool a_x_is_zero);

        /** One preconditioner application for the inner GMRES (one V-cycle
         *  on the block RHS at a_scomp of a_blk); result in m_xW[0]/m_xF[0]. */
        void ApplyMToW (const MF& a_blk, int a_scomp);

        /** Inner GMRES on the pair (<= inner_max its or inner_rtol); result
         *  in m_xW[0]/m_xF[0]. Classical Gram-Schmidt with
         *  reorthogonalization (CGS2): two batched projection passes per
         *  iteration -- two host syncs -- instead of the j+2 sequential
         *  reductions of modified GS. */
        void InnerGMRES ();

        /** Inner-GMRES iteration cap (0: no inner Krylov, m_vcycles
         *  V-cycles per application) and relative tolerance. */
        int m_inner_max = 8;
        RT m_inner_rtol = 0.3;
        int m_vcycles = 1;
        /** Jacobi damping of the E rows. */
        RT m_sigma = 1.3;
        /** Damping of the W-row update. The W eigenvalue of P^-1 A is 1 on
         *  curl-null content, so sigma_w > 1 under-damps W junk while
         *  sigma > 1 speeds the whistler band. */
        RT m_sigma_w = 1.2;
        /** Whistler coupling defect wdf: d_perp = wdf*l2 in the block
         *  (beta = d_perp*c_w) and in the perp-b part of the W increment
         *  (par-b keeps l2). Collocated grid only. */
        RT m_pair_wdf = RT(0.4);
        /** Floor of the local W-hat scale s_W (coef comp 14) as a fraction
         *  of the global scale max(beta, 1)/l2, beta = l2 max(c_w): 0 keeps
         *  the purely local weight (strong density-contrast profiles such
         *  as the FRC), 1 the global one (the GEM setting). */
        RT m_pair_sw_min = RT(0.0);

        /** Ion-response leg active: the deposited reduced mass-matrix
         *  diagonal (raw dJ/dE) is available via
         *  implicit_evolve.use_mass_matrices_pc = 1 (set in Update()). */
        bool m_have_mm = false;

        /** pe-row leg: when the solver vector carries the pe scalar row
         *  (implicit_evolve.pe_newton_unknown), precondition it with a
         *  block-upper-triangular leg instead of the bare identity:
         *  x_p = D_p^-1 b_p with the pointwise diagonal D_p = 1 + coef(10),
         *  coef(10) = theta*dt*[gamma*sum_d |ue_d|/dx_d + conduction], then
         *  b_E <- b_E - grad(x_p/kappa)/rho_e before the E-system solve (the
         *  Ohm gradient coupling dF_E/dpe the identity leg left entirely to
         *  GMRES). |ue| is clamped like the residual's pe advance, ue_c =
         *  c tanh(|ue|/c), c = implicit_evolve.pe_ue_cap_fac * dx_min /
         *  (theta dt). Collocated grids (Cartesian and RZ m = 0), in-loop pe
         *  closure only; inert for E-only vectors. */
        bool m_pe_row_active = false;
        std::unique_ptr<MF> m_pe_row_scr;   //!< ghosted dpe scratch (1 comp)
        amrex::Real m_pe_ue_cap_fac = 1.0;
        /** pc_hybrid_pic.pe_row_sweeps = n: follow the pointwise D_p^-1
         *  pe-row solve with n damped-Jacobi sweeps carrying the signed
         *  UPWIND advection stencil theta*dt*gamma ue0.grad(.) and the
         *  conduction diffusion stencil -c_k del2(.) (c_k sized so its
         *  Nyquist diagonal equals the conduction part of D_p): the
         *  off-diagonal content the pointwise diagonal drops for both legs,
         *  damped by the same D_p. Collocated grids (RZ: area-weighted radial
         *  Laplacian, even parity across the axis). */
        int m_pe_row_sweeps = 0;
        std::unique_ptr<MF> m_pe_row_scr2;  //!< sweep iterate scratch (1 comp)

        /** Pair E-half: per-level E unknown/RHS/residual. */
        std::vector<std::unique_ptr<MF>> m_xF;
        std::vector<std::unique_ptr<MF>> m_bF;
        std::vector<std::unique_ptr<MF>> m_rF;
        /** Ping-pong partners of m_xW/m_xF for the collocated smoother: its
         *  fused update reads the xW neighbours (hyper-resistive Laplacian
         *  inside HybridPICPCPairTcKk) of the iterate it writes, so an
         *  in-place update is a scheduling-dependent Jacobi/Gauss-Seidel
         *  mix. Each sweep writes the partner and the unique_ptrs are
         *  swapped (no copy), making the sweep a deterministic Jacobi
         *  iteration. */
        std::vector<std::unique_ptr<MF>> m_xW2;
        std::vector<std::unique_ptr<MF>> m_xF2;

        bool m_is_defined = false;
        int m_mg_floor = 4;
        int m_verbose = 0;
        /** Per level: single-kernel ghost fill eligible (see FillGhosts). */
        std::vector<bool> m_fastfill_ok;
        /** True when the level solves need no pre-zeroed x: the collocated
         *  sweep writes x = 0 + sigma dE on the first sweep of a zero
         *  initial guess and never reads x before (ghosts are refilled from
         *  the written valid region), so the setVal kernels per level per
         *  V-cycle are skipped. */
        bool m_zero_free = false;

        Ops* m_ops = nullptr;

        int m_nlevels = 1;
        amrex::Vector<amrex::Geometry> m_geom;

        /** Per level: l2, the Nyquist symbol of -del2 (see the class docs). */
        std::vector<RT> m_l2;
        /** Per level: l1 = sum_d 2/dx_d (Nyquist symbol of the first
         *  derivative; sizes the block's convection diagonal). */
        std::vector<RT> m_l1;
        /** Per level: the stencil kernels' grid constants. */
        std::vector<HybridPICPCGrid> m_grid;

        /** Per-node frozen coefficients (per level, kNCoef comps):
         *  comp 0-2  : bhat (unit total B),
         *  comp 3    : c_w = theta*dt |B| / (mu0 rho_e),
         *  comp 4    : a_e = theta*dt eta(rho, |J0|) / mu0,
         *  comp 5    : a_H = theta*dt eta_H / mu0,
         *  comp 6-8  : w_c = Sigma_cc |B| / rho_e (deposited-mass-matrix
         *              ion response through the Hall term; zero unless
         *              implicit_evolve.use_mass_matrices_pc = 1),
         *  comp 9    : rho_e (the floored charge density; used by the
         *              staggered Hall chain, which -- like the residual --
         *              crosses at the nodes WITHOUT the density and divides
         *              by rho interpolated to the component location),
         *  comp 10   : pe-row diagonal D_p - 1 (level 0 only),
         *  comp 11-13: theta*dt ue0 = theta*dt (J_i - J_plasma)/rho_e
         *              (pair convection),
         *  comp 14   : local W-hat scale s_W = max(c_w, 1/l2) (pair),
         *  comp 15   : row mask, 1 at solved nodes and 0 at EB-covered nodes
         *              (set to 1 in Update, zeroed with the other
         *              coefficients by the EB closure, injected to the coarse
         *              levels): the pair couplings tc/kk and the iterates are
         *              multiplied by it so covered rows are identity rows on
         *              every level while the first-curl F stays live there
         *              (the residual's Faraday update is not EB-masked, so
         *              the wall-adjacent delta-J does read the covered
         *              delta-B). */
        std::vector<std::unique_ptr<MF>> m_coef;

        /** Per level (all 3-component): scaled W-half iterate (kNGrow
         *  ghosts), RHS (0), residual (1, for restriction), the staged F =
         *  curl xE, and the staggered node-sandwich scratch. */
        std::vector<std::unique_ptr<MF>> m_xW;
        std::vector<std::unique_ptr<MF>> m_bW;
        std::vector<std::unique_ptr<MF>> m_rW;
        std::vector<std::unique_ptr<MF>> m_tmp;
        std::vector<std::unique_ptr<MF>> m_tmp2;

        /** Fine level: ghosted copy of the PC input b (2 ghosts; doubles as
         *  the B scratch of the coefficient build). */
        std::unique_ptr<MF> m_bE;

        /** Inner-GMRES storage in BLOCK form: every Krylov object carries
         *  kNKB components -- the W-half in comps 0..2, the E-half in comps
         *  3..5 -- so each vector operation is ONE kernel, and the whole
         *  Arnoldi basis lives in a single MultiFab (basis vector j = comps
         *  [j*kNKB, (j+1)*kNKB)), so the Gram-Schmidt projections onto all
         *  j+1 basis vectors are a single batched kernel. */
        std::unique_ptr<MF> m_kryB;   //!< (inner_max+1)*kNKB comps, 0 ghosts
        std::unique_ptr<MF> m_bKB;    //!< RHS block, 0 ghosts
        std::unique_ptr<MF> m_wB;     //!< A M v_j, 0 ghosts
        std::unique_ptr<MF> m_zB;     //!< M v_j, kNGrow ghosts (matvec input)
        /** Batched dot-product scratch: per-block partial sums, slot-major
         *  [slot][global block], one device->host copy per pass. */
        amrex::Gpu::DeviceVector<amrex::Real> m_dot_dev;
        amrex::Gpu::PinnedVector<amrex::Real> m_dot_host;
        std::vector<int> m_dot_nblocks;      //!< per local box
        std::vector<int> m_dot_box_offset;   //!< prefix sum of m_dot_nblocks
        int m_dot_nblocks_total = 0;

        /** Physical-BC closure state: per index dimension and side, whether
         *  the face is a PEC(_Insulator) or PMC (Neumann) wall the closure
         *  must model; the index-dim -> vector-component map; and the EB mask
         *  (the nodal eb_update_E flag on the collocated grid; nullptr
         *  without EB). */
        bool m_has_wall = false;
        /** RZ: the r-lo face is the symmetry axis (rmin == 0). */
        bool m_has_axis = false;
        /** Staggered (Yee) grid: solver-vector components carry their
         *  native staggerings; internally everything is stored on the
         *  nodal-superset layout with per-component staggered
         *  interpretation. E_c is cell-centered along the index dimension
         *  that maps to component c (m_dim2comp) and nodal elsewhere. */
        bool m_staggered = false;
        bool m_wall_pec[AMREX_SPACEDIM][2] = {};
        /** hybrid_pic_model.pec_normal_E_from_interior on the collocated grid: the
         *  residual takes the wall-normal E at a PEC wall node from the first interior
         *  node, so that row is an identity row (dead column) in the preconditioner as
         *  well, like the tangential rows. */
        bool m_pec_normal_from_interior = false;
        /** PMC (Neumann) faces: modeled with the dual parities (collocated
         *  grid; the staggered path requires periodic BCs). */
        bool m_wall_pmc[AMREX_SPACEDIM][2] = {};
        /** pc_hybrid_pic.pmc_closure (default true): model PMC faces; false
         *  keeps the zero-extension fallback there (attribution A/B). */
        bool m_pmc_closure = true;
        /** Non-periodic faces under the zero-extension fallback: on the
         *  staggered grid the hi-side wall plane of components nodal along
         *  the face normal lives in the container's ghost region and is not
         *  modeled -- those rows are made exact identity rows instead. */
        bool m_wall_open[AMREX_SPACEDIM][2] = {};
        int m_dim2comp[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 1, 2)};
        const amrex::iMultiFab* m_eb_flag = nullptr;

};

template <class T, class Ops>
void HybridPICPC<T,Ops>::Define ( const T& a_U, Ops* a_ops )
{
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined, "HybridPICPC::Define() called on defined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.IsDefined(), "HybridPICPC::Define() called with undefined solver vector");

    m_ops = a_ops;

    auto& warpx = WarpX::GetInstance();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.get_pointer_HybridPICModel() != nullptr,
        "pc_hybrid_pic requires the hybrid-PIC (Ohm's law) field solver");
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ABORT_WITH_MESSAGE("pc_hybrid_pic is not implemented for 1D radial geometries");
#elif defined(WARPX_DIM_RZ)
    // RZ support is m = 0 only, with the axis at r = 0 (the residual's
    // CylindricalYee stencils and on-axis branches are mirrored exactly)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::n_rz_azimuthal_modes == 1,
        "pc_hybrid_pic in RZ supports the m = 0 azimuthal mode only");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.Geom(0).ProbLo(0) == 0.0,
        "pc_hybrid_pic in RZ requires the axis (r = 0) in the domain");
#endif
    m_staggered =
        (WarpX::grid_type != ablastr::utils::enums::GridType::Collocated);
    if (m_staggered) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::grid_type == ablastr::utils::enums::GridType::Staggered,
            "pc_hybrid_pic supports the collocated and staggered (Yee) "
            "grids only");
    }

    // Physical-BC closure: detect PEC(_Insulator) faces so the PC's operator
    // reproduces the residual's boundary rows (identity wall-tangential-E
    // rows with dead columns; parity-mirrored ghosts for the delta-B /
    // delta-J stages). Other non-periodic BCs keep the zero-extension
    // closure (warned once).
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    m_dim2comp[0] = 0; m_dim2comp[1] = 2;
#elif defined(WARPX_DIM_1D_Z)
    m_dim2comp[0] = 2;
#endif
#if defined(WARPX_DIM_RZ)
    m_has_axis = true;
#endif
    {
        // read before the face classification below uses it
        const amrex::ParmParse pp_bc("pc_hybrid_pic");
        pp_bc.query("pmc_closure", m_pmc_closure);
        m_pec_normal_from_interior =
            warpx.get_pointer_HybridPICModel()->m_pec_normal_E_from_interior && !m_staggered;
        const auto& fbc_lo = a_ops->GetFieldBoundaryLo();
        const auto& fbc_hi = a_ops->GetFieldBoundaryHi();
        for (int d = 0; d < AMREX_SPACEDIM; d++) {
            if (warpx.Geom(0).isPeriodic(d)) { continue; }
            for (int side = 0; side < 2; side++) {
                const FieldBoundaryType bt =
                    (side == 0) ? fbc_lo[d] : fbc_hi[d];
                if (bt == FieldBoundaryType::PEC
                    || bt == FieldBoundaryType::PEC_Insulator) {
                    m_wall_pec[d][side] = true;
                    m_has_wall = true;
                } else if (bt == FieldBoundaryType::PMC && !m_staggered
                           && m_pmc_closure) {
                    // Neumann is an alias of PMC
                    m_wall_pmc[d][side] = true;
                    m_has_wall = true;
                } else {
                    m_wall_open[d][side] = true;
                    ablastr::warn_manager::WMRecordWarning("HybridPICPC",
                        std::string("pc_hybrid_pic: non-periodic, non-PEC "
                        "field boundary detected; the preconditioner uses ")
                        + (m_staggered
                           ? "a homogeneous zero-extension ghost closure with "
                             "exact identity rows on the unmodeled hi-side "
                             "wall planes on that face (approximate boundary "
                             "rows)."
                           : "a homogeneous zero-extension closure on that "
                             "face (approximate boundary rows)."));
                }
            }
        }
    }

    const amrex::ParmParse pp("pc_hybrid_pic");
    pp.query("verbose", m_verbose);
    pp.query("mg_floor", m_mg_floor);
    pp.query("inner_max", m_inner_max);
    pp.query("vcycles", m_vcycles);
    pp.query("sigma", m_sigma);
    pp.query("pe_row_sweeps", m_pe_row_sweeps);
    {
        amrex::Real swm = static_cast<amrex::Real>(m_pair_sw_min);
        pp.query("pair_sw_min", swm);
        m_pair_sw_min = static_cast<RT>(swm);
        amrex::Real irtol = static_cast<amrex::Real>(m_inner_rtol);
        pp.query("inner_rtol", irtol);
        m_inner_rtol = static_cast<RT>(irtol);
        amrex::Real sgw = static_cast<amrex::Real>(m_sigma_w);
        pp.query("sigma_w", sgw);
        m_sigma_w = static_cast<RT>(sgw);
        amrex::Real wdf = static_cast<amrex::Real>(m_pair_wdf);
        pp.query("pair_whistler_defect", wdf);
        m_pair_wdf = static_cast<RT>(wdf);
        // mirror the residual's marker-CFL cap factor (ThetaImplicitHybrid)
        const amrex::ParmParse pp_impl("implicit_evolve");
        pp_impl.query("pe_ue_cap_fac", m_pe_ue_cap_fac);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_sigma > 0.0 && m_sigma < 2.0,
        "pc_hybrid_pic.sigma must be in (0, 2)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_sigma_w > RT(0.0) && m_sigma_w < RT(2.0),
        "pc_hybrid_pic.sigma_w must be in (0, 2)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_pair_wdf > RT(0.0) && m_pair_wdf <= RT(1.0),
        "pc_hybrid_pic.pair_whistler_defect must be in (0, 1]");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_inner_max >= 0 && m_inner_max + 1 <= kMaxKry,
        "pc_hybrid_pic.inner_max must be in [0, 63] (Krylov coefficients are "
        "passed to the block kernels by value)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_inner_rtol > RT(0.0) && m_vcycles > 0,
        "pc_hybrid_pic.inner_rtol and vcycles must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_pair_sw_min >= RT(0.0) && m_pair_sw_min <= RT(1.0),
        "pc_hybrid_pic.pair_sw_min must be in [0, 1]");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_pe_row_sweeps >= 0
                                     && !(m_pe_row_sweeps > 0 && m_staggered),
        "pc_hybrid_pic.pe_row_sweeps must be >= 0 (collocated grid only)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mg_floor >= 4,
        "pc_hybrid_pic.mg_floor must be >= 4");
    if (m_inner_max > 0) {
        // The inner GMRES stops at an RHS-dependent iteration count, so the
        // preconditioner is nonlinear: classical (fixed-M) right-PC GMRES is
        // silently invalid. Enforce a flexible outer solver; the KSP type
        // itself is checked after KSPSetFromOptions in WarpX_PETSc.cpp.
        std::string nlsolver = "newton";
        const amrex::ParmParse pp_ie("implicit_evolve");
        pp_ie.query("nonlinear_solver", nlsolver);
        if (nlsolver == "newton") {
            std::string linsolver = "amrex_gmres";
            const amrex::ParmParse pp_newton("newton");
            pp_newton.query("linear_solver", linsolver);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(linsolver == "petsc_ksp",
                "pc_hybrid_pic with an inner Krylov solve (inner_max > 0) is a "
                "nonlinear preconditioner and requires a flexible outer Krylov "
                "solver: set newton.linear_solver = petsc_ksp and "
                "PETSC_OPTIONS=\"-ksp_type fgmres\", or set inner_max = 0");
        }
    }
    // Physical boundaries (PEC walls, open zero-extension faces, the RZ
    // axis) are modeled by the closures of the collocated path: the E-half
    // is EField-classed at every fill, the RHS b~ is closured before its
    // copy into the E-half, and the output passes through
    // OverrideIdentityRows. The staggered node-sandwich staging and output
    // exchange assume periodic ghosts.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_staggered || warpx.Geom(0).isAllPeriodic(),
        "pc_hybrid_pic on the staggered (Yee) grid requires fully periodic "
        "BCs (physical boundaries and RZ: collocated grid only)");

    // Storage on the solver-vector layout (single AMR level); the MG
    // hierarchy coarsens that grid while every box stays coarsenable,
    // keeping the DistributionMapping so transfers are local.
    const amrex::MultiFab* E0 = a_U.getArrayVec()[0][0];
    const amrex::DistributionMapping& dm = E0->DistributionMap();

    // Container layout: nodal on the collocated grid (the solver vector's
    // own staggering); CELL-indexed on the staggered grid (see class docs)
    amrex::Vector<amrex::BoxArray> ba_node;
    ba_node.push_back(amrex::convert(E0->boxArray(),
        m_staggered ? amrex::IntVect::TheCellVector()
                    : amrex::IntVect::TheNodeVector()));
    m_geom.clear();
    m_geom.push_back(warpx.Geom(0));
    {
        amrex::BoxArray ba_cell = amrex::convert(ba_node[0], amrex::IntVect::TheCellVector());
        while (true) {
            const amrex::Geometry& fg = m_geom.back();
            // min box width mg_floor >= 4: periodic exchanges of kNGrow
            // ghosts must not wrap more than one period
            if (!fg.Domain().coarsenable(2, m_mg_floor)
                || !ba_cell.coarsenable(2, m_mg_floor)) { break; }
            ba_cell.coarsen(2);
            ba_node.push_back(amrex::convert(ba_cell, ba_node[0].ixType()));
            amrex::Array<int,AMREX_SPACEDIM> is_per{};
            for (int d = 0; d < AMREX_SPACEDIM; d++) { is_per[d] = fg.isPeriodic(d); }
            m_geom.emplace_back(amrex::coarsen(fg.Domain(), 2), fg.ProbDomain(),
                                static_cast<int>(fg.Coord()), is_per);
        }
        if (m_geom.size() == 1) {
            amrex::Print() << "pc_hybrid_pic: grid is not coarsenable; each V-cycle is "
                           << kNuSingle << " smoothing sweeps\n";
        }
    }
    m_nlevels = static_cast<int>(m_geom.size());

    // single-kernel ghost fill eligibility (FillGhosts): one box spanning
    // the whole (index-type-converted) domain on that level
    m_fastfill_ok.assign(m_nlevels, false);
    for (int lev = 0; lev < m_nlevels; lev++) {
        const amrex::BoxArray& ba = ba_node[lev];
        const amrex::Box dom_t = amrex::convert(m_geom[lev].Domain(), ba.ixType());
        m_fastfill_ok[lev] = (ba.size() == 1) && (ba[0] == dom_t);
    }
    m_zero_free = !m_staggered;   // EB (set in Update) checked at use

#if defined(WARPX_DIM_3D)
    const int dirs[3] = {0, 1, 2};
    constexpr int ndirs = 3;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const int dirs[2] = {0, 2};
    constexpr int ndirs = 2;
#else // WARPX_DIM_1D_Z
    const int dirs[1] = {2};
    constexpr int ndirs = 1;
#endif
    m_l2.assign(m_nlevels, 0.0);
    m_l1.assign(m_nlevels, 0.0);
    m_grid.assign(m_nlevels, HybridPICPCGrid{});
    for (int lev = 0; lev < m_nlevels; lev++) {
        const auto* dx = m_geom[lev].CellSize();
        for (int d = 0; d < ndirs; d++) {
            const amrex::Real h = dx[d];
            m_l1[lev] += 2.0_rt / h;
            m_grid[lev].inv2dx[dirs[d]] = 0.5_rt / h;
            m_grid[lev].invdx2[dirs[d]] = 1.0_rt / (h*h);
#if defined(WARPX_DIM_RZ)
            // the RZ curl-curl is the compact down/up composition, whose
            // Nyquist symbol reaches 4/h^2 per direction (the Cartesian
            // wide central composition peaks at half that): size the
            // defect diagonal accordingly so the sigma < 2 damping bound
            // keeps its Cartesian semantics. The same applies to the
            // staggered (Yee) composition in Cartesian geometry.
            m_l2[lev] += 4.0_rt / (h*h);
#else
            m_l2[lev] += (m_staggered ? 4.0_rt : 2.0_rt) / (h*h);
#endif
        }
#if defined(WARPX_DIM_RZ)
        m_grid[lev].rmin = static_cast<amrex::Real>(m_geom[lev].ProbLo(0));
        m_grid[lev].dr = dx[0];
        m_grid[lev].inv_dr = 1.0_rt / dx[0];
        m_grid[lev].inv_dz = 1.0_rt / dx[1];
#endif
    }

    m_coef.resize(m_nlevels);
    m_xW.resize(m_nlevels);
    m_bW.resize(m_nlevels);
    m_rW.resize(m_nlevels);
    m_tmp.resize(m_nlevels);
    m_xF.resize(m_nlevels);
    m_bF.resize(m_nlevels);
    m_rF.resize(m_nlevels);
    if (m_staggered) {
        m_tmp2.resize(m_nlevels);
    } else {
        m_xW2.resize(m_nlevels);
        m_xF2.resize(m_nlevels);
    }

    for (int lev = 0; lev < m_nlevels; lev++) {
        const amrex::BoxArray& ba = ba_node[lev];
        // kNGrow coefficient ghost layers: the staged F = curl xE is
        // evaluated in a 2-ring around the valid region, and the staggered
        // up-averages read the coefficients one ring beyond that; internal
        // ghosts are exchanged in Update(), physical ghosts follow the
        // closures
        m_coef[lev] = std::make_unique<MF>(ba, dm, kNCoef, kNGrow);
        m_coef[lev]->setVal(0.0);
        // Domain (non-periodic) ghosts stay zero: homogeneous Dirichlet
        // closure on every level (exchanges fill periodic/internal only)
        auto alloc = [&] (std::vector<std::unique_ptr<MF>>& a_v, int a_ngrow) {
            a_v[lev] = std::make_unique<MF>(ba, dm, 3, a_ngrow);
            a_v[lev]->setVal(0.0);
        };
        alloc(m_xW, kNGrow);
        alloc(m_xF, kNGrow);
        alloc(m_tmp, kNGrow);
        alloc(m_rW, 1);
        alloc(m_rF, 1);
        alloc(m_bW, 0);
        alloc(m_bF, 0);
        if (m_staggered) {
            alloc(m_tmp2, kNGrow);
        } else {
            // ping-pong partners of the iterates (see m_xW2)
            alloc(m_xW2, kNGrow);
            alloc(m_xF2, kNGrow);
        }
    }
    m_bE = std::make_unique<MF>(ba_node[0], dm, 3, 2);
    m_bE->setVal(0.0);
    m_pe_row_scr = std::make_unique<MF>(ba_node[0], dm, 1, 1);
    m_pe_row_scr->setVal(0.0);
    if (m_pe_row_sweeps > 0) {
        m_pe_row_scr2 = std::make_unique<MF>(ba_node[0], dm, 1, 1);
        m_pe_row_scr2->setVal(0.0);
    }

    if (m_inner_max > 0) {
        // block Krylov storage (see m_kryB)
        m_kryB = std::make_unique<MF>(ba_node[0], dm, (m_inner_max + 1) * kNKB, 0);
        m_bKB  = std::make_unique<MF>(ba_node[0], dm, kNKB, 0);
        m_wB   = std::make_unique<MF>(ba_node[0], dm, kNKB, 0);
        m_zB   = std::make_unique<MF>(ba_node[0], dm, kNKB, kNGrow);
        m_kryB->setVal(0.0);
        m_bKB->setVal(0.0);
        m_wB->setVal(0.0);
        m_zB->setVal(0.0);
        // batched-dot launch layout: ~2 cells per thread, at most 512
        // blocks per box; partial sums are slot-major over all local boxes
        m_dot_nblocks.clear();
        m_dot_box_offset.clear();
        m_dot_nblocks_total = 0;
        for (amrex::MFIter mfi(*m_wB, false); mfi.isValid(); ++mfi) {
            const amrex::Long npts = mfi.validbox().numPts();
            int nb = static_cast<int>((npts + 2*kDotThreads - 1) / (2*kDotThreads));
            nb = std::max(1, std::min(nb, 512));
            m_dot_box_offset.push_back(m_dot_nblocks_total);
            m_dot_nblocks.push_back(nb);
            m_dot_nblocks_total += nb;
        }
        const std::size_t ndot = static_cast<std::size_t>(m_inner_max + 2)
            * static_cast<std::size_t>(std::max(m_dot_nblocks_total, 1));
        m_dot_dev.resize(ndot);
        m_dot_host.resize(ndot);
    }

    m_is_defined = true;
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::printParameters () const
{
    using namespace amrex;
    Print() << "Hybrid-PIC EMHD preconditioner (pc_hybrid_pic):\n";
    Print() << "  grid: " << (m_staggered ? "staggered (Yee)" : "collocated")
            << "\n";
    Print() << "  formulation: primal pair (xE, W), joint block Jacobi "
            << "(convection exact)\n";
    if (m_inner_max > 0) {
        Print() << "  solve: MG V(" << kNuPre << "," << kNuPost << ") (" << kNuBottom
                << " bottom sweeps, " << m_nlevels << " level(s)) inside GMRES(<= "
                << m_inner_max << " its, rtol " << m_inner_rtol
                << " in the s_W-weighted block norm); requires a flexible outer "
                << "solver\n";
    } else {
        Print() << "  solve: " << m_vcycles << " MG V(" << kNuPre << "," << kNuPost
                << ") cycle(s) per application (" << kNuBottom << " bottom sweeps, "
                << m_nlevels << " level(s)); a fixed linear operator\n";
    }
    for (int lev = 0; lev < m_nlevels; lev++) {
        Print() << "    level " << lev << ": " << m_geom[lev].Domain()
                << ", diag l2 = " << m_l2[lev] << "\n";
    }
    Print() << "  damping: " << m_sigma << " (E rows), " << m_sigma_w
            << " (W rows)\n";
    Print() << "  block defects: hyper-resistive Nyquist 2*l2, whistler "
            << m_pair_wdf << "*l2 (pair_whistler_defect)\n";
    if (m_pair_sw_min > RT(0.0)) {
        Print() << "  W-hat scale floor: " << m_pair_sw_min << " x global (pair_sw_min)\n";
    }
    if (m_pe_row_sweeps > 0) {
        Print() << "  pe-row upwind/conduction sweeps: " << m_pe_row_sweeps << "\n";
    }
    Print() << "  ion response (deposited mass-matrix diagonal, "
            << "implicit_evolve.use_mass_matrices_pc): "
            << ((m_ops != nullptr && m_ops->GetMassMatricesCoeff() != nullptr)
                ? "true" : "false")
            << "\n";
    Print() << "  boundary closure: "
            << (m_has_wall
                ? "PEC/PMC wall model (identity tangential-E rows at PEC, "
                  "normal-E rows at PMC; dead columns, parity-mirrored "
                  "delta-B/delta-J ghosts)"
                : "periodic/zero-extension")
            << (EB::enabled() ? " + EB staircase mask (eb_update_E)" : "")
            << "\n";
    Print() << "  single-kernel ghost fill: "
            << (m_fastfill_ok.empty() || !m_fastfill_ok[0] ? "off" : "on")
            << " (single domain-spanning box per level)\n";
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::Update ( const T& a_U )
{
    using namespace amrex;
    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    BL_PROFILE("HybridPICPC::Update()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(), "HybridPICPC::Update() called on undefined object");
    amrex::ignore_unused(a_U);

    auto& warpx = WarpX::GetInstance();
    const HybridPICModel* hybrid = warpx.get_pointer_HybridPICModel();

    // EB mask (staircase): the nodal eb_update_E flag the residual's Ohm
    // solves consume; on the collocated grid all three components share the
    // nodal staggering, so component 0 serves as the per-node mask
    m_eb_flag = nullptr;
    if (EB::enabled()) {
        m_eb_flag = warpx.GetEBUpdateEFlag()[0][0].get();
    }

    // Bfield_fp holds the internal B^{n+theta} at this point of the Newton
    // iteration (external fields are subtracted at the end of ComputeRHS);
    // the whistler rides on the total field, so add B_external here.
    const bool add_external = hybrid->m_add_external_fields;
    ablastr::fields::VectorField B_fp = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    ablastr::fields::VectorField B_ext;
    if (add_external) {
        B_ext = warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0);
    }
    const amrex::MultiFab* rho = warpx.m_fields.get(FieldType::rho_fp, 0);

    // pe-row leg (see m_pe_row_active): Update() cannot see whether the
    // solver vector carries the scalar row, so activate on the model gates
    // only; Apply() additionally gates on the scalar's presence, making
    // this inert for E-only vectors.
    // collocated grids only (Cartesian and RZ m = 0: the leg is a pointwise
    // diagonal plus a central gradient of a scalar, which carries no metric
    // factor in RZ; the r-gradient is zeroed on the axis by symmetry)
    m_pe_row_active = !m_staggered && !hybrid->m_implicit_use_algebraic_closure;
    const bool pe_row_on = m_pe_row_active;
    // the frozen electron velocity of the convection term
    const ablastr::fields::VectorField Ji_fp =
        warpx.m_fields.get_alldirs(FieldType::current_fp, 0);
    const ablastr::fields::VectorField Jp_fp =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
    const RT gamma_e = static_cast<RT>(hybrid->m_gamma);

    const RT theta_dt = m_ops->GetTheta() * this->m_dt;
    const RT t_now = this->m_time;
    const RT rho_floor = hybrid->m_n_floor * PhysConst::q_e;
    // same C1 smooth floor as the Ohm kernels: the PC coefficients see the
    // identical rho_lim field the residual differentiates
    const RT floor_w = static_cast<RT>(hybrid->m_n_floor_smooth_width) * rho_floor;
    const RT one_over_mu0 = 1.0_rt / PhysConst::mu0;
    const auto eta = hybrid->m_eta;
    const auto eta_h = hybrid->m_eta_h;
    const bool include_hyper = hybrid->m_include_hyper_resistivity_term;
    const bool holmstrom = hybrid->m_holmstrom_vacuum_region;
    const Real vac_rref = static_cast<Real>(hybrid->m_vacuum_weight_r_ref);
    const Real vac_maxf = static_cast<Real>(hybrid->m_vacuum_weight_max_factor);
    const HybridPICPCGrid g0 = m_grid[0];
    amrex::ignore_unused(vac_rref, vac_maxf, g0);

    // J0 = curl(B_tot)/mu0 at the linearization state, for freezing the
    // (possibly J-dependent) resistivity at |J0| rather than |J| = 0
    // (m_bE doubles as ghosted scratch here; Apply() rewrites it)
    for (int n = 0; n < 3; n++) {
        if (m_staggered) {
            CopyCompIn(*B_fp[n], *m_bE, n, false);
            if (add_external) { CopyCompIn(*B_ext[n], *m_bE, n, true); }
        } else {
            amrex::MultiFab::Copy(*m_bE, *B_fp[n], 0, n, 1, 0);
            if (add_external) {
                amrex::MultiFab::Add(*m_bE, *B_ext[n], 0, n, 1, 0);
            }
        }
    }
    FillGhosts(*m_bE, 0);
    ApplyDomainClosure(*m_bE, 0, ClosureClass::BCoef);
    ComputeCurl(*m_bE, *m_tmp[0], 0, 0, CurlKind::BtoJ);   // m_tmp[0] = mu0 J0

    // Reduced mass-matrix diagonal (raw dJ/dE, deposited/synced each Newton
    // iteration in PreLinearSolve) for the optional ion-response leg,
    // provided through implicit_evolve.use_mass_matrices_pc = 1
    const auto* mmpc = m_ops->GetMassMatricesCoeff();
    m_have_mm = (mmpc != nullptr) && !mmpc->empty();
    const bool have_mm = m_have_mm;
    const amrex::MultiFab* Sxx_mf = have_mm ? (*mmpc)[0][0] : nullptr;
    const amrex::MultiFab* Syy_mf = have_mm ? (*mmpc)[0][1] : nullptr;
    const amrex::MultiFab* Szz_mf = have_mm ? (*mmpc)[0][2] : nullptr;
    const int cxx = have_mm ? (Sxx_mf->nComp() - 1)/2 : 0;
    const int cyy = have_mm ? (Syy_mf->nComp() - 1)/2 : 0;
    const int czz = have_mm ? (Szz_mf->nComp() - 1)/2 : 0;
    if (have_mm) {
        // the deposited diagonals are read index-space (per-component E
        // staggerings on the Yee grid, nodal when collocated): the grids
        // must align cell-wise with the coefficient container
        AMREX_ALWAYS_ASSERT(
            amrex::convert(Sxx_mf->boxArray(), amrex::IntVect::TheCellVector())
                == amrex::convert(m_coef[0]->boxArray(),
                                  amrex::IntVect::TheCellVector())
            && Sxx_mf->DistributionMap() == m_coef[0]->DistributionMap());
    }
    const bool stag = m_staggered;
    const auto dxi_pc = m_geom[0].InvCellSizeArray();
    const RT l2_pc = m_l2[0];
    // marker-CFL clamp of |ue| for the pe-row diagonal (see m_pe_row_active):
    // c = pe_ue_cap_fac * dx_min / (theta dt), the residual's cap; <= 0 = none
    amrex::Real dx_min_pc = m_geom[0].CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        dx_min_pc = std::min(dx_min_pc, m_geom[0].CellSize(d));
    }
    const amrex::Real ue_cap_pc =
        static_cast<amrex::Real>(m_pe_ue_cap_fac * dx_min_pc / theta_dt);
    // representative constant conductivity for the PC leg: a spatially
    // varying kappa_e(rho,Te) is sampled at the ambient reference state
    // (n0_ref, Te0) -- the PC need not track the exact coefficient field
    const RT kappa_pc = static_cast<RT>(
        hybrid->m_has_kappa_e_expression
            ? hybrid->m_kappa(hybrid->m_n0_ref * PhysConst::q_e,
                              hybrid->m_elec_temp / PhysConst::q_e)
            : hybrid->m_kappa_e);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_coef[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& coef = m_coef[0]->array(mfi);
        Array4<const Real> const& Bx = B_fp[0]->const_array(mfi);
        Array4<const Real> const& By = B_fp[1]->const_array(mfi);
        Array4<const Real> const& Bz = B_fp[2]->const_array(mfi);
        Array4<const Real> const& r = rho->const_array(mfi);
        Array4<const Real> Bxe, Bye, Bze;
        if (add_external) {
            Bxe = B_ext[0]->const_array(mfi);
            Bye = B_ext[1]->const_array(mfi);
            Bze = B_ext[2]->const_array(mfi);
        }
        Array4<const Real> const& cJ = m_tmp[0]->const_array(mfi);
        Array4<const Real> Sxx, Syy, Szz;
        if (have_mm) {
            Sxx = Sxx_mf->const_array(mfi);
            Syy = Syy_mf->const_array(mfi);
            Szz = Szz_mf->const_array(mfi);
        }
        Array4<const Real> const& Jix = Ji_fp[0]->const_array(mfi);
        Array4<const Real> const& Jiy = Ji_fp[1]->const_array(mfi);
        Array4<const Real> const& Jiz = Ji_fp[2]->const_array(mfi);
        Array4<const Real> const& Jpx = Jp_fp[0]->const_array(mfi);
        Array4<const Real> const& Jpy = Jp_fp[1]->const_array(mfi);
        Array4<const Real> const& Jpz = Jp_fp[2]->const_array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // staggered: down-average each B component to the node over
                // its cell-centered dimensions (the residual's ghost-filled
                // B provides the lo-side reads; on the axis the odd
                // B_theta ghosts average to the physical zero)
                auto bnx = [&] (Array4<const Real> const& B) {
#if defined(WARPX_DIM_3D)
                    return 0.25_rt*(B(i,j-1,k-1) + B(i,j,k-1)
                                  + B(i,j-1,k) + B(i,j,k));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    return 0.5_rt*(B(i,j-1,k) + B(i,j,k));
#else
                    return 0.5_rt*(B(i-1,j,k) + B(i,j,k));
#endif
                };
                auto bny = [&] (Array4<const Real> const& B) {
#if defined(WARPX_DIM_3D)
                    return 0.25_rt*(B(i-1,j,k-1) + B(i,j,k-1)
                                  + B(i-1,j,k) + B(i,j,k));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    return 0.25_rt*(B(i-1,j-1,k) + B(i,j-1,k)
                                  + B(i-1,j,k) + B(i,j,k));
#else
                    return 0.5_rt*(B(i-1,j,k) + B(i,j,k));
#endif
                };
                auto bnz = [&] (Array4<const Real> const& B) {
#if defined(WARPX_DIM_3D)
                    return 0.25_rt*(B(i-1,j-1,k) + B(i,j-1,k)
                                  + B(i-1,j,k) + B(i,j,k));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    return 0.5_rt*(B(i-1,j,k) + B(i,j,k));
#else
                    return B(i,j,k);
#endif
                };
                Real bx_v = stag ? bnx(Bx) : Bx(i,j,k);
                Real by_v = stag ? bny(By) : By(i,j,k);
                Real bz_v = stag ? bnz(Bz) : Bz(i,j,k);
                if (add_external) {
                    bx_v += stag ? bnx(Bxe) : Bxe(i,j,k);
                    by_v += stag ? bny(Bye) : Bye(i,j,k);
                    bz_v += stag ? bnz(Bze) : Bze(i,j,k);
                }
                const Real Bmag = std::sqrt(bx_v*bx_v + by_v*by_v + bz_v*bz_v);
                const Real rho_raw = r(i,j,k,0);
                const Real rho_e = HybridSmoothFloor(rho_raw, rho_floor, floor_w);
                // Holmstrom vacuum region: the residual weights the Hall and
                // pressure terms by HybridExtSubWeight(rho) (0 below the floor),
                // so the whistler, the convection and the Hall-folded ion
                // response carry the same weight; only the resistive and
                // hyper-resistive legs survive in vacuum cells
                Real wvac = 1.0_rt;
                if (holmstrom) {
#if defined(WARPX_DIM_RZ)
                    // statistics-aware floor near the axis (HybridVacuumFloor)
                    const Real rf = HybridVacuumFloor(rho_floor, g0.rmin + i*g0.dr,
                                                      vac_rref, vac_maxf, g0.dr);
#else
                    const Real rf = static_cast<Real>(rho_floor);
#endif
                    wvac = HybridExtSubWeight(rho_raw, rf, floor_w*(rf/rho_floor));
                }
                if (Bmag > 0.0_rt) {
                    coef(i,j,k,0) = bx_v / Bmag;
                    coef(i,j,k,1) = by_v / Bmag;
                    coef(i,j,k,2) = bz_v / Bmag;
                } else {
                    coef(i,j,k,0) = 0.0_rt;
                    coef(i,j,k,1) = 0.0_rt;
                    coef(i,j,k,2) = 0.0_rt;
                }
                coef(i,j,k,3) = wvac * theta_dt * Bmag * one_over_mu0 / rho_e;
                // Resistivity frozen at the linearization state |J0|
                // (|J| = 0 would underestimate anomalous eta(J) in current
                // sheets). The parsers see the raw (unfloored) rho, matching
                // the Ohm's-law solve: flooring would deselect vacuum
                // branches if(rho < rho_vac, ...).
                const Real J0mag = std::sqrt(cJ(i,j,k,0)*cJ(i,j,k,0)
                                           + cJ(i,j,k,1)*cJ(i,j,k,1)
                                           + cJ(i,j,k,2)*cJ(i,j,k,2)) * one_over_mu0;
                coef(i,j,k,4) = theta_dt * eta(rho_raw, J0mag, t_now) * one_over_mu0;
                coef(i,j,k,5) = include_hyper ?
                    theta_dt * eta_h(rho_raw, Bmag) * one_over_mu0 : 0.0_rt;
                // ion-response strengths w_c = Sigma_cc |B_tot| / rho_c
                // (S v = (Sigma v) x B / rho_c through the Hall term; the
                // raw dJ/dE deposit already carries the theta*dt of the
                // implicit velocity integration, so w ~ Omega_ci theta*dt)
                Real w6 = 0.0_rt, w7 = 0.0_rt, w8 = 0.0_rt;
                if (have_mm) {
                    const Real bfac = wvac * Bmag / rho_e;
                    w6 = Sxx(i,j,k,cxx) * bfac;
                    w7 = Syy(i,j,k,cyy) * bfac;
                    w8 = Szz(i,j,k,czz) * bfac;
                }
                coef(i,j,k,6) = w6;
                coef(i,j,k,7) = w7;
                coef(i,j,k,8) = w8;
                coef(i,j,k,9) = rho_e;
                // pe row: pointwise diagonal D_p = 1 + coef(10), coef(10) =
                // theta*dt*[gamma*sum_d|ue_d|/dx_d + conduction]. |ue| from
                // the floored density: in floored cells the fictitious ue
                // only GROWS the diagonal, further damping the correction
                // there -- the safe direction for a PC.
                if (pe_row_on) {
                    const Real irho = 1.0_rt / rho_e;
                    // |ue_d| clamped like the residual's pe advance
                    auto uec = [=] (Real u) {
                        return (ue_cap_pc > 0.0_rt) ? ue_cap_pc * std::tanh(u / ue_cap_pc) : u;
                    };
                    Real adv = 0.0_rt;
#if defined(WARPX_DIM_1D_Z)
                    adv = uec(std::abs(Jiz(i,j,k) - Jpz(i,j,k)) * irho) * dxi_pc[0];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    adv = uec(std::abs(Jix(i,j,k) - Jpx(i,j,k)) * irho) * dxi_pc[0]
                        + uec(std::abs(Jiz(i,j,k) - Jpz(i,j,k)) * irho) * dxi_pc[1];
#elif defined(WARPX_DIM_3D)
                    adv = uec(std::abs(Jix(i,j,k) - Jpx(i,j,k)) * irho) * dxi_pc[0]
                        + uec(std::abs(Jiy(i,j,k) - Jpy(i,j,k)) * irho) * dxi_pc[1]
                        + uec(std::abs(Jiz(i,j,k) - Jpz(i,j,k)) * irho) * dxi_pc[2];
#endif
                    // + the electron-conduction stiffness (kappa_e grad(pe/n)
                    // diffusion diagonal) so D_p tracks the pe row when
                    // conduction dominates the grid scale
                    coef(i,j,k,10) = theta_dt * (gamma_e * adv
                        + (gamma_e - 1.0_rt) * kappa_pc
                          * PhysConst::q_e * l2_pc / rho_e);
                } else {
                    coef(i,j,k,10) = 0.0_rt;
                }
                // frozen electron velocity of the convection term, stored
                // pre-scaled as theta*dt * ue0 = theta*dt (J_i - J_plasma)/rho_c.
                // staggered: each J-like component is down-averaged to the
                // node over its cell-centered dimension (the residual's
                // ghost-filled J provides the lo-side reads), matching the
                // B down-averages above
                {
                    const Real irho = wvac * theta_dt / rho_e;
                    if (stag) {
                        auto jn = [&] (Array4<const Real> const& Ja,
                                       Array4<const Real> const& Jb,
                                       int c) -> Real {
                            amrex::ignore_unused(c);
#if defined(WARPX_DIM_3D)
                            if (c == 0) {
                                return 0.5_rt*(Ja(i-1,j,k) - Jb(i-1,j,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            if (c == 1) {
                                return 0.5_rt*(Ja(i,j-1,k) - Jb(i,j-1,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            return 0.5_rt*(Ja(i,j,k-1) - Jb(i,j,k-1)
                                         + Ja(i,j,k)   - Jb(i,j,k));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                            if (c == 0) {
                                return 0.5_rt*(Ja(i-1,j,k) - Jb(i-1,j,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            if (c == 2) {
                                return 0.5_rt*(Ja(i,j-1,k) - Jb(i,j-1,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            return Ja(i,j,k) - Jb(i,j,k);
#else
                            if (c == 2) {
                                return 0.5_rt*(Ja(i-1,j,k) - Jb(i-1,j,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            return Ja(i,j,k) - Jb(i,j,k);
#endif
                        };
                        coef(i,j,k,11) = jn(Jix, Jpx, 0) * irho;
                        coef(i,j,k,12) = jn(Jiy, Jpy, 1) * irho;
                        coef(i,j,k,13) = jn(Jiz, Jpz, 2) * irho;
                    } else {
                        coef(i,j,k,11) = (Jix(i,j,k) - Jpx(i,j,k)) * irho;
                        coef(i,j,k,12) = (Jiy(i,j,k) - Jpy(i,j,k)) * irho;
                        coef(i,j,k,13) = (Jiz(i,j,k) - Jpz(i,j,k)) * irho;
                    }
                }
                // LOCAL W-hat scale s_W = max(beta_loc, 1)/l2 = max(c_w,
                // 1/l2), so every node's W-residual is weighted by its own
                // E-row coupling ||U_loc|| (a single global scale cannot
                // weight both the vacuum, beta ~ 1/rho_floor, and the plasma
                // of a strong density-contrast profile)
                coef(i,j,k,14) = amrex::max(coef(i,j,k,3), Real(1.0)/Real(l2_pc));
                // row mask: 1 here, 0 at EB-covered nodes after the closure
                coef(i,j,k,15) = 1.0_rt;
            });
    }

    if (m_verbose) {
        // coefficient-field summary of the linearization state (level 0,
        // valid region): the whistler, resistive and hyper-resistive scales
        // in units of the Laplacian symbol l2, the floored density, the
        // convection speed and the W-hat scale, plus the field magnitudes
        // they were built from
        const RT l2v = m_l2[0];
        amrex::Real bfp = 0.0_rt, bext = 0.0_rt;
        for (int n = 0; n < 3; ++n) {
            bfp = amrex::max(bfp, B_fp[n]->norminf(0, 0));
            if (add_external) { bext = amrex::max(bext, B_ext[n]->norminf(0, 0)); }
        }
        amrex::Real ue = 0.0_rt;
        for (int c = 11; c <= 13; ++c) {
            ue = amrex::max(ue, m_coef[0]->norminf(c, 0));
        }
        amrex::Print() << "  pc_hybrid_pic coefficients: |B_fp| " << bfp
                       << ", |B_ext| " << bext
                       << "; c_w l2 [" << m_coef[0]->min(3, 0)*l2v << ", "
                       << m_coef[0]->max(3, 0)*l2v << "]"
                       << ", c_eta l2 [" << m_coef[0]->min(4, 0)*l2v << ", "
                       << m_coef[0]->max(4, 0)*l2v << "]"
                       << ", c_h l2^2 [" << m_coef[0]->min(5, 0)*l2v*l2v << ", "
                       << m_coef[0]->max(5, 0)*l2v*l2v << "]"
                       << ", rho_e [" << m_coef[0]->min(9, 0) << ", "
                       << m_coef[0]->max(9, 0) << "]"
                       << ", |theta dt ue| max " << ue
                       << " (" << ue*std::sqrt(l2v) << " cells)"
                       << ", s_W [" << m_coef[0]->min(14, 0) << ", "
                       << m_coef[0]->max(14, 0) << "]"
                       << (m_coef[0]->contains_nan(0, m_coef[0]->nComp(), 0)
                           || m_coef[0]->contains_inf(0, m_coef[0]->nComp(), 0)
                           ? "  NON-FINITE" : "") << "\n";
    }

    // optional floor of the local W-hat scale at pair_sw_min x the global
    // scale max(beta, 1)/l2, beta = l2 max(c_w) (the peak whistler CFL);
    // before the coefficient exchange and the coarse injection, so every
    // level inherits the floored value
    if (m_pair_sw_min > RT(0.0)) {
        const RT beta = static_cast<RT>(m_l2[0] * m_coef[0]->max(3, 0));
        const amrex::Real swfl = static_cast<amrex::Real>(
            m_pair_sw_min * amrex::max(beta, RT(1.0)) / m_l2[0]);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*m_coef[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<Real> const& c = m_coef[0]->array(mfi);
            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    c(i,j,k,14) = amrex::max(c(i,j,k,14), swfl);
                });
        }
    }

    // Coarse operators by rediscretization: inject the coefficients down the
    // hierarchy (l2 is recomputed per level in Define()) and exchange their
    // internal ghosts for the stencils.
    FillGhosts(*m_coef[0], 0);
    ApplyDomainClosure(*m_coef[0], 0, ClosureClass::Coef);
    for (int lev = 1; lev < m_nlevels; lev++) {
        RestrictCoef(lev);
        FillGhosts(*m_coef[lev], lev);
        ApplyDomainClosure(*m_coef[lev], lev, ClosureClass::Coef);
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::RestrictCoef ( int a_lev )
{
    using namespace amrex;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& c = m_coef[a_lev]->array(mfi);
        Array4<const Real> const& f = m_coef[a_lev-1]->const_array(mfi);
        amrex::ParallelFor(bx, kNCoef - 1,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int m)
            {
                // comps 0-9 (bhat, c_w, a_e, a_H, MM, rho) + the pair's
                // electron velocity (11-13) and local W-hat scale (14);
                // the level-0-only pe-row diagonal (10) is skipped
                const int n = (m < 10) ? m : (m + 1);
#if defined(WARPX_DIM_3D)
                c(i,j,k,n) = f(2*i,2*j,2*k,n);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                c(i,j,k,n) = f(2*i,2*j,k,n);
#else // WARPX_DIM_1D_Z
                c(i,j,k,n) = f(2*i,j,k,n);
#endif
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ComputeCurl ( const MF& a, MF& out, int a_lev, int a_grow,
                                       CurlKind a_kind, int a_scomp ) const
{
    using namespace amrex;

    const HybridPICPCGrid g = m_grid[a_lev];
#if defined(WARPX_DIM_RZ)
    const bool faraday = (a_kind == CurlKind::EtoB);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(a_grow);
        Array4<const Real> const& av = a.const_array(mfi, a_scomp);
        Array4<Real> const& cv = out.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::ignore_unused(k);
                Real cr, ct, cz;
                if (faraday) {
                    hybridpcrz::CurlF(av, cr, ct, cz, i, j, g);
                } else {
                    hybridpcrz::CurlA(av, cr, ct, cz, i, j, g);
                }
                cv(i,j,0,0) = cr;
                cv(i,j,0,1) = ct;
                cv(i,j,0,2) = cz;
            });
    }
#else
    // staggered: compact up (E->B) / down (B->J) index-space differences
    const bool faraday = (a_kind == CurlKind::EtoB);
    const bool stag = m_staggered;
    const auto inv2dx = g.inv2dx;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(a_grow);
        Array4<const Real> const& av = a.const_array(mfi, a_scomp);
        Array4<Real> const& cv = out.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
#if defined(WARPX_DIM_3D)
                if (stag) {
                    const int  s  = faraday ? 0 : -1;
                    const Real ix = 2.0_rt*inv2dx[0];
                    const Real iy = 2.0_rt*inv2dx[1];
                    const Real iz = 2.0_rt*inv2dx[2];
                    cv(i,j,k,0) = (av(i,j+1+s,k,2) - av(i,j+s,k,2))*iy
                                - (av(i,j,k+1+s,1) - av(i,j,k+s,1))*iz;
                    cv(i,j,k,1) = (av(i,j,k+1+s,0) - av(i,j,k+s,0))*iz
                                - (av(i+1+s,j,k,2) - av(i+s,j,k,2))*ix;
                    cv(i,j,k,2) = (av(i+1+s,j,k,1) - av(i+s,j,k,1))*ix
                                - (av(i,j+1+s,k,0) - av(i,j+s,k,0))*iy;
                } else {
                    const Real dyaz = (av(i,j+1,k,2) - av(i,j-1,k,2)) * inv2dx[1];
                    const Real dzay = (av(i,j,k+1,1) - av(i,j,k-1,1)) * inv2dx[2];
                    const Real dzax = (av(i,j,k+1,0) - av(i,j,k-1,0)) * inv2dx[2];
                    const Real dxaz = (av(i+1,j,k,2) - av(i-1,j,k,2)) * inv2dx[0];
                    const Real dxay = (av(i+1,j,k,1) - av(i-1,j,k,1)) * inv2dx[0];
                    const Real dyax = (av(i,j+1,k,0) - av(i,j-1,k,0)) * inv2dx[1];
                    cv(i,j,k,0) = dyaz - dzay;
                    cv(i,j,k,1) = dzax - dxaz;
                    cv(i,j,k,2) = dxay - dyax;
                }
#elif defined(WARPX_DIM_XZ)
                // (i,j) = (x,z); d/dy = 0
                if (stag) {
                    const int  s  = faraday ? 0 : -1;
                    const Real ix = 2.0_rt*inv2dx[0];
                    const Real iz = 2.0_rt*inv2dx[2];
                    cv(i,j,k,0) = -(av(i,j+1+s,k,1) - av(i,j+s,k,1))*iz;
                    cv(i,j,k,1) = (av(i,j+1+s,k,0) - av(i,j+s,k,0))*iz
                                - (av(i+1+s,j,k,2) - av(i+s,j,k,2))*ix;
                    cv(i,j,k,2) = (av(i+1+s,j,k,1) - av(i+s,j,k,1))*ix;
                } else {
                    const Real dzay = (av(i,j+1,k,1) - av(i,j-1,k,1)) * inv2dx[2];
                    const Real dzax = (av(i,j+1,k,0) - av(i,j-1,k,0)) * inv2dx[2];
                    const Real dxaz = (av(i+1,j,k,2) - av(i-1,j,k,2)) * inv2dx[0];
                    const Real dxay = (av(i+1,j,k,1) - av(i-1,j,k,1)) * inv2dx[0];
                    cv(i,j,k,0) = -dzay;
                    cv(i,j,k,1) = dzax - dxaz;
                    cv(i,j,k,2) = dxay;
                }
#else // WARPX_DIM_1D_Z
                // i = z; d/dx = d/dy = 0
                if (stag) {
                    const int  s  = faraday ? 0 : -1;
                    const Real iz = 2.0_rt*inv2dx[2];
                    cv(i,j,k,0) = -(av(i+1+s,j,k,1) - av(i+s,j,k,1))*iz;
                    cv(i,j,k,1) = (av(i+1+s,j,k,0) - av(i+s,j,k,0))*iz;
                    cv(i,j,k,2) = 0.0_rt;
                } else {
                    const Real dzay = (av(i+1,j,k,1) - av(i-1,j,k,1)) * inv2dx[2];
                    const Real dzax = (av(i+1,j,k,0) - av(i-1,j,k,0)) * inv2dx[2];
                    cv(i,j,k,0) = -dzay;
                    cv(i,j,k,1) = dzax;
                    cv(i,j,k,2) = 0.0_rt;
                }
#endif
            });
    }
#endif // WARPX_DIM_RZ
}



template <class T, class Ops>
void HybridPICPC<T,Ops>::StagePair ( int a_lev )
{
    // F = curl xE on grown(gU), so consumers can read its first ring for
    // the inline kk = sW curl F
    const int gU = kNGrow - 1;
    ComputeCurl(*m_xF[a_lev], *m_tmp[a_lev], a_lev, gU, CurlKind::EtoB);
    ApplyDomainClosure(*m_tmp[a_lev], a_lev, ClosureClass::BField);
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::SmoothLevel ( int a_lev, int a_nsweeps, bool a_x_is_zero )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::SmoothLevel()");

    // Joint per-node 2x2 block update (see the class docs): tc and kk are
    // evaluated inline (HybridPICPCPairTcKk) from the staged F = curl xE
    // and the xW neighbors.
    //   rE = bF - xE - tc, rW = bW - xW + kk
    //   P dE = rE - (1/sW) U_loc rW ; dW = rW + sW l2 dE
    const RT l2 = m_l2[a_lev];
    const RT l1p = m_l1[a_lev];
    const RT sigma = m_sigma;
    const RT sigmaw = m_sigma_w;
    const Real wdl2 = static_cast<Real>(m_pair_wdf*l2);
    const bool ionp = m_have_mm;
    const HybridPICPCGrid g = m_grid[a_lev];
#if defined(WARPX_DIM_RZ)
    // RZ collocated block symbols. The RZ curl-curl is the compact down/up
    // composition with max symbol m_l2 = sum_d 4/h_d^2, whereas the Cartesian
    // collocated l2 = sum_d 2/h_d^2 is TWICE the max symbol (sum_d 1/h_d^2) of
    // its wide central composition: the block defect wdf*l2 that is stable
    // under sigma = 1.3 in Cartesian geometry (0.8 x max symbol at wdf 0.4)
    // was only 0.4 x max symbol here and the damped Jacobi over-relaxed the
    // whistler band (RZ pair diverged for c_w l2 >~ 10: the formation coil
    // field, the stiff mirror em-modes reproduction, 2026-09-14). The block
    // is therefore sized on 2 m_l2 (Cartesian semantics) and the
    // hyper-resistive Nyquist on the exact compact Laplacian symbol m_l2.
    // Axis rows: the (1/r) d_r (r .) compositions carry the radial symbol
    // 8/dr^2 instead of 4/dr^2 (curl-curl z-row through the J_z
    // regularization 4 F_theta/dr, hyper-Laplacian r-row through the -1/r^2
    // curvature term at r = dr/2): both symbols are raised accordingly there.
    const bool has_axis = m_has_axis;
    const int iax = m_geom[a_lev].Domain().smallEnd(0);
    const Real wdf = static_cast<Real>(m_pair_wdf);
#endif

    for (int sweep = 0; sweep < a_nsweeps; sweep++) {

        const bool r_is_b = (sweep == 0 && a_x_is_zero);
        if (!r_is_b) {
            // one exchange per half feeds the whole sweep: W is delta-J-like
            // (parity ghosts, no wall write), E gets the E-field closure
            FillGhosts(*m_xW[a_lev], a_lev);
            ApplyDomainClosure(*m_xW[a_lev], a_lev, ClosureClass::JField);
            FillGhosts(*m_xF[a_lev], a_lev);
            ApplyDomainClosure(*m_xF[a_lev], a_lev, ClosureClass::EField);
            StagePair(a_lev);
        }

        if (m_staggered) {
            // NODE-SANDWICH block: on the Yee grid the pointwise block's
            // cross-couplings would act between samples h/2 apart -- in
            // Fourier space a phase e^{i(theta_a-theta_b)/2} the real
            // rotation cannot represent. Every component-coupling term
            // of the update therefore goes through the operator's own
            // down-average/cross/up-average chain (symbol c_a c_b: real,
            // damped, phase-free); every diagonal term stays pointwise.
            //
            // stage A: exact residuals rE -> m_rF, rW -> m_rW (valid),
            //          then one exchange each fills the gather ghosts
            for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
                Array4<const Real> const& bW = m_bW[a_lev]->const_array(mfi);
                Array4<const Real> const& bF = m_bF[a_lev]->const_array(mfi);
                Array4<const Real> const& Fc = m_tmp[a_lev]->const_array(mfi);
                Array4<const Real> const& xW = m_xW[a_lev]->const_array(mfi);
                Array4<const Real> const& xE = m_xF[a_lev]->const_array(mfi);
                Array4<Real> const& rE = m_rF[a_lev]->array(mfi);
                Array4<Real> const& rW = m_rW[a_lev]->array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (r_is_b) {
                            for (int c = 0; c < 3; c++) {
                                rE(i,j,k,c) = bF(i,j,k,c);
                                rW(i,j,k,c) = bW(i,j,k,c);
                            }
                            return;
                        }
                        Real tcx, tcy, tcz, kkx, kky, kkz;
                        HybridPICPCPairTcKkStag(coef, xW, Fc, i, j, k, ionp, g,
                                                tcx, tcy, tcz, kkx, kky, kkz);
                        rE(i,j,k,0) = bF(i,j,k,0) - xE(i,j,k,0) - tcx;
                        rE(i,j,k,1) = bF(i,j,k,1) - xE(i,j,k,1) - tcy;
                        rE(i,j,k,2) = bF(i,j,k,2) - xE(i,j,k,2) - tcz;
                        rW(i,j,k,0) = bW(i,j,k,0) - xW(i,j,k,0) + kkx;
                        rW(i,j,k,1) = bW(i,j,k,1) - xW(i,j,k,1) + kky;
                        rW(i,j,k,2) = bW(i,j,k,2) - xW(i,j,k,2) + kkz;
                    });
            }
            FillGhosts(*m_rF[a_lev], a_lev);
            FillGhosts(*m_rW[a_lev], a_lev);
            // stage B: node coupling q = Q sD~ + P^-1 sX~ -> m_tmp2 on
            // grow(1) (stage C reads q at the two nodes of each edge)
            for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.growntilebox(1);
                Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
                Array4<const Real> const& rE = m_rF[a_lev]->const_array(mfi);
                Array4<const Real> const& rW = m_rW[a_lev]->const_array(mfi);
                Array4<Real> const& qq = m_tmp2[a_lev]->array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        // A- gather of both residual halves to the node
                        // (each component averaged along its own
                        // cell-centered dimension)
                        Real gEx, gEy, gEz, gWx, gWy, gWz;
#if defined(WARPX_DIM_3D)
                        gEx = 0.5_rt*(rE(i-1,j,k,0) + rE(i,j,k,0));
                        gEy = 0.5_rt*(rE(i,j-1,k,1) + rE(i,j,k,1));
                        gEz = 0.5_rt*(rE(i,j,k-1,2) + rE(i,j,k,2));
                        gWx = 0.5_rt*(rW(i-1,j,k,0) + rW(i,j,k,0));
                        gWy = 0.5_rt*(rW(i,j-1,k,1) + rW(i,j,k,1));
                        gWz = 0.5_rt*(rW(i,j,k-1,2) + rW(i,j,k,2));
#elif defined(WARPX_DIM_XZ)
                        gEx = 0.5_rt*(rE(i-1,j,k,0) + rE(i,j,k,0));
                        gEy = rE(i,j,k,1);
                        gEz = 0.5_rt*(rE(i,j-1,k,2) + rE(i,j,k,2));
                        gWx = 0.5_rt*(rW(i-1,j,k,0) + rW(i,j,k,0));
                        gWy = rW(i,j,k,1);
                        gWz = 0.5_rt*(rW(i,j-1,k,2) + rW(i,j,k,2));
#else
                        gEx = rE(i,j,k,0);
                        gEy = rE(i,j,k,1);
                        gEz = 0.5_rt*(rE(i-1,j,k,2) + rE(i,j,k,2));
                        gWx = rW(i,j,k,0);
                        gWy = rW(i,j,k,1);
                        gWz = 0.5_rt*(rW(i-1,j,k,2) + rW(i,j,k,2));
#endif
                        const Real bhx = coef(i,j,k,0);
                        const Real bhy = coef(i,j,k,1);
                        const Real bhz = coef(i,j,k,2);
                        const Real cw = coef(i,j,k,3);
                        const Real aU = coef(i,j,k,4) + coef(i,j,k,5)*l2;
                        const Real uem = std::sqrt(
                            coef(i,j,k,11)*coef(i,j,k,11)
                          + coef(i,j,k,12)*coef(i,j,k,12)
                          + coef(i,j,k,13)*coef(i,j,k,13));
                        const Real alpha = 1.0_rt + l1p*uem + l2*aU;
                        const Real beta = l2*cw;
                        const Real isW = 1.0_rt/coef(i,j,k,14);
                        // sD = gE - (1/sW) aU gW ; sX = -(1/sW) cw gW x b
                        const Real sDx = gEx - isW*aU*gWx;
                        const Real sDy = gEy - isW*aU*gWy;
                        const Real sDz = gEz - isW*aU*gWz;
                        const Real sXx = -isW*cw*(gWy*bhz - gWz*bhy);
                        const Real sXy = -isW*cw*(gWz*bhx - gWx*bhz);
                        const Real sXz = -isW*cw*(gWx*bhy - gWy*bhx);
                        // q = Q sD + P^-1 sX, with t = sD + sX:
                        //   q = [alpha sX - beta (t x b)
                        //        + (beta^2/alpha)(t.b) b
                        //        - (beta^2/alpha) sD] / (alpha^2+beta^2)
                        const Real tx = sDx + sXx;
                        const Real ty = sDy + sXy;
                        const Real tz = sDz + sXz;
                        const Real txb_x = ty*bhz - tz*bhy;
                        const Real txb_y = tz*bhx - tx*bhz;
                        const Real txb_z = tx*bhy - ty*bhx;
                        const Real tdb = tx*bhx + ty*bhy + tz*bhz;
                        const Real b2oa = beta*beta/alpha;
                        const Real inv_a2b2 = 1.0_rt/(alpha*alpha + beta*beta);
                        qq(i,j,k,0) = inv_a2b2*(alpha*sXx - beta*txb_x
                                      + b2oa*(tdb*bhx - sDx));
                        qq(i,j,k,1) = inv_a2b2*(alpha*sXy - beta*txb_y
                                      + b2oa*(tdb*bhy - sDy));
                        qq(i,j,k,2) = inv_a2b2*(alpha*sXz - beta*txb_z
                                      + b2oa*(tdb*bhz - sDz));
                    });
            }
            // stage C: dE = sD/alpha + A+ q ; xE += sigma dE,
            // xW += sigma_w (rW + sW l2 dE)
            for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
                Array4<const Real> const& rE = m_rF[a_lev]->const_array(mfi);
                Array4<const Real> const& rW = m_rW[a_lev]->const_array(mfi);
                Array4<const Real> const& qq = m_tmp2[a_lev]->const_array(mfi);
                Array4<Real> const& xW = m_xW[a_lev]->array(mfi);
                Array4<Real> const& xE = m_xF[a_lev]->array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        const Real aU = coef(i,j,k,4) + coef(i,j,k,5)*l2;
                        const Real uem = std::sqrt(
                            coef(i,j,k,11)*coef(i,j,k,11)
                          + coef(i,j,k,12)*coef(i,j,k,12)
                          + coef(i,j,k,13)*coef(i,j,k,13));
                        const Real ialpha = 1.0_rt/(1.0_rt + l1p*uem + l2*aU);
                        const Real swl = coef(i,j,k,14);
                        const Real isW = 1.0_rt/swl;
                        const Real l2s = static_cast<Real>(l2)*swl;
                        // A+ scatter of q to each component's edge
                        Real qx, qy, qz;
#if defined(WARPX_DIM_3D)
                        qx = 0.5_rt*(qq(i,j,k,0) + qq(i+1,j,k,0));
                        qy = 0.5_rt*(qq(i,j,k,1) + qq(i,j+1,k,1));
                        qz = 0.5_rt*(qq(i,j,k,2) + qq(i,j,k+1,2));
#elif defined(WARPX_DIM_XZ)
                        qx = 0.5_rt*(qq(i,j,k,0) + qq(i+1,j,k,0));
                        qy = qq(i,j,k,1);
                        qz = 0.5_rt*(qq(i,j,k,2) + qq(i,j+1,k,2));
#else
                        qx = qq(i,j,k,0);
                        qy = qq(i,j,k,1);
                        qz = 0.5_rt*(qq(i,j,k,2) + qq(i+1,j,k,2));
#endif
                        const Real dEx = (rE(i,j,k,0) - isW*aU*rW(i,j,k,0))*ialpha + qx;
                        const Real dEy = (rE(i,j,k,1) - isW*aU*rW(i,j,k,1))*ialpha + qy;
                        const Real dEz = (rE(i,j,k,2) - isW*aU*rW(i,j,k,2))*ialpha + qz;
                        xE(i,j,k,0) += sigma*dEx;
                        xE(i,j,k,1) += sigma*dEy;
                        xE(i,j,k,2) += sigma*dEz;
                        xW(i,j,k,0) += sigmaw*(rW(i,j,k,0) + l2s*dEx);
                        xW(i,j,k,1) += sigmaw*(rW(i,j,k,1) + l2s*dEy);
                        xW(i,j,k,2) += sigmaw*(rW(i,j,k,2) + l2s*dEz);
                    });
            }
            continue;
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
            Array4<const Real> const& bW = m_bW[a_lev]->const_array(mfi);
            Array4<const Real> const& bF = m_bF[a_lev]->const_array(mfi);
            Array4<const Real> const& Fc = m_tmp[a_lev]->const_array(mfi);
            // reads from the current iterate, writes into the ping-pong
            // partner (a pure Jacobi sweep; see m_xW2)
            Array4<const Real> const& xW = m_xW[a_lev]->const_array(mfi);
            Array4<const Real> const& xE = m_xF[a_lev]->const_array(mfi);
            Array4<Real> const& xWo = m_xW2[a_lev]->array(mfi);
            Array4<Real> const& xEo = m_xF2[a_lev]->array(mfi);
            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real rEx, rEy, rEz, rWx, rWy, rWz;
                    // current iterate; zero on the first sweep of a zero
                    // initial guess (the level's x is NOT pre-zeroed, see
                    // m_zero_free)
                    Real xEx = 0.0_rt, xEy = 0.0_rt, xEz = 0.0_rt;
                    Real xWx = 0.0_rt, xWy = 0.0_rt, xWz = 0.0_rt;
                    if (r_is_b) {
                        rEx = bF(i,j,k,0); rEy = bF(i,j,k,1); rEz = bF(i,j,k,2);
                        rWx = bW(i,j,k,0); rWy = bW(i,j,k,1); rWz = bW(i,j,k,2);
                    } else {
                        Real tcx, tcy, tcz, kkx, kky, kkz;
                        HybridPICPCPairTcKk(coef, xW, Fc, i, j, k, ionp, g,
                                            tcx, tcy, tcz, kkx, kky, kkz);
                        xEx = xE(i,j,k,0); xEy = xE(i,j,k,1); xEz = xE(i,j,k,2);
                        xWx = xW(i,j,k,0); xWy = xW(i,j,k,1); xWz = xW(i,j,k,2);
                        rEx = bF(i,j,k,0) - xEx - tcx;
                        rEy = bF(i,j,k,1) - xEy - tcy;
                        rEz = bF(i,j,k,2) - xEz - tcz;
                        rWx = bW(i,j,k,0) - xWx + kkx;
                        rWy = bW(i,j,k,1) - xWy + kky;
                        rWz = bW(i,j,k,2) - xWz + kkz;
                    }
                    const Real bhx = coef(i,j,k,0);
                    const Real bhy = coef(i,j,k,1);
                    const Real bhz = coef(i,j,k,2);
                    const Real aU = coef(i,j,k,4) + coef(i,j,k,5)*l2;
                    // rotation coefficient of the block: the whistler plus the
                    // Hall-folded ion response to the dissipative push-field
                    // coupling, S (U_diss W) ~ w_iso aU (W x bhat) (isotropic
                    // average of the deposited diagonal; the residuals carry
                    // the exact anisotropic form)
                    const Real cw = coef(i,j,k,3)
                        + (coef(i,j,k,6) + coef(i,j,k,7) + coef(i,j,k,8))
                          * (1.0_rt/3.0_rt) * aU;
                    // local W-hat scale (coef comp 14)
                    const Real swl = coef(i,j,k,14);
                    const Real isW = 1.0_rt/swl;
                    // s = rE - (1/sW) U_loc rW (the E-row couples to
                    // the scaled W-hat through U_loc/sW)
                    const Real sx = rEx - isW*(cw*(rWy*bhz - rWz*bhy) + aU*rWx);
                    const Real sy = rEy - isW*(cw*(rWz*bhx - rWx*bhz) + aU*rWy);
                    const Real sz = rEz - isW*(cw*(rWx*bhy - rWy*bhx) + aU*rWz);
                    // P^-1 s with the block defects: hyper-resistive at
                    // the Laplacian Nyquist symbol 2*l2, whistler at wdl2
                    const Real uem = std::sqrt(
                        coef(i,j,k,11)*coef(i,j,k,11)
                      + coef(i,j,k,12)*coef(i,j,k,12)
                      + coef(i,j,k,13)*coef(i,j,k,13));
#if defined(WARPX_DIM_RZ)
                    const bool on_axis = has_axis && (i == iax);
                    const Real l2n = 2.0_rt*static_cast<Real>(l2)
                                   + (on_axis ? 8.0_rt*g.invdx2[0] : 0.0_rt);
                    const Real l2h = static_cast<Real>(l2)
                                   + (on_axis ? 4.0_rt*g.invdx2[0] : 0.0_rt);
                    const Real wdl2n = wdf*l2n;
                    amrex::ignore_unused(wdl2);
#else
                    const Real l2n = static_cast<Real>(l2);
                    const Real l2h = 2.0_rt*static_cast<Real>(l2);
                    const Real wdl2n = wdl2;
#endif
                    // Hyper-resistive part of the block diagonal: the node's own
                    // coefficient times the Nyquist row sum of the (non-conservative)
                    // c_i del2 stencil, which is exactly c_i l2h whatever the
                    // neighbours' coefficients (a banded eta_H changes the block
                    // elsewhere, not here); the RZ axis z-row carries 8/dr^2.
                    const Real aUb = coef(i,j,k,4) + coef(i,j,k,5)*l2h;
                    const Real alpha = 1.0_rt + l1p*uem + l2n*aUb;
                    const Real beta = wdl2n*cw;
                    const Real sxb_x = sy*bhz - sz*bhy;
                    const Real sxb_y = sz*bhx - sx*bhz;
                    const Real sxb_z = sx*bhy - sy*bhx;
                    const Real sdb = sx*bhx + sy*bhy + sz*bhz;
                    const Real inv_a2b2 = 1.0_rt / (alpha*alpha + beta*beta);
                    const Real b2oa = beta*beta/alpha;
                    const Real dEx = inv_a2b2*(alpha*sx - beta*sxb_x + b2oa*sdb*bhx);
                    const Real dEy = inv_a2b2*(alpha*sy - beta*sxb_y + b2oa*sdb*bhy);
                    const Real dEz = inv_a2b2*(alpha*sz - beta*sxb_z + b2oa*sdb*bhz);
                    // EB-covered nodes (row mask 0) hold zero on every level
                    const Real mk = coef(i,j,k,15);
                    xEo(i,j,k,0) = mk*(xEx + sigma*dEx);
                    xEo(i,j,k,1) = mk*(xEy + sigma*dEy);
                    xEo(i,j,k,2) = mk*(xEz + sigma*dEz);
                    // perp-b part of dE couples through the whistler
                    // defect, the par-b part through the full l2
                    const Real dEb = dEx*bhx + dEy*bhy + dEz*bhz;
                    const Real dl = l2n - wdl2n;
                    xWo(i,j,k,0) = mk*(xWx + sigmaw*(rWx + swl*(wdl2n*dEx + dl*dEb*bhx)));
                    xWo(i,j,k,1) = mk*(xWy + sigmaw*(rWy + swl*(wdl2n*dEy + dl*dEb*bhy)));
                    xWo(i,j,k,2) = mk*(xWz + sigmaw*(rWz + swl*(wdl2n*dEz + dl*dEb*bhz)));
                });
        }
        // the partner now holds the new iterate (valid region); its
        // ghosts are refilled by the next FillGhosts before any read
        std::swap(m_xW[a_lev], m_xW2[a_lev]);
        std::swap(m_xF[a_lev], m_xF2[a_lev]);
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ResidualLevel ( int a_lev )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::ResidualLevel()");

    const bool stag = m_staggered;
    const bool ionp = m_have_mm;
    const HybridPICPCGrid g = m_grid[a_lev];

    FillGhosts(*m_xW[a_lev], a_lev);
    ApplyDomainClosure(*m_xW[a_lev], a_lev, ClosureClass::JField);
    FillGhosts(*m_xF[a_lev], a_lev);
    ApplyDomainClosure(*m_xF[a_lev], a_lev, ClosureClass::EField);
    StagePair(a_lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_rW[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
        Array4<const Real> const& bW = m_bW[a_lev]->const_array(mfi);
        Array4<const Real> const& bF = m_bF[a_lev]->const_array(mfi);
        Array4<const Real> const& xW = m_xW[a_lev]->const_array(mfi);
        Array4<const Real> const& xE = m_xF[a_lev]->const_array(mfi);
        Array4<const Real> const& Fc = m_tmp[a_lev]->const_array(mfi);
        Array4<Real> const& rW = m_rW[a_lev]->array(mfi);
        Array4<Real> const& rF = m_rF[a_lev]->array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real tcx, tcy, tcz, kkx, kky, kkz;
                if (stag) {
                    HybridPICPCPairTcKkStag(coef, xW, Fc, i, j, k, ionp, g,
                                            tcx, tcy, tcz, kkx, kky, kkz);
                } else {
                    HybridPICPCPairTcKk(coef, xW, Fc, i, j, k, ionp, g,
                                        tcx, tcy, tcz, kkx, kky, kkz);
                }
                rF(i,j,k,0) = bF(i,j,k,0) - xE(i,j,k,0) - tcx;
                rF(i,j,k,1) = bF(i,j,k,1) - xE(i,j,k,1) - tcy;
                rF(i,j,k,2) = bF(i,j,k,2) - xE(i,j,k,2) - tcz;
                rW(i,j,k,0) = bW(i,j,k,0) - xW(i,j,k,0) + kkx;
                rW(i,j,k,1) = bW(i,j,k,1) - xW(i,j,k,1) + kky;
                rW(i,j,k,2) = bW(i,j,k,2) - xW(i,j,k,2) + kkz;
            });
    }
}


template <class T, class Ops>
void HybridPICPC<T,Ops>::RestrictLevel ( int a_lev )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::RestrictLevel()");

    // nodal full weighting of the fine residual(s); fine ghosts are filled
    // (periodic/internal) or zero (physical: homogeneous Dirichlet). The
    // same index-space transfer serves the staggered components.
    auto restrict_fw = [&] (const MF& a_fine, MF& a_coarse)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(a_coarse, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<const Real> const& fr = a_fine.const_array(mfi);
            Array4<Real> const& cb = a_coarse.array(mfi);
            amrex::ParallelFor(bx, 3,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
#if defined(WARPX_DIM_3D)
                    Real acc = 0.0_rt;
                    for (int oz = -1; oz <= 1; oz++) {
                    for (int oy = -1; oy <= 1; oy++) {
                    for (int ox = -1; ox <= 1; ox++) {
                        const Real w = (ox == 0 ? 0.5_rt : 0.25_rt)
                                     * (oy == 0 ? 0.5_rt : 0.25_rt)
                                     * (oz == 0 ? 0.5_rt : 0.25_rt);
                        acc += w * fr(2*i+ox, 2*j+oy, 2*k+oz, n);
                    }}}
                    cb(i,j,k,n) = acc;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    Real acc = 0.0_rt;
                    for (int oz = -1; oz <= 1; oz++) {
                    for (int ox = -1; ox <= 1; ox++) {
                        const Real w = (ox == 0 ? 0.5_rt : 0.25_rt)
                                     * (oz == 0 ? 0.5_rt : 0.25_rt);
                        acc += w * fr(2*i+ox, 2*j+oz, k, n);
                    }}
                    cb(i,j,k,n) = acc;
#else // WARPX_DIM_1D_Z
                    cb(i,j,k,n) = 0.5_rt*fr(2*i,j,k,n)
                                + 0.25_rt*(fr(2*i-1,j,k,n) + fr(2*i+1,j,k,n));
#endif
                });
        }
    };

    restrict_fw(*m_rW[a_lev], *m_bW[a_lev+1]);
    restrict_fw(*m_rF[a_lev], *m_bF[a_lev+1]);
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ProlongLevel ( int a_lev )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::ProlongLevel()");

    // The nodal (tri)linear prolongation reads coarse nodes ic and ic+1 <=
    // N/2 only -- never a coarse ghost -- so the exchange is needed only on
    // the cell-indexed staggered container (fine cell N-1 reads coarse
    // cell N/2, the periodic image of cell 0). The closure still applies
    // the wall-plane rows of the coarse correction. Level-0 index space
    // starts at 0, so i/2 floors correctly.
    auto prolong_lin = [&] (const MF& a_coarse, MF& a_fine)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(a_fine, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<const Real> const& xc = a_coarse.const_array(mfi);
            Array4<Real> const& xf = a_fine.array(mfi);
            // fine-level row mask (coef comp 15): no correction lands on
            // EB-covered nodes
            Array4<const Real> const& mk = m_coef[a_lev]->const_array(mfi);
            amrex::ParallelFor(bx, 3,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
#if defined(WARPX_DIM_3D)
                    const int ic = i/2; const int fx = i - 2*ic;
                    const int jc = j/2; const int fy = j - 2*jc;
                    const int kc = k/2; const int fz = k - 2*kc;
                    Real acc = 0.0_rt;
                    for (int c = 0; c <= fz; c++) {
                    for (int b = 0; b <= fy; b++) {
                    for (int a = 0; a <= fx; a++) {
                        acc += xc(ic+a, jc+b, kc+c, n);
                    }}}
                    xf(i,j,k,n) += mk(i,j,k,15) * acc * (fx ? 0.5_rt : 1.0_rt)
                                       * (fy ? 0.5_rt : 1.0_rt)
                                       * (fz ? 0.5_rt : 1.0_rt);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    const int ic = i/2; const int fx = i - 2*ic;
                    const int jc = j/2; const int fy = j - 2*jc;
                    Real acc = 0.0_rt;
                    for (int b = 0; b <= fy; b++) {
                    for (int a = 0; a <= fx; a++) {
                        acc += xc(ic+a, jc+b, k, n);
                    }}
                    xf(i,j,k,n) += mk(i,j,k,15) * acc * (fx ? 0.5_rt : 1.0_rt)
                                       * (fy ? 0.5_rt : 1.0_rt);
#else // WARPX_DIM_1D_Z
                    const int ic = i/2; const int fx = i - 2*ic;
                    Real acc = 0.0_rt;
                    for (int a = 0; a <= fx; a++) { acc += xc(ic+a, j, k, n); }
                    xf(i,j,k,n) += mk(i,j,k,15) * acc * (fx ? 0.5_rt : 1.0_rt);
#endif
                });
        }
    };

    if (m_staggered) {
        FillGhosts(*m_xW[a_lev+1], a_lev+1);
        FillGhosts(*m_xF[a_lev+1], a_lev+1);
    }
    ApplyDomainClosure(*m_xW[a_lev+1], a_lev+1, ClosureClass::JField);
    ApplyDomainClosure(*m_xF[a_lev+1], a_lev+1, ClosureClass::EField);
    prolong_lin(*m_xW[a_lev+1], *m_xW[a_lev]);
    prolong_lin(*m_xF[a_lev+1], *m_xF[a_lev]);
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::Vcycle ( int a_lev, bool a_x_is_zero )
{
    if (a_lev == m_nlevels - 1) {
        const int nb = (m_nlevels == 1) ? kNuSingle : kNuBottom;
        SmoothLevel(a_lev, nb, a_x_is_zero);
        return;
    }

    SmoothLevel(a_lev, kNuPre, a_x_is_zero);

    ResidualLevel(a_lev);
    FillGhosts(*m_rW[a_lev], a_lev);
    ApplyDomainClosure(*m_rW[a_lev], a_lev, ClosureClass::JField);
    FillGhosts(*m_rF[a_lev], a_lev);
    ApplyDomainClosure(*m_rF[a_lev], a_lev, ClosureClass::EField);
    RestrictLevel(a_lev);
    if (!(m_zero_free && m_eb_flag == nullptr)) {
        m_xW[a_lev+1]->setVal(0.0);
        m_xF[a_lev+1]->setVal(0.0);
    }
    Vcycle(a_lev + 1, true);
    ProlongLevel(a_lev);

    SmoothLevel(a_lev, kNuPost, false);
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ApplyMToW ( const MF& a_blk, int a_scomp )
{
    // block input -> m_bW[0] + m_bF[0] in one kernel, then one V-cycle
    SplitBlock(a_blk, a_scomp);
    if (!(m_zero_free && m_eb_flag == nullptr)) {
        m_xW[0]->setVal(0.0);
        m_xF[0]->setVal(0.0);
    }
    Vcycle(0, true);
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ApplyReducedOp ( const MF& a_blk, MF& a_out )
{
    using namespace amrex;

    // a_blk must be ghost-filled (kNGrow layers) by the caller; the W-half
    // is comps 0..2, the E-half comps 3..5. out_W = W - kk ; out_E = xE +
    // tc, with F = curl xE staged from the block's E-half
    const int gU = kNGrow - 1;
    ComputeCurl(a_blk, *m_tmp[0], 0, gU, CurlKind::EtoB, 3);
    ApplyDomainClosure(*m_tmp[0], 0, ClosureClass::BField);
    const bool ionp = m_have_mm;
    const bool stag = m_staggered;
    const HybridPICPCGrid g = m_grid[0];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a_out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<const Real> const& coef = m_coef[0]->const_array(mfi);
        Array4<const Real> const& av = a_blk.const_array(mfi);
        Array4<const Real> const& zE = a_blk.const_array(mfi, 3);
        Array4<const Real> const& Fc = m_tmp[0]->const_array(mfi);
        Array4<Real> const& ow = a_out.array(mfi);
        Array4<Real> const& oe = a_out.array(mfi, 3);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real tcx, tcy, tcz, kkx, kky, kkz;
                if (stag) {
                    HybridPICPCPairTcKkStag(coef, av, Fc, i, j, k, ionp, g,
                                            tcx, tcy, tcz, kkx, kky, kkz);
                } else {
                    HybridPICPCPairTcKk(coef, av, Fc, i, j, k, ionp, g,
                                        tcx, tcy, tcz, kkx, kky, kkz);
                }
                ow(i,j,k,0) = av(i,j,k,0) - kkx;
                ow(i,j,k,1) = av(i,j,k,1) - kky;
                ow(i,j,k,2) = av(i,j,k,2) - kkz;
                oe(i,j,k,0) = zE(i,j,k,0) + tcx;
                oe(i,j,k,1) = zE(i,j,k,1) + tcy;
                oe(i,j,k,2) = zE(i,j,k,2) + tcz;
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::SplitBlock ( const MF& a_blk, int a_scomp )
{
    using namespace amrex;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_bW[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const& blk = a_blk.const_array(mfi, a_scomp);
        Array4<Real> const& bw = m_bW[0]->array(mfi);
        Array4<Real> const& bf = m_bF[0]->array(mfi);
        amrex::ParallelFor(bx, 3,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int c)
            {
                bw(i,j,k,c) = blk(i,j,k,c);
                bf(i,j,k,c) = blk(i,j,k,3+c);
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::MergeBlock ( MF& a_blk ) const
{
    using namespace amrex;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a_blk, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& blk = a_blk.array(mfi);
        Array4<Real const> const& xw = m_xW[0]->const_array(mfi);
        Array4<Real const> const& xf = m_xF[0]->const_array(mfi);
        amrex::ParallelFor(bx, 3,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int c)
            {
                blk(i,j,k,c) = xw(i,j,k,c);
                blk(i,j,k,3+c) = xf(i,j,k,c);
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::BlockAxpy ( MF& a_dst, int a_dcomp, const MF* a_src, int a_nv,
                                     const amrex::GpuArray<amrex::Real, kMaxKry>& a_c,
                                     amrex::Real a_scale ) const
{
    using namespace amrex;
    const int nv = a_nv;
    constexpr int nkb = kNKB;
    const Real scale = a_scale;
    const bool have_src = (a_src != nullptr);
    const auto c = a_c;
    // Aliasing contract: the source is read at comps 0..nkb-1 and the
    // destination written at a_dcomp..a_dcomp+nkb-1, so in-place use is safe
    // only when both ranges coincide (a_dcomp == 0) or are disjoint; writing
    // into the Krylov block must not overlap the basis vectors being read.
    AMREX_ALWAYS_ASSERT(a_src != &a_dst || a_dcomp == 0 || a_dcomp >= nkb);
    AMREX_ALWAYS_ASSERT(&a_dst != m_kryB.get() || a_dcomp >= nv * nkb);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a_dst, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& d = a_dst.array(mfi, a_dcomp);
        Array4<Real const> const& V = m_kryB->const_array(mfi);
        Array4<Real const> const s = have_src ? a_src->const_array(mfi)
                                              : Array4<Real const>{};
        amrex::ParallelFor(bx, nkb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int cc)
            {
                // in-place use (dst == src, a_dcomp == 0) is safe: each
                // thread then reads and writes the same element
                Real t = have_src ? s(i,j,k,cc) : 0.0_rt;
                for (int n = 0; n < nv; ++n) {
                    t -= c[n] * V(i,j,k, n*nkb + cc);
                }
                d(i,j,k,cc) = scale * t;
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::BatchDot ( const MF& a_w, int a_nv, bool a_with_ww, RT* a_out )
{
    using namespace amrex;
    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    constexpr int nkb = kNKB;
    const int nslots = a_nv + (a_with_ww ? 1 : 0);
    if (nslots == 0) { return; }
    // collocated grid: the nodal owner mask of Efield_fp (duplicated
    // periodic/box-shared nodes counted once); staggered containers are
    // cell-indexed and need no mask
    const bool use_mask = !m_staggered;
    const amrex::iMultiFab* mask = nullptr;
    if (use_mask) {
        mask = WarpX::GetInstance().getFieldDotMaskPointer(
            FieldType::Efield_fp, 0, Direction{0});
    }
    std::vector<amrex::Real> res(nslots, 0.0);

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    if (Gpu::inLaunchRegion()) {
        constexpr int NV = 8;   // basis vectors per register chunk
        const int nbt = m_dot_nblocks_total;
        amrex::Real* part = m_dot_dev.data();
        for (MFIter mfi(a_w, false); mfi.isValid(); ++mfi) {
            const Box bx = mfi.validbox();
            const int ib = mfi.LocalIndex();
            const int nblocks = m_dot_nblocks[ib];
            const int boff = m_dot_box_offset[ib];
            Array4<Real const> const w = a_w.const_array(mfi);
            Array4<Real const> const V = m_kryB->const_array(mfi);
            Array4<int const> const mk = use_mask ? mask->const_array(mfi)
                                                  : Array4<int const>{};
            const auto lo = amrex::lbound(bx);
            const auto len = amrex::length(bx);
            const Long npts = bx.numPts();
            const int nv = a_nv;
            const bool with_ww = a_with_ww;
            amrex::launch<kDotThreads>(nblocks, Gpu::gpuStream(),
                [=] AMREX_GPU_DEVICE () noexcept
                {
                    const int tid = static_cast<int>(threadIdx.x);
                    const Long stride = Long(blockDim.x) * Long(gridDim.x);
                    const Long start = Long(blockDim.x) * Long(blockIdx.x) + tid;
                    const Long lxy = Long(len.x) * Long(len.y);
                    // chunks of NV basis vectors; the <w,w> slot rides
                    // with the first chunk. Every thread executes every
                    // block reduction (uniform control flow).
                    const int nchunks = amrex::max(1, (nv + NV - 1) / NV);
                    for (int ch = 0; ch < nchunks; ++ch) {
                        const int n0 = ch * NV;
                        const bool do_ww = with_ww && (ch == 0);
                        Real s[NV];
                        for (int n = 0; n < NV; ++n) { s[n] = 0.0_rt; }
                        Real sww = 0.0_rt;
                        for (Long ic = start; ic < npts; ic += stride) {
                            const int k = static_cast<int>(ic / lxy);
                            const Long r = ic - Long(k) * lxy;
                            const int j = static_cast<int>(r / len.x);
                            const int i = static_cast<int>(r - Long(j) * len.x);
                            const int ii = i + lo.x;
                            const int jj = j + lo.y;
                            const int kk = k + lo.z;
                            if (use_mask && mk(ii,jj,kk) == 0) { continue; }
                            for (int c = 0; c < nkb; ++c) {
                                const Real wc = w(ii,jj,kk,c);
                                if (do_ww) { sww += wc * wc; }
                                for (int n = 0; n < NV; ++n) {
                                    if (n0 + n < nv) {
                                        s[n] += wc * V(ii,jj,kk,(n0 + n)*nkb + c);
                                    }
                                }
                            }
                        }
                        for (int n = 0; n < NV; ++n) {
                            const Real rs = Gpu::blockReduceSum<kDotThreads>(s[n]);
                            if (tid == 0 && n0 + n < nv) {
                                part[std::size_t(n0 + n) * nbt + boff + blockIdx.x] = rs;
                            }
                        }
                        if (do_ww) {
                            const Real rs = Gpu::blockReduceSum<kDotThreads>(sww);
                            if (tid == 0) {
                                part[std::size_t(nv) * nbt + boff + blockIdx.x] = rs;
                            }
                        }
                    }
                });
        }
        // the one host synchronization of the pass; fixed-order host sum
        const std::size_t ncopy = static_cast<std::size_t>(nslots) * nbt;
        Gpu::dtoh_memcpy(m_dot_host.data(), m_dot_dev.data(), ncopy * sizeof(amrex::Real));
        for (int n = 0; n < nslots; ++n) {
            amrex::Real ssum = 0.0_rt;
            for (int b = 0; b < nbt; ++b) {
                ssum += m_dot_host[static_cast<std::size_t>(n) * nbt + b];
            }
            res[n] = ssum;
        }
    } else
#endif
    {
        for (int n = 0; n < nslots; ++n) {
            const bool ww = (n == a_nv);
            const int vc = ww ? 0 : n * nkb;
            const MF& b = ww ? a_w : *m_kryB;
            res[n] = use_mask
                ? amrex::MultiFab::Dot(*mask, a_w, 0, b, vc, nkb, 0, true)
                : amrex::MultiFab::Dot(a_w, 0, b, vc, nkb, 0, true);
        }
    }
    if (amrex::ParallelContext::NProcsSub() > 1) {
        amrex::ParallelAllReduce::Sum(res.data(), nslots,
                                      amrex::ParallelContext::CommunicatorSub());
    }
    for (int n = 0; n < nslots; ++n) { a_out[n] = static_cast<RT>(res[n]); }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::RotateValid ( MF& a ) const
{
    using namespace amrex;
    // a <- (I+S)^-1 a pointwise (deposited-mass-matrix ion response, see
    // the class docs)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<const Real> const& coef = m_coef[0]->const_array(mfi);
        Array4<Real> const& v = a.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real vx = v(i,j,k,0);
                Real vy = v(i,j,k,1);
                Real vz = v(i,j,k,2);
                HybridPICPCInvIpS(coef(i,j,k,0), coef(i,j,k,1), coef(i,j,k,2),
                                  coef(i,j,k,6), coef(i,j,k,7), coef(i,j,k,8),
                                  vx, vy, vz);
                v(i,j,k,0) = vx;
                v(i,j,k,1) = vy;
                v(i,j,k,2) = vz;
            });
    }
}


template <class T, class Ops>
void HybridPICPC<T,Ops>::CopyCompIn ( const MF& a_src, MF& a_dst, int a_comp,
                                      bool a_add ) const
{
    using namespace amrex;
    // valid region of the container only; ghosts (including the top nodal
    // planes, which are ghost indices of the cell-indexed container) are
    // re-derived by FillBoundary and the closures
    const int nc = a_comp;
    const bool add = a_add;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a_dst, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<const Real> const& sv = a_src.const_array(mfi);
        Array4<Real> const& dv = a_dst.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                dv(i,j,k,nc) = (add ? dv(i,j,k,nc) : 0.0_rt) + sv(i,j,k);
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::WriteXOut ( const MF& a_src, T& a_x ) const
{
    using namespace amrex;
    for (int n = 0; n < 3; n++) {
        MF& x = *a_x.getArrayVec()[0][n];
        const int nc = n;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(x, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<const Real> const& sv = a_src.const_array(mfi);
            Array4<Real> const& xv = x.array(mfi);
            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    xv(i,j,k) = sv(i,j,k,nc);
                });
        }
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::FillGhosts ( MF& a, int a_lev ) const
{
    using namespace amrex;

    const amrex::Geometry& geom = m_geom[a_lev];
    if (!m_fastfill_ok[a_lev]) {
        a.FillBoundary(geom.periodicity());
        return;
    }

    // Single domain-spanning box: every ghost cell that lies inside the
    // periodic domain image is the wrap of a valid cell. Index arithmetic
    // per dimension d with the CELL domain length N_d:
    //   in range [lo, hi]           -> unchanged (hi = lo+N nodal, lo+N-1 cell)
    //   out of range, periodic      -> lo + ((idx-lo) mod N)   in [lo, lo+N)
    //   out of range, not periodic  -> not written (closure territory)
    // A duplicated nodal plane index lo+N is mapped to lo so every ghost
    // derives from the primary plane. The valid region is never written.
    const int ncomp = a.nComp();
    GpuArray<int,3> per{0, 0, 0};
    GpuArray<int,3> len{1, 1, 1};
    for (int d = 0; d < AMREX_SPACEDIM; d++) {
        per[d] = geom.isPeriodic(d) ? 1 : 0;
        len[d] = geom.Domain().length(d);
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box vbx = mfi.validbox();
        // the grown tile (== the whole fab box without tiling, i.e. on GPU)
        const Box gbx = mfi.growntilebox();
        GpuArray<int,3> lo{0, 0, 0};
        GpuArray<int,3> hi{0, 0, 0};
        for (int d = 0; d < AMREX_SPACEDIM; d++) {
            lo[d] = vbx.smallEnd(d);
            hi[d] = vbx.bigEnd(d);
        }
        Array4<Real> const& v = a.array(mfi);
        amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                int s[3] = {i, j, k};
                bool ghost = false;
                for (int d = 0; d < AMREX_SPACEDIM; d++) {
                    if (s[d] < lo[d] || s[d] > hi[d]) {
                        if (!per[d]) { return; }
                        ghost = true;
                        int r = (s[d] - lo[d]) % len[d];
                        if (r < 0) { r += len[d]; }
                        s[d] = lo[d] + r;
                    } else if (per[d] && s[d] == lo[d] + len[d]) {
                        s[d] = lo[d];   // duplicated nodal plane -> primary
                    }
                }
                if (!ghost) { return; }
                for (int n = 0; n < ncomp; n++) {
                    v(i,j,k,n) = v(s[0],s[1],s[2],n);
                }
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ApplyDomainClosure ( MF& a, int a_lev,
                                              ClosureClass a_cls,
                                              int a_scomp, int a_ncomp ) const
{
    using namespace amrex;

    // component range: the closure logic indexes components RELATIVE to
    // a_scomp (so a 6-comp Krylov block gets its W-half and E-half closed
    // with their own classes by two calls), the storage by a_scomp + n
    const int ncl = (a_ncomp < 0) ? a.nComp() : a_ncomp;
    const int sc = a_scomp;

    if (m_has_wall) {
        const Box dom = amrex::convert(m_geom[a_lev].Domain(),
                                       a.boxArray().ixType());
        for (int d = 0; d < AMREX_SPACEDIM; d++) {
            for (int side = 0; side < 2; side++) {
                if (!m_wall_pec[d][side] && !m_wall_pmc[d][side]) { continue; }
                // wall NODE index: the staggered containers are
                // cell-indexed, so the hi-side wall plane sits one past
                // the cell bigEnd
                const int w = (side == 0)
                    ? dom.smallEnd(d)
                    : dom.bigEnd(d) + (m_staggered ? 1 : 0);
                const int nc = m_dim2comp[d];
                const bool lo_side = (side == 0);
                const int cls = static_cast<int>(a_cls);
                const int dd = d;
                const bool stag = m_staggered;
                // PMC (Neumann): the residual applies the dual parity rules,
                // so the normal/tangential roles swap for every vector class
                const bool pmc = m_wall_pmc[d][side];
                // PEC wall node whose normal E the residual takes from the interior:
                // its column is dead in the preconditioner too
                const bool pec_normal_dead = m_pec_normal_from_interior && m_wall_pec[d][side];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    const Box& bx = mfi.growntilebox();
                    // component window: relative index n, storage sc + n
                    Array4<Real> const& v = a.array(mfi, sc);
                    amrex::ParallelFor(bx, ncl,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                        {
                            const int id = (dd == 0) ? i : ((dd == 1) ? j : k);
                            const bool outside =
                                lo_side ? (id < w) : (id > w);
                            // Coef: comps 0-2 carry bhat (vector), the rest
                            // are scalars
                            const bool scalar =
                                (cls == 4/*Coef*/) && (n >= 3);
                            // effective "normal" role: swapped at PMC faces
                            const bool is_norm = !scalar && ((n == nc) != pmc);
                            if (stag) {
                                // Yee: components NODAL along dd (wall-
                                // tangential E/J, wall-normal B) carry a
                                // wall-plane DOF that is zero at PEC (dead
                                // tangential-E columns, zero normal B, and
                                // zero mirror-image tangential J) and
                                // mirror ODD about the wall node;
                                // components CELL-centered along dd have no
                                // wall DOF and mirror EVEN about the wall
                                // plane. The node-built coefficients keep a
                                // smooth even cell-mirror (their ghost ring
                                // only shapes U there).
                                bool nodal_d;
                                Real sgn;
                                if (cls == 0/*EField*/ || cls == 2/*JField*/) {
                                    nodal_d = !is_norm;
                                    sgn = is_norm ? 1.0_rt : -1.0_rt;
                                } else if (cls == 4/*Coef*/) {
                                    nodal_d = false;
                                    sgn = 1.0_rt;
                                } else { // BField, BCoef
                                    nodal_d = is_norm;
                                    sgn = is_norm ? -1.0_rt : 1.0_rt;
                                }
                                int im;
                                if (nodal_d) {
                                    const bool beyond =
                                        lo_side ? (id < w) : (id > w);
                                    if (!beyond) {
                                        if (id == w) { v(i,j,k,n) = 0.0_rt; }
                                        return;
                                    }
                                    im = 2*w - id;
                                } else {
                                    const bool beyond =
                                        lo_side ? (id < w) : (id >= w);
                                    if (!beyond) { return; }
                                    im = 2*w - 1 - id;
                                }
                                const int mi = (dd == 0) ? im : i;
                                const int mj = (dd == 1) ? im : j;
                                const int mk = (dd == 2) ? im : k;
                                v(i,j,k,n) = sgn * v(mi,mj,mk,n);
                                return;
                            }
                            if (!outside) {
                                if (id != w) { return; }
                                // wall plane: EField zeroes tangential
                                // (identity rows / dead columns), BField
                                // zeroes the normal component
                                if (cls == 0/*EField*/ && !scalar
                                    && (!is_norm || pec_normal_dead)) { v(i,j,k,n) = 0.0_rt; }
                                if (cls == 1/*BField*/ && is_norm) {
                                    v(i,j,k,n) = 0.0_rt;
                                }
                                return;
                            }
                            // ghost planes: signed mirror about the wall node
                            const int im = 2*w - id;
                            Real sign = 1.0_rt;
                            if (!scalar) {
                                if (cls == 0/*EField*/ || cls == 2/*JField*/) {
                                    sign = is_norm ? 1.0_rt : -1.0_rt;
                                } else { // BField, BCoef, Coef bhat comps
                                    sign = is_norm ? -1.0_rt : 1.0_rt;
                                }
                            }
                            const int mi = (dd == 0) ? im : i;
                            const int mj = (dd == 1) ? im : j;
                            const int mk = (dd == 2) ? im : k;
                            v(i,j,k,n) = sign * v(mi,mj,mk,n);
                        });
                }
            }
        }
    }

#if defined(WARPX_DIM_RZ)
    // Symmetry axis (r = 0): signed-mirror guards about the axis node --
    // r/theta components odd, z even, the same parities for E-, B- and
    // J-like objects at m = 0 (matching ApplyFieldBoundaryOnAxis). The Ohm
    // solve zeroes E_theta ON the axis, so EField-class objects get that
    // row zeroed here as well (restored as an identity row on the output).
    if (m_has_axis) {
        const Box adom = amrex::convert(m_geom[a_lev].Domain(),
                                        a.boxArray().ixType());
        const int w = adom.smallEnd(0);
        const int cls = static_cast<int>(a_cls);
        const bool stag = m_staggered;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.growntilebox();
            Array4<Real> const& v = a.array(mfi, sc);
            amrex::ParallelFor(bx, ncl,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    if (i > w) { return; }
                    const bool scalar = (cls == 4/*Coef*/) && (n >= 3);
                    // Yee: components CELL-centered in r (the E/J r
                    // component, the B theta/z components; never the
                    // node-built coefficients) have no axis DOF and mirror
                    // with the residual's ishift = 1
                    bool cell_r = false;
                    if (stag && cls != 4/*Coef*/) {
                        cell_r = (cls == 0 || cls == 2) ? (n == 0)
                                                        : (n != 0);
                    }
                    if (i == w) {
                        // the Ohm solve zeroes E_theta on the axis on both
                        // grids and E_r there on the collocated grid (odd
                        // component with an axis DOF)
                        if (cls == 0/*EField*/ && (n == 1 || (n == 0 && !stag))) {
                            v(i,j,k,n) = 0.0_rt;
                        }
                        return;
                    }
                    const int im = cell_r ? (2*w - 1 - i) : (2*w - i);
                    const Real sign =
                        (scalar || n == 2) ? 1.0_rt : -1.0_rt;
                    v(i,j,k,n) = sign * v(im,j,k,n);
                });
        }
    }
#endif

    // EB (staircase): zero covered/frozen nodes of the E-, J- and
    // coefficient-like objects (the residual's masked Ohm and Ampere kernels
    // skip those locations; the coefficient row mask comp 15 becomes 0 there
    // and makes the covered rows identity on every level). The first-curl
    // delta-B (BField) is NOT zeroed: the residual's Faraday update is not
    // EB-masked, so the covered delta-B, built from the uncovered E
    // neighbours, is read by the wall-adjacent Ampere stencils. Fine level
    // only; masked up to the flag's guards. Exception: the pair's local
    // W-hat scale (Coef comp 14) is a strictly positive SCALING, divided by
    // in the joint block -- zeroing it at covered nodes makes 1/sW = inf
    // there (NaN poisoning that the level-0-only closure cannot clean off
    // the coarse levels).
    if (a_lev == 0 && m_eb_flag != nullptr && a_cls != ClosureClass::BField) {
        amrex::IntVect ngm = a.nGrowVect();
        ngm.min(m_eb_flag->nGrowVect());
        const bool is_coef = (a_cls == ClosureClass::Coef);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.growntilebox(ngm);
            Array4<Real> const& v = a.array(mfi, sc);
            Array4<const int> const& fl = m_eb_flag->const_array(mfi);
            amrex::ParallelFor(bx, ncl,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    if (is_coef && n == 14) { return; }
                    if (fl(i,j,k) == 0) { v(i,j,k,n) = 0.0_rt; }
                });
        }
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::OverrideIdentityRows ( T& a_out, const T& a_in ) const
{
    using namespace amrex;

    for (int d = 0; d < AMREX_SPACEDIM; d++) {
        for (int side = 0; side < 2; side++) {
            // PEC faces: exact identity tangential rows (both grids); PMC
            // faces: exact identity normal row (the residual zeroes the
            // normal E on the wall). Staggered hi-side zero-extension
            // faces: the nodal wall plane is an unmodeled ghost plane of
            // the cell-indexed container -- the preconditioner would
            // return zero there (a singular preconditioner); make those
            // rows identity.
            const bool open_stag = m_staggered && (side == 1)
                && m_wall_open[d][side];
            const bool pmc = m_wall_pmc[d][side];
            if (!m_wall_pec[d][side] && !pmc && !open_stag) { continue; }
            const int nc = m_dim2comp[d];
            // PEC wall node whose normal E the residual takes from the interior
            // (hybrid_pic_model.pec_normal_E_from_interior): identity row as well
            const bool pec_normal_identity = m_pec_normal_from_interior && m_wall_pec[d][side];
            for (int n = 0; n < 3; n++) {
                // tangential rows at PEC / open faces, the normal row at PMC (and at
                // PEC with the interior normal-E closure)
                const bool normal_row = (n == nc);
                const bool take = pmc ? normal_row : (!normal_row || pec_normal_identity);
                if (!take) { continue; }
                MF& o = *a_out.getArrayVec()[0][n];
                const MF& s = *a_in.getArrayVec()[0][n];
                // wall plane in this component's own staggering (the
                // tangential components are nodal along d on both grids)
                const Box dom = amrex::convert(m_geom[0].Domain(),
                                               o.boxArray().ixType());
                const int w = (side == 0) ? dom.smallEnd(d) : dom.bigEnd(d);
                Box plane = dom;
                plane.setSmall(d, w);
                plane.setBig(d, w);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(o, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    const Box bx = mfi.tilebox() & plane;
                    if (bx.isEmpty()) { continue; }
                    Array4<Real> const& ov = o.array(mfi);
                    Array4<const Real> const& sv = s.const_array(mfi);
                    amrex::ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        { ov(i,j,k) = sv(i,j,k); });
                }
            }
        }
    }

#if defined(WARPX_DIM_RZ)
    // E_theta on the axis (both grids) and E_r on the axis (collocated grid,
    // where E_r has an axis DOF) are fixed to zero by the Ohm solve: identity rows
    if (m_has_axis) {
        for (int c = 0; c < 2; ++c) {
            if (c == 0 && m_staggered) { continue; }
            const Box adom = amrex::convert(m_geom[0].Domain(),
                a_out.getArrayVec()[0][c]->boxArray().ixType());
            Box plane = adom;
            plane.setBig(0, adom.smallEnd(0));
            MF& o = *a_out.getArrayVec()[0][c];
            const MF& si = *a_in.getArrayVec()[0][c];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(o, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box bx = mfi.tilebox() & plane;
                if (bx.isEmpty()) { continue; }
                Array4<Real> const& ov = o.array(mfi);
                Array4<const Real> const& sv = si.const_array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    { ov(i,j,k) = sv(i,j,k); });
            }
        }
    }
#endif

    if (m_eb_flag != nullptr) {
        for (int n = 0; n < 3; n++) {
            MF& o = *a_out.getArrayVec()[0][n];
            const MF& s = *a_in.getArrayVec()[0][n];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(o, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                Array4<Real> const& ov = o.array(mfi);
                Array4<const Real> const& sv = s.const_array(mfi);
                Array4<const int> const& fl = m_eb_flag->const_array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (fl(i,j,k) == 0) { ov(i,j,k) = sv(i,j,k); }
                    });
            }
        }
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::InnerGMRES ()
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::InnerGMRES()");

    const int m = m_inner_max;
    constexpr int nkb = kNKB;
    // the s_W-weighted block norm is E-space accurate: inner_rtol applies
    // directly
    const RT rtol = m_inner_rtol;

    // v_0 = b/||b|| (the RHS block was assembled by Apply() into m_bKB)
    amrex::MultiFab::Copy(*m_kryB, *m_bKB, 0, 0, nkb, 0);
    m_xW[0]->setVal(0.0);
    m_xF[0]->setVal(0.0);
    std::vector<RT> d1(m + 2, RT(0.0));
    std::vector<RT> d2(m + 2, RT(0.0));
    BatchDot(*m_kryB, 0, true, d1.data());
    const RT beta = std::sqrt(d1[0]);
    if (beta <= RT(0.0)) { return; }
    m_kryB->mult(RT(1.0)/beta, 0, nkb, 0);

    std::vector<std::vector<RT>> H(m + 1, std::vector<RT>(m, RT(0.0)));
    std::vector<RT> cs(m, RT(0.0));
    std::vector<RT> sn(m, RT(0.0));
    std::vector<RT> g(m + 1, RT(0.0));
    g[0] = beta;
    amrex::GpuArray<amrex::Real, kMaxKry> cb{};

    int its = 0;
    for (int j = 0; j < m; j++) {
        its = j + 1;

        // z = M v_j (one V-cycle; result merged into the ghost-filled block
        // m_zB), w = A z
        ApplyMToW(*m_kryB, j*nkb);
        MergeBlock(*m_zB);
        FillGhosts(*m_zB, 0);
        ApplyDomainClosure(*m_zB, 0, ClosureClass::JField, 0, 3);
        ApplyDomainClosure(*m_zB, 0, ClosureClass::EField, 3, 3);
        ApplyReducedOp(*m_zB, *m_wB);

        // Classical Gram-Schmidt with reorthogonalization (CGS2):
        //   pass 1: d1 = V^T w,        w1 = w  - V d1
        //   pass 2: d2 = V^T w1,       w2 = w1 - V d2,  H_j = d1 + d2,
        //   h_{j+1,j} = ||w2|| = sqrt(||w1||^2 - ||d2||^2)  (Pythagoras; an
        //   explicit norm is taken when the second projection is not tiny)
        // Two batched reductions (two host syncs) per iteration instead of
        // the j+2 sequential ones of modified Gram-Schmidt; identical in
        // exact arithmetic.
        BatchDot(*m_wB, j + 1, false, d1.data());
        for (int i = 0; i <= j; i++) { cb[i] = static_cast<amrex::Real>(d1[i]); }
        BlockAxpy(*m_wB, 0, m_wB.get(), j + 1, cb, 1.0_rt);
        BatchDot(*m_wB, j + 1, true, d2.data());
        RT d2sq = RT(0.0);
        for (int i = 0; i <= j; i++) {
            H[i][j] = d1[i] + d2[i];
            d2sq += d2[i]*d2[i];
        }
        RT h2 = d2[j+1] - d2sq;
        const bool explicit_norm = (d2sq > RT(1.0e-8) * d2[j+1]) || (h2 <= RT(0.0));
        if (explicit_norm) {
            // near-breakdown regime: form w2 and measure it directly
            for (int i = 0; i <= j; i++) { cb[i] = static_cast<amrex::Real>(d2[i]); }
            BlockAxpy(*m_wB, 0, m_wB.get(), j + 1, cb, 1.0_rt);
            RT ww[1] = {RT(0.0)};
            BatchDot(*m_wB, 0, true, ww);
            h2 = ww[0];
        }
        const RT hj1 = std::sqrt(amrex::max(h2, RT(0.0)));

        for (int i = 0; i < j; i++) {
            const RT t = cs[i]*H[i][j] + sn[i]*H[i+1][j];
            H[i+1][j] = -sn[i]*H[i][j] + cs[i]*H[i+1][j];
            H[i][j] = t;
        }
        const RT denom = std::sqrt(H[j][j]*H[j][j] + hj1*hj1);
        cs[j] = H[j][j] / denom;
        sn[j] = hj1 / denom;
        H[j][j] = denom;
        g[j+1] = -sn[j]*g[j];
        g[j] = cs[j]*g[j];

        const bool converged = (std::abs(g[j+1]) <= rtol * beta);
        const bool breakdown = (hj1 <= RT(1.0e-14) * beta);
        if (converged || breakdown || j + 1 == m) { break; }

        // v_{j+1} = w2 / h_{j+1,j}, fused with the second projection when
        // w2 was not materialized above
        if (explicit_norm) {
            BlockAxpy(*m_kryB, (j+1)*nkb, m_wB.get(), 0, cb, 1.0_rt/static_cast<amrex::Real>(hj1));
        } else {
            for (int i = 0; i <= j; i++) { cb[i] = static_cast<amrex::Real>(d2[i]); }
            BlockAxpy(*m_kryB, (j+1)*nkb, m_wB.get(), j + 1, cb,
                      1.0_rt/static_cast<amrex::Real>(hj1));
        }
    }

    std::vector<RT> y(its, RT(0.0));
    for (int i = its - 1; i >= 0; i--) {
        RT s = g[i];
        for (int k = i + 1; k < its; k++) { s -= H[i][k]*y[k]; }
        y[i] = s / H[i][i];
    }

    // x = M (V y): the V-cycle is linear, so apply it once to the summed
    // basis (classical right-preconditioned GMRES update); one fused
    // linear-combination kernel forms V y (BlockAxpy with no source term
    // computes -sum c_i v_i, hence the sign)
    for (int i = 0; i < its; i++) { cb[i] = -static_cast<amrex::Real>(y[i]); }
    BlockAxpy(*m_zB, 0, nullptr, its, cb, 1.0_rt);
    ApplyMToW(*m_zB, 0);   // final W and E in m_xW[0]/m_xF[0]

    if (m_verbose) {
        amrex::Print() << "  pc_hybrid_pic inner GMRES: " << its << " its, rel res "
                       << std::abs(g[its])/beta << "\n";
    }
    if (m_verbose > 1) { PrintInnerResidualMap(); }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::PrintInnerResidualMap ()
{
    using namespace amrex;
    constexpr int nkb = kNKB;
    // Diagnostic (verbose >= 2): where the un-reduced inner residual
    // r = b - M x lives. Share of ||r_E||^2 and of the RHS ||b_E||^2 per
    // block of an 8 x 8 partition of the nodal domain (rows: dim-1 blocks
    // lo -> hi, cols: dim-0 blocks lo -> hi), the W/E split, and the share
    // of ||r_E||^2 on the four domain-boundary planes.
    MergeBlock(*m_zB);
    FillGhosts(*m_zB, 0);
    ApplyDomainClosure(*m_zB, 0, ClosureClass::JField, 0, 3);
    ApplyDomainClosure(*m_zB, 0, ClosureClass::EField, 3, 3);
    ApplyReducedOp(*m_zB, *m_wB);
    amrex::MultiFab::Xpay(*m_wB, -1.0_rt, *m_bKB, 0, 0, nkb, 0);   // r = b - M x
    const Box ndom = amrex::convert(m_geom[0].Domain(), amrex::IntVect::TheNodeVector());
    const auto dlo = ndom.smallEnd();
    const auto dhi = ndom.bigEnd();
    constexpr int NB = 8;
    constexpr int NACC = 2*NB*NB + 7;
    amrex::Gpu::DeviceVector<amrex::Real> acc(NACC, 0.0_rt);
    amrex::Real* pa = acc.data();
    const int n0 = dhi[0] - dlo[0] + 1;
#if (AMREX_SPACEDIM > 1)
    const int n1 = dhi[1] - dlo[1] + 1;
#endif
    for (MFIter mfi(*m_wB, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<const Real> const& r = m_wB->const_array(mfi);
        Array4<const Real> const& b = m_bKB->const_array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const Real rE = r(i,j,k,3)*r(i,j,k,3) + r(i,j,k,4)*r(i,j,k,4)
                              + r(i,j,k,5)*r(i,j,k,5);
                const Real rW = r(i,j,k,0)*r(i,j,k,0) + r(i,j,k,1)*r(i,j,k,1)
                              + r(i,j,k,2)*r(i,j,k,2);
                const Real bE = b(i,j,k,3)*b(i,j,k,3) + b(i,j,k,4)*b(i,j,k,4)
                              + b(i,j,k,5)*b(i,j,k,5);
                const int b0 = amrex::min(NB - 1, ((i - dlo[0]) * NB) / n0);
#if (AMREX_SPACEDIM > 1)
                const int b1 = amrex::min(NB - 1, ((j - dlo[1]) * NB) / n1);
#else
                const int b1 = 0;
                amrex::ignore_unused(j);
#endif
                amrex::ignore_unused(k);
                amrex::HostDevice::Atomic::Add(pa + b1*NB + b0, rE);
                amrex::HostDevice::Atomic::Add(pa + NB*NB + b1*NB + b0, bE);
                amrex::HostDevice::Atomic::Add(pa + 2*NB*NB + 0, rW);
                amrex::HostDevice::Atomic::Add(pa + 2*NB*NB + 1, rE);
                amrex::HostDevice::Atomic::Add(pa + 2*NB*NB + 2, bE);
                if (i == dlo[0]) { amrex::HostDevice::Atomic::Add(pa + 2*NB*NB + 3, rE); }
                if (i == dhi[0]) { amrex::HostDevice::Atomic::Add(pa + 2*NB*NB + 4, rE); }
#if (AMREX_SPACEDIM > 1)
                if (j == dlo[1]) { amrex::HostDevice::Atomic::Add(pa + 2*NB*NB + 5, rE); }
                if (j == dhi[1]) { amrex::HostDevice::Atomic::Add(pa + 2*NB*NB + 6, rE); }
#endif
            });
    }
    amrex::Gpu::streamSynchronize();
    std::vector<amrex::Real> h(NACC, 0.0_rt);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, acc.begin(), acc.end(), h.begin());
    amrex::ParallelDescriptor::ReduceRealSum(h.data(), NACC);
    const amrex::Real rW2 = h[2*NB*NB], rE2 = h[2*NB*NB + 1], bE2 = h[2*NB*NB + 2];
    const amrex::Real irE = (rE2 > 0.0_rt) ? 1.0_rt/rE2 : 0.0_rt;
    const amrex::Real ibE = (bE2 > 0.0_rt) ? 1.0_rt/bE2 : 0.0_rt;
    amrex::Print() << "  pc_hybrid_pic inner residual: ||r_E||/||b_E|| "
                   << std::sqrt(rE2*ibE) << ", ||r_W||/||r_E|| "
                   << std::sqrt(rW2*irE) << "; r_E share on planes lo0 "
                   << h[2*NB*NB + 3]*irE << " hi0 " << h[2*NB*NB + 4]*irE
                   << " lo1 " << h[2*NB*NB + 5]*irE << " hi1 " << h[2*NB*NB + 6]*irE
                   << "\n  block shares (dim-1 rows lo->hi | dim-0 cols lo->hi):"
                   << "  r_E  ||  b_E\n";
    for (int b1 = 0; b1 < NB; ++b1) {
        amrex::Print() << "   ";
        for (int b0 = 0; b0 < NB; ++b0) {
            amrex::Print() << " " << std::fixed << std::setprecision(3)
                           << h[b1*NB + b0]*irE;
        }
        amrex::Print() << "  ||";
        for (int b0 = 0; b0 < NB; ++b0) {
            amrex::Print() << " " << std::fixed << std::setprecision(3)
                           << h[NB*NB + b1*NB + b0]*ibE;
        }
        amrex::Print() << std::defaultfloat << "\n";
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::Apply ( T& a_x, const T& a_b )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::Apply()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(), "HybridPICPC::Apply() called on undefined object");

    // pe-as-Newton-unknown triangular leg (see m_pe_row_active): solve the
    // pe row with the pointwise diagonal D_p, then fold its Ohm gradient
    // coupling into the E-row RHS below (block upper-triangular approximate
    // Jacobian; the reverse pe <- E coupling is dropped). With the leg
    // inactive, or an E-only vector, the pe row falls through to the
    // identity passthrough at the end of this function.
    const bool pe_row = m_pe_row_active
        && (a_x.getScalarVecType() != warpx::fields::FieldType::None);
    if (pe_row) {
        amrex::MultiFab& xp = *a_x.getScalarVec()[0];
        const amrex::MultiFab& bp = *a_b.getScalarVec()[0];
        // the scheme scales the pe Newton row; the E leg needs physical dpe
        const amrex::Real inv_kappa =
            1.0_rt / m_ops->GetHybridPICModel()->m_pe_newton_scale;
        m_pe_row_scr->setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(xp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box vbx = mfi.tilebox();
            auto const& x  = xp.array(mfi);
            auto const& b  = bp.const_array(mfi);
            auto const& cf = m_coef[0]->const_array(mfi);
            auto const& s  = m_pe_row_scr->array(mfi);
            amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const Real v = b(i,j,k) / (1.0_rt + cf(i,j,k,10));
                    x(i,j,k) = v;
                    s(i,j,k) = v * inv_kappa;   // physical dpe for the E leg
                });
        }
        if (m_pe_row_sweeps > 0) {
            // damped-Jacobi sweeps carrying the signed UPWIND advection
            // stencil theta*dt*gamma ue0.grad(.) (comps 11-13 hold
            // theta*dt ue0), whose diagonal is exactly the advective part
            // of D_p = 1 + cf10, PLUS the conduction diffusion stencil
            // -c_k del2(.) whose Nyquist diagonal is exactly the conduction
            // part of D_p (c_k = (cf10 - adiag)/l2, so the sweep operator's
            // diagonal stays 1 + cf10): the off-diagonal content the
            // pointwise leg drops for both legs. Domain ghosts zero-extend.
            const amrex::Real gam = m_ops->GetHybridPICModel()->m_gamma;
            const amrex::Real l2_pe = static_cast<amrex::Real>(m_l2[0]);
            const auto dxi = m_geom[0].InvCellSizeArray();
            // marker-CFL clamp in the units of comps 11-13 (theta*dt ue):
            // c = pe_ue_cap_fac * dx_min; <= 0 = no clamp
            amrex::Real dx_min_pe = m_geom[0].CellSize(0);
            for (int d = 1; d < AMREX_SPACEDIM; ++d) {
                dx_min_pe = std::min(dx_min_pe, m_geom[0].CellSize(d));
            }
            const amrex::Real capl = m_pe_ue_cap_fac * dx_min_pe;
#if defined(WARPX_DIM_RZ)
            // RZ m = 0: area-weighted radial Laplacian neighbours, even
            // parity across the axis (u_r = 0 there by symmetry)
            const bool sw_axis = m_has_axis;
            const int sw_iax = m_geom[0].Domain().smallEnd(0);
            const HybridPICPCGrid gsw = m_grid[0];
#endif
            amrex::MultiFab& x2 = *m_pe_row_scr2;
            amrex::MultiFab::Copy(x2, xp, 0, 0, 1, 0);
            for (int sw = 0; sw < m_pe_row_sweeps; sw++) {
                FillGhosts(x2, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(xp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    const Box vbx = mfi.tilebox();
                    auto const& xo = xp.array(mfi);
                    auto const& xi = x2.const_array(mfi);
                    auto const& b  = bp.const_array(mfi);
                    auto const& cf = m_coef[0]->const_array(mfi);
                    amrex::ParallelFor(vbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        {
                            auto clamp_u = [=] (Real u) {
                                return (capl > 0.0_rt) ? capl * std::tanh(u / capl) : u;
                            };
                            // gamma * upwind(theta*dt ue . grad x) and the
                            // advective diagonal of D_p it contains
                            Real up = 0.0_rt;
                            Real adiag = 0.0_rt;
                            Real nbr = 0.0_rt;   // Laplacian neighbour sum
#if defined(WARPX_DIM_1D_Z)
                            const Real uz = clamp_u(cf(i,j,k,13));
                            up += gam * dxi[0] * (uz > 0.0_rt
                                ? uz*(xi(i,j,k) - xi(i-1,j,k))
                                : uz*(xi(i+1,j,k) - xi(i,j,k)));
                            adiag = gam * std::abs(uz) * dxi[0];
                            nbr = (xi(i+1,j,k) + xi(i-1,j,k))*dxi[0]*dxi[0];
#elif defined(WARPX_DIM_XZ)
                            const Real ux = clamp_u(cf(i,j,k,11));
                            const Real uz = clamp_u(cf(i,j,k,13));
                            up += gam * dxi[0] * (ux > 0.0_rt
                                ? ux*(xi(i,j,k) - xi(i-1,j,k))
                                : ux*(xi(i+1,j,k) - xi(i,j,k)));
                            up += gam * dxi[1] * (uz > 0.0_rt
                                ? uz*(xi(i,j,k) - xi(i,j-1,k))
                                : uz*(xi(i,j+1,k) - xi(i,j,k)));
                            adiag = gam * (std::abs(ux) * dxi[0] + std::abs(uz) * dxi[1]);
                            nbr = (xi(i+1,j,k) + xi(i-1,j,k))*dxi[0]*dxi[0]
                                + (xi(i,j+1,k) + xi(i,j-1,k))*dxi[1]*dxi[1];
#elif defined(WARPX_DIM_RZ)
                            const bool on_ax = sw_axis && (i == sw_iax);
                            const Real ur = on_ax ? 0.0_rt : clamp_u(cf(i,j,k,11));
                            const Real uz = clamp_u(cf(i,j,k,13));
                            up += gam * dxi[0] * (ur > 0.0_rt
                                ? ur*(xi(i,j,k) - xi(i-1,j,k))
                                : ur*(xi(i+1,j,k) - xi(i,j,k)));
                            up += gam * dxi[1] * (uz > 0.0_rt
                                ? uz*(xi(i,j,k) - xi(i,j-1,k))
                                : uz*(xi(i,j+1,k) - xi(i,j,k)));
                            adiag = gam * (std::abs(ur) * dxi[0] + std::abs(uz) * dxi[1]);
                            if (on_ax) {
                                nbr = 2.0_rt*xi(i+1,j,k)*dxi[0]*dxi[0];
                            } else {
                                const Real rn = gsw.rmin + i*gsw.dr;
                                nbr = ((rn + 0.5_rt*gsw.dr)*xi(i+1,j,k)
                                     + (rn - 0.5_rt*gsw.dr)*xi(i-1,j,k))/rn*dxi[0]*dxi[0];
                            }
                            nbr += (xi(i,j+1,k) + xi(i,j-1,k))*dxi[1]*dxi[1];
#elif defined(WARPX_DIM_3D)
                            const Real ux = clamp_u(cf(i,j,k,11));
                            const Real uy = clamp_u(cf(i,j,k,12));
                            const Real uz = clamp_u(cf(i,j,k,13));
                            up += gam * dxi[0] * (ux > 0.0_rt
                                ? ux*(xi(i,j,k) - xi(i-1,j,k))
                                : ux*(xi(i+1,j,k) - xi(i,j,k)));
                            up += gam * dxi[1] * (uy > 0.0_rt
                                ? uy*(xi(i,j,k) - xi(i,j-1,k))
                                : uy*(xi(i,j+1,k) - xi(i,j,k)));
                            up += gam * dxi[2] * (uz > 0.0_rt
                                ? uz*(xi(i,j,k) - xi(i,j,k-1))
                                : uz*(xi(i,j,k+1) - xi(i,j,k)));
                            adiag = gam * (std::abs(ux) * dxi[0] + std::abs(uy) * dxi[1]
                                         + std::abs(uz) * dxi[2]);
                            nbr = (xi(i+1,j,k) + xi(i-1,j,k))*dxi[0]*dxi[0]
                                + (xi(i,j+1,k) + xi(i,j-1,k))*dxi[1]*dxi[1]
                                + (xi(i,j,k+1) + xi(i,j,k-1))*dxi[2]*dxi[2];
#endif
                            // conduction: -c_k del2 x = c_k (l2 x - nbr) with
                            // c_k = cond/l2, cond = cf10 - adiag; its
                            // diagonal is inside the (1 + cond) x term, so
                            // only the neighbour sum enters the residual
                            const Real ck = (cf(i,j,k,10) - adiag) / l2_pe;
                            const Real r = b(i,j,k)
                                - (1.0_rt + cf(i,j,k,10) - adiag)*xi(i,j,k)
                                + ck*nbr
                                - up;
                            xo(i,j,k) = xi(i,j,k) + r / (1.0_rt + cf(i,j,k,10));
                        });
                }
                if (sw + 1 < m_pe_row_sweeps) {
                    amrex::MultiFab::Copy(x2, xp, 0, 0, 1, 0);
                }
            }
            // refresh the physical dpe carried to the E leg
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(xp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box vbx = mfi.tilebox();
                auto const& x = xp.const_array(mfi);
                auto const& s = m_pe_row_scr->array(mfi);
                amrex::ParallelFor(vbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        s(i,j,k) = x(i,j,k) * inv_kappa;
                    });
            }
        }
        FillGhosts(*m_pe_row_scr, 0);
    }

    // b~: the PC input b (rotated by (I+S)^-1 with the ion response), with
    // the pe-row gradient folded in and the E-field closure applied
    for (int n = 0; n < 3; n++) {
        if (m_staggered) {
            CopyCompIn(*a_b.getArrayVec()[0][n], *m_bE, n, false);
        } else {
            amrex::MultiFab::Copy(*m_bE, *a_b.getArrayVec()[0][n], 0, n, 1, 0);
        }
    }
    if (pe_row) {
        // b_E <- b_E - grad(dpe)/rho_e: the dF_E/dpe Ohm coupling the
        // identity pe leg left entirely to GMRES. Domain ghosts of the dpe
        // scratch are zero-extended; the affected wall-tangential E rows are
        // identity-overridden on output anyway.
        const auto dxi = m_geom[0].InvCellSizeArray();
        const bool pe_axis = m_has_axis;
        const int pe_iax = m_geom[0].Domain().smallEnd(0);
        amrex::ignore_unused(pe_axis, pe_iax);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*m_bE, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box vbx = mfi.tilebox();
            auto const& be = m_bE->array(mfi);
            auto const& s  = m_pe_row_scr->const_array(mfi);
            auto const& cf = m_coef[0]->const_array(mfi);
            amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const Real ir = 1.0_rt / cf(i,j,k,9);
#if defined(WARPX_DIM_1D_Z)
                    be(i,j,k,2) -= ir * (s(i+1,j,k) - s(i-1,j,k)) * 0.5_rt * dxi[0];
#elif defined(WARPX_DIM_XZ)
                    be(i,j,k,0) -= ir * (s(i+1,j,k) - s(i-1,j,k)) * 0.5_rt * dxi[0];
                    be(i,j,k,2) -= ir * (s(i,j+1,k) - s(i,j-1,k)) * 0.5_rt * dxi[1];
#elif defined(WARPX_DIM_RZ)
                    // d_r pe = 0 on the axis (even scalar; the zero-extended
                    // ghost would fake a gradient there)
                    if (!(pe_axis && i == pe_iax)) {
                        be(i,j,k,0) -= ir * (s(i+1,j,k) - s(i-1,j,k)) * 0.5_rt * dxi[0];
                    }
                    be(i,j,k,2) -= ir * (s(i,j+1,k) - s(i,j-1,k)) * 0.5_rt * dxi[1];
#elif defined(WARPX_DIM_3D)
                    be(i,j,k,0) -= ir * (s(i+1,j,k) - s(i-1,j,k)) * 0.5_rt * dxi[0];
                    be(i,j,k,1) -= ir * (s(i,j+1,k) - s(i,j-1,k)) * 0.5_rt * dxi[1];
                    be(i,j,k,2) -= ir * (s(i,j,k+1) - s(i,j,k-1)) * 0.5_rt * dxi[2];
#else
                    amrex::ignore_unused(i, j, k, ir, be, s, dxi);
#endif
                });
        }
    }
    if (m_have_mm) {
        // the pair is posed on b~ = (I+S)^-1 b (see the class docs)
        RotateValid(*m_bE);
    }
    FillGhosts(*m_bE, 0);
    // PEC dead tangential-E columns (wall values zeroed; the identity rows
    // are restored on the output below) + parity ghosts; EB nodes zeroed
    ApplyDomainClosure(*m_bE, 0, ClosureClass::EField);

    // the E-row RHS is b~, the W-row RHS is zero
    if (m_inner_max > 0) {
        // block RHS (W-half comps 0..2, E-half comps 3..5) for the inner GMRES
        m_bKB->setVal(0.0, 0, 3, 0);
        amrex::MultiFab::Copy(*m_bKB, *m_bE, 0, 3, 3, 0);
        InnerGMRES();
    } else {
        // fixed linear operator: m_vcycles V-cycles from a zero initial guess
        amrex::MultiFab::Copy(*m_bF[0], *m_bE, 0, 0, 3, 0);
        m_bW[0]->setVal(0.0);
        if (!(m_zero_free && m_eb_flag == nullptr)) {
            m_xW[0]->setVal(0.0);
            m_xF[0]->setVal(0.0);
        }
        Vcycle(0, true);
        for (int c = 1; c < m_vcycles; c++) { Vcycle(0, false); }
    }

    // the E-half is the output
    if (m_staggered) {
        // WriteXOut reads the top nodal planes from the first ghost ring of
        // the cell container: fill it (periodic grids only)
        FillGhosts(*m_xF[0], 0);
        WriteXOut(*m_xF[0], a_x);
    } else {
        for (int n = 0; n < 3; n++) {
            amrex::MultiFab::Copy(*a_x.getArrayVec()[0][n], *m_xF[0], n, 0, 1, 0);
        }
    }
    // exact identity rows of the true Jacobian: x = b on wall-tangential E
    // at PEC faces and on EB-covered nodes
    OverrideIdentityRows(a_x, a_b);

    // pe-as-Newton-unknown fallback: with the triangular pe-row leg
    // inactive, pass the scalar (pe) row through as identity so the
    // right-preconditioned operator is nonsingular in those rows (the
    // precond entry zeroes a_x first, so skipping this would pin dpe = 0).
    // With the leg active, a_x's scalar already holds the D_p solve above.
    if (!pe_row && a_x.getScalarVecType() != warpx::fields::FieldType::None) {
        amrex::MultiFab::Copy(*a_x.getScalarVec()[0], *a_b.getScalarVec()[0],
                              0, 0, 1, 0);
    }
}

#endif
