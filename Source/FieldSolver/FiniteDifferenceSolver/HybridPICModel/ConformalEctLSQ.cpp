/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 *
 * Accurate conformal-EB hybrid-PIC Ampere current ("conformal_ect_lsq", "Path 1").
 *
 * The masked Yee curl J = curl(B)/mu0 is only O(1)-accurate at a curved cut wall.
 * This scheme decouples accuracy from divergence-consistency:
 *
 *   1. the existing PEC covered-B fill (normal odd -> B_n=0, tangential even ->
 *      Neumann) feeds the STANDARD Yee curl (already correct, driven by the flag);
 *   2. the near-wall (cut-band) J edges are OVERWRITTEN by a noise-free weighted
 *      least-squares CENTROID reconstruction of that curl, fit over the surrounding
 *      genuine-fluid edges of the same family in the local wall-normal frame and
 *      evaluated at each edge's fluid centroid -- this recovers a converging wall
 *      current where the bare curl is O(1)-wrong;
 *   3. the matched cut-metric divergence clean (centroid-aware divergence D paired
 *      with its exact transpose grad := -D^T, on the agglomerated cut complex)
 *      restores divergence-consistency.
 *
 * This file holds the shared LSQ centroid-weight machinery (used by both the
 * wall-J overwrite and the cut-metric divergence D). The weights are GEOMETRY-ONLY
 * (they depend on the cut geometry + the wall-normal frame + which neighbour edges
 * are fluid, not on the field values), so they are precomputed once at init/regrid
 * and cached as a per-target list of (neighbour offset, weight) taps; the per-step
 * apply is then a cheap sparse weighted sum -- the same cache pattern as the
 * quadratic b-curl-fill gather (EBJBoundary EBFillStatus).
 *
 * Faithful C++ port of the validated prototypes
 * Docs/eb_fill_review/{lsq_centroid_div_3d,accurate_input_matched_clean_3d}.py
 * (functions lsq_weights / gather_lsq).
 */

#include "HybridPICModel.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/utils/Communication.H>

#include <AMReX_REAL.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_Math.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Reduce.H>

#include <array>
#include <cmath>
#include <memory>

namespace warpx::hybrid::conformal_lsq
{
using amrex::Real;
using namespace amrex::literals;

/** Max polynomial coefficients carried by the LSQ centroid fit: a degree-2
 *  polynomial in 3 local coordinates {1, x, y, z, x^2, y^2, z^2, xy, yz, zx}. */
static constexpr int max_ncoef = 10;

/** Number of polynomial coefficients for a degree-d fit in 3D (d in {0,1,2}). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ncoef (int deg)
{
    // deg 0 -> 1 (constant), deg 1 -> 4 (+ x,y,z), deg 2 -> 10 (+ 6 quadratics)
    return (deg <= 0) ? 1 : (deg == 1) ? 4 : max_ncoef;
}

/** Evaluate the polynomial basis row at a local-frame point (sx, sy, sz).
 *  Index 0 is the constant term, so the fit's constant coefficient IS the value
 *  at the centroid (s = 0). Fills b[0..ncoef(deg)-1]. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void poly_basis (Real sx, Real sy, Real sz, int deg, Real* b)
{
    b[0] = 1.0_rt;
    if (deg >= 1) {
        b[1] = sx; b[2] = sy; b[3] = sz;
    }
    if (deg >= 2) {
        b[4] = sx*sx; b[5] = sy*sy; b[6] = sz*sz;
        b[7] = sx*sy; b[8] = sy*sz; b[9] = sz*sx;
    }
}

/** In-place Cholesky factorization of an SPD matrix G (nc x nc, row-major in a
 *  flat max_ncoef*max_ncoef buffer); returns false if not positive-definite
 *  (pivot <= 0), in which case the caller falls back to a lower degree. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool cholesky_factor (Real* G, int nc)
{
    for (int j = 0; j < nc; ++j) {
        Real d = G[j*max_ncoef + j];
        for (int k = 0; k < j; ++k) { d -= G[j*max_ncoef + k] * G[j*max_ncoef + k]; }
        if (d <= 0.0_rt) { return false; }
        d = std::sqrt(d);
        G[j*max_ncoef + j] = d;
        Real const inv_d = 1.0_rt / d;
        for (int i = j+1; i < nc; ++i) {
            Real s = G[i*max_ncoef + j];
            for (int k = 0; k < j; ++k) { s -= G[i*max_ncoef + k] * G[j*max_ncoef + k]; }
            G[i*max_ncoef + j] = s * inv_d;
        }
    }
    return true;
}

/** Solve G x = rhs in place (rhs -> x) given the Cholesky factor L (lower) of G
 *  produced by cholesky_factor (G now holds L). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cholesky_solve (Real const* L, int nc, Real* x)
{
    // forward solve L y = rhs
    for (int i = 0; i < nc; ++i) {
        Real s = x[i];
        for (int k = 0; k < i; ++k) { s -= L[i*max_ncoef + k] * x[k]; }
        x[i] = s / L[i*max_ncoef + i];
    }
    // back solve L^T x = y
    for (int i = nc-1; i >= 0; --i) {
        Real s = x[i];
        for (int k = i+1; k < nc; ++k) { s -= L[k*max_ncoef + i] * x[k]; }
        x[i] = s / L[i*max_ncoef + i];
    }
}

/** Weighted-least-squares CENTROID weights for one target.
 *
 *  Given \p k sample taps with local-frame coordinates \p sloc (k x 3, the
 *  neighbour-edge centroid minus the target centroid, rotated into the wall-normal
 *  frame) and nonnegative Gaussian weights \p wmask (k; 0 = excluded / non-fluid),
 *  returns in \p w0 the linear map sample-values -> the fit value at s = 0 (the
 *  centroid). That is, reconstructed_centroid_value = sum_t w0[t] * sample[t].
 *
 *  Solves the ridge-regularized weighted normal equations
 *      G = B^T W B + ridge*tr(G)/nc * I ,   w0 = (e0^T G^{-1}) B^T W ,
 *  with e0 the constant-term selector. Degree auto-drops 2 -> 1 -> 0 when there are
 *  too few fluid taps or the Gram is not SPD; degree 0 is the inverse-distance
 *  weighted mean. \p w0 must hold at least \p k entries; entries are zero for
 *  excluded taps. Mirrors prototype lsq_weights() (ridge = 1e-9 * tr/nc). */
AMREX_GPU_HOST_DEVICE inline
int lsq_centroid_weights (int k, Real const* sloc, Real const* wmask,
                          int deg, Real* w0, Real ridge = 1.e-9_rt)
{
    for (int t = 0; t < k; ++t) { w0[t] = 0.0_rt; }

    int n_fluid = 0;
    for (int t = 0; t < k; ++t) { if (wmask[t] > 0.0_rt) { ++n_fluid; } }
    if (n_fluid == 0) { return -1; }

    for (int d = deg; d >= 1; --d) {
        int const nc = ncoef(d);
        if (n_fluid < nc) { continue; }

        // Weighted Gram G = B^T W B (basis row recomputed per tap below).
        Real G[max_ncoef*max_ncoef] = {};
        for (int t = 0; t < k; ++t) {
            Real const w = wmask[t];
            if (w <= 0.0_rt) { continue; }
            Real b[max_ncoef];
            poly_basis(sloc[3*t+0], sloc[3*t+1], sloc[3*t+2], d, b);
            for (int a = 0; a < nc; ++a) {
                Real const wb = w * b[a];
                for (int c = a; c < nc; ++c) { G[a*max_ncoef + c] += wb * b[c]; }
            }
        }
        // symmetrize lower triangle + ridge on the diagonal (ridge * tr/nc)
        Real tr = 0.0_rt;
        for (int a = 0; a < nc; ++a) { tr += G[a*max_ncoef + a]; }
        Real const lam = ridge * tr / static_cast<Real>(nc);
        for (int a = 0; a < nc; ++a) {
            G[a*max_ncoef + a] += lam;
            for (int c = 0; c < a; ++c) { G[a*max_ncoef + c] = G[c*max_ncoef + a]; }
        }
        // x = G^{-1} e0  (e0 = constant-term selector)
        Real x[max_ncoef] = {};
        x[0] = 1.0_rt;
        if (!cholesky_factor(G, nc)) { continue; }  // not SPD -> drop a degree
        cholesky_solve(G, nc, x);
        // w0[t] = (G^{-1} e0)^T (B^T W)_{.,t} = sum_a x[a] * w_t * b_a(s_t)
        for (int t = 0; t < k; ++t) {
            Real const w = wmask[t];
            if (w <= 0.0_rt) { continue; }
            Real b[max_ncoef];
            poly_basis(sloc[3*t+0], sloc[3*t+1], sloc[3*t+2], d, b);
            Real acc = 0.0_rt;
            for (int a = 0; a < nc; ++a) { acc += x[a] * b[a]; }
            w0[t] = w * acc;
        }
        return d;
    }

    // degree-0 fallback: inverse-distance (Gaussian) weighted mean
    Real wsum = 0.0_rt;
    for (int t = 0; t < k; ++t) { if (wmask[t] > 0.0_rt) { wsum += wmask[t]; } }
    Real const inv = 1.0_rt / (wsum + 1.e-300_rt);
    for (int t = 0; t < k; ++t) { w0[t] = (wmask[t] > 0.0_rt) ? wmask[t] * inv : 0.0_rt; }
    return 0;
}

/** Build a right-handed orthonormal frame R (row-major 3x3) whose first row is the
 *  unit wall normal n and whose other two rows span the tangent plane. The LSQ fit is
 *  invariant to the choice/sign of tangent axes, so any consistent frame works. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void make_frame (Real nx, Real ny, Real nz, Real* R)
{
    Real const an = std::sqrt(nx*nx + ny*ny + nz*nz) + 1.e-300_rt;
    nx /= an; ny /= an; nz /= an;
    // pick the axis least aligned with n to seed a tangent
    Real ax = 0.0_rt, ay = 0.0_rt, az = 0.0_rt;
    Real const axn = amrex::Math::abs(nx), ayn = amrex::Math::abs(ny), azn = amrex::Math::abs(nz);
    if (axn <= ayn && axn <= azn) { ax = 1.0_rt; }
    else if (ayn <= azn)          { ay = 1.0_rt; }
    else                          { az = 1.0_rt; }
    // t1 = normalize(a - (a.n) n)
    Real const adotn = ax*nx + ay*ny + az*nz;
    Real t1x = ax - adotn*nx, t1y = ay - adotn*ny, t1z = az - adotn*nz;
    Real const at1 = std::sqrt(t1x*t1x + t1y*t1y + t1z*t1z) + 1.e-300_rt;
    t1x /= at1; t1y /= at1; t1z /= at1;
    // t2 = n x t1
    Real const t2x = ny*t1z - nz*t1y;
    Real const t2y = nz*t1x - nx*t1z;
    Real const t2z = nx*t1y - ny*t1x;
    R[0] = nx;  R[1] = ny;  R[2] = nz;
    R[3] = t1x; R[4] = t1y; R[5] = t1z;
    R[6] = t2x; R[7] = t2y; R[8] = t2z;
}

/** Trilinearly interpolate the NODAL signed-distance field \p phi at an arbitrary
 *  world point (x,y,z). Used to sub-sample the cut geometry of the node-dual control
 *  volumes and their facets (the AMReX cell-EB metrics are the wrong staggering for a
 *  node-centred dual complex). The point must lie within one cell of a valid (ghost-
 *  filled) node; callers only sub-sample inside the wall band where distance_to_eb is
 *  a genuine sub-grid distance, so trilinear blending is locally ~2nd order. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real sdf_at (amrex::Array4<Real const> const& phi,
             amrex::GpuArray<Real,3> const& plo, amrex::GpuArray<Real,3> const& dx,
             amrex::Dim3 const& lo, amrex::Dim3 const& hi,
             Real x, Real y, Real z)
{
    // continuous node-index coordinates
    Real const fi = (x - plo[0]) / dx[0];
    Real const fj = (y - plo[1]) / dx[1];
    Real const fk = (z - plo[2]) / dx[2];
    int i0 = static_cast<int>(std::floor(fi));
    int j0 = static_cast<int>(std::floor(fj));
    int k0 = static_cast<int>(std::floor(fk));
    // clamp the base cell so the 8 corners stay in range (ubound inclusive)
    i0 = amrex::min(amrex::max(i0, lo.x), hi.x - 1);
    j0 = amrex::min(amrex::max(j0, lo.y), hi.y - 1);
    k0 = amrex::min(amrex::max(k0, lo.z), hi.z - 1);
    Real const tx = fi - static_cast<Real>(i0);
    Real const ty = fj - static_cast<Real>(j0);
    Real const tz = fk - static_cast<Real>(k0);
    Real const c000 = phi(i0,   j0,   k0  ), c100 = phi(i0+1, j0,   k0  );
    Real const c010 = phi(i0,   j0+1, k0  ), c110 = phi(i0+1, j0+1, k0  );
    Real const c001 = phi(i0,   j0,   k0+1), c101 = phi(i0+1, j0,   k0+1);
    Real const c011 = phi(i0,   j0+1, k0+1), c111 = phi(i0+1, j0+1, k0+1);
    Real const c00 = c000*(1-tx) + c100*tx, c10 = c010*(1-tx) + c110*tx;
    Real const c01 = c001*(1-tx) + c101*tx, c11 = c011*(1-tx) + c111*tx;
    Real const c0 = c00*(1-ty) + c10*ty, c1 = c01*(1-ty) + c11*ty;
    return c0*(1-tz) + c1*tz;
}

/** Fluid (sdf<0) fraction of the cell-sized rectangle centred at (cx,cy,cz) lying in
 *  the plane perpendicular to axis \p nax (the two in-plane axes span their cell sizes
 *  dx[a1] x dx[a2], sub-sampled M x M). The open area fraction of a node-dual facet. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real frac_square (amrex::Array4<Real const> const& phi,
                  amrex::GpuArray<Real,3> const& plo, amrex::GpuArray<Real,3> const& dx,
                  amrex::Dim3 const& lo, amrex::Dim3 const& hi,
                  Real cx, Real cy, Real cz, int nax, int M)
{
    int const a1 = (nax + 1) % 3;
    int const a2 = (nax + 2) % 3;
    int cnt = 0;
    for (int p = 0; p < M; ++p) {
        Real const s1 = ((p + 0.5_rt) / M - 0.5_rt) * dx[a1];
        for (int q = 0; q < M; ++q) {
            Real const s2 = ((q + 0.5_rt) / M - 0.5_rt) * dx[a2];
            Real o[3] = {0.0_rt, 0.0_rt, 0.0_rt};
            o[a1] = s1; o[a2] = s2;
            if (sdf_at(phi, plo, dx, lo, hi, cx + o[0], cy + o[1], cz + o[2]) < 0.0_rt) { ++cnt; }
        }
    }
    return static_cast<Real>(cnt) / static_cast<Real>(M * M);
}

} // namespace warpx::hybrid::conformal_lsq

using namespace amrex;
namespace clsq = warpx::hybrid::conformal_lsq;

void HybridPICModel::ApplyConformalLSQOverwrite (
    ablastr::fields::VectorField const& Jfield,
    const int lev) const
{
#if defined(WARPX_DIM_3D)
    using warpx::fields::FieldType;
    if (!EB::enabled()) { return; }

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const Real hmax = amrex::max(dx[0], amrex::max(dx[1], dx[2]));

    // de-risk-validated LSQ parameters (rad=2, deg=2, sfac=0.4, isotropic)
    constexpr int rad = 2;
    constexpr int deg = 2;
    constexpr int ntap = (2*rad+1)*(2*rad+1)*(2*rad+1);   // 125
    const Real sigma = rad * hmax * 0.4_rt;
    const Real aniso = 1.0_rt;
    const Real d_lo = 0.05_rt * hmax;   // exclude the immediate wall layer
    const Real d_hi = 2.0_rt * hmax;    // wall band outer edge
    const Real inv_2sig2 = 1.0_rt / (2.0_rt * sigma * sigma);

    amrex::MultiFab const* phi_mf = warpx.m_fields.get(FieldType::distance_to_eb, lev);
    auto eco = warpx.m_fields.get_alldirs(FieldType::edge_cent_offset, lev);

    for (int c = 0; c < 3; ++c) {
        // Snapshot the standard curl for this family so the overwrite reads the
        // ORIGINAL values (a wall-band tap may itself be an overwrite target).
        amrex::MultiFab Jsnap(Jfield[c]->boxArray(), Jfield[c]->DistributionMap(),
                              1, amrex::IntVect(rad));
        amrex::MultiFab::Copy(Jsnap, *Jfield[c], 0, 0, 1, 0);
        Jsnap.FillBoundary(geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*Jfield[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const tb = mfi.tilebox(Jfield[c]->ixType().toIntVect());
            amrex::Array4<Real> const& J = Jfield[c]->array(mfi);
            amrex::Array4<Real const> const& Js = Jsnap.const_array(mfi);
            amrex::Array4<Real const> const& ph = phi_mf->const_array(mfi);
            amrex::Array4<Real const> const& of = eco[c]->const_array(mfi);

            // bounds of the snapshot / nodal-phi arrays for safe neighbour reads
            // (ubound is INCLUSIVE)
            amrex::Dim3 const jslo = amrex::lbound(Js), jshi = amrex::ubound(Js);
            amrex::Dim3 const phlo = amrex::lbound(ph), phhi = amrex::ubound(ph);

            amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // unit offset along the edge (cell-centred) direction c
                int const e0 = (c==0), e1 = (c==1), e2 = (c==2);

                // phi at this edge = mean of its two endpoint nodes (nodal phi)
                auto phi_at = [=] (int a, int b, int cc) -> Real {
                    if (a < phlo.x || a+e0 > phhi.x || b < phlo.y || b+e1 > phhi.y
                        || cc < phlo.z || cc+e2 > phhi.z) { return amrex::Real(-1.e30); }
                    return 0.5_rt * (ph(a,b,cc) + ph(a+e0,b+e1,cc+e2));
                };
                Real const d_e = phi_at(i,j,k);
                if (d_e <= d_lo || d_e >= d_hi) { return; }   // not a wall-band fluid edge

                // wall normal = grad(phi) at the edge (nodal central differences,
                // averaged over the two endpoint nodes for the cross directions)
                Real g[3];
                g[c] = (ph(i+e0,j+e1,k+e2) - ph(i,j,k)) / dx[c];
                for (int d = 0; d < 3; ++d) {
                    if (d == c) { continue; }
                    int da=(d==0), db=(d==1), dc=(d==2);
                    Real const s0 = ph(i+da,j+db,k+dc) - ph(i-da,j-db,k-dc);
                    Real const s1 = ph(i+e0+da,j+e1+db,k+e2+dc) - ph(i+e0-da,j+e1-db,k+e2-dc);
                    g[d] = 0.5_rt * (s0 + s1) / (2.0_rt * dx[d]);
                }
                Real R[9];
                clsq::make_frame(g[0], g[1], g[2], R);

                // target = edge midpoint shifted to the fluid centroid along c
                Real p[3];
                p[0] = plo[0] + (i + 0.5_rt*e0) * dx[0];
                p[1] = plo[1] + (j + 0.5_rt*e1) * dx[1];
                p[2] = plo[2] + (k + 0.5_rt*e2) * dx[2];
                p[c] += of(i,j,k) * dx[c];

                // gather the rad-box of same-family edges
                Real sloc[3*ntap];
                Real wmask[ntap];
                Real samp[ntap];
                int t = 0;
                for (int dk = -rad; dk <= rad; ++dk) {
                for (int dj = -rad; dj <= rad; ++dj) {
                for (int di = -rad; di <= rad; ++di, ++t) {
                    int const ni = i+di, nj = j+dj, nk = k+dk;
                    sloc[3*t+0] = 0.0_rt; sloc[3*t+1] = 0.0_rt; sloc[3*t+2] = 0.0_rt;
                    wmask[t] = 0.0_rt; samp[t] = 0.0_rt;
                    // in-range for the snapshot read?
                    if (ni < jslo.x || ni > jshi.x || nj < jslo.y || nj > jshi.y
                        || nk < jslo.z || nk > jshi.z) { continue; }
                    Real const d_n = phi_at(ni,nj,nk);
                    if (d_n <= d_lo) { continue; }            // covered / wall-layer -> excluded
                    // neighbour edge midpoint
                    Real np[3];
                    np[0] = plo[0] + (ni + 0.5_rt*e0) * dx[0];
                    np[1] = plo[1] + (nj + 0.5_rt*e1) * dx[1];
                    np[2] = plo[2] + (nk + 0.5_rt*e2) * dx[2];
                    Real const dv0 = np[0]-p[0], dv1 = np[1]-p[1], dv2 = np[2]-p[2];
                    // rotate into the wall-normal frame (row 0 = normal)
                    Real sn = R[0]*dv0 + R[1]*dv1 + R[2]*dv2;
                    Real st1 = R[3]*dv0 + R[4]*dv1 + R[5]*dv2;
                    Real st2 = R[6]*dv0 + R[7]*dv1 + R[8]*dv2;
                    sloc[3*t+0] = sn; sloc[3*t+1] = st1; sloc[3*t+2] = st2;
                    Real const sn_s = aniso * sn;
                    Real const r2 = sn_s*sn_s + st1*st1 + st2*st2;
                    wmask[t] = std::exp(-r2 * inv_2sig2);
                    samp[t] = Js(ni,nj,nk);
                }}}

                Real w0[ntap];
                clsq::lsq_centroid_weights(ntap, sloc, wmask, deg, w0);
                Real acc = 0.0_rt;
                for (int tt = 0; tt < ntap; ++tt) { acc += w0[tt] * samp[tt]; }
                J(i,j,k) = acc;
            });
        }
    }
#else
    amrex::ignore_unused(Jfield, lev);
#endif
}

#if defined(WARPX_DIM_3D)
namespace
{
    /** out = A * in, where A is the open-area-fraction-weighted nodal Laplacian
     *  A = -(flux_balance . grad_node) (symmetric positive-semidefinite; its only null
     *  space is constants per connected fluid component). `in` must be FillBoundary'd by
     *  the caller; `area` is the 3-component nodal dual-facet-area MultiFab (component c =
     *  the +c facet, sitting at the same index as the c-edge). */
    void apply_area_laplacian (amrex::MultiFab& out, amrex::MultiFab const& in,
                               amrex::MultiFab const& area)
    {
        using namespace amrex;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const tb = mfi.tilebox();
            auto const& o = out.array(mfi);
            auto const& p = in.const_array(mfi);
            auto const& a = area.const_array(mfi);
            ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                o(i,j,k) =
                    a(i,j,k,0)*(p(i,j,k)-p(i+1,j,k)) + a(i-1,j,k,0)*(p(i,j,k)-p(i-1,j,k))
                  + a(i,j,k,1)*(p(i,j,k)-p(i,j+1,k)) + a(i,j-1,k,1)*(p(i,j,k)-p(i,j-1,k))
                  + a(i,j,k,2)*(p(i,j,k)-p(i,j,k+1)) + a(i,j,k-1,2)*(p(i,j,k)-p(i,j,k-1));
            });
        }
    }

    /** dst = a .* b elementwise (single component, valid region). */
    void elem_mult (amrex::MultiFab& dst, amrex::MultiFab const& a, amrex::MultiFab const& b)
    {
        using namespace amrex;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const tb = mfi.tilebox();
            auto const& d = dst.array(mfi);
            auto const& aa = a.const_array(mfi);
            auto const& bb = b.const_array(mfi);
            ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                d(i,j,k) = aa(i,j,k) * bb(i,j,k);
            });
        }
    }

    /** DIAGNOSTIC: RMS over the wall band (-2h < phi < 0) of (a) the cut-metric
     *  flux-balance divergence of J and (b) the plain Cartesian Yee divergence of J,
     *  to compare which metric the clean reduces. J ghosts must be filled by caller. */
    void probe_div_norms (ablastr::fields::VectorField const& J,
                          amrex::MultiFab const& area, amrex::MultiFab const& phi_eb,
                          amrex::Real hmax, amrex::Real& cut_rms, amrex::Real& cart_rms)
    {
        using namespace amrex;
        ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum> rop;
        ReduceData<Real, Real, Real> rdat(rop);
        using RT = ReduceData<Real, Real, Real>::Type;
        for (MFIter mfi(area, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Box const tb = mfi.tilebox();
            auto const& a = area.const_array(mfi);
            auto const& jx = J[0]->const_array(mfi);
            auto const& jy = J[1]->const_array(mfi);
            auto const& jz = J[2]->const_array(mfi);
            auto const& ph = phi_eb.const_array(mfi);
            rop.eval(tb, rdat, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> RT
            {
                Real const d = ph(i,j,k);
                if (!(d < 0.0_rt && d > -2.0_rt*hmax)) { return {0.0_rt, 0.0_rt, 0.0_rt}; }
                Real const cut = a(i,j,k,0)*jx(i,j,k) - a(i-1,j,k,0)*jx(i-1,j,k)
                               + a(i,j,k,1)*jy(i,j,k) - a(i,j-1,k,1)*jy(i,j-1,k)
                               + a(i,j,k,2)*jz(i,j,k) - a(i,j,k-1,2)*jz(i,j,k-1);
                Real const cart = (jx(i,j,k)-jx(i-1,j,k)) + (jy(i,j,k)-jy(i,j-1,k))
                                + (jz(i,j,k)-jz(i,j,k-1));
                return {cut*cut, cart*cart, 1.0_rt};
            });
        }
        RT hv = rdat.value();
        Real scut = amrex::get<0>(hv), scart = amrex::get<1>(hv), cnt = amrex::get<2>(hv);
        ParallelDescriptor::ReduceRealSum(scut);
        ParallelDescriptor::ReduceRealSum(scart);
        ParallelDescriptor::ReduceRealSum(cnt);
        cnt = amrex::max(cnt, 1.0_rt);
        cut_rms  = std::sqrt(scut  / cnt);
        cart_rms = std::sqrt(scart / cnt);
    }
} // anonymous namespace
#endif

void HybridPICModel::BuildConformalDualAreas (const int lev) const
{
#if defined(WARPX_DIM_3D)
    using warpx::fields::FieldType;
    if (!EB::enabled()) { return; }
    if (static_cast<int>(m_eb_dual_area_built.size()) <= lev) {
        m_eb_dual_area_built.resize(lev+1, 0);
        m_eb_dual_area.resize(lev+1);
    }
    if (m_eb_dual_area_built[lev]) { return; }

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const Real hmax = amrex::max(dx[0], amrex::max(dx[1], dx[2]));
    const Real gate = hmax;                       // band within which a facet can be cut
    const int M = amrex::max(2, m_conformal_divclean_subsample);
    // Cartesian mode: binary fluid-area operator (1 in fluid / 0 covered) -> the clean
    // targets the plain Yee divergence instead of the cut-metric flux-balance one.
    const bool cartesian = (m_conformal_divclean_cartesian != 0);

    amrex::MultiFab const* phi_mf = warpx.m_fields.get(FieldType::distance_to_eb, lev);

    // nodal, 3 components (one open-facet area fraction per +direction), 1 ghost
    m_eb_dual_area[lev] = std::make_unique<amrex::MultiFab>(
        phi_mf->boxArray(), phi_mf->DistributionMap(), 3, 1);
    auto& A = *m_eb_dual_area[lev];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(A, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const tb = mfi.tilebox();
        amrex::Array4<Real> const& Aa = A.array(mfi);
        amrex::Array4<Real const> const& ph = phi_mf->const_array(mfi);
        amrex::Dim3 const plb = amrex::lbound(ph), pub = amrex::ubound(ph);
        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            for (int c = 0; c < 3; ++c) {
                int const e0=(c==0), e1=(c==1), e2=(c==2);
                Real const cx = plo[0] + (i + 0.5_rt*e0) * dx[0];
                Real const cy = plo[1] + (j + 0.5_rt*e1) * dx[1];
                Real const cz = plo[2] + (k + 0.5_rt*e2) * dx[2];
                Real const d = clsq::sdf_at(ph, plo, dx, plb, pub, cx, cy, cz);
                Real frac;
                if (cartesian)       { frac = (d < 0.0_rt) ? 1.0_rt : 0.0_rt; }
                else if (d <= -gate) { frac = 1.0_rt; }
                else if (d >=  gate) { frac = 0.0_rt; }
                else { frac = clsq::frac_square(ph, plo, dx, plb, pub, cx, cy, cz, c, M); }
                Aa(i,j,k,c) = frac;
            }
        });
    }
    A.FillBoundary(geom.periodicity());
    m_eb_dual_area_built[lev] = 1;
#else
    amrex::ignore_unused(lev);
#endif
}

void HybridPICModel::ApplyConformalDivClean (
    ablastr::fields::VectorField const& Jfield, const int lev) const
{
#if defined(WARPX_DIM_3D)
    using warpx::fields::FieldType;
    if (!EB::enabled()) { return; }
    if (m_conformal_divclean_iters <= 0) { return; }   // A/B knob: skip the clean

    BuildConformalDualAreas(lev);

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    amrex::MultiFab const& area = *m_eb_dual_area[lev];

    amrex::MultiFab const* phi_mf = warpx.m_fields.get(FieldType::distance_to_eb, lev);
    amrex::BoxArray const& nba = phi_mf->boxArray();          // nodal
    amrex::DistributionMapping const& ndm = phi_mf->DistributionMap();

    amrex::MultiFab phi  (nba, ndm, 1, 1);
    amrex::MultiFab res  (nba, ndm, 1, 0);
    amrex::MultiFab zz   (nba, ndm, 1, 0);
    amrex::MultiFab pdir (nba, ndm, 1, 1);
    amrex::MultiFab Ap   (nba, ndm, 1, 0);
    amrex::MultiFab idiag(nba, ndm, 1, 0);
    amrex::MultiFab rhs  (nba, ndm, 1, 0);
    phi.setVal(0.0_rt);
    pdir.setVal(0.0_rt);

    // Inverse Jacobi diagonal: 1/diag(A); diag = sum of the node's six facet areas.
    // Zero where the node has no open facet (deep covered) -> decoupled, phi stays 0.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(idiag, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const tb = mfi.tilebox();
        auto const& id = idiag.array(mfi);
        auto const& a = area.const_array(mfi);
        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real const diag = a(i,j,k,0)+a(i-1,j,k,0) + a(i,j,k,1)+a(i,j-1,k,1)
                            + a(i,j,k,2)+a(i,j,k-1,2);
            id(i,j,k) = (diag > 1.e-12_rt) ? 1.0_rt/diag : 0.0_rt;
        });
    }

    // rhs = -b = -flux_balance(J): the cut-metric divergence of the current (negated).
    // The standard Yee curl is identically div-free in the full interior, so b (and the
    // whole solve) is supported only in the cut wall band.
    for (int c = 0; c < 3; ++c) { Jfield[c]->FillBoundary(geom.periodicity()); }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const tb = mfi.tilebox();
        auto const& b = rhs.array(mfi);
        auto const& a = area.const_array(mfi);
        auto const& jx = Jfield[0]->const_array(mfi);
        auto const& jy = Jfield[1]->const_array(mfi);
        auto const& jz = Jfield[2]->const_array(mfi);
        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            b(i,j,k) = -( a(i,j,k,0)*jx(i,j,k) - a(i-1,j,k,0)*jx(i-1,j,k)
                        + a(i,j,k,1)*jy(i,j,k) - a(i,j-1,k,1)*jy(i,j-1,k)
                        + a(i,j,k,2)*jz(i,j,k) - a(i,j,k-1,2)*jz(i,j,k-1) );
        });
    }

    // Jacobi-preconditioned CG on A phi = rhs (A = apply_area_laplacian, SPD).
    auto mask = amrex::OwnerMask(phi, geom.periodicity());
    auto dot = [&] (amrex::MultiFab const& x, amrex::MultiFab const& y) -> Real {
        return amrex::MultiFab::Dot(*mask, x, 0, y, 0, 1, 0);
    };

    amrex::MultiFab::Copy(res, rhs, 0, 0, 1, 0);             // res = rhs - A*0 = rhs
    elem_mult(zz, res, idiag);                               // z = M^{-1} res
    amrex::MultiFab::Copy(pdir, zz, 0, 0, 1, 0);
    Real rz = dot(res, zz);
    Real const res0 = std::sqrt(amrex::max(dot(res, res), Real(0.0)));
    Real const rtol = m_conformal_divclean_rtol;
    int const maxit = m_conformal_divclean_iters;
    Real resn = res0;
    int it_done = 0;

    if (res0 > 0.0_rt && rz > 0.0_rt) {
        for (int it = 0; it < maxit; ++it) {
            pdir.FillBoundary(geom.periodicity());
            apply_area_laplacian(Ap, pdir, area);
            Real const pAp = dot(pdir, Ap);
            if (!(amrex::Math::abs(pAp) > 1.e-300_rt)) { break; }
            Real const alpha = rz / pAp;
            amrex::MultiFab::Saxpy(phi, alpha, pdir, 0, 0, 1, 0);   // phi += alpha*pdir
            amrex::MultiFab::Saxpy(res, -alpha, Ap, 0, 0, 1, 0);    // res -= alpha*Ap
            resn = std::sqrt(amrex::max(dot(res, res), Real(0.0)));
            it_done = it + 1;
            if (resn <= rtol * res0) { break; }
            elem_mult(zz, res, idiag);
            Real const rzn = dot(res, zz);
            Real const beta = rzn / rz;
            amrex::MultiFab::Xpay(pdir, beta, zz, 0, 0, 1, 0);      // pdir = zz + beta*pdir
            rz = rzn;
        }
    }

    // DIAGNOSTIC (first few calls): area sanity + CG convergence + which div metric moves.
    static int s_diag_calls = 0;
    bool const diag = (s_diag_calls < 3);
    Real cut_b = 0.0_rt, cart_b = 0.0_rt, cut_a = 0.0_rt, cart_a = 0.0_rt;
    if (diag) {
        Real const hmax = amrex::max(geom.CellSize(0),
                                     amrex::max(geom.CellSize(1), geom.CellSize(2)));
        probe_div_norms(Jfield, area, *phi_mf, hmax, cut_b, cart_b);
        amrex::Print() << "[divclean] call " << s_diag_calls
                       << " area[min,max]=" << area.min(0) << "," << area.max(0)
                       << "  CG it=" << it_done << " res0=" << res0
                       << " resN=" << resn << " (rtol*res0=" << rtol*res0 << ")\n";
    }

    // Correction J -= grad_node(phi): drives flux_balance(J) -> 0 while leaving curl(J)
    // unchanged (a pure gradient). Supported only where phi != 0 (the wall band).
    phi.FillBoundary(geom.periodicity());
    for (int c = 0; c < 3; ++c) {
        int const e0=(c==0), e1=(c==1), e2=(c==2);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*Jfield[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const tb = mfi.tilebox();
            auto const& J = Jfield[c]->array(mfi);
            auto const& p = phi.const_array(mfi);
            amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                J(i,j,k) -= (p(i+e0,j+e1,k+e2) - p(i,j,k));
            });
        }
    }

    if (diag) {
        for (int c = 0; c < 3; ++c) { Jfield[c]->FillBoundary(geom.periodicity()); }
        Real const hmax = amrex::max(geom.CellSize(0),
                                     amrex::max(geom.CellSize(1), geom.CellSize(2)));
        probe_div_norms(Jfield, area, *phi_mf, hmax, cut_a, cart_a);
        amrex::Print() << "[divclean] call " << s_diag_calls
                       << " CUT-metric div band-RMS: " << cut_b << " -> " << cut_a
                       << "   CARTESIAN div band-RMS: " << cart_b << " -> " << cart_a << "\n";
        ++s_diag_calls;
    }
#else
    amrex::ignore_unused(Jfield, lev);
#endif
}
