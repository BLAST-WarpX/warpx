/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * Fluid ion response of the hybrid theta-implicit Jacobian
 * (implicit_evolve.use_fluid_ion_response).
 *
 * Every Jacobian action needs the ion current response dJ_g to a field increment dE_g. With the
 * particle positions frozen at the Newton iterate, the implicit push makes the mid-step velocity
 * affine in the field, and the response is dJ_g = sum_g' M_gg' dE_g' with
 *
 *   M_gg' = (1/dV) sum_p q_p alpha_p R_p S(xbar_p - x_g') S(xbar_p - x_g),
 *   alpha = (q/m) dt/2,  b = alpha B,  R = (I + b b^T - [b]x) / (1 + b^2).
 *
 * Replacing the particle sum by its expectation for a smooth density, and evaluating the bracket
 * at the output node, gives
 *
 *   M_gg' = alpha rho_g R(b_g) K_{g-g'},   K = prod_d S_{2s+1}(offset_d),
 *
 * with K the autocorrelation of the shape factor of order s (rows sum to one: the long-wavelength
 * limit is the cold-fluid response). The coefficients alpha*rho and b come from the grid moments
 * of the linearization point, once per Newton iteration; each Jacobian action is a (2s+1)^d
 * stencil on E - E0 and a 3x3 product per node, with no particle work. The nonlinear residual
 * keeps the particles, so Newton converges to the same root. In cylindrical geometry (azimuthal
 * mode 0) the radial weights are int u S(u-i) S(u-i') du (u = r/dr): ((i+i')/2) S_{2s+1}(i-i')
 * away from the axis and the outer wall, tabulated next to them.
 */
#include "ThetaImplicitHybrid.H"

#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
    /** Charge-to-mass ratio of the charged species: the moment form uses the total deposited
     *  charge density, so it needs a common q/m. */
    amrex::Real CommonChargeToMass (const MultiParticleContainer& mpc)
    {
        amrex::Real qom = 0.0_rt;
        for (int is = 0; is < mpc.nSpecies(); ++is) {
            const auto& pc = mpc.GetParticleContainer(is);
            if (pc.getCharge() == 0 || pc.getMass() == 0) { continue; }
            const amrex::Real qom_s = pc.getCharge()/pc.getMass();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                qom == 0.0_rt || std::abs(qom_s - qom) <= 1.e-10_rt*std::abs(qom),
                "implicit_evolve.use_fluid_ion_response requires all charged species "
                "to share the same charge-to-mass ratio");
            qom = qom_s;
        }
        return qom;
    }

    /** K per direction at offsets 0..s: S_{2s+1}, the autocorrelation of the shape factor of
     *  order s (the same-node value is the preconditioner's ion-block weight) */
    amrex::GpuArray<amrex::Real,5> ShapeAutocorrelation (const int s)
    {
        amrex::GpuArray<amrex::Real,5> w{1.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
        if (s == 1) { w[0] = 2._rt/3._rt; w[1] = 1._rt/6._rt; }
        else if (s == 2) { w[0] = 11._rt/20._rt; w[1] = 13._rt/60._rt; w[2] = 1._rt/120._rt; }
        else if (s == 3) {
            w[0] = 151._rt/315._rt; w[1] = 397._rt/1680._rt;
            w[2] = 1._rt/42._rt; w[3] = 1._rt/5040._rt;
        }
        else if (s == 4) {
            w[0] = 15619._rt/36288._rt; w[1] = 44117._rt/181440._rt; w[2] = 913._rt/22680._rt;
            w[3] = 251._rt/181440._rt; w[4] = 1._rt/362880._rt;
        }
        return w;
    }

#if defined(WARPX_DIM_RZ)
    /** Centred cardinal B-spline of degree s at x: the particle shape factor of order s */
    double CentredBSpline (const int s, const double x)
    {
        double sum = 0.0, binom = 1.0, fact = 1.0;
        for (int k = 1; k <= s; ++k) { fact *= k; }
        for (int k = 0; k <= s + 1; ++k) {
            const double t = x + 0.5*(s + 1) - k;
            if (t > 0.0) { sum += ((k % 2) ? -1.0 : 1.0)*binom*std::pow(t, s); }
            binom = binom*(s + 1 - k)/(k + 1);
        }
        return sum/fact;
    }

    // 8-point Gauss-Legendre on half-cell intervals: exact for the piecewise polynomials below
    constexpr double gl_x[4] = {0.1834346424956498, 0.5255324099163290,
                                0.7966664774136267, 0.9602898564975363};
    constexpr double gl_w[4] = {0.3626837833783620, 0.3137066458778873,
                                0.2223810344533745, 0.1012285362903763};

    /** Radial weights next to the axis (nodal r). T(i,d) = int_0^inf u S(u-i) S(u-i-d) du is the
     *  raw (per dr dz) deposit of uniformly distributed particles at r > 0 onto node i for a unit
     *  field at node i+d; rows i = -s..s (the guard rows below the axis are folded back by
     *  ApplyInverseVolumeScalingToCurrentDensity), offsets d = -s..s, index (i+s)(2s+1) + d+s.
     *  c(i), i = 0..s, converts the node density WarpX reports, whose volume is 2 pi i dr
     *  (pi dr axis_factor on the axis), to the local particle density. */
    void AxisKernelTable (const int s, const double axis_factor,
                          amrex::GpuArray<amrex::Real,81>& T, amrex::GpuArray<amrex::Real,5>& c)
    {
        const int nd = 2*s + 1;
        for (int i = -s; i <= s; ++i) {
            for (int d = -s; d <= s; ++d) {
                double sum = 0.0;
                const double umax = std::min(i, i + d) + 0.5*(s + 1);
                for (double a = 0.0; a < umax - 1.e-12; a += 0.5) {
                    for (int q = 0; q < 8; ++q) {
                        const double x = (q < 4) ? -gl_x[q] : gl_x[q - 4];
                        const double u = a + 0.25*(1.0 + x);
                        sum += 0.25*gl_w[q % 4]*u*CentredBSpline(s, u - i)
                               *CentredBSpline(s, u - i - d);
                    }
                }
                T[(i + s)*nd + (d + s)] = static_cast<amrex::Real>(sum);
            }
        }
        for (int i = 0; i <= s; ++i) {
            double m1 = 0.0;
            for (int d = -s; d <= s; ++d) { m1 += T[(i + s)*nd + (d + s)]; }
            const double vol = (i == 0) ? 0.5*axis_factor : static_cast<double>(i);
            c[i] = static_cast<amrex::Real>((m1 > 0.0) ? vol/m1 : 1.0);
        }
    }

    /** One-sided radial weights next to the outer radial boundary, where the particles fill only
     *  u < u_w: T(i, i+d) = int_{-inf}^{u_w} u S(u-i) S(u-i-d) du = u_i A(k,d) + B(k,d) with
     *  k = i_w - i the node distance from the boundary (0..s), A = int_{t<k} S(t) S(t-d) dt and
     *  B = int_{t<k} t S(t) S(t-d) dt; index k(2s+1) + d+s. */
    void WallKernelTable (const int s, amrex::GpuArray<amrex::Real,81>& A,
                          amrex::GpuArray<amrex::Real,81>& B)
    {
        const int nd = 2*s + 1;
        for (int k = 0; k <= s; ++k) {
            for (int d = -s; d <= s; ++d) {
                double sa = 0.0, sb = 0.0;
                const double tlo = std::max(-0.5*(s + 1), d - 0.5*(s + 1));
                const double thi = std::min(std::min(0.5*(s + 1), d + 0.5*(s + 1)),
                                            static_cast<double>(k));
                for (double a = tlo; a < thi - 1.e-12; a += 0.5) {
                    for (int q = 0; q < 8; ++q) {
                        const double x = (q < 4) ? -gl_x[q] : gl_x[q - 4];
                        const double t = a + 0.25*(1.0 + x);
                        const double f = 0.25*gl_w[q % 4]*CentredBSpline(s, t)
                                         *CentredBSpline(s, t - d);
                        sa += f;
                        sb += f*t;
                    }
                }
                A[k*nd + (d + s)] = static_cast<amrex::Real>(sa);
                B[k*nd + (d + s)] = static_cast<amrex::Real>(sb);
            }
        }
    }
#endif
}

void ThetaImplicitHybrid::InitFluidIonResponse ()
{
    const int lev = 0;
    const int ns = WarpX::nox;
    const ablastr::fields::VectorField E = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
    for (int n = 0; n < 3; ++n) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(E[n]->ixType().nodeCentered(),
            "implicit_evolve.use_fluid_ion_response requires the collocated grid");
        m_fir_E0[n] = std::make_unique<amrex::MultiFab>(
            E[n]->boxArray(), E[n]->DistributionMap(), 1, E[n]->nGrowVect());
    }
    // alpha*rho and b = alpha*B at the nodes
    m_fir_coef = std::make_unique<amrex::MultiFab>(E[0]->boxArray(), E[0]->DistributionMap(), 4, 0);
    m_fir_coef->setVal(0.0_rt);
    m_fir_dE = std::make_unique<amrex::MultiFab>(E[0]->boxArray(), E[0]->DistributionMap(), 3, ns);
    // nodes shared by several boxes are written by their owner only: the current then goes
    // through SumBoundary as the particle deposit does
    m_fir_owner = amrex::OwnerMask(*m_fir_coef, m_WarpX->Geom(lev).periodicity());
    m_fir_qom = CommonChargeToMass(m_WarpX->GetPartContainer());
}

const amrex::MultiFab* ThetaImplicitHybrid::GetFluidIonResponseForPC (const int lev) const
{
    return (m_fluid_ion_response && lev == 0) ? m_fir_coef.get() : nullptr;
}

void ThetaImplicitHybrid::SaveFluidIonResponseE0 ()
{
    // the push field of this (nonlinear) evaluation; filtered as the Jacobian actions filter
    // theirs before E - E0 is formed
    const int lev = 0;
    const ablastr::fields::VectorField E = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
    for (int n = 0; n < 3; ++n) {
        amrex::MultiFab::Copy(*m_fir_E0[n], *E[n], 0, 0, 1, E[n]->nGrowVect());
    }
    if (WarpX::use_filter) {
        ablastr::fields::MultiLevelVectorField E0(1);
        E0[0] = {m_fir_E0[0].get(), m_fir_E0[1].get(), m_fir_E0[2].get()};
        m_WarpX->ApplyFilterMF(E0, lev);
    }
}

void ThetaImplicitHybrid::FillFluidIonResponseCoeff ()
{
    BL_PROFILE("ThetaImplicitHybrid::FillFluidIonResponseCoeff()");
    using namespace ablastr::coarsen::sample;

    const int lev = 0;
    const amrex::Real alpha = 0.5_rt*m_fir_qom*m_dt;
    // post-push (half-time) charge density and the total B^{n+theta} of the linearization point
    // (Bfield_fp holds the total field between the external-field add and subtract)
    const amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
    const int rho_comp = rho->nComp() - rho->nComp()/2;
    const ablastr::fields::VectorField B = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
    const amrex::GpuArray<int,3> coarsen = {1, 1, 1};
    amrex::GpuArray<int,3> Bs0{}, Bs1{}, Bs2{}, rs{}, dst{};
    for (int d = 0; d < 3; ++d) {
        const int dd = std::min(d, AMREX_SPACEDIM-1);
        Bs0[d] = (d < AMREX_SPACEDIM) ? B[0]->ixType().nodeCentered(dd) : 0;
        Bs1[d] = (d < AMREX_SPACEDIM) ? B[1]->ixType().nodeCentered(dd) : 0;
        Bs2[d] = (d < AMREX_SPACEDIM) ? B[2]->ixType().nodeCentered(dd) : 0;
        rs[d]  = (d < AMREX_SPACEDIM) ? rho->ixType().nodeCentered(dd) : 0;
        dst[d] = (d < AMREX_SPACEDIM) ? 1 : 0;
    }
    amrex::MultiFab& coef = *m_fir_coef;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(coef, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& c = coef.array(mfi);
        amrex::Array4<amrex::Real const> const& r = rho->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Bx = B[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& By = B[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Bz = B[2]->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            c(i,j,k,0) = alpha*Interp(r, rs, dst, coarsen, i, j, k, rho_comp);
            c(i,j,k,1) = alpha*Interp(Bx, Bs0, dst, coarsen, i, j, k, 0);
            c(i,j,k,2) = alpha*Interp(By, Bs1, dst, coarsen, i, j, k, 0);
            c(i,j,k,3) = alpha*Interp(Bz, Bs2, dst, coarsen, i, j, k, 0);
        });
    }
}

void ThetaImplicitHybrid::FluidIonResponseCurrent ()
{
    BL_PROFILE("ThetaImplicitHybrid::FluidIonResponseCurrent()");

    const int lev = 0;
    const int ns = WarpX::nox;
    const ablastr::fields::VectorField J =
        m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
    const ablastr::fields::VectorField E = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
    amrex::MultiFab& dE = *m_fir_dE;
    const amrex::MultiFab& coef = *m_fir_coef;
    const amrex::iMultiFab& owner = *m_fir_owner;
    for (int n = 0; n < 3; ++n) { J[n]->setVal(0.0_rt); }

    // E - E0 on the valid nodes; zero outside the domain, neighbours by FillBoundary
    dE.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(dE, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& d = dE.array(mfi);
        amrex::Array4<amrex::Real const> const& Ex = E[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Ey = E[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Ez = E[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& E0x = m_fir_E0[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& E0y = m_fir_E0[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& E0z = m_fir_E0[2]->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            d(i,j,k,0) = Ex(i,j,k) - E0x(i,j,k);
            d(i,j,k,1) = Ey(i,j,k) - E0y(i,j,k);
            d(i,j,k,2) = Ez(i,j,k) - E0z(i,j,k);
        });
    }
    dE.FillBoundary(m_WarpX->Geom(lev).periodicity());

#if defined(WARPX_DIM_RZ)
    // Azimuthal mode 0: the kernel produces the raw (per dr dz) current that uniformly distributed
    // particles at r > 0 deposit, including the guard rows below the axis, so that
    // ApplyInverseVolumeScalingToCurrentDensity folds and scales it exactly like the particle
    // deposit. Below the axis, E - E0 is continued with the mode-0 parity.
    const amrex::Geometry& geom = m_WarpX->Geom(lev);
    const amrex::Real dr = geom.CellSize(0);
    const bool axis_at_lo = (geom.ProbLo(0) == 0.0_rt);
    const int ilo_domain = geom.Domain().smallEnd(0);
    const amrex::Real r0_over_dr = geom.ProbLo(0)/dr;
    bool verboncoeur = true;
    {
        const amrex::ParmParse pp_boundary("boundary");
        pp_boundary.query("verboncoeur_axis_correction", verboncoeur);
    }
    const amrex::Real axis_factor = verboncoeur ? 1._rt/3._rt : 1._rt/4._rt;
    amrex::GpuArray<amrex::Real,81> Tax{}, Awall{}, Bwall{};
    amrex::GpuArray<amrex::Real,5> cax{};
    AxisKernelTable(ns, axis_factor, Tax, cax);
    WallKernelTable(ns, Awall, Bwall);
    const int iw_r = geom.Domain().length(0);   // outer boundary node, counted from the domain edge
    const int nd = 2*ns + 1;
    if (axis_at_lo) {
        for (amrex::MFIter mfi(dE); mfi.isValid(); ++mfi) {
            const amrex::Box& vb = mfi.validbox();
            if (vb.smallEnd(0) != ilo_domain) { continue; }
            amrex::Box gb = vb;
            gb.setSmall(0, ilo_domain - ns);
            gb.setBig(0, ilo_domain - 1);
            gb.grow(1, ns);
            amrex::Array4<amrex::Real> const& d = dE.array(mfi);
            amrex::ParallelFor(gb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const int im = 2*ilo_domain - i;
                d(i,j,k,0) = -d(im,j,k,0);
                d(i,j,k,1) = -d(im,j,k,1);
                d(i,j,k,2) =  d(im,j,k,2);
            });
        }
    }
#endif

    const amrex::GpuArray<amrex::Real,5> w = ShapeAutocorrelation(ns);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(dE, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box bx = mfi.tilebox();
#if defined(WARPX_DIM_RZ)
        // guard rows below the axis (folded back later)
        if (axis_at_lo && bx.smallEnd(0) == ilo_domain) { bx.growLo(0, ns); }
#endif
        amrex::Array4<amrex::Real const> const& d = dE.const_array(mfi);
        amrex::Array4<amrex::Real const> const& c = coef.const_array(mfi);
        amrex::Array4<int const> const& own = owner.const_array(mfi);
        amrex::Array4<amrex::Real> const& Jx = J[0]->array(mfi);
        amrex::Array4<amrex::Real> const& Jy = J[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Jz = J[2]->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            amrex::Real F0 = 0.0_rt, F1 = 0.0_rt, F2 = 0.0_rt;
#if defined(WARPX_DIM_RZ)
            const int ir = i - ilo_domain;                 // node index from the domain edge
            const amrex::Real ui = r0_over_dr + ir;        // r_i/dr
            const bool near_axis = axis_at_lo && ir <= ns;
            const bool near_wall = !near_axis && ir >= iw_r - ns;
            if (ir >= 0 && own(i,j,k) == 0) { return; }
            // coefficients of the guard rows from the mirror node
            const int ic = (ir < 0) ? ilo_domain - ir : i;
            for (int o2 = -ns; o2 <= ns; ++o2) {
            for (int o1 = -ns; o1 <= ns; ++o1) {
                const amrex::Real wr = near_axis ? Tax[(ir + ns)*nd + (o1 + ns)]
                                     : near_wall ? ui*Awall[(iw_r - ir)*nd + (o1 + ns)]
                                                   + Bwall[(iw_r - ir)*nd + (o1 + ns)]
                                                 : (ui + 0.5_rt*o1)*w[(o1 < 0) ? -o1 : o1];
                if (wr == 0.0_rt) { continue; }
                const amrex::Real wt = wr*w[(o2 < 0) ? -o2 : o2];
                F0 += wt*d(i+o1,j+o2,k,0);
                F1 += wt*d(i+o1,j+o2,k,1);
                F2 += wt*d(i+o1,j+o2,k,2);
            }}
            // raw current: 2 pi dr times the radial moment above, with the density correction
            // next to the axis
            const amrex::Real pref =
                2.0_rt*MathConst::pi*dr*(near_axis ? cax[(ir < 0) ? -ir : ir] : 1.0_rt);
#else
            if (own(i,j,k) == 0) { return; }
            const int ic = i;
            constexpr amrex::Real pref = 1.0_rt;
#if defined(WARPX_DIM_3D)
            for (int o3 = -ns; o3 <= ns; ++o3) {
            for (int o2 = -ns; o2 <= ns; ++o2) {
            for (int o1 = -ns; o1 <= ns; ++o1) {
                const amrex::Real wt = w[(o1 < 0) ? -o1 : o1]*w[(o2 < 0) ? -o2 : o2]
                                       *w[(o3 < 0) ? -o3 : o3];
                F0 += wt*d(i+o1,j+o2,k+o3,0);
                F1 += wt*d(i+o1,j+o2,k+o3,1);
                F2 += wt*d(i+o1,j+o2,k+o3,2);
            }}}
#elif defined(WARPX_DIM_XZ)
            for (int o2 = -ns; o2 <= ns; ++o2) {
            for (int o1 = -ns; o1 <= ns; ++o1) {
                const amrex::Real wt = w[(o1 < 0) ? -o1 : o1]*w[(o2 < 0) ? -o2 : o2];
                F0 += wt*d(i+o1,j+o2,k,0);
                F1 += wt*d(i+o1,j+o2,k,1);
                F2 += wt*d(i+o1,j+o2,k,2);
            }}
#else
            for (int o1 = -ns; o1 <= ns; ++o1) {
                const amrex::Real wt = w[(o1 < 0) ? -o1 : o1];
                F0 += wt*d(i+o1,j,k,0);
                F1 += wt*d(i+o1,j,k,1);
                F2 += wt*d(i+o1,j,k,2);
            }
#endif
#endif
            const amrex::Real bx_ = c(ic,j,k,1);
            const amrex::Real by_ = c(ic,j,k,2);
            const amrex::Real bz_ = c(ic,j,k,3);
            const amrex::Real a = pref*c(ic,j,k,0)/(1.0_rt + bx_*bx_ + by_*by_ + bz_*bz_);
            // alpha*rho*R(b) F with R = (I + b b^T - [b]x)/(1 + b^2)
            const amrex::Real bF = bx_*F0 + by_*F1 + bz_*F2;
            Jx(i,j,k) += a*(F0 + bx_*bF - (by_*F2 - bz_*F1));
            Jy(i,j,k) += a*(F1 + by_*bF - (bz_*F0 - bx_*F2));
            Jz(i,j,k) += a*(F2 + bz_*bF - (bx_*F1 - by_*F0));
        });
    }

    // the treatment of the particle deposit (volume scaling, filter and guard sums, PEC
    // reflection), then J = J_base + dJ
#if defined(WARPX_DIM_RZ)
    m_WarpX->ApplyInverseVolumeScalingToCurrentDensity(J[0], J[1], J[2], lev);
#endif
    m_WarpX->SyncCurrent("current_fp");
    m_WarpX->ApplyJfieldBoundary(lev, J[0], J[1], J[2], PatchType::fine);
    for (int n = 0; n < 3; ++n) {
        amrex::MultiFab::Add(*J[n], *m_J_base[n], 0, 0, J[n]->nComp(), J[n]->nGrowVect());
    }
}
