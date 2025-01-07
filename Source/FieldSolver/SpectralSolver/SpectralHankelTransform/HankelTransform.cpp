/* Copyright 2019 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HankelTransform.H"

#include "BesselRoots.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include "Utils/WarpXProfilerWrapper.H"

#include <blas.hh>
#include <lapack.hh>

using namespace amrex::literals;

HankelTransform::HankelTransform (int const hankel_order,
                                  int const azimuthal_mode,
                                  int const nr,
                                  const amrex::Real rmax)
: m_nr(nr), m_nk(nr)
{

    WARPX_PROFILE("HankelTransform::HankelTransform");

    // Check that azimuthal_mode has a valid value
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(hankel_order-1 <= azimuthal_mode && azimuthal_mode <= hankel_order+1,
                                     "azimuthal_mode must be either hankel_order-1, hankel_order or hankel_order+1");

#ifdef AMREX_USE_GPU
    // BLAS setup
    //   SYCL note: we need to double check AMReX device ID conventions and
    //   BLAS++ device ID conventions are the same
    int const device_id = amrex::Gpu::Device::deviceId();
    blas::Queue::stream_t stream_id = amrex::Gpu::gpuStream();
    m_queue = std::make_unique<blas::Queue>( device_id, stream_id );
#endif

    amrex::Vector<amrex::Real> alphas;
    amrex::Vector<int> alpha_errors;

    GetBesselRoots(azimuthal_mode, m_nk, alphas, alpha_errors);
    // Add of check of alpha_errors, should be all zeros
    AMREX_ALWAYS_ASSERT(std::all_of(alpha_errors.begin(), alpha_errors.end(), [](int i) { return i == 0; }));

    // Calculate the spectral grid
    amrex::Vector<amrex::Real> kr(m_nk);
    for (int ik=0 ; ik < m_nk ; ik++) {
        kr[ik] = alphas[ik]/rmax;
    }

    // Calculate the spatial grid (Uniform grid with a half-cell offset)
    amrex::Vector<amrex::Real> rmesh(m_nr);
    const amrex::Real dr = rmax/m_nr;
    for (int ir=0 ; ir < m_nr ; ir++) {
        rmesh[ir] = dr*(ir + 0.5_rt);
    }

    // Calculate and store the inverse matrix invM
    // (imposed by the constraints on the DHT of Bessel modes)
    // NB: When compared with the FBPIC article, all the matrices here
    // are calculated in transposed form. This is done so as to use the
    // `dot` and `gemm` functions, in the `transform` method.
    const int p_denom = (hankel_order == azimuthal_mode)?(hankel_order + 1):(hankel_order);

    amrex::Vector<amrex::Real> denom(m_nk);
    for (int ik=0 ; ik < m_nk ; ik++) {
        const auto jna = static_cast<amrex::Real>(jn(p_denom, alphas[ik]));
        denom[ik] = MathConst::pi*rmax*rmax*jna*jna;
    }

    amrex::Vector<amrex::Real> num(m_nk*m_nr);
    for (int ir=0 ; ir < m_nr ; ir++) {
        for (int ik=0 ; ik < m_nk ; ik++) {
            int const ii = ik + ir*m_nk;
            num[ii] = static_cast<amrex::Real>(jn(hankel_order, rmesh[ir]*kr[ik]));
        }
    }

    // Get the inverse matrix
    amrex::Vector<amrex::Real> invM(m_nk*m_nr);
    if (azimuthal_mode > 0) {
        for (int ir=0 ; ir < m_nr ; ir++) {
            for (int ik=1 ; ik < m_nk ; ik++) {
                int const ii = ik + ir*m_nk;
                invM[ii] = num[ii]/denom[ik];
            }
        }
        // ik = 0
        // In this case, the functions are represented by Bessel functions
        // *and* an additional mode (below) which satisfies the same
        // algebric relations for curl/div/grad as the regular Bessel modes,
        // with the value kperp=0.
        // The normalization of this mode is arbitrary, and is chosen
        // so that the condition number of invM is close to 1
        if (hankel_order == azimuthal_mode-1) {
            for (int ir=0 ; ir < m_nr ; ir++) {
                int const ii = ir*m_nk;
                invM[ii] = static_cast<amrex::Real>(
                    std::pow(rmesh[ir], (azimuthal_mode-1))/(MathConst::pi*std::pow(rmax, (azimuthal_mode+1))));
            }
        } else {
            for (int ir=0 ; ir < m_nr ; ir++) {
                int const ii = ir*m_nk;
                invM[ii] = 0.;
            }
        }
    } else {
        for (int ir=0 ; ir < m_nr ; ir++) {
            for (int ik=0 ; ik < m_nk ; ik++) {
                int const ii = ik + ir*m_nk;
                invM[ii] = num[ii]/denom[ik];
            }
        }
    }

    amrex::Vector<amrex::Real> M;

    // Calculate the matrix M by inverting invM
    if (azimuthal_mode !=0 && hankel_order != azimuthal_mode-1) {
        // In this case, invM is singular, thus we calculate the pseudo-inverse.
        // The Moore-Penrose pseudo-inverse is calculated using the SVD method.

        M.resize(m_nk*m_nr, 0.);
        amrex::Vector<amrex::Real> invMcopy(invM);
        amrex::Vector<amrex::Real> sdiag(m_nk-1, 0.);
        amrex::Vector<amrex::Real> u((m_nk-1)*(m_nk-1), 0.);
        amrex::Vector<amrex::Real> vt((m_nr)*(m_nr), 0.);
        amrex::Vector<amrex::Real> sp((m_nr)*(m_nk-1), 0.);
        amrex::Vector<amrex::Real> temp((m_nr)*(m_nk-1), 0.);

        // Calculate the singular-value-decomposition of invM (leaving out the first row).
        // invM = u*sdiag*vt
        // Note that invMcopy.dataPtr()+1 is passed in so that the first ik row is skipped
        // A copy is passed in since the matrix is destroyed
        lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec,
                      m_nk-1, m_nr, invMcopy.dataPtr()+1, m_nk,
                      sdiag.dataPtr(), u.dataPtr(), m_nk-1,
                      vt.dataPtr(), m_nr);

        // Calculate the pseudo-inverse of sdiag, which is trivial since it
        // only has values of the diagonal.
        for (int i=0 ; i < m_nk-1 ; i++) {
            if (sdiag[i] != 0.) {
                int const j = i + i*m_nk;
                sp[j] = 1._rt/sdiag[i];
            }
        }

        // M can be calculated from the SVD
        // M = v*sp*u_transpose

        // Do the second multiply, temp = sp*u_transpose
        // temp(1:n,1:m) = matmul(transpose(vt(1:n,1:n)), matmul(sp(1:n,1:m), transpose(u(1:m,1:m))))
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                   m_nr, m_nk-1, m_nk-1, 1._rt,
                   sp.dataPtr(), m_nr,
                   u.dataPtr(), m_nk-1, 0._rt,
                   temp.dataPtr(), m_nr);

        // Do the first mutiply, M = vt*temp
        // Note that M.dataPtr()+m_nr is passed in so that the first ir column is skipped
        blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   m_nr, m_nk-1, m_nr, 1.,
                   vt.dataPtr(), m_nr,
                   temp.dataPtr(), m_nr, 0._rt,
                   M.dataPtr()+m_nr, m_nr);

    } else {
        // In this case, invM is invertible; calculate the inverse.
        // getrf calculates the LU decomposition
        // getri calculates the inverse from the LU decomposition

        M = invM;
        amrex::Vector<int64_t> ipiv(m_nr);
        lapack::getrf(m_nk, m_nr, M.dataPtr(), m_nk, ipiv.dataPtr());
        lapack::getri(m_nr, M.dataPtr(), m_nr, ipiv.dataPtr());

    }

    m_kr.resize(kr.size());
    m_invM.resize(invM.size());
    m_M.resize(M.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, kr.begin(), kr.end(), m_kr.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, invM.begin(), invM.end(), m_invM.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, M.begin(), M.end(), m_M.begin());
    amrex::Gpu::synchronize();
}

void
HankelTransform::HankelForwardTransform (amrex::FArrayBox const& F, int const F_icomp,
                                         amrex::FArrayBox      & G, int const G_icomp)
{
    WARPX_PROFILE("HankelTransform::HankelForwardTransform");

    amrex::Box const& F_box = F.box();
    amrex::Box const& G_box = G.box();

    int const nrF = F_box.length(0);
    int const nz = F_box.length(1);
    int const ngr = G_box.smallEnd(0) - F_box.smallEnd(0);

    AMREX_ALWAYS_ASSERT(m_nr == G_box.length(0));
    AMREX_ALWAYS_ASSERT(nz == G_box.length(1));
    AMREX_ALWAYS_ASSERT(ngr >= 0);
    AMREX_ALWAYS_ASSERT(F_box.bigEnd(0)+1 >= m_nr);

    // We perform stream synchronization since `gemm` may be running
    // on a different stream.
    amrex::Gpu::streamSynchronize();

    // Note that M is flagged to be transposed since it has dimensions (m_nr, m_nk)
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               m_nk, nz, m_nr, 1._rt,
               m_M.dataPtr(), m_nk,
               F.dataPtr(F_icomp)+ngr, nrF, 0._rt,
               G.dataPtr(G_icomp), m_nk
#ifdef AMREX_USE_GPU
               , *m_queue // Calls the GPU version of blas::gemm
#endif
           );

    // We perform stream synchronization since `gemm` may be running
    // on a different stream.
    amrex::Gpu::streamSynchronize();

}

void
HankelTransform::HankelInverseTransform (amrex::FArrayBox const& G, int const G_icomp,
                                         amrex::FArrayBox      & F, int const F_icomp)
{
    WARPX_PROFILE("HankelTransform::HankelInverseTransform");

    amrex::Box const& G_box = G.box();
    amrex::Box const& F_box = F.box();

    int const nrF = F_box.length(0);
    int const nz = F_box.length(1);
    int const ngr = G_box.smallEnd(0) - F_box.smallEnd(0);

    AMREX_ALWAYS_ASSERT(m_nr == G_box.length(0));
    AMREX_ALWAYS_ASSERT(nz == G_box.length(1));
    AMREX_ALWAYS_ASSERT(ngr >= 0);
    AMREX_ALWAYS_ASSERT(F_box.bigEnd(0)+1 >= m_nr);

    // We perform stream synchronization since `gemm` may be running
    // on a different stream.
    amrex::Gpu::streamSynchronize();

    // Note that m_invM is flagged to be transposed since it has dimensions (m_nk, m_nr)
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               m_nr, nz, m_nk, 1._rt,
               m_invM.dataPtr(), m_nr,
               G.dataPtr(G_icomp), m_nk, 0._rt,
               F.dataPtr(F_icomp)+ngr, nrF
#ifdef AMREX_USE_GPU
               , *m_queue // Calls the GPU version of blas::gemm
#endif
           );

    // We perform stream synchronization since `gemm` may be running
    // on a different stream.
    amrex::Gpu::streamSynchronize();
}
