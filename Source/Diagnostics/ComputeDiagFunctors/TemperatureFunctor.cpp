
#include "TemperatureFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

TemperatureFunctor::TemperatureFunctor (const int lev,
        const amrex::IntVect crse_ratio, const int ispec, const int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_lev(lev), m_ispec(ispec)
{
    // Write only in one output component.
    AMREX_ALWAYS_ASSERT(ncomp == 1);
}

void
TemperatureFunctor::operator() (amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/) const
{
    using namespace amrex::literals;
    auto& warpx = WarpX::GetInstance();

    // Guard cell is set to 1 for generality. However, for a cell-centered
    // output Multifab, mf_dst, the guard-cell data is not needed especially considering
    // the operations performend in the CoarsenAndInterpolate function.
    constexpr int ng = 1;

    // Temporary cell-centered, multi-component MultiFab for storing particles sums and result
    amrex::MultiFab sum_mf(warpx.boxArray(m_lev), warpx.DistributionMap(m_lev), 7, ng);

    auto& pc = warpx.GetPartContainer().GetParticleContainer(m_ispec);
    amrex::Real const mass = pc.getMass();  // Note, implicit conversion from ParticleReal

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mass > 0.,
        "The temperature diagnostic can not be calculated for a massless species.");

    // Calculate the averages in two steps, first the average velocity <u>, then the
    // average velocity squared <u - <u>>**2. This method is more robust than the
    // single step using <u**2> - <u>**2 when <u> >> u_rms.
    ParticleToMesh(pc, sum_mf, m_lev,
            [=] AMREX_GPU_DEVICE (const WarpXParticleContainer::SuperParticleType& p,
                amrex::Array4<amrex::Real> const& out_array,
                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
            {
                // Get position in AMReX convention to calculate corresponding index.
                const auto [ii, jj, kk] = amrex::getParticleCell(p, plo, dxi).dim3();

                const amrex::ParticleReal w  = p.rdata(PIdx::w);
                const amrex::ParticleReal ux = p.rdata(PIdx::ux);
                const amrex::ParticleReal uy = p.rdata(PIdx::uy);
                const amrex::ParticleReal uz = p.rdata(PIdx::uz);
                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 0), (amrex::Real)(w));
                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 1), (amrex::Real)(w*ux));
                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 2), (amrex::Real)(w*uy));
                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 3), (amrex::Real)(w*uz));
            });

    // Divide value by number of particles for average
    for (amrex::MFIter mfi(sum_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.tilebox();
        amrex::Array4<amrex::Real> const& out_array = sum_mf.array(mfi);
        amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (out_array(i,j,k,0) > 0) {
                        const amrex::Real invsum = 1._rt/out_array(i,j,k,0);
                        out_array(i,j,k,1) *= invsum;
                        out_array(i,j,k,2) *= invsum;
                        out_array(i,j,k,3) *= invsum;
                    }
                });
    }

    // Calculate the sum of the squares, subtracting the averages
    // These loops must be written out since ParticleToMesh always zeros out the mf.
    const auto plo = pc.Geom(m_lev).ProbLoArray();
    const auto dxi = pc.Geom(m_lev).InvCellSizeArray();
    for (WarpXParIter pti(pc, m_lev); pti.isValid(); ++pti)
    {
        const long np = pti.numParticles();
        auto& tile = pti.GetParticleTile();
        auto ptd = tile.getParticleTileData();
        amrex::ParticleReal* wp = pti.GetAttribs(PIdx::w).dataPtr();
        amrex::ParticleReal* uxp = pti.GetAttribs(PIdx::ux).dataPtr();
        amrex::ParticleReal* uyp = pti.GetAttribs(PIdx::uy).dataPtr();
        amrex::ParticleReal* uzp = pti.GetAttribs(PIdx::uz).dataPtr();

        amrex::Array4<amrex::Real> const& out_array = sum_mf.array(pti);

        amrex::ParallelFor(np,
            [=] AMREX_GPU_DEVICE (long ip) {
                // Get position in AMReX convention to calculate corresponding index.
                const auto p = WarpXParticleContainer::ParticleType(ptd, ip);
                const auto [ii, jj, kk] = getParticleCell(p, plo, dxi).dim3();

                const amrex::ParticleReal w  = wp[ip];
                const amrex::ParticleReal ux = uxp[ip] - out_array(ii, jj, kk, 1);
                const amrex::ParticleReal uy = uyp[ip] - out_array(ii, jj, kk, 2);
                const amrex::ParticleReal uz = uzp[ip] - out_array(ii, jj, kk, 3);
                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 4), (amrex::Real)(w*ux*ux));
                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 5), (amrex::Real)(w*uy*uy));
                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 6), (amrex::Real)(w*uz*uz));
            });
    }

    // Divide the squares by number of particles for average and calculate the temperature
    for (amrex::MFIter mfi(sum_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.tilebox();
        amrex::Array4<amrex::Real> const& out_array = sum_mf.array(mfi);
        amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (out_array(i,j,k,0) > 0) {
                        const amrex::Real invsum = 1._rt/out_array(i,j,k,0);
                        out_array(i,j,k,4) *= invsum;
                        out_array(i,j,k,5) *= invsum;
                        out_array(i,j,k,6) *= invsum;
                        out_array(i,j,k,0) = mass*(out_array(i,j,k,4) + out_array(i,j,k,5) + out_array(i,j,k,6))/(3._rt*PhysConst::q_e);
                    }
                });
    }

    // Coarsen and interpolate from sum_mf to the output diagnostic MultiFab, mf_dst.
    ablastr::coarsen::sample::Coarsen(mf_dst, sum_mf, dcomp, 0, nComp(), 0, m_crse_ratio);

}
