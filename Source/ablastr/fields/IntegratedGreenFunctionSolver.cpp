/* Copyright 2019-2024 Arianna Formenti, Remi Lehe
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "IntegratedGreenFunctionSolver.H"

#include <ablastr/constant.H>
#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/math/fft/AnyFFT.H>
#include <ablastr/utils/TextMsg.H>

#include <AMReX_Array4.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#if defined(ABLASTR_USE_HEFFTE)
#include <heffte.h>
#endif


namespace ablastr::fields {

void
computePhiIGF ( amrex::MultiFab const & rho,
                amrex::MultiFab & phi,
                std::array<amrex::Real, 3> const & cell_size,
                amrex::BoxArray const & ba,
                bool const is_2d_slices,
                bool const is_3d_distributed)
{
    using namespace amrex::literals;

    BL_PROFILE_VAR_NS("ablastr::fields::computePhiIGF: FFTs", timer_ffts);
    BL_PROFILE_VAR_NS("ablastr::fields::computePhiIGF: FFT plans", timer_plans);
    BL_PROFILE_VAR_NS("ablastr::fields::computePhiIGF: parallel copies", timer_pcopies);
    BL_PROFILE_VAR_NS("ablastr::fields::computePhiIGF: Green functions", timer_Gfunc);

    BL_PROFILE("ablastr::fields::computePhiIGF");

    // Check that the user-defined inputs are consistent
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(!(is_2d_slices && is_3d_distributed), \
    "Must choose between the 2d slice solver and the fully 3D solver.");

    // Define box that encompasses the full domain
    amrex::Box domain = ba.minimalBox();
    domain.surroundingNodes(); // get nodal points, since `phi` and `rho` are nodal
    domain.grow( phi.nGrowVect() ); // include guard cells

    int const nx = domain.length(0);
    int const ny = domain.length(1);
    int const nz = domain.length(2);

    // Allocate 2x wider (_big) arrays for the convolution of rho with the Green function
    // The arrays are doubled in z only if the solver is full 3D
    int const nx_big = 2*nx-1;
    int const ny_big = 2*ny-1;
    int const nz_big = (!is_2d_slices) ? 2*nz-1 : nz;

    amrex::Box const realspace_box = amrex::Box(
        {domain.smallEnd(0), domain.smallEnd(1), domain.smallEnd(2)},
        {nx_big+domain.smallEnd(0), ny_big+domain.smallEnd(1), nz_big+domain.smallEnd(2)},
        amrex::IntVect::TheNodeVector() );

    amrex::BoxArray realspace_ba;
    amrex::DistributionMapping dm_global_fft;

    if(is_2d_slices || is_3d_distributed){
        // Define a new distribution mapping which is decomposed purely along z
        // and has one box per MPI rank
        int const nprocs = amrex::ParallelDescriptor::NProcs();

        int const realspace_nx = realspace_box.length(0);
        int const realspace_ny = realspace_box.length(1);
        int const realspace_nz = realspace_box.length(2);
        int const minsize_z = realspace_nz / nprocs;
        int const nleft_z = realspace_nz - minsize_z*nprocs;

        AMREX_ALWAYS_ASSERT(realspace_nz >= nprocs);
        // We are going to split realspace_box in such a way that the first
        // nleft boxes has minsize_z+1 nodes and the others minsize
        // nodes. We do it this way instead of BoxArray::maxSize to make
        // sure there are exactly nprocs boxes and there are no overlaps.
        amrex::BoxList bl(amrex::IndexType::TheNodeType());
        for (int iproc = 0; iproc < nprocs; ++iproc) {
            int zlo, zhi;
            if (iproc < nleft_z) {
                zlo = iproc*(minsize_z+1);
                zhi = zlo + minsize_z;
            } else {
                zlo = iproc*minsize_z + nleft_z;
                zhi = zlo + minsize_z - 1;
            }
            amrex::Box tbx(amrex::IntVect(0,0,zlo),amrex::IntVect(realspace_nx-1,realspace_ny-1,zhi),amrex::IntVect(1));
            tbx.shift(realspace_box.smallEnd());
            bl.push_back(tbx);
        }
        realspace_ba.define(std::move(bl));
        amrex::Vector<int> pmap(nprocs);
        std::iota(pmap.begin(), pmap.end(), 0);
        dm_global_fft.define(std::move(pmap));
    }
    else{
        // In 3D without distributed FFTs (i.e. without heFFTe):
        // allocate the 2x wider array on a single box
        realspace_ba = amrex::BoxArray(realspace_box);
        // Define a distribution mapping for the global FFT, with only one box
        dm_global_fft.define(realspace_ba);
    }

    // Allocate required arrays
    amrex::MultiFab tmp_rho = amrex::MultiFab(realspace_ba, dm_global_fft, 1, 0);
    tmp_rho.setVal(0);
    amrex::MultiFab tmp_G = amrex::MultiFab(realspace_ba, dm_global_fft, 1, 0);
    tmp_G.setVal(0);

    BL_PROFILE_VAR_START(timer_pcopies);
    // Copy from rho including its ghost cells to tmp_rho
    tmp_rho.ParallelCopy( rho, 0, 0, 1, amrex::IntVect::TheZeroVector(), amrex::IntVect::TheZeroVector() );
    BL_PROFILE_VAR_STOP(timer_pcopies);

    amrex::BoxArray const& igf_compute_box = (is_2d_slices || is_3d_distributed) ? tmp_G.boxArray() : amrex::BoxArray(domain);

    // Compute the integrated Green function
    BL_PROFILE_VAR_START(timer_Gfunc);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(igf_compute_box, dm_global_fft, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        amrex::Box const bx = mfi.tilebox();

        amrex::IntVect const lo = realspace_box.smallEnd();
        amrex::IntVect const hi = realspace_box.bigEnd();

        amrex::Real const dx = cell_size[0];
        amrex::Real const dy = cell_size[1];
        amrex::Real const dz = cell_size[2];

        amrex::Real const x_hi = dx*(hi[0]+2);
        amrex::Real const y_hi = dy*(hi[1]+2);
        amrex::Real const z_hi = dz*(hi[2]+2);
        amrex::ignore_unused(z_hi);

        amrex::Array4<amrex::Real> const tmp_G_arr = tmp_G.array(mfi);

        amrex::ParallelFor( bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                int const i0 = i - lo[0];
                int const j0 = j - lo[1];
                amrex::Real const x = i0*dx;
                amrex::Real const y = j0*dy;

                if (is_2d_slices) {
                    // 2D
                    if ((i0< nx)&&(j0< ny)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential2D(x,      y,      dx, dy); }
                    if ((i0< nx)&&(j0> ny)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential2D(x,      y_hi-y, dx, dy); }
                    if ((i0> nx)&&(j0> ny)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential2D(x_hi-x, y_hi-y, dx, dy); }
                    if ((i0> nx)&&(j0< ny)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential2D(x_hi-x, y,      dx, dy); }
                } else {
                    int const k0 = k - lo[2];
                    amrex::Real const z = k0*dz;
#if defined(ABLASTR_USE_HEFFTE)
                    if (is_3d_distributed) {
                        // 3D with distributed FFTs (i.e. with heFFTe):
                        if ((i0< nx)&&(j0< ny)&&(k0< nz)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential3D(x     , y     , z     , dx, dy, dz); }
                        if ((i0< nx)&&(j0> ny)&&(k0< nz)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential3D(x     , y_hi-y, z     , dx, dy, dz); }
                        if ((i0< nx)&&(j0< ny)&&(k0> nz)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential3D(x     , y     , z_hi-z, dx, dy, dz); }
                        if ((i0> nx)&&(j0> ny)&&(k0< nz)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential3D(x_hi-x, y_hi-y, z     , dx, dy, dz); }
                        if ((i0< nx)&&(j0> ny)&&(k0> nz)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential3D(x     , y_hi-y, z_hi-z, dx, dy, dz); }
                        if ((i0> nx)&&(j0< ny)&&(k0> nz)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential3D(x_hi-x, y     , z_hi-z, dx, dy, dz); }
                        if ((i0> nx)&&(j0> ny)&&(k0> nz)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential3D(x_hi-x, y_hi-y, z_hi-z, dx, dy, dz); }
                        if ((i0> nx)&&(j0< ny)&&(k0< nz)) { tmp_G_arr(i,j,k) = SumOfIntegratedPotential3D(x_hi-x, y     , z     , dx, dy, dz); }
                    } else
#endif
                    {
                        // 3D without distributed FFTs (i.e. without heFFTe):
                        amrex::Real const G_value = SumOfIntegratedPotential3D(x     , y     , z     , dx, dy, dz);
                        tmp_G_arr(i,j,k) = G_value;
                        // Fill the rest of the array by periodicity
                        if (i0>0) {tmp_G_arr(hi[0]+1-i0, j         , k         ) = G_value;}
                        if (j0>0) {tmp_G_arr(i         , hi[1]+1-j0, k         ) = G_value;}
                        if (k0>0) {tmp_G_arr(i         , j         , hi[2]+1-k0) = G_value;}
                        if ((i0>0)&&(j0>0)) {tmp_G_arr(hi[0]+1-i0, hi[1]+1-j0, k         ) = G_value;}
                        if ((j0>0)&&(k0>0)) {tmp_G_arr(i         , hi[1]+1-j0, hi[2]+1-k0) = G_value;}
                        if ((i0>0)&&(k0>0)) {tmp_G_arr(hi[0]+1-i0, j         , hi[2]+1-k0) = G_value;}
                        if ((i0>0)&&(j0>0)&&(k0>0)) {tmp_G_arr(hi[0]+1-i0, hi[1]+1-j0, hi[2]+1-k0) = G_value;}
                    }
                }
            }
        );
    }
    BL_PROFILE_VAR_STOP(timer_Gfunc);

    // Prepare to perform global FFT
    // Since there is 1 MPI rank per box, here each MPI rank obtains its local box and the associated boxid
    const int local_boxid = amrex::ParallelDescriptor::MyProc(); // because of how we made the DistributionMapping
    if (local_boxid < realspace_ba.size()) {
        // When not distributed, there is only one box (the global box)
        // It is taken care of my MPI rank 0 ; other ranks have no work (hence the if condition)

        const amrex::Box local_nodal_box = realspace_ba[local_boxid];
        amrex::Box local_box(local_nodal_box.smallEnd(), local_nodal_box.bigEnd());
        local_box.shift(-realspace_box.smallEnd()); // This simplifies the setup because the global lo is zero now
        // Since we the domain decompostion is in the z-direction, setting up c_local_box is simple.
        amrex::Box c_local_box = local_box;
        c_local_box.setBig(0, local_box.length(0)/2+1);

        // Allocate array in spectral space
        using SpectralField = amrex::BaseFab< amrex::GpuComplex< amrex::Real > > ;
        SpectralField tmp_rho_fft(c_local_box, 1, amrex::The_Device_Arena());
        SpectralField tmp_G_fft(c_local_box, 1, amrex::The_Device_Arena());
        tmp_rho_fft.shift(realspace_box.smallEnd());
        tmp_G_fft.shift(realspace_box.smallEnd());

        // Number of cells in real space
        const int nrx = local_box.length(0);
        const int nry = local_box.length(1);
        const int nrz = local_box.length(2);

        // Number of cells in spectral space
        const int nsx = c_local_box.length(0);
        const int nsy = c_local_box.length(1);
        const int nsz = c_local_box.length(2);

        if(is_2d_slices){
            // Create many plans at once
            // The 2D slice solver performes, for every z, a 2D transform in the transverse plane (x,y).
            // Hence, we use AMREX_SPACEDIM-1=2 and not AMREX_SPACEDIM=3 in the batched FFT calculations.
            int fft_size[] = {nry, nrx};
            BL_PROFILE_VAR_START(timer_plans);
            ablastr::math::anyfft::FFTplan forward_plan_rho = ablastr::math::anyfft::CreatePlanMany(
                fft_size, tmp_rho[local_boxid].dataPtr(),
                reinterpret_cast<ablastr::math::anyfft::Complex*>(tmp_rho_fft.dataPtr()),
                ablastr::math::anyfft::direction::R2C, AMREX_SPACEDIM-1,
                nrz, nullptr, 1, nrx*nry, nullptr, 1, nsx*nsy);
            ablastr::math::anyfft::FFTplan forward_plan_G = ablastr::math::anyfft::CreatePlanMany(
                fft_size, tmp_G[local_boxid].dataPtr(),
                reinterpret_cast<ablastr::math::anyfft::Complex*>(tmp_G_fft.dataPtr()),
                ablastr::math::anyfft::direction::R2C, AMREX_SPACEDIM-1,
                nrz, nullptr, 1, nrx*nry, nullptr, 1, nsx*nsy);
            ablastr::math::anyfft::FFTplan backward_plan = ablastr::math::anyfft::CreatePlanMany(
                fft_size, tmp_G[local_boxid].dataPtr(),
                reinterpret_cast<ablastr::math::anyfft::Complex*>(tmp_G_fft.dataPtr()),
                ablastr::math::anyfft::direction::C2R, AMREX_SPACEDIM-1,
                nsz, nullptr, 1, nsx*nsy, nullptr, 1, nrx*nry);
            BL_PROFILE_VAR_STOP(timer_plans);

            // Forward transforms of rho and G
            BL_PROFILE_VAR_START(timer_ffts);
            ablastr::math::anyfft::Execute(forward_plan_rho);
            ablastr::math::anyfft::Execute(forward_plan_G);
            BL_PROFILE_VAR_STOP(timer_ffts);

            // Multiply tmp_G_fft and tmp_rho_fft in spectral space
            // Store the result in-place in tmp_G_fft, to save memory
            tmp_G_fft.template mult<amrex::RunOn::Device>(tmp_rho_fft, 0, 0, 1);
            amrex::Gpu::streamSynchronize();

            // Backtransform the product of FFT(rho) and FFT(G)
            BL_PROFILE_VAR_START(timer_ffts);
            ablastr::math::anyfft::Execute(backward_plan);
            BL_PROFILE_VAR_STOP(timer_ffts);

            // Remove plans
            ablastr::math::anyfft::DestroyPlan(forward_plan_G);
            ablastr::math::anyfft::DestroyPlan(forward_plan_rho);
            ablastr::math::anyfft::DestroyPlan(backward_plan);

            // Normalize, since (FFT + inverse FFT) results in a factor N
            // Note that we use nrx*nry instead of realspace_box.numPts()
            // because here FFTs are in 2D along x and y
            const amrex::Real normalization = 1._rt / (nrx*nry);
            tmp_G.mult( normalization );
        } else {
#if defined(ABLASTR_USE_HEFFTE)
            if (is_3d_distributed) { // Fully 3d solver on many processes
                 // Create plans
                BL_PROFILE_VAR_START(timer_plans);
#if defined(AMREX_USE_CUDA)
                heffte::fft3d_r2c<heffte::backend::cufft> fft
#elif defined(AMREX_USE_HIP)
                heffte::fft3d_r2c<heffte::backend::rocfft> fft
#else
                heffte::fft3d_r2c<heffte::backend::fftw> fft
#endif
                ({{local_box.smallEnd(0), local_box.smallEnd(1), local_box.smallEnd(2)},
                {local_box.bigEnd(0), local_box.bigEnd(1), local_box.bigEnd(2)}},
                {{c_local_box.smallEnd(0), c_local_box.smallEnd(1), c_local_box.smallEnd(2)},
                {c_local_box.bigEnd(0), c_local_box.bigEnd(1), c_local_box.bigEnd(2)}},
                0, amrex::ParallelDescriptor::Communicator());
                BL_PROFILE_VAR_STOP(timer_plans);

                // Create heffte-friendly objects
                using heffte_complex = typename heffte::fft_output<amrex::Real>::type;
                heffte_complex* rho_fft_data = (heffte_complex*) tmp_rho_fft.dataPtr();
                heffte_complex* G_fft_data = (heffte_complex*) tmp_G_fft.dataPtr();

                // Forward transforms of rho and G
                BL_PROFILE_VAR_START(timer_ffts);
                fft.forward(tmp_rho[local_boxid].dataPtr(), rho_fft_data);
                fft.forward(tmp_G[local_boxid].dataPtr(), G_fft_data);
                BL_PROFILE_VAR_STOP(timer_ffts);

                // Multiply tmp_G_fft and tmp_rho_fft in spectral space
                // Store the result in-place in tmp_G_fft, to save memory
                tmp_G_fft.template mult<amrex::RunOn::Device>(tmp_rho_fft, 0, 0, 1);
                amrex::Gpu::streamSynchronize();

                // Backtransform the product of FFT(rho) and FFT(G)
                BL_PROFILE_VAR_START(timer_ffts);
                fft.backward(G_fft_data, tmp_G[local_boxid].dataPtr());
                BL_PROFILE_VAR_STOP(timer_ffts);

                // Normalize, since (FFT + inverse FFT) results in a factor N
                const amrex::Real normalization = 1._rt / realspace_box.numPts();
                tmp_G.mult( normalization );
            } else
#endif
            { // Fully 3D solver on 1 process
            // Create the FFT plans
            BL_PROFILE_VAR_START(timer_plans);
            const amrex::IntVect fft_size = realspace_ba[local_boxid].length();
            ablastr::math::anyfft::FFTplan forward_plan_rho = ablastr::math::anyfft::CreatePlan(
                fft_size, tmp_rho[local_boxid].dataPtr(),
                reinterpret_cast<ablastr::math::anyfft::Complex*>(tmp_rho_fft.dataPtr()),
                ablastr::math::anyfft::direction::R2C, AMREX_SPACEDIM);
            ablastr::math::anyfft::FFTplan forward_plan_G = ablastr::math::anyfft::CreatePlan(
                fft_size, tmp_G[local_boxid].dataPtr(),
                reinterpret_cast<ablastr::math::anyfft::Complex*>(tmp_G_fft.dataPtr()),
                ablastr::math::anyfft::direction::R2C, AMREX_SPACEDIM);
            ablastr::math::anyfft::FFTplan backward_plan = ablastr::math::anyfft::CreatePlan(
                fft_size, tmp_G[local_boxid].dataPtr(),
                reinterpret_cast<ablastr::math::anyfft::Complex*>( tmp_G_fft.dataPtr()),
                ablastr::math::anyfft::direction::C2R, AMREX_SPACEDIM);
            BL_PROFILE_VAR_STOP(timer_plans);

            // Perform forward FFTs of rho and G
            BL_PROFILE_VAR_START(timer_ffts);
            ablastr::math::anyfft::Execute(forward_plan_rho);
            ablastr::math::anyfft::Execute(forward_plan_G);
            BL_PROFILE_VAR_STOP(timer_ffts);

            // Multiply tmp_G_fft and tmp_rho_fft in spectral space
            // Store the result in-place in tmp_G_fft, to save memory
            tmp_G_fft.template mult<amrex::RunOn::Device>(tmp_rho_fft, 0, 0, 1);
            amrex::Gpu::streamSynchronize();

            // Backtransform the product of FFT(rho) and FFT(G)
            BL_PROFILE_VAR_START(timer_ffts);
            ablastr::math::anyfft::Execute(backward_plan);
            BL_PROFILE_VAR_STOP(timer_ffts);

            // Get rid of the plans
            ablastr::math::anyfft::DestroyPlan(forward_plan_G);
            ablastr::math::anyfft::DestroyPlan(forward_plan_rho);
            ablastr::math::anyfft::DestroyPlan(backward_plan);

            // Normalize, since (FFT + inverse FFT) results in a factor N
            const amrex::Real normalization = 1._rt / realspace_box.numPts();
            tmp_G.mult( normalization );
            }
        }
    }

    // Copy from tmp_G to phi
    BL_PROFILE_VAR_START(timer_pcopies);
    phi.ParallelCopy( tmp_G, 0, 0, 1, amrex::IntVect::TheZeroVector(), phi.nGrowVect() );
    BL_PROFILE_VAR_STOP(timer_pcopies);

} //computePhiIGF
} // namespace ablastr::fields
