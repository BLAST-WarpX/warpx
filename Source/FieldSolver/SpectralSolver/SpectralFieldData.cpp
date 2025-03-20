/* Copyright 2019 Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SpectralFieldData.H"

#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXUtil.H"
#include "WarpX.H"

#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Dim3.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_PODVector.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#if WARPX_USE_FFT

using namespace amrex;

SpectralFieldIndex::SpectralFieldIndex (const bool update_with_rho,
                                        const bool time_averaging,
                                        const JInTime J_in_time,
                                        const RhoInTime rho_in_time,
                                        const bool dive_cleaning,
                                        const bool divb_cleaning,
                                        const bool pml,
                                        const bool pml_rz)
{
    // TODO Use these to allocate rho_old, rho_new, F, and G only when needed
    amrex::ignore_unused(update_with_rho);

    int c = 0;

    if (!pml)
    {
        Ex = c++; Ey = c++; Ez = c++;
        Bx = c++; By = c++; Bz = c++;

        // TODO Allocate rho_old and rho_new only when needed

        // Reuse data corresponding to index Bx = 3 to avoid storing extra memory
        divE = 3;

        if (time_averaging)
        {
            Ex_avg = c++; Ey_avg = c++; Ez_avg = c++;
            Bx_avg = c++; By_avg = c++; Bz_avg = c++;
        }

        if (dive_cleaning) { F = c++; }

        if (divb_cleaning) { G = c++; }

        if (J_in_time == JInTime::Constant)
        {
            Jx_mid = c++; Jy_mid = c++; Jz_mid = c++;
        }
        else if (J_in_time == JInTime::Linear)
        {
            Jx_old = c++; Jy_old = c++; Jz_old = c++;
            Jx_new = c++; Jy_new = c++; Jz_new = c++;
        }

        if (rho_in_time == RhoInTime::Constant)
        {
            rho_mid = c++;
        }
        else if (rho_in_time == RhoInTime::Linear)
        {
            rho_old = c++;
            rho_new = c++;
        }

        if (pml_rz)
        {
            Er_pml = c++; Et_pml = c++;
            Br_pml = c++; Bt_pml = c++;
        }

    }
    else // PML
    {
        Exy = c++; Exz = c++; Eyx = c++; Eyz = c++; Ezx = c++; Ezy = c++;
        Bxy = c++; Bxz = c++; Byx = c++; Byz = c++; Bzx = c++; Bzy = c++;

        if (dive_cleaning)
        {
            Exx = c++; Eyy = c++; Ezz = c++;
            Fx  = c++; Fy  = c++; Fz  = c++;
        }

        if (divb_cleaning)
        {
            Bxx = c++; Byy = c++; Bzz = c++;
            Gx  = c++; Gy  = c++; Gz  = c++;
        }
    }

    // This is the number of arrays that will be actually allocated in spectral space
    n_fields = c;
}

/* \brief Initialize fields in spectral space, and FFT plans */
SpectralFieldData::SpectralFieldData( const int lev,
                                      const amrex::BoxArray& realspace_ba,
                                      const SpectralKSpace& k_space,
                                      const amrex::DistributionMapping& dm,
                                      const int n_field_required,
                                      const bool periodic_single_box):
    m_periodic_single_box{periodic_single_box}
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    const bool do_costs = WarpXUtilLoadBalance::doCosts(cost, realspace_ba, dm);

    const BoxArray& spectralspace_ba = k_space.spectralspace_ba;

    // Allocate the arrays that contain the fields in spectral space
    // (one component per field)
    fields = SpectralField(spectralspace_ba, dm, n_field_required, 0);

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    tmpRealField = MultiFab(realspace_ba, dm, 1, 0);
    tmpSpectralField = SpectralField(spectralspace_ba, dm, 1, 0);

    // By default, we assume the FFT is done from/to a nodal grid in real space
    // If the FFT is performed from/to a cell-centered grid in real space,
    // a correcting "shift" factor must be applied in spectral space.
    shift0_FFTfromCell = k_space.getSpectralShiftFactor(dm, 0,
                                    ShiftType::TransformFromCellCentered);
    shift0_FFTtoCell = k_space.getSpectralShiftFactor(dm, 0,
                                    ShiftType::TransformToCellCentered);
#if AMREX_SPACEDIM > 1
    shift1_FFTfromCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformFromCellCentered);
    shift1_FFTtoCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformToCellCentered);
#if AMREX_SPACEDIM > 2
    shift2_FFTfromCell = k_space.getSpectralShiftFactor(dm, 2,
                                    ShiftType::TransformFromCellCentered);
    shift2_FFTtoCell = k_space.getSpectralShiftFactor(dm, 2,
                                    ShiftType::TransformToCellCentered);
#endif
#endif

    // Allocate and initialize the FFT plans
    forward_plan = ablastr::math::anyfft::FFTplans(spectralspace_ba, dm);
    backward_plan = ablastr::math::anyfft::FFTplans(spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc
    for ( MFIter mfi(spectralspace_ba, dm); mfi.isValid(); ++mfi ){
        if (do_costs)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Note: the size of the real-space box and spectral-space box
        // differ when using real-to-complex FFT. When initializing
        // the FFT plan, the valid dimensions are those of the real-space box.
        const IntVect fft_size = realspace_ba[mfi].length();

        forward_plan[mfi] = ablastr::math::anyfft::CreatePlan(
            fft_size, tmpRealField[mfi].dataPtr(),
            reinterpret_cast<ablastr::math::anyfft::Complex*>( tmpSpectralField[mfi].dataPtr()),
            ablastr::math::anyfft::direction::R2C, AMREX_SPACEDIM);

        backward_plan[mfi] = ablastr::math::anyfft::CreatePlan(
            fft_size, tmpRealField[mfi].dataPtr(),
            reinterpret_cast<ablastr::math::anyfft::Complex*>( tmpSpectralField[mfi].dataPtr()),
            ablastr::math::anyfft::direction::C2R, AMREX_SPACEDIM);

        if (do_costs)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}


SpectralFieldData::~SpectralFieldData()
{
    if (!tmpRealField.empty()){
        for ( MFIter mfi(tmpRealField); mfi.isValid(); ++mfi ){
            ablastr::math::anyfft::DestroyPlan(forward_plan[mfi]);
            ablastr::math::anyfft::DestroyPlan(backward_plan[mfi]);
        }
    }
}

/* \brief Transform the component `i_comp` of MultiFab `mf`
 *  to spectral space, and store the corresponding result internally
 *  (in the spectral field specified by `field_index`) */
void
SpectralFieldData::ForwardTransform (const int lev,
                                     const MultiFab& mf, const int field_index,
                                     const int i_comp)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    const bool do_costs = WarpXUtilLoadBalance::doCosts(cost, mf.boxArray(), mf.DistributionMap());

    // Check field index type, in order to apply proper shift in spectral space
    const bool is_nodal_0 = mf.is_nodal(0);
#if AMREX_SPACEDIM > 1
    const bool is_nodal_1 = mf.is_nodal(1);
#if AMREX_SPACEDIM > 2
    const bool is_nodal_2 = mf.is_nodal(2);
#endif
#endif

    // Loop over boxes
    // Note: we do NOT OpenMP parallelize here, since we use OpenMP threads for
    //       the FFTs on each box!
    for ( MFIter mfi(mf); mfi.isValid(); ++mfi ){
        if (do_costs)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Copy the real-space field `mf` to the temporary field `tmpRealField`
        // This ensures that all fields have the same number of points
        // before the Fourier transform.
        // As a consequence, the copy discards the *last* point of `mf`
        // in any direction that has *nodal* index type.
        {
            Box realspace_bx;
            if (m_periodic_single_box) {
                realspace_bx = mfi.validbox(); // Discard guard cells
            } else {
                realspace_bx = mf[mfi].box(); // Keep guard cells
            }
            realspace_bx.enclosedCells(); // Discard last point in nodal direction
            AMREX_ALWAYS_ASSERT( realspace_bx.contains(tmpRealField[mfi].box()) );
            const Array4<const Real> mf_arr = mf[mfi].array();
            const Array4<Real> tmp_arr = tmpRealField[mfi].array();
            ParallelFor( tmpRealField[mfi].box(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                tmp_arr(i,j,k) = mf_arr(i,j,k,i_comp);
            });
        }

        // Perform Fourier transform from `tmpRealField` to `tmpSpectralField`
        ablastr::math::anyfft::Execute(forward_plan[mfi]);

        // Copy the spectral-space field `tmpSpectralField` to the appropriate
        // index of the FabArray `fields` (specified by `field_index`)
        // and apply correcting shift factor if the real space data comes
        // from a cell-centered grid in real space instead of a nodal grid.
        {
            const Array4<Complex> fields_arr = SpectralFieldData::fields[mfi].array();
            const Array4<const Complex> tmp_arr = tmpSpectralField[mfi].array();

            const Complex* shift0_arr = shift0_FFTfromCell[mfi].dataPtr();
#if AMREX_SPACEDIM > 1
            const Complex* shift1_arr = shift1_FFTfromCell[mfi].dataPtr();
#if AMREX_SPACEDIM > 2
            const Complex* shift2_arr = shift2_FFTfromCell[mfi].dataPtr();
#endif
#endif
            // Loop over indices within one box
            const Box spectralspace_bx = tmpSpectralField[mfi].box();

            ParallelFor( spectralspace_bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Complex spectral_field_value = tmp_arr(i,j,k);
                // Apply proper shift in each dimension
                if (!is_nodal_0) { spectral_field_value *= shift0_arr[i]; }
#if AMREX_SPACEDIM > 1
                if (!is_nodal_1) { spectral_field_value *= shift1_arr[j]; }
#if AMREX_SPACEDIM > 2
                if (!is_nodal_2) { spectral_field_value *= shift2_arr[k]; }
#endif
#endif
                // Copy field into the right index
                fields_arr(i,j,k,field_index) = spectral_field_value;
            });
        }

        if (do_costs)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}


/* \brief Transform spectral field specified by `field_index` back to
 * real space, and store it in the component `i_comp` of `mf` */
void
SpectralFieldData::BackwardTransform (const int lev,
                                      MultiFab& mf,
                                      const int field_index,
                                      const amrex::IntVect& fill_guards,
                                      const int i_comp)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    const bool do_costs = WarpXUtilLoadBalance::doCosts(cost, mf.boxArray(), mf.DistributionMap());

    // Check field index type, in order to apply proper shift in spectral space
    const bool is_nodal_0 = mf.is_nodal(0);
    const bool is_nodal_1 = (AMREX_SPACEDIM > 1 ? mf.is_nodal(1) : 0);
    const bool is_nodal_2 = (AMREX_SPACEDIM > 2 ? mf.is_nodal(2) : 0);

    // Numbers of guard cells
    const amrex::IntVect& mf_ng = mf.nGrowVect();

    // Loop over boxes
    // Note: we do NOT OpenMP parallelize here, since we use OpenMP threads for
    //       the iFFTs on each box!
    for ( MFIter mfi(mf); mfi.isValid(); ++mfi ){
        if (do_costs)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Copy the spectral-space field `tmpSpectralField` to the appropriate
        // field (specified by the input argument field_index)
        // and apply correcting shift factor if the field is to be transformed
        // to a cell-centered grid in real space instead of a nodal grid.
        {
            const Array4<const Complex> field_arr = SpectralFieldData::fields[mfi].array();
            const Array4<Complex> tmp_arr = tmpSpectralField[mfi].array();
            const Complex* shift0_arr = shift0_FFTtoCell[mfi].dataPtr();
#if AMREX_SPACEDIM > 1
            const Complex* shift1_arr = shift1_FFTtoCell[mfi].dataPtr();
#if AMREX_SPACEDIM > 2
            const Complex* shift2_arr = shift2_FFTtoCell[mfi].dataPtr();
#endif
#endif
            // Loop over indices within one box
            const Box spectralspace_bx = tmpSpectralField[mfi].box();

            ParallelFor( spectralspace_bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Complex spectral_field_value = field_arr(i,j,k,field_index);
                // Apply proper shift in each dimension
                if (!is_nodal_0) { spectral_field_value *= shift0_arr[i]; }
#if AMREX_SPACEDIM > 1
                if (!is_nodal_1) { spectral_field_value *= shift1_arr[j]; }
#if AMREX_SPACEDIM > 2
                if (!is_nodal_2) { spectral_field_value *= shift2_arr[k]; }
#endif
#endif
                // Copy field into temporary array
                tmp_arr(i,j,k) = spectral_field_value;
            });
        }

        // Perform Fourier transform from `tmpSpectralField` to `tmpRealField`
        ablastr::math::anyfft::Execute(backward_plan[mfi]);

        // Copy the temporary field tmpRealField to the real-space field mf and
        // normalize, dividing by N, since (FFT + inverse FFT) results in a factor N
        {
            amrex::Box mf_box = (m_periodic_single_box) ? mfi.validbox() : mfi.fabbox();
            const amrex::Array4<amrex::Real> mf_arr = mf[mfi].array();
            const amrex::Array4<const amrex::Real> tmp_arr = tmpRealField[mfi].array();

            const amrex::Real inv_N = 1._rt / tmpRealField[mfi].box().numPts();

            // Total number of cells, including ghost cells (nj represents ny in 3D and nz in 2D)
            const int ni = mf_box.length(0);
            const int nj = (AMREX_SPACEDIM > 1 ? mf_box.length(1) : 1);
            const int nk = (AMREX_SPACEDIM > 2 ? mf_box.length(2) : 1);

            const int si = (is_nodal_0) ? 1 : 0;
            const int sj = (is_nodal_1) ? 1 : 0;
            const int sk = (is_nodal_2) ? 1 : 0;

            // Lower bound of the box (lo_j represents lo_y in 3D and lo_z in 2D)
            const int lo_i = amrex::lbound(mf_box).x;
            const int lo_j = (AMREX_SPACEDIM > 1 ? amrex::lbound(mf_box).y : 0);
            const int lo_k = (AMREX_SPACEDIM > 2 ? amrex::lbound(mf_box).z : 0);

            // If necessary, do not fill the guard cells
            // (shrink box by passing negative number of cells)
            if (!m_periodic_single_box)
            {
                for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
                {
                    if ((fill_guards[dir]) == 0) { mf_box.grow(dir, -mf_ng[dir]); }
                }
            }

            // Loop over cells within full box, including ghost cells
            ParallelFor(mf_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                // Assume periodicity and set the last outer guard cell equal to the first one:
                // this is necessary in order to get the correct value along a nodal direction,
                // because the last point along a nodal direction is always discarded when FFTs
                // are computed, as the real-space box is always cell-centered.
                const int ii = (i == lo_i + ni - si) ? lo_i : i;
                const int jj = (j == lo_j + nj - sj) ? lo_j : j;
                const int kk = (k == lo_k + nk - sk) ? lo_k : k;
                // Copy and normalize field
                mf_arr(i,j,k,i_comp) = inv_N * tmp_arr(ii,jj,kk);
            });
        }

        if (do_costs)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#endif // WARPX_USE_FFT
