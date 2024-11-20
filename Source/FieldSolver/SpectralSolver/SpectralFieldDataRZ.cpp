/* Copyright 2019-2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SpectralFieldDataRZ.H"

#include "Utils/WarpXUtil.H"
#include "WarpX.H"

#include <ablastr/math/fft/AnyFFT.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Config.H>

using namespace amrex::literals;

/* \brief Initialize fields in spectral space, and FFT plans
 *
 * \param realspace_ba Box array that corresponds to the decomposition
 *  * of the fields in real space (cell-centered ; includes guard cells only in z)
 * \param k_space Defined the domain of the k space
 * \param dm Indicates which MPI proc owns which box, in realspace_ba
 * \param n_field_required Specifies the number of fields that will be transformed
 * \param n_modes Number of cylindrical modes
 * */
SpectralFieldDataRZ::SpectralFieldDataRZ (const int lev,
                                          amrex::BoxArray const & realspace_ba,
                                          SpectralKSpaceRZ const & k_space,
                                          amrex::DistributionMapping const & dm,
                                          int const n_field_required,
                                          int const n_modes)
    : n_rz_azimuthal_modes(n_modes),
      m_ncomps(2 * n_modes - 1),
      m_n_fields(n_field_required)
{
    amrex::BoxArray const & spectralspace_ba = k_space.spectralspace_ba;

    // Allocate the arrays that contain the fields in spectral space.
    // SpectralField is comparable to a MultiFab but stores complex numbers.
    // This stores all of the transformed fields in one place, with the last dimension
    // being the list of fields, defined by SpectralFieldIndex, for all of the modes.
    // The fields of each mode are grouped together, so that the index of a
    // field for a specific mode is given by field_index + mode*n_fields.
    fields = SpectralField(spectralspace_ba, dm, n_rz_azimuthal_modes*n_field_required, 0);

    // Allocate temporary arrays - in real space and spectral space.
    // These complex arrays will store the data just before/after the z FFT.
    // Note that the realspace_ba should not include the radial guard cells.
    tempHTransformed = SpectralField(realspace_ba, dm, n_rz_azimuthal_modes, 0);
    tmpSpectralField = SpectralField(spectralspace_ba, dm, n_rz_azimuthal_modes, 0);

    // By default, we assume the z FFT is done from/to a nodal grid in real space.
    // If the FFT is performed from/to a cell-centered grid in real space,
    // a correcting "shift" factor must be applied in spectral space.
    zshift_FFTfromCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformFromCellCentered);
    zshift_FFTtoCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformToCellCentered);

    // Allocate and initialize the FFT plans and Hankel transformer.
    forward_plan = FFTplans(spectralspace_ba, dm);
#ifndef AMREX_USE_CUDA
    // The backward plan is not needed with CUDA since it would be the same
    // as the forward plan anyway.
    backward_plan = FFTplans(spectralspace_ba, dm);
#endif
    multi_spectral_hankel_transformer = MultiSpectralHankelTransformer(spectralspace_ba, dm);

    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc.
    for (amrex::MFIter mfi(spectralspace_ba, dm); mfi.isValid(); ++mfi){
        amrex::IntVect grid_size = realspace_ba[mfi].length();
#if defined(AMREX_USE_CUDA)
        // Create cuFFT plan.
        // This is alway complex to complex.
        // This plan is for one azimuthal mode only.
        cufftResult result;
        int fft_length[] = {grid_size[1]};
        int inembed[] = {grid_size[1]};
        int istride = grid_size[0];
        int idist = 1;
        int onembed[] = {grid_size[1]};
        int ostride = grid_size[0];
        int odist = 1;
        int batch = grid_size[0]; // number of ffts
#  ifdef AMREX_USE_FLOAT
        auto cufft_type = CUFFT_C2C;
#  else
        auto cufft_type = CUFFT_Z2Z;
#  endif
        result = cufftPlanMany(&forward_plan[mfi], 1, fft_length, inembed, istride, idist,
                               onembed, ostride, odist, cufft_type, batch);
        if (result != CUFFT_SUCCESS) {
            ablastr::warn_manager::WMRecordWarning("Spectral solver",
                "cufftPlanMany failed!", ablastr::warn_manager::WarnPriority::high);
        }
        // The backward plane is the same as the forward since the direction is passed when executed.
#elif defined(AMREX_USE_HIP)
        const std::size_t fft_length[] = {static_cast<std::size_t>(grid_size[1])};
        const std::size_t stride[] = {static_cast<std::size_t>(grid_size[0])};
        rocfft_plan_description description;
        rocfft_status result;
        result = rocfft_plan_description_create(&description);
        result = rocfft_plan_description_set_data_layout(description,
                                                         rocfft_array_type_complex_interleaved,
                                                         rocfft_array_type_complex_interleaved,
                                                         nullptr, nullptr,
                                                         1, stride, 1,
                                                         1, stride, 1);

        result = rocfft_plan_create(&(forward_plan[mfi]),
                                    rocfft_placement_notinplace,
                                    rocfft_transform_type_complex_forward,
#ifdef AMREX_USE_FLOAT
                                    rocfft_precision_single,
#else
                                    rocfft_precision_double,
#endif
                                    1, fft_length,
                                    grid_size[0], // number of transforms
                                    description);
        if (result != rocfft_status_success) {
            ablastr::warn_manager::WMRecordWarning("Spectral solver",
                "rocfft_plan_create failed!\n",
                ablastr::warn_manager::WarnPriority::high);
        }

        result = rocfft_plan_create(&(backward_plan[mfi]),
                                    rocfft_placement_notinplace,
                                    rocfft_transform_type_complex_inverse,
#ifdef AMREX_USE_FLOAT
                                    rocfft_precision_single,
#else
                                    rocfft_precision_double,
#endif
                                    1, fft_length,
                                    grid_size[0], // number of transforms
                                    description);
        if (result != rocfft_status_success) {
            ablastr::warn_manager::WMRecordWarning("Spectral solver",
                "rocfft_plan_create failed!\n",
                ablastr::warn_manager::WarnPriority::high);
        }

        result = rocfft_plan_description_destroy(description);
        if (result != rocfft_status_success) {
            ablastr::warn_manager::WMRecordWarning("Spectral solver",
                "rocfft_plan_description_destroy failed!\n",
                ablastr::warn_manager::WarnPriority::high);
        }
#else
        // Create FFTW plans.
        fftw_iodim dims[1];
        fftw_iodim howmany_dims[2];
        dims[0].n = grid_size[1];
        dims[0].is = grid_size[0];
        dims[0].os = grid_size[0];
        howmany_dims[0].n = n_rz_azimuthal_modes;
        howmany_dims[0].is = grid_size[0]*grid_size[1];
        howmany_dims[0].os = grid_size[0]*grid_size[1];
        howmany_dims[1].n = grid_size[0];
        howmany_dims[1].is = 1;
        howmany_dims[1].os = 1;
        forward_plan[mfi] =
            // Note that AMReX FAB are Fortran-order.
#ifdef AMREX_USE_FLOAT
            fftwf_plan_guru_dft
#else
            fftw_plan_guru_dft
#endif
                              (1, // int rank
                               dims,
                               2, // int howmany_rank,
                               howmany_dims,
                               reinterpret_cast<
                                ablastr::math::anyfft::Complex*>(tempHTransformed[mfi].dataPtr()), // complex *in
                               reinterpret_cast<
                                ablastr::math::anyfft::Complex*>(tmpSpectralField[mfi].dataPtr()), // complex *out
                               FFTW_FORWARD, // int sign
                               FFTW_ESTIMATE); // unsigned flags
        backward_plan[mfi] =
#ifdef AMREX_USE_FLOAT
            fftwf_plan_guru_dft
#else
            fftw_plan_guru_dft
#endif
                              (1, // int rank
                               dims,
                               2, // int howmany_rank,
                               howmany_dims,
                               reinterpret_cast<
                                ablastr::math::anyfft::Complex*>(tmpSpectralField[mfi].dataPtr()), // complex *in
                               reinterpret_cast<
                                ablastr::math::anyfft::Complex*>(tempHTransformed[mfi].dataPtr()), // complex *out
                               FFTW_BACKWARD, // int sign
                               FFTW_ESTIMATE); // unsigned flags
#endif

        // Create the Hankel transformer for each box.
        amrex::XDim3 const xyzmax = WarpX::UpperCorner(mfi.tilebox(), lev, 0._rt);
        multi_spectral_hankel_transformer[mfi] = SpectralHankelTransformer(grid_size[0], n_rz_azimuthal_modes, xyzmax.x);
    }
}


SpectralFieldDataRZ::~SpectralFieldDataRZ()
{
    if (!fields.empty()){
        for (amrex::MFIter mfi(fields); mfi.isValid(); ++mfi){
#if defined(AMREX_USE_CUDA)
            // Destroy cuFFT plans.
            cufftDestroy(forward_plan[mfi]);
            // cufftDestroy(backward_plan[mfi]); // This was never allocated.
#elif defined(AMREX_USE_HIP)
            rocfft_plan_destroy(forward_plan[mfi]);
            rocfft_plan_destroy(backward_plan[mfi]);
#else
            // Destroy FFTW plans.
#  ifdef AMREX_USE_FLOAT
            fftwf_destroy_plan(forward_plan[mfi]);
            fftwf_destroy_plan(backward_plan[mfi]);
#  else
            fftw_destroy_plan(forward_plan[mfi]);
            fftw_destroy_plan(backward_plan[mfi]);
#  endif
#endif
        }
    }
}

/* \brief Z Transform the FAB to spectral space,
 *  and store the corresponding result internally
 *  (in the spectral field specified by `field_index`)
 *  The input, tempHTransformedSplit, is the complex, Hankel transformed
 *  data, which is stored wih the real and imaginary parts split.
 *  The input should include the imaginary component of mode 0
 *  (even though it is all zeros). */
void
SpectralFieldDataRZ::FABZForwardTransform (amrex::MFIter const & mfi, amrex::Box const & realspace_bx,
                                           amrex::MultiFab const & tempHTransformedSplit,
                                           int const field_index, const bool is_nodal_z)
{
    // Copy the split complex to the interleaved complex.

    amrex::Array4<const amrex::Real> const& split_arr = tempHTransformedSplit[mfi].array();
    amrex::Array4<Complex> const& complex_arr = tempHTransformed[mfi].array();

    int const modes = n_rz_azimuthal_modes;
    ParallelFor(realspace_bx, modes,
    [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
        int const mode_r = 2*mode;
        int const mode_i = 2*mode + 1;
        complex_arr(i,j,k,mode) = Complex{split_arr(i,j,k,mode_r), split_arr(i,j,k,mode_i)};
    });

    // Perform Fourier transform from `tempHTransformed` to `tmpSpectralField`.
#if defined(AMREX_USE_CUDA)
    // Perform Fast Fourier Transform on GPU using cuFFT.
    // Make sure that this is done on the same
    // GPU stream as the above copy.
    cufftResult result;
    cudaStream_t stream = amrex::Gpu::Device::cudaStream();
    cufftSetStream(forward_plan[mfi], stream);
    for (int mode=0 ; mode < n_rz_azimuthal_modes ; mode++) {
#  ifdef AMREX_USE_FLOAT
        result = cufftExecC2C(forward_plan[mfi],
#  else
        result = cufftExecZ2Z(forward_plan[mfi],
#  endif
                              reinterpret_cast<
                                ablastr::math::anyfft::Complex*>(tempHTransformed[mfi].dataPtr(mode)), // Complex *in
                              reinterpret_cast<
                                ablastr::math::anyfft::Complex*>(tmpSpectralField[mfi].dataPtr(mode)), // Complex *out
                              CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            ablastr::warn_manager::WMRecordWarning("Spectral solver",
                "forward transform using cufftExecZ2Z failed!",
                ablastr::warn_manager::WarnPriority::high);
        }
    }
#elif defined(AMREX_USE_HIP)
    rocfft_execution_info execinfo = nullptr;
    rocfft_status result = rocfft_execution_info_create(&execinfo);
    std::size_t buffersize = 0;
    result = rocfft_plan_get_work_buffer_size(forward_plan[mfi], &buffersize);
    void* buffer = amrex::The_Arena()->alloc(buffersize);
    result = rocfft_execution_info_set_work_buffer(execinfo, buffer, buffersize);
    result = rocfft_execution_info_set_stream(execinfo, amrex::Gpu::gpuStream());

    for (int mode=0 ; mode < n_rz_azimuthal_modes ; mode++) {
        void* in_array[] = {(void*)(tempHTransformed[mfi].dataPtr(mode))};
        void* out_array[] = {(void*)(tmpSpectralField[mfi].dataPtr(mode))};
        result = rocfft_execute(forward_plan[mfi], in_array, out_array, execinfo);
        if (result != rocfft_status_success) {
            ablastr::warn_manager::WMRecordWarning("Spectral solver",
                "forward transform using rocfft_execute failed!",
                ablastr::warn_manager::WarnPriority::high);
        }
    }

    amrex::Gpu::streamSynchronize();
    amrex::The_Arena()->free(buffer);
    result = rocfft_execution_info_destroy(execinfo);
#else
#  ifdef AMREX_USE_FLOAT
    fftwf_execute(forward_plan[mfi]);
#  else
    fftw_execute(forward_plan[mfi]);
#  endif
#endif

    // Copy the spectral-space field `tmpSpectralField` to the appropriate
    // index of the FabArray `fields` (specified by `field_index`)
    // and apply correcting shift factor if the real space data comes
    // from a cell-centered grid in real space instead of a nodal grid.
    amrex::Array4<const Complex> const& tmp_arr = tmpSpectralField[mfi].array();
    amrex::Array4<Complex> const& fields_arr = fields[mfi].array();
    Complex const* zshift_arr = zshift_FFTfromCell[mfi].dataPtr();

    // Loop over indices within one box, all components.
    // The fields are organized so that the fields for each mode
    // are grouped together in memory.
    amrex::Box const& spectralspace_bx = tmpSpectralField[mfi].box();
    int const nz = spectralspace_bx.length(1);
    const amrex::Real inv_nz = 1._rt/nz;
    const int n_fields = m_n_fields;

    ParallelFor(spectralspace_bx, modes,
    [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
        Complex spectral_field_value = tmp_arr(i,j,k,mode);
        // Apply proper shift.
        if (!is_nodal_z) { spectral_field_value *= zshift_arr[j]; }
        // Copy field into the correct index.
        int const ic = field_index + mode*n_fields;
        fields_arr(i,j,k,ic) = spectral_field_value*inv_nz;
    });
}

/* \brief Backward Z Transform the data from the fields
 * (in the spectral field specified by `field_index`)
 * to physical space, and return the resulting FArrayBox.
 *  The output, tempHTransformedSplit, is the complex, Hankel transformed
 *  data, which is stored wih the real and imaginary parts split.
 *  The output includes the imaginary component of mode 0
 *  (even though it is all zeros). */
void
SpectralFieldDataRZ::FABZBackwardTransform (amrex::MFIter const & mfi, amrex::Box const & realspace_bx,
                                            int const field_index,
                                            amrex::MultiFab & tempHTransformedSplit,
                                            const bool is_nodal_z)
{
    // Copy the spectral-space field from the appropriate index of the FabArray
    // `fields` (specified by `field_index`) to field `tmpSpectralField`
    // and apply correcting shift factor if the real space data is on
    // a cell-centered grid in real space instead of a nodal grid.
    amrex::Array4<const Complex> const& fields_arr = fields[mfi].array();
    amrex::Array4<Complex> const& tmp_arr = tmpSpectralField[mfi].array();
    Complex const* zshift_arr = zshift_FFTtoCell[mfi].dataPtr();

    // Loop over indices within one box, all components.
    amrex::Box const& spectralspace_bx = tmpSpectralField[mfi].box();

    int const modes = n_rz_azimuthal_modes;
    const int n_fields = m_n_fields;
    ParallelFor(spectralspace_bx, modes,
    [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
        int const ic = field_index + mode*n_fields;
        Complex spectral_field_value = fields_arr(i,j,k,ic);
        // Apply proper shift.
        if (!is_nodal_z) { spectral_field_value *= zshift_arr[j]; }
        // Copy field into the right index.
        tmp_arr(i,j,k,mode) = spectral_field_value;
    });

    // Perform Fourier transform from `tmpSpectralField` to `tempHTransformed`.
#if defined(AMREX_USE_CUDA)
    // Perform Fast Fourier Transform on GPU using cuFFT.
    // Make sure that this is done on the same
    // GPU stream as the above copy.
    cufftResult result;
    cudaStream_t stream = amrex::Gpu::Device::cudaStream();
    cufftSetStream(forward_plan[mfi], stream);
    for (int mode=0 ; mode < n_rz_azimuthal_modes ; mode++) {
#  ifdef AMREX_USE_FLOAT
        result = cufftExecC2C(forward_plan[mfi],
#  else
        result = cufftExecZ2Z(forward_plan[mfi],
#  endif
                              reinterpret_cast<
                                ablastr::math::anyfft::Complex*>(tmpSpectralField[mfi].dataPtr(mode)), // Complex *in
                              reinterpret_cast<
                                ablastr::math::anyfft::Complex*>(tempHTransformed[mfi].dataPtr(mode)), // Complex *out
                              CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS) {
            ablastr::warn_manager::WMRecordWarning("Spectral solver",
                "backwardtransform using cufftExecZ2Z failed!",
                ablastr::warn_manager::WarnPriority::high);
        }
    }
#elif defined(AMREX_USE_HIP)
    rocfft_execution_info execinfo = nullptr;
    rocfft_status result = rocfft_execution_info_create(&execinfo);
    std::size_t buffersize = 0;
    result = rocfft_plan_get_work_buffer_size(forward_plan[mfi], &buffersize);
    void* buffer = amrex::The_Arena()->alloc(buffersize);
    result = rocfft_execution_info_set_work_buffer(execinfo, buffer, buffersize);
    result = rocfft_execution_info_set_stream(execinfo, amrex::Gpu::gpuStream());

    for (int mode=0 ; mode < n_rz_azimuthal_modes ; mode++) {
        void* in_array[] = {(void*)(tmpSpectralField[mfi].dataPtr(mode))};
        void* out_array[] = {(void*)(tempHTransformed[mfi].dataPtr(mode))};
        result = rocfft_execute(backward_plan[mfi], in_array, out_array, execinfo);
        if (result != rocfft_status_success) {
            ablastr::warn_manager::WMRecordWarning("Spectral solver",
                "forward transform using rocfft_execute failed!",
                ablastr::warn_manager::WarnPriority::high);
        }
    }

    amrex::Gpu::streamSynchronize();
    amrex::The_Arena()->free(buffer);
    result = rocfft_execution_info_destroy(execinfo);
#else
#  ifdef AMREX_USE_FLOAT
    fftwf_execute(backward_plan[mfi]);
#  else
    fftw_execute(backward_plan[mfi]);
#  endif
#endif

    // Copy the interleaved complex to the split complex.
    amrex::Array4<amrex::Real> const& split_arr = tempHTransformedSplit[mfi].array();
    amrex::Array4<const Complex> const& complex_arr = tempHTransformed[mfi].array();

    ParallelFor(realspace_bx, modes,
    [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
        int const mode_r = 2*mode;
        int const mode_i = 2*mode + 1;
        split_arr(i,j,k,mode_r) = complex_arr(i,j,k,mode).real();
        split_arr(i,j,k,mode_i) = complex_arr(i,j,k,mode).imag();
    });

}

/* \brief Transform the component `i_comp` of MultiFab `field_mf`
 *  to spectral space, and store the corresponding result internally
 *  (in the spectral field specified by `field_index`) */
void
SpectralFieldDataRZ::ForwardTransform (const int lev,
                                       amrex::MultiFab const & field_mf, int const field_index,
                                       int const i_comp)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    const bool do_costs = WarpXUtilLoadBalance::doCosts(cost, field_mf.boxArray(), field_mf.DistributionMap());

    // Check field index type, in order to apply proper shift in spectral space.
    // Only cell centered in r is supported.
    bool const is_nodal_z = field_mf.is_nodal(1);

    // Create a copy of the input multifab since the shape of field_mf
    // might not be what is needed in transform.
    // For example, with periodic_single_box_fft, field_mf will have guard cells but
    // the transformed array does not.
    // Note that the copy will not include the imaginary part of mode 0 as
    // PhysicalToSpectral_Scalar expects.
    amrex::MultiFab field_mf_copy(tempHTransformed.boxArray(), field_mf.DistributionMap(), m_ncomps, 0);

    // This will hold the Hankel transformed data, with the real and imaginary parts split.
    // A full multifab is created so that each GPU stream has its own temp space.
    amrex::MultiFab tempHTransformedSplit(tempHTransformed.boxArray(), tempHTransformed.DistributionMap(), 2*n_rz_azimuthal_modes, 0);

    // Loop over boxes.
    for (amrex::MFIter mfi(field_mf); mfi.isValid(); ++mfi){

        if (do_costs)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Perform the Hankel transform first.
        // tempHTransformedSplit includes the imaginary component of mode 0.
        // field_mf does not.
        amrex::Box const& realspace_bx = tempHTransformed[mfi].box();

        if ( !(field_mf[mfi].box().contains(field_mf_copy[mfi].box())) ) {
            // If field_mf[mfi] is smaller than field_mf_copy[mfi], then fill field_mf_copy[mfi] with
            // zeros so that all of it is initialized.
            field_mf_copy[mfi].setVal<amrex::RunOn::Device>(0._rt, realspace_bx, 0, m_ncomps);
            }
        field_mf_copy[mfi].copy<amrex::RunOn::Device>(field_mf[mfi], i_comp*m_ncomps, 0, m_ncomps);
        multi_spectral_hankel_transformer[mfi].PhysicalToSpectral_Scalar(field_mf_copy[mfi], tempHTransformedSplit[mfi]);

        FABZForwardTransform(mfi, realspace_bx, tempHTransformedSplit, field_index, is_nodal_z);

        if (do_costs)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

/* \brief Transform the coupled components of MultiFabs `field_mf_r` and `field_mf_t`
 *  to spectral space, and store the corresponding result internally
 *  (in the spectral fields specified by `field_index_r` and `field_index_t`) */
void
SpectralFieldDataRZ::ForwardTransform (const int lev,
                                       amrex::MultiFab const & field_mf_r, int const field_index_r,
                                       amrex::MultiFab const & field_mf_t, int const field_index_t)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    const bool do_costs = WarpXUtilLoadBalance::doCosts(cost, field_mf_r.boxArray(), field_mf_r.DistributionMap());

    // Check field index type, in order to apply proper shift in spectral space.
    // Only cell centered in r is supported.
    const bool is_nodal_z = field_mf_r.is_nodal(1);

    // Create copies of the input multifabs. The copies will include the imaginary part of mode 0.
    // Also note that the Hankel transform will overwrite the copies.
    // Full multifabs are created for the temps so that each GPU stream has its own temp space.
    amrex::MultiFab field_mf_r_copy(tempHTransformed.boxArray(), field_mf_r.DistributionMap(), 2*n_rz_azimuthal_modes, 0);
    amrex::MultiFab field_mf_t_copy(tempHTransformed.boxArray(), field_mf_t.DistributionMap(), 2*n_rz_azimuthal_modes, 0);

    amrex::MultiFab tempHTransformedSplit_p(tempHTransformed.boxArray(), tempHTransformed.DistributionMap(), 2*n_rz_azimuthal_modes, 0);
    amrex::MultiFab tempHTransformedSplit_m(tempHTransformed.boxArray(), tempHTransformed.DistributionMap(), 2*n_rz_azimuthal_modes, 0);

    // Loop over boxes.
    for (amrex::MFIter mfi(field_mf_r); mfi.isValid(); ++mfi){

        if (do_costs)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        amrex::Box const& realspace_bx = tempHTransformed[mfi].box();

        if ( !(field_mf_r[mfi].box().contains(field_mf_r_copy[mfi].box())) ) {
            // If field_mf_r[mfi] is smaller than field_mf_r_copy[mfi], then fill field_mf_r_copy[mfi] with
            // zeros so that all of it is initialized.
            field_mf_r_copy[mfi].setVal<amrex::RunOn::Device>(0._rt, realspace_bx, 0, 2*n_rz_azimuthal_modes);
            field_mf_t_copy[mfi].setVal<amrex::RunOn::Device>(0._rt, realspace_bx, 0, 2*n_rz_azimuthal_modes);
            }
        field_mf_r_copy[mfi].copy<amrex::RunOn::Device>(field_mf_r[mfi], 0, 0, 1); // Real part of mode 0
        field_mf_t_copy[mfi].copy<amrex::RunOn::Device>(field_mf_t[mfi], 0, 0, 1); // Real part of mode 0
        field_mf_r_copy[mfi].setVal<amrex::RunOn::Device>(0._rt, realspace_bx, 1, 1); // Imaginary part of mode 0 (all zero)
        field_mf_t_copy[mfi].setVal<amrex::RunOn::Device>(0._rt, realspace_bx, 1, 1); // Imaginary part of mode 0 (all zero)
        const int ncomps_left = 2 * (n_rz_azimuthal_modes - 1);  // mode zero with an additional imaginary part already handled
        field_mf_r_copy[mfi].copy<amrex::RunOn::Device>(field_mf_r[mfi], 1, 2, ncomps_left);
        field_mf_t_copy[mfi].copy<amrex::RunOn::Device>(field_mf_t[mfi], 1, 2, ncomps_left);

        // Perform the Hankel transform first.
        multi_spectral_hankel_transformer[mfi].PhysicalToSpectral_Vector(realspace_bx,
                                                           field_mf_r_copy[mfi], field_mf_t_copy[mfi],
                                                           tempHTransformedSplit_p[mfi], tempHTransformedSplit_m[mfi]);

        FABZForwardTransform(mfi, realspace_bx, tempHTransformedSplit_p, field_index_r, is_nodal_z);
        FABZForwardTransform(mfi, realspace_bx, tempHTransformedSplit_m, field_index_t, is_nodal_z);

        if (do_costs)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

/* \brief Transform spectral field specified by `field_index` back to
 * real space, and store it in the component `i_comp` of `field_mf` */
void
SpectralFieldDataRZ::BackwardTransform (const int lev,
                                        amrex::MultiFab& field_mf, int const field_index,
                                        int const i_comp)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    const bool do_costs = WarpXUtilLoadBalance::doCosts(cost, field_mf.boxArray(), field_mf.DistributionMap());

    // Check field index type, in order to apply proper shift in spectral space.
    const bool is_nodal_z = field_mf.is_nodal(1);

    // A full multifab is created so that each GPU stream has its own temp space.
    amrex::MultiFab tempHTransformedSplit(tempHTransformed.boxArray(), tempHTransformed.DistributionMap(), 2*n_rz_azimuthal_modes, 0);

    // Create a temporary to hold the inverse Hankel transform field.
    // This allows the final result to have a different shape than the transformed field.
    amrex::MultiFab field_mf_copy(tempHTransformed.boxArray(), tempHTransformed.DistributionMap(), m_ncomps, 0);

    // Loop over boxes.
    for (amrex::MFIter mfi(field_mf); mfi.isValid(); ++mfi){

        if (do_costs)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        amrex::Box realspace_bx = tempHTransformed[mfi].box();

        FABZBackwardTransform(mfi, realspace_bx, field_index, tempHTransformedSplit, is_nodal_z);

        // Perform the Hankel inverse transform last.
        // tempHTransformedSplit includes the imaginary component of mode 0.
        // field_mf does not.
        multi_spectral_hankel_transformer[mfi].SpectralToPhysical_Scalar(tempHTransformedSplit[mfi], field_mf_copy[mfi]);

        amrex::Array4<amrex::Real> const & field_mf_array = field_mf[mfi].array();
        amrex::Array4<amrex::Real> const & field_mf_copy_array = field_mf_copy[mfi].array();

        // The box will be extended to include the guards cells below the axis
        // so that they can be filled in. This will not be a simple copy of the
        // fields since the signs will change when there is anti-symmetry.
        amrex::Box const& realspace_bx_with_guards = field_mf[mfi].box();
        const int* lo_with_guards = realspace_bx_with_guards.loVect();

        // Grow the lower side of realspace_bx by the number of guard cells.
        // This assumes that the box extends over the full extent radially, so
        // lo_with_guards[0] will be equal to minus the number of guard cells radially.
        const int nguard_r = -lo_with_guards[0];
        realspace_bx.growLo(0, nguard_r);

        // Get the intersection of the two boxes in case the field_mf has fewer z-guard cells
        realspace_bx &= realspace_bx_with_guards;

        ParallelFor(realspace_bx, m_ncomps,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int icomp) noexcept {
            int ii = i;
            amrex::Real sign = +1._rt;
            if (i < 0) {
                ii = -i - 1;
                if (icomp == 0) {
                    // Mode zero is symmetric
                    sign = +1._rt;
                } else {
                    // Odd modes are anti-symmetric
                    const auto imode = (icomp + 1)/2;
                    sign = static_cast<amrex::Real>(std::pow(-1._rt, imode));
                }
            }
            const auto ic = icomp + i_comp;
            field_mf_array(i,j,k,ic) = sign*field_mf_copy_array(ii,j,k,icomp);
        });

        if (do_costs)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

/* \brief Transform spectral fields specified by `field_index_r` and
 * `field_index_t` back to real space, and store them in `field_mf_r` and `field_mf_t` */
void
SpectralFieldDataRZ::BackwardTransform (const int lev,
                                        amrex::MultiFab& field_mf_r, int const field_index_r,
                                        amrex::MultiFab& field_mf_t, int const field_index_t)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    const bool do_costs = WarpXUtilLoadBalance::doCosts(cost, field_mf_r.boxArray(), field_mf_r.DistributionMap());

    // Check field index type, in order to apply proper shift in spectral space.
    bool const is_nodal_z = field_mf_r.is_nodal(1);

    // Full multifabs are created for the temps so that each GPU stream has its own temp space.
    amrex::MultiFab tempHTransformedSplit_p(tempHTransformed.boxArray(), tempHTransformed.DistributionMap(), 2*n_rz_azimuthal_modes, 0);
    amrex::MultiFab tempHTransformedSplit_m(tempHTransformed.boxArray(), tempHTransformed.DistributionMap(), 2*n_rz_azimuthal_modes, 0);

    // Create copies of the input multifabs. The copies will include the imaginary part of mode 0.
    amrex::MultiFab field_mf_r_copy(tempHTransformed.boxArray(), field_mf_r.DistributionMap(), 2*n_rz_azimuthal_modes, 0);
    amrex::MultiFab field_mf_t_copy(tempHTransformed.boxArray(), field_mf_t.DistributionMap(), 2*n_rz_azimuthal_modes, 0);

    // Loop over boxes.
    for (amrex::MFIter mfi(field_mf_r); mfi.isValid(); ++mfi){

        if (do_costs)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        amrex::Box realspace_bx = tempHTransformed[mfi].box();

        FABZBackwardTransform(mfi, realspace_bx, field_index_r, tempHTransformedSplit_p, is_nodal_z);
        FABZBackwardTransform(mfi, realspace_bx, field_index_t, tempHTransformedSplit_m, is_nodal_z);

        // Perform the Hankel inverse transform last.
        // tempHTransformedSplit includes the imaginary component of mode 0.
        // field_mf_[ri] do not.
        multi_spectral_hankel_transformer[mfi].SpectralToPhysical_Vector(realspace_bx,
                                                           tempHTransformedSplit_p[mfi], tempHTransformedSplit_m[mfi],
                                                           field_mf_r_copy[mfi], field_mf_t_copy[mfi]);

        amrex::Array4<amrex::Real> const & field_mf_r_array = field_mf_r[mfi].array();
        amrex::Array4<amrex::Real> const & field_mf_t_array = field_mf_t[mfi].array();
        amrex::Array4<amrex::Real> const & field_mf_r_copy_array = field_mf_r_copy[mfi].array();
        amrex::Array4<amrex::Real> const & field_mf_t_copy_array = field_mf_t_copy[mfi].array();

        // The box will be extended to include the guards cells below the axis
        // so that they can be filled in. This will not be a simple copy of the
        // fields since the signs will change when there is anti-symmetry.
        amrex::Box const& realspace_bx_with_guards = field_mf_r[mfi].box();
        const int* lo_with_guards = realspace_bx_with_guards.loVect();

        // Grow the lower side of realspace_bx by the number of guard cells.
        // This assumes that the box extends over the full extent radially, so
        // lo_with_guards[0] will be equal to minus the number of guard cells radially.
        const int nguard_r = -lo_with_guards[0];
        realspace_bx.growLo(0, nguard_r);

        // Get the intersection of the two boxes in case the field_mf has fewer z-guard cells
        realspace_bx &= realspace_bx_with_guards;

        ParallelFor(realspace_bx, m_ncomps,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int icomp) noexcept {
            int ii = i;
            amrex::Real sign = +1._rt;
            if (i < 0) {
                ii = -i - 1;
                if (icomp == 0) {
                    // Mode zero is anti-symmetric
                    sign = -1._rt;
                } else {
                    // Even modes are anti-symmetric
                    const int imode = (icomp + 1)/2;
                    sign = static_cast<amrex::Real>(std::pow(-1._rt, imode+1));
                }
            }
            if (icomp == 0) {
                // mode zero
                field_mf_r_array(i,j,k,icomp) = sign*field_mf_r_copy_array(ii,j,k,icomp);
                field_mf_t_array(i,j,k,icomp) = sign*field_mf_t_copy_array(ii,j,k,icomp);
            } else {
                field_mf_r_array(i,j,k,icomp) = sign*field_mf_r_copy_array(ii,j,k,icomp+1);
                field_mf_t_array(i,j,k,icomp) = sign*field_mf_t_copy_array(ii,j,k,icomp+1);
            }
        });

        if (do_costs)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

}

/* \brief Initialize arrays used for filtering */
void
SpectralFieldDataRZ::InitFilter (amrex::IntVect const & filter_npass_each_dir, bool const compensation,
                                 SpectralKSpaceRZ const & k_space)
{
    binomialfilter = BinomialFilter(multi_spectral_hankel_transformer.boxArray(),
                                    multi_spectral_hankel_transformer.DistributionMap());

    auto const & dx = k_space.getCellSize();
    auto const & kz = k_space.getKzArray();

    for (amrex::MFIter mfi(binomialfilter); mfi.isValid(); ++mfi){
        binomialfilter[mfi].InitFilterArray(multi_spectral_hankel_transformer[mfi].getKrArray(),
                                            kz[mfi], dx, filter_npass_each_dir, compensation);
    }
}

/* \brief Apply K-space filtering on a scalar */
void
SpectralFieldDataRZ::ApplyFilter (const int lev, int const field_index)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    const bool do_costs = WarpXUtilLoadBalance::doCosts(cost, binomialfilter.boxArray(), binomialfilter.DistributionMap());

    for (amrex::MFIter mfi(binomialfilter); mfi.isValid(); ++mfi){

        if (do_costs)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        auto const & filter_r = binomialfilter[mfi].getFilterArrayR();
        auto const & filter_z = binomialfilter[mfi].getFilterArrayZ();
        auto const & filter_r_arr = filter_r.dataPtr();
        auto const & filter_z_arr = filter_z.dataPtr();

        amrex::Array4<Complex> const& fields_arr = fields[mfi].array();

        int const modes = n_rz_azimuthal_modes;
        const int n_fields = m_n_fields;

        amrex::Box const& spectralspace_bx = fields[mfi].box();
        int const nr = spectralspace_bx.length(0);

        ParallelFor(spectralspace_bx, modes,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
            int const ic = field_index + mode*n_fields;
            int const ir = i + nr*mode;
            fields_arr(i,j,k,ic) *= filter_r_arr[ir]*filter_z_arr[j];
        });

        if (do_costs)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

/* \brief Apply K-space filtering on a vector */
void
SpectralFieldDataRZ::ApplyFilter (const int lev, int const field_index1,
                                  int const field_index2, int const field_index3)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
    const bool do_costs = WarpXUtilLoadBalance::doCosts(cost, binomialfilter.boxArray(), binomialfilter.DistributionMap());

    for (amrex::MFIter mfi(binomialfilter); mfi.isValid(); ++mfi){

        if (do_costs)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        auto const & filter_r = binomialfilter[mfi].getFilterArrayR();
        auto const & filter_z = binomialfilter[mfi].getFilterArrayZ();
        auto const & filter_r_arr = filter_r.dataPtr();
        auto const & filter_z_arr = filter_z.dataPtr();

        amrex::Array4<Complex> const& fields_arr = fields[mfi].array();

        int const modes = n_rz_azimuthal_modes;
        const int n_fields = m_n_fields;

        amrex::Box const& spectralspace_bx = fields[mfi].box();
        int const nr = spectralspace_bx.length(0);

        ParallelFor(spectralspace_bx, modes,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
            int const ic1 = field_index1 + mode*n_fields;
            int const ic2 = field_index2 + mode*n_fields;
            int const ic3 = field_index3 + mode*n_fields;
            int const ir = i + nr*mode;
            fields_arr(i,j,k,ic1) *= filter_r_arr[ir]*filter_z_arr[j];
            fields_arr(i,j,k,ic2) *= filter_r_arr[ir]*filter_z_arr[j];
            fields_arr(i,j,k,ic3) *= filter_r_arr[ir]*filter_z_arr[j];
        });

        if (do_costs)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}
