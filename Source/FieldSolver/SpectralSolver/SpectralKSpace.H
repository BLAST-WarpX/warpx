/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPECTRAL_K_SPACE_H_
#define WARPX_SPECTRAL_K_SPACE_H_

#include "SpectralKSpace_fwd.H"

#include "Utils/WarpX_Complex.H"

#include <ablastr/utils/Enums.H>

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_LayoutData.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

// `KVectorComponent` and `SpectralShiftFactor` hold one 1D array
// ("DeviceVector") for each box ("LayoutData"). The arrays are
// only allocated if the corresponding box is owned by the local MPI rank.
using RealKVector = amrex::Gpu::DeviceVector<amrex::Real>;
using KVectorComponent = amrex::LayoutData< RealKVector >;
using SpectralShiftFactor = amrex::LayoutData<
                           amrex::Gpu::DeviceVector<Complex> >;

// Indicate the type of correction "shift" factor to apply
// when the FFT is performed from/to a cell-centered grid in real space.
struct ShiftType {
    enum{ TransformFromCellCentered=0, TransformToCellCentered=1 };
};

/**
 * \brief Class that represents the spectral space.
 *
 * (Contains info about the size of the spectral space corresponding
 * to each box in `realspace_ba`, as well as the value of the
 * corresponding k coordinates)
 */
class SpectralKSpace
{
    public:
        amrex::BoxArray spectralspace_ba;
        SpectralKSpace() : dx(amrex::RealVect::Zero) {}

        SpectralKSpace( const amrex::BoxArray& realspace_ba,
                        const amrex::DistributionMapping& dm,
                        amrex::RealVect realspace_dx );

        KVectorComponent getKComponent(
            const amrex::DistributionMapping& dm,
            const amrex::BoxArray& realspace_ba,
            int i_dim, bool only_positive_k ) const;

        KVectorComponent getModifiedKComponent (
            const amrex::DistributionMapping& dm,
            int i_dim,
            int n_order,
            ablastr::utils::enums::GridType grid_type) const;

        SpectralShiftFactor getSpectralShiftFactor(
            const amrex::DistributionMapping& dm, int i_dim,
            int shift_type ) const;

    protected:
        amrex::Array<KVectorComponent, AMREX_SPACEDIM> k_vec;
        // 3D: k_vec is an Array of 3 components, corresponding to kx, ky, kz
        // 2D: k_vec is an Array of 2 components, corresponding to kx, kz
        amrex::RealVect dx;
};

#endif
