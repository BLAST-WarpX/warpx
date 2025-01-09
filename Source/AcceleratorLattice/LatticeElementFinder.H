/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ACCELERATORLATTICE_LATTICEELEMENTS_LATTICEELEMENTFINDER_H_
#define WARPX_ACCELERATORLATTICE_LATTICEELEMENTS_LATTICEELEMENTFINDER_H_

#include "LatticeElements/HardEdgedQuadrupole.H"
#include "LatticeElements/HardEdgedPlasmaLens.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"

#include <AMReX_REAL.H>
#include <AMReX_GpuContainers.H>

class AcceleratorLattice;
struct LatticeElementFinderDevice;

// Instances of the LatticeElementFinder class are saved in the AcceleratorLattice class
// as the objects in a LayoutData.
// The LatticeElementFinder handles the lookup needed to find the lattice elements at
// particle locations.

struct LatticeElementFinder
{

    /**
     * \brief Initialize the element finder at the level and grid
     *
     * @param[in] lev the refinement level
     * @param[in] gamma_boost the Lorentz factor of the boosted frame
     * @param[in] a_mfi specifies the grid where the finder is defined
     * @param[in] accelerator_lattice a reference to the accelerator lattice at the refinement level
     */
    void InitElementFinder (int lev, amrex::Real gamma_boost,
                            amrex::MFIter const& a_mfi,
                            AcceleratorLattice const& accelerator_lattice);

    /**
     * \brief Allocate the index lookup tables for each element type
     *
     * @param[in] accelerator_lattice a reference to the accelerator lattice at the refinement level
     */
    void AllocateIndices (AcceleratorLattice const& accelerator_lattice);

    /**
     * \brief Update the index lookup tables for each element type, filling in the values
     *
     * @param[in] lev the refinement level
     * @param[in] a_mfi specifies the grid where the finder is defined
     * @param[in] accelerator_lattice a reference to the accelerator lattice at the refinement level
     */
    void UpdateIndices (int lev, amrex::MFIter const& a_mfi,
                        AcceleratorLattice const& accelerator_lattice);

    /* Define the location and size of the index lookup table */
    /* Use the type Real to be consistent with the way the main grid is defined */
    int m_nz;
    amrex::Real m_zmin;
    amrex::Real m_dz;

    /* Parameters needed for the Lorentz transforms into and out of the boosted frame */
    /* The time for m_time is consistent with the main time variable */
    amrex::ParticleReal m_gamma_boost;
    amrex::ParticleReal m_uz_boost;
    amrex::Real m_time;

    /**
     * \brief Get the device level instance associated with this instance
     *
     * @param[in] a_pti specifies the grid where the finder is defined
     * @param[in] a_offset particle index offset needed to access particle info
     * @param[in] accelerator_lattice a reference to the accelerator lattice at the refinement level
     */
    [[nodiscard]] LatticeElementFinderDevice GetFinderDeviceInstance (
        WarpXParIter const& a_pti, int a_offset, AcceleratorLattice const& accelerator_lattice) const;

    /* The index lookup tables for each lattice element type */
    amrex::Gpu::DeviceVector<int> d_quad_indices;
    amrex::Gpu::DeviceVector<int> d_plasmalens_indices;

    /**
     * \brief Fill in the index lookup tables
     * This loops over the grid (in z) and finds the lattice element closest to each grid point
     *
     * @param[in] zs list of the starts of the lattice elements
     * @param[in] ze list of the ends of the lattice elements
     * @param[in] indices the index lookup table to be filled in
     */
    void setup_lattice_indices (amrex::Gpu::DeviceVector<amrex::ParticleReal> const & zs,
                                amrex::Gpu::DeviceVector<amrex::ParticleReal> const & ze,
                                amrex::Gpu::DeviceVector<int> & indices) const;
};

/**
 * \brief The lattice element finder class that can be trivially copied to the device.
 * This only has simple data and pointers.
 */
struct LatticeElementFinderDevice
{

    /**
     * \brief Initialize the data needed to do the lookups
     *
     * @param[in] a_pti specifies the grid where the finder is defined
     * @param[in] a_offset particle index offset needed to access particle info
     * @param[in] accelerator_lattice a reference to the accelerator lattice at the refinement level
     * @param[in] h_finder The host level instance of the element finder that this is associated with
     */
    void
    InitLatticeElementFinderDevice (WarpXParIter const& a_pti, int a_offset,
                                    AcceleratorLattice const& accelerator_lattice,
                                    LatticeElementFinder const & h_finder);

    /* Size and location of the index lookup table */
    amrex::Real m_zmin;
    amrex::Real m_dz;
    amrex::Real m_dt;

    /* Parameters needed for the Lorentz transforms into and out of the boosted frame */
    amrex::ParticleReal m_gamma_boost;
    amrex::ParticleReal m_uz_boost;
    amrex::Real m_time;

    GetParticlePosition<PIdx> m_get_position;
    const amrex::ParticleReal* AMREX_RESTRICT m_ux = nullptr;
    const amrex::ParticleReal* AMREX_RESTRICT m_uy = nullptr;
    const amrex::ParticleReal* AMREX_RESTRICT m_uz = nullptr;

    /* Device level instances for each lattice element type */
    HardEdgedQuadrupoleDevice d_quad;
    HardEdgedPlasmaLensDevice d_plasmalens;

    /* Device level index lookup tables for each element type */
    int const* d_quad_indices_arr = nullptr;
    int const* d_plasmalens_indices_arr = nullptr;

    /**
     * \brief Gather the field for the particle from the lattice elements
     *
     * @param[in] i the particle index
     * @param[out] field_Ex,field_Ey,field_Ez,field_Bx,field_By,field_Bz the gathered E and B fields
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator () (const long i,
                      amrex::ParticleReal& field_Ex,
                      amrex::ParticleReal& field_Ey,
                      amrex::ParticleReal& field_Ez,
                      amrex::ParticleReal& field_Bx,
                      amrex::ParticleReal& field_By,
                      amrex::ParticleReal& field_Bz) const noexcept
    {

        using namespace amrex::literals;

        amrex::ParticleReal x, y, z;
        m_get_position(i, x, y, z);

        // Find location of partice in the indices grid
        // (which is in the boosted frame)
        const int iz = static_cast<int>((z - m_zmin)/m_dz);

        constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
        amrex::ParticleReal const gamma = std::sqrt(1._prt + (m_ux[i]*m_ux[i] + m_uy[i]*m_uy[i] + m_uz[i]*m_uz[i])*inv_c2);
        amrex::ParticleReal const vzp = m_uz[i]/gamma;

        amrex::ParticleReal zpvdt = z + vzp*m_dt;

        // The position passed to the get_field methods needs to be in the lab frame.
        if (m_gamma_boost > 1._prt) {
            z = m_gamma_boost*z + m_uz_boost*m_time;
            zpvdt = m_gamma_boost*zpvdt + m_uz_boost*(m_time + m_dt);
        }

        amrex::ParticleReal Ex_sum = 0._prt;
        amrex::ParticleReal Ey_sum = 0._prt;
        const amrex::ParticleReal Ez_sum = 0._prt;
        amrex::ParticleReal Bx_sum = 0._prt;
        amrex::ParticleReal By_sum = 0._prt;
        const amrex::ParticleReal Bz_sum = 0._prt;

        if (d_quad.nelements > 0) {
            if (d_quad_indices_arr[iz] > -1) {
                const auto ielement = d_quad_indices_arr[iz];
                amrex::ParticleReal Ex, Ey, Bx, By;
                d_quad.get_field(ielement, x, y, z, zpvdt, Ex, Ey, Bx, By);
                Ex_sum += Ex;
                Ey_sum += Ey;
                Bx_sum += Bx;
                By_sum += By;
            }
        }

        if (d_plasmalens.nelements > 0) {
            if (d_plasmalens_indices_arr[iz] > -1) {
                const auto ielement = d_plasmalens_indices_arr[iz];
                amrex::ParticleReal Ex, Ey, Bx, By;
                d_plasmalens.get_field(ielement, x, y, z, zpvdt, Ex, Ey, Bx, By);
                Ex_sum += Ex;
                Ey_sum += Ey;
                Bx_sum += Bx;
                By_sum += By;
            }
        }

        if (m_gamma_boost > 1._prt) {
            // The fields returned from get_field is in the lab frame
            // Transform the fields to the boosted frame
            const amrex::ParticleReal Ex_boost = m_gamma_boost*Ex_sum - m_uz_boost*By_sum;
            const amrex::ParticleReal Ey_boost = m_gamma_boost*Ey_sum + m_uz_boost*Bx_sum;
            const amrex::ParticleReal Bx_boost = m_gamma_boost*Bx_sum + m_uz_boost*Ey_sum*inv_c2;
            const amrex::ParticleReal By_boost = m_gamma_boost*By_sum - m_uz_boost*Ex_sum*inv_c2;
            Ex_sum = Ex_boost;
            Ey_sum = Ey_boost;
            Bx_sum = Bx_boost;
            By_sum = By_boost;
        }

        field_Ex += Ex_sum;
        field_Ey += Ey_sum;
        field_Ez += Ez_sum;
        field_Bx += Bx_sum;
        field_By += By_sum;
        field_Bz += Bz_sum;

    }

};

#endif // WARPX_ACCELERATORLATTICE_LATTICEELEMENTS_LATTICEELEMENTFINDER_H_
