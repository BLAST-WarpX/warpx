#include "BoundaryConditions/WarpX_PEC.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>

using namespace amrex;
using namespace amrex::literals;

namespace
{
    /**
     * \brief Calculates the number of grid points the given index is pass the
     *        domain boundary i.e. a value of +1 means the current cell is
     *        outside of the simulation domain by 1 cell. Note that the high
     *        side domain boundary is between cell dom_hi and dom_hi+1 for cell
     *        centered grids and on cell dom_hi+1 for nodal grid. This is why
     *        (dom_hi[idim] + is_nodal[idim]) is used below.
     *
     * \param[in] dom_lo, dom_hi  Domain boundaries
     * \param[in] ijk_vec         Cell coordinates
     * \param[in] is_nodal        Whether the field of interest is nodal
     * \param[in] idim            Dimension of interest
     * \param[in] iside           0 for low and 1 for high
     *
     * \returns number of grid points to the boundary
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    int get_cell_count_to_boundary (const amrex::IntVect& dom_lo,
        const amrex::IntVect& dom_hi, const amrex::IntVect& ijk_vec,
        const amrex::IntVect& is_nodal, const int idim, const int iside)
    {
        return ((iside == 0) ? (dom_lo[idim] - ijk_vec[idim])
                             : (ijk_vec[idim] - (dom_hi[idim] + is_nodal[idim])));
    }


    /**
     * \brief Sets the electric field value tangential to the PEC boundary to zero. The
     *        tangential Efield components in the guard cells outside the
     *        domain boundary are set equal and opposite to the field in the valid cells
     *        at their mirrored locations. The normal Efield components in the guard cells
     *        are set equal to the field in the valid cells at their mirrored locations.
     *        The number or depth of guard cells updated is equal to the shape factor of
     *        particles in each dimension.
     *        For corner cells with mixed boundaries, the mirror location could be outside
     *        valid region, while still ensuring PEC condition is maintained across the
     *        PEC boundary, and the necessary sign change is accounted for depending on
     *        if the component, icomp, is tangential or normal to the PEC boundary.
     *
     *        For 3D :
     *            x component is tangential to the y-boundary and z-boundary
     *            y component is tangential to the x-boundary and z-boundary
     *            z component is tangential to the x-boundary and y-boundary
     *            x component is normal to the x-boundary
     *            y component is normal to the y-boundary
     *            z component is normal to the z-boundary
     *            where, x-boundary is the yz-plane at x=xmin and x=xmax
     *                   y-boundary is the xz-plane at y=ymin and y=ymax
     *                   z-boundary is the xy-plane at z=zmin and z=zmax
     *
     *        For 2D : WarpX uses X-Z as the two dimensions
     *            x component is tangential to the z-boundary
     *            y component is tangential to the x-boundary and z-boundary
     *            z component is tangential to the x-boundary
     *            x component is normal to the x-boundary
     *            y component is not normal to any boundary (Only xz dimensions in 2D)
     *            z component is normal to the z-boundary
     *            where, x-boundary is along the line z at x=xmin and x=xmax
     *                   z-boundary is along the line x at z=zmin and z=zmax
     *
     *        For 1D : WarpX uses Z as the only dimension
     *            x component is tangential to the z-boundary
     *            y component is tangential to the z-boundary
     *            z component is not tangential to the z-boundary
     *            x component is not normal to any boundary (Only z dimension in 1D)
     *            y component is not normal to any boundary (Only z dimension in 1D)
     *            z component is normal to the z-boundary
     *            where, z-boundary is a point at z=zmin and z=zmax
     *
     *        For RZ : WarpX uses R-Z as the two dimensions
     *            r component is tangential to the z-boundary
     *            theta_component is tangential to the r-boundary and z-boundary
     *            z component is tangential to the r-boundary
     *            r component is normal to the r-boundary
     *            theta_component is not normal to any boundary (on RZ dimensions are modeled)
     *            z component is normal to the z-boundary
     *            where, r-boundary is along the line z at r=rmin and r=rmax
     *                   z-boundary is along the line r at z=zmin and z=zmax
     *
     *        For RCYLINDER : WarpX uses R as the one dimension
     *            theta_component is tangential to the r-boundary
     *            z component is tangential to the r-boundary
     *            r component is normal to the r-boundary
     *            theta_component is not normal to any boundary (only r dimension)
     *            where, r-boundary is at r=rmin and r=rmax
     *
     *        For RSPHERE : WarpX uses R as the one dimension
     *            theta_component is tangential to the r-boundary
     *            phi component is tangential to the r-boundary
     *            r component is normal to the r-boundary
     *            theta_component is not normal to any boundary (only r dimension)
     *            phi_component is not normal to any boundary (only r dimension)
     *            where, r-boundary is at r=rmin and r=rmax
     *
     * \param[in] icomp        component of the Efield being updated
     *                         (0=x, 1=y, 2=z in Cartesian)
     *                         (0=r, 1=theta, 2=z in RZ and RCYLINDER)
     * \param[in] dom_lo       index value of the lower domain boundary (cell-centered)
     * \param[in] dom_hi       index value of the higher domain boundary (cell-centered)
     * \param[in] ijk_vec      indices along the x(i), y(j), z(k) of Efield Array4
     * \param[in] n            index of the MultiFab component being updated
     * \param[in] Efield       field data to be updated if (ijk) is at the boundary or a guard cell
     * \param[in] is_nodal     staggering of the field data being updated.
     * \param[in] fbndry_lo    Field boundary type at the lower boundaries
     * \param[in] fbndry_hi    Field boundary type at the upper boundaries
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void SetEfieldOnPEC (const int icomp, const amrex::IntVect & dom_lo,
                                const amrex::IntVect &dom_hi,
                                const amrex::IntVect &ijk_vec, const int n,
                                amrex::Array4<amrex::Real> const& Efield,
                                const amrex::IntVect& is_nodal,
                                amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_lo,
                                amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_hi,
                                FieldBoundaryType bc_type)
    {
        // Tangential Efield components in guard cells set equal and opposite to cells
        // in the mirror locations across the PEC boundary, whereas normal E-field
        // components are set equal to values in the mirror locations across the PEC
        // boundary. Here we just initialize it.
        amrex::IntVect ijk_mirror = ijk_vec;
        bool OnPECBoundary = false;
        bool GuardCell = false;
        amrex::Real sign = 1._rt;
        // Loop over all the dimensions
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            // Loop over sides, iside = 0 (lo), iside = 1 (hi)
            for (int iside = 0; iside < 2; ++iside) {
                const bool isPECBoundary = ( (iside == 0)
                    ? fbndry_lo[idim] == bc_type
                    : fbndry_hi[idim] == bc_type );
#if (defined WARPX_DIM_XZ) || (defined WARPX_DIM_RZ)
                // For 2D : for icomp==1, (Ey in XZ, Etheta in RZ),
                //          icomp=1 is tangential to both x and z boundaries
                //          The logic below ensures that the flags are set right for 2D
                const bool is_tangent_to_PEC = (icomp != AMREX_SPACEDIM*idim);
#elif (defined WARPX_DIM_1D_Z)
                // For 1D : icomp=0 and icomp=1 (Ex and Ey are tangential to the z boundary)
                //          The logic below ensures that the flags are set right for 1D
                const bool is_tangent_to_PEC = (icomp != idim+2);
#else
                const bool is_tangent_to_PEC = (icomp != idim);
#endif
                if (isPECBoundary) {
                    // grid point ijk_vec is ig number of points pass the
                    // domain boundary in direction, idim
                    const int ig = ::get_cell_count_to_boundary(
                        dom_lo, dom_hi, ijk_vec, is_nodal, idim, iside);

                    if (ig == 0) {
                        if (is_tangent_to_PEC && is_nodal[idim] == 1) {
                            OnPECBoundary = true;
                        }
                    } else if (ig > 0) {
                        // Find mirror location across PEC boundary
                        ijk_mirror[idim] = ( ( iside == 0)
                                        ? (dom_lo[idim] + ig - (1 - is_nodal[idim]))
                                        : (dom_hi[idim] + 1 - ig));
                        GuardCell = true;
                        // tangential components are inverted across PEC boundary
                        if (is_tangent_to_PEC) { sign *= -1._rt; }
                    }
                } // is PEC boundary
            } // loop over iside
        } // loop over dimensions
        if (OnPECBoundary) {
            // if ijk_vec is on a PEC boundary in any direction, set Etangential to 0.
            Efield(ijk_vec,n) = 0._rt;
        } else if (GuardCell) {
            Efield(ijk_vec,n) = sign * Efield(ijk_mirror,n);
        }
    }


    /**
     * \brief Sets the magnetic field value normal to the PEC boundary to zero. The
     *        tangential (and normal) field value of the guard cells outside the
     *        domain boundary are set equal (and opposite) to the respective field components
     *        in the valid cells at their mirrored locations.
     *        The number or depth of guard cells updated is equal to the shape factor of
     *        particles in each dimension.
     *
     *        For 3D :
     *            x component is tangential to the y-boundary and z-boundary
     *            y component is tangential to the x-boundary and z-boundary
     *            z component is tangential to the x-boundary and y-boundary
     *            x component is normal to the x-boundary
     *            y component is normal to the y-boundary
     *            z component is normal to the z-boundary
     *            where, x-boundary is the yz-plane at x=xmin and x=xmax
     *                   y-boundary is the xz-plane at y=ymin and y=ymax
     *                   z-boundary is the xy-plane at z=zmin and z=zmax
     *
     *        For 2D : WarpX uses X-Z as the two dimensions
     *            x component is tangential to the z-boundary
     *            y component is tangential to the x-boundary and z-boundary
     *            z component is tangential to the x-boundary
     *            x component is normal to the x-boundary
     *            y component is not normal to any boundary (Only xz dimensions in 2D)
     *            z component is normal to the z-boundary
     *            where, x-boundary is along the line z at x=xmin and x=xmax
     *                   z-boundary is along the line x at z=zmin and z=zmax
     *
     *        For 1D : WarpX uses Z as the only dimension
     *            x component is tangential to the z-boundary
     *            y component is tangential to the z-boundary
     *            z component is not tangential to the z-boundary
     *            x component is not normal to any boundary (Only z dimension in 1D)
     *            y component is not normal to any boundary (Only z dimension in 1D)
     *            z component is normal to the z-boundary
     *            where, z-boundary is a point at z=zmin and z=zmax
     *
     *        For RZ : WarpX uses R-Z as the two dimensions
     *            r component is tangential to the z-boundary
     *            theta_component is tangential to the r-boundary and z-boundary
     *            z component is tangential to the r-boundary
     *            r component is normal to the r-boundary
     *            theta_component is not normal to any boundary (on RZ dimensions are modeled)
     *            z component is normal to the z-boundary
     *            where, r-boundary is along the line z at r=rmin and r=rmax
     *                   z-boundary is along the line r at z=zmin and z=zmax
     *
     *        For RCYLINDER : WarpX uses R as the one dimension
     *            theta_component is tangential to the r-boundary
     *            z component is tangential to the r-boundary
     *            r component is normal to the r-boundary
     *            theta_component is not normal to any boundary (only r dimension)
     *            where, r-boundary is at r=rmin and r=rmax
     *
     *        For RSPHERE : WarpX uses R as the one dimension
     *            theta_component is tangential to the r-boundary
     *            phi component is tangential to the r-boundary
     *            r component is normal to the r-boundary
     *            theta_component is not normal to any boundary (only r dimension)
     *            phi_component is not normal to any boundary (only r dimension)
     *            where, r-boundary is at r=rmin and r=rmax
     *
     * \param[in] icomp        component of the Bfield being updated
     *                         (0=x, 1=y, 2=z in Cartesian)
     *                         (0=r, 1=theta, 2=z in RZ and RCYLINDER)
     * \param[in] dom_lo       index value of the lower domain boundary (cell-centered)
     * \param[in] dom_hi       index value of the higher domain boundary (cell-centered)
     * \param[in] ijk_vec      indices along the x(i), y(j), z(k) of Efield Array4
     * \param[in] n            index of the MultiFab component being updated
     * \param[in] Bfield       field data to be updated if (ijk) is at the boundary
                               or a guard cell
     * \param[in] is_nodal     staggering of the field data being updated.
     * \param[in] fbndry_lo    Field boundary type at the lower boundaries
     * \param[in] fbndry_hi    Field boundary type at the upper boundaries
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void SetBfieldOnPEC (const int icomp, const amrex::IntVect & dom_lo,
                           const amrex::IntVect & dom_hi,
                           const amrex::IntVect & ijk_vec, const int n,
                           amrex::Array4<amrex::Real> const& Bfield,
                           const amrex::IntVect & is_nodal,
                           amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_lo,
                           amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_hi,
                           FieldBoundaryType bc_type)
    {
        amrex::IntVect ijk_mirror = ijk_vec;
        bool OnPECBoundary = false;
        bool GuardCell = false;
        amrex::Real sign = 1._rt;
        // Loop over all dimensions
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            // Loop over sides, iside = 0 (lo), iside = 1 (hi)
            for (int iside = 0; iside < 2; ++iside) {
                const bool isPECBoundary = ( (iside == 0)
                    ? fbndry_lo[idim] == bc_type
                    : fbndry_hi[idim] == bc_type );
                if (isPECBoundary) {
#if (defined WARPX_DIM_XZ) || (defined WARPX_DIM_RZ)
                    // For 2D : for icomp==1, (By in XZ, Btheta in RZ),
                    //          icomp=1 is not normal to x or z boundary
                    //          The logic below ensures that the flags are set right for 2D
                    const bool is_normal_to_PEC = (icomp == (AMREX_SPACEDIM*idim));
#elif (defined WARPX_DIM_1D_Z)
                    // For 1D : icomp=0 and icomp=1 (Bx and By are not normal to the z boundary)
                    //          The logic below ensures that the flags are set right for 1D
                    const bool is_normal_to_PEC = (icomp == (idim+2));
#else
                    const bool is_normal_to_PEC = (icomp == idim);
#endif

                    // grid point ijk_vec is ig number of points pass the
                    // domain boundary in direction, idim
                    const int ig = ::get_cell_count_to_boundary(
                        dom_lo, dom_hi, ijk_vec, is_nodal, idim, iside);

                    if (ig == 0) {
                        // Only normal component is set to 0
                        if (is_normal_to_PEC && is_nodal[idim]==1) {
                            OnPECBoundary = true;
                        }
                    } else if ( ig > 0) {
                        // Mirror location inside the domain by "ig" number of cells
                        // across PEC boundary in direction, idim, and side, iside
                        ijk_mirror[idim] = ( (iside == 0)
                                        ? (dom_lo[idim] + ig - (1 - is_nodal[idim]))
                                        : (dom_hi[idim] + 1 - ig));
                        GuardCell = true;
                        // Sign of the normal component in guard cell is inverted
                        if (is_normal_to_PEC) { sign *= -1._rt; }
                    }
                } // if PEC Boundary
            } // loop over sides
        } // loop of dimensions

        if (OnPECBoundary) {
            // if ijk_vec is on a PEC boundary in any direction, set Bnormal to 0.
            Bfield(ijk_vec,n) = 0._rt;
        } else if (GuardCell) {
            // Bnormal and Btangential is set opposite and equal to the value
            // in the mirror location, respectively.
            Bfield(ijk_vec,n) = sign * Bfield(ijk_mirror,n);
        }
    }

    /**
     * \brief Reflect the J or Rho field values deposited to the guard cells to their
     *        mirror location inside the domain at PEC or PMC boundaries.
     *
     *        PMC: -Rho/J_parallel deposited to guard region is added to its mirror location.
     *             -J_perpindicular deposited to guard region is subtracted from its mirror location.
     *             -This is a symmetry boundary. Reflecting Rho/J as described above is
     *              equivalent to capturing Rho/J of the mirror charge of the same sign
     *              on the other side of the symmetry plane.
     *        PEC: -Rho/J_parallel deposited to guard region is subtracted from its mirror location.
     *             -J_perpindicular deposited to guard region is added to its mirror location.
     *             -This is an anti-symmetry boundary. Reflecting Rho/J as described above is
     *              equivalent to capturing Rho/J of the mirror charge of the opposite sign
     *              on the other side of the anti-symmetry plane.
     *
     * \param[in] n                 index of the MultiFab component being updated
     * \param[in] ijk_vec           indices along the x(i), y(j), z(k) of the Rho/J Array4
     * \param[in out] field         field data to be updated
     * \param[in] mirrorfac         mirror cell indices given by mirrorfac - ijk_vec
     * \param[in] is_reflective     whether the given boundary is reflective
     * \param[in] psign             sign for reflecting the field value across the boundary
     * \param[in] idim              boundary direction
     * \param[in] is_nodal_r        whether data is nodal along r
     * \param[in] fabbox            multifab box including ghost cells
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void ReflectJorRho (const int n,
                        const amrex::IntVect & ijk_vec,
                              amrex::Array4<amrex::Real> const& field,
                        const amrex::GpuArray<int,2> & mirrorfac,
                        const amrex::GpuArray<int,2> & is_reflective,
                        const amrex::GpuArray<amrex::Real,2> & psign,
                        [[maybe_unused]]int const idim,
                        [[maybe_unused]]int const is_nodal_r,
                              amrex::Box const& fabbox)
    {

        for (int iside = 0; iside < 2; ++iside) {

            if (!is_reflective[iside]) { continue; }

            // Get the mirror guard cell index
            amrex::IntVect ijk_mirror = ijk_vec;
            ijk_mirror[idim] = mirrorfac[iside] - ijk_vec[idim];

            // Update the cell if the mirror guard cell exists
            if (fabbox.contains(ijk_mirror)) {
                // Note that this includes the cells on the boundary
                amrex::Real rscale = 1._rt;
#if (defined WARPX_DIM_RZ) || (defined WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                if (idim == 0 && iside == 1) {
                    // Account for different dV at different radii
                    amrex::Real const rshift = (is_nodal_r ? 0.0_rt : 0.5_rt);
                    const amrex::Real rvalid = ijk_vec[idim] + rshift;
                    const amrex::Real rmirror = ijk_mirror[idim] + rshift;
                    rscale = rmirror/rvalid;
#if defined(WARPX_DIM_RSPHERE)
                    rscale *= rmirror/rvalid;
#endif
                }
#endif
                // Reflected J/rho deposited to guard cell to mirror valid cell
                field(ijk_vec,n) += rscale * psign[iside] * field(ijk_mirror,n);
            }
        }
    }

    /**
     * \brief Set the J or Rho field values in the guard cells consistent with the
     *        assumed symmetries associated with PEC or PMC boundaries.
     *
     *        PMC: -Rho/J_parallel in guard region is equal to Rho/J_parallel at
     *              its mirror location inside the domain.
     *             -J_perpindicular in guard region is equal and opposite to
     *              J_perpinducular at its mirror location inside the domain.
     *             -This is a symmetry boundary. Setting the BCs for Rho/J in this way is
     *              equivalent to the Rho/J of the mirror charge of the same sign
     *              on the other side of the symmetry plane.
     *        PEC: -Rho/J_parallel in guard region is equal and opposite to Rho/J_paralle
     *              at its mirror location inside the domain.
     *             -J_perpindicular in the guard region is equal to
     *              J_perpindicular at its mirror location inside the domain.
     *             -This is an anti-symmetry boundary. Setting the BCs for Rho/J in this
     *              way is equivalent to the Rho/J of the mirror charge of the opposite sign
     *              on the other side of the anti-symmetry plane.
     *
     * \param[in] n                 index of the MultiFab component being updated
     * \param[in] ijk_vec           indices along the x(i), y(j), z(k) of the Rho/J Array4
     * \param[in out] field         field data to be updated
     * \param[in] mirrorfac         mirror cell indices given by mirrorfac - ijk_vec
     * \param[in] is_reflective     whether the given boundary is reflective
     * \param[in] psign             sign for reflecting the field value across the boundary
     * \param[in] idim              boundary direction
     * \param[in] is_nodal_r        whether data is nodal along r
     * \param[in] fabbox            multifab box including ghost cells
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void SetJorRho (const int n,
                    const amrex::IntVect & ijk_vec,
                          amrex::Array4<amrex::Real> const& field,
                    const amrex::GpuArray<int,2> & mirrorfac,
                    const amrex::GpuArray<int,2> & is_reflective,
                    const amrex::GpuArray<amrex::Real,2> & psign,
                    [[maybe_unused]]int const idim,
                    [[maybe_unused]]int const is_nodal_r,
                          amrex::Box const& fabbox)
    {

        for (int iside = 0; iside < 2; ++iside) {

            if (!is_reflective[iside]) { continue; }

            // Get the mirror guard cell index
            amrex::IntVect ijk_mirror = ijk_vec;
            ijk_mirror[idim] = mirrorfac[iside] - ijk_vec[idim];

            // Update the cell if the mirror guard cell exists
            if (ijk_vec == ijk_mirror && psign[iside] == -1) {
                field(ijk_mirror,n) = 0.0;
            }
            else if ( (ijk_vec != ijk_mirror) && (fabbox.contains(ijk_mirror)) ) {
                amrex::Real inv_rscale = 1._rt;
#if (defined WARPX_DIM_RZ) || (defined WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                if (idim == 0 && iside == 1) {
                    // Account for different dV at different radii
                    amrex::Real const rshift = (is_nodal_r ? 0.0_rt : 0.5_rt);
                    const amrex::Real rvalid = ijk_vec[idim] + rshift;
                    const amrex::Real rmirror = ijk_mirror[idim] + rshift;
                    inv_rscale = rvalid/rmirror;
#if defined(WARPX_DIM_RSPHERE)
                    inv_rscale *= rvalid/rmirror;
#endif
                }
#endif
                field(ijk_mirror,n) = inv_rscale * psign[iside] * field(ijk_vec,n);
            }
        }
    }

    /**
     * \brief This function sets the given field value on a PEC boundary
     *        to enforce a Neumann boundary condition (zero derivative) in the
     *        normal direction.
     *
     * \param[in] n            index of the MultiFab component being updated
     * \param[in] ijk_vec      indices along the x(i), y(j), z(k) of the rho Array4
     * \param[in out] field    field data to be updated
     * \param[in] mirrorfac    mirror cell is given by mirrorfac - ijk_vec
     * \param[in] psign        Whether the field value should be flipped across the boundary
     * \param[in] is_pec       Whether the given boundary is PEC
     * \param[in] tangent_to_bndy    Whether a given direction is perpendicular to the boundary
     * \param[in] fabbox       multifab box including ghost cells
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void SetNeumannOnPEC (const int n,
                          const amrex::IntVect & ijk_vec,
                          amrex::Array4<amrex::Real> const& field,
                          amrex::GpuArray<GpuArray<int, 2>, AMREX_SPACEDIM> const& mirrorfac,
                          amrex::GpuArray<GpuArray<bool, 2>, AMREX_SPACEDIM> const& is_pec,
                          amrex::Box const& fabbox )
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            for (int iside = 0; iside < 2; ++iside)
            {
                if (!is_pec[idim][iside]) { continue; }

                // Get the mirror guard cell index
                amrex::IntVect iv_mirror = ijk_vec;
                iv_mirror[idim] = mirrorfac[idim][iside] - ijk_vec[idim];

                // On the PEC boundary the field value is set equal to the
                // first value in the domain (nodal fields)
                if (ijk_vec == iv_mirror) {
                    iv_mirror[idim] += (iside == 0) ? 1 : -1;
                    if (fabbox.contains(iv_mirror)) { field(ijk_vec, n) = field(iv_mirror, n); }
                }
                // otherwise set the mirror guard cell equal to the internal cell value
                else if (fabbox.contains(iv_mirror))
                {
                    field(iv_mirror, n) = field(ijk_vec, n);
                }
            }
        }
    }
}

void
PEC::ApplyPECtoEfield (
    std::array<amrex::MultiFab*, 3> Efield,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
    FieldBoundaryType bc_type,
    const amrex::IntVect& ng_fieldgather, const amrex::Geometry& geom,
    const int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios,
    const bool split_pml_field)
{
    amrex::Box domain_box = geom.Domain();
    if (patch_type == PatchType::coarse && (lev > 0)) {
        domain_box.coarsen(ref_ratios[lev-1]);
    }
    const amrex::IntVect domain_lo = domain_box.smallEnd();
    const amrex::IntVect domain_hi = domain_box.bigEnd();
    amrex::GpuArray<FieldBoundaryType, 3> fbndry_lo;
    amrex::GpuArray<FieldBoundaryType, 3> fbndry_hi;
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        fbndry_lo[idim] = field_boundary_lo[idim];
        fbndry_hi[idim] = field_boundary_hi[idim];
    }
    const amrex::IntVect Ex_nodal = Efield[0]->ixType().toIntVect();
    const amrex::IntVect Ey_nodal = Efield[1]->ixType().toIntVect();
    const amrex::IntVect Ez_nodal = Efield[2]->ixType().toIntVect();
    // For each Efield multifab, apply PEC boundary condition to ncomponents
    // If not split E-field, the PEC is applied to the regular Efield used in Maxwell's eq.
    // If split_pml_field is true, then PEC is applied to all the split field components of the tangential field.
    const int nComp_x = Efield[0]->nComp();
    const int nComp_y = Efield[1]->nComp();
    const int nComp_z = Efield[2]->nComp();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Efield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        // Extract field data
        amrex::Array4<amrex::Real> const& Ex = Efield[0]->array(mfi);
        amrex::Array4<amrex::Real> const& Ey = Efield[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Ez = Efield[2]->array(mfi);

        // Extract tileboxes for which to loop
        // if split field, the box includes nodal flag
        // For E-field used in Maxwell's update, nodal flag plus cells that particles
        // gather fields from in the guard-cell region are included.
        // Note that for simulations without particles or laser, ng_field_gather is 0
        // and the guard-cell values of the E-field multifab will not be modified.
        amrex::Box const& tex = (split_pml_field) ? mfi.tilebox(Efield[0]->ixType().toIntVect())
                                                  : mfi.tilebox(Efield[0]->ixType().toIntVect(), ng_fieldgather);
        amrex::Box const& tey = (split_pml_field) ? mfi.tilebox(Efield[1]->ixType().toIntVect())
                                                  : mfi.tilebox(Efield[1]->ixType().toIntVect(), ng_fieldgather);
        amrex::Box const& tez = (split_pml_field) ? mfi.tilebox(Efield[2]->ixType().toIntVect())
                                                  : mfi.tilebox(Efield[2]->ixType().toIntVect(), ng_fieldgather);

        // loop over cells and update fields
        amrex::ParallelFor(
            tex, nComp_x,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 0;
                ::SetEfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                           Ex, Ex_nodal, fbndry_lo, fbndry_hi, bc_type);
            },
            tey, nComp_y,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 1;
                ::SetEfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                           Ey, Ey_nodal, fbndry_lo, fbndry_hi, bc_type);
            },
            tez, nComp_z,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 2;
                ::SetEfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                           Ez, Ez_nodal, fbndry_lo, fbndry_hi, bc_type);
            }
        );
    }
}


void
PEC::ApplyPECtoBfield (
    std::array<amrex::MultiFab*, 3> Bfield,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
    FieldBoundaryType bc_type,
    const amrex::IntVect& ng_fieldgather, const amrex::Geometry& geom,
    const int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios,
    const bool split_pml_field)
{
    amrex::Box domain_box = geom.Domain();
    if (patch_type == PatchType::coarse && (lev > 0)) {
        domain_box.coarsen(ref_ratios[lev-1]);
    }
    const amrex::IntVect domain_lo = domain_box.smallEnd();
    const amrex::IntVect domain_hi = domain_box.bigEnd();
    amrex::GpuArray<FieldBoundaryType, 3> fbndry_lo;
    amrex::GpuArray<FieldBoundaryType, 3> fbndry_hi;
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        fbndry_lo[idim] = field_boundary_lo[idim];
        fbndry_hi[idim] = field_boundary_hi[idim];
    }
    const amrex::IntVect Bx_nodal = Bfield[0]->ixType().toIntVect();
    const amrex::IntVect By_nodal = Bfield[1]->ixType().toIntVect();
    const amrex::IntVect Bz_nodal = Bfield[2]->ixType().toIntVect();
    const int nComp_x = Bfield[0]->nComp();
    const int nComp_y = Bfield[1]->nComp();
    const int nComp_z = Bfield[2]->nComp();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Bfield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // Extract field data
        amrex::Array4<amrex::Real> const& Bx = Bfield[0]->array(mfi);
        amrex::Array4<amrex::Real> const& By = Bfield[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Bz = Bfield[2]->array(mfi);

        // Extract tileboxes for which to loop
        // For B-field used in Maxwell's update, nodal flag plus cells that particles
        // gather fields from in the guard-cell region are included.
        // Note that for simulations without particles or laser, ng_field_gather is 0
        // and the guard-cell values of the B-field multifab will not be modified.
        amrex::Box const& tbx = (split_pml_field) ? mfi.tilebox(Bfield[0]->ixType().toIntVect())
                                                  : mfi.tilebox(Bfield[0]->ixType().toIntVect(), ng_fieldgather);
        amrex::Box const& tby = (split_pml_field) ? mfi.tilebox(Bfield[1]->ixType().toIntVect())
                                                  : mfi.tilebox(Bfield[1]->ixType().toIntVect(), ng_fieldgather);
        amrex::Box const& tbz = (split_pml_field) ? mfi.tilebox(Bfield[2]->ixType().toIntVect())
                                                  : mfi.tilebox(Bfield[2]->ixType().toIntVect(), ng_fieldgather);

        // loop over cells and update fields
        amrex::ParallelFor(
            tbx, nComp_x,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 0;
                ::SetBfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                     Bx, Bx_nodal, fbndry_lo, fbndry_hi, bc_type);
            },
            tby, nComp_y,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 1;
                ::SetBfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                     By, By_nodal, fbndry_lo, fbndry_hi, bc_type);
            },
            tbz, nComp_z,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 2;
                ::SetBfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                     Bz, Bz_nodal, fbndry_lo, fbndry_hi, bc_type);
            }
        );
    }
}


/**
 * \brief Step 1: Reflect the Rho field values deposited to the guard cells to
 *                their mirror locations inside the domain at PEC and PMC boundaries.
 *        Step 2: Set the Rho field values in the guard cells consistent with the
 *                assumed symmetries associated with PEC and PMC boundaries.
 *
 *        PEC: This is an anti-symmetry boundary. Rho deposited to guard cells is
 *             subtracted from its mirror location inside the domain, which is
 *             equivalent to depositing Rho associated with the image charge of the
 *             opposite sign on the other side of the PEC boundary.
 *        PMC: This is a symmetry boundary. Rho deposited to guard cells is
 *             Added to its mirror location inside the domain, which is
 *             equivalent to depositing Rho associated with the image charge of the
 *             same sign on the other side of the PMC boundary.
 *
 **/
void
PEC::ApplyReflectiveBoundarytoRhofield (
    amrex::MultiFab* rho,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
    const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_lo,
    const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_hi,
    const amrex::Geometry& geom,
    const int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios)
{
    amrex::Box domain_box = geom.Domain();
    if (patch_type == PatchType::coarse && (lev > 0)) {
        domain_box.coarsen(ref_ratios[lev-1]);
    }
    domain_box.convert(rho->ixType());

    const amrex::IntVect domain_lo = domain_box.smallEnd();
    const amrex::IntVect domain_hi = domain_box.bigEnd();

    const amrex::IntVect rho_nodal = rho->ixType().toIntVect();
    const amrex::IntVect Ng = rho->nGrowVect();

    // Declare and assign GpuArrays before ifdef AMREX_USE_OMP
    amrex::GpuArray<GpuArray<int,2>,AMREX_SPACEDIM> is_reflective;
    amrex::GpuArray<GpuArray<Real,2>,AMREX_SPACEDIM> psign;
    amrex::GpuArray<GpuArray<int,2>,AMREX_SPACEDIM> mirrorfac;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

        // Check if boundary is reflective on lo side
        is_reflective[idim][0] = ( (particle_boundary_lo[idim] == ParticleBoundaryType::Reflecting)
                               ||  (particle_boundary_lo[idim] == ParticleBoundaryType::Thermal)
                               ||  (field_boundary_lo[idim] == FieldBoundaryType::PMC)
                               ||  (field_boundary_lo[idim] == FieldBoundaryType::PEC) );

        // Check if boundary is reflective on hi side
        is_reflective[idim][1] = ( (particle_boundary_hi[idim] == ParticleBoundaryType::Reflecting)
                               ||  (particle_boundary_hi[idim] == ParticleBoundaryType::Thermal)
                               ||  (field_boundary_hi[idim] == FieldBoundaryType::PMC)
                               ||  (field_boundary_hi[idim] == FieldBoundaryType::PEC) );

        // Set psign on lo side
        psign[idim][0] = ( (particle_boundary_lo[idim] == ParticleBoundaryType::Reflecting)
                       ||  (particle_boundary_lo[idim] == ParticleBoundaryType::Thermal)
                       ||  (field_boundary_lo[idim] == FieldBoundaryType::PMC) )
                           ? 1._rt : -1._rt;

        // Set psign on hi side
        psign[idim][1] = ( (particle_boundary_hi[idim] == ParticleBoundaryType::Reflecting)
                       ||  (particle_boundary_hi[idim] == ParticleBoundaryType::Thermal)
                       ||  (field_boundary_hi[idim] == FieldBoundaryType::PMC) )
                           ? 1._rt : -1._rt;

        // Set the mirror index offset on lo and hi sides
        mirrorfac[idim][0] = 2*domain_lo[idim] - (1 - rho_nodal[idim]);
        mirrorfac[idim][1] = 2*domain_hi[idim] - (1 - rho_nodal[idim]);

    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // The false flag here is to ensure that this loop does not use tiling.
    // The boxes are grown to include transverse ghost cells prior to the reflection.
    // Tiling is problematic because neighboring tiles will have overlapping boxes
    // in the direction transverse to the boundary, thereby reflecting the value multiple
    // times in the overlapping region.
    for (amrex::MFIter mfi(*rho, false); mfi.isValid(); ++mfi) {

        // Get the multifab box including ghost cells
        const amrex::Box& rho_fabbox = mfi.fabbox();

        // Get nodal box that does not include ghost cells
        const amrex::Box node_box = amrex::convert(mfi.validbox(),IntVect::TheNodeVector());

        //
        // Step 1: Reflect Rho deposited to guard cells back into the domain
        //
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

            if ( !is_reflective[idim][0] && !is_reflective[idim][1] ) { continue; }

            if ( (node_box.smallEnd()[idim] != domain_lo[idim]) &&
                 (node_box.bigEnd()[idim] != domain_hi[idim]) ) { continue; }

            // Get Rho box and grow to include guard cells in directions transverse
            // to this boundary. This is required to correctly reflect Rho at domain
            // corners that touch multiple PEC/PMC boundaries.
            amrex::Box rho_box = amrex::convert(mfi.validbox(),rho_nodal);
            for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
                if (jdim==idim) { continue; }
                rho_box.grow(jdim,Ng[jdim]);
            }

            auto const& rho_array = rho->array(mfi);

            // Loop over cells and reflect Rho
            amrex::ParallelFor(
            rho_box, rho->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                ::ReflectJorRho( n, iv, rho_array, mirrorfac[idim],
                                 is_reflective[idim], psign[idim], idim,
                                 rho_nodal[0], rho_fabbox );
            });

        }

        //
        // Step 2: Set Rho in the guard cells
        //
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

            if ( !is_reflective[idim][0] && !is_reflective[idim][1] ) { continue; }

            if ( (node_box.smallEnd()[idim] != domain_lo[idim]) &&
                 (node_box.bigEnd()[idim] != domain_hi[idim]) ) { continue; }

            // Get Rho box and grow to include guard cells in transverse dirs
            amrex::Box rho_box = amrex::convert(mfi.validbox(),rho_nodal);
            for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
                if (jdim==idim) { continue; }
                rho_box.grow(jdim,Ng[jdim]);
            }

            auto const& rho_array = rho->array(mfi);

            // Loop over cells and set Rho in guard cells
            amrex::ParallelFor(
            rho_box, rho->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                ::SetJorRho( n, iv, rho_array, mirrorfac[idim],
                             is_reflective[idim], psign[idim], idim,
                             rho_nodal[0], rho_fabbox );
            });

        }

    }

}

/**
 * \brief Step 1: Reflect the J field values deposited to the guard cells to
 *                their mirror locations inside the domain at PEC and PMC boundaries.
 *        Step 2: Set the J field values in the guard cells consistent with the
 *                assumed symmetries associated with PEC and PMC boundaries.
 *
 *        PEC: This is an anti-symmetry boundary. Jparallel/Jperp to a boundary deposited
 *             to guard cells is subtracted/added from/to its mirror location inside the
 *             domain, which is equivalent to depositing J associated with the image
 *             charge of the opposite sign on the other side of the PEC boundary.
 *        PMC: This is a symmetry boundary. Jparallel/Jperp to a boundary deposited
 *             to guard cells is added/subtracted to/from its mirror location inside the
 *             domain, which is equivalent to depositing J associated with the image
 *             charge of the opposite sign on the other side of the PEC boundary.
 *
 **/
void
PEC::ApplyReflectiveBoundarytoJfield (
    amrex::MultiFab* Jx, amrex::MultiFab* Jy, amrex::MultiFab* Jz,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
    const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_lo,
    const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_hi,
    const amrex::Geometry& geom,
    const int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios)
{
    amrex::Box domain_box = geom.Domain();
    if (patch_type == PatchType::coarse && (lev > 0)) {
        domain_box.coarsen(ref_ratios[lev-1]);
    }
    domain_box.convert(IntVect::TheNodeVector());

    const amrex::IntVect domain_lo = domain_box.smallEnd();
    const amrex::IntVect domain_hi = domain_box.bigEnd();

    const amrex::IntVect Jx_nodal = Jx->ixType().toIntVect();
    const amrex::IntVect Jy_nodal = Jy->ixType().toIntVect();
    const amrex::IntVect Jz_nodal = Jz->ixType().toIntVect();
    const amrex::IntVect Ng = Jx->nGrowVect();

    // Declare and assign GpuArrays before ifdef AMREX_USE_OMP
    bool is_tangent_to_bndy;
    amrex::GpuArray<GpuArray<int,2>,AMREX_SPACEDIM> is_reflective;
    amrex::GpuArray<GpuArray<GpuArray<Real,2>,3>,AMREX_SPACEDIM> psign;
    amrex::GpuArray<GpuArray<GpuArray<int,2>,3>,AMREX_SPACEDIM> mirrorfac;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

        // Check if boundary is reflective on lo side
        is_reflective[idim][0] = ( (particle_boundary_lo[idim] == ParticleBoundaryType::Reflecting)
                               ||  (particle_boundary_lo[idim] == ParticleBoundaryType::Thermal)
                               ||  (field_boundary_lo[idim] == FieldBoundaryType::PMC)
                               ||  (field_boundary_lo[idim] == FieldBoundaryType::PEC) );

        // Check if boundary is reflective on hi side
        is_reflective[idim][1] = ( (particle_boundary_hi[idim] == ParticleBoundaryType::Reflecting)
                               ||  (particle_boundary_hi[idim] == ParticleBoundaryType::Thermal)
                               ||  (field_boundary_hi[idim] == FieldBoundaryType::PMC)
                               ||  (field_boundary_hi[idim] == FieldBoundaryType::PEC) );

        for (int icomp = 0; icomp < 3; ++icomp) {
            // Set the psign value for each component of J for each direction
#if (defined WARPX_DIM_1D_Z)
            // For 1D : icomp=0 and icomp=1 (Jx and Jy are tangential to the z boundary)
            //          The logic below ensures that the flags are set right for 1D
            is_tangent_to_bndy = (icomp != (idim+2));
#elif (defined WARPX_DIM_XZ) || (defined WARPX_DIM_RZ)
            // For 2D : for icomp==1, (Jy in XZ, Jtheta in RZ),
            //          icomp=1 is tangential to both x and z boundaries
            //          The logic below ensures that the flags are set right for 2D
            is_tangent_to_bndy = (icomp != AMREX_SPACEDIM*idim);
#else
            is_tangent_to_bndy = (icomp != idim);
#endif

            amrex::Real pmc_sign = 1._rt;
            if (!is_tangent_to_bndy) { pmc_sign = -1._rt; }

            // Set psign on lo side
            psign[idim][icomp][0] = ( (particle_boundary_lo[idim] == ParticleBoundaryType::Reflecting)
                                  ||  (particle_boundary_lo[idim] == ParticleBoundaryType::Thermal)
                                  ||  (field_boundary_lo[idim] == FieldBoundaryType::PMC) )
                                      ? pmc_sign : -pmc_sign;

            // Set psign on hi side
            psign[idim][icomp][1] = ( (particle_boundary_hi[idim] == ParticleBoundaryType::Reflecting)
                                  ||  (particle_boundary_hi[idim] == ParticleBoundaryType::Thermal)
                                  ||  (field_boundary_hi[idim] == FieldBoundaryType::PMC) )
                                      ? pmc_sign : -pmc_sign;
        }

        // Set the mirror index offset on lo and hi sides
        mirrorfac[idim][0][0] = 2*domain_lo[idim] - (1 - Jx_nodal[idim]);
        mirrorfac[idim][0][1] = 2*domain_hi[idim] - (1 - Jx_nodal[idim]);
        mirrorfac[idim][1][0] = 2*domain_lo[idim] - (1 - Jy_nodal[idim]);
        mirrorfac[idim][1][1] = 2*domain_hi[idim] - (1 - Jy_nodal[idim]);
        mirrorfac[idim][2][0] = 2*domain_lo[idim] - (1 - Jz_nodal[idim]);
        mirrorfac[idim][2][1] = 2*domain_hi[idim] - (1 - Jz_nodal[idim]);

    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // The false flag here is to ensure that this loop does not use tiling.
    // The boxes are grown to include transverse ghost cells prior to the reflection.
    // Tiling is problematic because neighboring tiles will have overlapping boxes
    // in the direction transverse to the boundary, thereby reflecting the value multiple
    // times in the overlapping region.
    for (amrex::MFIter mfi(*Jx, false); mfi.isValid(); ++mfi) {

        // Get the multifab box including ghost cells
        const amrex::Box Jx_fabbox = mfi.fabbox().convert(Jx_nodal);
        const amrex::Box Jy_fabbox = mfi.fabbox().convert(Jy_nodal);
        const amrex::Box Jz_fabbox = mfi.fabbox().convert(Jz_nodal);

        // Get nodal box that does not include ghost cells
        const amrex::Box node_box = amrex::convert(mfi.validbox(),IntVect::TheNodeVector());

        //
        // Step 1: Reflect J deposited to guard cells back into the domain
        //
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

            if ( !is_reflective[idim][0] && !is_reflective[idim][1] ) { continue; }

            if ( (node_box.smallEnd()[idim] != domain_lo[idim]) &&
                 (node_box.bigEnd()[idim] != domain_hi[idim]) ) { continue; }

            // Get J boxes and grow to include guard cells in directions transverse
            // to this boundary. This is required to correctly reflect J at domain
            // corners that touch multiple PEC/PMC boundaries.
            amrex::Box Jx_box = amrex::convert(mfi.validbox(),Jx_nodal);
            amrex::Box Jy_box = amrex::convert(mfi.validbox(),Jy_nodal);
            amrex::Box Jz_box = amrex::convert(mfi.validbox(),Jz_nodal);
            for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
                if (jdim==idim) { continue; }
                Jx_box.grow(jdim,Ng[jdim]);
                Jy_box.grow(jdim,Ng[jdim]);
                Jz_box.grow(jdim,Ng[jdim]);
            }

            auto const& Jx_array = Jx->array(mfi);
            auto const& Jy_array = Jy->array(mfi);
            auto const& Jz_array = Jz->array(mfi);

            // Loop over cells and reflect J
            amrex::ParallelFor(
            Jx_box, Jx->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                ::ReflectJorRho( n, iv, Jx_array, mirrorfac[idim][0],
                                 is_reflective[idim], psign[idim][0], idim,
                                 Jx_nodal[0], Jx_fabbox );
            },
            Jy_box, Jy->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                ::ReflectJorRho( n, iv, Jy_array, mirrorfac[idim][1],
                                 is_reflective[idim], psign[idim][1], idim,
                                 Jy_nodal[0], Jy_fabbox );
            },
            Jz_box, Jz->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                ::ReflectJorRho( n, iv, Jz_array, mirrorfac[idim][2],
                                 is_reflective[idim], psign[idim][2], idim,
                                 Jz_nodal[0], Jz_fabbox );
            });

        }

        //
        // Step 2: Set J in the guard cells
        //
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

            if ( !is_reflective[idim][0] && !is_reflective[idim][1] ) { continue; }

            if ( (node_box.smallEnd()[idim] != domain_lo[idim]) &&
                 (node_box.bigEnd()[idim] != domain_hi[idim]) ) { continue; }

            // Get J boxes and grow to include guard cells in transverse dirs
            amrex::Box Jx_box = amrex::convert(mfi.validbox(),Jx_nodal);
            amrex::Box Jy_box = amrex::convert(mfi.validbox(),Jy_nodal);
            amrex::Box Jz_box = amrex::convert(mfi.validbox(),Jz_nodal);
            for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
                if (jdim==idim) { continue; }
                Jx_box.grow(jdim,Ng[jdim]);
                Jy_box.grow(jdim,Ng[jdim]);
                Jz_box.grow(jdim,Ng[jdim]);
            }

            auto const& Jx_array = Jx->array(mfi);
            auto const& Jy_array = Jy->array(mfi);
            auto const& Jz_array = Jz->array(mfi);

            // Loop over cells and set J in guard cells
            amrex::ParallelFor(
            Jx_box, Jx->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                ::SetJorRho( n, iv, Jx_array, mirrorfac[idim][0],
                             is_reflective[idim], psign[idim][0], idim,
                             Jx_nodal[0], Jx_fabbox );
            },
            Jy_box, Jy->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                ::SetJorRho( n, iv, Jy_array, mirrorfac[idim][1],
                             is_reflective[idim], psign[idim][1], idim,
                             Jy_nodal[0], Jy_fabbox );
            },
            Jz_box, Jz->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                ::SetJorRho( n, iv, Jz_array, mirrorfac[idim][2],
                             is_reflective[idim], psign[idim][2], idim,
                             Jz_nodal[0], Jz_fabbox );
            });

        }

    }

}

void
PEC::ApplyPECtoElectronPressure (
    amrex::MultiFab* Pefield,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
    const amrex::Geometry& geom,
    const int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios)
{
    amrex::Box domain_box = geom.Domain();
    if (patch_type == PatchType::coarse && (lev > 0)) {
        domain_box.coarsen(ref_ratios[lev-1]);
    }
    domain_box.convert(Pefield->ixType());

    amrex::IntVect domain_lo = domain_box.smallEnd();
    amrex::IntVect domain_hi = domain_box.bigEnd();

    amrex::IntVect Pe_nodal = Pefield->ixType().toIntVect();
    amrex::IntVect ng_fieldgather = Pefield->nGrowVect();

    // Create a copy of the domain which will be extended to include guard
    // cells for boundaries that are NOT PEC
    amrex::Box grown_domain_box = domain_box;

    amrex::GpuArray<GpuArray<bool,2>, AMREX_SPACEDIM> is_pec;
    amrex::GpuArray<GpuArray<int,2>, AMREX_SPACEDIM> mirrorfac;
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        is_pec[idim][0] = field_boundary_lo[idim] == FieldBoundaryType::PEC;
        is_pec[idim][1] = field_boundary_hi[idim] == FieldBoundaryType::PEC;
        if (!is_pec[idim][0]) { grown_domain_box.growLo(idim, ng_fieldgather[idim]); }
        if (!is_pec[idim][1]) { grown_domain_box.growHi(idim, ng_fieldgather[idim]); }

        mirrorfac[idim][0] = 2*domain_lo[idim] - (1 - Pe_nodal[idim]);
        mirrorfac[idim][1] = 2*domain_hi[idim] + (1 - Pe_nodal[idim]);
    }
    const int nComp = Pefield->nComp();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Pefield); mfi.isValid(); ++mfi) {

        // Get the multifab box including ghost cells
        Box const& fabbox = mfi.fabbox();

        // If grown_domain_box contains fabbox it means there are no PEC
        // boundaries to handle so continue to next box
        if (grown_domain_box.contains(fabbox)) { continue; }

        // Extract field data
        auto const& Pe_array = Pefield->array(mfi);

        // Loop over valid cells (i.e. cells inside the domain)
        amrex::ParallelFor(mfi.validbox(), nComp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            amrex::ignore_unused(j,k);
            // Store the array index
            const amrex::IntVect iv(AMREX_D_DECL(i,j,k));

            ::SetNeumannOnPEC(n, iv, Pe_array, mirrorfac, is_pec, fabbox);
        });
    }
}
