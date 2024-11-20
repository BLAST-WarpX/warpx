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
     *
     * \param[in] icomp        component of the Efield being updated
     *                         (0=x, 1=y, 2=z in Cartesian)
     *                         (0=r, 1=theta, 2=z in RZ)
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
                                amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_hi )
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
                    ? fbndry_lo[idim] == FieldBoundaryType::PEC
                    : fbndry_hi[idim] == FieldBoundaryType::PEC );
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
#if (defined WARPX_DIM_RZ)
                        if (icomp == 0 && idim == 0 && iside == 1) {
                            // Add radial scale so that drEr/dr = 0.
                            // This only works for the first guard cell and with
                            // Er cell centered in r.
                            const amrex::Real rguard = ijk_vec[idim] + 0.5_rt*(1._rt - is_nodal[idim]);
                            const amrex::Real rmirror = ijk_mirror[idim] + 0.5_rt*(1._rt - is_nodal[idim]);
                            sign *= rmirror/rguard;
                        }
#endif
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
     *
     * \param[in] icomp        component of the Bfield being updated
     *                         (0=x, 1=y, 2=z in Cartesian)
     *                         (0=r, 1=theta, 2=z in RZ)
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
                           amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_hi )
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
                    ? fbndry_lo[idim] == FieldBoundaryType::PEC
                    : fbndry_hi[idim] == FieldBoundaryType::PEC );
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
#if (defined WARPX_DIM_RZ)
                        if (icomp == 0 && idim == 0 && iside == 1) {
                            // Add radial scale so that drBr/dr = 0.
                            const amrex::Real rguard = ijk_vec[idim] + 0.5_rt*(1._rt - is_nodal[idim]);
                            const amrex::Real rmirror = ijk_mirror[idim] + 0.5_rt*(1._rt - is_nodal[idim]);
                            sign *= rmirror/rguard;
                        }
#endif
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
     * \brief Sets the rho or J field value in cells close to and on reflecting particle boundary
     *        or PEC field boundary. The charge/current density deposited
     *        in the guard cells are either reflected
     *        back into the simulation domain (if a reflecting particle
     *        boundary is used), or the opposite charge/current density is deposited
     *        back in the domain to capture the effect of an image charge.
     *        The charge/current density on the reflecting boundary is set to 0 while values
     *        in the guard cells are set equal (and opposite) to their mirror
     *        location inside the domain - representing image charges - in the
     *        normal (tangential) direction.
     *
     * \param[in] n                 index of the MultiFab component being updated
     * \param[in] ijk_vec           indices along the x(i), y(j), z(k) of the rho Array4
     * \param[in out] field         field data to be updated
     * \param[in] mirrorfac         mirror cell is given by mirrorfac - ijk_vec
     * \param[in] psign             Whether the field value should be flipped across the boundary
     * \param[in] is_reflective     Whether the given particle boundary is reflecting or field boundary is pec
     * \param[in] tangent_to_bndy   Whether a given direction is perpendicular to the boundary
     * \param[in] fabbox            multifab box including ghost cells
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void SetRhoOrJfieldFromPEC (const int n,
                                const amrex::IntVect & ijk_vec,
                                amrex::Array4<amrex::Real> const& field,
                                amrex::GpuArray<GpuArray<int, 2>, AMREX_SPACEDIM> const& mirrorfac,
                                amrex::GpuArray<GpuArray<amrex::Real, 2>, AMREX_SPACEDIM> const& psign,
                                amrex::GpuArray<GpuArray<bool, 2>, AMREX_SPACEDIM> const& is_reflective,
                                amrex::GpuArray<bool, AMREX_SPACEDIM> const& tangent_to_bndy,
                                amrex::Box const& fabbox)
    {
        // The boundary is handled in 2 steps:
        // 1) The cells internal to the domain are updated using the
        //    current deposited in the guard cells
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            for (int iside = 0; iside < 2; ++iside)
            {
                if (!is_reflective[idim][iside]) { continue; }

                // Get the mirror guard cell index
                amrex::IntVect iv_mirror = ijk_vec;
                iv_mirror[idim] = mirrorfac[idim][iside] - ijk_vec[idim];

                // On the PEC boundary the charge/current density is set to 0
                if (ijk_vec == iv_mirror) {
                    field(ijk_vec, n) = 0._rt;
                // otherwise update the internal cell if the mirror guard cell exists
                } else if (fabbox.contains(iv_mirror)) {
                    field(ijk_vec,n) += psign[idim][iside] * field(iv_mirror,n);
                }
            }
        }
        // 2) The guard cells are updated with the appropriate image
        //    charge based on the charge/current in the valid cells
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            for (int iside = 0; iside < 2; ++iside)
            {
                if (!is_reflective[idim][iside]) { continue; }

                amrex::IntVect iv_mirror = ijk_vec;
                iv_mirror[idim] = mirrorfac[idim][iside] - ijk_vec[idim];
                if (ijk_vec != iv_mirror && fabbox.contains(iv_mirror))
                {
                    if (tangent_to_bndy[idim]) {
                        field(iv_mirror, n) = -field(ijk_vec, n);
                    } else {
                        field(iv_mirror, n) = field(ijk_vec, n);
                    }
                }
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
                                           Ex, Ex_nodal, fbndry_lo, fbndry_hi);
            },
            tey, nComp_y,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 1;
                ::SetEfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                           Ey, Ey_nodal, fbndry_lo, fbndry_hi);
            },
            tez, nComp_z,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 2;
                ::SetEfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                           Ez, Ez_nodal, fbndry_lo, fbndry_hi);
            }
        );
    }
}


void
PEC::ApplyPECtoBfield (
    std::array<amrex::MultiFab*, 3> Bfield,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
    const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
    const amrex::IntVect& ng_fieldgather, const amrex::Geometry& geom,
    const int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios)
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
        amrex::Box const& tbx = mfi.tilebox(Bfield[0]->ixType().toIntVect(), ng_fieldgather);
        amrex::Box const& tby = mfi.tilebox(Bfield[1]->ixType().toIntVect(), ng_fieldgather);
        amrex::Box const& tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect(), ng_fieldgather);

        // loop over cells and update fields
        amrex::ParallelFor(
            tbx, nComp_x,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 0;
                ::SetBfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                     Bx, Bx_nodal, fbndry_lo, fbndry_hi);
            },
            tby, nComp_y,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 1;
                ::SetBfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                     By, By_nodal, fbndry_lo, fbndry_hi);
            },
            tbz, nComp_z,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j,k);
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                const int icomp = 2;
                ::SetBfieldOnPEC(icomp, domain_lo, domain_hi, iv, n,
                                     Bz, Bz_nodal, fbndry_lo, fbndry_hi);
            }
        );
    }
}


/**
 * \brief Sets the rho field value in cells close to and inside a PEC boundary.
 *        The charge density deposited in the guard cells are either reflected
 *        back into the simulation domain (if a reflecting particle
 *        boundary is used), or the opposite charge density is deposited
 *        back in the domain to capture the effect of an image charge.
 *        The charge density on the PEC boundary is set to 0 while values
 *        in the guard cells are set equal and opposite to their mirror
 *        location inside the domain - representing image charges.
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

    amrex::IntVect domain_lo = domain_box.smallEnd();
    amrex::IntVect domain_hi = domain_box.bigEnd();

    amrex::IntVect rho_nodal = rho->ixType().toIntVect();
    amrex::IntVect ng_fieldgather = rho->nGrowVect();

    // Create a copy of the domain which will be extended to include guard
    // cells for boundaries that are NOT PEC
    amrex::Box grown_domain_box = domain_box;

    amrex::GpuArray<GpuArray<bool,2>, AMREX_SPACEDIM> is_reflective;
    amrex::GpuArray<bool, AMREX_SPACEDIM> is_tangent_to_bndy;
    amrex::GpuArray<GpuArray<amrex::Real,2>, AMREX_SPACEDIM> psign;
    amrex::GpuArray<GpuArray<int,2>, AMREX_SPACEDIM> mirrorfac;
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        is_reflective[idim][0] = ( particle_boundary_lo[idim] == ParticleBoundaryType::Reflecting)
                              || ( particle_boundary_lo[idim] == ParticleBoundaryType::Thermal)
                              || ( field_boundary_lo[idim] == FieldBoundaryType::PEC);
        is_reflective[idim][1] = ( particle_boundary_hi[idim] == ParticleBoundaryType::Reflecting)
                              || ( particle_boundary_hi[idim] == ParticleBoundaryType::Thermal)
                              || ( field_boundary_hi[idim] == FieldBoundaryType::PEC);
        if (!is_reflective[idim][0]) { grown_domain_box.growLo(idim, ng_fieldgather[idim]); }
        if (!is_reflective[idim][1]) { grown_domain_box.growHi(idim, ng_fieldgather[idim]); }

        // rho values inside guard cells are updated the same as tangential
        // components of the current density
        is_tangent_to_bndy[idim] = true;

        psign[idim][0] = ((particle_boundary_lo[idim] == ParticleBoundaryType::Reflecting)
                        ||(particle_boundary_lo[idim] == ParticleBoundaryType::Thermal))
                         ? 1._rt : -1._rt;
        psign[idim][1] = ((particle_boundary_hi[idim] == ParticleBoundaryType::Reflecting)
                        ||(particle_boundary_hi[idim] == ParticleBoundaryType::Thermal))
                         ? 1._rt : -1._rt;
        mirrorfac[idim][0] = 2*domain_lo[idim] - (1 - rho_nodal[idim]);
        mirrorfac[idim][1] = 2*domain_hi[idim] + (1 - rho_nodal[idim]);
    }
    const int nComp = rho->nComp();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*rho); mfi.isValid(); ++mfi) {

        // Get the multifab box including ghost cells
        Box const& fabbox = mfi.fabbox();

        // If grown_domain_box contains fabbox it means there are no PEC
        // boundaries to handle so continue to next box
        if (grown_domain_box.contains(fabbox)) { continue; }

        // Extract field data
        auto const& rho_array = rho->array(mfi);

        // Loop over valid cells (i.e. cells inside the domain)
        amrex::ParallelFor(mfi.validbox(), nComp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            amrex::ignore_unused(j,k);
            // Store the array index
            const amrex::IntVect iv(AMREX_D_DECL(i,j,k));

            ::SetRhoOrJfieldFromPEC(
                n, iv, rho_array, mirrorfac, psign, is_reflective,
                is_tangent_to_bndy, fabbox
            );
        });
    }
}


void
PEC::ApplyReflectiveBoundarytoJfield(
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

    // Note: force domain box to be nodal to simplify the mirror cell
    // calculations below
    domain_box.convert(IntVect::TheNodeVector());

    amrex::IntVect domain_lo = domain_box.smallEnd();
    amrex::IntVect domain_hi = domain_box.bigEnd();

    // Get the nodal flag for each current component since it is needed to
    // determine the appropriate mirrorfac values below
    amrex::IntVect Jx_nodal = Jx->ixType().toIntVect();
    amrex::IntVect Jy_nodal = Jy->ixType().toIntVect();
    amrex::IntVect Jz_nodal = Jz->ixType().toIntVect();

    // Create a copy of the domain for each component of J which will
    // be extended to include guard cells for boundaries that are NOT PEC
    amrex::Box grown_domain_box = domain_box;

    // The number of ghost cells used is the same for nodal and cell-centered
    // directions of the current density multifab
    const amrex::IntVect ng_fieldgather = Jx->nGrowVect();

    amrex::GpuArray<GpuArray<bool, 2>, AMREX_SPACEDIM> is_reflective;
    amrex::GpuArray<GpuArray<bool, AMREX_SPACEDIM>, 3> is_tangent_to_bndy;
    amrex::GpuArray<GpuArray<GpuArray<amrex::Real, 2>, AMREX_SPACEDIM>, 3> psign;
    amrex::GpuArray<GpuArray<GpuArray<int, 2>, AMREX_SPACEDIM>, 3> mirrorfac;
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        is_reflective[idim][0] = ( particle_boundary_lo[idim] == ParticleBoundaryType::Reflecting)
                              || ( particle_boundary_lo[idim] == ParticleBoundaryType::Thermal)
                              || ( field_boundary_lo[idim] == FieldBoundaryType::PEC);
        is_reflective[idim][1] = ( particle_boundary_hi[idim] == ParticleBoundaryType::Reflecting)
                              || ( particle_boundary_hi[idim] == ParticleBoundaryType::Thermal)
                              || ( field_boundary_hi[idim] == FieldBoundaryType::PEC);
        if (!is_reflective[idim][0]) { grown_domain_box.growLo(idim, ng_fieldgather[idim]); }
        if (!is_reflective[idim][1]) { grown_domain_box.growHi(idim, ng_fieldgather[idim]); }

        for (int icomp=0; icomp < 3; ++icomp) {
            // Set the psign value correctly for each current component for each
            // simulation direction
#if (defined WARPX_DIM_1D_Z)
            // For 1D : icomp=0 and icomp=1 (Ex and Ey are tangential to the z boundary)
            //          The logic below ensures that the flags are set right for 1D
            is_tangent_to_bndy[icomp][idim] = (icomp != (idim+2));
#elif (defined WARPX_DIM_XZ) || (defined WARPX_DIM_RZ)
            // For 2D : for icomp==1, (Ey in XZ, Etheta in RZ),
            //          icomp=1 is tangential to both x and z boundaries
            //          The logic below ensures that the flags are set right for 2D
            is_tangent_to_bndy[icomp][idim] = (icomp != AMREX_SPACEDIM*idim);
#else
            is_tangent_to_bndy[icomp][idim] = (icomp != idim);
#endif

            if (is_tangent_to_bndy[icomp][idim]){
                psign[icomp][idim][0] = ( (particle_boundary_lo[idim] == ParticleBoundaryType::Reflecting)
                                        ||(particle_boundary_lo[idim] == ParticleBoundaryType::Thermal))
                                        ? 1._rt : -1._rt;
                psign[icomp][idim][1] = ( (particle_boundary_hi[idim] == ParticleBoundaryType::Reflecting)
                                        ||(particle_boundary_hi[idim] == ParticleBoundaryType::Thermal))
                                        ? 1._rt : -1._rt;
            }
            else {
                psign[icomp][idim][0] = ( (particle_boundary_lo[idim] == ParticleBoundaryType::Reflecting)
                                        ||(particle_boundary_lo[idim] == ParticleBoundaryType::Thermal))
                                        ? -1._rt : 1._rt;
                psign[icomp][idim][1] = ( (particle_boundary_hi[idim] == ParticleBoundaryType::Reflecting)
                                        ||(particle_boundary_hi[idim] == ParticleBoundaryType::Thermal))
                                        ? -1._rt : 1._rt;
            }
        }
        // Set the correct mirror cell calculation for each current
        // component. Seeing as domain was set to be nodal, nodal fields have
        // boundaries on domain_lo and domain_hi, but cell-centered fields
        // have valid cells ranging from domain_lo to domain_hi - 1.
        mirrorfac[0][idim][0] = 2*domain_lo[idim] - (1 - Jx_nodal[idim]);
        mirrorfac[0][idim][1] = 2*domain_hi[idim] - (1 - Jx_nodal[idim]);
        mirrorfac[1][idim][0] = 2*domain_lo[idim] - (1 - Jy_nodal[idim]);
        mirrorfac[1][idim][1] = 2*domain_hi[idim] - (1 - Jy_nodal[idim]);
        mirrorfac[2][idim][0] = 2*domain_lo[idim] - (1 - Jz_nodal[idim]);
        mirrorfac[2][idim][1] = 2*domain_hi[idim] - (1 - Jz_nodal[idim]);
    }

    // Each current component is handled separately below, starting with Jx.
    grown_domain_box.convert(Jx_nodal);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Jx); mfi.isValid(); ++mfi) {

        // Get the multifab box including ghost cells
        Box const& fabbox = mfi.fabbox();

        // If grown_domain_box contains fabbox it means there are no PEC
        // boundaries to handle so continue to next box
        if (grown_domain_box.contains(fabbox)) { continue; }

        // Extract field data
        auto const& Jx_array = Jx->array(mfi);

        // Loop over valid cells (i.e. cells inside the domain)
        amrex::ParallelFor(mfi.validbox(), Jx->nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            amrex::ignore_unused(j,k);
            // Store the array index
            const amrex::IntVect iv(AMREX_D_DECL(i,j,k));

            ::SetRhoOrJfieldFromPEC(
                n, iv, Jx_array, mirrorfac[0], psign[0], is_reflective,
                is_tangent_to_bndy[0], fabbox
            );
        });
    }

    // Handle Jy.
    grown_domain_box.convert(Jy_nodal);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Jy); mfi.isValid(); ++mfi) {

        // Get the multifab box including ghost cells
        Box const& fabbox = mfi.fabbox();

        // If grown_domain_box contains fabbox it means there are no PEC
        // boundaries to handle so continue to next box
        if (grown_domain_box.contains(fabbox)) { continue; }

        // Extract field data
        auto const& Jy_array = Jy->array(mfi);

        // Loop over valid cells (i.e. cells inside the domain)
        amrex::ParallelFor(mfi.validbox(), Jy->nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            amrex::ignore_unused(j,k);
            // Store the array index
            const amrex::IntVect iv(AMREX_D_DECL(i,j,k));

            ::SetRhoOrJfieldFromPEC(
                n, iv, Jy_array, mirrorfac[1], psign[1], is_reflective,
                is_tangent_to_bndy[1], fabbox
            );
        });
    }

    // Handle Jz.
    grown_domain_box.convert(Jz_nodal);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Jz); mfi.isValid(); ++mfi) {

        // Get the multifab box including ghost cells
        Box const& fabbox = mfi.fabbox();

        // If grown_domain_box contains fabbox it means there are no PEC
        // boundaries to handle so continue to next box
        if (grown_domain_box.contains(fabbox)) { continue; }

        // Extract field data
        auto const& Jz_array = Jz->array(mfi);

        // Loop over valid cells (i.e. cells inside the domain)
        amrex::ParallelFor(mfi.validbox(), Jz->nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            amrex::ignore_unused(j,k);
            // Store the array index
            const amrex::IntVect iv(AMREX_D_DECL(i,j,k));

            ::SetRhoOrJfieldFromPEC(
                n, iv, Jz_array, mirrorfac[2], psign[2], is_reflective,
                is_tangent_to_bndy[2], fabbox
            );
        });
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
