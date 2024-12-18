#include "BoundaryConditions/PEC_Insulator.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

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
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>

namespace
{
    /**
     * \brief At the specified grid location, apply either the PEC or insulator boundary condition if
     *        the cell is on the boundary or in the guard cells.
     *
     * \param[in] icomp        component of the field being updated
     *                         (0=x, 1=y, 2=z in Cartesian)
     *                         (0=r, 1=theta, 2=z in RZ)
     * \param[in] dom_lo       index value of the lower domain boundary (cell-centered)
     * \param[in] dom_hi       index value of the higher domain boundary (cell-centered)
     * \param[in] ijk_vec      indices along the x(i), y(j), z(k) of field Array4
     * \param[in] n            index of the MultiFab component being updated
     * \param[in] field        field data to be updated if (ijk) is at the boundary
     *                         or a guard cell
     * \param[in] E_like       whether the field behaves like E field or B field
     * \param[in] is_nodal     staggering of the field data being updated.
     * \param[in] is_insulator_lo Specifies whether lower boundaries are insulators
     * \param[in] is_insulator_hi Specifies whether upper boundaries are insulators
     * \param[in] field_lo     the values of the field for the lower insulator boundary cell
     * \param[in] field_hi     the values of the field for the upper insulator boundary cell
     * \param[in] set_field_lo whether to set the field for the direction on the lower boundary
     * \param[in] set_field_hi whether to set the field for the direction on the upper boundary
     * \param[in] fbndry_lo    specified values of the field at the lower boundaries in the insulator
     * \param[in] fbndry_hi    specified values of the field at the upper boundaries in the insulator
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void SetFieldOnPEC_Insulator (int icomp,
                                  amrex::IntVect const & dom_lo,
                                  amrex::IntVect const & dom_hi,
                                  amrex::IntVect const & ijk_vec, int n,
                                  amrex::Array4<amrex::Real> const & field,
                                  bool const E_like,
                                  amrex::IntVect const & is_nodal,
                                  amrex::IntVect const & is_insulator_lo,
                                  amrex::IntVect const & is_insulator_hi,
                                  amrex::RealVect const & field_lo,
                                  amrex::RealVect const & field_hi,
                                  amrex::IntVect const & set_field_lo,
                                  amrex::IntVect const & set_field_hi,
                                  amrex::GpuArray<FieldBoundaryType, 3> const fbndry_lo,
                                  amrex::GpuArray<FieldBoundaryType, 3> const fbndry_hi)
    {
        using namespace amrex::literals;
        amrex::IntVect ijk_next = ijk_vec;
        amrex::IntVect ijk_nextp1 = ijk_vec;
        amrex::IntVect ijk_mirror = ijk_vec;
        bool OnBoundary = false;
        bool GuardCell = false;
        bool isInsulatorBoundary = false;
        amrex::Real sign = +1._rt;
        bool is_normal_to_boundary;
        amrex::Real field_value = 0._rt;
        bool set_field = false;
        // Loop over all dimensions
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            // Loop over sides, iside = -1 (lo), iside = +1 (hi)
            for (int iside = -1; iside <= +1; iside += 2) {
                bool const isPEC_InsulatorBoundary = ( (iside == -1)
                    ? fbndry_lo[idim] == FieldBoundaryType::PECInsulator
                    : fbndry_hi[idim] == FieldBoundaryType::PECInsulator );
                if (isPEC_InsulatorBoundary) {
                    isInsulatorBoundary = ( (iside == -1)
                        ? is_insulator_lo[idim] == 1
                        : is_insulator_hi[idim] == 1 );
                }
                if (isPEC_InsulatorBoundary) {
                    // Calculates the number of grid points ijk_vec is beyond the
                    // domain boundary i.e. a value of +1 means the current cell is
                    // outside of the simulation domain by 1 cell. Note that the high
                    // side domain boundary is between cell dom_hi and dom_hi+1 for cell
                    // centered grids and on cell dom_hi+1 for nodal grid. This is why
                    // (dom_hi[idim] + is_nodal[idim]) is used.
                    int const ig = ((iside == -1) ? (dom_lo[idim] - ijk_vec[idim])
                                                  : (ijk_vec[idim] - (dom_hi[idim] + is_nodal[idim])));

#if (defined WARPX_DIM_XZ) || (defined WARPX_DIM_RZ)
                    // For 2D : for icomp==1, (Fy in XZ, Ftheta in RZ),
                    //          icomp=1 is not normal to x or z boundary
                    //          The logic below ensures that the flags are set right for 2D
                    is_normal_to_boundary = (icomp == (2*idim));
#elif (defined WARPX_DIM_1D_Z)
                    // For 1D : icomp=0 and icomp=1 (Fx and Fy are not normal to the z boundary)
                    //          The logic below ensures that the flags are set right for 1D
                    is_normal_to_boundary = (icomp == 2);
#else
                    is_normal_to_boundary = (icomp == idim);
#endif

                    if (ig == 0) {
                        // Check if field is on the boundary
                        if (is_nodal[idim] == 1) {
                            OnBoundary = true;
                        }
                    } else if (ig > 0) {
                        GuardCell = true;

                        // Location of the next cells inward
                        ijk_next[idim] = ijk_vec[idim] - ig*iside;
                        ijk_nextp1[idim] = ijk_next[idim] - ig*iside;

                        // Mirror location inside the domain by "ig" number of cells
                        ijk_mirror[idim] = ( (iside == -1)
                                        ? (dom_lo[idim] + ig - (1 - is_nodal[idim]))
                                        : (dom_hi[idim] - ig + 1));

                        // Check for components with even symmetry.
                        // True for E_like and tangential, and B_like and normal
                        if (E_like ^ is_normal_to_boundary) { sign *= -1._rt; }

                        field_value = ( (iside == -1) ? field_lo[idim] : field_hi[idim] );
                        set_field = ( (iside == -1) ? set_field_lo[idim]==1 : set_field_hi[idim]==1 );

#if (defined WARPX_DIM_RZ)
                        if (idim == 0 && iside == +1) {
                            // Upper radial boundary
                            amrex::Real const rguard = ijk_vec[idim] + 0.5_rt*(1._rt - is_nodal[idim]);
                            if (icomp == 0) {
                                // Add radial scale so that the divergence, drFr/dr, is 0.
                                // This only works for the first guard cell and with
                                // Fr cell centered in r.
                                amrex::Real const rmirror = ijk_mirror[idim] + 0.5_rt*(1._rt - is_nodal[idim]);
                                // Calculate radial scale factor
                                sign *= rmirror/rguard;
                            }
                            if (isInsulatorBoundary) {
                                // Apply radial scale factor
                                field_value *= dom_hi[idim]/rguard;
                            }
                        }
#endif
                    }
                } // is pec_insulator boundary
            } // loop over iside
        } // loop over dimensions

        if (isInsulatorBoundary) {
            if (is_normal_to_boundary) {
                // The value on the boundary is left unmodified
                // The values in the guard cells are extrapolated
                if (GuardCell) {
                    field(ijk_vec, n) = 2._rt*field(ijk_next, n) - field(ijk_nextp1, n);
                }
            } else if ((OnBoundary || GuardCell) && set_field) {
                field(ijk_vec, n) = field_value;
            } else if (GuardCell) {
                field(ijk_vec, n) = 2._rt*field(ijk_next, n) - field(ijk_nextp1, n);
            }
        } else {
            if (OnBoundary && (E_like ^ is_normal_to_boundary)) {
                // If ijk_vec is on a boundary, set to zero if
                // E_like and tangential or B_like and normal
                field(ijk_vec,n) = 0._rt;
            } else if (GuardCell) {
                // Fnormal and Ftangential is set opposite and equal to the value
                // in the mirror location, respectively.
                field(ijk_vec,n) = sign * field(ijk_mirror,n);
            }
        }
    }
}


bool
PEC_Insulator::ReadTangentialFieldParser (amrex::ParmParse const & pp_insulator,
                                          std::unique_ptr<amrex::Parser> & parser,
                                          std::string const & input_name,
                                          std::string const & coord1,
                                          std::string const & coord2)
{
    std::string str = "0";
    bool const specified = utils::parser::Query_parserString(pp_insulator, input_name, str);
    parser = std::make_unique<amrex::Parser>(utils::parser::makeParser(str, {coord1, coord2, "t"}));
    return specified;
}

PEC_Insulator::PEC_Insulator ()
{

    amrex::ParmParse const pp_insulator("insulator");

#if (AMREX_SPACEDIM > 1)
    std::string str_area_x_lo = "0";
    std::string str_area_x_hi = "0";
    utils::parser::Query_parserString( pp_insulator, "area_x_lo(y,z)", str_area_x_lo);
    utils::parser::Query_parserString( pp_insulator, "area_x_hi(y,z)", str_area_x_hi);
    m_insulator_area_lo.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_x_lo, {"y", "z"})));
    m_insulator_area_hi.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_x_hi, {"y", "z"})));

    m_set_B_x_lo |= ReadTangentialFieldParser(pp_insulator, m_By_x_lo, "By_x_lo(y,z,t)", "y", "z");
    m_set_B_x_lo |= ReadTangentialFieldParser(pp_insulator, m_Bz_x_lo, "Bz_x_lo(y,z,t)", "y", "z");
    m_set_B_x_hi |= ReadTangentialFieldParser(pp_insulator, m_By_x_hi, "By_x_hi(y,z,t)", "y", "z");
    m_set_B_x_hi |= ReadTangentialFieldParser(pp_insulator, m_Bz_x_hi, "Bz_x_hi(y,z,t)", "y", "z");

    m_set_E_x_lo |= ReadTangentialFieldParser(pp_insulator, m_Ey_x_lo, "Ey_x_lo(y,z,t)", "y", "z");
    m_set_E_x_lo |= ReadTangentialFieldParser(pp_insulator, m_Ez_x_lo, "Ez_x_lo(y,z,t)", "y", "z");
    m_set_E_x_hi |= ReadTangentialFieldParser(pp_insulator, m_Ey_x_hi, "Ey_x_hi(y,z,t)", "y", "z");
    m_set_E_x_hi |= ReadTangentialFieldParser(pp_insulator, m_Ez_x_hi, "Ez_x_hi(y,z,t)", "y", "z");
#endif
#if defined(WARPX_DIM_3D)
    std::string str_area_y_lo = "0";
    std::string str_area_y_hi = "0";
    utils::parser::Query_parserString( pp_insulator, "area_y_lo(x,z)", str_area_y_lo);
    utils::parser::Query_parserString( pp_insulator, "area_y_hi(x,z)", str_area_y_hi);
    m_insulator_area_lo.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_y_lo, {"x", "z"})));
    m_insulator_area_hi.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_y_hi, {"x", "z"})));

    m_set_B_y_lo |= ReadTangentialFieldParser(pp_insulator, m_Bx_y_lo, "Bx_y_lo(x,z,t)", "x", "z");
    m_set_B_y_lo |= ReadTangentialFieldParser(pp_insulator, m_Bz_y_lo, "Bz_y_lo(x,z,t)", "x", "z");
    m_set_B_y_hi |= ReadTangentialFieldParser(pp_insulator, m_Bx_y_hi, "Bx_y_hi(x,z,t)", "x", "z");
    m_set_B_y_hi |= ReadTangentialFieldParser(pp_insulator, m_Bz_y_hi, "Bz_y_hi(x,z,t)", "x", "z");

    m_set_E_y_lo |= ReadTangentialFieldParser(pp_insulator, m_Ex_y_lo, "Ex_y_lo(x,z,t)", "x", "z");
    m_set_E_y_lo |= ReadTangentialFieldParser(pp_insulator, m_Ez_y_lo, "Ez_y_lo(x,z,t)", "x", "z");
    m_set_E_y_hi |= ReadTangentialFieldParser(pp_insulator, m_Ex_y_hi, "Ex_y_hi(x,z,t)", "x", "z");
    m_set_E_y_hi |= ReadTangentialFieldParser(pp_insulator, m_Ez_y_hi, "Ez_y_hi(x,z,t)", "x", "z");
#endif

    std::string str_area_z_lo = "0";
    std::string str_area_z_hi = "0";
    utils::parser::Query_parserString( pp_insulator, "area_z_lo(x,y)", str_area_z_lo);
    utils::parser::Query_parserString( pp_insulator, "area_z_hi(x,y)", str_area_z_hi);
    m_insulator_area_lo.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_z_lo, {"x", "y"})));
    m_insulator_area_hi.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_z_hi, {"x", "y"})));

    m_set_B_z_lo |= ReadTangentialFieldParser(pp_insulator, m_Bx_z_lo, "Bx_z_lo(x,y,t)", "x", "y");
    m_set_B_z_lo |= ReadTangentialFieldParser(pp_insulator, m_By_z_lo, "By_z_lo(x,y,t)", "x", "y");
    m_set_B_z_hi |= ReadTangentialFieldParser(pp_insulator, m_Bx_z_hi, "Bx_z_hi(x,y,t)", "x", "y");
    m_set_B_z_hi |= ReadTangentialFieldParser(pp_insulator, m_By_z_hi, "By_z_hi(x,y,t)", "x", "y");

    m_set_E_z_lo |= ReadTangentialFieldParser(pp_insulator, m_Ex_z_lo, "Ex_z_lo(x,y,t)", "x", "y");
    m_set_E_z_lo |= ReadTangentialFieldParser(pp_insulator, m_Ey_z_lo, "Ey_z_lo(x,y,t)", "x", "y");
    m_set_E_z_hi |= ReadTangentialFieldParser(pp_insulator, m_Ex_z_hi, "Ex_z_hi(x,y,t)", "x", "y");
    m_set_E_z_hi |= ReadTangentialFieldParser(pp_insulator, m_Ey_z_hi, "Ey_z_hi(x,y,t)", "x", "y");

}

void
PEC_Insulator::ApplyPEC_InsulatortoEfield (
    std::array<amrex::MultiFab*, 3> Efield,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
    amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
    int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
    amrex::Real time,
    bool split_pml_field)
{
    bool const E_like = true;
    ApplyPEC_InsulatortoField(Efield, field_boundary_lo, field_boundary_hi, ng_fieldgather, geom,
                              lev, patch_type, ref_ratios, time, split_pml_field,
                              E_like,
#if (AMREX_SPACEDIM > 1)
                              m_set_E_x_lo, m_set_E_x_hi,
                              m_Ey_x_lo, m_Ez_x_lo, m_Ey_x_hi, m_Ez_x_hi,
#endif
#if defined(WARPX_DIM_3D)
                              m_set_E_y_lo, m_set_E_y_hi,
                              m_Ex_y_lo, m_Ez_y_lo, m_Ex_y_hi, m_Ez_y_hi,
#endif
                              m_set_E_z_lo, m_set_E_z_hi,
                              m_Ex_z_lo, m_Ey_z_lo, m_Ex_z_hi, m_Ey_z_hi);
}


void
PEC_Insulator::ApplyPEC_InsulatortoBfield (
    std::array<amrex::MultiFab*, 3> Bfield,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
    amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
    int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
    amrex::Real time)
{
    bool const E_like = false;
    bool const split_pml_field = false;
    ApplyPEC_InsulatortoField(Bfield, field_boundary_lo, field_boundary_hi, ng_fieldgather, geom,
                              lev, patch_type, ref_ratios, time, split_pml_field,
                              E_like,
#if (AMREX_SPACEDIM > 1)
                              m_set_B_x_lo, m_set_B_x_hi,
                              m_By_x_lo, m_Bz_x_lo, m_By_x_hi, m_Bz_x_hi,
#endif
#if defined(WARPX_DIM_3D)
                              m_set_B_y_lo, m_set_B_y_hi,
                              m_Bx_y_lo, m_Bz_y_lo, m_Bx_y_hi, m_Bz_y_hi,
#endif
                              m_set_B_z_lo, m_set_B_z_hi,
                              m_Bx_z_lo, m_By_z_lo, m_Bx_z_hi, m_By_z_hi);
}


void
PEC_Insulator::ApplyPEC_InsulatortoField (
    std::array<amrex::MultiFab*, 3> field,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
    amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
    int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
    amrex::Real time,
    bool split_pml_field,
    bool E_like,
#if (AMREX_SPACEDIM > 1)
    bool set_F_x_lo, bool set_F_x_hi,
    std::unique_ptr<amrex::Parser> const & a_Fy_x_lo, std::unique_ptr<amrex::Parser> const & a_Fz_x_lo,
    std::unique_ptr<amrex::Parser> const & a_Fy_x_hi, std::unique_ptr<amrex::Parser> const & a_Fz_x_hi,
#endif
#if defined(WARPX_DIM_3D)
    bool set_F_y_lo, bool set_F_y_hi,
    std::unique_ptr<amrex::Parser> const & a_Fx_y_lo, std::unique_ptr<amrex::Parser> const & a_Fz_y_lo,
    std::unique_ptr<amrex::Parser> const & a_Fx_y_hi, std::unique_ptr<amrex::Parser> const & a_Fz_y_hi,
#endif
    bool set_F_z_lo, bool set_F_z_hi,
    std::unique_ptr<amrex::Parser> const & a_Fx_z_lo, std::unique_ptr<amrex::Parser> const & a_Fy_z_lo,
    std::unique_ptr<amrex::Parser> const & a_Fx_z_hi, std::unique_ptr<amrex::Parser> const & a_Fy_z_hi)
{
    using namespace amrex::literals;
    amrex::Box domain_box = geom.Domain();
    if (patch_type == PatchType::coarse && (lev > 0)) {
        domain_box.coarsen(ref_ratios[lev-1]);
    }
    amrex::IntVect const domain_lo = domain_box.smallEnd();
    amrex::IntVect const domain_hi = domain_box.bigEnd();
    amrex::GpuArray<FieldBoundaryType, 3> fbndry_lo;
    amrex::GpuArray<FieldBoundaryType, 3> fbndry_hi;
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        fbndry_lo[idim] = field_boundary_lo[idim];
        fbndry_hi[idim] = field_boundary_hi[idim];
    }

#if (AMREX_SPACEDIM > 1)
    amrex::ParserExecutor<2> const area_parsers_x_lo = m_insulator_area_lo[0]->compile<2>();
    amrex::ParserExecutor<2> const area_parsers_x_hi = m_insulator_area_hi[0]->compile<2>();
#endif
#if defined(WARPX_DIM_3D)
    amrex::ParserExecutor<2> const area_parsers_y_lo = m_insulator_area_lo[1]->compile<2>();
    amrex::ParserExecutor<2> const area_parsers_y_hi = m_insulator_area_hi[1]->compile<2>();
#endif
    amrex::ParserExecutor<2> const area_parsers_z_lo = m_insulator_area_lo[WARPX_ZINDEX]->compile<2>();
    amrex::ParserExecutor<2> const area_parsers_z_hi = m_insulator_area_hi[WARPX_ZINDEX]->compile<2>();

#if (AMREX_SPACEDIM > 1)
    amrex::ParserExecutor<3> const Fy_x_lo_parser = a_Fy_x_lo->compile<3>();
    amrex::ParserExecutor<3> const Fz_x_lo_parser = a_Fz_x_lo->compile<3>();
    amrex::ParserExecutor<3> const Fy_x_hi_parser = a_Fy_x_hi->compile<3>();
    amrex::ParserExecutor<3> const Fz_x_hi_parser = a_Fz_x_hi->compile<3>();
#endif
#if defined(WARPX_DIM_3D)
    amrex::ParserExecutor<3> const Fx_y_lo_parser = a_Fx_y_lo->compile<3>();
    amrex::ParserExecutor<3> const Fz_y_lo_parser = a_Fz_y_lo->compile<3>();
    amrex::ParserExecutor<3> const Fx_y_hi_parser = a_Fx_y_hi->compile<3>();
    amrex::ParserExecutor<3> const Fz_y_hi_parser = a_Fz_y_hi->compile<3>();
#endif
    amrex::ParserExecutor<3> const Fx_z_lo_parser = a_Fx_z_lo->compile<3>();
    amrex::ParserExecutor<3> const Fy_z_lo_parser = a_Fy_z_lo->compile<3>();
    amrex::ParserExecutor<3> const Fx_z_hi_parser = a_Fx_z_hi->compile<3>();
    amrex::ParserExecutor<3> const Fy_z_hi_parser = a_Fy_z_hi->compile<3>();

    amrex::IntVect const Fx_nodal = field[0]->ixType().toIntVect();
    amrex::IntVect const Fy_nodal = field[1]->ixType().toIntVect();
    amrex::IntVect const Fz_nodal = field[2]->ixType().toIntVect();
    // For each field multifab, apply boundary condition to ncomponents
    // If not split field, the boundary condition is applied to the regular field used in Maxwell's eq.
    // If split_pml_field is true, then boundary condition is applied to all the split field components.
    int const nComp_x = field[0]->nComp();
    int const nComp_y = field[1]->nComp();
    int const nComp_z = field[2]->nComp();

    std::array<amrex::Real,3> const & dx = WarpX::CellSize(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*field[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        // Extract field data
        amrex::Array4<amrex::Real> const & Fx = field[0]->array(mfi);
        amrex::Array4<amrex::Real> const & Fy = field[1]->array(mfi);
        amrex::Array4<amrex::Real> const & Fz = field[2]->array(mfi);

        // Extract tileboxes for which to loop
        // if split field, the box includes nodal flag
        // For E-field used in Maxwell's update, nodal flag plus cells that particles
        // gather fields from in the guard-cell region are included.
        // Note that for simulations without particles or laser, ng_field_gather is 0
        // and the guard-cell values of the E-field multifab will not be modified.
        amrex::Box const & tex = (split_pml_field) ? mfi.tilebox(field[0]->ixType().toIntVect())
                                                   : mfi.tilebox(field[0]->ixType().toIntVect(), ng_fieldgather);
        amrex::Box const & tey = (split_pml_field) ? mfi.tilebox(field[1]->ixType().toIntVect())
                                                   : mfi.tilebox(field[1]->ixType().toIntVect(), ng_fieldgather);
        amrex::Box const & tez = (split_pml_field) ? mfi.tilebox(field[2]->ixType().toIntVect())
                                                   : mfi.tilebox(field[2]->ixType().toIntVect(), ng_fieldgather);

        const amrex::XDim3 xyzmin_x = WarpX::LowerCorner(tex, lev, 0._rt);
        const amrex::XDim3 xyzmin_y = WarpX::LowerCorner(tey, lev, 0._rt);
        const amrex::XDim3 xyzmin_z = WarpX::LowerCorner(tez, lev, 0._rt);
        amrex::IntVect const lo_x = tex.smallEnd();
        amrex::IntVect const lo_y = tey.smallEnd();
        amrex::IntVect const lo_z = tez.smallEnd();

        // loop over cells and update fields
        amrex::ParallelFor(
            tex, nComp_x,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j, k);

                amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                amrex::Real const shiftx = (Fx_nodal[0] ? 0._rt : 0.5_rt);
                amrex::Real const x = (AMREX_SPACEDIM > 1 ? xyzmin_x.x + (iv[0] - lo_x[0] + shiftx)*dx[0] : 0._rt);
                amrex::Real const shifty = (AMREX_SPACEDIM == 3 ? (Fx_nodal[1] ? 0._rt : 0.5_rt) : 0._rt);
                amrex::Real const y = (AMREX_SPACEDIM == 3 ? xyzmin_x.y + (iv[1] - lo_x[1] + shifty)*dx[1] : 0._rt);
#if (AMREX_SPACEDIM > 1)
                amrex::Real const shiftz = (Fx_nodal[WARPX_ZINDEX] ? 0._rt : 0.5_rt);
                amrex::Real const z = xyzmin_x.z + (iv[WARPX_ZINDEX] - lo_x[WARPX_ZINDEX] + shiftz)*dx[2];
#endif

                amrex::IntVect is_insulator_lo;
                amrex::IntVect is_insulator_hi;
                amrex::RealVect F_lo, F_hi;
                amrex::IntVect set_field_lo;
                amrex::IntVect set_field_hi;
#if (AMREX_SPACEDIM > 1)
                is_insulator_lo[0] = (area_parsers_x_lo(y, z) > 0._rt);
                is_insulator_hi[0] = (area_parsers_x_hi(y, z) > 0._rt);
                F_lo[0] = 0._rt;  // Will be unused
                F_hi[0] = 0._rt;  // Will be unused
                set_field_lo[0] = 0;  // Will be unused
                set_field_hi[0] = 0;  // Will be unused
#endif
#if defined(WARPX_DIM_3D)
                is_insulator_lo[1] = (area_parsers_y_lo(x, z) > 0._rt);
                is_insulator_hi[1] = (area_parsers_y_hi(x, z) > 0._rt);
                F_lo[1] = (set_F_y_lo ? Fx_y_lo_parser(x, z, time) : 0._rt);
                F_hi[1] = (set_F_y_hi ? Fx_y_hi_parser(x, z, time) : 0._rt);
                set_field_lo[1] = set_F_y_lo;
                set_field_hi[1] = set_F_y_hi;
#endif
                is_insulator_lo[WARPX_ZINDEX] = (area_parsers_z_lo(x, y) > 0._rt);
                is_insulator_hi[WARPX_ZINDEX] = (area_parsers_z_hi(x, y) > 0._rt);
                F_lo[WARPX_ZINDEX] = (set_F_z_lo ? Fx_z_lo_parser(x, y, time) : 0._rt);
                F_hi[WARPX_ZINDEX] = (set_F_z_hi ? Fx_z_hi_parser(x, y, time) : 0._rt);
                set_field_lo[WARPX_ZINDEX] = set_F_z_lo;
                set_field_hi[WARPX_ZINDEX] = set_F_z_hi;

                int const icomp = 0;
                ::SetFieldOnPEC_Insulator(icomp, domain_lo, domain_hi, iv, n,
                                          Fx, E_like, Fx_nodal, is_insulator_lo, is_insulator_hi,
                                          F_lo, F_hi, set_field_lo, set_field_hi,
                                          fbndry_lo, fbndry_hi);
            },
            tey, nComp_y,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j, k);

                amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                amrex::Real const shiftx = (Fy_nodal[0] ? 0._rt : 0.5_rt);
                amrex::Real const x = (AMREX_SPACEDIM > 1 ? xyzmin_y.x + (iv[0] - lo_y[0] + shiftx)*dx[0] : 0._rt);
                amrex::Real const shifty = (AMREX_SPACEDIM == 3 ? (Fy_nodal[1] ? 0._rt : 0.5_rt) : 0._rt);
                amrex::Real const y = (AMREX_SPACEDIM == 3 ? xyzmin_y.y + (iv[1] - lo_y[1] + shifty)*dx[1] : 0._rt);
#if (AMREX_SPACEDIM > 1)
                amrex::Real const shiftz = (Fy_nodal[WARPX_ZINDEX] ? 0._rt : 0.5_rt);
                amrex::Real const z = xyzmin_y.z + (iv[WARPX_ZINDEX] - lo_y[WARPX_ZINDEX] + shiftz)*dx[2];
#endif

                amrex::IntVect is_insulator_lo;
                amrex::IntVect is_insulator_hi;
                amrex::RealVect F_lo, F_hi;
                amrex::IntVect set_field_lo;
                amrex::IntVect set_field_hi;
#if (AMREX_SPACEDIM > 1)
                is_insulator_lo[0] = (area_parsers_x_lo(y, z) > 0._rt);
                is_insulator_hi[0] = (area_parsers_x_hi(y, z) > 0._rt);
                F_lo[0] = (set_F_x_lo ? Fy_x_lo_parser(y, z, time) : 0._rt);
                F_hi[0] = (set_F_x_hi ? Fy_x_hi_parser(y, z, time) : 0._rt);
                set_field_lo[0] = set_F_x_lo;
                set_field_hi[0] = set_F_x_hi;
#endif
#if defined(WARPX_DIM_3D)
                is_insulator_lo[1] = (area_parsers_y_lo(x, z) > 0._rt);
                is_insulator_hi[1] = (area_parsers_y_hi(x, z) > 0._rt);
                F_lo[1] = 0._rt;  // Will be unused
                F_hi[1] = 0._rt;  // Will be unused
                set_field_lo[1] = 0;  // Will be unused
                set_field_hi[1] = 0;  // Will be unused
#endif
                is_insulator_lo[WARPX_ZINDEX] = (area_parsers_z_lo(x, y) > 0._rt);
                is_insulator_hi[WARPX_ZINDEX] = (area_parsers_z_hi(x, y) > 0._rt);
                F_lo[WARPX_ZINDEX] = (set_F_z_lo ? Fy_z_lo_parser(x, y, time) : 0._rt);
                F_hi[WARPX_ZINDEX] = (set_F_z_hi ? Fy_z_hi_parser(x, y, time) : 0._rt);
                set_field_lo[WARPX_ZINDEX] = set_F_z_lo;
                set_field_hi[WARPX_ZINDEX] = set_F_z_hi;

                int const icomp = 1;
                ::SetFieldOnPEC_Insulator(icomp, domain_lo, domain_hi, iv, n,
                                          Fy, E_like, Fy_nodal, is_insulator_lo, is_insulator_hi,
                                          F_lo, F_hi, set_field_lo, set_field_hi,
                                          fbndry_lo, fbndry_hi);
            },
            tez, nComp_z,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::ignore_unused(j, k);

                amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                amrex::Real const shiftx = (Fz_nodal[0] ? 0._rt : 0.5_rt);
                amrex::Real const x = (AMREX_SPACEDIM > 1 ? xyzmin_z.x + (iv[0] - lo_z[0] + shiftx)*dx[0] : 0._rt);
                amrex::Real const shifty = (AMREX_SPACEDIM == 3 ? (Fz_nodal[1] ? 0._rt : 0.5_rt) : 0._rt);
                amrex::Real const y = (AMREX_SPACEDIM == 3 ? xyzmin_z.y + (iv[1] - lo_z[1] + shifty)*dx[1] : 0._rt);
#if (AMREX_SPACEDIM > 1)
                amrex::Real const shiftz = (Fz_nodal[WARPX_ZINDEX] ? 0._rt : 0.5_rt);
                amrex::Real const z = xyzmin_z.z + (iv[WARPX_ZINDEX] - lo_z[WARPX_ZINDEX] + shiftz)*dx[2];
#endif

                amrex::IntVect is_insulator_lo;
                amrex::IntVect is_insulator_hi;
                amrex::RealVect F_lo, F_hi;
                amrex::IntVect set_field_lo;
                amrex::IntVect set_field_hi;
#if (AMREX_SPACEDIM > 1)
                is_insulator_lo[0] = (area_parsers_x_lo(y, z) > 0._rt);
                is_insulator_hi[0] = (area_parsers_x_hi(y, z) > 0._rt);
                F_lo[0] = (set_F_x_lo ? Fz_x_lo_parser(y, z, time) : 0._rt);
                F_hi[0] = (set_F_x_hi ? Fz_x_hi_parser(y, z, time) : 0._rt);
                set_field_lo[0] = set_F_x_lo;
                set_field_hi[0] = set_F_x_hi;
#endif
#if defined(WARPX_DIM_3D)
                is_insulator_lo[1] = (area_parsers_y_lo(x, z) > 0._rt);
                is_insulator_hi[1] = (area_parsers_y_hi(x, z) > 0._rt);
                F_lo[1] = (set_F_y_lo ? Fz_y_lo_parser(x, z, time) : 0._rt);
                F_hi[1] = (set_F_y_hi ? Fz_y_hi_parser(x, z, time) : 0._rt);
                set_field_lo[1] = set_F_y_lo;
                set_field_hi[1] = set_F_y_hi;
#endif
                is_insulator_lo[WARPX_ZINDEX] = (area_parsers_z_lo(x, y) > 0._rt);
                is_insulator_hi[WARPX_ZINDEX] = (area_parsers_z_hi(x, y) > 0._rt);
                F_lo[WARPX_ZINDEX] = 0._rt;  // Will be unused
                F_hi[WARPX_ZINDEX] = 0._rt;  // Will be unused
                set_field_lo[WARPX_ZINDEX] = 0;  // Will be unused
                set_field_hi[WARPX_ZINDEX] = 0;  // Will be unused

                int const icomp = 2;
                ::SetFieldOnPEC_Insulator(icomp, domain_lo, domain_hi, iv, n,
                                          Fz, E_like, Fz_nodal, is_insulator_lo, is_insulator_hi,
                                          F_lo, F_hi, set_field_lo, set_field_hi,
                                          fbndry_lo, fbndry_hi);
            }
        );
    }
}
