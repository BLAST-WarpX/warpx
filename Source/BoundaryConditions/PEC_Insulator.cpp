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

    // \brief Converts the grid indices to spatial coordinates
    //
    // \param[in] iv      vector of indices
    // \param[in] xyzmin  vector of grid mins
    // \param[in] dx      vector of grid sizes
    // \param[in] lo      vector of grid index mins
    // \param[in] nodal   vector of nodel flags to each dimension
    // \return the coordinates

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::XDim3 ConvertIndexToCoordinate(amrex::IntVect const & iv,
                                          amrex::XDim3 const & xyzmin,
                                          std::array<amrex::Real,3> const & dx,
                                          amrex::IntVect const & lo,
                                          amrex::IntVect const & nodal)
    {
        using namespace amrex::literals;
        amrex::XDim3 result = {0._rt, 0._rt, 0._rt};

#if defined(WARPX_DIM_3D) || defined(WARPX_ZINDEX)
        amrex::Real const shiftx = (nodal[0] ? 0._rt : 0.5_rt);
        result.x = (AMREX_SPACEDIM > 1 ? xyzmin.x + (iv[0] - lo[0] + shiftx)*dx[0] : 0._rt);
#endif
        amrex::Real const shifty = (AMREX_SPACEDIM == 3 ? (nodal[1] ? 0._rt : 0.5_rt) : 0._rt);
        result.y = (AMREX_SPACEDIM == 3 ? xyzmin.y + (iv[1] - lo[1] + shifty)*dx[1] : 0._rt);
#ifndef WARPX_DIM_1D_Z
#if defined(WARPX_ZINDEX)
        amrex::Real const shiftz = (nodal[WARPX_ZINDEX] ? 0._rt : 0.5_rt);
        result.z = xyzmin.z + (iv[WARPX_ZINDEX] - lo[WARPX_ZINDEX] + shiftz)*dx[2];
#endif
#endif

        return result;
    }

    // Convenient structure to the hold the two transverse coordinates
    struct XDimTransverse { amrex::Real t1; amrex::Real t2; };

    // \brief Returns the two coordinates transverse from the specified dimension
    //
    // \param[im] idim    the dimension number
    // \param[in] coords  the coordinate of each dimension
    // \return the transverse coordinates
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    XDimTransverse GetTransverseCoordinates(int idim, amrex::XDim3 const & coords)
    {
        // Transverse coordinates
        amrex::Real t1 = 0.;
        amrex::Real t2 = 0.;

#ifndef WARPX_DIM_1D_Z
        if (idim == 0) {
            t1 = coords.y;
            t2 = coords.z;
        }
#endif
#if defined(WARPX_DIM_3D)
        if (idim == 1) {
            t1 = coords.x;
            t2 = coords.z;
        }
#endif
#if defined(WARPX_ZINDEX)
        if (idim == WARPX_ZINDEX) {
            t1 = coords.x;
            t2 = coords.y;
        }
#endif
        return XDimTransverse {t1, t2};
    }

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
     * \param[in] is_insulator_boundary Specifies whether the boundary is an insulator
     * \param[in] field_value  the value of the field for the insulator boundary cell
     * \param[in] set_field    whether to set the field on the lower boundary
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void SetFieldOnPEC_Insulator (int idim,
                                  int iside,
                                  int icomp,
                                  amrex::IntVect const & dom_lo,
                                  amrex::IntVect const & dom_hi,
                                  amrex::IntVect const & ijk_vec,
                                  int n,
                                  amrex::Array4<amrex::Real> const & field,
                                  bool const E_like,
                                  amrex::IntVect const & is_nodal,
                                  const bool is_insulator_boundary,
                                  amrex::Real const field_value,
                                  bool const set_field)
    {
        using namespace amrex::literals;

        // iside = -1 (lo), iside = +1 (hi)

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
        bool const is_normal_to_boundary = (icomp == (2*idim));
#elif (defined WARPX_DIM_1D_Z)
        // For 1D : icomp=0 and icomp=1 (Fx and Fy are not normal to the z boundary)
        //          The logic below ensures that the flags are set right for 1D
        bool const is_normal_to_boundary = (icomp == 2);
#else
        bool const is_normal_to_boundary = (icomp == idim);
#endif

        // For B fields, the parallel fields are cell centered, so are on the boundary
        // when ig == 1, the first cell beyond the boundary.
        bool const on_cell_boundary = (ig == 1);

        // For E fields, the parallel fields are node centered, so are on the boundary
        // when ig == 0.
        bool const on_nodal_boundary = (ig == 0);

        amrex::IntVect ijk_mirror = ijk_vec;
        amrex::IntVect ijk_boundary = ijk_vec;

        bool const guard_cell = (ig > 0);
        if (guard_cell) {

            // Mirror location inside the domain by "ig" number of cells
            ijk_mirror[idim] = ( (iside == -1)
                            ? (dom_lo[idim] + ig - (1 - is_nodal[idim]))
                            : (dom_hi[idim] - ig + 1));

            ijk_boundary[idim] = ( (iside == -1) ? dom_lo[idim] : dom_hi[idim] + is_nodal[idim]);


        }

        if (E_like) {
            if (is_normal_to_boundary) {
                if (guard_cell) {
                    // E-normal is cell-centered so is mirrored for both PEC and insulator
                    field(ijk_vec, n) = field(ijk_mirror, n);
                }
            } else {
                if (on_nodal_boundary) {
                    if (is_insulator_boundary) {
                        if (set_field) {
                            // E-parallel on the boundary is only modified if field is being set,
                            // otherwise it is evolved
                            field(ijk_vec, n) = field_value;
                        }
                    } else {
                        // E-parallel is set to zero on the boundary for PEC
                        field(ijk_vec, n) = 0._rt;
                    }
                } else if (guard_cell) {
                    if (is_insulator_boundary) {
                        amrex::Real const field_boundary = (set_field ? field_value : field(ijk_boundary, n));
                        field(ijk_vec, n) = 2._rt*field_boundary - field(ijk_mirror, n);
                    } else {
                        // For PEC, field_boundary = 0
                        field(ijk_vec, n) = -field(ijk_mirror, n);
                    }
                }
            }
        } else {
            // B-field
            if (is_normal_to_boundary) {
                if (on_nodal_boundary) {
                    // For insulator, B-normal is evolved so is unmodified here
                    if (!is_insulator_boundary) {
                        // With PEC, B-normal is zeroed out.
                        field(ijk_vec, n) = 0._rt;
                    }
                } else if (guard_cell) {
                    // B-normal is nodal
                    if (is_insulator_boundary) {
                        // Extrapolate from evolved value on the boundary
                        field(ijk_vec, n) = 2._rt*field(ijk_boundary, n) - field(ijk_mirror, n);
                    } else {
                        // Extrapolate from zero on the boundary
                        field(ijk_vec, n) = -field(ijk_mirror, n);
                    }
                }
            } else {
                if (on_cell_boundary && is_insulator_boundary && set_field) {
                    // B-parallel in the boundary cell is only modified if field is being set,
                    // otherwise it is mirrored.
                    field(ijk_vec, n) = field_value;
                } else if (guard_cell) {
                    if (is_insulator_boundary && set_field) {
                        // Location of the next cells inward
                        amrex::IntVect ijk_next = ijk_vec;
                        amrex::IntVect ijk_nextp1 = ijk_vec;
                        ijk_next[idim] = ijk_vec[idim] - ig*iside;
                        ijk_nextp1[idim] = ijk_next[idim] - ig*iside;
                        field(ijk_vec, n) = 2._rt*field(ijk_next, n) + field(ijk_nextp1, n);
                    } else {
                        field(ijk_vec, n) = field(ijk_mirror, n);
                    }
                }
            }
        }
    }

    /* \brief Sets up the parsers, taking the input data and arranging it as needed
     *        for the loops, and compiling the parser expressions.
     *
     * \param[in] set_field_lo      flags whether the insulator expressions were specified
     * \param[in] set_field_hi      flags whether the insulator expressions were specified
     * \param[in] parser_field1_lo  the parser for the first transverse field at the low boundary
     * \param[in] parser_field2_lo  the parser for the second transverse field at the low boundary
     * \param[in] parser_field1_hi  the parser for the first transverse field at the high boundary
     * \param[in] parser_field2_hi  the parser for the second transverse field at the high boundary
     * \param[out] set_fields_x_lo  the flags for the field along x at the lower boundary
     * \param[out] set_fields_y_lo  the flags for the field along y at the lower boundary
     * \param[out] set_fields_z_lo  the flags for the field along z at the lower boundary
     * \param[out] set_fields_x_hi  the flags for the field along x at the upper boundary
     * \param[out] set_fields_y_hi  the flags for the field along y at the upper boundary
     * \param[out] set_fields_z_hi  the flags for the field along z at the upper boundary
     * \param[out] Fx_parsers_lo    the parsers for the field along x at the lower boundary
     * \param[out] Fy_parsers_lo    the parsers for the field along y at the lower boundary
     * \param[out] Fz_parsers_lo    the parsers for the field along z at the lower boundary
     * \param[out] Fx_parsers_hi    the parsers for the field along x at the upper boundary
     * \param[out] Fy_parsers_hi    the parsers for the field along y at the upper boundary
     * \param[out] Fz_parsers_hi    the parsers for the field along z at the upper boundary
    */
    void SetupFieldParsers(amrex::Vector<int> const & set_field_lo,
                           amrex::Vector<int> const & set_field_hi,
                           amrex::Vector<std::unique_ptr<amrex::Parser>> const & parser_field1_lo,
                           amrex::Vector<std::unique_ptr<amrex::Parser>> const & parser_field2_lo,
                           amrex::Vector<std::unique_ptr<amrex::Parser>> const & parser_field1_hi,
                           amrex::Vector<std::unique_ptr<amrex::Parser>> const & parser_field2_hi,
                           amrex::Vector<int> & set_fields_x_lo,
                           amrex::Vector<int> & set_fields_y_lo,
                           amrex::Vector<int> & set_fields_z_lo,
                           amrex::Vector<int> & set_fields_x_hi,
                           amrex::Vector<int> & set_fields_y_hi,
                           amrex::Vector<int> & set_fields_z_hi,
                           [[maybe_unused]]amrex::Vector<amrex::ParserExecutor<3>> & Fx_parsers_lo,
                           [[maybe_unused]]amrex::Vector<amrex::ParserExecutor<3>> & Fy_parsers_lo,
                           [[maybe_unused]]amrex::Vector<amrex::ParserExecutor<3>> & Fz_parsers_lo,
                           [[maybe_unused]]amrex::Vector<amrex::ParserExecutor<3>> & Fx_parsers_hi,
                           [[maybe_unused]]amrex::Vector<amrex::ParserExecutor<3>> & Fy_parsers_hi,
                           [[maybe_unused]]amrex::Vector<amrex::ParserExecutor<3>> & Fz_parsers_hi)
    {
        set_fields_x_lo.resize(AMREX_SPACEDIM, false);
        set_fields_y_lo.resize(AMREX_SPACEDIM, false);
        set_fields_z_lo.resize(AMREX_SPACEDIM, false);
        set_fields_x_hi.resize(AMREX_SPACEDIM, false);
        set_fields_y_hi.resize(AMREX_SPACEDIM, false);
        set_fields_z_hi.resize(AMREX_SPACEDIM, false);

#ifndef WARPX_DIM_1D_Z
        set_fields_y_lo[0] = set_field_lo[0];
        set_fields_z_lo[0] = set_field_lo[0];
        set_fields_y_hi[0] = set_field_hi[0];
        set_fields_z_hi[0] = set_field_hi[0];
        Fy_parsers_lo.push_back(parser_field1_lo[0]->compile<3>());
        Fz_parsers_lo.push_back(parser_field2_lo[0]->compile<3>());
        Fy_parsers_hi.push_back(parser_field1_hi[0]->compile<3>());
        Fz_parsers_hi.push_back(parser_field2_hi[0]->compile<3>());
#endif
#if defined(WARPX_DIM_3D)
        set_fields_x_lo[1] = set_field_lo[1];
        set_fields_z_lo[1] = set_field_lo[1];
        set_fields_x_hi[1] = set_field_hi[1];
        set_fields_z_hi[1] = set_field_hi[1];
        Fx_parsers_lo.push_back(parser_field1_lo[1]->compile<3>());
        Fz_parsers_lo.push_back(parser_field2_lo[1]->compile<3>());
        Fx_parsers_hi.push_back(parser_field1_hi[1]->compile<3>());
        Fz_parsers_hi.push_back(parser_field2_hi[1]->compile<3>());
#endif
#if defined(WARPX_ZINDEX)
        set_fields_x_lo[WARPX_ZINDEX] = set_field_lo[WARPX_ZINDEX];
        set_fields_y_lo[WARPX_ZINDEX] = set_field_lo[WARPX_ZINDEX];
        set_fields_x_hi[WARPX_ZINDEX] = set_field_hi[WARPX_ZINDEX];
        set_fields_y_hi[WARPX_ZINDEX] = set_field_hi[WARPX_ZINDEX];
        Fx_parsers_lo.push_back(parser_field1_lo[WARPX_ZINDEX]->compile<3>());
        Fy_parsers_lo.push_back(parser_field2_lo[WARPX_ZINDEX]->compile<3>());
        Fx_parsers_hi.push_back(parser_field1_hi[WARPX_ZINDEX]->compile<3>());
        Fy_parsers_hi.push_back(parser_field2_hi[WARPX_ZINDEX]->compile<3>());
#endif
    }

}

bool
PEC_Insulator::ReadTangentialFieldParser (amrex::ParmParse const & pp_insulator,
                                          amrex::Vector<std::unique_ptr<amrex::Parser>> & parsers,
                                          std::string const & input_name,
                                          std::string const & coord1,
                                          std::string const & coord2)
{
    std::string str = "0";
    bool const specified = utils::parser::Query_parserString(pp_insulator, input_name, str);
    parsers.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str, {coord1, coord2, "t"})));
    return specified;
}

PEC_Insulator::PEC_Insulator ()
{

    amrex::ParmParse const pp_insulator("insulator");

#ifndef WARPX_DIM_1D_Z
    std::string str_area_x_lo = "0";
    std::string str_area_x_hi = "0";
    utils::parser::Query_parserString( pp_insulator, "area_x_lo(y,z)", str_area_x_lo);
    utils::parser::Query_parserString( pp_insulator, "area_x_hi(y,z)", str_area_x_hi);
    m_insulator_area_lo.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_x_lo, {"y", "z"})));
    m_insulator_area_hi.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_x_hi, {"y", "z"})));

    m_set_B_lo[0] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B1_lo, "By_x_lo(y,z,t)", "y", "z");
    m_set_B_lo[0] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B2_lo, "Bz_x_lo(y,z,t)", "y", "z");
    m_set_B_hi[0] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B1_hi, "By_x_hi(y,z,t)", "y", "z");
    m_set_B_hi[0] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B2_hi, "Bz_x_hi(y,z,t)", "y", "z");

    m_set_E_lo[0] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E1_lo, "Ey_x_lo(y,z,t)", "y", "z");
    m_set_E_lo[0] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E2_lo, "Ez_x_lo(y,z,t)", "y", "z");
    m_set_E_hi[0] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E1_hi, "Ey_x_hi(y,z,t)", "y", "z");
    m_set_E_hi[0] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E2_hi, "Ez_x_hi(y,z,t)", "y", "z");
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

    m_set_B_lo[1] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B1_lo, "Bx_y_lo(x,z,t)", "x", "z");
    m_set_B_lo[1] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B2_lo, "Bz_y_lo(x,z,t)", "x", "z");
    m_set_B_hi[1] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B1_hi, "Bx_y_hi(x,z,t)", "x", "z");
    m_set_B_hi[1] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B2_hi, "Bz_y_hi(x,z,t)", "x", "z");

    m_set_E_lo[1] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E1_lo, "Ex_y_lo(x,z,t)", "x", "z");
    m_set_E_lo[1] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E2_lo, "Ez_y_lo(x,z,t)", "x", "z");
    m_set_E_hi[1] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E1_hi, "Ex_y_hi(x,z,t)", "x", "z");
    m_set_E_hi[1] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E2_hi, "Ez_y_hi(x,z,t)", "x", "z");
#endif

#if defined(WARPX_ZINDEX)
    std::string str_area_z_lo = "0";
    std::string str_area_z_hi = "0";
    utils::parser::Query_parserString( pp_insulator, "area_z_lo(x,y)", str_area_z_lo);
    utils::parser::Query_parserString( pp_insulator, "area_z_hi(x,y)", str_area_z_hi);
    m_insulator_area_lo.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_z_lo, {"x", "y"})));
    m_insulator_area_hi.push_back(
        std::make_unique<amrex::Parser>(utils::parser::makeParser(str_area_z_hi, {"x", "y"})));

    m_set_B_lo[WARPX_ZINDEX] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B1_lo, "Bx_z_lo(x,y,t)", "x", "y");
    m_set_B_lo[WARPX_ZINDEX] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B2_lo, "By_z_lo(x,y,t)", "x", "y");
    m_set_B_hi[WARPX_ZINDEX] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B1_hi, "Bx_z_hi(x,y,t)", "x", "y");
    m_set_B_hi[WARPX_ZINDEX] |= ReadTangentialFieldParser(pp_insulator, m_parsers_B2_hi, "By_z_hi(x,y,t)", "x", "y");

    m_set_E_lo[WARPX_ZINDEX] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E1_lo, "Ex_z_lo(x,y,t)", "x", "y");
    m_set_E_lo[WARPX_ZINDEX] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E2_lo, "Ey_z_lo(x,y,t)", "x", "y");
    m_set_E_hi[WARPX_ZINDEX] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E1_hi, "Ex_z_hi(x,y,t)", "x", "y");
    m_set_E_hi[WARPX_ZINDEX] |= ReadTangentialFieldParser(pp_insulator, m_parsers_E2_hi, "Ey_z_hi(x,y,t)", "x", "y");
#endif

    for(const auto & area_parser : m_insulator_area_lo) {
        m_area_parsers_lo.push_back(area_parser->compile<2>());
    }
    for(const auto & area_parser : m_insulator_area_hi) {
        m_area_parsers_hi.push_back(area_parser->compile<2>());
    }

    ::SetupFieldParsers(m_set_B_lo, m_set_B_hi,
                        m_parsers_B1_lo, m_parsers_B2_lo, m_parsers_B1_hi, m_parsers_B2_hi,
                        m_set_Bfields_x_lo, m_set_Bfields_y_lo, m_set_Bfields_z_lo,
                        m_set_Bfields_x_hi, m_set_Bfields_y_hi, m_set_Bfields_z_hi,
                        m_Bx_parsers_lo, m_By_parsers_lo, m_Bz_parsers_lo,
                        m_Bx_parsers_hi, m_By_parsers_hi, m_Bz_parsers_hi);

    ::SetupFieldParsers(m_set_E_lo, m_set_E_hi,
                        m_parsers_E1_lo, m_parsers_E2_lo, m_parsers_E1_hi, m_parsers_E2_hi,
                        m_set_Efields_x_lo, m_set_Efields_y_lo, m_set_Efields_z_lo,
                        m_set_Efields_x_hi, m_set_Efields_y_hi, m_set_Efields_z_hi,
                        m_Ex_parsers_lo, m_Ey_parsers_lo, m_Ez_parsers_lo,
                        m_Ex_parsers_hi, m_Ey_parsers_hi, m_Ez_parsers_hi);
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
                              m_set_Efields_x_lo, m_set_Efields_y_lo, m_set_Efields_z_lo,
                              m_set_Efields_x_hi, m_set_Efields_y_hi, m_set_Efields_z_hi,
                              m_Ex_parsers_lo, m_Ey_parsers_lo, m_Ez_parsers_lo,
                              m_Ex_parsers_hi, m_Ey_parsers_hi, m_Ez_parsers_hi);
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
                              m_set_Bfields_x_lo, m_set_Bfields_y_lo, m_set_Bfields_z_lo,
                              m_set_Bfields_x_hi, m_set_Bfields_y_hi, m_set_Bfields_z_hi,
                              m_Bx_parsers_lo, m_By_parsers_lo, m_Bz_parsers_lo,
                              m_Bx_parsers_hi, m_By_parsers_hi, m_Bz_parsers_hi);
}


void
PEC_Insulator::ApplyPEC_InsulatortoField (
    std::array<amrex::MultiFab*, 3> field,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
    amrex::IntVect const & ng_fieldgather,
    amrex::Geometry const & geom,
    int lev,
    PatchType patch_type,
    amrex::Vector<amrex::IntVect> const & ref_ratios,
    amrex::Real time,
    bool split_pml_field,
    bool E_like,
    amrex::Vector<int> const & set_fields_x_lo,
    amrex::Vector<int> const & set_fields_y_lo,
    amrex::Vector<int> const & set_fields_z_lo,
    amrex::Vector<int> const & set_fields_x_hi,
    amrex::Vector<int> const & set_fields_y_hi,
    amrex::Vector<int> const & set_fields_z_hi,
    amrex::Vector<amrex::ParserExecutor<3>> const & Fx_parsers_lo,
    amrex::Vector<amrex::ParserExecutor<3>> const & Fy_parsers_lo,
    amrex::Vector<amrex::ParserExecutor<3>> const & Fz_parsers_lo,
    amrex::Vector<amrex::ParserExecutor<3>> const & Fx_parsers_hi,
    amrex::Vector<amrex::ParserExecutor<3>> const & Fy_parsers_hi,
    amrex::Vector<amrex::ParserExecutor<3>> const & Fz_parsers_hi)
{
    using namespace amrex::literals;
    amrex::Box domain_box = geom.Domain();
    if (patch_type == PatchType::coarse && (lev > 0)) {
        domain_box.coarsen(ref_ratios[lev-1]);
    }
    amrex::IntVect const domain_lo = domain_box.smallEnd();
    amrex::IntVect const domain_hi = domain_box.bigEnd();

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
    // The false flag here is to ensure that this loop does not use tiling.
    // The boxes are grown to include transverse ghost cells prior to the reflection.
    // Tiling is problematic because neighboring tiles will have overlapping boxes
    // in the direction transverse to the boundary, thereby reflecting the value multiple
    // times in the overlapping region.
    for (amrex::MFIter mfi(*field[0], false); mfi.isValid(); ++mfi) {
        // Extract field data
        amrex::Array4<amrex::Real> const & Fx = field[0]->array(mfi);
        amrex::Array4<amrex::Real> const & Fy = field[1]->array(mfi);
        amrex::Array4<amrex::Real> const & Fz = field[2]->array(mfi);

        // Get nodal box that does not include ghost cells
        amrex::Box const & valid_box = mfi.validbox();
        amrex::Box const node_box = amrex::convert(valid_box, amrex::IntVect::TheNodeVector());

        // The lower end of the box is the same for nodal and cell centered,
        // so the same lower end can be used for the three fields even though
        // they will have different centerings.
        amrex::XDim3 const xyzmin = WarpX::LowerCorner(valid_box, lev, 0._rt);
        amrex::IntVect const lo = valid_box.smallEnd();

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

            if (!(field_boundary_lo[idim] == FieldBoundaryType::PECInsulator) &&
                !(field_boundary_hi[idim] == FieldBoundaryType::PECInsulator)) { continue; }

            if ( (node_box.smallEnd()[idim] > domain_lo[idim]) &&
                 (node_box.bigEnd()[idim] < domain_hi[idim]) ) { continue; }

            // Extract tileboxes for which to loop
            // if split field, the box includes nodal flag
            // For E-field used in Maxwell's update, nodal flag plus cells that particles
            // gather fields from in the guard-cell region are included.
            // Note that for simulations without particles or laser, ng_field_gather is 0
            // and the guard-cell values of the E-field multifab will not be modified.
            amrex::Box tex = (split_pml_field) ? mfi.tilebox(field[0]->ixType().toIntVect())
                                               : mfi.tilebox(field[0]->ixType().toIntVect(), ng_fieldgather);
            amrex::Box tey = (split_pml_field) ? mfi.tilebox(field[1]->ixType().toIntVect())
                                               : mfi.tilebox(field[1]->ixType().toIntVect(), ng_fieldgather);
            amrex::Box tez = (split_pml_field) ? mfi.tilebox(field[2]->ixType().toIntVect())
                                               : mfi.tilebox(field[2]->ixType().toIntVect(), ng_fieldgather);

            // Loop over sides, iside = -1 (lo), iside = +1 (hi)
            for (int iside = -1; iside <= +1; iside += 2) {

                if ((iside == -1 && (field_boundary_lo[idim] != FieldBoundaryType::PECInsulator)) ||
                    (iside == +1 && (field_boundary_hi[idim] != FieldBoundaryType::PECInsulator))) { continue; }

                if ((iside == -1 && (node_box.smallEnd()[idim] > domain_lo[idim])) ||
                    (iside == +1 && (node_box.bigEnd()[idim] < domain_hi[idim]))) { continue; }

                amrex::Box tex_guard = tex;
                amrex::Box tey_guard = tey;
                amrex::Box tez_guard = tez;

                // Shrink the box to only include the guard and boundary cells
                if (iside == -1) {
                    tex_guard.setBig(idim, node_box.smallEnd(idim));
                    tey_guard.setBig(idim, node_box.smallEnd(idim));
                    tez_guard.setBig(idim, node_box.smallEnd(idim));
                } else {
                    tex_guard.setSmall(idim, node_box.bigEnd(idim));
                    tey_guard.setSmall(idim, node_box.bigEnd(idim));
                    tez_guard.setSmall(idim, node_box.bigEnd(idim));
                }

                bool const set_fieldx = ( (iside == -1) ? set_fields_x_lo[idim] : set_fields_x_hi[idim]);
                bool const set_fieldy = ( (iside == -1) ? set_fields_y_lo[idim] : set_fields_y_hi[idim]);
                bool const set_fieldz = ( (iside == -1) ? set_fields_z_lo[idim] : set_fields_z_hi[idim]);

                amrex::ParserExecutor<2> const & area_parser = ( (iside == -1) ? m_area_parsers_lo[idim] : m_area_parsers_hi[idim]);

                // A special check is needed for Fx and Fz since in 1D cases no parsers will be defined for one
                // or the other. Instead, create a dummy ParserExecutor.
                amrex::ParserExecutor<3> const & Fx_parser = ( Fx_parsers_lo.size() > 0 ?
                                                             ( (iside == -1) ? Fx_parsers_lo[idim] : Fx_parsers_hi[idim]) :
                                                             amrex::ParserExecutor<3>() );
                amrex::ParserExecutor<3> const & Fy_parser = ( (iside == -1) ? Fy_parsers_lo[idim] : Fy_parsers_hi[idim]);
                amrex::ParserExecutor<3> const & Fz_parser = ( Fz_parsers_lo.size() > 0 ?
                                                             ( (iside == -1) ? Fz_parsers_lo[idim] : Fz_parsers_hi[idim]) :
                                                             amrex::ParserExecutor<3>() );

                // loop over cells and update fields
                amrex::ParallelFor(
                    tex_guard, nComp_x,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                        amrex::ignore_unused(j, k);

                        amrex::IntVect const iv(AMREX_D_DECL(i, j, k));

                        amrex::XDim3 const coords = ::ConvertIndexToCoordinate(iv, xyzmin, dx, lo, Fx_nodal);
                        ::XDimTransverse tcoords = ::GetTransverseCoordinates(idim, coords);

                        bool const is_insulator = (area_parser(tcoords.t1, tcoords.t2) > 0._rt);
                        amrex::Real const field_value = (set_fieldx ? Fx_parser(tcoords.t1, tcoords.t2, time) : 0._rt);

                        int const icomp = 0;
                        ::SetFieldOnPEC_Insulator(idim, iside, icomp, domain_lo, domain_hi, iv, n,
                                                  Fx, E_like, Fx_nodal, is_insulator,
                                                  field_value, set_fieldx);
                    },
                    tey_guard, nComp_y,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                        amrex::ignore_unused(j, k);

                        amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                        amrex::XDim3 const coords = ::ConvertIndexToCoordinate(iv, xyzmin, dx, lo, Fy_nodal);
                        ::XDimTransverse tcoords = ::GetTransverseCoordinates(idim, coords);

                        bool const is_insulator = (area_parser(tcoords.t1, tcoords.t2) > 0._rt);
                        amrex::Real const field_value = (set_fieldy ? Fy_parser(tcoords.t1, tcoords.t2, time) : 0._rt);

                        int const icomp = 1;
                        ::SetFieldOnPEC_Insulator(idim, iside, icomp, domain_lo, domain_hi, iv, n,
                                                  Fy, E_like, Fy_nodal, is_insulator,
                                                  field_value, set_fieldy);
                    },
                    tez_guard, nComp_z,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                        amrex::ignore_unused(j, k);

                        amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                        amrex::XDim3 const coords = ::ConvertIndexToCoordinate(iv, xyzmin, dx, lo, Fz_nodal);
                        ::XDimTransverse tcoords = ::GetTransverseCoordinates(idim, coords);

                        bool const is_insulator = (area_parser(tcoords.t1, tcoords.t2) > 0._rt);
                        amrex::Real const field_value = (set_fieldz ? Fz_parser(tcoords.t1, tcoords.t2, time) : 0._rt);

                        int const icomp = 2;
                        ::SetFieldOnPEC_Insulator(idim, iside, icomp, domain_lo, domain_hi, iv, n,
                                                  Fz, E_like, Fz_nodal, is_insulator,
                                                  field_value, set_fieldz);
                    }
                );
            }
        }
    }
}
