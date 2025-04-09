#ifndef PEC_INSULATOR_H_
#define PEC_INSULATOR_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

class PEC_Insulator
{
public:

    PEC_Insulator();

    /**
     * \brief Apply either the PEC or insulator boundary condition on the boundary and in the
     *        guard cells.
     *        In the PEC, the nodal fields (in a Yee mesh) are made even relative to the boundary,
     *        the non-nodal fields are made odd.
     *        In the insulator, the tangential fields are set to the value if specified, otherwise unchanged,
     *        and the normal fields extrapolated from the valid cells.
     *
     * \param[in,out] Efield
     * \param[in]     field_boundary_lo   lower field boundary conditions
     * \param[in]     field_boundary_hi   upper field boundary conditions
     * \param[in]     ng_fieldgather      number of guard cells used by field gather
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     * \param[in]     time                current time of the simulation
     * \param[in]     split_pml_field     whether pml the multifab is the regular Efield or
     *                                    split pml field
     */
    void ApplyPEC_InsulatortoEfield (std::array<amrex::MultiFab*, 3> Efield,
                                     amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
                                     amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
                                     amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
                                     int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
                                     amrex::Real time,
                                     bool split_pml_field = false);
    /**
     * \brief Apply either the PEC or insulator boundary condition on the boundary and in the
     *        guard cells.
     *        In the PEC, the nodal fields (in a Yee mesh) are made even relative to the boundary,
     *        the non-nodal fields are made odd.
     *        In the insulator, the tangential fields are set to the value if specified, otherwise unchanged,
     *        and the normal fields extrapolated from the valid cells.
     *
     * \param[in,out] Bfield
     * \param[in]     field_boundary_lo   lower field boundary conditions
     * \param[in]     field_boundary_hi   upper field boundary conditions
     * \param[in]     ng_fieldgather      number of guard cells used by field gather
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     * \param[in]     time                current time of the simulation
     */
    void ApplyPEC_InsulatortoBfield (std::array<amrex::MultiFab*, 3> Bfield,
                                     amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
                                     amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
                                     amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
                                     int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
                                     amrex::Real time);

    /**
     * \brief The work routine applying the boundary condition
     *
     * \param[in,out] field
     * \param[in]     field_boundary_lo   lower field boundary conditions
     * \param[in]     field_boundary_hi   upper field boundary conditions
     * \param[in]     ng_fieldgather      number of guard cells used by field gather
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     * \param[in]     time                current time of the simulation
     * \param[in]     split_pml_field     whether pml the multifab is the regular Efield or
     *                                    split pml field
     * \param[in]     E_like whether the field is E like or B like
     * \param[in]     set_F_x_lo whether the tangential field at the boundary was specified
     * \param[in]     set_F_x_hi whether the tangential field at the boundary was specified
     * \param[in]     a_Fy_x_lo the parser for the tangential field at the boundary
     * \param[in]     a_Fz_x_lo the parser for the tangential field at the boundary
     * \param[in]     a_Fy_x_hi the parser for the tangential field at the boundary
     * \param[in]     a_Fz_x_hi the parser for the tangential field at the boundary
     * \param[in]     set_F_y_lo whether the tangential field at the boundary was specified
     * \param[in]     set_F_y_hi whether the tangential field at the boundary was specified
     * \param[in]     a_Fx_y_lo the parser for the tangential field at the boundary
     * \param[in]     a_Fz_y_lo the parser for the tangential field at the boundary
     * \param[in]     a_Fx_y_hi the parser for the tangential field at the boundary
     * \param[in]     a_Fz_y_hi the parser for the tangential field at the boundary
     * \param[in]     set_F_z_lo whether the tangential field at the boundary was specified
     * \param[in]     set_F_z_hi whether the tangential field at the boundary was specified
     * \param[in]     a_Fx_z_lo the parser for the tangential field at the boundary
     * \param[in]     a_Fy_z_lo the parser for the tangential field at the boundary
     * \param[in]     a_Fx_z_hi the parser for the tangential field at the boundary
     * \param[in]     a_Fy_z_hi the parser for the tangential field at the boundary
     */
    void
    ApplyPEC_InsulatortoField (std::array<amrex::MultiFab*, 3> field,
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
                               std::unique_ptr<amrex::Parser> const & a_Fx_z_hi, std::unique_ptr<amrex::Parser> const & a_Fy_z_hi);

private:

    /* \brief Reads in the parsers for the tangential fields, returning whether
     *        the input parameter was specified.
     * \param[in]  pp_insulator ParmParse instance
     * \param[out] parser the parser generated from the input
     * \param[in]  input_name the name of the input parameter
     * \param[in]  coord1 the first coordinate in the plane
     * \param[in]  coord2 the second coordinate in the plane
     */
    bool ReadTangentialFieldParser (amrex::ParmParse const & pp_insulator,
                                    std::unique_ptr<amrex::Parser> & parser,
                                    std::string const & input_name,
                                    std::string const & coord1,
                                    std::string const & coord2);

    std::vector<std::unique_ptr<amrex::Parser>> m_insulator_area_lo;
    std::vector<std::unique_ptr<amrex::Parser>> m_insulator_area_hi;

#if (AMREX_SPACEDIM > 1)
    bool m_set_B_x_lo = false, m_set_B_x_hi = false;
    std::unique_ptr<amrex::Parser> m_By_x_lo, m_Bz_x_lo;
    std::unique_ptr<amrex::Parser> m_By_x_hi, m_Bz_x_hi;
#endif
#if defined(WARPX_DIM_3D)
    bool m_set_B_y_lo = false, m_set_B_y_hi = false;
    std::unique_ptr<amrex::Parser> m_Bx_y_lo, m_Bz_y_lo;
    std::unique_ptr<amrex::Parser> m_Bx_y_hi, m_Bz_y_hi;
#endif
    bool m_set_B_z_lo = false, m_set_B_z_hi = false;
    std::unique_ptr<amrex::Parser> m_Bx_z_lo, m_By_z_lo;
    std::unique_ptr<amrex::Parser> m_Bx_z_hi, m_By_z_hi;


#if (AMREX_SPACEDIM > 1)
    bool m_set_E_x_lo = false, m_set_E_x_hi = false;
    std::unique_ptr<amrex::Parser> m_Ey_x_lo, m_Ez_x_lo;
    std::unique_ptr<amrex::Parser> m_Ey_x_hi, m_Ez_x_hi;
#endif
#if defined(WARPX_DIM_3D)
    bool m_set_E_y_lo = false, m_set_E_y_hi = false;
    std::unique_ptr<amrex::Parser> m_Ex_y_lo, m_Ez_y_lo;
    std::unique_ptr<amrex::Parser> m_Ex_y_hi, m_Ez_y_hi;
#endif
    bool m_set_E_z_lo = false, m_set_E_z_hi = false;
    std::unique_ptr<amrex::Parser> m_Ex_z_lo, m_Ey_z_lo;
    std::unique_ptr<amrex::Parser> m_Ex_z_hi, m_Ey_z_hi;


};
#endif // PEC_INSULATOR_H_
