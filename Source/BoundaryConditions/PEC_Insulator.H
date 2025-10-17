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
     * \param[in]     split_pml_field     whether pml the multifab is the regular Efield or split pml field
     * \param[in]     E_like              whether the field is E like or B like
     * \param[in]     set_fields_x_lo     the flags for the field along x at the lower boundary
     * \param[in]     set_fields_y_lo     the flags for the field along y at the lower boundary
     * \param[in]     set_fields_z_lo     the flags for the field along z at the lower boundary
     * \param[in]     set_fields_x_hi     the flags for the field along x at the upper boundary
     * \param[in]     set_fields_y_hi     the flags for the field along y at the upper boundary
     * \param[in]     set_fields_z_hi     the flags for the field along z at the upper boundary
     * \param[in]     Fx_parsers_lo       the parsers for the field along x at the lower boundary
     * \param[in]     Fy_parsers_lo       the parsers for the field along y at the lower boundary
     * \param[in]     Fz_parsers_lo       the parsers for the field along z at the lower boundary
     * \param[in]     Fx_parsers_hi       the parsers for the field along x at the upper boundary
     * \param[in]     Fy_parsers_hi       the parsers for the field along y at the upper boundary
     * \param[in]     Fz_parsers_hi       the parsers for the field along z at the upper boundary
     */
    void
    ApplyPEC_InsulatortoField (std::array<amrex::MultiFab*, 3> field,
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
                               amrex::Vector<amrex::ParserExecutor<3>> const & Fz_parsers_hi);

private:

    /* \brief Read in the parsers for the tangential fields, returning whether
     *        the input parameter was specified.
     * \param[in]  pp_insulator ParmParse instance
     * \param[out] parsers vector holding the parsers generated from the input
     * \param[in]  input_name the name of the input parameter
     * \param[in]  coord1 the first coordinate in the plane
     * \param[in]  coord2 the second coordinate in the plane
     * \return wether the parameter was specified
     */
    bool ReadTangentialFieldParser (amrex::ParmParse const & pp_insulator,
                                    amrex::Vector<std::unique_ptr<amrex::Parser>> & parsers,
                                    std::string const & input_name,
                                    std::string const & coord1,
                                    std::string const & coord2);

    amrex::Vector<std::unique_ptr<amrex::Parser>> m_insulator_area_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_insulator_area_hi;

    amrex::Vector<amrex::ParserExecutor<2>> m_area_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<2>> m_area_parsers_hi;

    amrex::Vector<int> m_set_B_lo = {0, 0, 0};
    amrex::Vector<int> m_set_B_hi = {0, 0, 0};

    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_B1_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_B2_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_B1_hi;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_B2_hi;

    amrex::Vector<int> m_set_E_lo = {0, 0, 0};
    amrex::Vector<int> m_set_E_hi = {0, 0, 0};

    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_E1_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_E2_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_E1_hi;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_E2_hi;

    amrex::Vector<int> m_set_Bfields_x_lo;
    amrex::Vector<int> m_set_Bfields_y_lo;
    amrex::Vector<int> m_set_Bfields_z_lo;

    amrex::Vector<int> m_set_Bfields_x_hi;
    amrex::Vector<int> m_set_Bfields_y_hi;
    amrex::Vector<int> m_set_Bfields_z_hi;

    amrex::Vector<amrex::ParserExecutor<3>> m_Bx_parsers_lo, m_By_parsers_lo, m_Bz_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<3>> m_Bx_parsers_hi, m_By_parsers_hi, m_Bz_parsers_hi;

    amrex::Vector<int> m_set_Efields_x_lo;
    amrex::Vector<int> m_set_Efields_y_lo;
    amrex::Vector<int> m_set_Efields_z_lo;

    amrex::Vector<int> m_set_Efields_x_hi;
    amrex::Vector<int> m_set_Efields_y_hi;
    amrex::Vector<int> m_set_Efields_z_hi;

    amrex::Vector<amrex::ParserExecutor<3>> m_Ex_parsers_lo, m_Ey_parsers_lo, m_Ez_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<3>> m_Ex_parsers_hi, m_Ey_parsers_hi, m_Ez_parsers_hi;

};
#endif // PEC_INSULATOR_H_
