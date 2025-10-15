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
     * \param[in]     E_like whether the field is E like or B like
     * \param[in]     set_field_lo whether the tangential field at the low boundary was specified
     * \param[in]     set_field_hi whether the tangential field at the high boundary was specified
     * \param[in]     parsers_field1_lo the parser for the first tangential field at the low boundary
     * \param[in]     parsers_field2_lo the parser for the second tangential field at the low boundary
     * \param[in]     parsers_field1_hi the parser for the first tangential field at the high boundary
     * \param[in]     parsers_field2_hi the parser for the second tangential field at the high boundary
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
                               std::vector<int> & set_field_lo,
                               std::vector<int> & set_field_hi,
                               std::vector<std::unique_ptr<amrex::Parser>> const & parsers_field1_lo,
                               std::vector<std::unique_ptr<amrex::Parser>> const & parsers_field2_lo,
                               std::vector<std::unique_ptr<amrex::Parser>> const & parsers_field1_hi,
                               std::vector<std::unique_ptr<amrex::Parser>> const & parsers_field2_hi);

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
                                    std::vector<std::unique_ptr<amrex::Parser>> & parsers,
                                    std::string const & input_name,
                                    std::string const & coord1,
                                    std::string const & coord2);

    std::vector<std::unique_ptr<amrex::Parser>> m_insulator_area_lo;
    std::vector<std::unique_ptr<amrex::Parser>> m_insulator_area_hi;

    std::vector<int> m_set_B_lo = {0, 0, 0};
    std::vector<int> m_set_B_hi = {0, 0, 0};

    std::vector<std::unique_ptr<amrex::Parser>> m_parsers_B1_lo;
    std::vector<std::unique_ptr<amrex::Parser>> m_parsers_B2_lo;
    std::vector<std::unique_ptr<amrex::Parser>> m_parsers_B1_hi;
    std::vector<std::unique_ptr<amrex::Parser>> m_parsers_B2_hi;

    std::vector<int> m_set_E_lo = {0, 0, 0};
    std::vector<int> m_set_E_hi = {0, 0, 0};

    std::vector<std::unique_ptr<amrex::Parser>> m_parsers_E1_lo;
    std::vector<std::unique_ptr<amrex::Parser>> m_parsers_E2_lo;
    std::vector<std::unique_ptr<amrex::Parser>> m_parsers_E1_hi;
    std::vector<std::unique_ptr<amrex::Parser>> m_parsers_E2_hi;

};
#endif // PEC_INSULATOR_H_
