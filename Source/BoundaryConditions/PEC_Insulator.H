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
     * \brief Zeros the parallel fields on the boundary
     *        When the E field is being set, the field is zeroed everywhere, in both insulator and conductor.
     *        When the B field id being set, the field is zeroed only in the conductor.
     *
     * \param[in,out] field
     * \param[in]     field_boundary_lo   lower field boundary conditions
     * \param[in]     field_boundary_hi   upper field boundary conditions
     * \param[in]     ng_fieldgather      number of guard cells used by field gather
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     */
    void ZeroParallelFieldInConductor (std::array<amrex::MultiFab*, 3> field,
                                       amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
                                       amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
                                       amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
                                       int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios);

    /**
     * \brief Zeros the scalar on the boundary
     *        When the E field is being set, the scalar is zeroed everywhere, in both insulator and conductor.
     *        When the B field id being set, the scalar is zeroed only in the conductor.
     *
     * \param[in,out] scalar
     * \param[in]     field_boundary_lo   lower field boundary conditions
     * \param[in]     field_boundary_hi   upper field boundary conditions
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     */
    void ZeroParallelScalarInConductor (amrex::MultiFab* scalar,
                                        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
                                        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
                                        amrex::Geometry const & geom, int lev, PatchType patch_type,
                                        amrex::Vector<amrex::IntVect> const & ref_ratios);

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
     * \param[in]     only_zero_parallel_field only zero the parallel field on the boundary
     * \param[in]     set_Fx_lo           the flags for the x-field at the lower boundaries
     * \param[in]     set_Fy_lo           the flags for the y-field at the lower boundaries
     * \param[in]     set_Fz_lo           the flags for the z-field at the lower boundaries
     * \param[in]     set_Fx_hi           the flags for the x-field at the upper boundaries
     * \param[in]     set_Fy_hi           the flags for the y-field at the upper boundaries
     * \param[in]     set_Fz_hi           the flags for the z-field at the upper boundaries
     * \param[in]     Fx_parsers_lo       the parsers for the x-field at the lower boundaries
     * \param[in]     Fy_parsers_lo       the parsers for the y-field at the lower boundaries
     * \param[in]     Fz_parsers_lo       the parsers for the z-field at the lower boundaries
     * \param[in]     Fx_parsers_hi       the parsers for the x-field at the upper boundaries
     * \param[in]     Fy_parsers_hi       the parsers for the y-field at the upper boundaries
     * \param[in]     Fz_parsers_hi       the parsers for the z-field at the upper boundaries
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
                               bool only_zero_parallel_field,
                               amrex::Vector<int> const & set_Fx_lo,
                               amrex::Vector<int> const & set_Fy_lo,
                               amrex::Vector<int> const & set_Fz_lo,
                               amrex::Vector<int> const & set_Fx_hi,
                               amrex::Vector<int> const & set_Fy_hi,
                               amrex::Vector<int> const & set_Fz_hi,
                               amrex::Vector<amrex::ParserExecutor<3>> const & Fx_parsers_lo,
                               amrex::Vector<amrex::ParserExecutor<3>> const & Fy_parsers_lo,
                               amrex::Vector<amrex::ParserExecutor<3>> const & Fz_parsers_lo,
                               amrex::Vector<amrex::ParserExecutor<3>> const & Fx_parsers_hi,
                               amrex::Vector<amrex::ParserExecutor<3>> const & Fy_parsers_hi,
                               amrex::Vector<amrex::ParserExecutor<3>> const & Fz_parsers_hi);

    /* \brief Returns whether any E field is set on a boundary
     * \param[in] idim dimension to check
     * \param[in] iside side to check, 0 is lower, 1 is upper
     * \return whether the E field is set on the boundary
     */
    int IsESet(int idim, int iside) const { return ( iside == 1 ? m_set_E_hi[idim] : m_set_E_lo[idim] ); }

    /* \brief Returns whether any B field is set on a boundary
     * \param[in] idim dimension to check
     * \param[in] iside side to check, 0 is lower, 1 is upper
     * \return whether the B field is set on the boundary
     */
    int IsBSet(int idim, int iside) const { return ( iside == 1 ? m_set_B_hi[idim] : m_set_B_lo[idim] ); }

    /* \brief Returns whether the E field is set on a boundary
     * \param[in] idim dimension to check
     * \param[in] iside side to check, 0 is lower, 1 is upper
     * \param[in] ifield which field to check, 0, 1, or 2
     * \return whether the E field is set on the boundary
     */
    int IsESet(int idim, int iside, int ifield) const;

    /* \brief Returns whether the B field is set on a boundary
     * \param[in] idim dimension to check
     * \param[in] iside side to check, 0 is lower, 1 is upper
     * \param[in] ifield which field to check, 0, 1, or 2
     * \return whether the B field is set on the boundary
     */
    int IsBSet(int idim, int iside, int ifield) const;

private:


    // Parsers specifying the region of each boundary that is insulator
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_insulator_area_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_insulator_area_hi;

    amrex::Vector<amrex::ParserExecutor<2>> m_area_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<2>> m_area_parsers_hi;

    // Flags whether any B fields are specified on each boundary
    amrex::Vector<int> m_set_B_lo = {0, 0, 0};
    amrex::Vector<int> m_set_B_hi = {0, 0, 0};

    // Parsers for the two B fields tangential to each boundary
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_B1_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_B2_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_B1_hi;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_B2_hi;

    // Flags whether any E fields are specified on each boundary
    amrex::Vector<int> m_set_E_lo = {0, 0, 0};
    amrex::Vector<int> m_set_E_hi = {0, 0, 0};

    // Parsers for the two B fields tangential to each boundary
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_E1_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_E2_lo;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_E1_hi;
    amrex::Vector<std::unique_ptr<amrex::Parser>> m_parsers_E2_hi;

    // For each B field, the flags specifying whether it is set on each boundary
    amrex::Vector<int> m_set_Bx_lo;
    amrex::Vector<int> m_set_By_lo;
    amrex::Vector<int> m_set_Bz_lo;
    amrex::Vector<int> m_set_Bx_hi;
    amrex::Vector<int> m_set_By_hi;
    amrex::Vector<int> m_set_Bz_hi;

    // For each B field, the parsers for the field value on each boundary
    amrex::Vector<amrex::ParserExecutor<3>> m_Bx_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<3>> m_By_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<3>> m_Bz_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<3>> m_Bx_parsers_hi;
    amrex::Vector<amrex::ParserExecutor<3>> m_By_parsers_hi;
    amrex::Vector<amrex::ParserExecutor<3>> m_Bz_parsers_hi;

    // For each E field, the flags specifying whether it is set on each boundary
    amrex::Vector<int> m_set_Ex_lo;
    amrex::Vector<int> m_set_Ey_lo;
    amrex::Vector<int> m_set_Ez_lo;
    amrex::Vector<int> m_set_Ex_hi;
    amrex::Vector<int> m_set_Ey_hi;
    amrex::Vector<int> m_set_Ez_hi;

    // For each E field, the parsers for the field value on each boundary
    amrex::Vector<amrex::ParserExecutor<3>> m_Ex_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<3>> m_Ey_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<3>> m_Ez_parsers_lo;
    amrex::Vector<amrex::ParserExecutor<3>> m_Ex_parsers_hi;
    amrex::Vector<amrex::ParserExecutor<3>> m_Ey_parsers_hi;
    amrex::Vector<amrex::ParserExecutor<3>> m_Ez_parsers_hi;

};
#endif // PEC_INSULATOR_H_
