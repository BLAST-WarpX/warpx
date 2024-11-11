/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "WarpX.H"

using warpx::fields::FieldType;

WarpXSolverVec::~WarpXSolverVec ()
{
    for (auto & lvl : m_array_vec)
    {
        for (int i =0; i<3; ++i)
        {
            delete lvl[i];
        }
    }
}

void WarpXSolverVec::Define ( WarpX*  a_WarpX,
                         const std::string&  a_vector_type_name,
                         const std::string&  a_scalar_type_name )
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "WarpXSolverVec::Define() called on already defined WarpXSolverVec");

    // Define static member pointer to WarpX
    if (!m_warpx_ptr_defined) {
        m_WarpX = a_WarpX;
        m_warpx_ptr_defined = true;
    }

    m_num_amr_levels = 1;

    m_vector_type_name = a_vector_type_name;
    m_scalar_type_name = a_scalar_type_name;

    if (m_vector_type_name=="Efield_fp") {
        m_array_type = FieldType::Efield_fp;
    }
    else if (m_vector_type_name=="Bfield_fp") {
        m_array_type = FieldType::Bfield_fp;
    }
    else if (m_vector_type_name=="vector_potential_fp_nodal") {
        m_array_type = FieldType::vector_potential_fp;
    }
    else if (m_vector_type_name!="none") {
        WARPX_ABORT_WITH_MESSAGE(a_vector_type_name
                    +"is not a valid option for array type used in Definining"
                    +"a WarpXSolverVec. Valid array types are: Efield_fp, Bfield_fp,"
                    +"and vector_potential_fp_nodal");
    }

    if (m_scalar_type_name=="phi_fp") {
        m_scalar_type = FieldType::phi_fp;
    }
    else if (m_scalar_type_name!="none") {
        WARPX_ABORT_WITH_MESSAGE(a_scalar_type_name
                    +"is not a valid option for scalar type used in Definining"
                    +"a WarpXSolverVec. Valid scalar types are: phi_fp");
    }

    m_array_vec.resize(m_num_amr_levels);
    m_scalar_vec.resize(m_num_amr_levels);

    // Define the 3D vector field data container
    if (m_array_type != FieldType::None) {

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            isFieldArray(m_array_type),
            "WarpXSolverVec::Define() called with array_type not an array field");

        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            const ablastr::fields::VectorField this_array = m_WarpX->m_fields.get_alldirs(m_vector_type_name, lev);
            for (int n = 0; n < 3; n++) {
                m_array_vec[lev][n] = new amrex::MultiFab( this_array[n]->boxArray(),
                                                           this_array[n]->DistributionMap(),
                                                           this_array[n]->nComp(),
                                                           amrex::IntVect::TheZeroVector() );
            }
        }

    }

    // Define the scalar data container
    if (m_scalar_type != FieldType::None) {

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !isFieldArray(m_scalar_type),
            "WarpXSolverVec::Define() called with scalar_type not a scalar field ");

        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            const amrex::MultiFab* this_mf = m_WarpX->m_fields.get(m_scalar_type_name,lev);
            m_scalar_vec[lev] = new amrex::MultiFab( this_mf->boxArray(),
                                                     this_mf->DistributionMap(),
                                                     this_mf->nComp(),
                                                     amrex::IntVect::TheZeroVector() );
        }

    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_array_type != FieldType::None ||
        m_scalar_type != FieldType::None,
        "WarpXSolverVec cannot be defined with both array and scalar vecs FieldType::None");

    m_is_defined = true;
}

void WarpXSolverVec::Copy ( FieldType  a_array_type,
                            FieldType  a_scalar_type )
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::Copy() called on undefined WarpXSolverVec");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_array_type==m_array_type &&
        a_scalar_type==m_scalar_type,
        "WarpXSolverVec::Copy() called with vecs of different types");

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            const ablastr::fields::VectorField this_array = m_WarpX->m_fields.get_alldirs(m_vector_type_name, lev);
            for (int n = 0; n < 3; ++n) {
                amrex::MultiFab::Copy( *m_array_vec[lev][n], *this_array[n], 0, 0, m_ncomp,
                                       amrex::IntVect::TheZeroVector() );
            }
        }
        if (m_scalar_type != FieldType::None) {
            const amrex::MultiFab* this_mf = m_WarpX->m_fields.get(m_scalar_type_name,lev);
            amrex::MultiFab::Copy( *m_scalar_vec[lev], *this_mf, 0, 0, m_ncomp,
                                   amrex::IntVect::TheZeroVector() );
        }
    }
}

[[nodiscard]] amrex::Real WarpXSolverVec::dotProduct ( const WarpXSolverVec&  a_X ) const
{
    assertIsDefined( a_X );
    assertSameType( a_X );

    amrex::Real result = 0.0;
    const bool local = true;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) {
                const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(m_array_type,lev,n);
                auto rtmp = amrex::MultiFab::Dot( *dotMask,
                                                  *m_array_vec[lev][n], 0,
                                                  *a_X.getArrayVec()[lev][n], 0, 1, 0, local);
                result += rtmp;
            }
        }
        if (m_scalar_type != FieldType::None) {
            const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(m_scalar_type,lev,0);
            auto rtmp = amrex::MultiFab::Dot( *dotMask,
                                              *m_scalar_vec[lev], 0,
                                              *a_X.getScalarVec()[lev], 0, 1, 0, local);
            result += rtmp;
        }
    }
    amrex::ParallelAllReduce::Sum(result, amrex::ParallelContext::CommunicatorSub());
    return result;
}
