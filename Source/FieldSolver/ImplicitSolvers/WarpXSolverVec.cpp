/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "WarpX.H"

using warpx::fields::FieldType;
std::unique_ptr<WarpXSolverDOF> WarpXSolverVec::m_dofs = nullptr;

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

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_WarpX != nullptr,
        "WarpXSolverVec::Define() called with a nullptr WarpX instance");

    m_WarpX = a_WarpX;

    m_num_amr_levels =
        m_WarpX->evolve_scheme == EvolveScheme::Theta_Implicit_EM ? m_WarpX->finestLevel() + 1 : 1;

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
        WARPX_ABORT_WITH_MESSAGE(a_vector_type_name+" "
                    +"is not a valid option for array type used in Definining "
                    +"a WarpXSolverVec. Valid array types are: Efield_fp, Bfield_fp, "
                    +"and vector_potential_fp_nodal");
    }

    if (m_scalar_type_name=="phi_fp") {
        m_scalar_type = FieldType::phi_fp;
    }
    else if (m_scalar_type_name!="none") {
        WARPX_ABORT_WITH_MESSAGE(a_scalar_type_name+" "
                    +"is not a valid option for scalar type used in Definining "
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

    if (m_array_type == FieldType::Efield_fp &&
        m_WarpX->evolve_scheme == EvolveScheme::Theta_Implicit_EM) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (auto const type :
                 {FieldType::Efield_cp, FieldType::pml_E_fp, FieldType::pml_E_cp}) {
                if (!m_WarpX->m_fields.has_vector(type, lev)) {
                    continue;
                }
                ExtraField block;
                block.type = type;
                block.level = lev;
                auto const source = m_WarpX->m_fields.get_alldirs(type, lev);
                bool const coarse = type == FieldType::Efield_cp || type == FieldType::pml_E_cp;
                auto const& geometry = m_WarpX->Geom(coarse ? lev - 1 : lev);
                for (int n = 0; n < 3; ++n) {
                    block.values[n] = std::make_unique<amrex::MultiFab>(
                        source[n]->boxArray(), source[n]->DistributionMap(), source[n]->nComp(), 0);
                    block.dot_masks[n] = source[n]->OwnerMask(geometry.periodicity());
                    if (type != FieldType::Efield_cp) {
                        // Exchange gives regular cells ownership of the shared PML boundary.
                        auto const regular_type =
                            coarse ? FieldType::Efield_cp : FieldType::Efield_fp;
                        auto const& regular =
                            m_WarpX->m_fields.get(regular_type, ablastr::fields::Direction{n}, lev)
                                ->boxArray();
                        for (amrex::MFIter mfi(*block.dot_masks[n]); mfi.isValid(); ++mfi) {
                            auto const mask = block.dot_masks[n]->array(mfi);
                            for (auto const& intersection : regular.intersections(mfi.validbox())) {
                                amrex::ParallelFor(intersection.second,
                                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                                       mask(i, j, k) = 0;
                                                   });
                            }
                        }
                    }
                }
                m_extra_fields.push_back(std::move(block));
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
    if (m_dofs == nullptr) {
        m_dofs = std::make_unique<WarpXSolverDOF>();
        m_dofs->Define(m_WarpX, m_num_amr_levels, m_vector_type_name, m_scalar_type_name);
        amrex::ExecOnFinalize([p=&m_dofs] () { p->reset(); });
    }

    m_is_defined = true;
}

void WarpXSolverVec::Copy ( warpx::fields::FieldType  a_array_type,
                            warpx::fields::FieldType  a_scalar_type,
                            bool allow_type_mismatch)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::Copy() called on undefined WarpXSolverVec");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_array_type==m_array_type &&
        a_scalar_type==m_scalar_type) || allow_type_mismatch,
        "WarpXSolverVec::Copy() called with vecs of different types");

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            const ablastr::fields::VectorField this_array = m_WarpX->m_fields.get_alldirs(a_array_type, lev);
            for (int n = 0; n < 3; ++n) {
                amrex::MultiFab::Copy( *m_array_vec[lev][n], *this_array[n], 0, 0, m_ncomp,
                                       amrex::IntVect::TheZeroVector() );
            }
        }
        if (m_scalar_type != FieldType::None) {
            const amrex::MultiFab* this_mf = m_WarpX->m_fields.get(a_scalar_type,lev);
            amrex::MultiFab::Copy( *m_scalar_vec[lev], *this_mf, 0, 0, m_ncomp,
                                   amrex::IntVect::TheZeroVector() );
        }
    }
    // CP/PML guesses are reset to the current fields at each step, including restart.
    for (auto& block : m_extra_fields) {
        auto const source = m_WarpX->m_fields.get_alldirs(block.type, block.level);
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*block.values[n], *source[n], 0, 0, source[n]->nComp(), 0);
        }
    }
}

void WarpXSolverVec::copyFrom ( const amrex::Real* const a_arr)
{
    BL_PROFILE("WarpXSolverVec::copyFrom");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_extra_fields.empty(),
        "Packing coarse-patch or PML unknowns into PETSc arrays is not implemented.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::CopyFrom() called on undefined WarpXSolverVec");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_dofs != nullptr),
        "WarpXSolverVec::CopyFrom() DOF object is a nullptr");
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) {
                auto ncomp = m_array_vec[lev][n]->nComp();
                for (amrex::MFIter mfi(*(m_dofs->m_array)[lev][n]); mfi.isValid(); ++mfi) {
                    auto bx = mfi.tilebox();
                    auto data_arr = m_array_vec[lev][n]->array(mfi);
                    auto dof_arr = m_dofs->m_array[lev][n]->const_array(mfi);
                    ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for (int v = 0; v < ncomp; v++) {
                            const  int dof = dof_arr(i,j,k,2*v); // local
                            if (dof >= 0) {
                                data_arr(i,j,k,v) = a_arr[dof];
                            }
                        }
                    });
                }
                m_array_vec[lev][n]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
            }
        }
        if (m_scalar_type != FieldType::None) {
            auto ncomp = m_scalar_vec[lev]->nComp();
            for (amrex::MFIter mfi(*(m_dofs->m_scalar)[lev]); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                auto data_arr = m_scalar_vec[lev]->array(mfi);
                auto dof_arr = m_dofs->m_scalar[lev]->const_array(mfi);
                ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    for (int v = 0; v < ncomp; v++) {
                        const int dof = dof_arr(i,j,k,2*v); // local
                        if (dof >= 0) {
                            data_arr(i,j,k,v) = a_arr[dof];
                        }
                    }
                });
            }
            m_scalar_vec[lev]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
        }
    }
}

void WarpXSolverVec::copyTo ( amrex::Real* const a_arr) const
{
    BL_PROFILE("WarpXSolverVec::copyTo");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_extra_fields.empty(),
        "Packing coarse-patch or PML unknowns into PETSc arrays is not implemented.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::CopyTo() called on undefined WarpXSolverVec");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_dofs != nullptr),
        "WarpXSolverVec::CopyTo() DOF object is a nullptr");
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) {
                auto ncomp = m_array_vec[lev][n]->nComp();
                for (amrex::MFIter mfi(*(m_dofs->m_array)[lev][n]); mfi.isValid(); ++mfi) {
                    auto bx = mfi.tilebox();
                    auto data_arr = m_array_vec[lev][n]->const_array(mfi);
                    auto dof_arr = m_dofs->m_array[lev][n]->const_array(mfi);
                    ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for (int v = 0; v < ncomp; v++) {
                            const int dof = dof_arr(i,j,k,2*v); // local
                            if (dof >= 0) {
                                a_arr[dof] = data_arr(i,j,k,v);
                            }
                        }
                    });
                }
            }
        }
        if (m_scalar_type != FieldType::None) {
            auto ncomp = m_scalar_vec[lev]->nComp();
            for (amrex::MFIter mfi(*(m_dofs->m_scalar)[lev]); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                auto data_arr = m_scalar_vec[lev]->const_array(mfi);
                auto dof_arr = m_dofs->m_scalar[lev]->const_array(mfi);
                ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    for (int v = 0; v < ncomp; v++) {
                        const int dof = dof_arr(i,j,k,2*v); // local
                        if (dof >= 0) {
                            a_arr[dof] = data_arr(i,j,k,v);
                        }
                    }
                });
            }
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
                const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(m_array_type, lev, ablastr::fields::Direction{n});
                auto rtmp = amrex::MultiFab::Dot( *dotMask,
                                                  *m_array_vec[lev][n], 0,
                                                  *a_X.getArrayVec()[lev][n], 0, 1, 0, local);
                result += rtmp;
            }
        }
        if (m_scalar_type != FieldType::None) {
            const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(m_scalar_type,lev, ablastr::fields::Direction{0});
            auto rtmp = amrex::MultiFab::Dot( *dotMask,
                                              *m_scalar_vec[lev], 0,
                                              *a_X.getScalarVec()[lev], 0, 1, 0, local);
            result += rtmp;
        }
    }
    for (std::size_t b = 0; b < m_extra_fields.size(); ++b) {
        for (int n = 0; n < 3; ++n) {
            auto const& block = m_extra_fields[b];
            result += amrex::MultiFab::Dot(*block.dot_masks[n], *block.values[n], 0,
                                           *a_X.m_extra_fields[b].values[n], 0,
                                           block.values[n]->nComp(), 0, local);
        }
    }
    amrex::ParallelAllReduce::Sum(result, amrex::ParallelContext::CommunicatorSub());
    return result;
}

ablastr::fields::VectorField
WarpXSolverVec::getExtraField (FieldType type, int level) const
{
    for (auto const& block : m_extra_fields) {
        if (block.type == type && block.level == level) {
            return block.get();
        }
    }
    WARPX_ABORT_WITH_MESSAGE("The implicit solver does not contain the requested field block.");
    return {};
}

void
WarpXSolverVec::ExtraLinComb (RT a, WarpXSolverVec const& X, RT b, WarpXSolverVec const& Y)
{
    for (std::size_t block = 0; block < m_extra_fields.size(); ++block) {
        for (int n = 0; n < 3; ++n) {
            auto& field = *m_extra_fields[block].values[n];
            amrex::MultiFab::LinComb(field, a, *X.m_extra_fields[block].values[n], 0, b,
                                     *Y.m_extra_fields[block].values[n], 0, 0, field.nComp(), 0);
        }
    }
}

void
WarpXSolverVec::ExtraIncrement (WarpXSolverVec const& X, RT a)
{
    for (std::size_t block = 0; block < m_extra_fields.size(); ++block) {
        for (int n = 0; n < 3; ++n) {
            auto& field = *m_extra_fields[block].values[n];
            amrex::MultiFab::Saxpy(field, a, *X.m_extra_fields[block].values[n], 0, 0,
                                   field.nComp(), 0);
        }
    }
}
