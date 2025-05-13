/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/WarpXSolverDOF.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/utils/SignalHandling.H>
#include <ablastr/warn_manager/WarnManager.H>

using warpx::fields::FieldType;

void WarpXSolverDOF::Define ( WarpX* const        a_WarpX,
                              const int           a_num_amr_levels,
                              const std::string&  a_vector_type_name,
                              const std::string&  a_scalar_type_name )
{
    if (a_vector_type_name=="Efield_fp") {
        m_array_type = FieldType::Efield_fp;
    } else if (a_vector_type_name=="Bfield_fp") {
        m_array_type = FieldType::Bfield_fp;
    } else if (a_vector_type_name=="vector_potential_fp_nodal") {
        m_array_type = FieldType::vector_potential_fp;
    } else if (a_vector_type_name!="none") {
        WARPX_ABORT_WITH_MESSAGE(a_vector_type_name
                    +"is not a valid option for array type used in Definining"
                    +"a WarpXSolverDOF. Valid array types are: Efield_fp, Bfield_fp,"
                    +"and vector_potential_fp_nodal");
    }

    if (a_scalar_type_name=="phi_fp") {
        m_scalar_type = FieldType::phi_fp;
    } else if (a_scalar_type_name!="none") {
        WARPX_ABORT_WITH_MESSAGE(a_scalar_type_name
                    +"is not a valid option for scalar type used in Defining"
                    +"a WarpXSolverDOF. Valid scalar types are: phi_fp");
    }

    m_array.resize(a_num_amr_levels);
    m_scalar.resize(a_num_amr_levels);

    amrex::Long offset = 0;
    m_nDoFs_l = 0;

    // Define the 3D vector field data container
    if (m_array_type != FieldType::None) {

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            isFieldArray(m_array_type),
            "WarpXSolverDOF::Define() called with array_type not an array field");

        for (int lev = 0; lev < a_num_amr_levels; ++lev) {
            const ablastr::fields::VectorField this_array = a_WarpX->m_fields.get_alldirs(a_vector_type_name, lev);
            for (int n = 0; n < 3; n++) {
                auto ncomp = this_array[n]->nComp();
                m_array[lev][n] = new amrex::MultiFab( this_array[n]->boxArray(),
                                                                this_array[n]->DistributionMap(),
                                                                2*ncomp, // {local, global} for each comp
                                                                amrex::IntVect::TheUnitVector() );
                m_nDoFs_g += this_array[n]->boxArray().numPts()*ncomp;

                m_array[lev][n]->setVal(-1.0);
                amrex::Long offset_mf = 0;
                for (amrex::MFIter mfi(*m_array[lev][n]); mfi.isValid(); ++mfi) {
                    auto bx = mfi.tilebox();
                    auto dof_arr = m_array[lev][n]->array(mfi);
                    ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for (int v = 0; v < ncomp; v++) {
                            dof_arr(i,j,k,2*v) = (amrex::Real) bx.index(amrex::IntVect(AMREX_D_DECL(i, j, k))) * ncomp
                                                 + (amrex::Real) offset_mf
                                                 + (amrex::Real) offset;
                        }
                    });
                    offset_mf += bx.numPts()*ncomp;
                }
                offset += offset_mf;
                m_nDoFs_l += offset_mf;
            }
        }

    }

    // Define the scalar data container
    if (m_scalar_type != FieldType::None) {

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !isFieldArray(m_scalar_type),
            "WarpXSolverDOF::Define() called with scalar_type not a scalar field ");

        for (int lev = 0; lev < a_num_amr_levels; ++lev) {
            const amrex::MultiFab* this_mf = a_WarpX->m_fields.get(a_scalar_type_name,lev);
            auto ncomp = this_mf->nComp();
            m_scalar[lev] = new amrex::MultiFab( this_mf->boxArray(),
                                                          this_mf->DistributionMap(),
                                                          2*ncomp, // {local, global} for each comp
                                                          amrex::IntVect::TheUnitVector() );
            m_nDoFs_g += this_mf->boxArray().numPts()*ncomp;

            m_scalar[lev]->setVal(-1.0);
            amrex::Long offset_mf = 0;
            for (amrex::MFIter mfi(*m_scalar[lev]); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                auto dof_arr = m_scalar[lev]->array(mfi);
                ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    for (int v = 0; v < ncomp; v++) {
                        dof_arr(i,j,k,2*v) = (amrex::Real) bx.index(amrex::IntVect(AMREX_D_DECL(i, j, k))) * ncomp
                                             + (amrex::Real) offset_mf
                                             + (amrex::Real) offset;
                    }
                });
                offset_mf += bx.numPts()*ncomp;
            }
            offset += offset_mf;
            m_nDoFs_l += offset_mf;
        }

    }

    auto nDoFs_g = m_nDoFs_l;
    amrex::ParallelDescriptor::ReduceLongSum(&nDoFs_g,1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_nDoFs_g == nDoFs_g,
        "WarpXSolverDOF::Define(): something has gone wrong in DoF counting");

    auto num_procs = amrex::ParallelDescriptor::NProcs();
    auto my_proc = amrex::ParallelDescriptor::MyProc();
    amrex::Vector<int> dof_proc_arr(num_procs,0);
    dof_proc_arr[my_proc] = m_nDoFs_l;
    amrex::ParallelDescriptor::ReduceIntSum(dof_proc_arr.data(), num_procs);

    int offset_global = 0;
    for (int i = 0; i < my_proc; i++) { offset_global += dof_proc_arr[i]; }

    if (m_array_type != FieldType::None) {
        for (int lev = 0; lev < a_num_amr_levels; ++lev) {
            const ablastr::fields::VectorField this_array = a_WarpX->m_fields.get_alldirs(a_vector_type_name, lev);
            for (int n = 0; n < 3; n++) {
                auto ncomp = this_array[n]->nComp();
                for (amrex::MFIter mfi(*m_array[lev][n]); mfi.isValid(); ++mfi) {
                    auto bx = mfi.tilebox();
                    auto dof_arr = m_array[lev][n]->array(mfi);
                    ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for (int v = 0; v < ncomp; v++) {
                            dof_arr(i,j,k,2*v+1) = dof_arr(i,j,k,2*v) + (amrex::Real) offset_global;
                        }
                    });
                }
            }
        }
    }
    if (m_scalar_type != FieldType::None) {
        for (int lev = 0; lev < a_num_amr_levels; ++lev) {
            const amrex::MultiFab* this_mf = a_WarpX->m_fields.get(a_scalar_type_name,lev);
            auto ncomp = this_mf->nComp();
            for (amrex::MFIter mfi(*m_scalar[lev]); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                auto dof_arr = m_scalar[lev]->array(mfi);
                ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    for (int v = 0; v < ncomp; v++) {
                        dof_arr(i,j,k,2*v+1) = dof_arr(i,j,k,2*v) + (amrex::Real) offset_global;
                    }
                });
            }
        }
    }

    amrex::Print() << "Defined DOF object for linear solves (total DOFs = " << m_nDoFs_g << ").\n";
}
