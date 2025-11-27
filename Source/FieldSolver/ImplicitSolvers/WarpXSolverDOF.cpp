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

#include <AMReX_Scan.H>

using warpx::fields::FieldType;
using namespace amrex;

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
                m_array[lev][n] = std::make_unique<amrex::iMultiFab>(this_array[n]->boxArray(),
                                                                     this_array[n]->DistributionMap(),
                                                                     2*ncomp, // {local, global} for each comp
                                                                     this_array[n]->nGrowVect() );

                auto* mask = a_WarpX->getFieldDotMaskPointer(m_array_type, lev, ablastr::fields::Direction{n});
                fill_local_dof(*m_array[lev][n], *mask);
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
            m_scalar[lev] = std::make_unique<amrex::iMultiFab>(this_mf->boxArray(),
                                                               this_mf->DistributionMap(),
                                                               2*ncomp, // {local, global} for each comp
                                                               this_mf->nGrowVect() );

            auto* mask = a_WarpX->getFieldDotMaskPointer(m_scalar_type, lev, ablastr::fields::Direction{0});
            fill_local_dof(*m_scalar[lev], *mask);
        }

    }

    fill_global_dof();

    for (int lev = 0; lev < a_num_amr_levels; ++lev) {
        for (int n = 0; n < 3; n++) {
            if (auto* dof = m_array[lev][n].get()) {
                for (int comp = 1; comp < dof->nComp(); comp += 2) { // Only call this on global id
                    dof->FillBoundaryAndSync(comp, 1, dof->nGrowVect(), a_WarpX->Geom(lev).periodicity());
                }
            }
        }
        if (auto* dof = m_scalar[lev].get()) {
            for (int comp = 1; comp < dof->nComp(); comp += 2) { // Only call this on global id
                dof->FillBoundaryAndSync(comp, 1, dof->nGrowVect(), a_WarpX->Geom(lev).periodicity());
            }
        }
    }

    amrex::Print() << "Defined DOF object for linear solves (total DOFs = " << m_nDoFs_g << ").\n";
}

void WarpXSolverDOF::fill_local_dof (iMultiFab& dof, iMultiFab const& mask)
{
    int ncomp = dof.nComp() / 2; // /2 because both local and global ids are stored in dof

    AMREX_ALWAYS_ASSERT(dof.boxArray().numPts()*ncomp < static_cast<Long>(std::numeric_limits<int>::max()));

    dof.setVal(std::numeric_limits<int>::lowest());

#ifdef AMREX_USE_MPI
    int nprocs = ParallelDescriptor::NProcs();
#endif

    for (MFIter mfi(dof); mfi.isValid(); ++mfi) {
        Box const& vbx = mfi.validbox();
        int npts = vbx.numPts();
        BoxIndexer boxindex(vbx);
        auto const& m = mask.const_array(mfi);
        auto const& d = dof.array(mfi);
        auto start_id = m_nDoFs_l;
        auto ndofs = Scan::PrefixSum<int>(
            npts,
            [=] AMREX_GPU_DEVICE (int offset) -> int
            {
                auto [i,j,k] = boxindex(offset);
                return m(i,j,k) ? 1 : 0;
            },
            [=] AMREX_GPU_DEVICE (int offset, int ps)
            {
                auto [i,j,k] = boxindex(offset);
                if (m(i,j,k)) {
                    d(i,j,k,0) = ps + start_id;
#ifdef AMREX_USE_MPI
                    if (nprocs == 1)
#endif
                    {
                        d(i,j,k,1) = ps + start_id;
                    }
                }
            },
            Scan::Type::exclusive, Scan::retSum);
        if (ncomp > 1) {
            ParallelFor(vbx, ncomp-1, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                if (m(i,j,k)) {
                    d(i,j,k,2*(n+1)) = d(i,j,k,0) + ndofs*(n+1);
#ifdef AMREX_USE_MPI
                    if (nprocs == 1)
#endif
                    {
                        d(i,j,k,2*(n+1)+1) = d(i,j,k,0) + ndofs*(n+1);
                    }
                }
            });
        }
        m_nDoFs_l += Long(ndofs)*ncomp;
    }
}

void WarpXSolverDOF::fill_global_dof ()
{
#ifndef AMREX_USE_MPI
    m_nDoFs_g = m_nDoFs_l;
#else
    int nprocs = ParallelDescriptor::NProcs();
    if (nprocs == 1) {
        m_nDoFs_g = m_nDoFs_l;
    } else {
        Vector<Long> ndofs_allprocs(nprocs);
        MPI_Allgather(&m_nDoFs_l, 1, ParallelDescriptor::Mpi_typemap<Long>::type(),
                      ndofs_allprocs.data(), 1, ParallelDescriptor::Mpi_typemap<Long>::type(),
                      ParallelDescriptor::Communicator());
        Long proc_begin = 0;
        int myproc = ParallelDescriptor::MyProc();
        m_nDoFs_g = 0;
        for (int iproc = 0; iproc < nprocs; ++iproc) {
            if (iproc < myproc) {
                proc_begin += ndofs_allprocs[iproc];
            }
            m_nDoFs_g += ndofs_allprocs[iproc];
        }
        for (auto& x : m_array) {
            for (auto& y : x) {
                if (y) {
                    auto const& dof = y->arrays();
                    auto ncomp = y->nComp() / 2;
                    ParallelFor(*y, IntVect(0), ncomp, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k, int n)
                    {
                        dof[b](i,j,k,2*n+1) = dof[b](i,j,k,2*n) + int(proc_begin);
                    });
                }
            }
        }
        for (auto& x : m_scalar) {
            if (x) {
                auto const& dof = x->arrays();
                auto ncomp = x->nComp() / 2;
                ParallelFor(*x, IntVect(0), ncomp, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k, int n)
                {
                    dof[b](i,j,k,2*n+1) = dof[b](i,j,k,2*n) + int(proc_begin);
                });
            }
        }
        Gpu::streamSynchronize();
    }
#endif

}
