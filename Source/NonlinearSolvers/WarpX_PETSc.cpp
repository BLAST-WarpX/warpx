/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "Preconditioner.H"

#include <AMReX.H>
#include <AMReX_Config.H>
#include <AMReX_REAL.H>
#include <AMReX_ParallelContext.H>

#ifdef AMREX_USE_PETSC

#include <petscksp.h> // must include before WarpX_PETSc.H
#include <petscmat.h> // must include before WarpX_PETSc.H
#include <petscvec.h> // must include before WarpX_PETSc.H
#include "WarpX_PETSc.H"

namespace warpx_petsc {

//! Wrapper for PETSc KSP object
struct KSPObj
{
    KSPObj () = default;
    ~KSPObj () { if (obj) { KSPDestroy(&obj); } }
    KSPObj (KSPObj const&) = delete;
    KSPObj (KSPObj &&) = delete;
    KSPObj& operator= (KSPObj const&) = delete;
    KSPObj& operator= (KSPObj &&) = delete;
    KSP obj = nullptr;
};

//! Wrapper for PETSc Mat object
struct MatObj
{
    MatObj () = default;
    ~MatObj () { if (obj) { MatDestroy(&obj); } }
    MatObj (MatObj const&) = delete;
    MatObj (MatObj &&) = delete;
    MatObj& operator= (MatObj const&) = delete;
    MatObj& operator= (MatObj &&) = delete;
    Mat obj = nullptr;
};

//! Wrapper for PETSc Vec object
struct VecObj
{
    VecObj () = default;
    ~VecObj () { if (obj) { VecDestroy(&obj); } }
    VecObj (VecObj const&) = delete;
    VecObj (VecObj &&) = delete;
    VecObj& operator= (VecObj const&) = delete;
    VecObj& operator= (VecObj &&) = delete;
    Vec obj = nullptr;
};

//! Copy a PETSc vector to a WarpX vector
void copyVec(VecType& a_wvec, const Vec& a_pvec)
{
    BL_PROFILE("warpx_petsc::copyVec()");
    const PetscScalar* Yarr;
    VecGetArrayRead(a_pvec,&Yarr);
    a_wvec.copyFrom( static_cast<const amrex::Real*>(Yarr) );
    VecRestoreArrayRead(a_pvec,&Yarr);
}

//! Copy a WarpX vector to a PETSc vector
void copyVec( Vec& a_pvec, const VecType& a_wvec)
{
    BL_PROFILE("warpx_petsc::copyVec()");
    PetscScalar* Yarr;
    VecGetArray(a_pvec,&Yarr);
    a_wvec.copyTo( static_cast<amrex::Real*>(Yarr) );
    VecRestoreArray(a_pvec,&Yarr);
}

//! Apply matrix-free linear operator
PetscErrorCode applyMatOp(Mat a_A, Vec a_U, Vec a_F)
{
    BL_PROFILE("warpx_petsc::applyMatOp()");

    KSP_impl *context;
    MatShellGetContext(a_A,&context);

    copyVec( context->m_U, a_U );
    context->applyOp( context->m_F, context->m_U );
    copyVec( a_F, context->m_F);

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Apply native preconditioner
PetscErrorCode applyNativePC( PC  a_pc, Vec a_X, Vec a_Y )
{
    BL_PROFILE("warpx_petsc::applyNativePC()");

    KSP_impl *context;
    PCShellGetContext(a_pc, &context);

    copyVec( context->m_U, a_X );
    context->applyPC( context->m_F, context->m_U );
    copyVec( a_Y, context->m_F );

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Print KSP residuals
PetscErrorCode printKSPResidual(KSP a_ksp, PetscInt a_n, PetscReal a_rnorm, void *a_ctxt)
{
    amrex::ignore_unused(a_ctxt);
    amrex::ignore_unused(a_ksp);
    static amrex::Real norm0 = 0;
    if (a_n == 0) { norm0 = a_rnorm; }
    amrex::Print() << "GMRES (PETSc KSP): iter = " << a_n << ", residual = " << a_rnorm
                   << ", " << a_rnorm / norm0 << " (rel.)\n";
    PetscFunctionReturn(PETSC_SUCCESS);
}

KSP_impl::KSP_impl(LinOpType& a_op)
{
    m_op = &a_op;
    PETSC_COMM_WORLD = amrex::ParallelContext::CommunicatorSub();
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    MPI_Comm_size(PETSC_COMM_WORLD, &m_num_procs);
    MPI_Comm_rank(PETSC_COMM_WORLD, &m_myid);
    amrex::Print() << "KSP_impl: Initialized PETSc with "
                   << m_num_procs << " MPI ranks.\n";

    m_ksp = new KSPObj;
    m_A = new MatObj;
    m_P = new MatObj;
    m_x = new VecObj;
    m_b = new VecObj;
}

KSP_impl::~KSP_impl()
{
    delete m_ksp;
    delete m_A;
    delete m_P;
    delete m_x;
    delete m_b;

    PetscFinalize();
    amrex::Print() << "KSP_impl: Finalized PETSc.\n";
}

void KSP_impl::applyOp( VecType& a_F, const VecType& a_U)
{
    AMREX_ALWAYS_ASSERT(isDefined());
    m_op->apply(a_F, a_U);
}

void KSP_impl::applyPC( VecType& a_F, const VecType& a_U)
{
    AMREX_ALWAYS_ASSERT(isDefined());
    a_F.zero();
    m_op->precond(a_F, a_U);
}

void KSP_impl::createObjects(const VecType& a_vec)
{
    BL_PROFILE("KSP_impl::createObjects()");
    AMREX_ALWAYS_ASSERT(!isDefined());

    // define work vector
    m_U.Define(a_vec);
    m_F.Define(a_vec);
    // find local and global vector sizes
    m_ndofs_l = m_U.nDOF_local();
    m_ndofs_g = m_U.nDOF_global();

    // create vectors
    VecCreateMPI(PETSC_COMM_WORLD, m_ndofs_l, m_ndofs_g, &m_x->obj);
    VecDuplicate(m_x->obj, &m_b->obj);
    m_is_defined = true;

    // create matrix operator
    MatCreateShell( PETSC_COMM_WORLD,
                    m_ndofs_l,
                    m_ndofs_l,
                    m_ndofs_g,
                    m_ndofs_g,
                    this,
                    &m_A->obj );
    MatShellSetOperation( m_A->obj, MATOP_MULT,
                          (void (*)(void))applyMatOp );
    MatSetUp(m_A->obj);

    // create KSP object
    KSPCreate( PETSC_COMM_WORLD, &m_ksp->obj );
    KSPSetOperators( m_ksp->obj, m_A->obj, m_A->obj );
    KSPSetTolerances( m_ksp->obj, m_rtol, m_atol, PETSC_CURRENT, m_maxits );
    KSPSetNormType( m_ksp->obj, KSP_NORM_UNPRECONDITIONED );
    if (m_verbose > 0) {
        KSPMonitorSet( m_ksp->obj, printKSPResidual, NULL, NULL );
    }

    // set PC
    PC pc;
    KSPGetPC(m_ksp->obj, &pc);
    auto pc_type = m_op->pcType();
    if (pc_type != PreconditionerType::pc_petsc) {
        // use native implementation (or no PC, if
        // native PC is turned off)
        PCSetType(pc, PCSHELL);
        PCShellSetApply(pc, applyNativePC);
        PCShellSetContext(pc, this);
    } else {
        // use PETSc options and implementation for PC
        PCSetFromOptions(pc);
        // set up the PC sparse matrix
        MatCreate( PETSC_COMM_WORLD, &m_P->obj );
        MatSetSizes( m_P->obj,
                     m_ndofs_l, m_ndofs_l,
                     PETSC_DETERMINE, PETSC_DETERMINE );
        MatSetType( m_P->obj, MATAIJ );
        MatMPIAIJSetPreallocation( m_P->obj, 1 /*a_ops->numPCMatBands()*/, NULL,
                                             1 /*a_ops->numPCMatBands()-1*/, NULL);
        MatSeqAIJSetPreallocation( m_P->obj, 1 /*a_ops->numPCMatBands()*/, NULL);
        MatSetOption(m_P->obj, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
        MatSetOption(m_P->obj, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        MatSetUp(m_P->obj);
    }

    // it is now defined
    m_is_defined = true;

}

void KSP_impl::setTolerances(const amrex::Real a_rtol,
                             const amrex::Real a_atol,
                             const int         a_its )
{
    BL_PROFILE("KSP_impl::setTolerances()");
    m_atol = a_atol;
    m_rtol = a_rtol;
    if (a_its > 0) { m_maxits = a_its; }

    if (isDefined()) {
        KSPSetTolerances( m_ksp->obj,
                          m_rtol,
                          m_atol,
                          PETSC_CURRENT,
                          (a_its > 0 ? a_its : PETSC_CURRENT) );
    }
}

void KSP_impl::setMaxIters(const int a_its )
{
    BL_PROFILE("KSP_impl::setMaxIters()");
    m_maxits = a_its;
    if (isDefined()) {
        KSPSetTolerances( m_ksp->obj,
                          PETSC_CURRENT,
                          PETSC_CURRENT,
                          PETSC_CURRENT,
                          a_its );
    }
}

void KSP_impl::solve(VecType& a_Y, const VecType& a_R)
{
    BL_PROFILE("KSP_impl::solve()");

    AMREX_ALWAYS_ASSERT(isDefined());
    copyVec(m_x->obj, a_Y);
    copyVec(m_b->obj, a_R);

    if (m_op->pcType() == PreconditionerType::pc_petsc) {
        assemblePCMat(a_Y);
    }

    KSPSolve(m_ksp->obj, m_b->obj, m_x->obj);
    copyVec(a_Y, m_x->obj);

    KSPGetIterationNumber( m_ksp->obj, &m_niters );
    KSPConvergedReason reason;
    KSPGetConvergedReason( m_ksp->obj, &reason );
    m_status = (int)reason;
    KSPGetResidualNorm( m_ksp->obj, &m_norm );
}

void KSP_impl::setVerbose(int a_v)
{
    m_verbose = a_v;
    if (a_v > 0 && isDefined()) {
        KSPMonitorSet( m_ksp->obj,
                       (PetscErrorCode (*)(KSP, PetscInt, PetscReal, void *))KSPMonitorResidual,
                       NULL, NULL );
    }
}

void KSP_impl::assemblePCMat(const VecType& a_Y)
{
    BL_PROFILE("KSP_impl::assemblePCMat()");
    amrex::ignore_unused(a_Y);

    AMREX_ALWAYS_ASSERT(isDefined());
    AMREX_ALWAYS_ASSERT(m_P != nullptr);

    PetscInt n = -1;
    PetscInt ncols_max = -1;
    amrex::Gpu::DeviceVector<int> r_indices_g; // global row indices
    amrex::Gpu::DeviceVector<int> n_nz_cols; // number of non-zero columns for each row
    amrex::Gpu::DeviceVector<int> c_indices_g; // non-zero column indices (row-major)
    amrex::Gpu::DeviceVector<amrex::Real> a_ij;

    m_op->getPCMatrix( r_indices_g, n_nz_cols, c_indices_g, a_ij, n, ncols_max );

    {
        std::vector<int> h_r_indices_g(r_indices_g.size());
        std::vector<int> h_n_nz_cols(n_nz_cols.size());
        std::vector<int> h_c_indices_g(c_indices_g.size());
        std::vector<amrex::Real> h_a_ij(a_ij.size());
        amrex::Gpu::copy( amrex::Gpu::deviceToHost,
                          r_indices_g.begin(), r_indices_g.end(),
                          h_r_indices_g.begin() );
        amrex::Gpu::copy( amrex::Gpu::deviceToHost,
                          n_nz_cols.begin(), n_nz_cols.end(),
                          h_n_nz_cols.begin() );
        amrex::Gpu::copy( amrex::Gpu::deviceToHost,
                          c_indices_g.begin(), c_indices_g.end(),
                          h_c_indices_g.begin() );
        amrex::Gpu::copy( amrex::Gpu::deviceToHost,
                          a_ij.begin(), a_ij.end(),
                          h_a_ij.begin() );
        for (int i = 0; i < n; i++) {
            MatSetValues( m_P->obj,
                          1,
                          &h_r_indices_g[i],
                          h_n_nz_cols[i],
                          &h_c_indices_g[i*ncols_max],
                          &h_a_ij[i*ncols_max],
                          INSERT_VALUES );
        }
    }

//    const auto r_indices_g_ptr = r_indices_g.data();
//    const auto n_nz_cols_ptr = n_nz_cols.data();
//    const auto c_indices_g_ptr = c_indices_g.data();
//    const auto a_ij_ptr = a_ij.data();
//
//    auto mat = m_P->obj;
//    amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
//    {
//        MatSetValues( mat,
//                      1,
//                      &r_indices_g_ptr[i],
//                      n_nz_cols_ptr[i],
//                      &c_indices_g_ptr[i*ncols_max],
//                      &a_ij_ptr[i*ncols_max],
//                      INSERT_VALUES );
//    });

    MatAssemblyBegin(m_P->obj, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_P->obj, MAT_FINAL_ASSEMBLY);
}

}

#endif
