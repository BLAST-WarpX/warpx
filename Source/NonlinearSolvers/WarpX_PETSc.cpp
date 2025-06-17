/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "Preconditioner.H"

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
    m_x = new VecObj;
    m_b = new VecObj;
}

KSP_impl::~KSP_impl()
{
    delete m_ksp;
    delete m_A;
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
        PCSetType(pc, PCSHELL);
        PCShellSetApply(pc, applyNativePC);
        PCShellSetContext(pc, this);
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

}

#endif
