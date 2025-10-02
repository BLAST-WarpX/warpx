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

#include <petscsnes.h> // must include before WarpX_PETSc.H
#include <petscksp.h> // must include before WarpX_PETSc.H
#include <petscmat.h> // must include before WarpX_PETSc.H
#include <petscvec.h> // must include before WarpX_PETSc.H
#include "WarpX_PETSc.H"

namespace warpx_petsc {

//! Wrapper for PETSc SNES object
struct SNESObj
{
    SNESObj () = default;
    ~SNESObj () { if (obj) { SNESDestroy(&obj); } }
    SNESObj (SNESObj const&) = delete;
    SNESObj (SNESObj &&) = delete;
    SNESObj& operator= (SNESObj const&) = delete;
    SNESObj& operator= (SNESObj &&) = delete;
    SNES obj = nullptr;
};

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

//! Compute RHS function
PetscErrorCode RHSFunction( SNES a_solver, Vec a_U, Vec a_F, void* ctxt)
{
    BL_PROFILE("warpx_petsc::RHSFunction()");
    amrex::ignore_unused(a_solver);

    SNES_impl *context = (SNES_impl*) ctxt;
    copyVec(context->m_U, a_U);
    context->computeRHS(context->m_F, context->m_U);
    copyVec(a_F, context->m_F);
    VecAXPBY(a_F, 1.0, -1.0, a_U);

    if (!context->m_fd_jac_comput) {
        ((JacobianFunctionMF<VecType,TIType>*)context->m_linop)->updatePreCondMat(context->m_U);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Compute Jacobian
PetscErrorCode JacobianFunction( SNES a_solver,
                                 Vec a_U,
                                 Mat a_A,
                                 Mat a_P,
                                 void* ctxt )
{
    BL_PROFILE("warpx_petsc::JacobianFunction()");
    amrex::ignore_unused(a_A);
    amrex::ignore_unused(a_P);

    SNES_impl *context = (SNES_impl*) ctxt;
    KSP lin_solver;
    SNESGetKSP(a_solver, &lin_solver);
    PC pc;
    KSPGetPC(lin_solver, &pc);
    PCType pctype;
    PCGetType(pc, &pctype);

    if (strcmp(pctype,PCNONE) && strcmp(pctype,PCSHELL)) {
        copyVec(context->m_U, a_U);
        context->assemblePCMatrix();
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Apply matrix-free Jacobian
PetscErrorCode applyJacobian(Mat a_A, Vec a_U, Vec a_F)
{
    BL_PROFILE("warpx_petsc::applyJacobian()");

    SNES_impl *context;
    MatShellGetContext(a_A, &context);
    copyVec(context->m_U, a_U);
    context->applyOp(context->m_F, context->m_U);
    copyVec(a_F, context->m_F);

    PetscFunctionReturn(PETSC_SUCCESS);
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

    PETScSolver_impl *context;
    PCShellGetContext(a_pc, &context);

    copyVec( context->m_U, a_X );
    context->applyPC( context->m_F, context->m_U );
    copyVec( a_Y, context->m_F );

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Print SNES residuals
PetscErrorCode printSNESResidual(SNES a_snes, PetscInt a_n, PetscReal a_rnorm, void *a_ctxt)
{
    BL_PROFILE("printSNESResidual()");
    amrex::ignore_unused(a_ctxt);
    amrex::ignore_unused(a_snes);
    static amrex::Real norm0 = 0;
    if (a_n == 0) { norm0 = a_rnorm; }
    amrex::Print() << "Newton (PETSc SNES): iter = " << a_n << ", residual = " << a_rnorm
                   << ", " << a_rnorm / norm0 << " (rel.)\n";
    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Print KSP residuals
PetscErrorCode printKSPResidual(KSP a_ksp, PetscInt a_n, PetscReal a_rnorm, void *a_ctxt)
{
    BL_PROFILE("printKSPSResidual()");
    amrex::ignore_unused(a_ctxt);
    amrex::ignore_unused(a_ksp);
    static amrex::Real norm0 = 0;
    if (a_n == 0) { norm0 = a_rnorm; }
    amrex::Print() << "GMRES (PETSc KSP): iter = " << a_n << ", residual = " << a_rnorm
                   << ", " << a_rnorm / norm0 << " (rel.)\n";
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Apply Jacovian operator
void PETScSolver_impl::applyOp(VecType& a_F, const VecType& a_U) const
{
    BL_PROFILE("PETScSolver_impl::applyOp()");
    AMREX_ALWAYS_ASSERT(m_is_defined);
    m_linop->apply(a_F, a_U);
}

// Apply preconditioner
void PETScSolver_impl::applyPC( VecType& a_F, const VecType& a_U) const
{
    BL_PROFILE("PETScSolver_impl::applypC()");
    AMREX_ALWAYS_ASSERT(m_is_defined);
    a_F.zero();
    m_linop->precond(a_F, a_U);
}

//! Assemble preconditioner matrix
void PETScSolver_impl::assemblePCMatrix()
{
    BL_PROFILE("PETScSolver_impl::assemblePCMatrix()");

    AMREX_ALWAYS_ASSERT(m_is_defined);
    AMREX_ALWAYS_ASSERT(m_pc_type == PreconditionerType::pc_petsc);
    AMREX_ALWAYS_ASSERT(m_P != nullptr);

    PetscInt n = -1;
    PetscInt ncols_max = -1;
    amrex::Gpu::DeviceVector<int> r_indices_g; // global row indices
    amrex::Gpu::DeviceVector<int> n_nz_cols; // number of non-zero columns for each row
    amrex::Gpu::DeviceVector<int> c_indices_g; // non-zero column indices (row-major)
    amrex::Gpu::DeviceVector<amrex::Real> a_ij;

    m_linop->getPCMatrix( r_indices_g, n_nz_cols, c_indices_g, a_ij, n, ncols_max );

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
//    amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
//    {
//        MatSetValues( m_P->obj,
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

KSP_impl::KSP_impl(LinOpType& a_op)
{
    BL_PROFILE("KSP_impl::KSP_impl()");
    this->m_linop = &a_op;
    amrex::Print() << "KSP_impl: Initialized PETSc's KSP solver.\n";

    m_ksp = new KSPObj;
    this->m_A = new MatObj;
    this->m_P = new MatObj;
    this->m_x = new VecObj;
    this->m_b = new VecObj;
}

KSP_impl::~KSP_impl()
{
    BL_PROFILE("KSP_impl::~KSP_impl()");
    delete m_ksp;
    delete this->m_A;
    delete this->m_P;
    delete this->m_x;
    delete this->m_b;
}

void KSP_impl::createObjects(const VecType& a_vec)
{
    BL_PROFILE("KSP_impl::createObjects()");
    AMREX_ALWAYS_ASSERT(!isDefined());

    // define work vector
    this->m_U.Define(a_vec);
    m_F.Define(a_vec);
    // find local and global vector sizes
    this->m_ndofs_l = this->m_U.nDOF_local();
    this->m_ndofs_g = this->m_U.nDOF_global();

    // create vectors
    VecCreate(PETSC_COMM_WORLD, &this->m_x->obj);
#ifdef AMREX_USE_GPU
#ifdef AMREX_USE_CUDA
    VecSetType(this->m_x->obj, VECCUDA);
#elif defined AMREX_USE_HIP
    VecSetType(this->m_x->obj, VECHIP);
#else
    WARPX_ABORT_WITH_MESSAGE("KSP_impl::createObjects() - not yet implemented for non-CUDA/HIP architectures");
#endif
#else
    VecSetType(this->m_x->obj, VECSTANDARD);
#endif
    VecSetSizes(this->m_x->obj, this->m_ndofs_l, this->m_ndofs_g);
    VecSetFromOptions(this->m_x->obj);
    VecDuplicate(this->m_x->obj, &this->m_b->obj);

    // create matrix operator
    MatCreateShell( PETSC_COMM_WORLD,
                    this->m_ndofs_l,
                    this->m_ndofs_l,
                    this->m_ndofs_g,
                    this->m_ndofs_g,
                    this,
                    &this->m_A->obj );
    MatShellSetOperation( this->m_A->obj, MATOP_MULT,
                          (void (*)(void))applyMatOp );
    MatSetUp(this->m_A->obj);

    // create KSP and PC object
    KSPCreate( PETSC_COMM_WORLD, &m_ksp->obj );
    PC pc;
    KSPGetPC(m_ksp->obj, &pc);
    KSPSetPCSide(m_ksp->obj, PC_RIGHT);
    this->m_pc_type = this->m_linop->pcType();
    if (this->m_pc_type != PreconditionerType::pc_petsc) {
        // use native implementation (or no PC, if
        // native PC is turned off)
        PCSetType(pc, PCSHELL);
        PCShellSetApply(pc, applyNativePC);
        PCShellSetContext(pc, this);
        amrex::Print() << "KSP_impl: Using native preconditioner through PETSc's PCShell interface.\n";
        KSPSetOperators( m_ksp->obj, this->m_A->obj, this->m_A->obj );
    } else {
        // use PETSc options and implementation for PC
        PCSetFromOptions(pc);
        PCType pctype;
        PCGetType(pc, &pctype);
        amrex::Print() << "KSP_impl: Using PETSc preconditioner - " << pctype << ".\n";
        // set up the PC sparse matrix
        MatCreate( PETSC_COMM_WORLD, &this->m_P->obj );
        MatSetSizes( this->m_P->obj,
                     this->m_ndofs_l, this->m_ndofs_l,
                     PETSC_DETERMINE, PETSC_DETERMINE );
#if defined AMREX_USE_GPU
#if defined AMREX_USE_CUDA
        MatSetType( this->m_P->obj, MATAIJCUSPARSE);
#elif defined AMREX_USE_HIP
        MatSetType( this->m_P->obj, MATAIJHIPSPARSE);
#else
        WARPX_ABORT_WITH_MESSAGE("KSP_impl::createObjects() - not yet implemented for non-CUDA/HIP architectures");
#endif
#else
        MatSetType( this->m_P->obj, MATAIJ );
        MatMPIAIJSetPreallocation( this->m_P->obj, 1 /*a_ops->numPCMatBands()*/, NULL,
                                                   1 /*a_ops->numPCMatBands()-1*/, NULL);
#endif
        MatSetOption(this->m_P->obj, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
        MatSetOption(this->m_P->obj, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        MatSetUp(this->m_P->obj);
        MatSetFromOptions(this->m_P->obj);
        KSPSetOperators( m_ksp->obj, this->m_A->obj, this->m_P->obj );
    }
    KSPSetTolerances( m_ksp->obj, m_rtol, m_atol, PETSC_CURRENT, m_maxits );
    KSPSetNormType( m_ksp->obj, KSP_NORM_UNPRECONDITIONED );
    if (m_verbose > 1) {
        KSPMonitorSet( m_ksp->obj, printKSPResidual, NULL, NULL );
    }
    KSPSetFromOptions(m_ksp->obj);

    // it is now defined
    this->m_is_defined = true;

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
    copyVec(this->m_x->obj, a_Y);
    copyVec(this->m_b->obj, a_R);

    if (this->m_linop->pcType() == PreconditionerType::pc_petsc) {
        assemblePCMatrix();
    }

    KSPSolve(m_ksp->obj, this->m_b->obj, this->m_x->obj);
    copyVec(a_Y, this->m_x->obj);

    KSPGetIterationNumber( m_ksp->obj, &m_niters );
    KSPConvergedReason reason;
    KSPGetConvergedReason( m_ksp->obj, &reason );
    m_status = (int)reason;
    KSPGetResidualNorm( m_ksp->obj, &m_norm );

    const char* conv_reason;
    KSPGetConvergedReasonString(m_ksp->obj, &conv_reason);
    if (m_verbose > 0) {
        amrex::Print() << "GMRES (PETSc KSP): exited due to \""
                       << conv_reason << "\" "
                       << "(abs. norm=" << m_norm << ").\n";
    }
}

void KSP_impl::setVerbose(int a_v)
{
    BL_PROFILE("KSP_impl::setVerbose()");
    m_verbose = a_v;
    if (a_v > 0 && isDefined()) {
        KSPMonitorSet( m_ksp->obj,
                       (PetscErrorCode (*)(KSP, PetscInt, PetscReal, void *))KSPMonitorResidual,
                       NULL, NULL );
    }
}

SNES_impl::SNES_impl(const VecType& a_vec, TIType* a_op)
{
    BL_PROFILE("SNES_impl::SNES_impl()");
    amrex::Print() << "SNES_impl: Initialized PETSc's SNES solver.\n";

    const amrex::ParmParse pp_newton("newton");
    pp_newton.query("verbose",             m_verbose);
    pp_newton.query("absolute_tolerance",  m_atol);
    pp_newton.query("relative_tolerance",  m_rtol);
    pp_newton.query("max_iterations",      m_maxits);

    const amrex::ParmParse pp_gmres("gmres");
    pp_gmres.query("absolute_tolerance",  m_atol_l);
    pp_gmres.query("relative_tolerance",  m_rtol_l);
    pp_gmres.query("max_iterations",      m_maxits_l);

    const amrex::ParmParse pp_jac("jacobian");
    pp_jac.query("pc_type", this->m_pc_type);

    this->m_U.Define(a_vec);
    m_F.Define(a_vec);
    this->m_ndofs_l = this->m_U.nDOF_local();
    this->m_ndofs_g = this->m_U.nDOF_global();

    AMREX_ALWAYS_ASSERT(a_op != nullptr);
    m_op = a_op;

    this->m_linop = new JacobianFunctionMF<VecType,TIType>;
    this->m_linop->define(m_F, m_op, this->m_pc_type);

    m_snes = new SNESObj;
    this->m_A = new MatObj;
    this->m_P = new MatObj;
    this->m_x = new VecObj;
    this->m_b = new VecObj;

    VecCreate(PETSC_COMM_WORLD, &this->m_x->obj);
#if defined AMREX_USE_GPU
#if defined AMREX_USE_CUDA
    VecSetType(this->m_x->obj, VECCUDA);
#elif defined AMREX_USE_HIP
    VecSetType(this->m_x->obj, VECHIP);
#else
    WARPX_ABORT_WITH_MESSAGE("SNES_impl::SNES_impl() - not yet implemented for non-CUDA/HIP architectures");
#endif
#else
    VecSetType(this->m_x->obj, VECSTANDARD);
#endif
    VecSetSizes(this->m_x->obj, this->m_ndofs_l, this->m_ndofs_g);
    VecDuplicate(this->m_x->obj, &this->m_b->obj);

    SNESCreate(PETSC_COMM_WORLD, &m_snes->obj);
    SNESSetType( m_snes->obj, SNESNEWTONLS );
    SNESLineSearch linesearch;
    SNESGetLineSearch( m_snes->obj, &linesearch );
    SNESLineSearchSetType( linesearch, SNESLINESEARCHBASIC );
    SNESSetFunction(m_snes->obj, nullptr, RHSFunction, this);

    MatCreateShell( PETSC_COMM_WORLD,
                    this->m_ndofs_l,
                    this->m_ndofs_l,
                    this->m_ndofs_g,
                    this->m_ndofs_g,
                    this,
                    &this->m_A->obj );
    MatShellSetOperation( this->m_A->obj, MATOP_MULT,
                          (void (*)(void))applyJacobian );
    MatSetUp(this->m_A->obj);

    KSP ksp;
    SNESGetKSP(m_snes->obj, &ksp);
    PC pc;
    KSPGetPC(ksp, &pc);
    KSPSetPCSide(ksp, PC_RIGHT);
    if (this->m_pc_type != PreconditionerType::pc_petsc) {
        // use native implementation (or no PC, if
        // native PC is turned off)
        PCSetType(pc, PCSHELL);
        PCShellSetApply(pc, applyNativePC);
        PCShellSetContext(pc, this);
        amrex::Print() << "SNES_impl: Using native preconditioner through PETSc's PCShell interface.\n";
        SNESSetJacobian(m_snes->obj, this->m_A->obj, this->m_A->obj, JacobianFunction, this);
    } else {
        // use PETSc options and implementation for PC
        PCSetFromOptions(pc);
        PCType pctype;
        PCGetType(pc, &pctype);
        amrex::Print() << "SNES_impl: Using PETSc preconditioner - " << pctype << ".\n";
        // set up the PC sparse matrix
        MatCreate( PETSC_COMM_WORLD, &this->m_P->obj );
        MatSetSizes( this->m_P->obj,
                     this->m_ndofs_l, this->m_ndofs_l,
                     PETSC_DETERMINE, PETSC_DETERMINE );
#if defined AMREX_USE_GPU
#if defined AMREX_USE_CUDA
        MatSetType( this->m_P->obj, MATAIJCUSPARSE);
#elif defined AMREX_USE_HIP
        MatSetType( this->m_P->obj, MATAIJHIPSPARSE);
#else
        WARPX_ABORT_WITH_MESSAGE("SNES_impl::SNES_impl() - not yet implemented for non-CUDA/HIP architectures");
#endif
#else
        MatSetType( this->m_P->obj, MATAIJ );
        MatMPIAIJSetPreallocation( this->m_P->obj, 1 /*a_ops->numPCMatBands()*/, NULL,
                                                   1 /*a_ops->numPCMatBands()-1*/, NULL);
#endif
        MatSetOption(this->m_P->obj, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
        MatSetOption(this->m_P->obj, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        MatSetUp(this->m_P->obj);
        SNESSetJacobian(m_snes->obj, this->m_A->obj, this->m_P->obj, JacobianFunction, this);
    }

    setTolerances(m_rtol, m_atol, m_maxits, m_rtol_l, m_atol_l, m_maxits_l);
    setMaxIters(m_maxits, m_maxits_l);
    if (m_verbose) {
        SNESMonitorSet(m_snes->obj, printSNESResidual, NULL,NULL );
        SNESGetKSP(m_snes->obj, &ksp);
        KSPMonitorSet( ksp, printKSPResidual, NULL, NULL );
    }

    PetscOptionsSetValue(nullptr, "-ksp_converged_reason", nullptr);

    SNESSetFromOptions(m_snes->obj);
    this->m_is_defined = true;

    PetscBool is_specified = PETSC_FALSE;
    PetscOptionsHasName(nullptr, nullptr, "-snes_fd", &is_specified);
    m_fd_jac_comput = (is_specified == PETSC_TRUE);
}

SNES_impl::~SNES_impl()
{
    BL_PROFILE("SNES_impl::~SNES_impl()");
    delete m_snes;
    delete this->m_A;
    delete this->m_P;
    delete this->m_x;
    delete this->m_b;
    delete this->m_linop;
}

void SNES_impl::printParams () const
{
    amrex::Print()     << "SNES_impl verbose:             " << (m_verbose?"true":"false") << "\n";
    amrex::Print()     << "SNES_impl max iterations:      " << m_maxits << "\n";
    amrex::Print()     << "SNES_impl relative tolerance:  " << m_rtol << "\n";
    amrex::Print()     << "SNES_impl absolute tolerance:  " << m_atol << "\n";
    amrex::Print()     << "KSP (SNES_impl) max iterations:     " << m_maxits_l << "\n";
    amrex::Print()     << "KSP (SNES_impl) relative tolerance: " << m_rtol_l << "\n";
    amrex::Print()     << "KSP (SNES_impl) absolute tolerance: " << m_atol_l << "\n";
    amrex::Print()     << "Preconditioner type:      " << amrex::getEnumNameString(this->m_pc_type) << "\n";

    ((JacobianFunctionMF<VecType,TIType>*)this->m_linop)->printParams();
}

void SNES_impl::setTolerances( const amrex::Real a_rtol,
                               const amrex::Real a_atol,
                               const int  a_its,
                               const amrex::Real a_rtol_l,
                               const amrex::Real a_atol_l,
                               const int a_its_l )
{
    BL_PROFILE("SNES_impl::setTolerances()");
    m_atol = a_atol;
    m_rtol = a_rtol;
    m_atol_l = a_atol_l;
    m_rtol_l = a_rtol_l;
    if (a_its > 0) { m_maxits = a_its; }
    if (a_its_l > 0) { m_maxits_l = a_its_l; }

    if (isDefined()) {
        SNESSetTolerances( m_snes->obj,
                           m_rtol,
                           m_atol,
                           m_stol,
                           (a_its > 0 ? a_its : PETSC_CURRENT),
                           PETSC_CURRENT );
        KSP ksp;
        SNESGetKSP(m_snes->obj, &ksp);
        KSPSetTolerances( ksp,
                          m_rtol_l,
                          m_atol_l,
                          PETSC_CURRENT,
                          (a_its > 0 ? a_its : PETSC_CURRENT) );
    }
}

void SNES_impl::setMaxIters(const int a_its, const int a_its_l )
{
    BL_PROFILE("SNES_impl::setMaxIters()");
    m_maxits = a_its;
    m_maxits_l = a_its_l;
    if (isDefined()) {
        SNESSetTolerances( m_snes->obj,
                           PETSC_CURRENT,
                           PETSC_CURRENT,
                           PETSC_CURRENT,
                           a_its,
                           PETSC_CURRENT );
        KSP ksp;
        SNESGetKSP(m_snes->obj, &ksp);
        KSPSetTolerances( ksp,
                          PETSC_CURRENT,
                          PETSC_CURRENT,
                          PETSC_CURRENT,
                          a_its );
    }
}

bool SNES_impl::usePC() const
{
    AMREX_ALWAYS_ASSERT(isDefined());
    return ((JacobianFunctionMF<VecType,TIType>*)this->m_linop)->usePreconditioner();
}

void SNES_impl::solve(  VecType& a_U,
                        const VecType& a_B,
                        amrex::Real a_time,
                        amrex::Real a_dt,
                        int a_step ) const
{
    BL_PROFILE("SNES_impl::solve()");
    AMREX_ALWAYS_ASSERT(isDefined());
    amrex::ignore_unused(a_dt);

    m_time = a_time;
    m_iter = a_step;
    ((JacobianFunctionMF<VecType,TIType>*)this->m_linop)->curTimeStep(a_dt);

    copyVec(this->m_x->obj, a_U);
    copyVec(this->m_b->obj, a_B);
    SNESSolve(m_snes->obj, this->m_b->obj, this->m_x->obj);

    SNESGetIterationNumber(m_snes->obj, &m_niters);
    SNESGetLinearSolveIterations(m_snes->obj, &m_niters_l);

    SNESConvergedReason reason;
    SNESGetConvergedReason( m_snes->obj, &reason );
    m_status = (int) reason;
    SNESGetFunctionNorm(m_snes->obj, &m_norm);

    const char* conv_reason;
    SNESGetConvergedReasonString(m_snes->obj, &conv_reason);
    if (m_verbose) {
        amrex::Print() << "Newton (PETSc SNES): exited due to \""
                       << conv_reason << "\" "
                       << "(abs. norm = " << m_norm << ").\n";
    }

    m_total_iters += m_niters;
    m_total_linsol_iters += m_niters_l;
}

void SNES_impl::computeRHS(VecType& a_F, const VecType& a_U) const
{
    BL_PROFILE("SNES_impl::computeRHS()");
    AMREX_ALWAYS_ASSERT(isDefined());

    if (m_fd_jac_comput) {
        static bool first_call = true;
        m_op->ComputeRHS( a_F, a_U, m_time, m_iter, !first_call);
        first_call = false;
    } else {
        m_op->ComputeRHS( a_F, a_U, m_time, m_iter, false);
    }

    ((JacobianFunctionMF<VecType,TIType>*)this->m_linop)->setBaseSolution(a_U);
    ((JacobianFunctionMF<VecType,TIType>*)this->m_linop)->setBaseRHS(a_F);
}

void SNES_impl::setVerbose(bool a_v)
{
    BL_PROFILE("SNES_impl::setVerbose()");
    m_verbose = a_v;
    if (a_v && isDefined()) {
        SNESMonitorSet(m_snes->obj, printSNESResidual, NULL,NULL );
    }
}

}

#endif
