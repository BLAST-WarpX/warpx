/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "NewtonSolverParams.H"
#include "NonlinearSolver.H"
#include "Preconditioner.H"

#include <ablastr/warn_manager/WarnManager.H>

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
    PetscFunctionBeginUser;
    BL_PROFILE("warpx_petsc::RHSFunction()");
    amrex::ignore_unused(a_solver);

    SNES_impl *context = (SNES_impl*) ctxt;
    copyVec(context->m_U, a_U);
    context->computeRHS(context->m_F, context->m_U);
    copyVec(a_F, context->m_F);
    VecAXPBY(a_F, 1.0, -1.0, a_U);

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Called at the start of each nonlinear iteration, before the Jacobian is computed
PetscErrorCode PreLinearSolveFunction( SNES a_solver, PetscInt a_iter )
{
    PetscFunctionBeginUser;
    BL_PROFILE("warpx_petsc::PreLinearSolveFunction()");
    amrex::ignore_unused(a_iter);

    void* ctxt = nullptr;
    SNESGetApplicationContext(a_solver, &ctxt);
    static_cast<SNES_impl*>(ctxt)->preLinearSolve();

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Compute Jacobian
PetscErrorCode JacobianFunction( SNES a_solver,
                                 Vec a_U,
                                 Mat a_A,
                                 Mat a_P,
                                 void* ctxt )
{
    PetscFunctionBeginUser;
    BL_PROFILE("warpx_petsc::JacobianFunction()");
    amrex::ignore_unused(a_A);
    amrex::ignore_unused(a_P);

    SNES_impl *context = (SNES_impl*) ctxt;
    const bool update_pc = context->updatePC();

    if (update_pc) {
        dynamic_cast<JacobianFunctionMF<VecType,TIType>*>(context->m_linop.get())->updatePreCondMat();

        KSP lin_solver;
        SNESGetKSP(a_solver, &lin_solver);
        PC pc;
        KSPGetPC(lin_solver, &pc);
        PCType pctype;
        PCGetType(pc, &pctype);

        if (strcmp(pctype,PCNONE) && strcmp(pctype,PCSHELL)) {
            copyVec(context->m_U, a_U);
            auto err = context->assemblePCMatrix(context->m_linop.get());
            AMREX_ALWAYS_ASSERT(err == PETSC_SUCCESS);
        }
        context->setPCInitialized();
    }

    // Rebuild or reuse the PETSc preconditioner in the linear solve that follows
    SNESSetLagPreconditioner(a_solver, update_pc ? 1 : -1);

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Apply matrix-free Jacobian
PetscErrorCode applyJacobian(Mat a_A, Vec a_U, Vec a_F)
{
    PetscFunctionBeginUser;
    BL_PROFILE("warpx_petsc::applyJacobian()");

    SNES_impl *context;
    MatShellGetContext(a_A, &context);
    copyVec(context->m_U, a_U);
    context->applyOp(context->m_F, context->m_U, context->m_linop.get());
    copyVec(a_F, context->m_F);

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Apply matrix-free linear operator
PetscErrorCode applyMatOp(Mat a_A, Vec a_U, Vec a_F)
{
    PetscFunctionBeginUser;
    BL_PROFILE("warpx_petsc::applyMatOp()");

    KSP_impl *context;
    MatShellGetContext(a_A,&context);

    copyVec( context->m_U, a_U );
    context->applyOp( context->m_F, context->m_U, context->m_linop );
    copyVec( a_F, context->m_F);

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Apply native preconditioner
PetscErrorCode applyNativePC( PC  a_pc, Vec a_X, Vec a_Y )
{
    PetscFunctionBeginUser;
    BL_PROFILE("warpx_petsc::applyNativePC()");

    PETScSolver_impl *context;
    PCShellGetContext(a_pc, &context);

    LinOpType* linop = nullptr;
    if (dynamic_cast<KSP_impl*>(context) != nullptr) {
        linop = dynamic_cast<KSP_impl*>(context)->m_linop;
    } else if (dynamic_cast<SNES_impl*>(context) != nullptr) {
        linop = dynamic_cast<SNES_impl*>(context)->m_linop.get();
    } else {
        amrex::Abort("Error in warpx_petsc::applyNativePC - unable to case context pointer");
    }
    copyVec( context->m_U, a_X );
    context->applyPC( context->m_F, context->m_U, linop );
    copyVec( a_Y, context->m_F );

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Print SNES residuals
PetscErrorCode printSNESResidual(SNES a_snes, PetscInt a_n, PetscReal a_rnorm, void *a_ctxt)
{
    PetscFunctionBeginUser;
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
    PetscFunctionBeginUser;
    BL_PROFILE("printKSPSResidual()");
    amrex::ignore_unused(a_ctxt);
    amrex::ignore_unused(a_ksp);
    static amrex::Real norm0 = 0;
    if (a_n == 0) { norm0 = a_rnorm; }
    amrex::Print() << "GMRES (PETSc KSP): iter = " << a_n << ", residual = " << a_rnorm
                   << ", " << a_rnorm / norm0 << " (rel.)\n";
    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Constructor
//! Create the PETSc solution and right-hand-side vectors of a solver
void createVectors( PETScSolver_impl& a_solver, const std::string& a_name )
{
    amrex::ignore_unused(a_name);
    VecCreate(PETSC_COMM_WORLD, &a_solver.m_x->obj);
#ifdef AMREX_USE_GPU
#ifdef AMREX_USE_CUDA
    VecSetType(a_solver.m_x->obj, VECCUDA);
#elif defined AMREX_USE_HIP
    VecSetType(a_solver.m_x->obj, VECHIP);
#else
    WARPX_ABORT_WITH_MESSAGE(a_name + " - not yet implemented for non-CUDA/HIP architectures");
#endif
#else
    VecSetType(a_solver.m_x->obj, VECSTANDARD);
#endif
    VecSetSizes(a_solver.m_x->obj, a_solver.m_ndofs_l, a_solver.m_ndofs_g);
    VecSetFromOptions(a_solver.m_x->obj);
    VecDuplicate(a_solver.m_x->obj, &a_solver.m_b->obj);
}

/** Set up the PC of a KSP: the native PC through PETSc's PCShell interface, or a
 *  PETSc PC with its sparse matrix. Returns the matrix from which the PC is built. */
Mat setupPC( KSP a_ksp, PETScSolver_impl& a_solver, const std::string& a_name )
{
    PC pc;
    KSPGetPC(a_ksp, &pc);
    KSPSetPCSide(a_ksp, PC_RIGHT);
    if (a_solver.m_pc_type != PreconditionerType::pc_petsc) {
        // use native implementation (or no PC, if
        // native PC is turned off)
        PCSetType(pc, PCSHELL);
        PCShellSetApply(pc, applyNativePC);
        PCShellSetContext(pc, &a_solver);
        amrex::Print() << a_name << ": Using native preconditioner through PETSc's PCShell interface.\n";
        return a_solver.m_A->obj;
    }

    // use PETSc options and implementation for PC
    PCSetFromOptions(pc);
    PCType pctype;
    PCGetType(pc, &pctype);
    amrex::Print() << a_name << ": Using PETSc preconditioner - " << pctype << ".\n";
    // set up the PC sparse matrix
    MatCreate( PETSC_COMM_WORLD, &a_solver.m_P->obj );
    MatSetSizes( a_solver.m_P->obj,
                 a_solver.m_ndofs_l, a_solver.m_ndofs_l,
                 PETSC_DETERMINE, PETSC_DETERMINE );
#if defined AMREX_USE_GPU
#if defined AMREX_USE_CUDA
    MatSetType( a_solver.m_P->obj, MATAIJCUSPARSE);
#elif defined AMREX_USE_HIP
    MatSetType( a_solver.m_P->obj, MATAIJHIPSPARSE);
#else
    WARPX_ABORT_WITH_MESSAGE(a_name + " - not yet implemented for non-CUDA/HIP architectures");
#endif
#else
    MatSetType( a_solver.m_P->obj, MATAIJ );
    MatMPIAIJSetPreallocation( a_solver.m_P->obj, 1 /*a_ops->numPCMatBands()*/, NULL,
                                                  1 /*a_ops->numPCMatBands()-1*/, NULL);
#endif
    MatSetOption(a_solver.m_P->obj, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
    MatSetOption(a_solver.m_P->obj, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(a_solver.m_P->obj);
    MatSetFromOptions(a_solver.m_P->obj);
    return a_solver.m_P->obj;
}

PETScSolver_impl::PETScSolver_impl()
{
    m_A = std::make_unique<MatObj>();
    m_P = std::make_unique<MatObj>();
    m_x = std::make_unique<VecObj>();
    m_b = std::make_unique<VecObj>();
}

//! Destructor
PETScSolver_impl::~PETScSolver_impl() { }

// Set PETSc options from WarpX input file
void PETScSolver_impl::setOptions()
{
    BL_PROFILE("PETScSolver_impl::setOptions()");

    PetscBool pc_specified;
    PetscOptionsHasName( NULL, NULL, "-pc_type", &pc_specified );

    if ((!pc_specified) && (m_pc_type == PreconditionerType::pc_petsc)) {

        auto parsestr = amrex::getEnumNameString(PreconditionerType::pc_petsc);
        const amrex::ParmParse pp_pc(parsestr.c_str());

        std::string pctype = "asm"; // default
        {
            std::string val = pctype;
            pp_pc.query("type", val);
            PetscOptionsSetValue( NULL, "-pc_type", val.c_str());
            pctype = val;
        }

        if (pctype == "asm") {

            {
                int val = 0;
                pp_pc.query("asm_overlap", val);
                std::string valstr = std::to_string(val);
                PetscOptionsSetValue( NULL, "-pc_asm_overlap", valstr.c_str());
            }

            std::string subpctype = "ilu"; // default
            {
                std::string val = subpctype;
                pp_pc.query("sub_type", val);
                PetscOptionsSetValue( NULL, "-sub_pc_type", val.c_str());
                subpctype = val;
            }

            if (subpctype == "ilu") {
                int val = 2;
                pp_pc.query("ilu_factor_levels", val);
                std::string valstr = std::to_string(val);
                PetscOptionsSetValue( NULL, "-sub_pc_factor_levels", valstr.c_str());
            }

        } else if (pctype == "hypre") {

            std::string hyprepctype = "euclid"; // default
            {
                std::string val = hyprepctype;
                pp_pc.query("hypre_type", val);
                PetscOptionsSetValue( NULL, "-pc_hypre_type", val.c_str());
                hyprepctype = val;
            }

            if (hyprepctype == "euclid") {
                int val = 2;
                pp_pc.query("euclid_factor_levels", val);
                std::string valstr = std::to_string(val);
                PetscOptionsSetValue( NULL, "-pc_hypre_euclid_level", valstr.c_str());
            }

        } else {

            // some sanity checks
#ifdef AMREX_USE_GPU
            if (pctype == "lu") {
                WARPX_ABORT_WITH_MESSAGE("PETScSolver_impl::setOptions() - PC type LU not available on GPUs");
            }
#endif
        }

    }

}

void PETScSolver_impl::setRestartLength(int a_restart_length) const
{
    PetscBool restart_specified;
    PetscOptionsHasName( NULL, NULL, "-ksp_gmres_restart", &restart_specified );
    if (!restart_specified) {
        const std::string valstr = std::to_string(a_restart_length);
        PetscOptionsSetValue( NULL, "-ksp_gmres_restart", valstr.c_str());
    }
}

// Apply Jacobian operator
void PETScSolver_impl::applyOp( VecType& a_F,
                                const VecType& a_U,
                                LinOpType* const a_linop ) const
{
    BL_PROFILE("PETScSolver_impl::applyOp()");
    AMREX_ALWAYS_ASSERT(m_is_defined);
    a_linop->apply(a_F, a_U);
}

// Apply preconditioner
void PETScSolver_impl::applyPC( VecType& a_F,
                                const VecType& a_U,
                                LinOpType* const a_linop ) const
{
    BL_PROFILE("PETScSolver_impl::applypC()");
    AMREX_ALWAYS_ASSERT(m_is_defined);
    a_F.zero();
    a_linop->precond(a_F, a_U);
}

//! Assemble preconditioner matrix
int PETScSolver_impl::assemblePCMatrix(LinOpType* const a_linop)
{
    PetscFunctionBeginUser;
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

    a_linop->getPCMatrix( r_indices_g, n_nz_cols, c_indices_g, a_ij, n, ncols_max );

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
            PetscCall(MatSetValues( m_P->obj,
                                    1,
                                    &h_r_indices_g[i],
                                    h_n_nz_cols[i],
                                    &h_c_indices_g[i*ncols_max],
                                    &h_a_ij[i*ncols_max],
                                    INSERT_VALUES ));
        }
    }

    PetscCall(MatAssemblyBegin(m_P->obj, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(m_P->obj, MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Constructor
KSP_impl::KSP_impl(LinOpType& a_op)
{
    BL_PROFILE("KSP_impl::KSP_impl()");
    m_linop = &a_op;
    amrex::Print() << "KSP_impl: Initialized PETSc's KSP solver.\n";

    m_ksp = std::make_unique<KSPObj>();
}

KSP_impl::~KSP_impl() { }

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
    createVectors(*this, "KSP_impl");

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
    this->m_pc_type = m_linop->pcType();
    // set PETSc options from WarpX inputs
    setOptions();
    KSPSetOperators( m_ksp->obj, this->m_A->obj, setupPC(m_ksp->obj, *this, "KSP_impl") );
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

    if (m_linop->pcType() == PreconditionerType::pc_petsc) {
        if (m_pc_needs_update) {
            auto err = assemblePCMatrix(m_linop);
            AMREX_ALWAYS_ASSERT(err == PETSC_SUCCESS);
            m_pc_needs_update = false;
            // Tell PETSc NOT to reuse preconditioner - forces new factorization
            KSPSetReusePreconditioner(m_ksp->obj, PETSC_FALSE);
        } else {
            // Tell PETSc to reuse the existing preconditioner factorization
            KSPSetReusePreconditioner(m_ksp->obj, PETSC_TRUE);
        }
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

SNES_impl::SNES_impl(const VecType& a_vec, TIType* a_op,
                     const NonlinearSolverParams& a_params,
                     const NewtonSolverParams& a_newton_params)
    : m_params(a_params), m_newton_params(a_newton_params)
{
    BL_PROFILE("SNES_impl::SNES_impl()");
    amrex::Print() << "SNES_impl: Initialized PETSc's SNES solver.\n";

    m_verbose = m_params.verbose;
    setRestartLength(m_newton_params.linsol_restart_length);
    this->m_pc_type = m_newton_params.pc_type;

    this->m_U.Define(a_vec);
    m_F.Define(a_vec);
    this->m_ndofs_l = this->m_U.nDOF_local();
    this->m_ndofs_g = this->m_U.nDOF_global();

    AMREX_ALWAYS_ASSERT(a_op != nullptr);
    m_op = a_op;

    m_linop = std::make_unique<JacobianFunctionMF<VecType,TIType>>();
    m_linop->define(m_F, m_op, this->m_pc_type);

    m_snes = std::make_unique<SNESObj>();

    createVectors(*this, "SNES_impl");

    SNESCreate(PETSC_COMM_WORLD, &m_snes->obj);
    SNESSetType( m_snes->obj, SNESNEWTONLS );
    SNESLineSearch linesearch;
    SNESGetLineSearch( m_snes->obj, &linesearch );
    SNESLineSearchSetType( linesearch, SNESLINESEARCHNONE );
    SNESSetFunction(m_snes->obj, nullptr, RHSFunction, this);
    SNESSetApplicationContext(m_snes->obj, this);
    // Keep the residual norms of each solve; the first one gives the relative norm
    SNESSetConvergenceHistory(m_snes->obj, nullptr, nullptr, PETSC_DECIDE, PETSC_TRUE);
    SNESSetUpdate(m_snes->obj, PreLinearSolveFunction);

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
    // set PETSc options from WarpX inputs
    setOptions();
    SNESSetJacobian(m_snes->obj, this->m_A->obj, setupPC(ksp, *this, "SNES_impl"),
                    JacobianFunction, this);

    setTolerances();
    if (m_verbose) {
        SNESMonitorSet(m_snes->obj, printSNESResidual, NULL,NULL );
    }
    if (m_newton_params.linsol_verbose_int > 1) {
        KSPMonitorSet( ksp, printKSPResidual, NULL, NULL );
    }

    PetscOptionsSetValue(nullptr, "-ksp_converged_reason", nullptr);

    SNESSetFromOptions(m_snes->obj);
    this->m_is_defined = true;

    PetscBool is_specified = PETSC_FALSE;
    PetscOptionsHasName(nullptr, nullptr, "-snes_fd", &is_specified);
    m_fd_jac_comput = (is_specified == PETSC_TRUE);
}

SNES_impl::~SNES_impl() { }

void SNES_impl::printParams () const
{
    dynamic_cast<JacobianFunctionMF<VecType,TIType>*>(m_linop.get())->printParams();
}

void SNES_impl::setTolerances()
{
    BL_PROFILE("SNES_impl::setTolerances()");
    SNESSetTolerances( m_snes->obj,
                       m_params.atol,
                       m_params.rtol,
                       m_stol,
                       m_params.maxits,
                       PETSC_CURRENT );
    KSP ksp;
    SNESGetKSP(m_snes->obj, &ksp);
    KSPSetTolerances( ksp,
                      m_newton_params.linsol_rtol,
                      m_newton_params.linsol_atol,
                      PETSC_CURRENT,
                      m_newton_params.linsol_maxits );
}

bool SNES_impl::usePC() const
{
    AMREX_ALWAYS_ASSERT(isDefined());
    return dynamic_cast<JacobianFunctionMF<VecType,TIType>*>(m_linop.get())->usePreconditioner();
}

void SNES_impl::solve (VecType& a_U,
                       const VecType& a_B,
                       amrex::Real a_time,
                       amrex::Real a_dt,
                       int a_step) const
{
    BL_PROFILE("SNES_impl::solve()");
    AMREX_ALWAYS_ASSERT(isDefined());

    m_time = a_time;
    m_step = a_step;
    m_iter = 0;
    m_rhs_first_call = true;
    dynamic_cast<JacobianFunctionMF<VecType,TIType>*>(m_linop.get())->curTime(a_time);
    dynamic_cast<JacobianFunctionMF<VecType,TIType>*>(m_linop.get())->curTimeStep(a_dt);

    copyVec(this->m_x->obj, a_U);
    copyVec(this->m_b->obj, a_B);
    SNESSolve(m_snes->obj, this->m_b->obj, this->m_x->obj);
    copyVec(a_U, this->m_x->obj);

    SNESGetIterationNumber(m_snes->obj, &m_niters);
    SNESGetLinearSolveIterations(m_snes->obj, &m_niters_l);

    KSP ksp;
    SNESGetKSP(m_snes->obj, &ksp);
    KSPGetResidualNorm(ksp, &m_norm_lin);

    SNESConvergedReason reason;
    SNESGetConvergedReason( m_snes->obj, &reason );
    m_status = (int)reason;
    // Reaching the maximum number of iterations is a failure only if convergence is required
    if (reason == SNES_DIVERGED_MAX_IT && !m_params.require_convergence) { m_status = 0; }
    SNESGetFunctionNorm(m_snes->obj, &m_norm);

    // Norm relative to the initial residual norm, as in the native Newton solver
    PetscReal* norm_history = nullptr;
    PetscInt num_norms = 0;
    SNESGetConvergenceHistory(m_snes->obj, &norm_history, nullptr, &num_norms);
    const amrex::Real norm0 = (num_norms > 0 && norm_history[0] > 0.) ? norm_history[0] : 1.0;
    m_norm_rel = m_norm/norm0;

    const char* conv_reason;
    SNESGetConvergedReasonString(m_snes->obj, &conv_reason);
    // see https://petsc.org/release/docs/manualpages/SNES/SNESConvergedReason/
    if (m_verbose) {
        amrex::Print() << "Newton (PETSc SNES): exited due to \""
                       << conv_reason << "\" "
                       << "(abs. norm = " << m_norm << ").\n";
    }

    m_total_iters += m_niters;
    m_total_linsol_iters += m_niters_l;
}

void SNES_impl::computeRHS(VecType& a_F, const VecType& a_U)
{
    BL_PROFILE("SNES_impl::computeRHS()");
    AMREX_ALWAYS_ASSERT(isDefined());

    // With -snes_fd, evaluations after the first one of a solve are part of
    // the finite-difference Jacobian
    const bool from_jacobian = m_fd_jac_comput && !m_rhs_first_call;
    m_rhs_first_call = false;

    if (!from_jacobian) {
        // Same PC update schedule as the native Newton solver
        m_update_pc = m_newton_params.UpdatePC(m_step, m_iter, m_pc_initialized);
    }

    m_op->ComputeRHS(a_F, a_U, m_time, m_iter, from_jacobian,
                     m_update_pc && !from_jacobian);
    if (!from_jacobian) { m_iter++; }
}

void SNES_impl::preLinearSolve()
{
    BL_PROFILE("SNES_impl::preLinearSolve()");
    AMREX_ALWAYS_ASSERT(isDefined());

    // m_U and m_F hold the current iterate and its RHS from the last residual evaluation
    dynamic_cast<JacobianFunctionMF<VecType,TIType>*>(m_linop.get())->setBaseSolution(this->m_U);
    dynamic_cast<JacobianFunctionMF<VecType,TIType>*>(m_linop.get())->setBaseRHS(m_F);
    m_op->PreLinearSolve(m_update_pc);
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
