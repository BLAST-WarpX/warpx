/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PETSC_H_
#define WARPX_PETSC_H_

#include "LinearFunction.H"
#include "Preconditioner.H"

#include <AMReX_BLassert.H>
#include <AMReX_REAL.H>
#include <AMReX_Config.H>

#include  <cfloat>

#ifdef AMREX_USE_PETSC

// forward declarations
class WarpXSolverVec;
class ImplicitSolver;
// end of forward declarations

namespace warpx_petsc {

// forward declarations
class SNESObj;
class KSPObj;
class MatObj;
class VecObj;
// end of forward declarations

using TIType = ImplicitSolver;
using LinOpType = LinearFunction<WarpXSolverVec,ImplicitSolver>;
using VecType = WarpXSolverVec;

class PETScSolver_impl
{
    public:

        //! Constructor
        PETScSolver_impl () = default;

        //! Destructor
        virtual ~PETScSolver_impl () = default;

        // Prohibit Move and Copy operations
        PETScSolver_impl(const PETScSolver_impl&) = delete;
        PETScSolver_impl& operator=(const PETScSolver_impl&) = delete;
        PETScSolver_impl(PETScSolver_impl&&) noexcept = delete;
        PETScSolver_impl& operator=(PETScSolver_impl&&) noexcept = delete;

        //! Apply the action of the preconditioner
        void applyPC(VecType&, const VecType&) const;
        //! Apply the action of the Jacobian
        void applyOp(VecType& a_F, const VecType& a_U) const;

        //! Assemble the sparse matrix for preconditioner
        void assemblePCMatrix();

        //! Work vector
        mutable VecType m_U;
        //! Work vector
        mutable VecType m_F;

        //! Local number of DOFs
        long m_ndofs_l = 0;
        //! Global number of DOFs
        long m_ndofs_g = 0;

        //! Matrix (lin operator) object
        MatObj* m_A = nullptr;
        //! Matrix (PC) object
        MatObj* m_P = nullptr;
        //! solution vector
        VecObj* m_x = nullptr;
        //! right-hand-side vector
        VecObj* m_b = nullptr;

        //! Linear operator object
        LinOpType* m_linop = nullptr;

        //! is defined?
        bool m_is_defined = false;

        //! Preconditioner type
        PreconditionerType m_pc_type = PreconditionerType::pc_petsc;
};

class KSP_impl : public PETScSolver_impl
{
    public:

        //! Constructor
        KSP_impl (LinOpType&);

        //! Destructor
        ~KSP_impl () override;

        // Prohibit Move and Copy operations
        KSP_impl(const KSP_impl&) = delete;
        KSP_impl& operator=(const KSP_impl&) = delete;
        KSP_impl(KSP_impl&&) noexcept = delete;
        KSP_impl& operator=(KSP_impl&&) noexcept = delete;

        //! Create PETSc objects
        void createObjects (const VecType&);

        //! Set tolerances
        void setTolerances(amrex::Real, amrex::Real, int);

        //! Set max iters
        void setMaxIters(int);

        //! Solve
        void solve(VecType&, const VecType&);

        //! Set verbosity
        void setVerbose(int);

        //! Get number of iterations for recentmost solve
        inline int getNumIters() const { return m_niters; }

        //! Get status for recentmost solve
        inline int getStatus() const { return m_status; }

        //! Get relative norm for recentmost solve
        inline amrex::Real getResidualNorm() const { return m_norm; }

        //! Check if object is defined
        [[nodiscard]] bool isDefined () const
        {
            return (this->m_is_defined && (this->m_linop != nullptr));
        }

    private:

        //! absolute tolerance
        amrex::Real m_atol = 1e-12;
        //! relative tolerance
        amrex::Real m_rtol = 1e-8;
        //! maximum iterations
        int m_maxits= 100;
        //! verbosity
        int m_verbose = 1;

        //! number of iterations for recentmost solve
        int m_niters = -1;
        //! status for recentmost solve
        int m_status = -1;
        //! residual norm (relative) for recentmost solve
        amrex::Real m_norm = DBL_MAX;

        // Note: unable to use std::unique_ptr<> due to
        // incomplete-type compilation error.
        //! Solver object
        KSPObj* m_ksp = nullptr;
};

class SNES_impl : public PETScSolver_impl
{
    public:

        //! Constructor
        SNES_impl (const VecType&, TIType*);

        //! Destructor
        ~SNES_impl() override;

        //! Check if object is defined
        [[nodiscard]] bool isDefined () const
        {
            return (this->m_is_defined && (m_op != nullptr) && (this->m_linop != nullptr));
        }

        //! Set tolerances
        void setTolerances(amrex::Real, amrex::Real, int,
                           amrex::Real, amrex::Real, int);

        //! Set max iters
        void setMaxIters(int, int);

        //! Solve the system
        void solve(VecType&, const VecType&, amrex::Real, amrex::Real, int) const;

        //! Evaluate the RHS
        void computeRHS(VecType&, const VecType&) const;

        //! Returns if a preconditioner is being used
        [[nodiscard]] bool usePC() const;

        //! Set verbosity
        void setVerbose(bool);

        //! Get parameters
        AMREX_FORCE_INLINE
        void getSolverParams ( amrex::Real&  a_rtol,
                               amrex::Real&  a_atol,
                               int&          a_maxits ) const
        {
            a_rtol = m_rtol;
            a_atol = m_atol;
            a_maxits = m_maxits;
        }

        //! Print parameters
        void printParams () const;

    private:

        //! absolute tolerance
        amrex::Real m_atol = 0.;
        //! relative tolerance
        amrex::Real m_rtol = 1e-6;
        //! step tolerance
        amrex::Real m_stol = 1e-16;
        //! maximum iterations
        int m_maxits= 100;
        //! verbosity
        bool m_verbose = true;

        //! absolute tolerance
        amrex::Real m_atol_l = 0.;
        //! relative tolerance
        amrex::Real m_rtol_l = 1e-4;
        //! maximum iterations
        int m_maxits_l = 1000;

        //! number of iterations for recentmost solve
        mutable int m_niters = -1;
        //! number of linear solver iterations for recentmost solve
        mutable int m_niters_l = -1;
        //! status for recentmost solve
        mutable int m_status = -1;
        //! residual norm (relative) for recentmost solve
        mutable amrex::Real m_norm = DBL_MAX;

        //! Simulation time
        mutable amrex::Real m_time = 0.0;
        //! Simulation time iter
        mutable int m_iter = 0;

        //! Total nonlinear iterations for the diagnostic file
        mutable int m_total_iters = 0;
        //! Total linear iterations for the diagnostic file
        mutable int m_total_linsol_iters = 0;

        //! PETSc objects created?

        //! Implicit time integrator object
        TIType* m_op = nullptr;

        // Note: unable to use std::unique_ptr<> due to
        // incomplete-type compilation error.
        //! Solver object
        SNESObj* m_snes = nullptr;
};

}

#endif
#endif
