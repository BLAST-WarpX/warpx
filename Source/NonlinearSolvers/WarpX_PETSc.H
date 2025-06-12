/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PETSC_H_
#define WARPX_PETSC_H_

#include "LinearFunction.H"

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
class KSPObj;
class MatObj;
class VecObj;
// end of forward declarations

using LinOpType = LinearFunction<WarpXSolverVec,ImplicitSolver>;
using VecType = WarpXSolverVec;

class KSP_impl
{
    public:

        //! Constructor
        KSP_impl (LinOpType&);

        //! Destructor
        ~KSP_impl ();

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
            return m_is_defined && (m_op != nullptr);
        }

        //! Apply matrix op on given vector
        void applyOp( VecType& a_F, const VecType& a_U);

        //! Apply PC op on given vector
        void applyPC( VecType& a_F, const VecType& a_U);

        /* the following are not private to let them be accessible from
         * non-member functions passed to PETSc */

        //! Work vector
        VecType m_U;
        //! Work vector
        VecType m_F;

    private:

        //! Number of MPI ranks
        int m_num_procs = -1;
        //! MPI rank of this process
        int m_myid = -1;

        //! Local number of DOFs
        long m_ndofs_l = 0;
        //! Global number of DOFs
        long m_ndofs_g = 0;

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

        //! Linear operator object
        LinOpType* m_op = nullptr;

        // Note: unable to use std::unique_ptr<> due to
        // incomplete-type compilation error.
        //! Solver object
        KSPObj* m_ksp = nullptr;
        //! Matrix (lin operator) object
        MatObj* m_A = nullptr;
        //! solution vector
        VecObj* m_x = nullptr;
        //! right-hand-side vector
        VecObj* m_b = nullptr;

        //!PETSc objects created?
        bool m_is_defined = false;
};

}

#endif
#endif
