/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef PETSC_SNES_WRAPPER_H_
#define PETSC_SNES_WRAPPER_H_

#include "NonlinearSolver.H"
#include "WarpX_PETSc.H"

#include <AMReX_REAL.H>
#include <AMReX_Config.H>
#include <memory>

#ifdef AMREX_USE_PETSC

/**
 * \brief Wrapper for PETSc's SNES solver
 *
 *    This class is a wrapper for PETSc's SNES nonlinear solver that inheritis from
 *    WarpX's NonlinearSolver base class. See the documentation of PETSc's SNES
 *    solver for more details. This class is templated on a vector class Vec,
 *    and an operator class Ops.
 *
 *    PETScSNES is just a wrapper; the actual implementation is in SNES_impl in
 *    Warpx_PETSc.H/cpp. This is to avoid including PETSc's header files here
 *    and consequent conflicts between PETSc and WarpX variables.
 */

template <class Vec, class Ops>
class PETScSNES : public NonlinearSolver<Vec,Ops>
{
    public:

        using RT = amrex::Real;

        PETScSNES () = default;
        ~PETScSNES () override = default;

        // Prohibit Move and Copy operations
        PETScSNES(const PETScSNES&) = delete;
        PETScSNES& operator=(const PETScSNES&) = delete;
        PETScSNES(PETScSNES&&) noexcept = delete;
        PETScSNES& operator=(PETScSNES&&) noexcept = delete;

        AMREX_FORCE_INLINE
        void Define ( const Vec& a_U, Ops* a_op ) override
        {
            m_solver = std::make_unique<warpx_petsc::SNES_impl>(a_U, a_op);
            this->m_is_defined = m_solver->isDefined();
            this->m_usePC = m_solver->usePC();
        }

        AMREX_FORCE_INLINE
        void Solve ( Vec& a_U,
                     const Vec& a_b,
                     RT a_time,
                     RT a_dt,
                     int a_step)  const override
        {
            m_solver->solve(a_U, a_b, a_time, a_dt, a_step);
        }

        AMREX_FORCE_INLINE
        void PrintParams () const override
        {
            m_solver->printParams();
        }

        AMREX_FORCE_INLINE
        void GetSolverParams (RT& a_rtol, RT& a_atol, int& a_maxits) override
        {
            m_solver->getSolverParams(a_rtol, a_atol, a_maxits);
        }

        AMREX_FORCE_INLINE
        void Verbose (bool a_verbose) override
        {
            m_solver->setVerbose(a_verbose);
        }

    private:

        std::unique_ptr<warpx_petsc::SNES_impl> m_solver = nullptr;
};

#endif
#endif
