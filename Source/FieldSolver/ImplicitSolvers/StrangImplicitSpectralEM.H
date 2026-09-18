/* Copyright 2024 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef STRANG_IMPLICIT_SPECTRALEM_H_
#define STRANG_IMPLICIT_SPECTRALEM_H_

#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ImplicitSolver.H"

/** @file
 *  Implicit spectral electromagnetic time solver class. This is a fully implicit
 *  algorithm where both the fields and particles are treated implicitly.
 *
 *  The time stencil is
 *    Advance (E^n, B^n) -> (E^{n+1/2}, B^{n+1/2}) source free // E transverse
 *    Iterate:
 *      E^{n+1}  = E^n  + c^2*dt*( - mu0*J^{n+1/2} ) // E longitudinal
 *      xp^{n+1} = xp^n + dt*up^{n+1/2}/(0.5*(gammap^n + gammap^{n+1}))
 *      up^{n+1} = up^n + dt*qp/mp*(Ep^{n+1/2} + up^{n+1/2}/gammap^{n+1/2} x Bp^{n+1/2})
 *    Advance (E^{n+1/2}, B^{n+1/2}) -> (E^{n+1}, B^{n+1}) source free // E transverse
 *
 *  The algorithm is exactly energy conserving only with a single box, periodic fft (psatd.periodic_single_box_fft = 1).
 *  With multiple boxes, energy is not conserved since the ffts in each box assumes periodic in the box which
 *  is not consistent with the current.
 *  The algorithm is numerially stable for any time step.
 *  I.e., the CFL condition for light waves does not
 *  have to be satisifed and the time step is not limited by the plasma period. However, how
 *  efficiently the algorithm can use large time steps depends strongly on the nonlinear solver.
 *  Furthermore, the time step should always be such that particles do not travel outside the
 *  ghost region of the box they live in, which is an MPI-related limitation. The time step
 *  is always limited by the need to resolve the appropriate physics.
 *
 */

class StrangImplicitSpectralEM : public ImplicitSolver
{
public:

    StrangImplicitSpectralEM() = default;

    ~StrangImplicitSpectralEM() override = default;

    // Prohibit Move and Copy operations
    StrangImplicitSpectralEM(const StrangImplicitSpectralEM&) = delete;
    StrangImplicitSpectralEM& operator=(const StrangImplicitSpectralEM&) = delete;
    StrangImplicitSpectralEM(StrangImplicitSpectralEM&&) = delete;
    StrangImplicitSpectralEM& operator=(StrangImplicitSpectralEM&&) = delete;

    void Define (WarpX* a_WarpX, bool a_from_restart) override;

    void PrintParameters () const override;

    int OneStep (amrex::Real  start_time,
                 amrex::Real  a_dt,
                 int          a_step,
                 bool verbose_step) override;

    void ComputeRHS ( WarpXSolverVec&  a_RHS,
                const WarpXSolverVec&  a_E,
                      amrex::Real      start_time,
                      int              a_nl_iter,
                      bool             a_from_jacobian ) override;

    // This parameter is used for the time-step fraction in the PC for implicit
    // treatment of light waves in the curl-curl MLMG solver.
    // This function should return zero if light waves are not treated implicitly
    [[nodiscard]] amrex::Real  GetThetaForPC () const override { return 0.; }

private:

    /**
     * \brief Solver vectors to be used in the nonlinear solver to solve for the
     * electric field E. The main logic for determining which variables should be
     * WarpXSolverVec type is that it must have the same size and have the same
     * centering of the data as the variable being solved for, which is E here.
     * For example, if using a Yee grid then a container for curlB could be a
     * WarpXSovlerVec, but magnetic field B should not be.
     *
     * m_E holds the midpoint iterate for the implicit substep. After the solve,
     * it is retained as the initial guess for the next nonlinear solve.
     * m_Eold holds the field after the first source-free half step, which is
     * the reference state for the implicit solve. The separate E_old MultiFab
     * holds the same field for field finishing, diagnostics, and checkpointing.
     * Efield_fp reaches the full-step field E^{n+1} only after the second
     * source-free half step; E_old is not the full-step field E^n.
     */
    WarpXSolverVec m_E, m_Eold;

    /**
     * \brief B is a derived variable from E. Need to save Bold to update B during
     * the iterative nonlinear solve for E. Bold is owned here, but only used by WarpX.
     * It is not used directly by the nonlinear solver, nor is it the same size as the
     * solver vector (size E), and so it should not be WarpXSolverVec type.
     */
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3 > > m_Bold;

    /**
     * \brief Copy the implicit midpoint E into the WarpX-owned electric field and apply BCs
     */
    void UpdateWarpXFields ( WarpXSolverVec const&  a_E,
                             amrex::Real            half_time );

    /**
     * \brief Finish the implicit electric-field update using Efield_fp and E_old.
     * Preserve m_E as the initial guess for the next nonlinear solve. The second
     * source-free half step must still be applied to obtain the full-step fields.
     */
    void FinishFieldUpdate (amrex::Real end_time);

};

#endif
