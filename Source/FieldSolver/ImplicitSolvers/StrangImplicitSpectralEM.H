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
 *    Advance (Eg^n, Bg^n) -> (Eg^{n+1/2}, Bg^{n+1/2}) source free // E transverse
 *    Iterate:
 *      Eg^{n+1} = Eg^n + c^2*dt*( - mu0*Jg^{n+1/2} ) // E longitudinal
 *      xp^{n+1} = xp^n + dt*up^{n+1/2}/(0.5*(gammap^n + gammap^{n+1}))
 *      up^{n+1} = up^n + dt*qp/mp*(Ep^{n+1/2} + up^{n+1/2}/gammap^{n+1/2} x Bp^{n+1/2})
 *    Advance (Eg^n+1/2, Bg^n+1/2) -> (Eg^{n+1}, Bg^{n+1}) source free // E transverse
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

    void Define ( WarpX*  a_WarpX ) override;

    void PrintParameters () const override;

    void OneStep ( amrex::Real  start_time,
                   amrex::Real  a_dt,
                   int          a_step ) override;

    void ComputeRHS ( WarpXSolverVec&  a_RHS,
                const WarpXSolverVec&  a_E,
                      amrex::Real      half_time,
                      int              a_nl_iter,
                      bool             a_from_jacobian ) override;

private:

    /**
     * \brief Solver vectors to be used in the nonlinear solver to solve for the
     * electric field E. The main logic for determining which variables should be
     * WarpXSolverVec type is that it must have the same size and have the same
     * centering of the data as the variable being solved for, which is E here.
     * For example, if using a Yee grid then a container for curlB could be a
     * WarpXSovlerVec, but magnetic field B should not be.
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
     * \brief Update the E and B fields owned by WarpX
     */
    void UpdateWarpXFields ( WarpXSolverVec const&  a_E,
                             amrex::Real            half_time );

    /**
     * \brief Nonlinear solver is for the time-centered values of E. After
     * the solver, need to use m_E and m_Eold to compute E^{n+1}
     */
    void FinishFieldUpdate ( amrex::Real  a_new_time );

};

#endif
