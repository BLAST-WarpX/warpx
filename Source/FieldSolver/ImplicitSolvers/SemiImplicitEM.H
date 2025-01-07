/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef SEMI_IMPLICIT_EM_H_
#define SEMI_IMPLICIT_EM_H_

#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ImplicitSolver.H"

/** @file
 *  Semi-implicit electromagnetic time solver class. The electric field and the
 *  particles are implicitly coupled in this algorithm, but the magnetic field
 *  is advanced in the standard explicit leap-frog manner (hence semi-implicit).
 *
 *  The time stencil is
 *  Eg^{n+1} = Eg^n + c^2*dt*( curlBg^{n+1/2} - mu0*Jg^{n+1/2} )
 *  Bg^{n+3/2} = Bg^{n+1/2} - dt*curlEg^{n+1}
 *  xp^{n+1} = xp^n + dt*up^{n+1/2}/(0.5(gammap^n + gammap^{n+1}))
 *  up^{n+1} = up^n + dt*qp/mp*(Ep^{n+1/2} + up^{n+1/2}/gammap^{n+1/2} x Bp^{n+1/2})
 *  where f^{n+1/2} = (f^{n+1} + f^n)/2.0, for all but Bg, which lives at half steps
 *
 *  This algorithm is approximately energy conserving. It is exactly energy conserving
 *  using a non-standard definition for the magnetic field energy. The advantage of
 *  this method over the exactly energy-conserving theta-implicit EM method is that
 *  light wave dispersion is captured much better. However, the CFL condition for light
 *  waves has to be satisifed for numerical stability (and for the modified definition
 *  of the magnetic field energy to be well-posed).
 *
 *  See G. Chen, L. Chacon, L. Yin, B.J. Albright, D.J. Stark, R.F. Bird,
 *  "A semi-implicit energy- and charge-conserving particle-in-cell algorithm for the
 *  relativistic Vlasov-Maxwell equations.", JCP 407 (2020).
 */

class SemiImplicitEM : public ImplicitSolver
{
public:

    SemiImplicitEM() = default;

    ~SemiImplicitEM() override = default;

    // Prohibit Move and Copy operations
    SemiImplicitEM(const SemiImplicitEM&) = delete;
    SemiImplicitEM& operator=(const SemiImplicitEM&) = delete;
    SemiImplicitEM(SemiImplicitEM&&) = delete;
    SemiImplicitEM& operator=(SemiImplicitEM&&) = delete;

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
     * \brief Solver vectors for E and Eold
     */
    WarpXSolverVec m_E, m_Eold;

};

#endif
