/* Copyright 2019 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/* -------------------------------------------------------------------------
! program to calculate the first zeroes (root abscissas) of the first
! kind bessel function of integer order n using the subroutine rootj.
! --------------------------------------------------------------------------
! sample run:
!
! (calculate the first 10 zeroes of 1st kind bessel function of order 2).
!
! zeroes of bessel function of order: 2
!
! number of calculated zeroes: 10
!
! table of root abcissas (5 items per line)
!    5.135622    8.417244   11.619841   14.795952   17.959819
    21.116997   24.270112   27.420574   30.569204   33.716520
!
! table of error codes (5 items per line)
!   0   0   0   0   0
!   0   0   0   0   0
!
! --------------------------------------------------------------------------
! reference: from numath library by tuan dang trong in fortran 77
!            [bibli 18].
!
!                               c++ release 1.0 by j-p moreau, paris
!                                          (www.jpmoreau.fr)
! ------------------------------------------------------------------------ */

#ifndef WARPX_BESSEL_ROOTS_H_
#define WARPX_BESSEL_ROOTS_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

/*----------------------------------------------------------------------
 *    calculate the first nk zeroes of bessel function j(n, x)
 *    including the trivial root (when n > 0)
 *
 *    inputs:
 *      n    order of function j (integer >= 0)                  i*4
 *      nk   number of first zeroes  (integer > 0)               i*4
 *    outputs:
 *      roots(nk)  table of first zeroes (abcissas)              r*8
 *      ier(nk)    table of error codes (must be zeroes)         i*4
 *
 *    reference :
 *    abramowitz m. & stegun irene a.
 *    handbook of mathematical functions
 */
void GetBesselRoots(int n, int nk, amrex::Vector<amrex::Real>& roots, amrex::Vector<int> &ier);

#endif // WARPX_BESSEL_ROOTS_H_
