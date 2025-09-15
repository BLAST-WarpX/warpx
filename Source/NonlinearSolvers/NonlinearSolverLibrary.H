#ifndef NONLINEAR_SOLVER_LIBRARY_H_
#define NONLINEAR_SOLVER_LIBRARY_H_

#include "PicardSolver.H"  // IWYU pragma: export
#include "NewtonSolver.H"  // IWYU pragma: export

#include <AMReX_Enum.H>

/**
  * \brief struct to select the nonlinear solver for implicit schemes
  */
AMREX_ENUM (NonlinearSolverType,
    Picard,
    Newton
);

#endif
