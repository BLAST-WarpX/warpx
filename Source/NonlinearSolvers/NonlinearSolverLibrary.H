#ifndef NONLINEAR_SOLVER_LIBRARY_H_
#define NONLINEAR_SOLVER_LIBRARY_H_

#include "PicardSolver.H"  // IWYU pragma: export
#include "NewtonSolver.H"  // IWYU pragma: export

/**
  * \brief struct to select the nonlinear solver for implicit schemes
  */
enum NonlinearSolverType {
    Picard = 0,
    Newton = 1
};

#endif
