#ifndef PRECONDITIONER_LIBRARY_H_
#define PRECONDITIONER_LIBRARY_H_

#include <AMReX_Enum.H>

/**
 * \brief Types for preconditioners for field solvers
 */
AMREX_ENUM(PreconditionerType,
    pc_curl_curl_mlmg,
    pc_jacobi,
    pc_petsc,
    none
);

#endif
