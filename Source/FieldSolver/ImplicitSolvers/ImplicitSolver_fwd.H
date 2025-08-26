#ifndef _IMPLICIT_SOLVER_FWD_H_
#define _IMPLICIT_SOLVER_FWD_H_

struct ImplicitOptions {
    int max_particle_iterations;
    double particle_tolerance;
    bool deposit_mass_matrices;
    bool linear_stage_of_jfnk;
    int nonlinear_iteration;
};

#endif
