#ifndef IMPLICIT_OPTIONS_H_
#define IMPLICIT_OPTIONS_H_

#include <AMReX_REAL.H>

struct ImplicitOptions
{
    int max_particle_iterations = 21;
    amrex::ParticleReal particle_tolerance = static_cast<amrex::ParticleReal>(1.0e-10);
    bool print_unconverged_particle_details = false;
    bool use_mass_matrices_jacobian = false;
    bool use_mass_matrices_pc = false;
    bool linear_stage_of_jfnk = false;
    bool evolve_suborbit_particles_only = false;
    // Compose rho as rho_suborbit + rho_fp_non_suborbit (mass matrices + suborbits):
    // non-suborbit rho is cached once per nonlinear iteration and only suborbit
    // particles deposit rho during the linear stage of JFNK.
    bool use_rho_non_suborbit = false;
};

#endif
