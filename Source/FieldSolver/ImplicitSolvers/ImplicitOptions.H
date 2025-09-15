#ifndef IMPLICIT_OPTIONS_H_
#define IMPLICIT_OPTIONS_H_

#include <AMReX_Particles.H>
using namespace amrex::literals;

struct ImplicitOptions {
    int max_particle_iterations = 21;
    amrex::ParticleReal particle_tolerance = 1.0e-10_prt;
    bool print_unconverged_particle_details = false;
    bool deposit_mass_matrices = false;
    bool linear_stage_of_jfnk = false;
    int nonlinear_iteration = 0;
};

#endif
