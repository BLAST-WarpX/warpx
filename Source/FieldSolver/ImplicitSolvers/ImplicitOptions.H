#ifndef IMPLICIT_OPTIONS_H_
#define IMPLICIT_OPTIONS_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>

struct ImplicitOptions
{
    int max_particle_iterations = 21;
    amrex::ParticleReal particle_tolerance = static_cast<amrex::ParticleReal>(1.0e-10);
    bool print_unconverged_particle_details = false;
    bool use_mass_matrices_jacobian = false;
    bool use_mass_matrices_pc = false;
    bool linear_stage_of_jfnk = false;
    bool evolve_suborbit_particles_only = false;
    amrex::IntVect ncomp_xx = amrex::IntVect{0};
    amrex::IntVect ncomp_yy = amrex::IntVect{0};
    amrex::IntVect ncomp_zz = amrex::IntVect{0};
    amrex::IntVect ncomp_pc_xx = amrex::IntVect{0};
    amrex::IntVect ncomp_pc_yy = amrex::IntVect{0};
    amrex::IntVect ncomp_pc_zz = amrex::IntVect{0};
};

#endif
