/* Copyright 2021-2025 Yinjian Zhao, Luca Fedeli, Axel Huebl
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "RandomSeed.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Random.H>

#include <random>
#include <string>
#include <stdexcept>


void
ablastr::math::set_random_seed (std::string const & random_seed)
{
    if ( random_seed != "default" ) {
        const unsigned long myproc_1 = amrex::ParallelDescriptor::MyProc() + 1;
        if ( random_seed == "random" ) {
            std::random_device rd;
            std::uniform_int_distribution<int> dist(2, INT_MAX);
            const unsigned long cpu_seed = myproc_1 * dist(rd);
            const unsigned long gpu_seed = myproc_1 * dist(rd);
            amrex::ResetRandomSeed(cpu_seed, gpu_seed);
        } else if ( std::stoi(random_seed) > 0 ) {
            const unsigned long nprocs = amrex::ParallelDescriptor::NProcs();
            const unsigned long seed_long = std::stoul(random_seed);
            const unsigned long cpu_seed = myproc_1 * seed_long;
            const unsigned long gpu_seed = (myproc_1 + nprocs) * seed_long;
            amrex::ResetRandomSeed(cpu_seed, gpu_seed);
        } else {
            throw std::runtime_error("random_seed must be \"default\", \"random\" or an integer > 0.");
        }
    }
}
