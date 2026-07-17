/* Copyright 2021-2025 Yinjian Zhao, Luca Fedeli, Axel Huebl
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_MATH_RANDOM_SEED_H_
#define ABLASTR_MATH_RANDOM_SEED_H_

#include <string>


namespace ablastr::math
{
    /** Set the global random seed for AMReX.
     *
     * With "default", the random seed is fixed.
     * With "random", the random seed is determined using std::random_device and
     * std::clock() and thus every simulation run produces different random numbers.
     *
     * If a strictly positive integer is provided, the random seed for each MPI rank
     * is (mpi_rank+1) * n, where mpi_rank starts from 0. n = 1 and
     * random_seed = "default" produce the default random seed. Note that when GPU
     * simulations are run, one should not expect to obtain the same random numbers,
     * even if a fixed random_seed is provided.
     *
     * @param[in] random_seed allowed values are "default", "random", and strictly positive integers
     */
    void
    set_random_seed (std::string const & random_seed);

}

#endif  // ABLASTR_MATH_RANDOM_SEED_H_
