/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Particles/ElementaryProcess/Ionization.H"

#include <AMReX.H>
#include <AMReX_Random.H>

#include <array>

namespace
{
    struct MockParticleData
    {
        std::array<int*, 2> m_runtime_idata;
    };
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    int sentinel = 37;
    int ionization_level = 2;
    int destination_value = 0;
    MockParticleData source{{&sentinel, &ionization_level}};
    MockParticleData destination{{&destination_value, &destination_value}};

    IonizationTransformFunc const transform{/*ionization_level_comp=*/1};
#ifdef AMREX_USE_GPU
    amrex::RandomEngine const engine{nullptr};
#else
    amrex::RandomEngine const engine{};
#endif
    transform(destination, source, 0, 0, engine);

    bool const pass = sentinel == 37 && ionization_level == 3;
    amrex::Finalize();
    return pass ? 0 : 1;
}
