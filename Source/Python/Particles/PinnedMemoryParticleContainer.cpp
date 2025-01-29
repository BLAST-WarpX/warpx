/* Copyright 2021-2023 The WarpX Community
 *
 * Authors: Axel Huebl, Remi Lehe, Roelof Groenewald
 * License: BSD-3-Clause-LBNL
 */

#include "Python/pyWarpX.H"

#include <Particles/PinnedMemoryParticleContainer.H>


void init_PinnedMemoryParticleContainer (py::module& m)
{
    py::class_<
        PinnedMemoryParticleContainer,
        amrex::ParticleContainerPureSoA<PIdx::nattribs, 0, amrex::PinnedArenaAllocator>
    > pmpc (m, "PinnedMemoryParticleContainer");
    pmpc
        .def_property_readonly("real_comp_names",
            [](PinnedMemoryParticleContainer& pc)
            {
                return pc.getParticleComps();
            }
        )
        .def_property_readonly("int_comp_names",
            [](PinnedMemoryParticleContainer& pc)
            {
                return pc.getParticleiComps();
            }
        );
}
