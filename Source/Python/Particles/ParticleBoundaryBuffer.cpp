/* Copyright 2021-2023 The WarpX Community
 *
 * Authors: Axel Huebl, Remi Lehe, Roelof Groenewald
 * License: BSD-3-Clause-LBNL
 */

#include "Python/pyWarpX.H"

#include <Particles/ParticleBoundaryBuffer.H>

namespace warpx {
    class BoundaryBufferParIter
        : public amrex::ParIterSoA<PIdx::nattribs, 0, amrex::PinnedArenaAllocator>
    {
    public:
        using amrex::ParIterSoA<PIdx::nattribs, 0, amrex::PinnedArenaAllocator>::ParIterSoA;

        BoundaryBufferParIter(ContainerType& pc, int level) :
            amrex::ParIterSoA<PIdx::nattribs, 0, amrex::PinnedArenaAllocator>(pc, level) {}
    };
}

void init_BoundaryBufferParIter (py::module& m)
{
    py::class_<
        warpx::BoundaryBufferParIter,
        amrex::ParIterSoA<PIdx::nattribs, 0, amrex::PinnedArenaAllocator>
    >(m, "BoundaryBufferParIter")
        .def(py::init<amrex::ParIterSoA<PIdx::nattribs, 0, amrex::PinnedArenaAllocator>::ContainerType&, int>(),
            py::arg("particle_container"), py::arg("level")
        )
    ;
}

void init_ParticleBoundaryBuffer (py::module& m)
{
    py::class_<ParticleBoundaryBuffer>(m, "ParticleBoundaryBuffer")
        .def(py::init<>())
        .def("clear_particles",
            [](ParticleBoundaryBuffer& pbb) { pbb.clearParticles(); }
        )
        .def("get_particle_container",
            [](ParticleBoundaryBuffer& pbb,
            const std::string species_name, int boundary) {
                return &pbb.getParticleBuffer(species_name, boundary);
            },
            py::arg("species_name"), py::arg("boundary"),
            py::return_value_policy::reference_internal
        )
        .def("get_num_particles_in_container",
            [](ParticleBoundaryBuffer& pbb,
            const std::string species_name, int boundary, bool local)
            {
                return pbb.getNumParticlesInContainer(species_name, boundary, local);
            },
            py::arg("species_name"), py::arg("boundary"), py::arg("local")=true
        )
    ;
}
