/* Copyright 2021-2022 The WarpX Community
 *
 * Authors: Axel Huebl, Remi Lehe
 * License: BSD-3-Clause-LBNL
 */

#include "Python/pyWarpX.H"

#include <Particles/MultiParticleContainer.H>

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>


void init_MultiParticleContainer (py::module& m)
{
    py::class_<MultiParticleContainer>(m, "MultiParticleContainer")
        .def("get_particle_container_from_name",
            &MultiParticleContainer::GetParticleContainerFromName,
            py::arg("name"),
            py::return_value_policy::reference_internal
        )

        .def("set_plasma_lens_strength",
             [](MultiParticleContainer& mpc, int i_lens, amrex::Real strength_E, amrex::Real strength_B) {
                 mpc.h_repeated_plasma_lens_strengths_E.at(i_lens) = strength_E;
                 mpc.h_repeated_plasma_lens_strengths_B.at(i_lens) = strength_B;
                 amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                       mpc.h_repeated_plasma_lens_strengths_E.begin(), mpc.h_repeated_plasma_lens_strengths_E.end(),
                                       mpc.d_repeated_plasma_lens_strengths_E.begin());
                 amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                       mpc.h_repeated_plasma_lens_strengths_B.begin(), mpc.h_repeated_plasma_lens_strengths_B.end(),
                                       mpc.d_repeated_plasma_lens_strengths_B.begin());
                 amrex::Gpu::synchronize();
             },
             py::arg("i_lens"), py::arg("strength_E"), py::arg("strength_B"),
             R"pbdoc(Set the strength of the `i_lens`-th lens
Parameters
----------
i_lens: int
  Index of the lens to be modified
strength_E, strength_B: floats
  The electric and magnetic focusing strength of the lens)pbdoc"
        )

        .def("get_charge_density",
            [](MultiParticleContainer& mpc, int lev, bool local) {
                return mpc.GetChargeDensity(lev, local);
            },
            py::arg("lev"), py::arg("local")
        )
    ;
}
