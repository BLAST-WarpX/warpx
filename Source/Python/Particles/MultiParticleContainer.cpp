/* Copyright 2021-2022 The WarpX Community
 *
 * Authors: Axel Huebl, Remi Lehe
 * License: BSD-3-Clause-LBNL
 */

#include "Python/pyWarpX.H"

#include <Particles/MultiParticleContainer.H>
#include <Utils/WarpXAlgorithmSelection.H>

#include <AMReX_Enum.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <string>


void init_MultiParticleContainer (py::module& m)
{
    py::class_<MultiParticleContainer>(m, "MultiParticleContainer")
        .def("get",
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

        .def("push_p",
            [](MultiParticleContainer& mpc, int lev, amrex::Real dt,
               amrex::MultiFab const& Ex, amrex::MultiFab const& Ey, amrex::MultiFab const& Ez,
               amrex::MultiFab const& Bx, amrex::MultiFab const& By, amrex::MultiFab const& Bz,
               std::string const& momentum_push_type)
            {
                mpc.PushP(lev, dt, Ex, Ey, Ez, Bx, By, Bz,
                          amrex::getEnumCaseInsensitive<MomentumPushType>(momentum_push_type));
            },
            py::arg("lev"), py::arg("dt"),
            py::arg("Ex"), py::arg("Ey"), py::arg("Ez"),
            py::arg("Bx"), py::arg("By"), py::arg("Bz"),
            py::arg("momentum_push_type") = "Full",
            R"pbdoc(Push the momentum of the particles of all species, leaving their positions unchanged

The fields are gathered from the given MultiFabs, which must have their guard
cells filled, with the field gathering settings of the simulation.

Parameters
----------
lev: int
  Mesh refinement level of the particles to push
dt: float
  Time step over which to push the momentum
Ex, Ey, Ez: MultiFab
  Components of the electric field, with the staggering of ``Efield_fp``
Bx, By, Bz: MultiFab
  Components of the magnetic field, with the staggering of ``Bfield_fp``
momentum_push_type: str, optional
  ``"Full"`` (default) for a full step, ``"FirstHalf"`` or ``"SecondHalf"`` for the
  split push used by the collision algorithms)pbdoc"
        )
    ;
}
