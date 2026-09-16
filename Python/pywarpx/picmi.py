# Copyright 2018-2022 Andrew Myers, David Grote, Ligia Diana Amorim
# Maxence Thevenet, Remi Lehe, Revathi Jambunathan, Lorenzo Giacomel
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Classes following the PICMI standard"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self

import numpy as np
import periodictable
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

import picmistandard
import pywarpx
import pywarpx.callbacks
from picmistandard import Expression

codename = "warpx"
picmistandard.register_codename(codename)

# dictionary to map field boundary conditions from picmistandard to WarpX
BC_map = {
    "open": "pml",
    "dirichlet": "pec",
    "periodic": "periodic",
    "damped": "damped",
    "absorbing_silver_mueller": "absorbing_silver_mueller",
    "neumann": "neumann",
    "none": "none",
    None: "none",
}


class constants:
    # --- Put the constants in their own namespace
    # --- Values from WarpXConst.H
    c = 299792458.0
    ep0 = 8.8541878188e-12
    mu0 = 1.2566370612685e-06
    q_e = 1.602176634e-19
    m_e = 9.1093837139e-31
    m_p = 1.67262192595e-27
    hbar = 1.0545718176461565e-34
    kb = 1.380649e-23


picmistandard.register_constants(constants)


def _set_refined_region_inputs(refined_regions):
    if refined_regions:
        assert len(refined_regions) == 1, Exception(
            "WarpX only supports one refined region."
        )
        assert refined_regions[0][0] == 1, Exception(
            "The one refined region can only be level 1"
        )
        pywarpx.amr.max_level = 1
        pywarpx.warpx.fine_tag_lo = refined_regions[0][1]
        pywarpx.warpx.fine_tag_hi = refined_regions[0][2]
        if len(refined_regions[0]) == 4:
            pywarpx.amr.ref_ratio_vect = refined_regions[0][3]
    else:
        pywarpx.amr.max_level = 0


def _potential_not_in_geometry(name, geometry):
    """The potential on a boundary that the geometry does not have: always None.

    Assigning None is accepted, so that scripts can reset all potentials independently of the
    geometry.
    """

    def getter(self):
        return None

    def setter(self, value):
        if value is not None:
            raise AttributeError(f"{name} is not defined in {geometry} geometry")

    return property(
        getter, setter, doc=f"Not defined in {geometry} geometry (always None)"
    )


def warpx_options(picmi_base):
    """Expose the extra fields that a WarpX PICMI subclass adds as ``warpx_<name>`` options.

    A WarpX subclass extends a PICMI standard class with code-specific inputs. Rather than
    spelling out ``Field(alias="warpx_<name>")`` on every one of them, set this as the
    class' ``alias_generator``: any field the subclass adds becomes a ``warpx_<name>``
    keyword for the user, while the inherited PICMI standard fields keep their plain names.

    Usage::

        class Species(picmistandard.PICMI_Species):
            model_config = ConfigDict(
                alias_generator=warpx_options(picmistandard.PICMI_Species)
            )

            do_not_push: bool | None = None  # user passes warpx_do_not_push=...

    A field whose WarpX option name is not simply ``warpx_<field name>`` (for example
    ``warpx_potential_lo_x`` maps to the ``potential_xmin`` field) still declares an
    explicit ``Field(alias=...)``, which takes precedence over this generator.
    """
    standard_options = set(picmi_base.model_fields)
    return lambda name: name if name in standard_options else f"warpx_{name}"


class Species(picmistandard.PICMI_Species):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_Species)
    )

    # WarpX accepts code-specific string expressions (e.g. "q_e", "2*m_e") for the charge
    # and mass, in addition to plain floats, so widen the standard's float-only typing.
    charge: float | str | None = Field(
        default=None,
        description="Particle charge [C], or a WarpX expression such as 'q_e'. If not specified, determined from the particle type.",
    )
    mass: float | str | None = Field(
        default=None,
        description="Particle mass [kg], or a WarpX expression such as 'm_e'. If not specified, determined from the particle type.",
    )

    # --- WarpX-specific extension inputs (exposed to users as ``warpx_<name>``).
    boost_adjust_transverse_positions: bool | None = Field(
        default=None,
        description="Whether to adjust transverse positions when apply the boost to the simulation frame",
    )

    # For the relativistic electrostatic solver
    self_fields_required_precision: float | None = Field(
        default=None,
        description="Relative precision on the electrostatic solver (when using the relativistic solver)",
    )
    self_fields_absolute_tolerance: float | None = Field(
        default=None,
        description="Absolute precision on the electrostatic solver (when using the relativistic solver)",
    )
    self_fields_max_iters: int | None = Field(
        default=None,
        description="Maximum number of iterations for the electrostatic solver for the species",
    )
    self_fields_verbosity: int | None = Field(
        default=None, description="Level of verbosity for the electrostatic solver"
    )
    save_previous_position: bool | None = Field(
        default=None, description="Whether to save the old particle positions"
    )
    do_not_deposit: bool | None = Field(
        default=None,
        description="Whether or not to deposit the charge and current density for for this species",
    )
    do_not_push: bool | None = Field(
        default=None, description="Whether or not to push this species"
    )
    do_not_gather: bool | None = Field(
        default=None,
        description="Whether or not to gather the fields from grids for this species",
    )
    radial_numpercell_power: float | None = Field(
        default=None,
        description="With cylindrical geometry, specifies the radial power of the number of particles per cell",
    )
    random_theta: bool | None = Field(
        default=None,
        description="Whether or not to add random angle to the particles in theta when in RZ mode.",
    )

    # For particle reflection
    reflection_model_xlo: float | str | None = Field(
        default=None,
        description='Expression (in terms of the velocity "v") specifying the probability that the particle will reflect on the lower x boundary',
    )
    reflection_model_xhi: float | str | None = Field(
        default=None,
        description='Expression (in terms of the velocity "v") specifying the probability that the particle will reflect on the upper x boundary',
    )
    reflection_model_ylo: float | str | None = Field(
        default=None,
        description='Expression (in terms of the velocity "v") specifying the probability that the particle will reflect on the lower y boundary',
    )
    reflection_model_yhi: float | str | None = Field(
        default=None,
        description='Expression (in terms of the velocity "v") specifying the probability that the particle will reflect on the upper y boundary',
    )
    reflection_model_zlo: float | str | None = Field(
        default=None,
        description='Expression (in terms of the velocity "v") specifying the probability that the particle will reflect on the lower z boundary',
    )
    reflection_model_zhi: float | str | None = Field(
        default=None,
        description='Expression (in terms of the velocity "v") specifying the probability that the particle will reflect on the upper z boundary',
    )

    # For the scraper buffer
    save_particles_at_xlo: bool | None = Field(
        default=None,
        description="Whether to save particles lost at the lower x boundary",
    )
    save_particles_at_xhi: bool | None = Field(
        default=None,
        description="Whether to save particles lost at the upper x boundary",
    )
    save_particles_at_ylo: bool | None = Field(
        default=None,
        description="Whether to save particles lost at the lower y boundary",
    )
    save_particles_at_yhi: bool | None = Field(
        default=None,
        description="Whether to save particles lost at the upper y boundary",
    )
    save_particles_at_zlo: bool | None = Field(
        default=None,
        description="Whether to save particles lost at the lower z boundary",
    )
    save_particles_at_zhi: bool | None = Field(
        default=None,
        description="Whether to save particles lost at the upper z boundary",
    )
    save_particles_at_eb: bool | None = Field(
        default=None,
        description="Whether to save particles lost at the embedded boundary",
    )

    # Resampling settings
    do_resampling: bool | None = Field(
        default=None, description="Whether particles will be resampled"
    )
    resampling_algorithm: str | None = Field(
        default=None, description="Resampling algorithm to use."
    )
    resampling_min_ppc: int | None = Field(
        default=None,
        description="Cells with fewer particles than this number will be skipped during resampling.",
    )
    resampling_trigger_intervals: int | str | None = Field(
        default=None, description="Timesteps at which to resample"
    )
    # option name (warpx_resampling_trigger_max_avg_ppc) differs from the field name:
    resampling_triggering_max_avg_ppc: float | None = Field(
        default=None,
        alias="warpx_resampling_trigger_max_avg_ppc",
        description="Resampling will be done when the average number of particles per cell exceeds this number",
    )
    resampling_algorithm_target_ratio: float | None = Field(
        default=None,
        description="Roughly corresponds to the ratio between the number of particles before and after resampling. Only used with the `leveling_thinning` algorithm.",
    )
    resampling_algorithm_target_weight: float | None = Field(
        default=None,
        description="Weight that the product particles from resampling will not exceed.",
    )
    resampling_algorithm_velocity_grid_type: str | None = Field(
        default=None,
        description="Type of grid to use when clustering particles in velocity space. Only applicable with the `velocity_coincidence_thinning` algorithm.",
    )
    resampling_algorithm_delta_ur: float | None = Field(
        default=None,
        description='Size of velocity window used for clustering particles during grid-based merging, with `velocity_grid_type == "spherical"`.',
    )
    resampling_algorithm_n_theta: int | None = Field(
        default=None,
        description='Number of bins to use in theta when clustering particle velocities during grid-based merging, with `velocity_grid_type == "spherical"`.',
    )
    resampling_algorithm_n_phi: int | None = Field(
        default=None,
        description='Number of bins to use in phi when clustering particle velocities during grid-based merging, with `velocity_grid_type == "spherical"`.',
    )
    resampling_algorithm_delta_u: float | list[float] | None = Field(
        default=None,
        description='Size of velocity window used in ux, uy and uz for clustering particles during grid-based merging, with `velocity_grid_type == "cartesian"`. If a single number is given the same du value will be used in all three directions.',
    )

    # extra particle attributes (option names differ from the field names):
    extra_int_attributes: dict[str, str] | None = Field(
        default=None,
        alias="warpx_add_int_attributes",
        description="Dictionary of extra integer particle attributes initialized from an expression that is a function of the variables (x, y, z, ux, uy, uz, t).",
    )
    extra_real_attributes: dict[str, str] | None = Field(
        default=None,
        alias="warpx_add_real_attributes",
        description="Dictionary of extra real particle attributes initialized from an expression that is a function of the variables (x, y, z, ux, uy, uz, t).",
    )

    do_temperature_deposition: bool | None = Field(
        default=None,
        description="This flag is set per species to do another pass to deposit temperature on each timestep if required. Currently only works with Ohm's Law Hybrid Solver.",
    )

    # --- Runtime state (not user inputs; populated during/after initialization).
    _species_type: str | None = PrivateAttr(default=None)
    _element: periodictable.core.Element | None = PrivateAttr(default=None)
    _species_number: int | None = PrivateAttr(default=None)
    _species: pywarpx.Bucket.Bucket | None = PrivateAttr(default=None)

    @property
    def species(self):
        """The WarpX inputs of this species (available after the inputs are initialized)"""
        return self._species

    def model_post_init(self, context) -> None:
        super().model_post_init(context)

        self._species_type = None
        if self.particle_type in [
            "unspecified",
            "electron",
            "positron",
            "muon",
            "antimuon",
            "photon",
            "neutron",
            "proton",
            "antiproton",
            "alpha",
        ]:
            self._species_type = self.particle_type
        else:
            if self.charge is None and self.charge_state is not None:
                self.charge = f"{self.charge_state}*q_e"
            if self.particle_type is not None:
                # Match a string of the format '#nXx', with the '#n' optional isotope number.
                m = re.match(r"(?P<iso>#[\d+])*(?P<sym>[A-Za-z]+)", self.particle_type)
                if m is not None:
                    element = periodictable.elements.symbol(m["sym"])
                    if m["iso"] is not None:
                        element = element[m["iso"][1:]]
                    if self.charge_state is not None:
                        assert self.charge_state <= element.number, Exception(
                            "%s charge state not valid" % self.particle_type
                        )
                        try:
                            element = element.ion[self.charge_state]
                        except ValueError:
                            # Note that not all valid charge states are defined in elements,
                            # so this value error can be ignored.
                            pass
                    self._element = element
                    if self.mass is None:
                        self.mass = (
                            element.mass * periodictable.constants.atomic_mass_constant
                        )
                else:
                    raise Exception('The species "particle_type" is not known')

        if (
            self.resampling_algorithm_delta_u is not None
            and np.size(self.resampling_algorithm_delta_u) == 1
        ):
            self.resampling_algorithm_delta_u = [self.resampling_algorithm_delta_u] * 3

    def species_initialize_inputs(
        self,
        layout,
        initialize_self_fields=False,
        injection_plane_position=None,
        injection_plane_normal_vector=None,
    ):
        self._species_number = len(pywarpx.particles.species_names)

        if self.name is None:
            self.name = "species{}".format(self._species_number)

        pywarpx.particles.species_names.append(self.name)

        if initialize_self_fields is None:
            initialize_self_fields = False

        self._species = pywarpx.Bucket.Bucket(
            self.name,
            species_type=self._species_type,
            mass=self.mass,
            charge=self.charge,
            injection_style=None,
            initialize_self_fields=int(initialize_self_fields),
            boost_adjust_transverse_positions=self.boost_adjust_transverse_positions,
            self_fields_required_precision=self.self_fields_required_precision,
            self_fields_absolute_tolerance=self.self_fields_absolute_tolerance,
            self_fields_max_iters=self.self_fields_max_iters,
            self_fields_verbosity=self.self_fields_verbosity,
            save_particles_at_xlo=self.save_particles_at_xlo,
            save_particles_at_xhi=self.save_particles_at_xhi,
            save_particles_at_ylo=self.save_particles_at_ylo,
            save_particles_at_yhi=self.save_particles_at_yhi,
            save_particles_at_zlo=self.save_particles_at_zlo,
            save_particles_at_zhi=self.save_particles_at_zhi,
            save_particles_at_eb=self.save_particles_at_eb,
            save_previous_position=self.save_previous_position,
            do_not_deposit=self.do_not_deposit,
            do_not_push=self.do_not_push,
            do_not_gather=self.do_not_gather,
            radial_numpercell_power=self.radial_numpercell_power,
            random_theta=self.random_theta,
            do_resampling=self.do_resampling,
            resampling_algorithm=self.resampling_algorithm,
            resampling_min_ppc=self.resampling_min_ppc,
            resampling_trigger_intervals=self.resampling_trigger_intervals,
            resampling_trigger_max_avg_ppc=self.resampling_triggering_max_avg_ppc,
            resampling_algorithm_target_ratio=self.resampling_algorithm_target_ratio,
            resampling_algorithm_target_weight=self.resampling_algorithm_target_weight,
            resampling_algorithm_velocity_grid_type=self.resampling_algorithm_velocity_grid_type,
            resampling_algorithm_delta_ur=self.resampling_algorithm_delta_ur,
            resampling_algorithm_n_theta=self.resampling_algorithm_n_theta,
            resampling_algorithm_n_phi=self.resampling_algorithm_n_phi,
            resampling_algorithm_delta_u=self.resampling_algorithm_delta_u,
            do_temperature_deposition=self.do_temperature_deposition,
        )

        # add reflection models
        self._species.add_new_attr("reflection_model_xlo(E)", self.reflection_model_xlo)
        self._species.add_new_attr("reflection_model_xhi(E)", self.reflection_model_xhi)
        self._species.add_new_attr("reflection_model_ylo(E)", self.reflection_model_ylo)
        self._species.add_new_attr("reflection_model_yhi(E)", self.reflection_model_yhi)
        self._species.add_new_attr("reflection_model_zlo(E)", self.reflection_model_zlo)
        self._species.add_new_attr("reflection_model_zhi(E)", self.reflection_model_zhi)
        # self._species.add_new_attr("reflection_model_eb(E)", self.reflection_model_eb)

        # extra particle attributes
        if self.extra_int_attributes is not None:
            self._species.addIntegerAttributes = self.extra_int_attributes.keys()
            for attr, function in self.extra_int_attributes.items():
                self._species.add_new_attr(
                    "attribute." + attr + "(x,y,z,ux,uy,uz,t)", function
                )
        if self.extra_real_attributes is not None:
            self._species.addRealAttributes = self.extra_real_attributes.keys()
            for attr, function in self.extra_real_attributes.items():
                self._species.add_new_attr(
                    "attribute." + attr + "(x,y,z,ux,uy,uz,t)", function
                )

        pywarpx.Particles.particles_list.append(self._species)

        if self.initial_distribution is not None:
            # Note: PICMI objects are pydantic models, which are themselves iterable, so
            # check explicitly for an actual list/tuple of distributions/layouts here.
            distributions_is_list = isinstance(self.initial_distribution, (list, tuple))
            layout_is_list = isinstance(layout, (list, tuple))
            if not distributions_is_list and not layout_is_list:
                self.initial_distribution.distribution_initialize_inputs(
                    self._species_number, layout, self._species, self.density_scale, ""
                )
            elif distributions_is_list and (layout_is_list or layout is None):
                assert layout is None or (
                    len(self.initial_distribution) == len(layout)
                ), Exception(
                    "The initial distribution and layout lists must have the same lenth"
                )
                source_names = [
                    f"dist{i}" for i in range(len(self.initial_distribution))
                ]
                self._species.injection_sources = source_names
                for i, dist in enumerate(self.initial_distribution):
                    layout_i = layout[i] if layout is not None else None
                    dist.distribution_initialize_inputs(
                        self._species_number,
                        layout_i,
                        self._species,
                        self.density_scale,
                        source_names[i],
                    )
            else:
                raise Exception(
                    "The initial distribution and layout must both be scalars or both be lists"
                )

        if injection_plane_position is not None:
            if injection_plane_normal_vector is not None:
                assert (
                    injection_plane_normal_vector[0] == 0.0
                    and injection_plane_normal_vector[1] == 0.0
                ), Exception("Rigid injection can only be done along z")
            pywarpx.particles.rigid_injected_species.append(self.name)
            self._species.rigid_advance = 1
            self._species.zinject_plane = injection_plane_position


picmistandard.PICMI_MultiSpecies.Species_class = Species


class MultiSpecies(picmistandard.PICMI_MultiSpecies):
    def species_initialize_inputs(
        self,
        layout,
        initialize_self_fields=False,
        injection_plane_position=None,
        injection_plane_normal_vector=None,
    ):
        for species in self.species_instances_list:
            species.species_initialize_inputs(
                layout,
                initialize_self_fields,
                injection_plane_position,
                injection_plane_normal_vector,
            )


class GaussianBunchDistribution(picmistandard.PICMI_GaussianBunchDistribution):
    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_GaussianBunchDistribution)
    )

    do_symmetrize: bool | None = Field(
        default=None, description="Whether to symmetrize the bunch"
    )
    symmetrization_order: int | None = Field(
        default=None, description="The order of the symmetrization (4 or 8)"
    )

    def distribution_initialize_inputs(
        self, species_number, layout, species, density_scale, source_name
    ):
        species.add_new_group_attr(source_name, "injection_style", "gaussian_beam")
        species.add_new_group_attr(source_name, "x_m", self.centroid_position[0])
        species.add_new_group_attr(source_name, "y_m", self.centroid_position[1])
        species.add_new_group_attr(source_name, "z_m", self.centroid_position[2])
        species.add_new_group_attr(source_name, "x_rms", self.rms_bunch_size[0])
        species.add_new_group_attr(source_name, "y_rms", self.rms_bunch_size[1])
        species.add_new_group_attr(source_name, "z_rms", self.rms_bunch_size[2])

        # --- Only PseudoRandomLayout is supported
        species.add_new_group_attr(source_name, "npart", layout.n_macroparticles)

        # --- Total number of real particles
        species.add_new_group_attr(source_name, "npart_real", self.n_physical_particles)
        if density_scale is not None:
            species.add_new_group_attr(source_name, "npart_real", density_scale)

        # --- The PICMI standard doesn't yet have a way of specifying these values.
        # --- They should default to the size of the domain. They are not typically
        # --- necessary though since any particles outside the domain are rejected.
        # species.xmin
        # species.xmax
        # species.ymin
        # species.ymax
        # species.zmin
        # species.zmax

        # --- Note that WarpX takes gamma*beta as input
        if np.any(np.not_equal(self.velocity_divergence, 0.0)):
            u_over_x = self.velocity_divergence[0] / constants.c
            u_over_y = self.velocity_divergence[1] / constants.c
            u_over_z = self.velocity_divergence[2] / constants.c
            species.add_new_group_attr(
                source_name, "momentum_distribution_type", "parse_momentum_function"
            )
            species.add_new_group_attr(
                source_name, "momentum_function_ux(x,y,z)", f"{u_over_x}*x"
            )
            species.add_new_group_attr(
                source_name, "momentum_function_uy(x,y,z)", f"{u_over_y}*y"
            )
            species.add_new_group_attr(
                source_name, "momentum_function_uz(x,y,z)", f"{u_over_z}*z"
            )
        elif np.any(np.not_equal(self.rms_velocity, 0.0)):
            species.add_new_group_attr(
                source_name, "momentum_distribution_type", "gaussian"
            )
            species.add_new_group_attr(
                source_name, "ux_m", self.centroid_velocity[0] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uy_m", self.centroid_velocity[1] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uz_m", self.centroid_velocity[2] / constants.c
            )
            species.add_new_group_attr(
                source_name, "ux_th", self.rms_velocity[0] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uy_th", self.rms_velocity[1] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uz_th", self.rms_velocity[2] / constants.c
            )
        else:
            species.add_new_group_attr(
                source_name, "momentum_distribution_type", "constant"
            )
            species.add_new_group_attr(
                source_name, "ux", self.centroid_velocity[0] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uy", self.centroid_velocity[1] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uz", self.centroid_velocity[2] / constants.c
            )

        species.add_new_group_attr(source_name, "do_symmetrize", self.do_symmetrize)
        species.add_new_group_attr(
            source_name, "symmetrization_order", self.symmetrization_order
        )


class DensityDistributionBase(object):
    """This is a base class for several predefined density distributions. It
    captures universal initialization logic."""

    def set_mangle_dict(self):
        # The classes using this mixin declare the ``_mangle_dict`` private attribute.
        if hasattr(self, "user_defined_kw") and self._mangle_dict is None:
            # Only do this once so that the same variables can be used multiple
            # times
            self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)

    def set_species_attributes(self, species, layout, source_name):
        if isinstance(layout, GriddedLayout):
            # --- Note that the grid attribute of GriddedLayout is ignored
            species.add_new_group_attr(
                source_name, "injection_style", "nuniformpercell"
            )
            species.add_new_group_attr(
                source_name,
                "num_particles_per_cell_each_dim",
                layout.n_macroparticle_per_cell,
            )
        elif isinstance(layout, PseudoRandomLayout):
            assert layout.n_macroparticles_per_cell is not None, Exception(
                "WarpX only supports n_macroparticles_per_cell for the PseudoRandomLayout with this distribution"
            )
            species.add_new_group_attr(source_name, "injection_style", "nrandompercell")
            species.add_new_group_attr(
                source_name, "num_particles_per_cell", layout.n_macroparticles_per_cell
            )
        else:
            raise Exception(
                "WarpX does not support the specified layout for this distribution"
            )

        species.add_new_group_attr(source_name, "xmin", self.lower_bound[0])
        species.add_new_group_attr(source_name, "xmax", self.upper_bound[0])
        species.add_new_group_attr(source_name, "ymin", self.lower_bound[1])
        species.add_new_group_attr(source_name, "ymax", self.upper_bound[1])
        species.add_new_group_attr(source_name, "zmin", self.lower_bound[2])
        species.add_new_group_attr(source_name, "zmax", self.upper_bound[2])

        # the flux distributions are always injected continuously and have no fill_in
        if getattr(self, "fill_in", None):
            species.add_new_group_attr(source_name, "do_continuous_injection", 1)

        if hasattr(self, "momentum_spread_expressions") and np.any(
            np.not_equal(self.momentum_spread_expressions, None)
        ):
            species.add_new_group_attr(
                source_name, "momentum_distribution_type", "maxwellian"
            )
            # Mean drift: any axis left as None falls back to directed_velocity.
            species.add_new_group_attr(
                source_name, "maxwellian_u_mean_distribution_type", "parser"
            )
            self.setup_parse_momentum_functions(
                species,
                source_name,
                self.momentum_expressions,
                self.directed_velocity,
                "u{dir}_mean_function(x,y,z)",
            )
            # Thermal spread: any axis left as None falls back to zero.
            species.add_new_group_attr(
                source_name, "maxwellian_u_std_distribution_type", "parser"
            )
            self.setup_parse_momentum_functions(
                species,
                source_name,
                self.momentum_spread_expressions,
                [0.0, 0.0, 0.0],
                "u{dir}_std_function(x,y,z)",
            )
        elif hasattr(self, "momentum_expressions") and np.any(
            np.not_equal(self.momentum_expressions, None)
        ):
            species.add_new_group_attr(
                source_name, "momentum_distribution_type", "parse_momentum_function"
            )
            self.setup_parse_momentum_functions(
                species,
                source_name,
                self.momentum_expressions,
                self.directed_velocity,
                "momentum_function_u{dir}(x,y,z)",
            )
        elif np.any(np.not_equal(self.rms_velocity, 0.0)):
            species.add_new_group_attr(
                source_name, "momentum_distribution_type", "gaussian"
            )
            species.add_new_group_attr(
                source_name, "ux_m", self.directed_velocity[0] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uy_m", self.directed_velocity[1] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uz_m", self.directed_velocity[2] / constants.c
            )
            species.add_new_group_attr(
                source_name, "ux_th", self.rms_velocity[0] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uy_th", self.rms_velocity[1] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uz_th", self.rms_velocity[2] / constants.c
            )
        else:
            species.add_new_group_attr(
                source_name, "momentum_distribution_type", "constant"
            )
            species.add_new_group_attr(
                source_name, "ux", self.directed_velocity[0] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uy", self.directed_velocity[1] / constants.c
            )
            species.add_new_group_attr(
                source_name, "uz", self.directed_velocity[2] / constants.c
            )

        if hasattr(self, "density_min"):
            species.add_new_group_attr(source_name, "density_min", self.density_min)
        if hasattr(self, "density_max"):
            species.add_new_group_attr(source_name, "density_max", self.density_max)

    def setup_parse_momentum_functions(
        self, species, source_name, expressions, defaults, attr_pattern
    ):
        """Write per-component momentum parser expressions (divided by c) to the species.

        ``attr_pattern`` is a format string with a ``{dir}`` placeholder for the
        component, e.g. ``"momentum_function_u{dir}(x,y,z)"`` for the
        ``parse_momentum_function`` distribution or ``"u{dir}_mean_function(x,y,z)"``
        and ``"u{dir}_std_function(x,y,z)"`` for the ``maxwellian`` distribution.
        """
        for sdir, idir in zip(["x", "y", "z"], [0, 1, 2]):
            if expressions[idir] is not None:
                expression = pywarpx.my_constants.mangle_expression(
                    expressions[idir], self._mangle_dict
                )
            else:
                expression = f"{defaults[idir]}"
            species.add_new_group_attr(
                source_name,
                attr_pattern.format(dir=sdir),
                f"({expression})/{constants.c}",
            )


class UniformDistribution(
    picmistandard.PICMI_UniformDistribution, DensityDistributionBase
):
    # Runtime state populated by the DensityDistributionBase mixin.
    _mangle_dict: dict | None = PrivateAttr(default=None)

    def distribution_initialize_inputs(
        self, species_number, layout, species, density_scale, source_name
    ):
        self.set_mangle_dict()
        self.set_species_attributes(species, layout, source_name)

        # --- Only constant density is supported by this class
        species.add_new_group_attr(source_name, "profile", "constant")
        species.add_new_group_attr(source_name, "density", self.density)
        if density_scale is not None:
            species.add_new_group_attr(source_name, "density", density_scale)


class FluxDistributionBase(object):
    """This is a base class for both uniform and analytic flux distributions."""

    def initialize_flux_profile_func(self, species, density_scale, source_name):
        """Initialize the flux profile and flux function."""
        pass

    def distribution_initialize_inputs(
        self, species_number, layout, species, density_scale, source_name
    ):
        self.set_mangle_dict()
        self.set_species_attributes(species, layout, source_name)

        self.initialize_flux_profile_func(species, density_scale, source_name)

        if not self.inject_from_embedded_boundary:
            species.add_new_group_attr(
                source_name, "flux_normal_axis", self.flux_normal_axis
            )
            species.add_new_group_attr(
                source_name, "surface_flux_pos", self.surface_flux_position
            )
            species.add_new_group_attr(
                source_name, "flux_direction", self.flux_direction
            )
        else:
            species.add_new_group_attr(
                source_name, "inject_from_embedded_boundary", True
            )

        species.add_new_group_attr(source_name, "flux_tmin", self.flux_tmin)
        species.add_new_group_attr(source_name, "flux_tmax", self.flux_tmax)

        # --- Use specific attributes for flux injection
        species.add_new_group_attr(source_name, "injection_style", "nfluxpercell")
        assert isinstance(layout, PseudoRandomLayout), Exception(
            "UniformFluxDistribution only supports the PseudoRandomLayout in WarpX"
        )
        if self.gaussian_flux_momentum_distribution:
            species.add_new_group_attr(
                source_name, "momentum_distribution_type", "gaussianflux"
            )


class AnalyticFluxDistribution(
    picmistandard.PICMI_AnalyticFluxDistribution,
    FluxDistributionBase,
    DensityDistributionBase,
):
    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_AnalyticFluxDistribution)
    )

    inject_from_embedded_boundary: bool = Field(
        default=False,
        description="When true, the flux is injected from the embedded boundaries instead of a plane.",
    )

    # Runtime state populated by the DensityDistributionBase mixin.
    _mangle_dict: dict | None = PrivateAttr(default=None)

    def initialize_flux_profile_func(self, species, density_scale, source_name):
        species.add_new_group_attr(source_name, "flux_profile", "parse_flux_function")
        if density_scale is not None:
            species.add_new_group_attr(source_name, "flux", density_scale)
        expression = pywarpx.my_constants.mangle_expression(
            self.flux, self._mangle_dict
        )
        if density_scale is None:
            species.add_new_group_attr(
                source_name, "flux_function(x,y,z,t)", expression
            )
        else:
            species.add_new_group_attr(
                source_name,
                "flux_function(x,y,z,t)",
                "{}*({})".format(density_scale, expression),
            )


class UniformFluxDistribution(
    picmistandard.PICMI_UniformFluxDistribution,
    FluxDistributionBase,
    DensityDistributionBase,
):
    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_UniformFluxDistribution)
    )

    inject_from_embedded_boundary: bool = Field(
        default=False,
        description="When true, the flux is injected from the embedded boundaries instead of a plane.",
    )

    # Runtime state populated by the DensityDistributionBase mixin.
    _mangle_dict: dict | None = PrivateAttr(default=None)

    def initialize_flux_profile_func(self, species, density_scale, source_name):
        species.add_new_group_attr(source_name, "flux_profile", "constant")
        species.add_new_group_attr(source_name, "flux", self.flux)
        if density_scale is not None:
            species.add_new_group_attr(source_name, "flux", density_scale)


class AnalyticDistribution(
    picmistandard.PICMI_AnalyticDistribution, DensityDistributionBase
):
    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_AnalyticDistribution)
    )

    density_min: float | None = Field(
        default=None,
        description="Minimum plasma density. No particle is injected where the density is below this value.",
    )
    density_max: float | None = Field(
        default=None,
        description="Maximum plasma density. The density at each point is the minimum between the value given in the profile, and density_max.",
    )
    # Re-declare the standard ``momentum_spread_expressions`` under the WarpX alias to
    # preserve the historical ``warpx_`` spelling (both spellings are accepted).
    momentum_spread_expressions: list[Expression | None] = Field(
        default_factory=lambda: [None, None, None],
        alias="warpx_momentum_spread_expressions",
        description="Analytic expressions describing the gamma*velocity spread for each axis [m/s]. Expressions should be in terms of the position, written as 'x', 'y', and 'z'. Parameters can be used in the expression with the values given as keyword arguments. For any axis not supplied (set to None), zero will be used.",
    )

    # Runtime state populated by the DensityDistributionBase mixin.
    _mangle_dict: dict | None = PrivateAttr(default=None)

    def distribution_initialize_inputs(
        self, species_number, layout, species, density_scale, source_name
    ):
        self.set_mangle_dict()
        self.set_species_attributes(species, layout, source_name)

        species.add_new_group_attr(source_name, "profile", "parse_density_function")
        expression = pywarpx.my_constants.mangle_expression(
            self.density_expression, self._mangle_dict
        )
        if density_scale is None:
            species.add_new_group_attr(
                source_name, "density_function(x,y,z)", expression
            )
        else:
            species.add_new_group_attr(
                source_name,
                "density_function(x,y,z)",
                "{}*({})".format(density_scale, expression),
            )


class ParticleListDistribution(picmistandard.PICMI_ParticleListDistribution):
    def distribution_initialize_inputs(
        self, species_number, layout, species, density_scale, source_name
    ):
        species.add_new_group_attr(source_name, "injection_style", "multipleparticles")
        species.add_new_group_attr(source_name, "multiple_particles_pos_x", self.x)
        species.add_new_group_attr(source_name, "multiple_particles_pos_y", self.y)
        species.add_new_group_attr(source_name, "multiple_particles_pos_z", self.z)
        species.add_new_group_attr(
            source_name, "multiple_particles_ux", np.array(self.ux) / constants.c
        )
        species.add_new_group_attr(
            source_name, "multiple_particles_uy", np.array(self.uy) / constants.c
        )
        species.add_new_group_attr(
            source_name, "multiple_particles_uz", np.array(self.uz) / constants.c
        )
        species.add_new_group_attr(
            source_name, "multiple_particles_weight", self.weight
        )
        if density_scale is not None:
            species.add_new_group_attr(
                source_name,
                "multiple_particles_weight",
                np.asarray(self.weight) * density_scale,
            )


class FromFileDistribution(picmistandard.PICMI_FromFileDistribution):
    def distribution_initialize_inputs(
        self, species_number, layout, species, density_scale, source_name
    ):
        species.add_new_group_attr(source_name, "injection_style", "external_file")
        species.add_new_group_attr(source_name, "injection_file", self.file_path)


class ParticleDistributionPlanarInjector(
    picmistandard.PICMI_ParticleDistributionPlanarInjector
):
    pass


class GriddedLayout(picmistandard.PICMI_GriddedLayout):
    pass


class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        if self.seed is not None:
            print(
                "Warning: WarpX does not support specifying the random number seed in PseudoRandomLayout"
            )


class BinomialSmoother(picmistandard.PICMI_BinomialSmoother):
    def smoother_initialize_inputs(self, solver):
        pywarpx.warpx.use_filter = 1
        pywarpx.warpx.use_filter_compensation = bool(np.all(self.compensation))
        if self.n_pass is None:
            # If not specified, do at least one pass in each direction.
            self.n_pass = 1
        try:
            # Check if n_pass is a vector
            len(self.n_pass)
        except TypeError:
            # If not, make it a vector
            self.n_pass = solver.grid.number_of_dimensions * [self.n_pass]
        pywarpx.warpx.filter_npass_each_dir = self.n_pass


class CylindricalGrid(picmistandard.PICMI_CylindricalGrid):
    """
    This assumes that WarpX was compiled with USE_RZ = TRUE

    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_CylindricalGrid)
    )

    max_grid_size: int | list[int] = Field(
        default=32, description="Maximum block size in either direction"
    )
    max_grid_size_x: int | None = Field(
        default=None, description="Maximum block size in radial direction"
    )
    max_grid_size_y: int | None = Field(
        default=None, description="Maximum block size in longitudinal direction"
    )
    blocking_factor: int | list[int] | None = Field(
        default=None, description="Blocking factor (which controls the block size)"
    )
    blocking_factor_x: int | None = Field(
        default=None,
        description="Blocking factor (which controls the block size) in the radial direction",
    )
    blocking_factor_y: int | None = Field(
        default=None,
        description="Blocking factor (which controls the block size) in the longitudinal direction",
    )
    # option names (warpx_potential_lo/hi_r/z) differ from the field names:
    potential_xmin: float | str | None = Field(
        default=None,
        alias="warpx_potential_lo_r",
        description="Electrostatic potential on the lower radial boundary",
    )
    potential_xmax: float | str | None = Field(
        default=None,
        alias="warpx_potential_hi_r",
        description="Electrostatic potential on the upper radial boundary",
    )
    potential_zmin: float | str | None = Field(
        default=None,
        alias="warpx_potential_lo_z",
        description="Electrostatic potential on the lower longitudinal boundary",
    )
    potential_zmax: float | str | None = Field(
        default=None,
        alias="warpx_potential_hi_z",
        description="Electrostatic potential on the upper longitudinal boundary",
    )
    reflect_all_velocities: bool | None = Field(
        default=None,
        description="Whether the sign of all of the particle velocities are changed upon reflection on a boundary, or only the velocity normal to the surface",
    )
    start_moving_window_step: int | None = Field(
        default=None, description="The timestep at which the moving window starts"
    )
    end_moving_window_step: int | None = Field(
        default=None,
        description="The timestep at which the moving window ends. If -1, the moving window will continue until the end of the simulation.",
    )
    thermal_boundary_u_th: dict[str, float] | None = Field(
        default=None,
        alias="warpx_boundary_u_th",
        description="If a thermal boundary is used for particles, this dictionary should specify the thermal speed for each species in the form {`<species>`: u_th}. Note: u_th = sqrt(T*q_e/mass)/clight with T in eV.",
    )
    # RZ geometry has no second Cartesian axis.
    potential_ymin = _potential_not_in_geometry("potential_ymin", "RZ")
    potential_ymax = _potential_not_in_geometry("potential_ymax", "RZ")

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        # Geometry
        # Set these as soon as the information is available
        # (since these are needed to determine which shared object to load)
        pywarpx.geometry.dims = "RZ"
        pywarpx.geometry.prob_lo = self.lower_bound  # physical domain
        pywarpx.geometry.prob_hi = self.upper_bound

    def grid_initialize_inputs(self):
        pywarpx.amr.n_cell = self.number_of_cells

        # Maximum allowable size of each subdomain in the problem domain;
        #    this is used to decompose the domain for parallel calculations.
        pywarpx.amr.max_grid_size = self.max_grid_size
        pywarpx.amr.max_grid_size_x = self.max_grid_size_x
        pywarpx.amr.max_grid_size_y = self.max_grid_size_y
        pywarpx.amr.blocking_factor = self.blocking_factor
        pywarpx.amr.blocking_factor_x = self.blocking_factor_x
        pywarpx.amr.blocking_factor_y = self.blocking_factor_y

        assert self.lower_bound[0] >= 0.0, Exception(
            "Lower radial boundary must be >= 0."
        )
        assert (
            self.lower_boundary_conditions[0] != "periodic"
            and self.upper_boundary_conditions[0] != "periodic"
        ), Exception("Radial boundaries can not be periodic")

        pywarpx.warpx.n_rz_azimuthal_modes = self.n_azimuthal_modes

        # Boundary conditions
        pywarpx.boundary.field_lo = [
            BC_map[bc] for bc in self.lower_boundary_conditions
        ]
        pywarpx.boundary.field_hi = [
            BC_map[bc] for bc in self.upper_boundary_conditions
        ]
        pywarpx.boundary.particle_lo = self.lower_boundary_conditions_particles
        pywarpx.boundary.particle_hi = self.upper_boundary_conditions_particles
        pywarpx.boundary.reflect_all_velocities = self.reflect_all_velocities

        if self.thermal_boundary_u_th is not None:
            for name, val in self.thermal_boundary_u_th.items():
                pywarpx.boundary.__setattr__(f"{name}.u_th", val)

        if self.moving_window_velocity is not None and np.any(
            np.not_equal(self.moving_window_velocity, 0.0)
        ):
            pywarpx.warpx.do_moving_window = 1
            if self.moving_window_velocity[0] != 0.0:
                pywarpx.warpx.moving_window_dir = "r"
                pywarpx.warpx.moving_window_v = (
                    self.moving_window_velocity[0] / constants.c
                )  # in units of the speed of light
            if self.moving_window_velocity[1] != 0.0:
                pywarpx.warpx.moving_window_dir = "z"
                pywarpx.warpx.moving_window_v = (
                    self.moving_window_velocity[1] / constants.c
                )  # in units of the speed of light

            pywarpx.warpx.start_moving_window_step = self.start_moving_window_step
            pywarpx.warpx.end_moving_window_step = self.end_moving_window_step

        _set_refined_region_inputs(self.refined_regions)


class Cartesian1DGrid(picmistandard.PICMI_Cartesian1DGrid):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_Cartesian1DGrid)
    )

    max_grid_size: int | list[int] = Field(
        default=32, description="Maximum block size in either direction"
    )
    max_grid_size_x: int | None = Field(
        default=None, description="Maximum block size in longitudinal direction"
    )
    blocking_factor: int | list[int] | None = Field(
        default=None, description="Blocking factor (which controls the block size)"
    )
    blocking_factor_x: int | None = Field(
        default=None,
        description="Blocking factor (which controls the block size) in the longitudinal direction",
    )
    # option names (warpx_potential_lo/hi_z) differ from the field names:
    potential_zmin: float | str | None = Field(
        default=None,
        alias="warpx_potential_lo_z",
        description="Electrostatic potential on the lower longitudinal boundary",
    )
    potential_zmax: float | str | None = Field(
        default=None,
        alias="warpx_potential_hi_z",
        description="Electrostatic potential on the upper longitudinal boundary",
    )
    start_moving_window_step: int | None = Field(
        default=None, description="The timestep at which the moving window starts"
    )
    end_moving_window_step: int | None = Field(
        default=None,
        description="The timestep at which the moving window ends. If -1, the moving window will continue until the end of the simulation.",
    )
    thermal_boundary_u_th: dict[str, float] | None = Field(
        default=None,
        alias="warpx_boundary_u_th",
        description="If a thermal boundary is used for particles, this dictionary should specify the thermal speed for each species in the form {`<species>`: u_th}. Note: u_th = sqrt(T*q_e/mass)/clight with T in eV.",
    )
    # 1D geometry has only the longitudinal (z) axis.
    potential_xmin = _potential_not_in_geometry("potential_xmin", "1D")
    potential_xmax = _potential_not_in_geometry("potential_xmax", "1D")
    potential_ymin = _potential_not_in_geometry("potential_ymin", "1D")
    potential_ymax = _potential_not_in_geometry("potential_ymax", "1D")

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        # Geometry
        # Set these as soon as the information is available
        # (since these are needed to determine which shared object to load)
        pywarpx.geometry.dims = "1"
        pywarpx.geometry.prob_lo = self.lower_bound  # physical domain
        pywarpx.geometry.prob_hi = self.upper_bound

    def grid_initialize_inputs(self):
        pywarpx.amr.n_cell = self.number_of_cells

        # Maximum allowable size of each subdomain in the problem domain;
        #    this is used to decompose the domain for parallel calculations.
        pywarpx.amr.max_grid_size = self.max_grid_size
        pywarpx.amr.max_grid_size_x = self.max_grid_size_x
        pywarpx.amr.blocking_factor = self.blocking_factor
        pywarpx.amr.blocking_factor_x = self.blocking_factor_x

        # Boundary conditions
        pywarpx.boundary.field_lo = [
            BC_map[bc] for bc in self.lower_boundary_conditions
        ]
        pywarpx.boundary.field_hi = [
            BC_map[bc] for bc in self.upper_boundary_conditions
        ]
        pywarpx.boundary.particle_lo = self.lower_boundary_conditions_particles
        pywarpx.boundary.particle_hi = self.upper_boundary_conditions_particles

        if self.thermal_boundary_u_th is not None:
            for name, val in self.thermal_boundary_u_th.items():
                pywarpx.boundary.__setattr__(f"{name}.u_th", val)

        if self.moving_window_velocity is not None and np.any(
            np.not_equal(self.moving_window_velocity, 0.0)
        ):
            pywarpx.warpx.do_moving_window = 1
            if self.moving_window_velocity[0] != 0.0:
                pywarpx.warpx.moving_window_dir = "z"
                pywarpx.warpx.moving_window_v = (
                    self.moving_window_velocity[0] / constants.c
                )  # in units of the speed of light

            pywarpx.warpx.start_moving_window_step = self.start_moving_window_step
            pywarpx.warpx.end_moving_window_step = self.end_moving_window_step

        _set_refined_region_inputs(self.refined_regions)


class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_Cartesian2DGrid)
    )

    max_grid_size: int | list[int] = Field(
        default=32, description="Maximum block size in either direction"
    )
    max_grid_size_x: int | None = Field(
        default=None, description="Maximum block size in x direction"
    )
    max_grid_size_y: int | None = Field(
        default=None, description="Maximum block size in z direction"
    )
    blocking_factor: int | list[int] | None = Field(
        default=None, description="Blocking factor (which controls the block size)"
    )
    blocking_factor_x: int | None = Field(
        default=None,
        description="Blocking factor (which controls the block size) in the x direction",
    )
    blocking_factor_y: int | None = Field(
        default=None,
        description="Blocking factor (which controls the block size) in the z direction",
    )
    # option names (warpx_potential_lo/hi_x/z) differ from the field names:
    potential_xmin: float | str | None = Field(
        default=None,
        alias="warpx_potential_lo_x",
        description="Electrostatic potential on the lower x boundary",
    )
    potential_xmax: float | str | None = Field(
        default=None,
        alias="warpx_potential_hi_x",
        description="Electrostatic potential on the upper x boundary",
    )
    potential_zmin: float | str | None = Field(
        default=None,
        alias="warpx_potential_lo_z",
        description="Electrostatic potential on the lower z boundary",
    )
    potential_zmax: float | str | None = Field(
        default=None,
        alias="warpx_potential_hi_z",
        description="Electrostatic potential on the upper z boundary",
    )
    start_moving_window_step: int | None = Field(
        default=None, description="The timestep at which the moving window starts"
    )
    end_moving_window_step: int | None = Field(
        default=None,
        description="The timestep at which the moving window ends. If -1, the moving window will continue until the end of the simulation.",
    )
    thermal_boundary_u_th: dict[str, float] | None = Field(
        default=None,
        alias="warpx_boundary_u_th",
        description="If a thermal boundary is used for particles, this dictionary should specify the thermal speed for each species in the form {`<species>`: u_th}. Note: u_th = sqrt(T*q_e/mass)/clight with T in eV.",
    )
    # 2D geometry is the x-z plane; there is no second (y) axis.
    potential_ymin = _potential_not_in_geometry("potential_ymin", "2D")
    potential_ymax = _potential_not_in_geometry("potential_ymax", "2D")

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        # Geometry
        # Set these as soon as the information is available
        # (since these are needed to determine which shared object to load)
        pywarpx.geometry.dims = "2"
        pywarpx.geometry.prob_lo = self.lower_bound  # physical domain
        pywarpx.geometry.prob_hi = self.upper_bound

    def grid_initialize_inputs(self):
        pywarpx.amr.n_cell = self.number_of_cells

        # Maximum allowable size of each subdomain in the problem domain;
        #    this is used to decompose the domain for parallel calculations.
        pywarpx.amr.max_grid_size = self.max_grid_size
        pywarpx.amr.max_grid_size_x = self.max_grid_size_x
        pywarpx.amr.max_grid_size_y = self.max_grid_size_y
        pywarpx.amr.blocking_factor = self.blocking_factor
        pywarpx.amr.blocking_factor_x = self.blocking_factor_x
        pywarpx.amr.blocking_factor_y = self.blocking_factor_y

        # Boundary conditions
        pywarpx.boundary.field_lo = [
            BC_map[bc] for bc in self.lower_boundary_conditions
        ]
        pywarpx.boundary.field_hi = [
            BC_map[bc] for bc in self.upper_boundary_conditions
        ]
        pywarpx.boundary.particle_lo = self.lower_boundary_conditions_particles
        pywarpx.boundary.particle_hi = self.upper_boundary_conditions_particles

        if self.thermal_boundary_u_th is not None:
            for name, val in self.thermal_boundary_u_th.items():
                pywarpx.boundary.__setattr__(f"{name}.u_th", val)

        if self.moving_window_velocity is not None and np.any(
            np.not_equal(self.moving_window_velocity, 0.0)
        ):
            pywarpx.warpx.do_moving_window = 1
            if self.moving_window_velocity[0] != 0.0:
                pywarpx.warpx.moving_window_dir = "x"
                pywarpx.warpx.moving_window_v = (
                    self.moving_window_velocity[0] / constants.c
                )  # in units of the speed of light
            if self.moving_window_velocity[1] != 0.0:
                pywarpx.warpx.moving_window_dir = "z"
                pywarpx.warpx.moving_window_v = (
                    self.moving_window_velocity[1] / constants.c
                )  # in units of the speed of light

            pywarpx.warpx.start_moving_window_step = self.start_moving_window_step
            pywarpx.warpx.end_moving_window_step = self.end_moving_window_step

        _set_refined_region_inputs(self.refined_regions)


class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_Cartesian3DGrid)
    )

    max_grid_size: int | list[int] = Field(
        default=32, description="Maximum block size in either direction"
    )
    max_grid_size_x: int | None = Field(
        default=None, description="Maximum block size in x direction"
    )
    max_grid_size_y: int | None = Field(
        default=None, description="Maximum block size in z direction"
    )
    max_grid_size_z: int | None = Field(
        default=None, description="Maximum block size in z direction"
    )
    blocking_factor: int | list[int] | None = Field(
        default=None, description="Blocking factor (which controls the block size)"
    )
    blocking_factor_x: int | None = Field(
        default=None,
        description="Blocking factor (which controls the block size) in the x direction",
    )
    blocking_factor_y: int | None = Field(
        default=None,
        description="Blocking factor (which controls the block size) in the z direction",
    )
    blocking_factor_z: int | None = Field(
        default=None,
        description="Blocking factor (which controls the block size) in the z direction",
    )
    potential_xmin: float | str | None = Field(
        default=None,
        alias="warpx_potential_lo_x",
        description="Electrostatic potential on the lower x boundary",
    )
    potential_xmax: float | str | None = Field(
        default=None,
        alias="warpx_potential_hi_x",
        description="Electrostatic potential on the upper x boundary",
    )
    potential_ymin: float | str | None = Field(
        default=None,
        alias="warpx_potential_lo_y",
        description="Electrostatic potential on the lower z boundary",
    )
    potential_ymax: float | str | None = Field(
        default=None,
        alias="warpx_potential_hi_y",
        description="Electrostatic potential on the upper z boundary",
    )
    potential_zmin: float | str | None = Field(
        default=None,
        alias="warpx_potential_lo_z",
        description="Electrostatic potential on the lower z boundary",
    )
    potential_zmax: float | str | None = Field(
        default=None,
        alias="warpx_potential_hi_z",
        description="Electrostatic potential on the upper z boundary",
    )
    start_moving_window_step: int | None = Field(
        default=None, description="The timestep at which the moving window starts"
    )
    end_moving_window_step: int | None = Field(
        default=None,
        description="The timestep at which the moving window ends. If -1, the moving window will continue until the end of the simulation.",
    )
    thermal_boundary_u_th: dict[str, float] | None = Field(
        default=None,
        alias="warpx_boundary_u_th",
        description="If a thermal boundary is used for particles, this dictionary should specify the thermal speed for each species in the form {`<species>`: u_th}. Note: u_th = sqrt(T*q_e/mass)/clight with T in eV.",
    )

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        # Geometry
        # Set these as soon as the information is available
        # (since these are needed to determine which shared object to load)
        pywarpx.geometry.dims = "3"
        pywarpx.geometry.prob_lo = self.lower_bound  # physical domain
        pywarpx.geometry.prob_hi = self.upper_bound

    def grid_initialize_inputs(self):
        pywarpx.amr.n_cell = self.number_of_cells

        # Maximum allowable size of each subdomain in the problem domain;
        #    this is used to decompose the domain for parallel calculations.
        pywarpx.amr.max_grid_size = self.max_grid_size
        pywarpx.amr.max_grid_size_x = self.max_grid_size_x
        pywarpx.amr.max_grid_size_y = self.max_grid_size_y
        pywarpx.amr.max_grid_size_z = self.max_grid_size_z
        pywarpx.amr.blocking_factor = self.blocking_factor
        pywarpx.amr.blocking_factor_x = self.blocking_factor_x
        pywarpx.amr.blocking_factor_y = self.blocking_factor_y
        pywarpx.amr.blocking_factor_z = self.blocking_factor_z

        # Boundary conditions
        pywarpx.boundary.field_lo = [
            BC_map[bc] for bc in self.lower_boundary_conditions
        ]
        pywarpx.boundary.field_hi = [
            BC_map[bc] for bc in self.upper_boundary_conditions
        ]
        pywarpx.boundary.particle_lo = self.lower_boundary_conditions_particles
        pywarpx.boundary.particle_hi = self.upper_boundary_conditions_particles

        if self.thermal_boundary_u_th is not None:
            for name, val in self.thermal_boundary_u_th.items():
                pywarpx.boundary.__setattr__(f"{name}.u_th", val)

        if self.moving_window_velocity is not None and np.any(
            np.not_equal(self.moving_window_velocity, 0.0)
        ):
            pywarpx.warpx.do_moving_window = 1
            if self.moving_window_velocity[0] != 0.0:
                pywarpx.warpx.moving_window_dir = "x"
                pywarpx.warpx.moving_window_v = (
                    self.moving_window_velocity[0] / constants.c
                )  # in units of the speed of light
            if self.moving_window_velocity[1] != 0.0:
                pywarpx.warpx.moving_window_dir = "y"
                pywarpx.warpx.moving_window_v = (
                    self.moving_window_velocity[1] / constants.c
                )  # in units of the speed of light
            if self.moving_window_velocity[2] != 0.0:
                pywarpx.warpx.moving_window_dir = "z"
                pywarpx.warpx.moving_window_v = (
                    self.moving_window_velocity[2] / constants.c
                )  # in units of the speed of light

            pywarpx.warpx.start_moving_window_step = self.start_moving_window_step
            pywarpx.warpx.end_moving_window_step = self.end_moving_window_step

        _set_refined_region_inputs(self.refined_regions)


class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.

    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_ElectromagneticSolver)
    )

    pml_ncell: int | None = Field(
        default=None, description="The depth of the PML, in number of cells"
    )
    # option name (warpx_periodic_single_box_fft) differs from the field name:
    psatd_periodic_single_box_fft: bool | None = Field(
        default=None,
        alias="warpx_periodic_single_box_fft",
        description="Whether to do the spectral solver FFTs assuming a single simulation block",
    )
    # option name (warpx_current_correction) differs from the field name:
    psatd_current_correction: bool | None = Field(
        default=None,
        alias="warpx_current_correction",
        description="Whether to do the current correction for the spectral solver. See documentation for exceptions to the default value.",
    )
    psatd_update_with_rho: bool | None = Field(
        default=None,
        description="Whether to update with the actual rho for the spectral solver",
    )
    psatd_do_time_averaging: bool | None = Field(
        default=None,
        description="Whether to do the time averaging for the spectral solver",
    )
    psatd_JRhom: str | None = Field(
        default=None,
        description=(
            "This determines whether the PSATD JRhom algorithm is used. "
            "The parameter is a string composed by two characters and one digit. "
            "The first character represents the time dependency of J within the "
            "time step over which the electromagnetic fields are evolved, e.g., "
            '"C" for constant in time, "L" for linear in time, "Q" for quadratic '
            "in time. "
            "The second character represents the time dependency of rho within the "
            "time step over which the electromagnetic fields are evolved, following "
            "the same naming convention as for J. "
            "The last digit is an integer that represents the number of subintervals "
            "used in the JRhom algorithm. "
            'Examples: "CL1" (equivalent to the standard PSATD PIC algorithm), '
            '"CL2", "LL4", etc. '
            "By default, the string is empty and the JRhom algorithm is not used."
        ),
    )
    do_pml_in_domain: bool | None = Field(
        default=None,
        description="Whether to do the PML boundaries within the domain (versus in the guard cells)",
    )
    pml_has_particles: bool | None = Field(
        default=None, description="Whether to allow particles in the PML region"
    )
    do_pml_j_damping: bool | None = Field(
        default=None, description="Whether to do damping of J in the PML"
    )

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        assert self.method is None or self.method in [
            "Yee",
            "CKC",
            "PSATD",
            "ECT",
        ], Exception("Only 'Yee', 'CKC', 'PSATD', and 'ECT' are supported")

    def solver_initialize_inputs(self):
        self.grid.grid_initialize_inputs()

        pywarpx.warpx.pml_ncell = self.pml_ncell

        if self.method == "PSATD":
            pywarpx.psatd.periodic_single_box_fft = self.psatd_periodic_single_box_fft
            pywarpx.psatd.current_correction = self.psatd_current_correction
            pywarpx.psatd.update_with_rho = self.psatd_update_with_rho
            pywarpx.psatd.do_time_averaging = self.psatd_do_time_averaging
            pywarpx.psatd.JRhom = self.psatd_JRhom

            if self.grid.guard_cells is not None:
                pywarpx.psatd.nx_guard = self.grid.guard_cells[0]
                if self.grid.number_of_dimensions == 3:
                    pywarpx.psatd.ny_guard = self.grid.guard_cells[1]
                pywarpx.psatd.nz_guard = self.grid.guard_cells[-1]

            if self.stencil_order is not None:
                pywarpx.psatd.nox = self.stencil_order[0]
                if self.grid.number_of_dimensions == 3:
                    pywarpx.psatd.noy = self.stencil_order[1]
                pywarpx.psatd.noz = self.stencil_order[-1]

            if self.galilean_velocity is not None:
                if self.grid.number_of_dimensions == 2:
                    self.galilean_velocity = [
                        self.galilean_velocity[0],
                        0.0,
                        self.galilean_velocity[1],
                    ]
                pywarpx.psatd.v_galilean = (
                    np.array(self.galilean_velocity) / constants.c
                )

        # --- Same method names are used, though mapped to lower case.
        pywarpx.algo.maxwell_solver = self.method

        pywarpx.warpx.cfl = self.cfl

        if self.source_smoother is not None:
            self.source_smoother.smoother_initialize_inputs(self)

        pywarpx.warpx.do_dive_cleaning = self.divE_cleaning
        pywarpx.warpx.do_divb_cleaning = self.divB_cleaning

        pywarpx.warpx.do_pml_dive_cleaning = self.pml_divE_cleaning
        pywarpx.warpx.do_pml_divb_cleaning = self.pml_divB_cleaning

        pywarpx.warpx.do_pml_in_domain = self.do_pml_in_domain
        pywarpx.warpx.pml_has_particles = self.pml_has_particles
        pywarpx.warpx.do_pml_j_damping = self.do_pml_j_damping


class EvolveSchemeBase(picmistandard.PICMI_Extension):
    """Base class of the evolve schemes, accepted as ``Simulation.warpx_evolve_scheme``"""

    def solver_scheme_initialize_inputs(self):
        raise NotImplementedError


class ExplicitEvolveScheme(EvolveSchemeBase):
    """
    Sets up the explicit evolve scheme
    """

    def solver_scheme_initialize_inputs(self):
        pywarpx.algo.evolve_scheme = "explicit"


class LinearSolverBase(picmistandard.PICMI_Extension):
    """Base class of the linear solvers"""

    def linear_solver_initialize_inputs(self, nonlinear_solver=None):
        raise NotImplementedError


class PreconditionerBase(picmistandard.PICMI_Extension):
    """Base class of the preconditioners"""

    # Name of the WarpX preconditioner type, set by subclasses.
    name: ClassVar[str | None] = None

    # Whether this preconditioner can be selected directly on a linear
    # solver (rather than only via a nonlinear solver's Jacobian).
    supports_direct_gmres: ClassVar[bool] = False

    def preconditioner_type_initialize_inputs(self, jacobian=None):
        if jacobian is not None:
            jacobian.pc_type = self.name
        bucket = pywarpx.warpx.get_bucket(self.name)
        for attr in type(self).model_fields:
            setattr(bucket, attr, getattr(self, attr))


class GMRESLinearSolver(LinearSolverBase):
    """
    Sets up the iterative GMRES linear solver for the implicit Newton nonlinear solver
    """

    verbose_int: int | None = Field(
        default=None, description="Level of verbosity of output (default 2)"
    )
    restart_length: int | None = Field(
        default=None,
        description="How often to restart the GMRES iterations (default 30)",
    )
    absolute_tolerance: float | None = Field(
        default=None, description="Absolute tolerance of the convergence (default 0.)"
    )
    relative_tolerance: float | None = Field(
        default=None,
        description="Relative tolerance of the convergence (default 1.e-4)",
    )
    max_iterations: int | None = Field(
        default=None, description="Maximum number of iterations (default 1000)"
    )
    pc_type: PreconditionerBase | None = Field(
        default=None,
        description="The preconditioner applied inside the GMRES iterations. This is only used by solvers that drive GMRES directly rather than through a nonlinear solver (currently the semi-implicit Darwin solver, which supports an instance of DarwinMLMGPreconditioner); with a nonlinear solver, pass the preconditioner to that solver instead.",
    )

    @field_validator("pc_type")
    @classmethod
    def _check_direct_gmres_support(cls, pc_type):
        if pc_type is not None and not pc_type.supports_direct_gmres:
            raise ValueError(
                f"{type(pc_type).__name__} cannot be selected directly on the "
                "GMRES solver; pass it to the nonlinear solver instead"
            )
        return pc_type

    def linear_solver_initialize_inputs(self, nonlinear_solver=None):
        if nonlinear_solver is not None:
            nonlinear_solver.linear_solver = "amrex_gmres"
        amrex_gmres = pywarpx.warpx.get_bucket("amrex_gmres")
        amrex_gmres.verbose_int = self.verbose_int
        amrex_gmres.restart_length = self.restart_length
        amrex_gmres.absolute_tolerance = self.absolute_tolerance
        amrex_gmres.relative_tolerance = self.relative_tolerance
        amrex_gmres.max_iterations = self.max_iterations

        if self.pc_type is not None:
            amrex_gmres.pc_type = self.pc_type.name
            self.pc_type.preconditioner_type_initialize_inputs()


class PETScKSPLinearSolver(LinearSolverBase):
    """
    Sets up the petsc_ksp linear solver for the implicit Newton nonlinear solver
    """

    def linear_solver_initialize_inputs(self, nonlinear_solver=None):
        if nonlinear_solver is not None:
            nonlinear_solver.linear_solver = "petsc_ksp"


class CurlCurlMLMGPreconditioner(PreconditionerBase):
    """
    Sets up the curl-curl multigrid preconditioner used during the nonlinear solver
    """

    name: ClassVar[str | None] = "pc_curl_curl_mlmg"

    verbose: bool | None = Field(
        default=None, description="Whether there is verbose output from the solver"
    )
    bottom_verbose: bool | None = Field(
        default=None,
        description="Whether there is verbose output from the bottom solver",
    )
    agglomeration: bool | None = Field(
        default=None, description="Whether to use agglomeration"
    )
    consolidation: bool | None = Field(
        default=None, description="Whether to use consolidation"
    )
    max_iter: int | None = Field(
        default=None, description="Maximum number of iterations"
    )
    max_coarsening_level: int | None = Field(
        default=None, description="Maximum coarsening level"
    )
    relative_tolerance: float | None = Field(
        default=None, description="Relative tolerance of the convergence"
    )
    absolute_tolerance: float | None = Field(
        default=None, description="Absolute tolerance of the convergence"
    )


class DarwinMLMGPreconditioner(PreconditionerBase):
    """
    Sets up the factored-Laplacian multigrid preconditioner for the
    semi-implicit Darwin solver's GMRES iteration. Approximates the Darwin
    field operator by its constant-susceptibility factorization
    (-nabla^2)(-nabla^2 + chi) and applies it as two successive scalar
    multigrid solves (Poisson then Helmholtz with the spatially varying
    susceptibility) per vector component.
    """

    name: ClassVar[str | None] = "pc_darwin_mlmg"
    supports_direct_gmres: ClassVar[bool] = True

    verbose: bool | None = Field(
        default=None,
        description="Whether there is verbose output from the solver (default False)",
    )
    bottom_verbose: bool | None = Field(
        default=None,
        description="Whether there is verbose output from the bottom solver",
    )
    agglomeration: bool | None = Field(
        default=None, description="Whether to use agglomeration"
    )
    consolidation: bool | None = Field(
        default=None, description="Whether to use consolidation"
    )
    max_iter: int | None = Field(
        default=None,
        description="The fixed number of V-cycles used for each of the two multigrid solves per component (fixed so the preconditioner is a fixed linear operator across a GMRES solve) (default 2)",
    )
    max_coarsening_level: int | None = Field(
        default=None, description="Maximum coarsening level"
    )
    relative_tolerance: float | None = Field(
        default=None, description="Relative tolerance of the convergence"
    )
    absolute_tolerance: float | None = Field(
        default=None, description="Absolute tolerance of the convergence"
    )


class JacobiPreconditioner(PreconditionerBase):
    """
    Sets up the point Jacobi preconditioner used during the nonlinear solver
    """

    name: ClassVar[str | None] = "pc_jacobi"

    verbose: bool | None = Field(
        default=None, description="Whether there is verbose output from the solver"
    )
    max_iter: int | None = Field(
        default=None, description="Maximum number of iterations"
    )
    relative_tolerance: float | None = Field(
        default=None, description="Relative tolerance of the convergence"
    )
    absolute_tolerance: float | None = Field(
        default=None, description="Absolute tolerance of the convergence"
    )


class PETScPreconditioner(PreconditionerBase):
    """
    Sets up the PETSc preconditioner used during the nonlinear solver
    """

    name: ClassVar[str | None] = "pc_petsc"

    type: Literal["lu", "asm", "hypre"] | None = Field(
        default=None, description='PETSc solver type, one of "lu", "asm", or "hypre"'
    )
    asm_overlap: int | None = Field(
        default=None, description='Parameter for type is "asm"'
    )
    sub_type: Literal["ilu", "lu"] | None = Field(
        default=None,
        description='When type is "asm", one of "ilu" or "lu", default "ilu"',
    )
    ilu_factor_levels: int | None = Field(
        default=None, description='When type is "asm", and sub_type is "ilu"'
    )
    hypre_type: str | None = Field(
        default=None, description='When type is "hypre", default "euclid"'
    )
    euclid_factor_levels: int | None = Field(
        default=None, description='When type is "hypre" and hypre_type is "euclid"'
    )


class NonlinearSolverBase(picmistandard.PICMI_Extension):
    """Base class of the nonlinear solvers"""

    def nonlinear_solver_initialize_inputs(self):
        raise NotImplementedError


class NewtonNonlinearSolver(NonlinearSolverBase):
    """
    Sets up the iterative Newton nonlinear solver for the implicit evolve scheme
    """

    verbose: bool | None = Field(
        default=None,
        description="Whether there is verbose output from the solver (default True)",
    )
    linear_solver: LinearSolverBase | None = Field(
        default=None, description="Specifies input arguments to the linear solver"
    )
    require_convergence: bool | None = Field(
        default=None,
        description="Whether convergence is required. If True and convergence is not obtained, the code will exit. (default True)",
    )
    max_iterations: int | None = Field(
        default=None, description="Maximum number of iterations (default 100)"
    )
    relative_tolerance: float | None = Field(
        default=None,
        description="Relative tolerance of the convergence (default 1.e-6)",
    )
    absolute_tolerance: float | None = Field(
        default=None, description="Absolute tolerance of the convergence (default 0.)"
    )
    diagnostic_file: str | None = Field(
        default=None, description="File name where solver diagnostics are written"
    )
    diagnostic_interval: int | str | None = Field(
        default=None,
        description="The intervals for writing out solver diagnostics to the diagnostic file",
    )
    max_particle_iterations: int | None = Field(
        default=None, description="The maximum number of particle iterations"
    )
    particle_tolerance: float | None = Field(
        default=None,
        description="The tolerance of particle quantities for convergence",
    )
    particle_suborbits: bool | None = Field(
        default=None, description="Whether to use particle suborbits during the solve"
    )
    print_unconverged_particle_detail: bool | None = Field(
        default=None,
        description="Whether to print the details of unconverged particles during suborbits",
    )
    use_mass_matrices_jacobian: bool | None = Field(
        default=None,
        description="Whether to use mass-matrices during the linear stage of PS-JFNK",
    )
    skip_particle_picard_init: bool | None = Field(
        default=None,
        description="When use_mass_matrices_jacobian is True, whether to skip the particle picard iteration on the initial Newton step",
    )
    use_mass_matrices_pc: bool | None = Field(
        default=None,
        description="Whether to capture the plasma response in the preconditioner",
    )
    mass_matrices_pc_width: int | None = Field(
        default=None,
        description="When use_mass_matrices_pc is True, the width of the preconditioner mass matrices",
    )
    pc_type: PreconditionerBase | None = Field(
        default=None,
        description="The preconditioner type, An instance of either CurlCurlMLMGPreconditioner, JacobiPreconditioner, or PETScPreconditioner",
    )

    def nonlinear_solver_initialize_inputs(self):
        implicit_evolve = pywarpx.warpx.get_bucket("implicit_evolve")
        implicit_evolve.nonlinear_solver = "newton"
        implicit_evolve.max_particle_iterations = self.max_particle_iterations
        implicit_evolve.particle_tolerance = self.particle_tolerance
        implicit_evolve.particle_suborbits = self.particle_suborbits
        implicit_evolve.print_unconverged_particle_detail = (
            self.print_unconverged_particle_detail
        )
        implicit_evolve.use_mass_matrices_jacobian = self.use_mass_matrices_jacobian
        implicit_evolve.skip_particle_picard_init = self.skip_particle_picard_init
        implicit_evolve.use_mass_matrices_pc = self.use_mass_matrices_pc
        implicit_evolve.mass_matrices_pc_width = self.mass_matrices_pc_width

        newton = pywarpx.warpx.get_bucket("newton")
        newton.verbose = self.verbose
        newton.absolute_tolerance = self.absolute_tolerance
        newton.relative_tolerance = self.relative_tolerance
        newton.max_iterations = self.max_iterations
        newton.require_convergence = self.require_convergence
        newton.diagnostic_file = self.diagnostic_file
        newton.diagnostic_interval = self.diagnostic_interval

        if self.linear_solver is not None:
            self.linear_solver.linear_solver_initialize_inputs(newton)

        if self.pc_type is not None:
            jacobian = pywarpx.warpx.get_bucket("jacobian")
            self.pc_type.preconditioner_type_initialize_inputs(jacobian)


class PicardNonlinearSolver(NonlinearSolverBase):
    """
    Sets up the iterative Picard nonlinear solver for the implicit evolve scheme
    """

    verbose: bool | None = Field(
        default=None,
        description="Whether there is verbose output from the solver (default True)",
    )
    require_convergence: bool | None = Field(
        default=None,
        description="Whether convergence is required. If True and convergence is not obtained, the code will exit. (default True)",
    )
    max_iterations: int | None = Field(
        default=None, description="Maximum number of iterations (default 100)"
    )
    relative_tolerance: float | None = Field(
        default=None,
        description="Relative tolerance of the convergence (default 1.e-6)",
    )
    absolute_tolerance: float | None = Field(
        default=None, description="Absolute tolerance of the convergence (default 0.)"
    )
    diagnostic_file: str | None = Field(
        default=None, description="File name where solver diagnostics are written"
    )
    diagnostic_interval: int | str | None = Field(
        default=None,
        description="The intervals for writing out solver diagnostics to the diagnostic file",
    )

    def nonlinear_solver_initialize_inputs(self):
        implicit_evolve = pywarpx.warpx.get_bucket("implicit_evolve")
        implicit_evolve.nonlinear_solver = "picard"

        picard = pywarpx.warpx.get_bucket("picard")
        picard.verbose = self.verbose
        picard.require_convergence = self.require_convergence
        picard.max_iterations = self.max_iterations
        picard.relative_tolerance = self.relative_tolerance
        picard.absolute_tolerance = self.absolute_tolerance
        picard.diagnostic_file = self.diagnostic_file
        picard.diagnostic_interval = self.diagnostic_interval


class ThetaImplicitEMEvolveScheme(EvolveSchemeBase):
    """
    Sets up the "theta implicit" electromagnetic evolve scheme
    """

    nonlinear_solver: NonlinearSolverBase = Field(
        description="The nonlinear solver to use for the iterations"
    )
    theta: float | None = Field(
        default=None,
        description='The "theta" parameter, determining the level of implicitness',
    )

    def solver_scheme_initialize_inputs(self):
        pywarpx.algo.evolve_scheme = "theta_implicit_em"
        implicit_evolve = pywarpx.warpx.get_bucket("implicit_evolve")
        implicit_evolve.theta = self.theta

        self.nonlinear_solver.nonlinear_solver_initialize_inputs()


class SemiImplicitEMEvolveScheme(EvolveSchemeBase):
    """
    Sets up the "semi-implicit" electromagnetic evolve scheme
    """

    nonlinear_solver: NonlinearSolverBase = Field(
        description="The nonlinear solver to use for the iterations"
    )

    def solver_scheme_initialize_inputs(self):
        pywarpx.algo.evolve_scheme = "semi_implicit_em"

        self.nonlinear_solver.nonlinear_solver_initialize_inputs()


class SemiImplicitDarwinEvolveScheme(EvolveSchemeBase):
    """
    Sets up the semi-implicit Darwin evolve scheme.
    """

    # There is no nonlinear solver for the linear solver to attach to, which
    # PETScKSPLinearSolver requires.
    linear_solver: GMRESLinearSolver = Field(description="The GMRES linear solver")

    def solver_scheme_initialize_inputs(self):
        pywarpx.algo.evolve_scheme = "semi_implicit_darwin"
        self.linear_solver.linear_solver_initialize_inputs()


class HybridPICSolver(
    picmistandard.PICMI_SolverExtension, picmistandard.PICMI_ExpressionParameters
):
    """
    Hybrid-PIC solver based on Ohm's law.
    See `Theory Section <https://warpx.readthedocs.io/en/latest/theory/kinetic_fluid_hybrid_model.html>`_ for more information.

    Parameters used in the expressions can be given as additional keyword arguments.

    Notes
    -----
    **Required Parameters:**

    - ``Te`` must be specified when using the hybrid solver.
    - ``n0`` should be specified if ``gamma != 1``.

    **Best Practices:**

    - *Grid type:* Setting ``warpx_grid_type='collocated'`` is recommended.
    - *Particle shape:* Linear particles (``algo.particle_shape = 1``) are recommended.

    **Constraints and Limitations:**

    - *Mesh refinement:* Only one level is supported (no AMR). The solver will abort if more than one level is used.
    - *RZ geometry:* Only the m=0 azimuthal mode is supported in RZ geometry.
    - *External vector potential:* If ``A_external`` is provided, it must be non-empty.
    - *Time-dependent A fields:* When using expressions for external vector potentials, time variation must be specified via ``A_time_external_function``, not directly in the ``A[x,y,z]_external_function`` expressions.

    For complete parameter documentation, see the `Input Parameters section <https://warpx.readthedocs.io/en/latest/usage/parameters.html#maxwell-solver-kinetic-fluid-hybrid>`_.
    """

    method: ClassVar[str] = "hybrid"
    _expression_fields: ClassVar[tuple[str, ...]] = (
        "plasma_resistivity",
        "plasma_hyper_resistivity",
        "plasma_resistivity_species",
        "electron_ion_relaxation_rate",
        "Jx_external_function",
        "Jy_external_function",
        "Jz_external_function",
        "A_external",
    )

    grid: picmistandard.PICMI_AnyGrid = Field(description="Grid object for the solver")
    Te: float | None = Field(default=None, description="Electron temperature in eV.")
    n0: float | None = Field(
        default=None, description="Reference plasma density in m^-3."
    )
    gamma: float | None = Field(
        default=None,
        description="Exponent in calculation of electron pressure (default 5/3).",
    )
    n_floor: float | None = Field(
        default=None, description="Minimum density used in Ohm's law calculation."
    )
    plasma_resistivity: float | str | None = Field(
        default=None,
        description="Value or expression to use for the plasma resistivity in Ohm*m. Can be a constant value or an expression depending on ``rho`` (charge density), ``J`` (current density magnitude), and ``t`` (simulation time).",
    )
    plasma_hyper_resistivity: float | str | None = Field(
        default=None,
        description="Value or expression to use for the plasma hyper-resistivity in Ohm*m^3. Can be a constant value or an expression depending on ``rho`` (charge density) and ``B`` (magnetic field magnitude).",
    )
    plasma_resistivity_species: dict[str, float | str] | None = Field(
        default=None,
        description="Per-species resistivity overlays added on top of ``plasma_resistivity``, as a dictionary mapping a charged species name to a value or expression in Ohm*m. The expression may depend on ``rho_s`` (the species charge density), ``rho`` (total charge density which, by quasineutrality, equals the electron charge density), ``Te`` (electron temperature in Kelvin), ``J`` (plasma current density magnitude), ``J_s`` (the species current density magnitude), ``B`` (magnetic field magnitude) and ``t`` (time). The effective resistivity applied to species ``s`` in Ohm's law, the Joule heating source and the resistive drag is ``plasma_resistivity + plasma_resistivity_species[s]``.",
    )
    solve_electron_energy_equation: bool | None = Field(
        default=None,
        description="Solve the electron energy equation instead of the algebraic adiabatic pressure closure: the electron entropy ``K = Te * ne**(1-gamma)`` is transported each step by QDSMC markers advected with the electron fluid velocity, the source terms below are applied per cell, and ``Pe = ne * kB * Te`` is fed back into the Ohm's-law E-solve. (default False)",
    )
    include_joule_heating: bool | None = Field(
        default=None,
        description="Add the resistive (Joule) heating source to the electron temperature. Reduces to ``eta * J**2`` for a single ion species. Only used when ``solve_electron_energy_equation`` is True. (default False)",
    )
    joule_redirect_Te_threshold: float | None = Field(
        default=None,
        description="Electron temperature threshold in eV above which the Joule heating of a cell is routed to the ions (as an energy-conserving stochastic kick) instead of the electrons, allowing ``Ti > Te`` to develop. Specifying a value >= 0 enables the redirect (off by default). Requires ``include_joule_heating``.",
    )
    electron_ion_relaxation_rate: float | str | None = Field(
        default=None,
        description="Value or expression for the electron-ion energy-equilibration rate ``nu_ei`` in 1/s. Specifying it enables the electron-ion thermal equilibration ``Q_ei`` on the electron temperature, with the conjugate ion heating applied as an energy-conserving drag-diffusion kick on each ion (the required shape-aware ion temperature deposition is enabled automatically on every charged species). The expression may depend on ``rho`` (charge density in C/m^3), ``Te`` and ``Ti`` (temperatures in eV) and ``t`` (time). Only used when ``solve_electron_energy_equation`` is True.",
    )
    substeps: int | None = Field(
        default=None,
        description="Total number of substeps used to advance the B-field over one full timestep (split evenly between the two half-steps, so ``substeps/2`` RK4 steps are taken per half-step, each of duration ``dt / substeps``). Must be divisible by 2; if not, the value is automatically rounded up to the next even number. When ``use_rkf45`` is active (True or a non-empty interval string), this is instead used only as the initial substep count estimate for the adaptive solver. After each timestep on which ``use_rkf45`` is active, this value is updated based on ``n_attempts``, the total number of RKF45 sub-step attempts (accepted and rejected) taken in the most recent half-step: if the current value is less than ``2 * n_attempts``, it jumps immediately to ``2 * n_attempts``; otherwise it decays slowly toward that target via exponential smoothing (95% old, 5% of ``2 * n_attempts``). This warm-start guess also carries over to RK4 steps on timesteps where ``use_rkf45`` is not active. (default 10)",
    )
    use_rkf45: bool | str | None = Field(
        default=None,
        description='If True (or the WarpX time-interval string ``"::"``), use the adaptive Runge-Kutta-Fehlberg 4(5) (RKF45) integrator (Fehlberg 1969, NASA Technical Report R-315, https://ntrs.nasa.gov/citations/19690021375) for the B-field substep advance, with step-size control governed by ``substep_rtol`` and ``substep_atol``. If False, use the fixed-step classical RK4 integrator with ``substeps`` total substeps per timestep. A WarpX time-interval string (e.g. ``"1::5"`` to enable from step every 5 steps starting from step 1) may also be passed to activate RKF45 only on specific timesteps. (default False)',
    )
    substep_rtol: float | None = Field(
        default=None,
        description="Relative tolerance for the RKF45 adaptive step-size control. Only used when ``use_rkf45`` is active. (default 1e-4)",
    )
    substep_atol: float | None = Field(
        default=None,
        description="Absolute tolerance for the RKF45 adaptive step-size control. Only used when ``use_rkf45`` is active. (default 1e-8)",
    )
    substep_safety: float | None = Field(
        default=None,
        description="Safety factor applied to the step-size adjustment formula. Only used when ``use_rkf45`` is active. (default 0.9)",
    )
    substep_max_growth: float | None = Field(
        default=None,
        description="Maximum factor by which the substep size may grow after an accepted step. Only used when ``use_rkf45`` is active. (default 5.0)",
    )
    max_substep_attempts: int | None = Field(
        default=None,
        description="Maximum number of substep attempts (accepted + rejected combined) per half-step before the simulation aborts. Only used when ``use_rkf45`` is active. (default 250)",
    )
    holmstrom_vacuum_region: bool | None = Field(
        default=None,
        description="Flag to determine handling of vacuum region (where rho < n_floor*q_e). Setting to True will solve the simplified Generalized Ohm's Law dropping the Hall and pressure terms in the vacuum region. See `Holmstrom (2013) <https://arxiv.org/abs/1301.0272v1>`_. This flag is useful for suppressing vacuum region fluctuations. A large resistivity value must be used when rho <= rho_floor. (default False)",
    )
    vacuum_seam_switch_mode: Literal["edge", "node", "cell"] | None = Field(
        default=None,
        description='Sampling of the density that decides the vacuum-seam treatment -- the Holmstrom vacuum branch when holmstrom_vacuum_region is True and the density-floor selection of the guarded Hall term otherwise: "edge" (legacy per-component edge average), "node" (endpoint minimum), or "cell" (adjacent-cell minimum -- one decision for all three E components of an index, removing the per-component half-cell decision offsets at the plasma/vacuum seam). Cartesian only. (default "edge")',
    )
    Jx_external_function: float | str | None = Field(
        default=None,
        description="Function of space and time specifying external (non-plasma) currents.",
    )
    Jy_external_function: float | str | None = Field(
        default=None,
        description="Function of space and time specifying external (non-plasma) currents.",
    )
    Jz_external_function: float | str | None = Field(
        default=None,
        description="Function of space and time specifying external (non-plasma) currents.",
    )
    A_external: dict[str, dict[str, float | str | bool]] | None = Field(
        default=None,
        description="Function of space and time specifying external (non-plasma) vector potential fields. It is expected that a nested dictionary will be passed in for each separate vector potential that may have different spatial configuration or time dependence. Each field entry should contain either implicit functions with (x,y,z) dependence for 'Ax_external_function', 'Ay_external_function', 'Az_external_function', plus 'A_time_external_function' with (t) dependence, or alternatively 'load_from_file': True with a 'path' to an OpenPMD file along with 'A_time_external_function'.",
    )
    do_external_diva_cleaning: bool | None = Field(
        default=None,
        description="This flag can be used to disable divA cleaning. This may be necessary when using a non-periodic external A with periodic field boundary conditions. (default True)",
    )
    user_defined_kw: dict = Field(
        default_factory=dict,
        description="Constants referenced in the expressions, collected from otherwise-unrecognized keyword arguments.",
    )

    _mangle_dict: dict | None = PrivateAttr(default=None)

    def solver_initialize_inputs(self):
        # Add the user defined keywords to my_constants
        # The keywords are mangled if there is a conflicting variable already
        # defined in my_constants with the same name but different value.
        self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)

        self.grid.grid_initialize_inputs()

        pywarpx.algo.maxwell_solver = self.method

        pywarpx.hybridpicmodel.elec_temp = self.Te
        pywarpx.hybridpicmodel.n0_ref = self.n0
        pywarpx.hybridpicmodel.gamma = self.gamma
        pywarpx.hybridpicmodel.n_floor = self.n_floor
        pywarpx.hybridpicmodel.__setattr__(
            "plasma_resistivity(rho,J,t)",
            pywarpx.my_constants.mangle_expression(
                self.plasma_resistivity, self._mangle_dict
            ),
        )
        pywarpx.hybridpicmodel.__setattr__(
            "plasma_hyper_resistivity(rho,B)",
            pywarpx.my_constants.mangle_expression(
                self.plasma_hyper_resistivity, self._mangle_dict
            ),
        )
        if self.plasma_resistivity_species is not None:
            for name, expr in self.plasma_resistivity_species.items():
                pywarpx.hybridpicmodel.__setattr__(
                    f"plasma_resistivity_{name}(rho_s,rho,Te,J,J_s,B,t)",
                    pywarpx.my_constants.mangle_expression(expr, self._mangle_dict),
                )
        # Only emit the electron-energy-equation attributes that were
        # explicitly set, so the generated input deck contains only
        # user-specified parameters.
        if self.solve_electron_energy_equation is not None:
            pywarpx.hybridpicmodel.solve_electron_energy_equation = (
                self.solve_electron_energy_equation
            )
        if self.include_joule_heating is not None:
            pywarpx.hybridpicmodel.include_joule_heating = self.include_joule_heating
        if self.joule_redirect_Te_threshold is not None:
            pywarpx.hybridpicmodel.joule_redirect_Te_threshold = (
                self.joule_redirect_Te_threshold
            )
        if self.electron_ion_relaxation_rate is not None:
            pywarpx.hybridpicmodel.__setattr__(
                "electron_ion_relaxation_rate(rho,Te,Ti,t)",
                pywarpx.my_constants.mangle_expression(
                    self.electron_ion_relaxation_rate, self._mangle_dict
                ),
            )
        pywarpx.hybridpicmodel.substeps = self.substeps
        pywarpx.hybridpicmodel.use_rkf45 = self.use_rkf45
        pywarpx.hybridpicmodel.substep_rtol = self.substep_rtol
        pywarpx.hybridpicmodel.substep_atol = self.substep_atol
        pywarpx.hybridpicmodel.substep_safety = self.substep_safety
        pywarpx.hybridpicmodel.substep_max_growth = self.substep_max_growth
        pywarpx.hybridpicmodel.max_substep_attempts = self.max_substep_attempts
        pywarpx.hybridpicmodel.holmstrom_vacuum_region = self.holmstrom_vacuum_region
        pywarpx.hybridpicmodel.vacuum_seam_switch_mode = self.vacuum_seam_switch_mode
        pywarpx.hybridpicmodel.__setattr__(
            "Jx_external_grid_function(x,y,z,t)",
            pywarpx.my_constants.mangle_expression(
                self.Jx_external_function, self._mangle_dict
            ),
        )
        pywarpx.hybridpicmodel.__setattr__(
            "Jy_external_grid_function(x,y,z,t)",
            pywarpx.my_constants.mangle_expression(
                self.Jy_external_function, self._mangle_dict
            ),
        )
        pywarpx.hybridpicmodel.__setattr__(
            "Jz_external_grid_function(x,y,z,t)",
            pywarpx.my_constants.mangle_expression(
                self.Jz_external_function, self._mangle_dict
            ),
        )
        if self.A_external is not None:
            pywarpx.hybridpicmodel.add_external_fields = True
            pywarpx.external_vector_potential.__setattr__(
                "fields",
                pywarpx.my_constants.mangle_expression(
                    list(self.A_external.keys()), self._mangle_dict
                ),
            )
            pywarpx.external_vector_potential.do_diva_cleaning = (
                self.do_external_diva_cleaning
            )
            for field_name, field_dict in self.A_external.items():
                if field_dict.get("read_from_file", False):
                    pywarpx.external_vector_potential.__setattr__(
                        f"{field_name}.read_from_file", field_dict["read_from_file"]
                    )
                    pywarpx.external_vector_potential.__setattr__(
                        f"{field_name}.path", field_dict["path"]
                    )
                else:
                    pywarpx.external_vector_potential.__setattr__(
                        f"{field_name}.Ax_external_grid_function(x,y,z)",
                        pywarpx.my_constants.mangle_expression(
                            field_dict["Ax_external_function"], self._mangle_dict
                        ),
                    )
                    pywarpx.external_vector_potential.__setattr__(
                        f"{field_name}.Ay_external_grid_function(x,y,z)",
                        pywarpx.my_constants.mangle_expression(
                            field_dict["Ay_external_function"], self._mangle_dict
                        ),
                    )
                    pywarpx.external_vector_potential.__setattr__(
                        f"{field_name}.Az_external_grid_function(x,y,z)",
                        pywarpx.my_constants.mangle_expression(
                            field_dict["Az_external_function"], self._mangle_dict
                        ),
                    )
                pywarpx.external_vector_potential.__setattr__(
                    f"{field_name}.A_time_external_function(t)",
                    pywarpx.my_constants.mangle_expression(
                        field_dict["A_time_external_function"], self._mangle_dict
                    ),
                )


class ElectrostaticSolver(picmistandard.PICMI_ElectrostaticSolver):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.

    The standard PICMI parameters `required_precision` and `maximum_iterations` control the
    MLMG Poisson solver convergence for the labframe electrostatic solvers. When `warpx_magnetostatic=True`,
    these parameters are used as defaults for the magnetostatic solver but can be overridden
    with the explicit `warpx_magnetostatic_*` parameters.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_ElectrostaticSolver)
    )

    relativistic: bool = Field(
        default=False,
        description="Whether to use the relativistic solver or lab frame solver",
    )
    absolute_tolerance: float | None = Field(
        default=None,
        description="Absolute tolerance on the labframe electrostatic solver (default 0.)",
    )
    self_fields_verbosity: int | None = Field(
        default=None,
        description="Level of verbosity for the labframe electrostatic solver (default 2)",
    )
    magnetostatic: bool = Field(
        default=False,
        description="Whether to also solve for self-consistent magnetic fields from currents.",
    )
    # Explicit magnetostatic solver parameters (override self_fields_* defaults)
    magnetostatic_required_precision: float | None = Field(
        default=None,
        description="Relative precision for the magnetostatic solver. If not specified, defaults to the value of `required_precision`.",
    )
    magnetostatic_absolute_tolerance: float | None = Field(
        default=None,
        description="Absolute tolerance for the magnetostatic solver. If not specified, defaults to the value of `warpx_absolute_tolerance`.",
    )
    magnetostatic_max_iters: int | None = Field(
        default=None,
        description="Maximum iterations for the magnetostatic solver. If not specified, defaults to the value of `maximum_iterations`.",
    )
    magnetostatic_verbosity: int | None = Field(
        default=None,
        description="Verbosity level for the magnetostatic solver. If not specified, defaults to the value of `warpx_self_fields_verbosity`.",
    )
    effective_potential: bool = Field(
        default=False,
        description="Whether to use the effective potential Poisson solver (EP-PIC)",
    )
    effective_potential_factor: float | None = Field(
        default=None,
        description="If the effective potential Poisson solver is used, this sets the value of C_EP (the method is marginally stable at C_EP = 1) (default 4)",
    )
    effective_potential_time_filter_param: float | None = Field(
        default=None,
        description="Time filtering parameter used to filter sigma in the effective potential scheme. sigma is updated using: sigma^n = warpx_effective_potential_time_filter_param * sigma^n + (1 - warpx_effective_potential_time_filter_param) * sigma^n-1 (default 0.1)",
    )
    effective_potential_density_floor: float | None = Field(
        default=None,
        description="If given, this value will be used as the minimum density during the local calculation of sigma. (default 0)",
    )
    dt_update_interval: int | str | None = Field(
        default=None,
        description="How frequently the timestep is updated. Adaptive timestepping is disabled when this is <= 0. (default -1)",
    )
    cfl: float | None = Field(
        default=None,
        description="Fraction of the CFL condition for particle velocity vs grid size, used to set the timestep when `warpx_dt_update_interval > 0`.",
    )
    max_dt: float | None = Field(
        default=None,
        description="The maximum allowable timestep when `warpx_dt_update_interval > 0`.",
    )

    def solver_initialize_inputs(self):
        # Open BC means FieldBoundaryType::Open for electrostatic sims, rather than perfectly-matched layer
        BC_map["open"] = "open"

        self.grid.grid_initialize_inputs()

        # set adaptive timestepping parameters
        pywarpx.warpx.cfl = self.cfl
        pywarpx.warpx.dt_update_interval = self.dt_update_interval
        pywarpx.warpx.max_dt = self.max_dt

        if self.relativistic:
            pywarpx.warpx.do_electrostatic = "relativistic"
        else:
            if self.magnetostatic:
                pywarpx.warpx.do_electrostatic = "labframe-electromagnetostatic"
            elif self.effective_potential:
                pywarpx.warpx.do_electrostatic = "labframe-effective-potential"
                pywarpx.warpx.effective_potential_factor = (
                    self.effective_potential_factor
                )
                pywarpx.warpx.effective_potential_time_filter_param = (
                    self.effective_potential_time_filter_param
                )
                pywarpx.warpx.effective_potential_density_floor = (
                    self.effective_potential_density_floor
                )
            else:
                pywarpx.warpx.do_electrostatic = "labframe"
            pywarpx.warpx.self_fields_required_precision = self.required_precision
            pywarpx.warpx.self_fields_absolute_tolerance = self.absolute_tolerance
            pywarpx.warpx.self_fields_max_iters = self.maximum_iterations
            pywarpx.warpx.self_fields_verbosity = self.self_fields_verbosity
            # Explicit magnetostatic solver parameters (if provided)
            pywarpx.warpx.magnetostatic_solver_required_precision = (
                self.magnetostatic_required_precision
            )
            pywarpx.warpx.magnetostatic_solver_absolute_tolerance = (
                self.magnetostatic_absolute_tolerance
            )
            pywarpx.warpx.magnetostatic_solver_max_iters = self.magnetostatic_max_iters
            pywarpx.warpx.magnetostatic_solver_verbosity = self.magnetostatic_verbosity
            pywarpx.boundary.potential_lo_x = self.grid.potential_xmin
            pywarpx.boundary.potential_lo_y = self.grid.potential_ymin
            pywarpx.boundary.potential_lo_z = self.grid.potential_zmin
            pywarpx.boundary.potential_hi_x = self.grid.potential_xmax
            pywarpx.boundary.potential_hi_y = self.grid.potential_ymax
            pywarpx.boundary.potential_hi_z = self.grid.potential_zmax

        pywarpx.warpx.poisson_solver = self.method


class GaussianLaser(picmistandard.PICMI_GaussianLaser):
    # Runtime state populated during laser_initialize_inputs.
    _laser: pywarpx.Bucket.Bucket | None = PrivateAttr(default=None)
    _laser_number: int | None = PrivateAttr(default=None)

    @property
    def laser(self):
        """The WarpX inputs of this laser (available after the inputs are initialized)"""
        return self._laser

    def laser_initialize_inputs(self):
        self._laser_number = len(pywarpx.lasers.names) + 1
        if self.name is None:
            self.name = "laser{}".format(self._laser_number)

        self._laser = pywarpx.Lasers.newlaser(self.name)

        self._laser.profile = "Gaussian"
        self._laser.wavelength = (
            self.wavelength
        )  # The wavelength of the laser (in meters)
        self._laser.e_max = self.E0  # Maximum amplitude of the laser field (in V/m)
        self._laser.polarization = (
            self.polarization_direction
        )  # The main polarization vector
        self._laser.profile_waist = self.waist  # The waist of the laser (in meters)
        self._laser.profile_duration = (
            self.duration
        )  # The duration of the laser (in seconds)
        self._laser.direction = self.propagation_direction
        self._laser.zeta = self.zeta
        self._laser.beta = self.beta
        self._laser.phi2 = self.phi2
        self._laser.phi0 = self.phi0

        self._laser.do_continuous_injection = self.fill_in


class AnalyticLaser(picmistandard.PICMI_AnalyticLaser):
    # Runtime state populated during laser_initialize_inputs.
    _laser: pywarpx.Bucket.Bucket | None = PrivateAttr(default=None)
    _laser_number: int | None = PrivateAttr(default=None)
    _mangle_dict: dict | None = PrivateAttr(default=None)

    @property
    def laser(self):
        """The WarpX inputs of this laser (available after the inputs are initialized)"""
        return self._laser

    def laser_initialize_inputs(self):
        self._laser_number = len(pywarpx.lasers.names) + 1
        if self.name is None:
            self.name = "laser{}".format(self._laser_number)

        self._laser = pywarpx.Lasers.newlaser(self.name)

        self._laser.profile = "parse_field_function"
        self._laser.wavelength = (
            self.wavelength
        )  # The wavelength of the laser (in meters)
        self._laser.e_max = self.Emax  # Maximum amplitude of the laser field (in V/m)
        self._laser.polarization = (
            self.polarization_direction
        )  # The main polarization vector
        self._laser.direction = self.propagation_direction
        self._laser.do_continuous_injection = self.fill_in

        if self._mangle_dict is None:
            # Only do this once so that the same variables are used in this distribution
            # is used multiple times
            self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)
        expression = pywarpx.my_constants.mangle_expression(
            self.field_expression, self._mangle_dict
        )
        self._laser.__setattr__("field_function(X,Y,t)", expression)


class LaserAntenna(picmistandard.PICMI_LaserAntenna):
    def laser_antenna_initialize_inputs(self, laser):
        laser._laser.position = self.position  # This point is on the laser plane
        if self.normal_vector is not None and not np.allclose(
            laser._laser.direction, self.normal_vector
        ):
            raise AttributeError(
                "The specified laser direction does not match the "
                "specified antenna normal."
            )
        # The plane normal direction, as a unit vector
        normal_vector = np.asarray(laser._laser.direction, dtype=float)
        normal_vector /= np.linalg.norm(normal_vector)
        if isinstance(laser, GaussianLaser):
            # Focal displacement from the antenna (in meters)
            laser._laser.profile_focal_distance = (
                (laser.focal_position[0] - self.position[0]) * normal_vector[0]
                + (laser.focal_position[1] - self.position[1]) * normal_vector[1]
                + (laser.focal_position[2] - self.position[2]) * normal_vector[2]
            )
            # The time at which the laser reaches its peak (in seconds)
            laser._laser.profile_t_peak = (
                (self.position[0] - laser.centroid_position[0]) * normal_vector[0]
                + (self.position[1] - laser.centroid_position[1]) * normal_vector[1]
                + (self.position[2] - laser.centroid_position[2]) * normal_vector[2]
            ) / constants.c


class LoadInitialField(picmistandard.PICMI_LoadGriddedField):
    """
    Field Initializer that loads the initial field from a file.
    """

    do_initial_div_cleaning: bool | None = Field(
        default=None,
        alias="warpx_do_initial_div_cleaning",
        description="Flag that controls whether or not to execute the Projection based B-field divergence cleaner. (default True)",
    )
    div_cleaner_atol: float | None = Field(
        default=None,
        alias="warpx_projection_div_cleaner_atol",
        description="Controls the absolute tolerance used in the divergence cleaner solve.",
    )
    div_cleaner_rtol: float | None = Field(
        default=None,
        alias="warpx_projection_div_cleaner_rtol",
        description="Controls the relative tolerance used in the divergence cleaner solve.",
    )

    def applied_field_initialize_inputs(self):
        pywarpx.warpx.read_fields_from_path = self.read_fields_from_path
        if self.load_E:
            pywarpx.warpx.E_ext_grid_init_style = "read_from_file"
        if self.load_B:
            pywarpx.warpx.B_ext_grid_init_style = "read_from_file"
            pywarpx.warpx.do_initial_div_cleaning = self.do_initial_div_cleaning
            pywarpx.warpx.add_new_group_attr(
                "projection_div_cleaner", "atol", self.div_cleaner_atol
            )
            pywarpx.warpx.add_new_group_attr(
                "projection_div_cleaner", "rtol", self.div_cleaner_rtol
            )


class LoadInitialFieldFromPython(picmistandard.PICMI_AppliedFieldExtension):
    """
    Field Initializer that takes a function handle to be registered as a callback.
    The function is expected to write the E and/or B fields into the
    fields.Bx/y/zFPExternalWrapper() multifab. The callback is installed
    in the beforeInitEsolve hook. This should operate identically to loading from
    a file.
    """

    load_from_python: Callable[[], Any] = Field(
        description="Function that is called to write the E and/or B fields"
    )
    load_E: bool = Field(
        default=True,
        description="E field is expected to be loaded in the registered callback.",
    )
    load_B: bool = Field(
        default=True,
        description="B field is expected to be loaded in the registered callback.",
    )
    do_initial_div_cleaning: bool | None = Field(
        default=None,
        alias="warpx_do_initial_div_cleaning",
        description="Flag that controls whether or not to execute the Projection based B-field divergence cleaner. (default True)",
    )
    div_cleaner_atol: float | None = Field(
        default=None,
        alias="warpx_projection_div_cleaner_atol",
        description="Controls the absolute tolerance used in the divergence cleaner solve.",
    )
    div_cleaner_rtol: float | None = Field(
        default=None,
        alias="warpx_projection_div_cleaner_rtol",
        description="Controls the relative tolerance used in the divergence cleaner solve.",
    )

    def applied_field_initialize_inputs(self):
        if self.load_E:
            pywarpx.warpx.E_ext_grid_init_style = "load_from_python"
        if self.load_B:
            pywarpx.warpx.B_ext_grid_init_style = "load_from_python"
            pywarpx.warpx.do_initial_div_cleaning = self.do_initial_div_cleaning
            pywarpx.warpx.add_new_group_attr(
                "projection_div_cleaner", "atol", self.div_cleaner_atol
            )
            pywarpx.warpx.add_new_group_attr(
                "projection_div_cleaner", "rtol", self.div_cleaner_rtol
            )

        pywarpx.callbacks.installloadExternalFields(self.load_from_python)


class AnalyticInitialField(picmistandard.PICMI_AnalyticAppliedField):
    """
    Field Initializer that takes an implicit function to be loaded as an initial E/B field.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_AnalyticAppliedField)
    )

    maxlevel_extEMfield_init: int | None = Field(
        default=None,
        description="Maximum mesh-refinement level up to which the external fields are loaded",
    )
    do_initial_div_cleaning: bool | None = Field(
        default=None,
        alias="warpx_do_initial_div_cleaning",
        description="Flag that controls whether or not to execute the Projection based B-field divergence cleaner. (default True)",
    )
    div_cleaner_atol: float | None = Field(
        default=None,
        alias="warpx_projection_div_cleaner_atol",
        description="Controls the absolute tolerance used in the divergence cleaner solve.",
    )
    div_cleaner_rtol: float | None = Field(
        default=None,
        alias="warpx_projection_div_cleaner_rtol",
        description="Controls the relative tolerance used in the divergence cleaner solve.",
    )

    _mangle_dict: dict | None = PrivateAttr(default=None)

    def applied_field_initialize_inputs(self):
        # Note that lower and upper_bound are not used by WarpX
        pywarpx.warpx.maxlevel_extEMfield_init = self.maxlevel_extEMfield_init

        if self._mangle_dict is None:
            # Only do this once so that the same variables are used in this distribution
            # is used multiple times
            self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)

        if (
            self.Ex_expression is not None
            or self.Ey_expression is not None
            or self.Ez_expression is not None
        ):
            pywarpx.warpx.E_ext_grid_init_style = "parse_e_ext_grid_function"
            for sdir, expression in zip(
                ["x", "y", "z"],
                [self.Ex_expression, self.Ey_expression, self.Ez_expression],
            ):
                expression = pywarpx.my_constants.mangle_expression(
                    expression, self._mangle_dict
                )
                pywarpx.warpx.__setattr__(
                    f"E{sdir}_external_grid_function(x,y,z)", expression
                )

        if (
            self.Bx_expression is not None
            or self.By_expression is not None
            or self.Bz_expression is not None
        ):
            pywarpx.warpx.B_ext_grid_init_style = "parse_b_ext_grid_function"
            for sdir, expression in zip(
                ["x", "y", "z"],
                [self.Bx_expression, self.By_expression, self.Bz_expression],
            ):
                expression = pywarpx.my_constants.mangle_expression(
                    expression, self._mangle_dict
                )
                pywarpx.warpx.__setattr__(
                    f"B{sdir}_external_grid_function(x,y,z)", expression
                )
            pywarpx.warpx.do_initial_div_cleaning = self.do_initial_div_cleaning
            pywarpx.warpx.add_new_group_attr(
                "projection_div_cleaner", "atol", self.div_cleaner_atol
            )
            pywarpx.warpx.add_new_group_attr(
                "projection_div_cleaner", "rtol", self.div_cleaner_rtol
            )


class LoadAppliedField(
    picmistandard.PICMI_LoadAppliedField, picmistandard.PICMI_ExpressionParameters
):
    """
    Load external electromagnetic fields (E and/or B) from an openPMD file and
    optionally apply a time-dependent scaling.

    Multiple external field maps are supported by adding several
    ``LoadAppliedField`` objects to the simulation via repeated calls to
    ``Simulation.add_applied_field(...)``. Each instance contributes one
    independently scaled field map; the resulting fields are summed by WarpX.

    Example (multiple applied fields)::

        applied_field1 = picmi.LoadAppliedField(
            read_fields_from_path="diags/Bfield_map",
            load_E=False,
            load_B=True,
            warpx_B_time_function="cos(omega*t)",
        )

        applied_field2 = picmi.LoadAppliedField(
            read_fields_from_path="diags/Bfield_map",
            load_E=False,
            load_B=True,
            warpx_B_time_function="cos(2*omega*t)",
        )

        sim.add_applied_field(applied_field1)
        sim.add_applied_field(applied_field2)

    Internally, each object registers a uniquely named external field entry
    (``particles.<name>.*``), ensuring that multiple applied fields compose
    without overwriting each other.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_LoadAppliedField)
    )
    _expression_fields: ClassVar[tuple[str, ...]] = (
        "E_time_function",
        "B_time_function",
    )

    E_time_function: Expression | None = Field(
        default=None,
        description='AMReX parser expression in variable ``t`` (seconds) scaling the file-loaded electric field uniformly in space and per level. Defaults to ``"1.0"`` if not given.',
    )
    B_time_function: Expression | None = Field(
        default=None,
        description='AMReX parser expression in variable ``t`` (seconds) scaling the file-loaded magnetic field uniformly in space and per level. Defaults to ``"1.0"`` if not given.',
    )
    do_initial_div_cleaning: bool | None = Field(
        default=None,
        alias="warpx_do_initial_div_cleaning",
        description="Flag that controls whether or not to execute the Projection based B-field divergence cleaner. (default True)",
    )
    div_cleaner_atol: float | None = Field(
        default=None,
        alias="warpx_projection_div_cleaner_atol",
        description="Controls the absolute tolerance used in the divergence cleaner solve.",
    )
    div_cleaner_rtol: float | None = Field(
        default=None,
        alias="warpx_projection_div_cleaner_rtol",
        description="Controls the relative tolerance used in the divergence cleaner solve.",
    )
    user_defined_kw: dict = Field(
        default_factory=dict,
        description="Constants referenced in the time functions, collected from otherwise-unrecognized keyword arguments.",
    )

    _auto_field_counter: ClassVar[int] = 0

    def _next_auto_name(self):
        LoadAppliedField._auto_field_counter += 1
        return f"ext_field{LoadAppliedField._auto_field_counter}"

    def _as_list(self, x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, str):
            return [s for s in x.split() if s]
        return [x]

    def _append_names(self, list_key, new_names):
        existing = None
        try:
            existing = getattr(pywarpx.particles, list_key)
        except Exception:
            existing = None
        existing = self._as_list(existing)
        for n in new_names:
            if n not in existing:
                existing.append(n)
        pywarpx.particles.__setattr__(list_key, existing)

    def applied_field_initialize_inputs(self):
        mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)

        if not (self.load_E or self.load_B):
            pywarpx.particles.E_ext_particle_init_style = "none"
            pywarpx.particles.B_ext_particle_init_style = "none"
            return

        # Always register this object as a named external field so that multiple
        # LoadAppliedField objects compose (no overwrite of global keys).
        if not hasattr(self, "read_fields_from_path") or not self.read_fields_from_path:
            raise ValueError("[PICMI] read_fields_from_path must be provided.")

        # construct particles.<fname>.read_fields_from_path as needed for WarpX input
        fname = self._next_auto_name()
        pywarpx.particles.__setattr__(
            f"{fname}.read_fields_from_path", self.read_fields_from_path
        )

        if self.load_E:
            pywarpx.particles.E_ext_particle_init_style = "read_from_file"
            self._append_names("E_ext_particle_fields", [fname])

            dep_raw = self.E_time_function or "1.0"
            dep = pywarpx.my_constants.mangle_expression(dep_raw, mangle_dict)
            pywarpx.particles.__setattr__(f"{fname}.read_fields_E_dependency(t)", dep)
        else:
            pywarpx.particles.E_ext_particle_init_style = "none"

        if self.load_B:
            pywarpx.particles.B_ext_particle_init_style = "read_from_file"
            self._append_names("B_ext_particle_fields", [fname])

            # div cleaner knobs are global-ish: last set value wins
            pywarpx.warpx.do_initial_div_cleaning = self.do_initial_div_cleaning
            pywarpx.warpx.add_new_group_attr(
                "projection_div_cleaner", "atol", self.div_cleaner_atol
            )
            pywarpx.warpx.add_new_group_attr(
                "projection_div_cleaner", "rtol", self.div_cleaner_rtol
            )

            dep_raw = self.B_time_function or "1.0"
            dep = pywarpx.my_constants.mangle_expression(dep_raw, mangle_dict)
            pywarpx.particles.__setattr__(f"{fname}.read_fields_B_dependency(t)", dep)
        else:
            pywarpx.particles.B_ext_particle_init_style = "none"


class ConstantAppliedField(picmistandard.PICMI_ConstantAppliedField):
    def applied_field_initialize_inputs(self):
        # Note that lower and upper_bound are not used by WarpX

        if self.Ex is not None or self.Ey is not None or self.Ez is not None:
            pywarpx.particles.E_ext_particle_init_style = "constant"
            pywarpx.particles.E_external_particle = [
                self.Ex or 0.0,
                self.Ey or 0.0,
                self.Ez or 0.0,
            ]

        if self.Bx is not None or self.By is not None or self.Bz is not None:
            pywarpx.particles.B_ext_particle_init_style = "constant"
            pywarpx.particles.B_external_particle = [
                self.Bx or 0.0,
                self.By or 0.0,
                self.Bz or 0.0,
            ]


class AnalyticAppliedField(picmistandard.PICMI_AnalyticAppliedField):
    _mangle_dict: dict | None = PrivateAttr(default=None)

    def applied_field_initialize_inputs(self):
        # Note that lower and upper_bound are not used by WarpX

        if self._mangle_dict is None:
            # Only do this once so that the same variables are used in this distribution
            # is used multiple times
            self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)

        if (
            self.Ex_expression is not None
            or self.Ey_expression is not None
            or self.Ez_expression is not None
        ):
            pywarpx.particles.E_ext_particle_init_style = (
                "parse_e_ext_particle_function"
            )
            for sdir, expression in zip(
                ["x", "y", "z"],
                [self.Ex_expression, self.Ey_expression, self.Ez_expression],
            ):
                expression = pywarpx.my_constants.mangle_expression(
                    expression, self._mangle_dict
                )
                pywarpx.particles.__setattr__(
                    f"E{sdir}_external_particle_function(x,y,z,t)", expression
                )

        if (
            self.Bx_expression is not None
            or self.By_expression is not None
            or self.Bz_expression is not None
        ):
            pywarpx.particles.B_ext_particle_init_style = (
                "parse_b_ext_particle_function"
            )
            for sdir, expression in zip(
                ["x", "y", "z"],
                [self.Bx_expression, self.By_expression, self.Bz_expression],
            ):
                expression = pywarpx.my_constants.mangle_expression(
                    expression, self._mangle_dict
                )
                pywarpx.particles.__setattr__(
                    f"B{sdir}_external_particle_function(x,y,z,t)", expression
                )


class Mirror(picmistandard.PICMI_Mirror):
    def applied_field_initialize_inputs(self):
        try:
            pywarpx.warpx.num_mirrors
        except AttributeError:
            pywarpx.warpx.num_mirrors = 0
            pywarpx.warpx.mirror_z = []
            pywarpx.warpx.mirror_z_width = []
            pywarpx.warpx.mirror_z_npoints = []

        pywarpx.warpx.num_mirrors += 1
        pywarpx.warpx.mirror_z.append(self.z_front_location)
        pywarpx.warpx.mirror_z_width.append(self.depth)
        pywarpx.warpx.mirror_z_npoints.append(self.number_of_cells)


class FieldIonization(picmistandard.PICMI_FieldIonization):
    """
    WarpX only has ADK ionization model implemented.
    """

    def interaction_initialize_inputs(self):
        assert self.model == "ADK", "WarpX only has ADK ionization model implemented"
        self.ionized_species._species.do_field_ionization = 1
        self.ionized_species._species.physical_element = (
            self.ionized_species.particle_type
        )
        self.ionized_species._species.ionization_product_species = (
            self.product_species.name
        )
        self.ionized_species._species.ionization_initial_level = (
            self.ionized_species.charge_state
        )
        self.ionized_species._species.charge = "q_e"


class CollisionBase(picmistandard.PICMI_Extension):
    """Base class of the collisions, accepted by ``Simulation.warpx_collisions``"""

    name: str = Field(description="Name of instance (used in the inputs file)")

    @model_validator(mode="before")
    @classmethod
    def _removed_ndt(cls, data):
        if isinstance(data, dict) and "ndt" in data:
            raise ValueError(
                "`ndt` is no longer a valid option for collisions."
                "Please use `ndt_supercycle` instead (run collision every N PIC steps)."
            )
        return data

    def collision_initialize_inputs(self):
        raise NotImplementedError


class CoulombCollisions(CollisionBase):
    """
    Custom class to handle setup of binary Coulomb collisions in WarpX. If
    collision initialization is added to picmistandard this can be changed to
    inherit that functionality.
    """

    species: list[Species] = Field(
        min_length=2,
        max_length=2,
        description="The species involved in the collision. Must be of length 2.",
    )
    CoulombLog: float | None = Field(
        default=None,
        description="Value of the Coulomb log to use in the collision cross section. If not supplied, it is calculated from the local conditions.",
    )
    ndt_supercycle: int | None = Field(
        default=None,
        ge=1,
        description="Run collision once every ndt_supercycle PIC time steps (dt_collision = ndt_supercycle * dt_PIC). Mutually exclusive with ndt_subcycle. Default is 1.",
    )
    ndt_subcycle: int | None = Field(
        default=None,
        ge=1,
        description="Run collision ndt_subcycle times per PIC time step (dt_collision = dt_PIC / ndt_subcycle). Mutually exclusive with ndt_supercycle.",
    )

    def collision_initialize_inputs(self):
        collision = pywarpx.Collisions.newcollision(self.name)
        collision.type = "pairwisecoulomb"
        collision.species = [species.name for species in self.species]
        collision.CoulombLog = self.CoulombLog
        collision.ndt_supercycle = self.ndt_supercycle
        collision.ndt_subcycle = self.ndt_subcycle


class MCCCollisions(CollisionBase):
    """
    Custom class to handle setup of MCC collisions in WarpX. If collision
    initialization is added to picmistandard this can be changed to inherit
    that functionality.
    """

    species: Species = Field(description="The species involved in the collision")
    background_density: float | str = Field(
        description="The density of the background. An string expression as a function of (x, y, z, t) can be used."
    )
    background_temperature: float | str = Field(
        description="The temperature of the background. An string expression as a function of (x, y, z, t) can be used."
    )
    scattering_processes: dict[str, dict[str, Species | int | float | str]] = Field(
        description="The scattering process to use and any needed information"
    )
    background_mass: float | None = Field(
        default=None,
        description="The mass of the background particle. If not supplied, the default depends on the type of scattering process.",
    )
    max_background_density: float | None = Field(
        default=None,
        description="The maximum background density. When the background_density is an expression, this must also be specified.",
    )
    ndt_supercycle: int | None = Field(
        default=None,
        ge=1,
        description="Run collision once every ndt_supercycle PIC time steps (dt_collision = ndt_supercycle * dt_PIC). Mutually exclusive with ndt_subcycle. Default is 1.",
    )
    ndt_subcycle: int | None = Field(
        default=None,
        ge=1,
        description="Run collision ndt_subcycle times per PIC time step (dt_collision = dt_PIC / ndt_subcycle). Mutually exclusive with ndt_supercycle.",
    )

    def collision_initialize_inputs(self):
        collision = pywarpx.Collisions.newcollision(self.name)
        collision.type = "background_mcc"
        collision.species = self.species.name
        if isinstance(self.background_density, str):
            collision.__setattr__(
                "background_density(x,y,z,t)", self.background_density
            )
        else:
            collision.background_density = self.background_density
        if isinstance(self.background_temperature, str):
            collision.__setattr__(
                "background_temperature(x,y,z,t)", self.background_temperature
            )
        else:
            collision.background_temperature = self.background_temperature
        collision.background_mass = self.background_mass
        collision.max_background_density = self.max_background_density
        collision.ndt_supercycle = self.ndt_supercycle
        collision.ndt_subcycle = self.ndt_subcycle

        collision.scattering_processes = self.scattering_processes.keys()
        for process, kw in self.scattering_processes.items():
            for key, val in kw.items():
                if key == "species":
                    val = val.name
                collision.add_new_attr(process + "_" + key, val)


class DSMCCollisions(CollisionBase):
    """
    Custom class to handle setup of DSMC collisions in WarpX. If collision
    initialization is added to picmistandard this can be changed to inherit
    that functionality.
    """

    species: list[Species] = Field(description="The species involved in the collision")
    scattering_processes: dict[str, dict[str, Species | int | float | str]] = Field(
        description="The scattering process to use and any needed information"
    )
    product_species: list[Species] | None = Field(
        default=None,
        description="The species produced by collision processes (currently both ionization and charge-exchange require defining the product species).",
    )
    ndt_supercycle: int | None = Field(
        default=None,
        ge=1,
        description="Run collision once every ndt_supercycle PIC time steps (dt_collision = ndt_supercycle * dt_PIC). Mutually exclusive with ndt_subcycle. Default is 1.",
    )
    ndt_subcycle: int | None = Field(
        default=None,
        ge=1,
        description="Run collision ndt_subcycle times per PIC time step (dt_collision = dt_PIC / ndt_subcycle). Mutually exclusive with ndt_supercycle.",
    )

    def collision_initialize_inputs(self):
        collision = pywarpx.Collisions.newcollision(self.name)
        collision.type = "dsmc"
        collision.species = [species.name for species in self.species]
        if self.product_species is not None:
            collision.product_species = [
                species.name for species in self.product_species
            ]
        collision.ndt_supercycle = self.ndt_supercycle
        collision.ndt_subcycle = self.ndt_subcycle

        collision.scattering_processes = self.scattering_processes.keys()
        for process, kw in self.scattering_processes.items():
            for key, val in kw.items():
                if "species" in key:
                    val = val.name
                collision.add_new_attr(process + "_" + key, val)


class HybridResistiveDragCollisions(CollisionBase):
    """
    Custom class to handle setup of the hybrid-PIC resistive drag collision in
    WarpX. If collision initialization is added to picmistandard this can be
    changed to inherit that functionality.

    This is the ion-side half of the electron-ion friction operator of the
    hybrid-PIC (Ohm's law) solver: it relaxes the bulk velocity of the given
    ion species toward the electron fluid velocity at the rate implied by the
    resistivity of Ohm's law, pairing with the ``plasma_resistivity`` /
    ``plasma_resistivity_species`` parameters of :class:`HybridPICSolver`.
    With the drag registered, the resistive terms of Ohm's law are also
    included in the particle-push E-field, so when used the drag must be
    registered on every charged species (WarpX asserts this at
    initialization).
    """

    species: Species = Field(
        description="The (positive, current-depositing) ion species the drag acts on"
    )

    def collision_initialize_inputs(self):
        collision = pywarpx.Collisions.newcollision(self.name)
        collision.type = "hybrid_resistive_drag"
        collision.species = [self.species.name]


class InverseBremsstrahlungCollisions(CollisionBase):
    """
    Custom class to handle setup of inverse Bremsstrahlung collisions in WarpX. If
    collision initialization is added to picmistandard this can be changed to
    inherit that functionality.
    """

    species: list[Species] = Field(
        min_length=2,
        max_length=2,
        description="The species involved in the collision. Must be of length 2. The photon species must be given first, followed by the electron species.",
    )
    energy_fraction: float | None = Field(
        default=None,
        description="The fraction of the relative energy in the collision COM frame that is used in the distribution of the absorbed photon energy.",
    )
    ndt_supercycle: int | None = Field(
        default=None,
        ge=1,
        description="Run collision once every ndt_supercycle PIC time steps (dt_collision = ndt_supercycle * dt_PIC). Mutually exclusive with ndt_subcycle. Default is 1.",
    )
    ndt_subcycle: int | None = Field(
        default=None,
        ge=1,
        description="Run collision ndt_subcycle times per PIC time step (dt_collision = dt_PIC / ndt_subcycle). Mutually exclusive with ndt_supercycle.",
    )

    def collision_initialize_inputs(self):
        collision = pywarpx.Collisions.newcollision(self.name)
        collision.type = "inverse_bremsstrahlung"
        collision.species = [species.name for species in self.species]
        collision.energy_fraction = self.energy_fraction
        collision.ndt_supercycle = self.ndt_supercycle
        collision.ndt_subcycle = self.ndt_subcycle


class EmbeddedBoundary(
    picmistandard.PICMI_Extension, picmistandard.PICMI_ExpressionParameters
):
    """
    Custom class to handle set up of embedded boundaries specific to WarpX.
    If embedded boundary initialization is added to picmistandard this can be
    changed to inherit that functionality. The geometry can be specified either as
    an implicit function or as an STL file (ASCII or binary). In the latter case the
    geometry specified in the STL file can be scaled, translated and inverted.

    Parameters used in the analytic expressions should be given as additional keyword arguments.
    """

    _expression_fields: ClassVar[tuple[str, ...]] = ("implicit_function", "potential")

    implicit_function: Expression | None = Field(
        default=None, description="Analytic expression describing the embedded boundary"
    )
    stl_file: str | None = Field(
        default=None,
        description="STL file path (string), file contains the embedded boundary geometry",
    )
    stl_scale: float | None = Field(
        default=None, description="Factor by which the STL geometry is scaled"
    )
    stl_center: list[float] | None = Field(
        default=None,
        description="Vector by which the STL geometry is translated (in meters)",
    )
    stl_reverse_normal: bool = Field(
        default=False, description="If True inverts the orientation of the STL geometry"
    )
    potential: Expression | None = Field(
        default=None,
        description="Analytic expression defining the potential. Can only be specified when the solver is electrostatic. (default 0.)",
    )
    cover_multiple_cuts: bool | None = Field(
        default=None,
        description="Whether to cover cells with multiple cuts. (If False, this will raise an error if some cells have multiple cuts)",
    )
    user_defined_kw: dict = Field(
        default_factory=dict,
        description="Constants referenced in the expressions, collected from otherwise-unrecognized keyword arguments.",
    )

    _mangle_dict: dict | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_geometry(self) -> Self:
        assert self.stl_file is None or self.implicit_function is None, (
            "Only one between implicit_function and stl_file can be specified"
        )
        if self.stl_file is None:
            assert self.stl_scale is None, (
                "EB can only be scaled only when using an stl file"
            )
            assert self.stl_center is None, (
                "EB can only be translated only when using an stl file"
            )
            assert self.stl_reverse_normal is False, (
                "EB can only be reversed only when using an stl file"
            )
        return self

    def embedded_boundary_initialize_inputs(self, solver):
        # Add the user defined keywords to my_constants
        # The keywords are mangled if there is a conflicting variable already
        # defined in my_constants with the same name but different value.
        self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)

        if self.implicit_function is not None:
            expression = pywarpx.my_constants.mangle_expression(
                self.implicit_function, self._mangle_dict
            )
            pywarpx.warpx.eb_implicit_function = expression

        if self.stl_file is not None:
            pywarpx.eb2.geom_type = "stl"
            pywarpx.eb2.stl_file = self.stl_file
            pywarpx.eb2.stl_scale = self.stl_scale
            pywarpx.eb2.stl_center = self.stl_center
            pywarpx.eb2.stl_reverse_normal = self.stl_reverse_normal

        pywarpx.eb2.cover_multiple_cuts = self.cover_multiple_cuts

        if self.potential is not None:
            expression = pywarpx.my_constants.mangle_expression(
                self.potential, self._mangle_dict
            )
            pywarpx.warpx.__setattr__("eb_potential(x,y,z,t)", expression)


class MacroscopicProperty(
    picmistandard.PICMI_Extension, picmistandard.PICMI_ExpressionParameters
):
    """
    Custom class to handle set up of material property specific to WarpX.
    If macroscopic properties initialization is added to picmistandard this can be
    changed to inherit that functionality. The geometry can be specified either as
    an implicit function.  STL file (ASCII or binary) will be added in future. In
    the latter case the geometry specified in the STL file can be scaled,
    translated and inverted.

    This can be used for both Electromagnetic and electrostatic solvers.

    Parameters used in the analytic expressions should be given as additional keyword arguments.

    The parameters ``stl_file``, ``stl_scale``, ``stl_center``, and ``stl_reverse_normal``
    are not implemented yet.
    """

    _expression_fields: ClassVar[tuple[str, ...]] = ("implicit_function",)

    name: Literal["sigma", "epsilon", "mu"] = Field(
        default="epsilon",
        description='the macroscopic property name to set. One of "sigma", "epsilon", or "mu"',
    )
    implicit_function: Expression | None = Field(
        default=None,
        description="Analytic expression f(x,y,z) describing the sigma, epsilon, or mu",
    )
    value: float | None = Field(
        default=None, description="Value of sigma, epsilon, or mu if it is a constant"
    )
    method: Literal["backwardeuler", "laxwendroff"] | None = Field(
        default=None,
        description="The algorithm for updating electric field when algo.em_solver_medium is macroscopic. Available options for name = sigma are: backwardeuler and laxwendroff",
    )
    stl_file: str | None = Field(
        default=None,
        description="(not implemented) STL file path (string), file contains the embedded boundary geometry",
    )
    stl_scale: float | None = Field(
        default=None,
        description="(not implemented) Factor by which the STL geometry is scaled",
    )
    stl_center: list[float] | None = Field(
        default=None,
        description="(not implemented) Vector by which the STL geometry is translated (in meters)",
    )
    stl_reverse_normal: bool = Field(
        default=False,
        description="(not implemented) If True inverts the orientation of the STL geometry",
    )
    user_defined_kw: dict = Field(
        default_factory=dict,
        description="Constants referenced in the implicit function, collected from otherwise-unrecognized keyword arguments.",
    )

    _mangle_dict: dict | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_parameters(self) -> Self:
        assert (
            sum(
                [
                    self.stl_file is not None,
                    self.implicit_function is not None,
                    self.value is not None,
                ]
            )
            == 1
        ), "Exactly one one of implicit_function, stl_file, and value must be specified"
        if self.stl_file is None:
            assert self.stl_scale is None, (
                "Material property can only be scaled only when using an stl file"
            )
            assert self.stl_center is None, (
                "Material property  can only be translated only when using an stl file"
            )
            assert self.stl_reverse_normal is False, (
                "Material property  can only be reversed only when using an stl file"
            )
        # Validate method for conductivity (sigma)
        if self.method is not None and self.name != "sigma":
            raise ValueError("Input 'method' can only be used with 'sigma'")
        return self

    def material_property_initialize_inputs(self, solver):
        # Add the user defined keywords to my_constants
        # The keywords are mangled if there is a conflicting variable already
        # defined in my_constants with the same name but different value.
        self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)
        macroscopic = pywarpx.warpx.get_bucket("macroscopic")
        if self.implicit_function is not None:
            expression = pywarpx.my_constants.mangle_expression(
                self.implicit_function, self._mangle_dict
            )
            setattr(macroscopic, self.name + "_function(x,y,z)", expression)

        if self.value is not None:
            setattr(macroscopic, self.name, self.value)

        if self.stl_file is not None:
            raise NotImplementedError(
                "material property definition with stl file is not implemented yet"
            )

        if self.method is not None:
            setattr(
                pywarpx.algo,
                "macroscopic_" + self.name + "_method",
                self.method,
            )


class PlasmaLens(picmistandard.PICMI_AppliedFieldExtension):
    """
    Custom class to setup a plasma lens lattice.
    The applied fields are dependent only on the transverse position.

    The field that is applied depends on the transverse position of the particle, (x,y)

    - Ex = x*strengths_E

    - Ey = y*strengths_E

    - Bx = +y*strengths_B

    - By = -x*strengths_B
    """

    period: float = Field(
        description="Periodicity of the lattice (in lab frame, in meters)"
    )
    starts: list[float] = Field(
        description="The start of each lens relative to the periodic repeat"
    )
    lengths: list[float] = Field(description="The length of each lens")
    strengths_E: list[float] | None = Field(
        default=None,
        description="The electric field strength of each lens (default 0.)",
    )
    strengths_B: list[float] | None = Field(
        default=None,
        description="The magnetic field strength of each lens (default 0.)",
    )

    @model_validator(mode="after")
    def _check_strengths(self) -> Self:
        assert (self.strengths_E is not None) or (self.strengths_B is not None), (
            "One of strengths_E or strengths_B must be supplied"
        )
        return self

    def applied_field_initialize_inputs(self):
        pywarpx.particles.E_ext_particle_init_style = "repeated_plasma_lens"
        pywarpx.particles.B_ext_particle_init_style = "repeated_plasma_lens"
        pywarpx.particles.repeated_plasma_lens_period = self.period
        pywarpx.particles.repeated_plasma_lens_starts = self.starts
        pywarpx.particles.repeated_plasma_lens_lengths = self.lengths
        pywarpx.particles.repeated_plasma_lens_strengths_E = self.strengths_E
        pywarpx.particles.repeated_plasma_lens_strengths_B = self.strengths_B


class Simulation(picmistandard.PICMI_Simulation):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_Simulation)
    )

    # NOTE: In the future, LibWarpX objects may actually be owned by Simulation
    # objects to permit multiple WarpX runs simultaneously.
    extension: ClassVar[Any] = pywarpx.libwarpx
    """Handle to the running, compiled WarpX library (a ``LibWarpX`` instance).

    This is the low-level bridge from a PICMI input script to the live C++ WarpX
    instance. It is most useful for interactive runs (stepping the simulation from
    Python), where it exposes the running simulation state and the pybind11-bound API.

    Commonly used entry points:

    - ``extension.warpx`` -- the C++ ``WarpX`` instance, e.g.
      ``sim.extension.warpx.getistep(lev=0)`` (current step),
      ``sim.extension.warpx.gett_new(0)`` (current time),
      ``sim.extension.warpx.getdt(0)`` (time-step size), or
      ``sim.extension.warpx.set_potential_on_eb("2.")`` (set the embedded-boundary potential).
    - ``extension.amr`` -- the AMReX adaptive mesh-refinement core object.
    - ``extension.getNProcs()`` -- the number of MPI processes.
    - ``extension.libwarpx_so`` -- the raw pybind11 module for direct access to the
      compiled bindings.

    Field and particle data are usually accessed more conveniently through the
    ``fields`` and ``particles`` properties, which wrap this handle. See
    :ref:`usage-python-extend` for the full runtime-extension workflow.
    """

    # --- WarpX-specific extension inputs (typed, exposed under the ``warpx_`` alias).
    # --- See the class docstring above for a description of each.
    evolve_scheme: EvolveSchemeBase | None = Field(
        default=None, description="Which evolve scheme to use"
    )
    current_deposition_algo: str | None = Field(
        default=None,
        description="Current deposition algorithm. The default depends on conditions.",
    )
    charge_deposition_algo: str | None = Field(
        default=None, description="Charge deposition algorithm."
    )
    field_gathering_algo: str | None = Field(
        default=None,
        description="Field gathering algorithm. The default depends on conditions.",
    )
    particle_pusher_algo: str | None = Field(
        default=None, description="Particle pushing algorithm."
    )
    use_filter: bool | None = Field(
        default=None,
        description="Whether to use filtering. The default depends on the conditions.",
    )
    grid_type: str | None = Field(
        default=None,
        description="Whether to use a collocated grid (all fields defined at the cell nodes), a staggered grid (fields defined on a Yee grid), or a hybrid grid (fields and currents are interpolated back and forth between a staggered grid and a collocated grid, must be used with momentum-conserving field gathering algorithm).",
    )
    do_current_centering: bool | None = Field(
        default=None,
        description="If true, the current is deposited on a nodal grid and then centered to a staggered grid (Yee grid), using finite-order interpolation. Default: warpx.do_current_centering=0 with collocated or staggered grids, warpx.do_current_centering=1 with hybrid grids.",
    )
    field_centering_order: list[int] | None = Field(
        default=None,
        description="The order of interpolation used with staggered or hybrid grids (``warpx_grid_type=staggered`` or ``warpx_grid_type=hybrid``) and momentum-conserving field gathering (``warpx_field_gathering_algo=momentum-conserving``) to interpolate the electric and magnetic fields from the cell centers to the cell nodes, before gathering the fields from the cell nodes to the particle positions. Default: ``warpx_field_centering_no<x,y,z>=2`` with staggered grids, ``warpx_field_centering_no<x,y,z>=8`` with hybrid grids (typically necessary to ensure stability in boosted-frame simulations of relativistic plasmas and beams).",
    )
    current_centering_order: list[int] | None = Field(
        default=None,
        description="The order of interpolation used with hybrid grids (``warpx_grid_type=hybrid``) to interpolate the currents from the cell nodes to the cell centers when ``warpx_do_current_centering=1``, before pushing the Maxwell fields on staggered grids. Default: ``warpx_current_centering_no<x,y,z>=8`` with hybrid grids (typically necessary to ensure stability in boosted-frame simulations of relativistic plasmas and beams).",
    )
    serialize_initial_conditions: bool | None = Field(
        default=None,
        description="Controls the random numbers used for initialization. This parameter should only be used for testing and continuous integration.",
    )
    random_seed: int | str | None = Field(
        default=None, description="(See documentation)"
    )
    do_dynamic_scheduling: bool | None = Field(
        default=None, description="Whether to do dynamic scheduling with OpenMP"
    )
    roundrobin_sfc: bool | None = Field(
        default=None,
        description="Whether to use the RRSFC strategy for making DistributionMapping",
    )
    load_balance_intervals: int | str | None = Field(
        default=None, description="The intervals for doing load balancing"
    )
    load_balance_efficiency_ratio_threshold: float | None = Field(
        default=None, description="(See documentation)"
    )
    load_balance_with_sfc: bool | None = Field(
        default=None, description="(See documentation)"
    )
    load_balance_knapsack_factor: float | None = Field(
        default=None, description="(See documentation)"
    )
    load_balance_costs_update: str | None = Field(
        default=None, description="(See documentation)"
    )
    costs_heuristic_particles_wt: float | None = Field(
        default=None, description="(See documentation)"
    )
    costs_heuristic_cells_wt: float | None = Field(
        default=None, description="(See documentation)"
    )
    use_fdtd_nci_corr: bool | None = Field(
        default=None,
        description="Whether to use the NCI correction when using the FDTD solver",
    )
    amr_check_input: bool | None = Field(
        default=None,
        description="Whether AMReX should perform checks on the input (primarily related to the max grid size and blocking factors)",
    )
    amr_restart: str | None = Field(
        default=None, description="The name of the restart to use"
    )
    amrex_the_arena_is_managed: bool | None = Field(
        default=None, description="Whether to use managed memory in the AMReX Arena"
    )
    amrex_the_arena_init_size: int | None = Field(
        default=None,
        description="The amount of memory in bytes to allocate in the Arena.",
    )
    amrex_use_gpu_aware_mpi: bool | None = Field(
        default=None, description="Whether to use GPU-aware MPI communications"
    )
    do_device_synchronize: bool | None = Field(
        default=None,
        description="Whether to synchronize GPU threads at ends of profiling regions. Note that if this is set to False, the TinyProfiler table can be misleading.",
    )
    zmax_plasma_to_compute_max_step: float | None = Field(
        default=None,
        description="Sets the simulation run time based on the maximum z value",
    )
    compute_max_step_from_btd: bool | None = Field(
        default=None,
        description="If specified, automatically calculates the number of iterations required in the boosted frame for all back-transformed diagnostics to be completed.",
    )
    sort_intervals: int | str | None = Field(
        default=None,
        description="Using the Intervals parser syntax, this string defines the timesteps at which particles are sorted. If <=0, do not sort particles. It is turned on on GPUs for performance reasons (to improve memory locality).",
    )
    sort_particles_for_deposition: bool | None = Field(
        default=None,
        description="This option controls the type of sorting used if particle sorting is turned on, i.e. if sort_intervals is not <=0. If `true`, particles will be sorted by cell to optimize deposition with many particles per cell, in the order `x` -> `y` -> `z` -> `ppc`. If `false`, particles will be sorted by bin, using the sort_bin_size parameter below, in the order `ppc` -> `x` -> `y` -> `z`. `true` is recommended for best performance on NVIDIA GPUs, especially if there are many particles per cell.",
    )
    sort_idx_type: list[int] | None = Field(
        default=None,
        description=(
            "This controls the type of grid used to sort the particles when sort_particles_for_deposition is true.\n"
            "Possible values are:\n"
            "\n"
            "* idx_type = {0, 0, 0}: Sort particles to a cell centered grid,\n"
            "* idx_type = {1, 1, 1}: Sort particles to a node centered grid,\n"
            "* idx_type = {2, 2, 2}: Compromise between a cell and node centered grid.\n"
            "\n"
            "In 2D (XZ and RZ), only the first two elements are read. In 1D, only the first element is read."
        ),
    )
    sort_bin_size: list[int] | None = Field(
        default=None,
        description="If `sort_intervals` is activated and `sort_particles_for_deposition` is false, particles are sorted in bins of `sort_bin_size` cells. In 2D, only the first two elements are read.",
    )
    used_inputs_file: str | None = Field(
        default=None,
        description="The name of the text file that the used input parameters is written to,",
    )
    collisions: list[CollisionBase] | None = Field(
        default=None,
        description="The collision instance specifying the particle collisions",
    )
    collisions_split_momentum_push: bool | None = Field(
        default=None,
        description="If true, collisions are performed in the middle of the momentum push, which is split into two substeps. This improves energy conservation, as demonstrated in (Vay et al., Phys. Rev. E 111, 2025). This is only implemented for the explicit evolve scheme and is not available for the implicit evolve schemes.",
    )
    embedded_boundary: EmbeddedBoundary | None = Field(
        default=None, description="The embedded boundary of the simulation"
    )
    break_signals: list[str | int] | None = Field(
        default=None, description="Signals on which to break"
    )
    checkpoint_signals: list[str | int] | None = Field(
        default=None, description="Signals on which to write out a checkpoint"
    )
    numprocs: list[int] | None = Field(
        default=None,
        description="Domain decomposition on the coarsest level. The domain will be chopped into the exact number of pieces in each dimension as specified by this parameter. https://warpx.readthedocs.io/en/latest/usage/parameters.html#distribution-across-mpi-ranks-and-parallelization https://warpx.readthedocs.io/en/latest/usage/domain_decomposition.html#simple-method",
    )
    reduced_diags_path: str | None = Field(
        default=None,
        description="Sets the default path for reduced diagnostic output files",
    )
    reduced_diags_extension: str | None = Field(
        default=None,
        description="Sets the default extension for reduced diagnostic output files",
    )
    reduced_diags_intervals: int | str | None = Field(
        default=None,
        description="Sets the default intervals for reduced diagnostic output files",
    )
    reduced_diags_separator: str | None = Field(
        default=None,
        description="Sets the default separator for reduced diagnostic output files",
    )
    reduced_diags_precision: int | None = Field(
        default=None,
        description="Sets the default precision for reduced diagnostic output files",
    )
    synchronize_velocity: bool | None = Field(
        default=None,
        description="Flags whether the particle velocities are synchronized in time with the positions in the diagnostics. When False, the particles are one half step behind the positions (except for the final diagnostic).",
    )
    self_fields_required_precision: float | None = Field(
        default=None, alias="warpx_self_fields_required_precision"
    )
    self_fields_absolute_tolerance: float | None = Field(
        default=None, alias="warpx_self_fields_absolute_tolerance"
    )
    self_fields_max_iters: int | None = Field(
        default=None, alias="warpx_self_fields_max_iters"
    )
    self_fields_verbosity: int | None = Field(
        default=None, alias="warpx_self_fields_verbosity"
    )

    macroscopic_properties: list[MacroscopicProperty] = Field(
        default_factory=list,
        description="Macroscopic material properties added with add_macroscopic_property",
    )

    # --- Runtime state (not user inputs).
    _inputs_initialized: bool = PrivateAttr(default=False)
    _warpx_initialized: bool = PrivateAttr(default=False)
    _finalized: bool = PrivateAttr(default=False)

    def _check_not_finalized(self):
        if self._finalized:
            raise RuntimeError(
                "This Simulation was finalized. Create new PICMI objects to "
                "set up another simulation."
            )

    def initialize_inputs(self):
        self._check_not_finalized()
        if self._inputs_initialized:
            return

        self._inputs_initialized = True

        pywarpx.warpx.verbose = self.verbose
        if self.time_step_size is not None:
            pywarpx.warpx.const_dt = self.time_step_size

        if self.gamma_boost is not None:
            pywarpx.warpx.gamma_boost = self.gamma_boost
            pywarpx.warpx.boost_direction = "z"

        pywarpx.warpx.zmax_plasma_to_compute_max_step = (
            self.zmax_plasma_to_compute_max_step
        )
        pywarpx.warpx.compute_max_step_from_btd = self.compute_max_step_from_btd

        pywarpx.warpx.sort_intervals = self.sort_intervals
        pywarpx.warpx.sort_particles_for_deposition = self.sort_particles_for_deposition
        pywarpx.warpx.sort_idx_type = self.sort_idx_type
        pywarpx.warpx.sort_bin_size = self.sort_bin_size

        if self.evolve_scheme is not None:
            self.evolve_scheme.solver_scheme_initialize_inputs()

        pywarpx.algo.current_deposition = self.current_deposition_algo
        pywarpx.algo.charge_deposition = self.charge_deposition_algo
        pywarpx.algo.field_gathering = self.field_gathering_algo
        pywarpx.algo.particle_pusher = self.particle_pusher_algo
        pywarpx.algo.load_balance_intervals = self.load_balance_intervals
        pywarpx.algo.load_balance_efficiency_ratio_threshold = (
            self.load_balance_efficiency_ratio_threshold
        )
        pywarpx.algo.load_balance_with_sfc = self.load_balance_with_sfc
        pywarpx.algo.load_balance_knapsack_factor = self.load_balance_knapsack_factor
        pywarpx.algo.load_balance_costs_update = self.load_balance_costs_update
        pywarpx.algo.costs_heuristic_particles_wt = self.costs_heuristic_particles_wt
        pywarpx.algo.costs_heuristic_cells_wt = self.costs_heuristic_cells_wt

        pywarpx.warpx.grid_type = self.grid_type
        pywarpx.warpx.do_current_centering = self.do_current_centering
        pywarpx.warpx.use_filter = self.use_filter
        pywarpx.warpx.serialize_initial_conditions = self.serialize_initial_conditions
        pywarpx.warpx.random_seed = self.random_seed
        pywarpx.warpx.used_inputs_file = self.used_inputs_file

        pywarpx.warpx.do_dynamic_scheduling = self.do_dynamic_scheduling

        pywarpx.warpx.roundrobin_sfc = self.roundrobin_sfc

        pywarpx.particles.use_fdtd_nci_corr = self.use_fdtd_nci_corr

        pywarpx.amr.check_input = self.amr_check_input

        pywarpx.warpx.break_signals = self.break_signals
        pywarpx.warpx.checkpoint_signals = self.checkpoint_signals

        pywarpx.warpx.synchronize_velocity_for_diagnostics = self.synchronize_velocity

        pywarpx.warpx.numprocs = self.numprocs

        reduced_diags = pywarpx.warpx.get_bucket("reduced_diags")
        reduced_diags.path = self.reduced_diags_path
        reduced_diags.extension = self.reduced_diags_extension
        reduced_diags.intervals = self.reduced_diags_intervals
        reduced_diags.separator = self.reduced_diags_separator
        reduced_diags.precision = self.reduced_diags_precision

        particle_shape = self.particle_shape
        for s in self.species:
            if s.particle_shape is not None:
                assert particle_shape is None or particle_shape == s.particle_shape, (
                    Exception("WarpX only supports one particle shape for all species")
                )
                # --- If this was set for any species, use that value.
                particle_shape = s.particle_shape

        if particle_shape is not None and (
            len(self.species) > 0 or len(self.lasers) > 0
        ):
            if isinstance(particle_shape, str):
                interpolation_order = {
                    "NGP": 0,
                    "linear": 1,
                    "quadratic": 2,
                    "cubic": 3,
                }[particle_shape]
            else:
                interpolation_order = particle_shape
            pywarpx.algo.particle_shape = interpolation_order

        pywarpx.warpx.self_fields_required_precision = (
            self.self_fields_required_precision
        )
        pywarpx.warpx.self_fields_absolute_tolerance = (
            self.self_fields_absolute_tolerance
        )
        pywarpx.warpx.self_fields_max_iters = self.self_fields_max_iters
        pywarpx.warpx.self_fields_verbosity = self.self_fields_verbosity

        self.solver.solver_initialize_inputs()

        # Initialize warpx.field_centering_no<x,y,z> and warpx.current_centering_no<x,y,z>
        # if set by the user in the input (need to access grid info from solver attribute)
        # warpx.field_centering_no<x,y,z>
        if self.field_centering_order is not None:
            pywarpx.warpx.field_centering_nox = self.field_centering_order[0]
            if self.solver.grid.number_of_dimensions == 3:
                pywarpx.warpx.field_centering_noy = self.field_centering_order[1]
            pywarpx.warpx.field_centering_noz = self.field_centering_order[-1]
        # warpx.current_centering_no<x,y,z>
        if self.current_centering_order is not None:
            pywarpx.warpx.current_centering_nox = self.current_centering_order[0]
            if self.solver.grid.number_of_dimensions == 3:
                pywarpx.warpx.current_centering_noy = self.current_centering_order[1]
            pywarpx.warpx.current_centering_noz = self.current_centering_order[-1]

        for i in range(len(self.species)):
            self.species[i].species_initialize_inputs(
                self.layouts[i],
                self.initialize_self_fields[i],
                self.injection_plane_positions[i],
                self.injection_plane_normal_vectors[i],
            )

        for interaction in self.interactions:
            assert isinstance(interaction, FieldIonization)
            interaction.interaction_initialize_inputs()

        if self.collisions is not None:
            pywarpx.collisions.collision_names = []
            for collision in self.collisions:
                pywarpx.collisions.collision_names.append(collision.name)
                collision.collision_initialize_inputs()
            pywarpx.collisions.split_momentum_push = self.collisions_split_momentum_push

        if self.embedded_boundary is not None:
            self.embedded_boundary.embedded_boundary_initialize_inputs(self.solver)

        for i in range(len(self.lasers)):
            self.lasers[i].laser_initialize_inputs()
            self.laser_injection_methods[i].laser_antenna_initialize_inputs(
                self.lasers[i]
            )

        for applied_field in self.applied_fields:
            applied_field.applied_field_initialize_inputs()

        for diagnostic in self.diagnostics:
            diagnostic.diagnostic_initialize_inputs()

        if self.amr_restart:
            pywarpx.amr.restart = self.amr_restart

        if self.amrex_the_arena_is_managed is not None:
            pywarpx.amrex.the_arena_is_managed = self.amrex_the_arena_is_managed

        if self.amrex_the_arena_init_size is not None:
            pywarpx.amrex.the_arena_init_size = self.amrex_the_arena_init_size

        if self.amrex_use_gpu_aware_mpi is not None:
            pywarpx.amrex.use_gpu_aware_mpi = self.amrex_use_gpu_aware_mpi

        if self.do_device_synchronize is not None:
            pywarpx.warpx.do_device_synchronize = self.do_device_synchronize

        if len(self.macroscopic_properties) > 0:
            pywarpx.algo.em_solver_medium = "macroscopic"
            for prop in self.macroscopic_properties:
                prop.material_property_initialize_inputs(self.solver)

    def initialize_warpx(self, mpi_comm=None):
        self._check_not_finalized()
        if self._warpx_initialized:
            return

        self._warpx_initialized = True
        pywarpx.warpx.init(mpi_comm, max_step=self.max_steps, stop_time=self.max_time)

    def write_input_file(self, file_name="inputs"):
        self.initialize_inputs()
        pywarpx.warpx.write_inputs(
            file_name, max_step=self.max_steps, stop_time=self.max_time
        )

    def step(self, nsteps=None, mpi_comm=None):
        self.initialize_inputs()
        self.initialize_warpx(mpi_comm)
        if nsteps is None:
            if self.max_steps is not None:
                nsteps = self.max_steps
            else:
                nsteps = -1
        pywarpx.warpx.step(nsteps)

    def finalize(self):
        # unconditional: tearing down WarpX is a no-op if it was never
        # initialized, but the input state still needs to be cleared
        self._warpx_initialized = False
        self._finalized = True
        pywarpx.warpx.finalize()

    def add_macroscopic_property(self, macroscopic_property):
        """Add a macroscopic material property (an instance of MacroscopicProperty)"""
        self._append(macroscopic_properties=macroscopic_property)

    @property
    def fields(self):
        """
        This is a convenience property that returns the MultiFab registry, allowing
        easy fetching of the MultiFabs.
        """
        return self.extension.warpx.multifab_register()

    @property
    def particles(self):
        """
        This is a convenience property that returns the MultiParticleContainer, allowing
        easy fetching of the WarpXParticleContainer instances.
        """
        return self.extension.warpx.multi_particle_container()


# ----------------------------
# Simulation frame diagnostics
# ----------------------------


def _collect_warpx_constants(cls, data, expression_field):
    """Collect the constants referenced in an expression of a particle diagnostic.

    This allows variables to be used in the expression, but in order not to break other codes,
    the variables must begin with "warpx_".
    """
    if not isinstance(data, dict):
        return data
    data = dict(data)
    field = cls.model_fields[expression_field]
    expression = (
        data.get(field.alias or expression_field) or data.get(expression_field) or ""
    )
    # the field may be given by name or by its warpx_ alias (e.g., when loading a dump)
    user_defined_kw_key = (
        "warpx_user_defined_kw"
        if "warpx_user_defined_kw" in data
        else "user_defined_kw"
    )
    user_defined_kw = dict(data.get(user_defined_kw_key, {}))
    if expression:
        known = set()
        for fname, finfo in cls.model_fields.items():
            known.add(fname)
            if finfo.alias:
                known.add(finfo.alias)
        for k in list(data.keys()):
            if k in known:
                continue
            if k.startswith("warpx_") and re.search(r"\b%s\b" % k, expression):
                user_defined_kw[k] = data.pop(k)
    data[user_defined_kw_key] = user_defined_kw
    return data


class WarpXDiagnosticBase(object):
    """
    Base class for all WarpX diagnostic containing functionality shared by
    all WarpX diagnostic installations.

    The classes using this mixin declare the ``_diagnostic`` private attribute.
    """

    @property
    def diagnostic(self):
        """The WarpX inputs of this diagnostic (available after the inputs are initialized)"""
        return self._diagnostic

    def add_diagnostic(self):
        # reduced diagnostics go in a different bucket than regular diagnostics
        if isinstance(self, ReducedDiagnostic):
            bucket = pywarpx.reduced_diagnostics
            name_template = "reduced_diag"
        else:
            bucket = pywarpx.diagnostics
            name_template = "diag"

        name = getattr(self, "name", None)
        if name is None:
            diagnostics_number = len(bucket._diagnostics_dict) + 1
            self.name = f"{name_template}{diagnostics_number}"

        try:
            self._diagnostic = bucket._diagnostics_dict[self.name]
        except KeyError:
            self._diagnostic = pywarpx.Diagnostics.Diagnostic(
                self.name, _species_dict={}
            )
            bucket._diagnostics_dict[self.name] = self._diagnostic

    def set_write_dir(self):
        if self.write_dir is not None or self.file_prefix is not None:
            write_dir = self.write_dir or "diags"
            file_prefix = self.file_prefix or self.name
            self._diagnostic.file_prefix = os.path.join(write_dir, file_prefix)


@dataclass(frozen=True)
class ParticleFieldDiagnostic:
    """
    Class holding particle field diagnostic information to be processed in FieldDiagnostic below.

    Parameters
    ----------
    name: str
        Name of particle field diagnostic. If a component of a vector field, for the openPMD viewer
        to treat it as a vector, the coordinate (i.e x, y, z) should be the last character.

    func: parser str
        Parser function to be calculated for each particle per cell. Should be of the form
        f(x,y,z,ux,uy,uz)

    do_average: (0 or 1) optional, default 1
        Whether the diagnostic is averaged by the sum of particle weights in each cell

    filter: parser str, optional
        Parser function returning a boolean for whether to include a particle in the diagnostic.
        If not specified, all particles will be included. The function arguments are the same
        as the `func` above.
    """

    name: str
    func: str
    do_average: int = 1
    filter: str | None = None


class FieldDiagnostic(picmistandard.PICMI_FieldDiagnostic, WarpXDiagnosticBase):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_FieldDiagnostic)
    )

    period: int | str = Field(
        description="Period of time steps at which the diagnostic is performed; WarpX also accepts the Intervals parser string syntax (e.g. '::10')."
    )

    plot_raw_fields: bool | None = Field(
        default=None, description="Flag whether to dump the raw fields"
    )
    plot_raw_fields_guards: bool | None = Field(
        default=None,
        description="Flag whether the raw fields should include the guard cells",
    )
    plot_finepatch: bool | None = Field(default=None, alias="warpx_plot_finepatch")
    plot_crsepatch: bool | None = Field(default=None, alias="warpx_plot_crsepatch")
    format: str = Field(default="plotfile", description="Diagnostic file format")
    openpmd_backend: str | None = Field(
        default=None, description="Openpmd backend file format"
    )
    openpmd_encoding: str | None = Field(
        default=None,
        description="Only read if ``<diag_name>.format = openpmd``. openPMD file output encoding. File based: one file per timestep (slower), group/variable based: one file for all steps (faster)). Variable based is an experimental feature with ADIOS2. Default: `'f'`.",
    )
    file_prefix: str | None = Field(
        default=None, description="Prefix on the diagnostic file name"
    )
    file_min_digits: int | None = Field(
        default=None,
        description="Minimum number of digits for the time step number in the file name",
    )
    dump_rz_modes: bool | None = Field(
        default=None, description="Flag whether to dump the data for all RZ modes"
    )
    dump_last_timestep: bool | None = Field(
        default=None,
        description="If true, the last timestep is dumped regardless of the diagnostic period/intervals.",
    )
    particle_fields_to_plot: list[ParticleFieldDiagnostic] = Field(
        default_factory=list,
        description="List of ParticleFieldDiagnostic classes to install in the simulation. Error checking is handled in the class itself.",
    )
    particle_fields_species: list[str] | None = Field(
        default=None,
        description="Species for which to calculate particle_fields_to_plot functions. Fields will be calculated separately for each specified species. If not passed, default is all of the available particle species.",
    )
    verbose: int | None = Field(
        default=None,
        description="Verbosity level to use for printing diagnostic output information.",
    )

    # Runtime state populated during diagnostic_initialize_inputs / WarpXDiagnosticBase.
    _diagnostic: pywarpx.Diagnostics.Diagnostic | None = PrivateAttr(default=None)

    def diagnostic_initialize_inputs(self):
        self.add_diagnostic()

        self._diagnostic.diag_type = "Full"
        self._diagnostic.format = self.format
        self._diagnostic.openpmd_backend = self.openpmd_backend
        self._diagnostic.openpmd_encoding = self.openpmd_encoding
        self._diagnostic.file_min_digits = self.file_min_digits
        self._diagnostic.dump_rz_modes = self.dump_rz_modes
        self._diagnostic.dump_last_timestep = self.dump_last_timestep
        self._diagnostic.intervals = self.period
        self._diagnostic.set_or_replace_attr("verbose", self.verbose)
        self._diagnostic.diag_lo = self.lower_bound
        self._diagnostic.diag_hi = self.upper_bound
        if self.number_of_cells is not None:
            self._diagnostic.coarsening_ratio = (
                np.array(self.grid.number_of_cells) / np.array(self.number_of_cells)
            ).astype(int)

        # --- Use a set to ensure that fields don't get repeated.
        fields_to_plot = set()

        if pywarpx.geometry.dims == "RZ":
            E_fields_list = ["Er", "Et", "Ez"]
            B_fields_list = ["Br", "Bt", "Bz"]
            J_fields_list = ["Jr", "Jt", "Jz"]
            J_displacement_fields_list = [
                "Jr_displacement",
                "Jt_displacement",
                "Jz_displacement",
            ]
            A_fields_list = ["Ar", "At", "Az"]
        else:
            E_fields_list = ["Ex", "Ey", "Ez"]
            B_fields_list = ["Bx", "By", "Bz"]
            J_fields_list = ["Jx", "Jy", "Jz"]
            J_displacement_fields_list = [
                "Jx_displacement",
                "Jy_displacement",
                "Jz_displacement",
            ]
            A_fields_list = ["Ax", "Ay", "Az"]
        if self.data_list is not None:
            for dataname in self.data_list:
                if dataname == "E":
                    for field_name in E_fields_list:
                        fields_to_plot.add(field_name)
                elif dataname == "B":
                    for field_name in B_fields_list:
                        fields_to_plot.add(field_name)
                elif dataname == "J":
                    for field_name in J_fields_list:
                        fields_to_plot.add(field_name.lower())
                elif dataname == "J_displacement":
                    for field_name in J_displacement_fields_list:
                        fields_to_plot.add(field_name.lower())
                elif dataname == "A":
                    for field_name in A_fields_list:
                        fields_to_plot.add(field_name)
                elif dataname in J_fields_list:
                    fields_to_plot.add(dataname.lower())
                elif dataname in J_displacement_fields_list:
                    fields_to_plot.add(dataname.lower())
                elif dataname == "dive":
                    fields_to_plot.add("divE")
                elif dataname == "divb":
                    fields_to_plot.add("divB")
                elif dataname == "proc_number":
                    fields_to_plot.add("proc_num")
                elif dataname == "raw_fields":
                    self.plot_raw_fields = 1
                elif dataname == "raw_fields_guards":
                    self.plot_raw_fields_guards = 1
                elif dataname == "finepatch":
                    self.plot_finepatch = 1
                elif dataname == "crsepatch":
                    self.plot_crsepatch = 1
                else:
                    # Pass field names through to C++ for resolution and validation.
                    # This includes known diagnostic quantities as well as fields
                    # registered in the MultiFabRegister. C++ raises a descriptive
                    # error if the name is not valid.
                    fields_to_plot.add(dataname)

            # --- Convert the set to a sorted list so that the order
            # --- is the same on all processors.
            fields_to_plot = list(fields_to_plot)
            fields_to_plot.sort()
            self._diagnostic.set_or_replace_attr("fields_to_plot", fields_to_plot)

        particle_fields_to_plot_names = list()
        for pfd in self.particle_fields_to_plot:
            if pfd.name in particle_fields_to_plot_names:
                raise Exception("A particle fields name can not be repeated.")
            particle_fields_to_plot_names.append(pfd.name)
            self._diagnostic.__setattr__(
                f"particle_fields.{pfd.name}(x,y,z,ux,uy,uz)", pfd.func
            )
            self._diagnostic.__setattr__(
                f"particle_fields.{pfd.name}.do_average", pfd.do_average
            )
            self._diagnostic.__setattr__(
                f"particle_fields.{pfd.name}.filter(x,y,z,ux,uy,uz)", pfd.filter
            )

        # --- Convert to a sorted list so that the order
        # --- is the same on all processors.
        particle_fields_to_plot_names.sort()
        self._diagnostic.particle_fields_to_plot = particle_fields_to_plot_names
        self._diagnostic.particle_fields_species = self.particle_fields_species
        self._diagnostic.plot_raw_fields = self.plot_raw_fields
        self._diagnostic.plot_raw_fields_guards = self.plot_raw_fields_guards
        self._diagnostic.plot_finepatch = self.plot_finepatch
        self._diagnostic.plot_crsepatch = self.plot_crsepatch
        if "write_species" not in self._diagnostic.argvattrs:
            self._diagnostic.write_species = False
        self.set_write_dir()


ElectrostaticFieldDiagnostic = FieldDiagnostic


class TimeAveragedFieldDiagnostic(FieldDiagnostic):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_FieldDiagnostic)
    )

    time_average_mode: str | None = Field(
        default=None,
        description=(
            "Type of time averaging diagnostic\n"
            'Supported values include ``"none"``, ``"fixed_start"``, and ``"dynamic_start"``\n'
            "\n"
            '* ``"none"`` for no averaging (instantaneous fields)\n'
            '* ``"fixed_start"`` for a diagnostic that averages all fields between the current output step and a fixed point in time\n'
            '* ``"dynamic_start"`` for a constant averaging period and output at different points in time (non-overlapping)'
        ),
    )
    average_period_steps: int | None = Field(
        default=None,
        description='Configures the number of time steps in an averaging period. Set this only in the ``"dynamic_start"`` mode and only if ``warpx_average_period_time`` has not already been set. Will be ignored in the ``"fixed_start"`` mode (with warning).',
    )
    average_period_time: float | None = Field(
        default=None,
        description='Configures the time (SI units) in an averaging period. Set this only in the ``"dynamic_start"`` mode and only if ``average_period_steps`` has not already been set. Will be ignored in the ``"fixed_start"`` mode (with warning).',
    )
    average_start_step: int | None = Field(
        default=None,
        description='Configures the time step at which time-averaging begins. Set this only in the ``"fixed_start"`` mode. Will be ignored in the ``"dynamic_start"`` mode (with warning).',
    )

    def diagnostic_initialize_inputs(self):
        super().diagnostic_initialize_inputs()

        self._diagnostic.set_or_replace_attr("diag_type", "TimeAveraged")

        if "write_species" not in self._diagnostic.argvattrs:
            self._diagnostic.write_species = False

        self._diagnostic.time_average_mode = self.time_average_mode
        self._diagnostic.average_period_steps = self.average_period_steps
        self._diagnostic.average_period_time = self.average_period_time
        self._diagnostic.average_start_step = self.average_start_step


class Checkpoint(picmistandard.PICMI_DiagnosticExtension, WarpXDiagnosticBase):
    """
    Sets up checkpointing of the simulation, allowing for later restarts

    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    period: int | str = Field(
        default=1,
        description="Period of time steps at which the checkpoint is written; WarpX also accepts the Intervals parser string syntax (e.g. '::10').",
    )
    write_dir: str | None = Field(
        default=None, description="Directory where the checkpoints are written"
    )
    name: str = Field(
        default="chkpoint", description="Name of the checkpoint diagnostic"
    )
    file_prefix: str | None = Field(
        default=None,
        alias="warpx_file_prefix",
        description="The prefix to the checkpoint directory names",
    )
    file_min_digits: int | None = Field(
        default=None,
        alias="warpx_file_min_digits",
        description="Minimum number of digits for the time step number in the checkpoint directory name.",
    )
    verbose: int | None = Field(
        default=None,
        alias="warpx_verbose",
        description="Verbosity level to use for printing diagnostic output information.",
    )

    # Runtime state populated during diagnostic_initialize_inputs / WarpXDiagnosticBase.
    _diagnostic: pywarpx.Diagnostics.Diagnostic | None = PrivateAttr(default=None)

    def diagnostic_initialize_inputs(self):
        self.add_diagnostic()

        self._diagnostic.intervals = self.period
        self._diagnostic.diag_type = "Full"
        self._diagnostic.format = "checkpoint"
        self._diagnostic.file_min_digits = self.file_min_digits
        self._diagnostic.set_or_replace_attr("verbose", self.verbose)

        self.set_write_dir()


class ParticleDiagnostic(picmistandard.PICMI_ParticleDiagnostic, WarpXDiagnosticBase):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_ParticleDiagnostic)
    )

    format: str = Field(default="plotfile", description="Diagnostic file format")
    openpmd_backend: str | None = Field(
        default=None, description="Openpmd backend file format"
    )
    openpmd_encoding: str | None = Field(
        default=None,
        description="Only read if ``<diag_name>.format = openpmd``. openPMD file output encoding. File based: one file per timestep (slower), group/variable based: one file for all steps (faster)). Variable based is an experimental feature with ADIOS2. Default: `'f'`.",
    )
    file_prefix: str | None = Field(
        default=None, description="Prefix on the diagnostic file name"
    )
    file_min_digits: int | None = Field(
        default=None,
        description="Minimum number of digits for the time step number in the file name",
    )
    period: int | str = Field(
        description="Period of time steps at which the diagnostic is performed; WarpX also accepts the Intervals parser string syntax (e.g. '::10')."
    )
    random_fraction: float | dict[Species, float] | None = Field(
        default=None,
        description="Random fraction of particles to include in the diagnostic. If a float is given the same fraction will be used for all species, if a dictionary is given the keys should be species with the value specifying the random fraction for that species.",
    )
    uniform_stride: int | dict[Species, int] | None = Field(
        default=None,
        description="Stride to down select to the particles to include in the diagnostic. If an integer is given the same stride will be used for all species, if a dictionary is given the keys should be species with the value specifying the stride for that species.",
    )
    plot_filter_function: str | None = Field(
        default=None,
        description="Analytic expression to down select the particles to in the diagnostic",
    )
    dump_last_timestep: bool | None = Field(
        default=None,
        description="If true, the last timestep is dumped regardless of the diagnostic period/intervals.",
    )
    verbose: int | None = Field(
        default=None,
        description="Verbosity level to use for printing diagnostic output information.",
    )

    # ``warpx_``-prefixed constants referenced in plot_filter_function, collected from the
    # otherwise-unrecognized keyword arguments.
    user_defined_kw: dict = Field(default_factory=dict)

    # Runtime state populated during diagnostic_initialize_inputs / WarpXDiagnosticBase.
    _diagnostic: pywarpx.Diagnostics.Diagnostic | None = PrivateAttr(default=None)
    _mangle_dict: dict | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _collect_plot_filter_kw(cls, data):
        return _collect_warpx_constants(cls, data, "plot_filter_function")

    def diagnostic_initialize_inputs(self):
        self.add_diagnostic()

        self._diagnostic.diag_type = "Full"
        self._diagnostic.format = self.format
        self._diagnostic.openpmd_backend = self.openpmd_backend
        self._diagnostic.openpmd_encoding = self.openpmd_encoding
        self._diagnostic.file_min_digits = self.file_min_digits
        self._diagnostic.dump_last_timestep = self.dump_last_timestep
        self._diagnostic.intervals = self.period
        self._diagnostic.set_or_replace_attr("verbose", self.verbose)
        self._diagnostic.set_or_replace_attr("write_species", True)
        if "fields_to_plot" not in self._diagnostic.argvattrs:
            self._diagnostic.fields_to_plot = "none"
        self.set_write_dir()

        # --- Use a set to ensure that fields don't get repeated.
        variables = set()

        if self.data_list is not None:
            for dataname in self.data_list:
                if dataname == "position":
                    if pywarpx.geometry.dims != "1":  # because then it's WARPX_DIM_1D_Z
                        variables.add("x")
                    if pywarpx.geometry.dims == "3":
                        variables.add("y")
                    variables.add("z")
                    if pywarpx.geometry.dims == "RZ":
                        variables.add("theta")
                elif dataname == "momentum":
                    variables.add("ux")
                    variables.add("uy")
                    variables.add("uz")
                elif dataname == "weighting":
                    variables.add("w")
                elif dataname == "fields":
                    variables.add("Ex")
                    variables.add("Ey")
                    variables.add("Ez")
                    variables.add("Bx")
                    variables.add("By")
                    variables.add("Bz")
                elif dataname in [
                    "x",
                    "y",
                    "z",
                    "theta",
                    "ux",
                    "uy",
                    "uz",
                    "Ex",
                    "Ey",
                    "Ez",
                    "Bx",
                    "By",
                    "Bz",
                    "Er",
                    "Et",
                    "Br",
                    "Bt",
                ]:
                    if pywarpx.geometry.dims == "1" and (
                        dataname == "x" or dataname == "y"
                    ):
                        raise RuntimeError(
                            f"The attribute {dataname} is not available in mode WARPX_DIM_1D_Z"
                            f"chosen by dim={pywarpx.geometry.dims} in pywarpx."
                        )
                    elif pywarpx.geometry.dims != "3" and dataname == "y":
                        raise RuntimeError(
                            f"The attribute {dataname} is not available outside of mode WARPX_DIM_3D"
                            f"The chosen value was dim={pywarpx.geometry.dims} in pywarpx."
                        )
                    elif pywarpx.geometry.dims != "RZ" and dataname == "theta":
                        raise RuntimeError(
                            f"The attribute {dataname} is not available outside of mode WARPX_DIM_RZ."
                            f"The chosen value was dim={pywarpx.geometry.dims} in pywarpx."
                        )
                    else:
                        variables.add(dataname)
                else:
                    # possibly add user defined attributes
                    variables.add(dataname)

            # --- Convert the set to a sorted list so that the order
            # --- is the same on all processors.
            variables = list(variables)
            variables.sort()

        # species list
        if self.species is None:
            species_names = pywarpx.particles.species_names
        elif isinstance(self.species, (list, tuple)):
            species_names = [species.name for species in self.species]
        else:
            species_names = [self.species.name]

        # check if random fraction is specified and whether a value is given per species
        random_fraction = {}
        random_fraction_default = self.random_fraction
        if isinstance(self.random_fraction, dict):
            random_fraction_default = 1.0
            for key, val in self.random_fraction.items():
                random_fraction[key.name] = val

        # check if uniform stride is specified and whether a value is given per species
        uniform_stride = {}
        uniform_stride_default = self.uniform_stride
        if isinstance(self.uniform_stride, dict):
            uniform_stride_default = 1
            for key, val in self.uniform_stride.items():
                uniform_stride[key.name] = val

        if self._mangle_dict is None:
            # Only do this once so that the same variables are used in this distribution
            # is used multiple times
            self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)

        for name in species_names:
            diag = pywarpx.Bucket.Bucket(
                self.name + "." + name,
                variables=variables,
                random_fraction=random_fraction.get(name, random_fraction_default),
                uniform_stride=uniform_stride.get(name, uniform_stride_default),
            )
            expression = pywarpx.my_constants.mangle_expression(
                self.plot_filter_function, self._mangle_dict
            )
            diag.__setattr__("plot_filter_function(t,x,y,z,ux,uy,uz)", expression)
            self._diagnostic._species_dict[name] = diag


# ----------------------------
# Lab frame diagnostics
# ----------------------------


class LabFrameFieldDiagnostic(
    picmistandard.PICMI_LabFrameFieldDiagnostic, WarpXDiagnosticBase
):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html#backtransformed-diagnostics>`__
    for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_LabFrameFieldDiagnostic)
    )

    format: str | None = Field(
        default=None, description="Passed to <diagnostic name>.format"
    )
    openpmd_backend: str | None = Field(
        default=None, description="Passed to <diagnostic name>.openpmd_backend"
    )
    openpmd_encoding: Literal["f", "g"] | None = Field(
        default=None,
        description="Only read if ``<diag_name>.format = openpmd``. openPMD file output encoding: 'f' (file based) or 'g' (group based). File based: one file per timestep (slower), group/variable based: one file for all steps (faster)). Default: `'f'`.",
    )
    file_prefix: str | None = Field(
        default=None, description="Passed to <diagnostic name>.file_prefix"
    )
    intervals: int | str | None = Field(
        default=None,
        description='Selects the snapshots to be made, instead of using "num_snapshots" which makes all snapshots. "num_snapshots" is ignored.',
    )
    file_min_digits: int | None = Field(
        default=None, description="Passed to <diagnostic name>.file_min_digits"
    )
    buffer_size: int | None = Field(
        default=None, description="Passed to <diagnostic name>.buffer_size"
    )
    lower_bound: list[float] | None = Field(
        default=None, description="Passed to <diagnostic name>.lower_bound"
    )
    upper_bound: list[float] | None = Field(
        default=None, description="Passed to <diagnostic name>.upper_bound"
    )
    verbose: int | None = Field(
        default=None,
        description="Verbosity level to use for printing diagnostic output information.",
    )

    # Runtime state populated during diagnostic_initialize_inputs / WarpXDiagnosticBase.
    _diagnostic: pywarpx.Diagnostics.Diagnostic | None = PrivateAttr(default=None)

    def diagnostic_initialize_inputs(self):
        self.add_diagnostic()

        self._diagnostic.diag_type = "BackTransformed"
        self._diagnostic.format = self.format
        self._diagnostic.openpmd_backend = self.openpmd_backend
        self._diagnostic.openpmd_encoding = self.openpmd_encoding
        self._diagnostic.file_min_digits = self.file_min_digits
        self._diagnostic.diag_lo = self.lower_bound
        self._diagnostic.diag_hi = self.upper_bound
        self._diagnostic.set_or_replace_attr("verbose", self.verbose)

        self._diagnostic.do_back_transformed_fields = True
        self._diagnostic.dt_snapshots_lab = self.dt_snapshots
        self._diagnostic.buffer_size = self.buffer_size

        # intervals and num_snapshots_lab cannot both be set
        if self.intervals is not None:
            self._diagnostic.intervals = self.intervals
        else:
            self._diagnostic.num_snapshots_lab = self.num_snapshots

        # --- Use a set to ensure that fields don't get repeated.
        fields_to_plot = set()

        if pywarpx.geometry.dims == "RZ":
            E_fields_list = ["Er", "Et", "Ez"]
            B_fields_list = ["Br", "Bt", "Bz"]
            J_fields_list = ["Jr", "Jt", "Jz"]
        else:
            E_fields_list = ["Ex", "Ey", "Ez"]
            B_fields_list = ["Bx", "By", "Bz"]
            J_fields_list = ["Jx", "Jy", "Jz"]
        if self.data_list is not None:
            for dataname in self.data_list:
                if dataname == "E":
                    for field_name in E_fields_list:
                        fields_to_plot.add(field_name)
                elif dataname == "B":
                    for field_name in B_fields_list:
                        fields_to_plot.add(field_name)
                elif dataname == "J":
                    for field_name in J_fields_list:
                        fields_to_plot.add(field_name.lower())
                elif dataname in E_fields_list:
                    fields_to_plot.add(dataname)
                elif dataname in B_fields_list:
                    fields_to_plot.add(dataname)
                elif dataname in J_fields_list:
                    fields_to_plot.add(dataname.lower())
                elif dataname == "rho":
                    # Add rho diagnostic
                    fields_to_plot.add(dataname)

            # --- Convert the set to a sorted list so that the order
            # --- is the same on all processors.
            fields_to_plot = list(fields_to_plot)
            fields_to_plot.sort()
            self._diagnostic.set_or_replace_attr("fields_to_plot", fields_to_plot)

        if "write_species" not in self._diagnostic.argvattrs:
            self._diagnostic.write_species = False
        self.set_write_dir()


class LabFrameParticleDiagnostic(
    picmistandard.PICMI_LabFrameParticleDiagnostic, WarpXDiagnosticBase
):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html#backtransformed-diagnostics>`__
    for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(picmistandard.PICMI_LabFrameParticleDiagnostic)
    )

    format: str | None = Field(
        default=None, description="Passed to <diagnostic name>.format"
    )
    openpmd_backend: str | None = Field(
        default=None, description="Passed to <diagnostic name>.openpmd_backend"
    )
    openpmd_encoding: Literal["f", "g"] | None = Field(
        default=None,
        description="Only read if ``<diag_name>.format = openpmd``. openPMD file output encoding: 'f' (file based) or 'g' (group based). File based: one file per timestep (slower), group/variable based: one file for all steps (faster)). Default: `'f'`.",
    )
    file_prefix: str | None = Field(
        default=None, description="Passed to <diagnostic name>.file_prefix"
    )
    intervals: int | str | None = Field(
        default=None,
        description='Selects the snapshots to be made, instead of using "num_snapshots" which makes all snapshots. "num_snapshots" is ignored.',
    )
    file_min_digits: int | None = Field(
        default=None, description="Passed to <diagnostic name>.file_min_digits"
    )
    buffer_size: int | None = Field(
        default=None, description="Passed to <diagnostic name>.buffer_size"
    )
    verbose: int | None = Field(
        default=None,
        description="Verbosity level to use for printing diagnostic output information.",
    )

    # Runtime state populated during diagnostic_initialize_inputs / WarpXDiagnosticBase.
    _diagnostic: pywarpx.Diagnostics.Diagnostic | None = PrivateAttr(default=None)

    def diagnostic_initialize_inputs(self):
        self.add_diagnostic()

        self._diagnostic.diag_type = "BackTransformed"
        self._diagnostic.format = self.format
        self._diagnostic.openpmd_backend = self.openpmd_backend
        self._diagnostic.openpmd_encoding = self.openpmd_encoding
        self._diagnostic.file_min_digits = self.file_min_digits
        self._diagnostic.set_or_replace_attr("verbose", self.verbose)

        self._diagnostic.do_back_transformed_particles = True
        self._diagnostic.dt_snapshots_lab = self.dt_snapshots
        self._diagnostic.buffer_size = self.buffer_size

        # intervals and num_snapshots_lab cannot both be set
        if self.intervals is not None:
            self._diagnostic.intervals = self.intervals
        else:
            self._diagnostic.num_snapshots_lab = self.num_snapshots

        self._diagnostic.do_back_transformed_fields = False

        self._diagnostic.set_or_replace_attr("write_species", True)
        if "fields_to_plot" not in self._diagnostic.argvattrs:
            self._diagnostic.fields_to_plot = "none"

        self.set_write_dir()

        # --- Use a set to ensure that fields don't get repeated.
        variables = set()

        if self.data_list is not None:
            for dataname in self.data_list:
                if dataname == "position":
                    if pywarpx.geometry.dims != "1":  # because then it's WARPX_DIM_1D_Z
                        variables.add("x")
                    if pywarpx.geometry.dims == "3":
                        variables.add("y")
                    variables.add("z")
                    if pywarpx.geometry.dims == "RZ":
                        variables.add("theta")
                elif dataname == "momentum":
                    variables.add("ux")
                    variables.add("uy")
                    variables.add("uz")
                elif dataname == "weighting":
                    variables.add("w")
                elif dataname == "fields":
                    variables.add("Ex")
                    variables.add("Ey")
                    variables.add("Ez")
                    variables.add("Bx")
                    variables.add("By")
                    variables.add("Bz")
                elif dataname in [
                    "x",
                    "y",
                    "z",
                    "theta",
                    "ux",
                    "uy",
                    "uz",
                    "Ex",
                    "Ey",
                    "Ez",
                    "Bx",
                    "By",
                    "Bz",
                    "Er",
                    "Et",
                    "Br",
                    "Bt",
                ]:
                    if pywarpx.geometry.dims == "1" and (
                        dataname == "x" or dataname == "y"
                    ):
                        raise RuntimeError(
                            f"The attribute {dataname} is not available in mode WARPX_DIM_1D_Z"
                            f"chosen by dim={pywarpx.geometry.dims} in pywarpx."
                        )
                    elif pywarpx.geometry.dims != "3" and dataname == "y":
                        raise RuntimeError(
                            f"The attribute {dataname} is not available outside of mode WARPX_DIM_3D"
                            f"The chosen value was dim={pywarpx.geometry.dims} in pywarpx."
                        )
                    elif pywarpx.geometry.dims != "RZ" and dataname == "theta":
                        raise RuntimeError(
                            f"The attribute {dataname} is not available outside of mode WARPX_DIM_RZ."
                            f"The chosen value was dim={pywarpx.geometry.dims} in pywarpx."
                        )
                    else:
                        variables.add(dataname)

            # --- Convert the set to a sorted list so that the order
            # --- is the same on all processors.
            variables = list(variables)
            variables.sort()

        # species list
        if self.species is None:
            species_names = pywarpx.particles.species_names
        elif isinstance(self.species, (list, tuple)):
            species_names = [species.name for species in self.species]
        else:
            species_names = [self.species.name]

        for name in species_names:
            diag = pywarpx.Bucket.Bucket(self.name + "." + name, variables=variables)
            self._diagnostic._species_dict[name] = diag


class ReducedDiagnostic(
    picmistandard.PICMI_DiagnosticExtension,
    picmistandard.PICMI_ExpressionParameters,
    WarpXDiagnosticBase,
):
    """
    Sets up a reduced diagnostic in the simulation.

    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html#reduced-diagnostics>`__
    for more information.

    Which parameters are used depends on the type of the diagnostic. Parameters used in the
    expressions can be given as additional keyword arguments.
    """

    _simple_reduced_diagnostics: ClassVar[tuple[str, ...]] = (
        "ParticleEnergy",
        "ParticleMomentum",
        "FieldEnergy",
        "FieldMomentum",
        "FieldMaximum",
        "FieldPoyntingFlux",
        "RhoMaximum",
        "ParticleNumber",
        "LoadBalanceCosts",
        "LoadBalanceEfficiency",
        "Timestep",
    )
    # these diagnostics require a species
    _species_reduced_diagnostics: ClassVar[tuple[str, ...]] = (
        "BeamRelevant",
        "ParticleHistogram",
        "ParticleHistogram2D",
        "ParticleExtrema",
    )
    # per type: the parameters and their input names, with the arguments of expressions
    _type_inputs: ClassVar[dict[str, dict[str, str]]] = {
        "FieldProbe": {
            name: name
            for name in (
                "probe_geometry",
                "x_probe",
                "y_probe",
                "z_probe",
                "interp_order",
                "integrate",
                "do_moving_window_FP",
                "resolution",
                "x1_probe",
                "y1_probe",
                "z1_probe",
                "detector_radius",
                "target_normal_x",
                "target_normal_y",
                "target_normal_z",
                "target_up_x",
                "target_up_y",
                "target_up_z",
            )
        },
        "ParticleHistogram": {
            "bin_number": "bin_number",
            "bin_max": "bin_max",
            "bin_min": "bin_min",
            "normalization": "normalization",
            "histogram_function": "histogram_function(t,x,y,z,ux,uy,uz)",
            "filter_function": "filter_function(t,x,y,z,ux,uy,uz)",
        },
        "ParticleHistogram2D": {
            "bin_number_abs": "bin_number_abs",
            "bin_number_ord": "bin_number_ord",
            "bin_min_abs": "bin_min_abs",
            "bin_max_abs": "bin_max_abs",
            "bin_min_ord": "bin_min_ord",
            "bin_max_ord": "bin_max_ord",
            "histogram_function_abs": "histogram_function_abs(t,x,y,z,ux,uy,uz,w)",
            "histogram_function_ord": "histogram_function_ord(t,x,y,z,ux,uy,uz,w)",
            "filter_function": "filter_function(t,x,y,z,ux,uy,uz,w)",
            "value_function": "value_function(t,x,y,z,ux,uy,uz,w)",
        },
        "FieldReduction": {
            "reduction_type": "reduction_type",
            "reduced_function": "reduced_function(x,y,z,Ex,Ey,Ez,Bx,By,Bz,jx,jy,jz)",
        },
        "ChargeOnEB": {
            "weighting_function": "weighting_function(x,y,z)",
        },
    }
    _expression_fields: ClassVar[tuple[str, ...]] = (
        "histogram_function",
        "filter_function",
        "histogram_function_abs",
        "histogram_function_ord",
        "value_function",
        "reduced_function",
        "weighting_function",
    )

    diag_type: Literal[
        "ParticleEnergy",
        "ParticleMomentum",
        "FieldEnergy",
        "FieldMomentum",
        "FieldMaximum",
        "FieldPoyntingFlux",
        "RhoMaximum",
        "ParticleNumber",
        "LoadBalanceCosts",
        "LoadBalanceEfficiency",
        "Timestep",
        "BeamRelevant",
        "ParticleHistogram",
        "ParticleHistogram2D",
        "ParticleExtrema",
        "FieldProbe",
        "FieldReduction",
        "ChargeOnEB",
    ] = Field(
        description="The type of reduced diagnostic. See the link above for all the different types of reduced diagnostics available."
    )
    name: str | None = Field(
        default=None,
        description="The name of this diagnostic which will also be the name of the data file written to disk.",
    )
    period: int | str | None = Field(
        default=None,
        description="The simulation step interval at which to output this diagnostic.",
    )
    path: str | None = Field(
        default=None,
        description="The file path in which the diagnostic file should be written.",
    )
    extension: str | None = Field(
        default=None, description="The file extension used for the diagnostic output."
    )
    separator: str | None = Field(
        default=None, description="The separator between row values in the output file."
    )
    species: Species | None = Field(
        default=None,
        description="The species for which to calculate the diagnostic, required for diagnostic types 'BeamRelevant', 'ParticleHistogram', 'ParticleHistogram2D', and 'ParticleExtrema'",
    )

    # ParticleHistogram
    bin_number: int | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram', the number of bins used for the histogram",
    )
    bin_max: float | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram', the maximum value of the bins",
    )
    bin_min: float | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram', the minimum value of the bins",
    )
    normalization: (
        Literal["unity_particle_weight", "max_to_unity", "area_to_unity"] | None
    ) = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram', normalization method of the histogram.",
    )
    histogram_function: Expression | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram', the function evaluated to produce the histogram data",
    )
    filter_function: Expression | None = Field(
        default=None,
        description="For diagnostic types 'ParticleHistogram' and 'ParticleHistogram2D', the function to filter whether particles are included in the histogram",
    )

    # ParticleHistogram2D
    bin_max_abs: float | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram2D', the maximum value of the bins for the abscissa axis.",
    )
    bin_max_ord: float | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram2D', the maximum value of the bins for the ordinate axis.",
    )
    bin_min_abs: float | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram2D', the minimum value of the bins for the abscissa axis.",
    )
    bin_min_ord: float | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram2D', the minimum value of the bins for the ordinate axis.",
    )
    bin_number_abs: int | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram2D', the number of bins used for the histogram for the abscissa axis.",
    )
    bin_number_ord: int | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram2D', the number of bins used for the histogram for the ordinate axis.",
    )
    histogram_function_abs: Expression | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram2D', the histogram function for the abscissa axis.",
    )
    histogram_function_ord: Expression | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram2D', the histogram function for the ordinate axis.",
    )
    value_function: Expression | None = Field(
        default=None,
        description="For diagnostic type 'ParticleHistogram2D', the expression for the weight used to calculate the histogram.",
    )

    # FieldReduction
    reduced_function: Expression | None = Field(
        default=None,
        description="For diagnostic type 'FieldReduction', the function of the fields to evaluate",
    )
    reduction_type: Literal["Maximum", "Minimum", "Integral"] | None = Field(
        default=None,
        description="For diagnostic type 'FieldReduction', the type of reduction",
    )

    # ChargeOnEB
    weighting_function: Expression | None = Field(
        default=None,
        description="For diagnostic type 'ChargeOnEB', the function to weight contributions to the total charge",
    )

    # FieldProbe
    probe_geometry: str | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the geometry of the probe: 'Point', 'Line', or 'Plane'",
    )
    integrate: bool | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', whether the field is integrated (default False)",
    )
    do_moving_window_FP: bool | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', whether the moving window is followed (default False)",
    )
    x_probe: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', a probe location. For 'Point', the location of the point. For 'Line', the start of the line. For 'Plane', the center of the square detector.",
    )
    y_probe: float | None = Field(
        default=None, description="For diagnostic type 'FieldProbe', see ``x_probe``"
    )
    z_probe: float | None = Field(
        default=None, description="For diagnostic type 'FieldProbe', see ``x_probe``"
    )
    interp_order: int | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the interpolation order for 'Line' and 'Plane'",
    )
    resolution: int | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the number of points along the 'Line' or along each edge of the square 'Plane'",
    )
    x1_probe: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the end point for 'Line'",
    )
    y1_probe: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the end point for 'Line'",
    )
    z1_probe: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the end point for 'Line'",
    )
    detector_radius: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the detector \"radius\" (half edge length) of the 'Plane'",
    )
    target_normal_x: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the normal vector to the 'Plane'. Only applicable in 3D",
    )
    target_normal_y: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the normal vector to the 'Plane'. Only applicable in 3D",
    )
    target_normal_z: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the normal vector to the 'Plane'. Only applicable in 3D",
    )
    target_up_x: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the vector specifying up in the 'Plane'",
    )
    target_up_y: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the vector specifying up in the 'Plane'",
    )
    target_up_z: float | None = Field(
        default=None,
        description="For diagnostic type 'FieldProbe', the vector specifying up in the 'Plane'",
    )

    user_defined_kw: dict = Field(
        default_factory=dict,
        description="Constants referenced in the expressions, collected from otherwise-unrecognized keyword arguments.",
    )

    # Runtime state populated during diagnostic_initialize_inputs / WarpXDiagnosticBase.
    _diagnostic: pywarpx.Diagnostics.Diagnostic | None = PrivateAttr(default=None)
    _mangle_dict: dict | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_parameters_of_type(self) -> Self:
        required = []
        if self.diag_type in self._species_reduced_diagnostics:
            required.append("species")
        elif self.species is not None:
            raise ValueError(
                f"species is not used by the {self.diag_type} reduced diagnostic"
            )

        used = set(self._type_inputs.get(self.diag_type, {}))
        if self.diag_type == "ParticleHistogram":
            required += ["bin_number", "bin_max", "bin_min", "histogram_function"]
        elif self.diag_type == "ParticleHistogram2D":
            required += [
                "bin_number_abs",
                "bin_number_ord",
                "bin_min_abs",
                "bin_max_abs",
                "bin_min_ord",
                "bin_max_ord",
                "histogram_function_abs",
                "histogram_function_ord",
            ]
        elif self.diag_type == "FieldReduction":
            required += ["reduction_type", "reduced_function"]
        elif self.diag_type == "FieldProbe":
            required += ["probe_geometry", "z_probe"]
            geometry = (self.probe_geometry or "").lower()
            if geometry != "point":
                required.append("resolution")
            if geometry != "line":
                used -= {"x1_probe", "y1_probe", "z1_probe"}
            else:
                required.append("z1_probe")
            if geometry != "plane":
                used -= {
                    "detector_radius",
                    "target_normal_x",
                    "target_normal_y",
                    "target_normal_z",
                    "target_up_x",
                    "target_up_y",
                    "target_up_z",
                }
            else:
                required.append("detector_radius")

        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"The {self.diag_type} reduced diagnostic requires: {', '.join(missing)}"
            )
        type_specific = {
            name for inputs in self._type_inputs.values() for name in inputs
        }
        unused = sorted(
            name for name in type_specific - used if getattr(self, name) is not None
        )
        if unused:
            raise ValueError(
                f"Not used by the {self.diag_type} reduced diagnostic: {', '.join(unused)}"
            )
        return self

    def diagnostic_initialize_inputs(self):
        self.add_diagnostic()

        self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)

        self._diagnostic.type = self.diag_type
        self._diagnostic.intervals = self.period
        self._diagnostic.path = self.path
        self._diagnostic.extension = self.extension
        self._diagnostic.separator = self.separator
        if self.species is not None:
            self._diagnostic.species = self.species.name

        for field_name, input_name in self._type_inputs.get(self.diag_type, {}).items():
            value = getattr(self, field_name)
            if input_name.endswith(")"):
                # Analytic expressions require processing to deal with constants
                value = pywarpx.my_constants.mangle_expression(value, self._mangle_dict)
            self._diagnostic.__setattr__(input_name, value)


class ParticleBoundaryScrapingDiagnostic(
    picmistandard.PICMI_ParticleBoundaryScrapingDiagnostic, WarpXDiagnosticBase
):
    """
    See `Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html>`__ for more information.
    """

    model_config = ConfigDict(
        alias_generator=warpx_options(
            picmistandard.PICMI_ParticleBoundaryScrapingDiagnostic
        )
    )

    format: str = Field(default="openpmd", description="Diagnostic file format")
    openpmd_backend: Literal["bp", "h5", "json"] | None = Field(
        default=None, description="Openpmd backend file format"
    )
    openpmd_encoding: Literal["v", "f", "g"] | None = Field(
        default=None,
        description="Only read if ``<diag_name>.format = openpmd``. openPMD file output encoding: 'v' (variable based), 'f' (file based) or 'g' (group based). File based: one file per timestep (slower), group/variable based: one file for all steps (faster)). Variable based is an experimental feature with ADIOS2. Default: `'f'`.",
    )
    file_prefix: str | None = Field(
        default=None, description="Prefix on the diagnostic file name"
    )
    file_min_digits: int | None = Field(
        default=None,
        description="Minimum number of digits for the time step number in the file name",
    )
    random_fraction: float | dict[Species, float] | None = Field(
        default=None,
        description="Random fraction of particles to include in the diagnostic. If a float is given the same fraction will be used for all species, if a dictionary is given the keys should be species with the value specifying the random fraction for that species.",
    )
    uniform_stride: int | dict[Species, int] | None = Field(
        default=None,
        description="Stride to down select to the particles to include in the diagnostic. If an integer is given the same stride will be used for all species, if a dictionary is given the keys should be species with the value specifying the stride for that species.",
    )
    plot_filter_function: str | None = Field(
        default=None,
        description="Analytic expression to down select the particles to in the diagnostic",
    )
    dump_last_timestep: bool | None = Field(
        default=None,
        description="If true, the last timestep is dumped regardless of the diagnostic period/intervals.",
    )

    # ``warpx_``-prefixed constants referenced in plot_filter_function, collected from the
    # otherwise-unrecognized keyword arguments.
    user_defined_kw: dict = Field(default_factory=dict)

    # Runtime state populated during diagnostic_initialize_inputs / WarpXDiagnosticBase.
    _diagnostic: pywarpx.Diagnostics.Diagnostic | None = PrivateAttr(default=None)
    _mangle_dict: dict | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _collect_plot_filter_kw(cls, data):
        return _collect_warpx_constants(cls, data, "plot_filter_function")

    def diagnostic_initialize_inputs(self):
        self.add_diagnostic()

        self._diagnostic.diag_type = "BoundaryScraping"
        self._diagnostic.format = self.format
        self._diagnostic.openpmd_backend = self.openpmd_backend
        self._diagnostic.openpmd_encoding = self.openpmd_encoding
        self._diagnostic.file_min_digits = self.file_min_digits
        self._diagnostic.dump_last_timestep = self.dump_last_timestep
        self._diagnostic.intervals = self.period
        self._diagnostic.set_or_replace_attr("write_species", True)
        if "fields_to_plot" not in self._diagnostic.argvattrs:
            self._diagnostic.fields_to_plot = "none"

        self.set_write_dir()

        # --- Use a set to ensure that fields don't get repeated.
        variables = set()

        if self.data_list is not None:
            for dataname in self.data_list:
                if dataname == "position":
                    if pywarpx.geometry.dims != "1":  # because then it's WARPX_DIM_1D_Z
                        variables.add("x")
                    if pywarpx.geometry.dims == "3":
                        variables.add("y")
                    variables.add("z")
                    if pywarpx.geometry.dims == "RZ":
                        variables.add("theta")
                elif dataname == "momentum":
                    variables.add("ux")
                    variables.add("uy")
                    variables.add("uz")
                elif dataname == "weighting":
                    variables.add("w")
                elif dataname in ["x", "y", "z", "theta", "ux", "uy", "uz"]:
                    if pywarpx.geometry.dims == "1" and (
                        dataname == "x" or dataname == "y"
                    ):
                        raise RuntimeError(
                            f"The attribute {dataname} is not available in mode WARPX_DIM_1D_Z"
                            f"chosen by dim={pywarpx.geometry.dims} in pywarpx."
                        )
                    elif pywarpx.geometry.dims != "3" and dataname == "y":
                        raise RuntimeError(
                            f"The attribute {dataname} is not available outside of mode WARPX_DIM_3D"
                            f"The chosen value was dim={pywarpx.geometry.dims} in pywarpx."
                        )
                    elif pywarpx.geometry.dims != "RZ" and dataname == "theta":
                        raise RuntimeError(
                            f"The attribute {dataname} is not available outside of mode WARPX_DIM_RZ."
                            f"The chosen value was dim={pywarpx.geometry.dims} in pywarpx."
                        )
                    else:
                        variables.add(dataname)
                else:
                    # possibly add user defined attributes
                    variables.add(dataname)

            # --- Convert the set to a sorted list so that the order
            # --- is the same on all processors.
            variables = list(variables)
            variables.sort()

        # species list
        if self.species is None:
            species_names = pywarpx.particles.species_names
        elif isinstance(self.species, (list, tuple)):
            species_names = [species.name for species in self.species]
        else:
            species_names = [self.species.name]

        # check if random fraction is specified and whether a value is given per species
        random_fraction = {}
        random_fraction_default = self.random_fraction
        if isinstance(self.random_fraction, dict):
            random_fraction_default = 1.0
            for key, val in self.random_fraction.items():
                random_fraction[key.name] = val

        # check if uniform stride is specified and whether a value is given per species
        uniform_stride = {}
        uniform_stride_default = self.uniform_stride
        if isinstance(self.uniform_stride, dict):
            uniform_stride_default = 1
            for key, val in self.uniform_stride.items():
                uniform_stride[key.name] = val

        if self._mangle_dict is None:
            # Only do this once so that the same variables are used in this distribution
            # is used multiple times
            self._mangle_dict = pywarpx.my_constants.add_keywords(self.user_defined_kw)

        for name in species_names:
            diag = pywarpx.Bucket.Bucket(
                self.name + "." + name,
                variables=variables,
                random_fraction=random_fraction.get(name, random_fraction_default),
                uniform_stride=uniform_stride.get(name, uniform_stride_default),
            )
            expression = pywarpx.my_constants.mangle_expression(
                self.plot_filter_function, self._mangle_dict
            )
            diag.__setattr__("plot_filter_function(t,x,y,z,ux,uy,uz)", expression)
            self._diagnostic._species_dict[name] = diag
