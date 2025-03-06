#! /usr/bin/env python
# Copyright 2022 Neil Zaim, Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import re
import sys

import numpy as np
import scipy.constants as scc
import yt

## This script performs various checks for the fusion module. The simulation
## that we check is made of 2 different tests, each with different reactant and product species.
##
## The first test is performed in the center of mass frame of the reactant. It could correspond to the
## physical case of two beams colliding with each other. The kinetic energy of the colliding
## particles depends on the cell number in the z direction and varies in the few keV to few MeV
## range. All the particles within a cell have the exact same momentum. The reactant species have the same
## density and number of particles in this test. The number of product species is much smaller than
## the initial number of reactants
##
## The second test is performed in the rest frame of the second reactant. It corresponds to the
## physical case of a low density beam colliding with a high-density mixed target. The energy of the
## beam particles is varied in the few keV to few MeV range, depending on the cell number in the z
## direction. As in the previous case, all the particles within a cell have the exact same
## momentum. In this test, there are 100 immobile macroparticles for each reactant per cell, as well as 900
## of the beam reactant macroparticles per cell. The density of the immobile particles is 6 orders of
## magnitude higher than the number of beam particles, which means that they have a much higher
## weight. This test is similar to the example given in section 3 of Higginson et al.,
## Journal of Computation Physics, 388 439–453 (2019), which was found to be sensitive to the way
## unsampled pairs are accounted for. As before, the number of product particles is much smaller than
## the initial number of reactants.
##
## In all simulations, we check particle number, charge, momentum and energy conservation and
## perform basic checks regarding the produced particles. When possible, we also compare the number
## of produced macroparticles, fusion yield and energy of the produced particles to theoretical
## values.
##
## Please be aware that the relative tolerances are often set empirically in this analysis script,
## so it would not be surprising that some tolerances need to be increased in the future.

default_tol = 1.0e-12  # Default relative tolerance

## Some physical parameters
keV_to_Joule = scc.e * 1e3
MeV_to_Joule = scc.e * 1e6
barn_to_square_meter = 1.0e-28

## Checks whether this is the 2D or the 3D test
with open("./warpx_used_inputs", "r") as f:
    warpx_used_inputs = f.read()
if re.search("geometry.dims = RZ", warpx_used_inputs):
    is_RZ = True
else:
    is_RZ = False

## Check which kind of test we are doing: D+T or D+D
# Define reactants and products
if re.search("tritium", warpx_used_inputs):
    # If tritium appears in the file, than this is the D+T test
    reaction_type = "DT"
    reactant_species = ["deuterium", "tritium"]
    product_species = ["helium4", "neutron"]
    ntests = 2
    E_fusion = 17.5893 * MeV_to_Joule  # Energy released during the fusion reaction
else:
    # else, this is the D+D test
    reaction_type = "DD"
    reactant_species = ["deuterium", "hydrogen2"]
    product_species = ["helium3", "neutron"]
    ntests = 1
    E_fusion = 3.268911e6 * MeV_to_Joule

mass = {
    "deuterium": 2.01410177812 * scc.m_u,
    "hydrogen2": 2.01410177812 * scc.m_u,
    "tritium": 3.0160492779 * scc.m_u,
    "helium3": 3.016029 * scc.m_u,
    "helium4": 4.00260325413 * scc.m_u,
    "neutron": 1.0013784193052508 * scc.m_p,
}
m_reduced = np.product([mass[s] for s in reactant_species]) / np.sum(
    [mass[s] for s in reactant_species]
)

## Some numerical parameters for this test
size_x = 8
size_y = 8
size_z = 16
if is_RZ:
    dV_slice = np.pi * size_x**2
    yt_z_string = "particle_position_y"
    nppcell_1 = 10000 * 8
    nppcell_2 = 900 * 8
else:
    dV_slice = size_x * size_y
    yt_z_string = "particle_position_z"
    nppcell_1 = 10000
    nppcell_2 = 900
# Volume of a slice corresponding to a single cell in the z direction. In tests 1 and 2, all the
# particles of a given species in the same slice have the exact same momentum

# In test 1 and 2, the energy in cells number i (in z direction) is typically Energy_step * i**2
Energy_step = 22.0 * keV_to_Joule


def is_close(val1, val2, rtol=default_tol, atol=0.0):
    ## Wrapper around numpy.isclose, used to override the default tolerances.
    return np.isclose(val1, val2, rtol=rtol, atol=atol)


def add_existing_species_to_dict(yt_ad, data_dict, species_name, prefix, suffix):
    data_dict[prefix + "_px_" + suffix] = yt_ad[species_name, "particle_momentum_x"].v
    data_dict[prefix + "_py_" + suffix] = yt_ad[species_name, "particle_momentum_y"].v
    data_dict[prefix + "_pz_" + suffix] = yt_ad[species_name, "particle_momentum_z"].v
    data_dict[prefix + "_w_" + suffix] = yt_ad[species_name, "particle_weight"].v
    data_dict[prefix + "_id_" + suffix] = yt_ad[species_name, "particle_id"].v
    data_dict[prefix + "_cpu_" + suffix] = yt_ad[species_name, "particle_cpu"].v
    data_dict[prefix + "_z_" + suffix] = yt_ad[species_name, yt_z_string].v


def add_empty_species_to_dict(data_dict, species_name, prefix, suffix):
    data_dict[prefix + "_px_" + suffix] = np.empty(0)
    data_dict[prefix + "_py_" + suffix] = np.empty(0)
    data_dict[prefix + "_pz_" + suffix] = np.empty(0)
    data_dict[prefix + "_w_" + suffix] = np.empty(0)
    data_dict[prefix + "_id_" + suffix] = np.empty(0)
    data_dict[prefix + "_cpu_" + suffix] = np.empty(0)
    data_dict[prefix + "_z_" + suffix] = np.empty(0)


def add_species_to_dict(yt_ad, data_dict, species_name, prefix, suffix):
    try:
        ## If species exist, we add its data to the dictionary
        add_existing_species_to_dict(yt_ad, data_dict, species_name, prefix, suffix)
    except yt.utilities.exceptions.YTFieldNotFound:
        ## If species does not exist, we avoid python crash and add empty arrays to the
        ## dictionnary.
        add_empty_species_to_dict(data_dict, species_name, prefix, suffix)


def check_particle_number_conservation(data):
    # Check consumption of reactants
    total_w_reactant1_start = np.sum(data[reactant_species[0] + "_w_start"])
    total_w_reactant1_end = np.sum(data[reactant_species[0] + "_w_end"])
    total_w_reactant2_start = np.sum(data[reactant_species[1] + "_w_start"])
    total_w_reactant2_end = np.sum(data[reactant_species[1] + "_w_end"])
    consumed_reactant1 = total_w_reactant1_start - total_w_reactant1_end
    consumed_reactant2 = total_w_reactant2_start - total_w_reactant2_end
    assert consumed_reactant1 >= 0.0
    assert consumed_reactant2 >= 0.0
    ## Check that number of consumed reactants are equal
    assert_scale = max(total_w_reactant1_start, total_w_reactant2_start)
    assert is_close(
        consumed_reactant1,
        consumed_reactant2,
        rtol=0.0,
        atol=default_tol * assert_scale,
    )

    # That the number of products corresponds consumed particles
    for species_name in product_species:
        created_product = np.sum(data[species_name + "_w_end"])
        assert created_product >= 0.0
        assert is_close(
            total_w_reactant1_start, total_w_reactant1_end + created_product
        )
        assert is_close(
            total_w_reactant2_start, total_w_reactant2_end + created_product
        )


def compute_energy_array(data, species_name, suffix, m):
    ## Relativistic computation of kinetic energy for a given species
    psq_array = (
        data[species_name + "_px_" + suffix] ** 2
        + data[species_name + "_py_" + suffix] ** 2
        + data[species_name + "_pz_" + suffix] ** 2
    )
    rest_energy = m * scc.c**2
    return np.sqrt(psq_array * scc.c**2 + rest_energy**2) - rest_energy


def check_energy_conservation(data):
    total_energy_start = 0
    for species_name in reactant_species:
        total_energy_start += np.sum(
            data[species_name + "_w_start"]
            * compute_energy_array(data, species_name, "start", mass[species_name])
        )
    total_energy_end = 0
    for species_name in product_species + reactant_species:
        total_energy_end += np.sum(
            data[species_name + "_w_end"]
            * compute_energy_array(data, species_name, "end", mass[species_name])
        )
    n_fusion_reaction = np.sum(data[product_species[0] + "_w_end"])
    assert is_close(
        total_energy_end, total_energy_start + n_fusion_reaction * E_fusion, rtol=1.0e-8
    )


def check_momentum_conservation(data):
    total_px_start = 0
    total_py_start = 0
    total_pz_start = 0
    for species_name in reactant_species:
        total_px_start += np.sum(
            data[species_name + "_px_start"] * data[species_name + "_w_start"]
        )
        total_py_start += np.sum(
            data[species_name + "_py_start"] * data[species_name + "_w_start"]
        )
        total_pz_start += np.sum(
            data[species_name + "_pz_start"] * data[species_name + "_w_start"]
        )
    total_px_end = 0
    total_py_end = 0
    total_pz_end = 0
    for species_name in reactant_species + product_species:
        total_px_end += np.sum(
            data[species_name + "_px_end"] * data[species_name + "_w_end"]
        )
        total_py_end += np.sum(
            data[species_name + "_py_end"] * data[species_name + "_w_end"]
        )
        total_pz_end += np.sum(
            data[species_name + "_pz_end"] * data[species_name + "_w_end"]
        )

    ## Absolute tolerance is needed because sometimes the initial momentum is exactly 0
    assert is_close(total_px_start, total_px_end, atol=1.0e-15)
    assert is_close(total_py_start, total_py_end, atol=1.0e-15)
    assert is_close(total_pz_start, total_pz_end, atol=1.0e-15)


def check_id(data):
    ## Check that all created particles have unique id + cpu identifier (two particles with
    ## different cpu can have the same id)
    for species_name in product_species:
        complex_id = (
            data[species_name + "_id_end"] + 1j * data[species_name + "_cpu_end"]
        )
        assert complex_id.shape == np.unique(complex_id).shape


def generic_check(data):
    check_particle_number_conservation(data)
    check_energy_conservation(data)
    check_momentum_conservation(data)
    check_id(data)


def check_isotropy(data, relative_tolerance):
    ## Checks that the product particles are emitted isotropically
    for species_name in product_species:
        average_px_sq = np.average(
            data[species_name + "_px_end"] * data[species_name + "_px_end"]
        )
        average_py_sq = np.average(
            data[species_name + "_py_end"] * data[species_name + "_py_end"]
        )
        average_pz_sq = np.average(
            data[species_name + "_pz_end"] * data[species_name + "_pz_end"]
        )
        assert is_close(average_px_sq, average_py_sq, rtol=relative_tolerance)
        assert is_close(average_px_sq, average_pz_sq, rtol=relative_tolerance)


def check_xy_isotropy(data):
    ## Checks that the product particles are emitted isotropically in x and y
    for species_name in product_species:
        average_px_sq = np.average(
            data[species_name + "_px_end"] * data[species_name + "_px_end"]
        )
        average_py_sq = np.average(
            data[species_name + "_py_end"] * data[species_name + "_py_end"]
        )
        average_pz_sq = np.average(
            data[species_name + "_pz_end"] * data[species_name + "_pz_end"]
        )
        assert is_close(average_px_sq, average_py_sq, rtol=5.0e-2)
        assert average_pz_sq > average_px_sq
        assert average_pz_sq > average_py_sq


def cross_section(E_keV):
    ## Returns cross section in b, using the analytical fits given
    ## in H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
    joule_to_keV = 1.0e-3 / scc.e
    B_G = scc.pi * scc.alpha * np.sqrt(2.0 * m_reduced * scc.c**2 * joule_to_keV)
    if reaction_type == "DT":
        A1 = 6.927e4
        A2 = 7.454e8
        A3 = 2.050e6
        A4 = 5.2002e4
        A5 = 0
        B1 = 6.38e1
        B2 = -9.95e-1
        B3 = 6.981e-5
        B4 = 1.728e-4
    elif reaction_type == "DD":
        A1 = 5.3701e4
        A2 = 3.3027e2
        A3 = -1.2706e-1
        A4 = 2.9327e-5
        A5 = -2.5151e-9
        B1 = 0
        B2 = 0
        B3 = 0
        B4 = 0
    else:
        raise RuntimeError(f"Reaction type '{reaction_type}' not implemented.")

    astrophysical_factor = (
        A1 + E_keV * (A2 + E_keV * (A3 + E_keV * (A4 + E_keV * A5)))
    ) / (1 + E_keV * (B1 + E_keV * (B2 + E_keV * (B3 + E_keV * B4))))
    millibarn_to_barn = 1.0e-3
    return (
        millibarn_to_barn * astrophysical_factor / E_keV * np.exp(-B_G / np.sqrt(E_keV))
    )


def E_com_to_p_sq_com(m1, m2, E):
    ## E is the total (kinetic+mass) energy of a two particle (with mass m1 and m2) system in
    ## its center of mass frame, in J.
    ## Returns the square norm of the momentum of each particle in that frame.
    E_ratio = E / ((m1 + m2) * scc.c**2)
    return (
        m1 * m2 * scc.c**2 * (E_ratio**2 - 1)
        + (m1 - m2) ** 2 * scc.c**2 / 4 * (E_ratio - 1.0 / E_ratio) ** 2
    )


def compute_relative_v_com(E):
    ## E is the kinetic energy of reactants in the center of mass frame, in keV
    ## Returns the relative velocity between reactants in this frame, in m/s
    m0 = mass[reactant_species[0]]
    m1 = mass[reactant_species[1]]
    E_J = E * keV_to_Joule + (m0 + m1) * scc.c**2
    p_sq = E_com_to_p_sq_com(m0, m1, E_J)
    p = np.sqrt(p_sq)
    gamma0 = np.sqrt(1.0 + p_sq / (m0 * scc.c) ** 2)
    gamma1 = np.sqrt(1.0 + p_sq / (m1 * scc.c) ** 2)
    v0 = p / (gamma0 * m0)
    v1 = p / (gamma1 * m1)
    return v0 + v1


def expected_weight_com(E_com, reactant0_density, reactant1_density, dV, dt):
    ## Computes expected number of product particles as a function of energy E_com in the
    ## center of mass frame. E_com is in keV.
    assert np.all(E_com >= 0)
    ## Case E_com == 0 is handled manually to avoid division by zero
    conditions = [E_com == 0, E_com > 0]
    ## Necessary to avoid division by 0 warning when pb_cross_section is evaluated
    E_com_never_zero = np.clip(E_com, 1.0e-15, None)
    choices = [
        0.0,
        cross_section(E_com_never_zero) * compute_relative_v_com(E_com_never_zero),
    ]
    sigma_times_vrel = np.select(conditions, choices)
    return (
        reactant0_density
        * reactant1_density
        * sigma_times_vrel
        * barn_to_square_meter
        * dV
        * dt
    )


def check_macroparticle_number(
    data, fusion_probability_target_value, num_pair_per_cell
):
    ## Checks that the number of macroparticles is as expected for the first and second tests

    ## The first slice 0 < z < 1 does not contribute to alpha creation
    if is_RZ:
        numcells = size_x * (size_z - 1)
    else:
        numcells = size_x * size_y * (size_z - 1)
    ## In these tests, the fusion_multiplier is so high that the fusion probability per pair is
    ## equal to the parameter fusion_probability_target_value
    fusion_probability_per_pair = fusion_probability_target_value
    expected_fusion_number = numcells * num_pair_per_cell * fusion_probability_per_pair
    expected_macroparticle_number = 2 * expected_fusion_number
    std_macroparticle_number = 2 * np.sqrt(expected_fusion_number)
    actual_macroparticle_number = data[product_species[0] + "_w_end"].shape[0]
    # 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
    assert is_close(
        actual_macroparticle_number,
        expected_macroparticle_number,
        rtol=0.0,
        atol=5.0 * std_macroparticle_number,
    )

    ## used in subsequent function
    return expected_fusion_number


def p_sq_reactant1_frame_to_E_COM_frame(p_reactant0_sq):
    # Takes the reactant0 square norm of the momentum in the reactant1 rest frame and returns the total
    # kinetic energy in the center of mass frame. Everything is in SI units.
    m0 = mass[reactant_species[0]]
    m1 = mass[reactant_species[1]]

    # Total (kinetic + mass) energy in lab frame
    E_lab = np.sqrt(p_reactant0_sq * scc.c**2 + (m0 * scc.c**2) ** 2) + m1 * scc.c**2
    # Use invariant E**2 - p**2c**2 of 4-momentum norm to compute energy in center of mass frame
    E_com = np.sqrt(E_lab**2 - p_reactant0_sq * scc.c**2)
    # Corresponding kinetic energy
    E_com_kin = E_com - (m1 + m0) * scc.c**2
    return E_com_kin * (p_reactant0_sq > 0.0)


def p_sq_to_kinetic_energy(p_sq, m):
    ## Returns the kinetic energy of a particle as a function of its squared momentum.
    ## Everything is in SI units.
    return np.sqrt(p_sq * scc.c**2 + (m * scc.c**2) ** 2) - (m * scc.c**2)


def compute_E_com1(data):
    ## Computes kinetic energy (in Joule) in the center of frame for the first test

    ## Square norm of the momentum of reactant as a function of cell number in z direction
    p_sq = 2.0 * m_reduced * (Energy_step * np.arange(size_z) ** 2)
    Ekin = 0
    for species_name in reactant_species:
        Ekin += p_sq_to_kinetic_energy(p_sq, mass[species_name])
    return Ekin


def compute_E_com2(data):
    ## Computes kinetic energy (in Joule) in the center of frame for the second test

    ## Square norm of the momentum of reactant0 as a function of cell number in z direction
    p_reactant0_sq = (
        2.0 * mass[reactant_species[0]] * (Energy_step * np.arange(size_z) ** 2)
    )
    return p_sq_reactant1_frame_to_E_COM_frame(p_reactant0_sq)


def check_fusion_yield(
    data, expected_fusion_number, E_com, reactant0_density, reactant1_density, dt
):
    ## Checks that the fusion yield is as expected for the first and second tests.
    product_weight_theory = expected_weight_com(
        E_com / keV_to_Joule, reactant0_density, reactant1_density, dV_slice, dt
    )
    for species_name in product_species:
        product_weight_simulation = np.histogram(
            data[species_name + "_z_end"],
            bins=size_z,
            range=(0, size_z),
            weights=data[species_name + "_w_end"],
        )[0]
        ## -1 is here because the first slice 0 < z < 1 does not contribute to fusion
        expected_fusion_number_per_slice = expected_fusion_number / (size_z - 1)
        relative_std_weight = 1.0 / np.sqrt(expected_fusion_number_per_slice)

        # 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
        assert np.all(
            is_close(
                product_weight_theory,
                product_weight_simulation,
                rtol=5.0 * relative_std_weight,
            )
        )


def specific_check1(data, dt):
    if not is_RZ:
        check_isotropy(data, relative_tolerance=3.0e-2)
    expected_fusion_number = check_macroparticle_number(
        data, fusion_probability_target_value=0.002, num_pair_per_cell=nppcell_1
    )
    E_com = compute_E_com1(data)
    check_fusion_yield(
        data,
        expected_fusion_number,
        E_com,
        reactant0_density=1.0,
        reactant1_density=1.0,
        dt=dt,
    )


def specific_check2(data, dt):
    check_xy_isotropy(data)
    ## Only 900 particles pairs per cell here because we ignore the 10% of reactants that are at rest
    expected_fusion_number = check_macroparticle_number(
        data, fusion_probability_target_value=0.02, num_pair_per_cell=nppcell_2
    )
    E_com = compute_E_com2(data)
    check_fusion_yield(
        data,
        expected_fusion_number,
        E_com,
        reactant0_density=1.0e20,
        reactant1_density=1.0e26,
        dt=dt,
    )


def check_charge_conservation(rho_start, rho_end):
    assert np.all(is_close(rho_start, rho_end, rtol=2.0e-11))


def main():
    filename_end = sys.argv[1]
    filename_start = filename_end[:-4] + "0000"
    ds_end = yt.load(filename_end)
    ds_start = yt.load(filename_start)
    ad_end = ds_end.all_data()
    ad_start = ds_start.all_data()
    dt = float(ds_end.current_time - ds_start.current_time)  # noqa
    field_data_end = ds_end.covering_grid(
        level=0, left_edge=ds_end.domain_left_edge, dims=ds_end.domain_dimensions
    )
    field_data_start = ds_start.covering_grid(
        level=0, left_edge=ds_start.domain_left_edge, dims=ds_start.domain_dimensions
    )

    for i in range(1, ntests + 1):
        data = {}

        for species_name in reactant_species:
            add_species_to_dict(
                ad_start, data, species_name + "_" + str(i), species_name, "start"
            )
            add_species_to_dict(
                ad_end, data, species_name + "_" + str(i), species_name, "end"
            )

        for species_name in product_species:
            add_species_to_dict(
                ad_end, data, species_name + "_" + str(i), species_name, "end"
            )

        # General checks that are performed for all tests
        generic_check(data)

        # Checks that are specific to test number i
        eval("specific_check" + str(i) + "(data, dt)")

    rho_start = field_data_start["rho"].to_ndarray()
    rho_end = field_data_end["rho"].to_ndarray()
    check_charge_conservation(rho_start, rho_end)


if __name__ == "__main__":
    main()
