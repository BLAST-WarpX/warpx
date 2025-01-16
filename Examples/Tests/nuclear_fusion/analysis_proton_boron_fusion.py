#! /usr/bin/env python
# Copyright 2021 Neil Zaim
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import re
import sys

import numpy as np
import scipy.constants as scc
import yt

## This script performs various checks for the proton boron nuclear fusion module. The simulation
## that we check is made of 5 different tests, each with different proton, boron and alpha species.
##
## The first test is performed in the proton-boron center of mass frame. It could correspond to the
## physical case of a proton beam colliding with a boron beam. The kinetic energy of the colliding
## particles depends on the cell number in the z direction and varies in the few keV to few MeV
## range. All the particles within a cell have the exact same momentum, which allows detailed
## checks of the energy of produced alpha particles. The proton and boron species have the same
## density and number of particles in this test. The number of produced alphas is much smaller than
## the initial number of protons and borons.
##
## The second test is performed in the boron rest frame. It corresponds to the physical case of a
## low density proton beam colliding with a high-density proton+boron target. The energy of the
## proton beam is varied in the few keV to few MeV range, depending on the cell number in the z
## direction. As in the previous case, all the particles within a cell have the exact same
## momentum, which allows detailed checks of the energy of produced alpha particles. In this test,
## there are 100 immobile boron and 100 immobile proton macroparticles per cell, as well as 900
## beam proton macroparticles per cell. The density of the immobile particles is 6 orders of
## magnitude higher than the number of beam particles, which means that they have a much higher
## weight. This test is similar to the example given in section 3 of Higginson et al.,
## Journal of Computation Physics, 388 439–453 (2019), which was found to be sensitive to the way
## unsampled pairs are accounted for. As before, the number of produced alphas is much smaller than
## the initial number of protons and borons.
##
## The third test corresponds to a Maxwellian plasma with a 44 keV temperature. The alpha yield is
## directly compared to the analytical fits of A. Tentori and F. Belloni, Nuclear Fusion, 63, 086001
## (2023) for a thermal plasma.
##
## The fourth test corresponds to a plasma with an extremely small boron density, so that all boron
## macroparticles should have disappeared by the end of the simulation, which we verify.
##
## The fifth test is exactly the same as the fourth test, except that the
## fusion_probability_threshold parameter is increased to an excessive value. Because of that, we
## severely underestimate the fusion yield and boron macroparticles remain at the end of the
## simulation, which we verify.
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
m_p = 1.00782503223 * scc.m_u  # Proton mass
m_b = 11.00930536 * scc.m_u  # Boron 11 mass
m_reduced = m_p * m_b / (m_p + m_b)
m_a = 4.00260325413 * scc.m_u  # Alpha (He4) mass
m_be = (8.0053095729 + 0.00325283863) * scc.m_u  # Be8* mass (3.03 MeV ex. state)
Z_boron = 5.0
Z_proton = 1.0
E_Gamow = (
    (Z_boron * Z_proton * np.pi * scc.fine_structure) ** 2 * 2.0 * m_reduced * scc.c**2
)
E_Gamow_MeV = E_Gamow / MeV_to_Joule
E_Gamow_keV = E_Gamow / keV_to_Joule
E_fusion = 5.55610759 * MeV_to_Joule  # Energy released during p + B -> alpha + Be*
E_decay = 3.12600414 * MeV_to_Joule  # Energy released during Be* -> 2*alpha
E_fusion_total = E_fusion + E_decay  # Energy released during p + B -> 3*alpha

## Checks whether this is the 2D or the 3D test
with open("./warpx_used_inputs") as warpx_used_inputs:
    is_2D = re.search("geometry.dims\s*=\s*2", warpx_used_inputs.read())
warpx_used_inputs.close()

## Some numerical parameters for this test
size_x = 8
if is_2D:
    size_y = 1
else:
    size_y = 8
size_z = 16
dV_total = size_x * size_y * size_z  # Total simulation volume
# Volume of a slice corresponding to a single cell in the z direction. In tests 1 and 2, all the
# particles of a given species in the same slice have the exact same momentum
dV_slice = size_x * size_y
if is_2D:
    dt = 1.0 / (scc.c * np.sqrt(2.0))
    yt_z_string = "particle_position_y"
    nppcell_1 = 10000 * 8
    nppcell_2 = 900 * 8
else:
    dt = 1.0 / (scc.c * np.sqrt(3.0))
    yt_z_string = "particle_position_z"
    nppcell_1 = 10000
    nppcell_2 = 900
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
        ## dictionnary. Currently, this happens for the boron species in test number 4, which
        ## entirely fuses into alphas.
        add_empty_species_to_dict(data_dict, species_name, prefix, suffix)


def check_particle_number_conservation(data):
    total_w_proton_start = np.sum(data["proton_w_start"])
    total_w_proton_end = np.sum(data["proton_w_end"])
    total_w_boron_start = np.sum(data["boron_w_start"])
    total_w_boron_end = np.sum(data["boron_w_end"])
    consumed_proton = total_w_proton_start - total_w_proton_end
    consumed_boron = total_w_boron_start - total_w_boron_end
    created_alpha = np.sum(data["alpha_w_end"])
    assert consumed_proton >= 0.0
    assert consumed_boron >= 0.0
    assert created_alpha >= 0.0
    ## Check that number of consumed proton and consumed boron are equal
    assert_scale = max(total_w_proton_start, total_w_boron_start)
    assert is_close(
        consumed_proton, consumed_boron, rtol=0.0, atol=default_tol * assert_scale
    )
    ## Check that number of consumed particles corresponds to number of produced alpha
    ## Factor 3 is here because each nuclear fusion reaction produces 3 alphas
    assert is_close(total_w_proton_start, total_w_proton_end + created_alpha / 3.0)
    assert is_close(total_w_boron_start, total_w_boron_end + created_alpha / 3.0)


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
    proton_energy_start = compute_energy_array(data, "proton", "start", m_p)
    proton_energy_end = compute_energy_array(data, "proton", "end", m_p)
    boron_energy_start = compute_energy_array(data, "boron", "start", m_b)
    boron_energy_end = compute_energy_array(data, "boron", "end", m_b)
    alpha_energy_end = compute_energy_array(data, "alpha", "end", m_a)
    total_energy_start = np.sum(proton_energy_start * data["proton_w_start"]) + np.sum(
        boron_energy_start * data["boron_w_start"]
    )
    total_energy_end = (
        np.sum(proton_energy_end * data["proton_w_end"])
        + np.sum(boron_energy_end * data["boron_w_end"])
        + np.sum(alpha_energy_end * data["alpha_w_end"])
    )
    ## Factor 3 is here because each nuclear fusion reaction produces 3 alphas
    n_fusion_reaction = np.sum(data["alpha_w_end"]) / 3.0
    assert is_close(
        total_energy_end,
        total_energy_start + n_fusion_reaction * E_fusion_total,
        rtol=1.0e-8,
    )


def check_momentum_conservation(data):
    proton_total_px_start = np.sum(data["proton_px_start"] * data["proton_w_start"])
    proton_total_py_start = np.sum(data["proton_py_start"] * data["proton_w_start"])
    proton_total_pz_start = np.sum(data["proton_pz_start"] * data["proton_w_start"])
    proton_total_px_end = np.sum(data["proton_px_end"] * data["proton_w_end"])
    proton_total_py_end = np.sum(data["proton_py_end"] * data["proton_w_end"])
    proton_total_pz_end = np.sum(data["proton_pz_end"] * data["proton_w_end"])
    boron_total_px_start = np.sum(data["boron_px_start"] * data["boron_w_start"])
    boron_total_py_start = np.sum(data["boron_py_start"] * data["boron_w_start"])
    boron_total_pz_start = np.sum(data["boron_pz_start"] * data["boron_w_start"])
    boron_total_px_end = np.sum(data["boron_px_end"] * data["boron_w_end"])
    boron_total_py_end = np.sum(data["boron_py_end"] * data["boron_w_end"])
    boron_total_pz_end = np.sum(data["boron_pz_end"] * data["boron_w_end"])
    alpha_total_px_end = np.sum(data["alpha_px_end"] * data["alpha_w_end"])
    alpha_total_py_end = np.sum(data["alpha_py_end"] * data["alpha_w_end"])
    alpha_total_pz_end = np.sum(data["alpha_pz_end"] * data["alpha_w_end"])
    total_px_start = proton_total_px_start + boron_total_px_start
    total_py_start = proton_total_py_start + boron_total_py_start
    total_pz_start = proton_total_pz_start + boron_total_pz_start
    total_px_end = proton_total_px_end + boron_total_px_end + alpha_total_px_end
    total_py_end = proton_total_py_end + boron_total_py_end + alpha_total_py_end
    total_pz_end = proton_total_pz_end + boron_total_pz_end + alpha_total_pz_end
    ## Absolute tolerance is needed because sometimes the initial momentum is exactly 0
    assert is_close(total_px_start, total_px_end, atol=1.0e-15)
    assert is_close(total_py_start, total_py_end, atol=1.0e-15)
    assert is_close(total_pz_start, total_pz_end, atol=1.0e-15)


def check_id(data):
    ## Check that all created particles have unique id + cpu identifier (two particles with
    ## different cpu can have the same id)
    complex_id = data["alpha_id_end"] + 1j * data["alpha_cpu_end"]
    assert complex_id.shape == np.unique(complex_id).shape


def basic_product_particles_check(data):
    ## For each nuclear fusion reaction in the code, we create 6 alpha macroparticles. So the
    ## total number of alpha macroparticles must be a multiple of 6.
    num_alpha = data["alpha_w_end"].shape[0]
    assert num_alpha % 6 == 0

    ## The weight of the 6 macroparticles coming from a single fusion event should be the same.
    ## We verify this here.
    assert np.array_equal(data["alpha_w_end"][::6], data["alpha_w_end"][1::6])
    assert np.array_equal(data["alpha_w_end"][::6], data["alpha_w_end"][2::6])
    assert np.array_equal(data["alpha_w_end"][::6], data["alpha_w_end"][3::6])
    assert np.array_equal(data["alpha_w_end"][::6], data["alpha_w_end"][4::6])
    assert np.array_equal(data["alpha_w_end"][::6], data["alpha_w_end"][5::6])

    ## When we create 6 macroparticles, the first has the exact same momentum as the second, the
    ## third has the same as the fourth and the fifth has the same as the sixth. We verify this
    ## here
    assert np.array_equal(data["alpha_px_end"][::6], data["alpha_px_end"][1::6])
    assert np.array_equal(data["alpha_py_end"][::6], data["alpha_py_end"][1::6])
    assert np.array_equal(data["alpha_pz_end"][::6], data["alpha_pz_end"][1::6])
    assert np.array_equal(data["alpha_px_end"][2::6], data["alpha_px_end"][3::6])
    assert np.array_equal(data["alpha_py_end"][2::6], data["alpha_py_end"][3::6])
    assert np.array_equal(data["alpha_pz_end"][2::6], data["alpha_pz_end"][3::6])
    assert np.array_equal(data["alpha_px_end"][4::6], data["alpha_px_end"][5::6])
    assert np.array_equal(data["alpha_py_end"][4::6], data["alpha_py_end"][5::6])
    assert np.array_equal(data["alpha_pz_end"][4::6], data["alpha_pz_end"][5::6])


def generic_check(data):
    check_particle_number_conservation(data)
    check_energy_conservation(data)
    check_momentum_conservation(data)
    check_id(data)
    basic_product_particles_check(data)


def check_isotropy(data, relative_tolerance):
    ## Checks that the alpha particles are emitted isotropically
    average_px_sq = np.average(data["alpha_px_end"] * data["alpha_px_end"])
    average_py_sq = np.average(data["alpha_py_end"] * data["alpha_py_end"])
    average_pz_sq = np.average(data["alpha_pz_end"] * data["alpha_pz_end"])
    assert is_close(average_px_sq, average_py_sq, rtol=relative_tolerance)
    assert is_close(average_px_sq, average_pz_sq, rtol=relative_tolerance)


def astrophysical_factor_lowE(E):
    ## E is in keV
    ## Returns astrophysical factor in MeV b using the low energy fit in the range E < 400 keV
    ## described in equation (3) of A. Tentori and F. Belloni, Nuclear Fusion, 63, 086001 (2023)
    C0 = 197.0
    C1 = 0.269
    C2 = 2.54e-4
    AL = 1.82e4
    EL = 148.0
    dEL = 2.35
    return C0 + C1 * E + C2 * E**2 + AL / ((E - EL) ** 2 + dEL**2)


def astrophysical_factor_midE(E):
    ## E is in keV
    ## Returns astrophysical factor in MeV b using the mid energy fit in the range
    ## 400 keV < E < 668 keV described in equation (4) of A. Tentori and F. Belloni,
    ## Nuclear Fusion, 63, 086001 (2023)
    D0 = 346.0
    D1 = 150.0
    D2 = -59.9
    D5 = -0.460
    E_400 = 400.0
    E_100 = 100.0
    E_norm = (E - E_400) / E_100
    return D0 + D1 * E_norm + D2 * E_norm**2 + D5 * E_norm**5


def astrophysical_factor_highE(E):
    ## E is in keV
    ## Returns astrophysical factor in MeV b using the high energy fit in the range
    ## 668 keV < E < 9760 keV described in equation (5) of A. Tentori and F. Belloni,
    ## Nuclear Fusion, 63, 086001 (2023)
    A0 = 1.98e6
    A1 = 3.89e6
    A2 = 1.36e6
    A3 = 3.71e6
    E0 = 640.9
    E1 = 1211.0
    E2 = 2340.0
    E3 = 3294.0
    dE0 = 85.5
    dE1 = 414.0
    dE2 = 221.0
    dE3 = 351.0
    B = 0.381
    return (
        A0 / ((E - E0) ** 2 + dE0**2)
        + A1 / ((E - E1) ** 2 + dE1**2)
        + A2 / ((E - E2) ** 2 + dE2**2)
        + A3 / ((E - E3) ** 2 + dE3**2)
        + B
    )


def astrophysical_factor(E):
    ## E is in keV
    ## Returns astrophysical factor in MeV b using the fits described in A. Tentori
    ## and F. Belloni, Nuclear Fusion, 63, 086001 (2023)
    conditions = [E <= 400, E <= 668, E > 668]
    choices = [
        astrophysical_factor_lowE(E),
        astrophysical_factor_midE(E),
        astrophysical_factor_highE(E),
    ]
    return np.select(conditions, choices)


def pb_cross_section_buck_fit(E):
    ## E is in MeV
    ## Returns cross section in b using a power law fit of the data presented in Buck et al.,
    ## Nuclear Physics A, 398(2), 189-202 (1983) in the range E > 9.76 MeV.
    E_start_fit = 9.76
    ## Cross section at E = E_start_fit = 3.5 MeV
    cross_section_start_fit = 0.01277998
    slope_fit = -2.661840717596765
    return cross_section_start_fit * (E / E_start_fit) ** slope_fit


def pb_cross_section(E):
    ## E is in keV
    ## Returns cross section in b using the fits described in A. Tentori and F. Belloni,
    ## Nucl. Fusion, 63, 086001 (2023) for E < 9.76 MeV otherwise returns a power law fit
    ## of the data in Buck et al., Nuclear Physics A, 398(2), 189-202 (1983)
    E_MeV = E / 1.0e3
    conditions = [E <= 9760, E > 9760]
    choices = [
        astrophysical_factor(E) / E_MeV * np.exp(-np.sqrt(E_Gamow_MeV / E_MeV)),
        pb_cross_section_buck_fit(E_MeV),
    ]
    return np.select(conditions, choices)


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
    ## E is the kinetic energy of proton+boron in the center of mass frame, in keV
    ## Returns the relative velocity between proton and boron in this frame, in m/s
    E_J = E * keV_to_Joule + (m_p + m_b) * scc.c**2
    p_sq = E_com_to_p_sq_com(m_p, m_b, E_J)
    p = np.sqrt(p_sq)
    gamma_p = np.sqrt(1.0 + p_sq / (m_p * scc.c) ** 2)
    gamma_b = np.sqrt(1.0 + p_sq / (m_b * scc.c) ** 2)
    v_p = p / (gamma_p * m_p)
    v_b = p / (gamma_b * m_b)
    return v_p + v_b


def expected_alpha_weight_com(E_com, proton_density, boron_density, dV, dt):
    ## Computes expected number of produced alpha particles as a function of energy E_com in the
    ## center of mass frame. E_com is in keV.
    assert np.all(E_com >= 0)
    ## Case E_com == 0 is handled manually to avoid division by zero
    conditions = [E_com == 0, E_com > 0]
    ## Necessary to avoid division by 0 warning when pb_cross_section is evaluated
    E_com_never_zero = np.clip(E_com, 1.0e-15, None)
    choices = [
        0.0,
        pb_cross_section(E_com_never_zero) * compute_relative_v_com(E_com_never_zero),
    ]
    sigma_times_vrel = np.select(conditions, choices)
    ## Factor 3 is here because each fusion reaction produces 3 alphas
    return (
        3.0
        * proton_density
        * boron_density
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
    numcells = dV_total - dV_slice
    ## In these tests, the fusion_multiplier is so high that the fusion probability per pair is
    ## equal to the parameter fusion_probability_target_value
    fusion_probability_per_pair = fusion_probability_target_value
    expected_fusion_number = numcells * num_pair_per_cell * fusion_probability_per_pair
    ## Each fusion event produces 6 alpha macroparticles
    expected_macroparticle_number = 6.0 * expected_fusion_number
    std_macroparticle_number = 6.0 * np.sqrt(expected_fusion_number)
    actual_macroparticle_number = data["alpha_w_end"].shape[0]
    # 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
    assert is_close(
        actual_macroparticle_number,
        expected_macroparticle_number,
        rtol=0.0,
        atol=5.0 * std_macroparticle_number,
    )

    ## used in subsequent function
    return expected_fusion_number


def p_sq_boron_frame_to_E_COM_frame(p_proton_sq):
    # Takes the proton square norm of the momentum in the boron rest frame and returns the total
    # kinetic energy in the center of mass frame. Everything is in SI units.

    # Total (kinetic + mass) energy in lab frame
    E_lab = np.sqrt(p_proton_sq * scc.c**2 + (m_p * scc.c**2) ** 2) + m_b * scc.c**2
    # Use invariant E**2 - p**2c**2 of 4-momentum norm to compute energy in center of mass frame
    E_com = np.sqrt(E_lab**2 - p_proton_sq * scc.c**2)
    # Corresponding kinetic energy
    E_com_kin = E_com - (m_b + m_p) * scc.c**2
    return E_com_kin * (p_proton_sq > 0.0)


def p_sq_to_kinetic_energy(p_sq, m):
    ## Returns the kinetic energy of a particle as a function of its squared momentum.
    ## Everything is in SI units.
    return np.sqrt(p_sq * scc.c**2 + (m * scc.c**2) ** 2) - (m * scc.c**2)


def compute_E_com1(data):
    ## Computes kinetic energy (in Joule) in the center of frame for the first test

    ## Square norm of the momentum of proton/boron as a function of cell number in z direction
    p_sq = 2.0 * m_reduced * (Energy_step * np.arange(size_z) ** 2)
    return p_sq_to_kinetic_energy(p_sq, m_b) + p_sq_to_kinetic_energy(p_sq, m_p)


def compute_E_com2(data):
    ## Computes kinetic energy (in Joule) in the center of frame for the second test

    ## Square norm of the momentum of the proton as a function of cell number in z direction
    p_proton_sq = 2.0 * m_p * (Energy_step * np.arange(size_z) ** 2)
    return p_sq_boron_frame_to_E_COM_frame(p_proton_sq)


def check_alpha_yield(
    data, expected_fusion_number, E_com, proton_density, boron_density
):
    ## Checks that the fusion yield is as expected for the first and second tests.
    ## Proton and boron densities are in m^-3.

    alpha_weight_theory = expected_alpha_weight_com(
        E_com / keV_to_Joule, proton_density, boron_density, dV_slice, dt
    )
    alpha_weight_simulation = np.histogram(
        data["alpha_z_end"], bins=size_z, range=(0, size_z), weights=data["alpha_w_end"]
    )[0]

    ## -1 is here because the first slice 0 < z < 1 does not contribute to alpha creation
    expected_fusion_number_per_slice = expected_fusion_number / (size_z - 1)
    relative_std_alpha_weight = 1.0 / np.sqrt(expected_fusion_number_per_slice)

    # 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
    assert np.all(
        is_close(
            alpha_weight_theory,
            alpha_weight_simulation,
            rtol=5.0 * relative_std_alpha_weight,
        )
    )


def check_initial_energy1(data, E_com):
    ## In WarpX, the initial momentum of the alphas is computed assuming that the fusion process
    ## takes place in two steps:
    ## (1): proton + boron 11 -> alpha + beryllium 8
    ## (2): beryllium 8 -> alpha + alpha
    ## The alpha generated in the first step (labeled alpha1) generally has a different initial
    ## energy distribution than the alphas generated in the second step (labeled alpha2 and
    ## alpha3).
    ## In the first test, we are in the center of mass frame. Therefore, the momentum of alpha1 is
    ## entirely determined by the energy in the center of mass frame, so we check in this function
    ## that the energy of the alpha1 macroparticles is as expected. On the other hand, the energy
    ## of alpha2 and alpha3 follows a continuous distribution within a given range. In this test,
    ## we check that this range is as expected by comparing the maximum and minimum energy of the
    ## obtained macroparticles to the theoretical maximum and minimum.
    ## Note that in the simulations, 6 macroparticles are generated during for each fusion event.
    ## The first and second macroparticles are alpha1 particles. The third and fourth are alpha2.
    ## The fifth and sixth are alpha3.

    energy_alpha_simulation = compute_energy_array(data, "alpha", "end", m_a)
    z_alpha = data["alpha_z_end"]

    # Loop over all slices (i.e. cells in the z direction)
    for slice_number in range(1, size_z):
        ## Kinetic energy in the lab frame before fusion
        E_kinetic_com_before = E_com[slice_number]
        ## Total (kinetic + mass) energy in the lab frame after
        ## proton + boron 11 -> alpha + beryllium 8
        E_total_com_after = E_kinetic_com_before + E_fusion + (m_a + m_be) * scc.c**2
        ## Corresponding momentum norm squared of alpha1/beryllium
        p_sq_after = E_com_to_p_sq_com(m_a, m_be, E_total_com_after)
        ## Corresponding kinetic energy for alpha1
        energy_alpha1_theory = p_sq_to_kinetic_energy(p_sq_after, m_a)
        ## Corresponding kinetic energy for beryllium
        energy_beryllium_theory = p_sq_to_kinetic_energy(p_sq_after, m_be)
        ## Corresponding kinetic energy for alpha2 + alpha3 after beryllium decay
        energy_alpha2_plus_3_theory = energy_beryllium_theory + E_decay
        ## Compute the theoretical maximum and minimum energy of alpha2 and alpha3. This
        ## calculation is done nonrelativistically, by noting that the maximum (minimum) energy
        ## corresponds to an alpha emitted exactly in the (opposite) direction of the beryllium
        ## in the center of mass frame. This calculation involves solving a polynomial equation of
        ## order 2 in p_alpha23.
        max_p_alpha23 = 0.5 * (
            np.sqrt(p_sq_after)
            + np.sqrt(4 * m_a * energy_alpha2_plus_3_theory - p_sq_after)
        )
        min_p_alpha23 = 0.5 * (
            np.sqrt(p_sq_after)
            - np.sqrt(4 * m_a * energy_alpha2_plus_3_theory - p_sq_after)
        )
        max_energy_alpha23 = max_p_alpha23**2 / (2.0 * m_a)
        min_energy_alpha23 = min_p_alpha23**2 / (2.0 * m_a)

        ## Get the energy of all alphas in the slice
        energy_alpha_slice = energy_alpha_simulation[
            (z_alpha >= slice_number) * (z_alpha < (slice_number + 1))
        ]
        ## Energy of alphas1 (here, first macroparticle of each fusion event) in the slice
        energy_alpha1_simulation = energy_alpha_slice[::6]
        ## Energy of alphas2 (here, third macroparticle of each fusion event) in the slice
        energy_alpha2_simulation = energy_alpha_slice[2::6]
        ## Energy of alphas3 (here, fifth macroparticle of each fusion event) in the slice
        energy_alpha3_simulation = energy_alpha_slice[4::6]

        assert np.all(
            is_close(energy_alpha1_simulation, energy_alpha1_theory, rtol=5.0e-8)
        )
        ## Check that the max / min value are comparable to the analytical value
        ## The minimum value is checked to be within 20 keV of the analytical value
        ## The maximum value is checked to be within 1% of the analytical value
        assert is_close(
            np.amax(energy_alpha2_simulation), max_energy_alpha23, rtol=1.0e-2
        )
        assert is_close(
            np.amin(energy_alpha2_simulation), min_energy_alpha23, atol=3.218e-15
        )
        assert is_close(
            np.amax(energy_alpha3_simulation), max_energy_alpha23, rtol=1.0e-2
        )
        assert is_close(
            np.amin(energy_alpha3_simulation), min_energy_alpha23, atol=3.218e-15
        )


def check_initial_energy2(data):
    ## In WarpX, the initial momentum of the alphas is computed assuming that the fusion process
    ## takes place in two steps:
    ## (1): proton + boron 11 -> alpha + beryllium 8
    ## (2): beryllium 8 -> alpha + alpha
    ## The alpha generated in the first step (labeled alpha1) generally has a different initial
    ## energy distribution than the alphas generated in the second step (labeled alpha2 and
    ## alpha3).
    ## In the second test, we are in the boron rest frame. In this case, the momentum of each alpha
    ## follows a continuous distribution within a given range. In this function, we verify that
    ## this range is as expected by comparing the maximum and minimum energy of the obtained
    ## macroparticles to the theoretical maximum and minimum. Be aware that the range for alpha1
    ## is not the same as the range for alpha2 and alpha3 (typically alpha1 particles will carry
    ## more energy).
    ## Note that in the simulations, 6 macroparticles are generated during for each fusion event.
    ## The first and second macroparticles are alpha1 particles. The third and fourth are alpha2.
    ## The fifth and sixth are alpha3.

    energy_alpha_simulation = compute_energy_array(data, "alpha", "end", m_a)
    z_alpha = data["alpha_z_end"]

    # Loop over all slices (i.e. cells in the z direction)
    for slice_number in range(1, size_z):
        ## For simplicity, all the calculations in this function are done nonrelativistically
        ## Proton kinetic energy in the lab frame before fusion
        E_proton_nonrelativistic = Energy_step * slice_number**2
        ## Corresponding square norm of proton momentum
        p_proton_sq = 2.0 * m_p * E_proton_nonrelativistic
        ## Kinetic energy in the lab frame after
        ## proton + boron 11 -> alpha + beryllium 8
        E_after_fusion = E_proton_nonrelativistic + E_fusion

        ## Compute the theoretical maximum and minimum energy of alpha1 in the lab frame. This
        ## calculation is done by noting that the maximum (minimum) energy corresponds to an alpha
        ## emitted exactly in the (opposite) direction of the proton in the lab frame. This
        ## calculation involves solving a polynomial equation of order 2 in p_alpha1.
        max_p_alpha1 = (
            m_a / m_be * np.sqrt(p_proton_sq)
            + np.sqrt(
                -m_a / m_be * p_proton_sq
                + 2.0 * E_after_fusion * m_a * (m_a / m_be + 1.0)
            )
        ) / (m_a / m_be + 1.0)
        min_p_alpha1 = (
            m_a / m_be * np.sqrt(p_proton_sq)
            - np.sqrt(
                -m_a / m_be * p_proton_sq
                + 2.0 * E_after_fusion * m_a * (m_a / m_be + 1.0)
            )
        ) / (m_a / m_be + 1.0)
        max_energy_alpha1 = max_p_alpha1**2 / (2 * m_a)
        min_energy_alpha1 = min_p_alpha1**2 / (2 * m_a)

        ## Corresponding max/min kinetic energy of Beryllium in the lab frame
        max_E_beryllium = E_after_fusion - min_energy_alpha1
        min_E_beryllium = E_after_fusion - max_energy_alpha1
        ## Corresponding max/min momentum square of Beryllium in the lab frame
        max_p_sq_beryllium = 2.0 * m_be * max_E_beryllium
        min_p_sq_beryllium = 2.0 * m_be * min_E_beryllium
        ## Corresponding max/min kinetic energy in the lab frame for alpha2 + alpha3 after
        ## Beryllium decay
        max_energy_alpha2_plus_3 = max_E_beryllium + E_decay
        min_energy_alpha2_plus_3 = min_E_beryllium + E_decay

        ## Compute the theoretical maximum and minimum energy of alpha2 and alpha3 in the lab
        ## frame. This calculation is done by noting that the maximum (minimum) energy corresponds
        ## to an alpha emitted exactly in the (opposite) direction of a beryllium with energy
        ## max_E_beryllium (min_E_beryllium). This calculation involves solving a polynomial
        ## equation of order 2 in p_alpha23.
        max_p_alpha23 = 0.5 * (
            np.sqrt(max_p_sq_beryllium)
            + np.sqrt(4 * m_a * max_energy_alpha2_plus_3 - max_p_sq_beryllium)
        )
        min_p_alpha23 = 0.5 * (
            np.sqrt(min_p_sq_beryllium)
            - np.sqrt(4 * m_a * min_energy_alpha2_plus_3 - min_p_sq_beryllium)
        )
        max_energy_alpha23 = max_p_alpha23**2 / (2 * m_a)
        min_energy_alpha23 = min_p_alpha23**2 / (2 * m_a)

        ## Get the energy of all alphas in the slice
        energy_alpha_slice = energy_alpha_simulation[
            (z_alpha >= slice_number) * (z_alpha < (slice_number + 1))
        ]
        ## Energy of alphas1 (here, first macroparticle of each fusion event) in the slice
        energy_alpha1_simulation = energy_alpha_slice[::6]
        ## Energy of alphas2 (here, third macroparticle of each fusion event) in the slice
        energy_alpha2_simulation = energy_alpha_slice[2::6]
        ## Energy of alphas3 (here, fifth macroparticle of each fusion event) in the slice
        energy_alpha3_simulation = energy_alpha_slice[4::6]

        assert is_close(
            np.amax(energy_alpha1_simulation), max_energy_alpha1, rtol=1.0e-2
        )
        assert is_close(
            np.amin(energy_alpha1_simulation), min_energy_alpha1, rtol=1.0e-2
        )
        ## Check that the max / min value are comparable to the analytical value
        ## The minimum value is checked to be within 200 keV of the analytical value
        ## The maximum value is checked to be within 5% of the analytical value
        ## Tolerance is quite high because we don't have a lot of alphas to produce good
        ## statistics and an event like alpha1 emitted exactly in direction of proton & alpha2
        ## emitted exactly in direction opposite to Beryllium is somewhat rare.
        assert is_close(
            np.amax(energy_alpha2_simulation), max_energy_alpha23, rtol=5e-2
        )
        assert is_close(
            np.amin(energy_alpha2_simulation), min_energy_alpha23, atol=3.218e-14
        )
        assert is_close(
            np.amax(energy_alpha3_simulation), max_energy_alpha23, rtol=5e-2
        )
        assert is_close(
            np.amin(energy_alpha3_simulation), min_energy_alpha23, atol=3.218e-14
        )


def check_xy_isotropy(data):
    ## Checks that the alpha particles are emitted isotropically in x and y
    average_px_sq = np.average(data["alpha_px_end"] * data["alpha_px_end"])
    average_py_sq = np.average(data["alpha_py_end"] * data["alpha_py_end"])
    average_pz_sq = np.average(data["alpha_pz_end"] * data["alpha_pz_end"])
    assert is_close(average_px_sq, average_py_sq, rtol=5.0e-2)
    assert average_pz_sq > average_px_sq
    assert average_pz_sq > average_py_sq


def sigmav_thermal_fit_lowE_nonresonant(T):
    ## Temperature T is in keV
    ## Returns the nonresonant average of cross section multiplied by relative velocity in m^3/s,
    ## in the range T <= 70 keV, as described by equations 10-14 of A. Tentori and F. Belloni,
    ## Nuclear Fusion, 63, 086001 (2023).
    E0 = (E_Gamow_keV / 4.0) ** (1.0 / 3.0) * T ** (2.0 / 3.0)
    DE0 = 4.0 * np.sqrt(T * E0 / 3.0)
    C0 = 197.0 * 1.0e3
    C1 = 0.269 * 1.0e3
    C2 = 2.54e-4 * 1.0e3
    tau = 3.0 * E0 / T
    Seff = (
        C0 * (1.0 + 5.0 / (12.0 * tau))
        + C1 * (E0 + 35.0 / 36.0 * T)
        + C2 * (E0**2 + 89.0 / 36.0 * E0 * T)
    )
    ## nonresonant sigma times vrel, in barn meter per second
    sigmav_nr_bmps = (
        np.sqrt(2 * T * keV_to_Joule / m_reduced) * DE0 * Seff / T**2 * np.exp(-tau)
    )
    ## Return result in cubic meter per second
    return sigmav_nr_bmps * barn_to_square_meter


def sigmav_thermal_fit_lowE_resonant(T):
    ## Temperature T is in keV
    ## Returns the resonant average of cross section multiplied by relative velocity in m^3/s,
    ## in the range T <= 70 keV, as described by equation 15 of A. Tentori and F. Belloni,
    ## Nuclear Fusion, 63, 086001 (2023).
    return 5.41e-21 * np.exp(-148.0 / T) / T ** (3.0 / 2.0)


def sigmav_thermal_fit_lowE(T):
    ## Temperature T is in keV
    ## Returns the average of cross section multiplied by relative velocity in m^3/s, using the
    ## fits described in section 2.2 of A. Tentori and F. Belloni, Nuclear Fusion, 63, 086001 (2023).
    ## The fits are valid for T <= 70 keV.
    return sigmav_thermal_fit_lowE_nonresonant(T) + sigmav_thermal_fit_lowE_resonant(T)


def expected_alpha_thermal(T, proton_density, boron_density, dV, dt):
    ## Computes the expected number of produced alpha particles when the protons and borons follow
    ## a Maxwellian distribution with a temperature T, in keV. This uses the thermal fits described
    ## in A. Tentori and F. Belloni, Nuclear Fusion, 63, 086001 (2023).

    ## The fit used here is only valid in the range T <= 70 keV.
    assert (T >= 0) and (T <= 70)
    sigma_times_vrel = sigmav_thermal_fit_lowE(T)
    ## Factor 3 is here because each fusion event produces 3 alphas.
    return 3.0 * proton_density * boron_density * sigma_times_vrel * dV * dt


def check_thermal_alpha_yield(data):
    ## Checks that the number of alpha particles in test3 is as expected
    Temperature = 44.0  # keV
    proton_density = 1.0e28  # m^-3
    boron_density = 5.0e28  # m^-3

    alpha_weight_theory = expected_alpha_thermal(
        Temperature, proton_density, boron_density, dV_total, dt
    )
    alpha_weight_simulation = np.sum(data["alpha_w_end"])

    assert is_close(alpha_weight_theory, alpha_weight_simulation, rtol=2.0e-1)


def boron_remains(data):
    ## Checks whether there remains boron macroparticles at the end of the test
    n_boron_left = data["boron_w_end"].shape[0]
    return n_boron_left > 0


def specific_check1(data):
    check_isotropy(data, relative_tolerance=3.0e-2)
    expected_fusion_number = check_macroparticle_number(
        data, fusion_probability_target_value=0.002, num_pair_per_cell=nppcell_1
    )
    E_com = compute_E_com1(data)
    check_alpha_yield(
        data, expected_fusion_number, E_com, proton_density=1.0, boron_density=1.0
    )
    check_initial_energy1(data, E_com)


def specific_check2(data):
    check_xy_isotropy(data)
    ## Only 900 particles pairs per cell here because we ignore the 10% of protons that are at rest
    expected_fusion_number = check_macroparticle_number(
        data, fusion_probability_target_value=0.02, num_pair_per_cell=nppcell_2
    )
    E_com = compute_E_com2(data)
    check_alpha_yield(
        data, expected_fusion_number, E_com, proton_density=1.0e20, boron_density=1.0e26
    )
    check_initial_energy2(data)


def specific_check3(data):
    check_isotropy(data, relative_tolerance=1.0e-1)
    check_thermal_alpha_yield(data)


def specific_check4(data):
    ## In test 4, the boron initial density is so small that all borons should have fused within a
    ## timestep dt. We thus assert that no boron remains at the end of the simulation.
    assert not boron_remains(data)


def specific_check5(data):
    ## Test 5 is similar to test 4, expect that the parameter fusion_probability_threshold is
    ## increased to the point that we should severely underestimate the fusion yield. Consequently,
    ## there should still be borons at the end of the test, which we verify here.
    assert boron_remains(data)


def check_charge_conservation(rho_start, rho_end):
    assert np.all(is_close(rho_start, rho_end, rtol=2.0e-11))


def main():
    filename_end = sys.argv[1]
    filename_start = filename_end[:-4] + "0000"
    ds_end = yt.load(filename_end)
    ds_start = yt.load(filename_start)
    ad_end = ds_end.all_data()
    ad_start = ds_start.all_data()
    field_data_end = ds_end.covering_grid(
        level=0, left_edge=ds_end.domain_left_edge, dims=ds_end.domain_dimensions
    )
    field_data_start = ds_start.covering_grid(
        level=0, left_edge=ds_start.domain_left_edge, dims=ds_start.domain_dimensions
    )

    ntests = 5
    for i in range(1, ntests + 1):
        proton_species = "proton" + str(i)
        boron_species = "boron" + str(i)
        alpha_species = "alpha" + str(i)
        data = {}
        add_species_to_dict(ad_start, data, proton_species, "proton", "start")
        add_species_to_dict(ad_start, data, boron_species, "boron", "start")
        add_species_to_dict(ad_end, data, proton_species, "proton", "end")
        add_species_to_dict(ad_end, data, boron_species, "boron", "end")
        add_species_to_dict(ad_end, data, alpha_species, "alpha", "end")

        # General checks that are performed for all tests
        generic_check(data)

        # Checks that are specific to test number i
        eval("specific_check" + str(i) + "(data)")

    rho_start = field_data_start["rho"].to_ndarray()
    rho_end = field_data_end["rho"].to_ndarray()
    check_charge_conservation(rho_start, rho_end)


if __name__ == "__main__":
    main()
