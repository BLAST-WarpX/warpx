#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019 Luca Fedeli, Maxence Thevenet
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
import scipy.special as spe
import scipy.stats as st

# This script performs detailed checks of the Breit-Wheeler pair production process.
# Four populations of photons are initialized with different momenta in different
# directions in a background EM field (with non-zero components along each direction).
# Specifically the script checks that:
#
# - The expected number of generated pairs n_pairs is in agreement with theory
#   (the maximum tolerated error is 5*sqrt(n_pairs)
# - The weight of the generated particles is equal to the weight of the photon
# - Momenta of the residual photons are still equal to the original momentum
# - The generated particles are emitted in the right direction
# - Total energy is conserved in each event
# - The energy distribution of the generated particles is in agreement with theory
# - The optical depths of the product species are correctly initialized (QED effects are
#   enabled for product species too).
#
# More details on the theoretical formulas used in this script can be found in
# the Jupyter notebook picsar/src/multi_physics/QED_tests/validation/validation.ipynb
#
# References:
# 1) R. Duclous et al 2011 Plasma Phys. Control. Fusion 53 015009
# 2) A. Gonoskov et al. 2015 Phys. Rev. E 92, 023305
# 3) M. Lobet. PhD thesis "Effets radiatifs et d'electrodynamique
#    quantique dans l'interaction laser-matiere ultra-relativiste"
#    URL: https://tel.archives-ouvertes.fr/tel-01314224


# Tolerances
tol = 1.0e-8
tol_red = 2.0e-2

# Physical constants (from CODATA 2018, see: https://physics.nist.gov/cuu/Constants/index.html )
me = 9.1093837015e-31  # electron mass
c = 299792458  # speed of light
hbar = 6.62607015e-34 / (2 * np.pi)  # reduced Plank constant
fine_structure = 7.2973525693e-3  # fine structure constant
qe = 1.602176634e-19  # elementary charge
E_s = (me**2 * c**3) / (qe * hbar)  # Schwinger E field
B_s = E_s / c  # Schwinger B field

mec = me * c
mec2 = mec * c
# ______________

# Initial parameters
spec_names_phot = ["p1", "p2", "p3", "p4"]
spec_names_ele = ["ele1", "ele2", "ele3", "ele4"]
spec_names_pos = ["pos1", "pos2", "pos3", "pos4"]
initial_momenta = [
    np.array([2000.0, 0, 0]) * mec,
    np.array([0.0, 5000.0, 0.0]) * mec,
    np.array([0.0, 0.0, 10000.0]) * mec,
    np.array([57735.02691896, 57735.02691896, 57735.02691896]) * mec,
]
initial_particle_number = 1048576

E_f = np.array([-2433321316961438.0, 973328526784575.0, 1459992790176863.0])
B_f = np.array([2857142.85714286, 4285714.28571428, 8571428.57142857])

NNS = [128, 128, 128, 128]  # bins for energy distribution comparison.
# ______________


# Returns all the species names and if they are photon species or not
def get_all_species_names_and_types():
    names = spec_names_phot + spec_names_ele + spec_names_pos
    types = [True] * len(spec_names_phot) + [False] * (
        len(spec_names_ele) + len(spec_names_pos)
    )
    return names, types


def calc_chi_gamma(p, E, B):
    pnorm = np.linalg.norm(p)
    v = c * (p / pnorm)
    EpvvecB = E + np.cross(v, B)
    vdotEoverc = np.dot(v, E) / c
    ff = np.sqrt(np.dot(EpvvecB, EpvvecB) - np.dot(vdotEoverc, vdotEoverc))
    gamma_phot = pnorm / mec
    return gamma_phot * ff / E_s


# Auxiliary functions
@np.vectorize
def BW_inner(x):
    return integ.quad(
        lambda s: np.sqrt(s) * spe.kv(1.0 / 3.0, 2.0 / 3.0 * s ** (3.0 / 2.0)),
        x,
        np.inf,
    )[0]


def BW_X(chi_phot, chi_ele):
    div = chi_ele * (chi_phot - chi_ele)
    div = np.where(np.logical_and(chi_phot > chi_ele, chi_ele != 0), div, 1.0)
    res = np.where(
        np.logical_and(chi_phot > chi_ele, chi_ele != 0),
        np.power(chi_phot / div, 2.0 / 3.0),
        np.inf,
    )
    return res


def BW_F(chi_phot, chi_ele):
    X = BW_X(chi_phot, chi_ele)
    res = np.where(
        np.logical_or(chi_phot == chi_ele, chi_ele == 0),
        0.0,
        BW_inner(X)
        - (2.0 - chi_phot * X ** (3.0 / 2.0))
        * spe.kv(2.0 / 3.0, 2.0 / 3.0 * X ** (3.0 / 2.0)),
    )
    return res


@np.vectorize
def BW_T(chi_phot):
    coeff = 1.0 / (np.pi * np.sqrt(3.0) * (chi_phot**2))
    return coeff * integ.quad(lambda chi_ele: BW_F(chi_phot, chi_ele), 0, chi_phot)[0]


def small_diff(vv, val):
    if val != 0.0:
        return np.max(np.abs((vv - val) / val)) < tol
    else:
        return np.max(np.abs(vv)) < tol


# __________________


# Breit-Wheeler total and differential cross sections
def BW_dN_dt(chi_phot, gamma_phot):
    coeff_BW = fine_structure * me * c**2 / hbar
    return coeff_BW * BW_T(chi_phot) * (chi_phot / gamma_phot)


def BW_d2N_dt_dchi(chi_phot, gamma_phot, chi_ele):
    coeff_BW = fine_structure * me * c**2 / hbar
    return coeff_BW * BW_F(chi_phot, chi_ele) * (gamma_phot / gamma_phot)


# __________________

# Individual tests


def check_number_of_pairs(
    particle_data,
    phot_name,
    ele_name,
    pos_name,
    chi_phot,
    gamma_phot,
    dt,
    particle_number,
):
    dNBW_dt_theo = BW_dN_dt(chi_phot, gamma_phot)
    expected_pairs = (1.0 - np.exp(-dNBW_dt_theo * dt)) * particle_number
    expected_pairs_tolerance = 5.0 * np.sqrt(expected_pairs)
    n_ele = len(particle_data[ele_name]["w"])
    n_pos = len(particle_data[pos_name]["w"])
    n_phot = len(particle_data[phot_name]["w"])
    n_lost = initial_particle_number - n_phot
    assert (n_ele == n_pos) and (n_ele == n_lost)
    assert np.abs(n_ele - expected_pairs) < expected_pairs_tolerance
    print("  [OK] generated pair number is within expectations")
    return n_lost


def check_weights(phot_data, ele_data, pos_data):
    assert np.all(phot_data["w"] == phot_data["w"][0])
    assert np.all(ele_data["w"] == phot_data["w"][0])
    assert np.all(pos_data["w"] == phot_data["w"][0])
    print("  [OK] particles weights are the expected ones")


def check_momenta(phot_data, ele_data, pos_data, p0, p_ele, p_pos):
    assert small_diff(phot_data["px"], p0[0])
    assert small_diff(phot_data["py"], p0[1])
    assert small_diff(phot_data["pz"], p0[2])
    print("  [OK] residual photons still have initial momentum")

    pdir = p0 / np.linalg.norm(p0)
    assert small_diff(ele_data["px"] / p_ele, pdir[0])
    assert small_diff(ele_data["py"] / p_ele, pdir[1])
    assert small_diff(ele_data["pz"] / p_ele, pdir[2])
    assert small_diff(pos_data["px"] / p_pos, pdir[0])
    assert small_diff(pos_data["py"] / p_pos, pdir[1])
    assert small_diff(pos_data["pz"] / p_pos, pdir[2])
    print("  [OK] pairs move along the initial photon direction")


def check_energy(energy_phot, energy_ele, energy_pos):
    # Sorting the arrays is required because electrons and positrons are not
    # necessarily dumped in the same order.
    s_energy_ele = np.sort(energy_ele)
    is_energy_pos = np.sort(energy_pos)[::-1]
    product_energy = s_energy_ele + is_energy_pos
    assert small_diff(product_energy, energy_phot)
    print("  [OK] energy is conserved in each event")


def check_opt_depths(phot_data, ele_data, pos_data):
    data = (phot_data, ele_data, pos_data)
    for dd in data:
        # Remove the negative optical depths that correspond to photons that will decay into pairs
        # at the beginning of the next timestep
        loc, scale = st.expon.fit(dd["opt"][dd["opt"] > 0])
        assert np.abs(loc - 0) < tol_red
        assert np.abs(scale - 1) < tol_red
    print("  [OK] optical depth distributions are still exponential")


def check_energy_distrib(
    energy_ele, energy_pos, gamma_phot, chi_phot, n_lost, NN, idx, do_plot=False
):
    gamma_min = 1.0001
    gamma_max = gamma_phot - 1.0001
    h_gamma_ele, c_gamma = np.histogram(
        energy_ele / mec2, bins=NN, range=[gamma_min, gamma_max]
    )
    h_gamma_pos, _ = np.histogram(
        energy_pos / mec2, bins=NN, range=[gamma_min, gamma_max]
    )

    cchi_part_min = chi_phot * (gamma_min - 1) / (gamma_phot - 2)
    cchi_part_max = chi_phot * (gamma_max - 1) / (gamma_phot - 2)

    # Rudimentary integration over npoints for each bin
    npoints = 20
    aux_chi = np.linspace(cchi_part_min, cchi_part_max, NN * npoints)
    distrib = BW_d2N_dt_dchi(chi_phot, gamma_phot, aux_chi)
    distrib = np.sum(distrib.reshape(-1, npoints), 1)
    distrib = n_lost * distrib / np.sum(distrib)

    if do_plot:
        # Visual comparison of distributions
        c_gamma_centered = 0.5 * (c_gamma[1:] + c_gamma[:-1])
        plt.clf()
        plt.xlabel("γ_particle")
        plt.ylabel("N")
        plt.title("χ_photon = {:f}".format(chi_phot))
        plt.plot(c_gamma_centered, distrib, label="theory")
        plt.plot(c_gamma_centered, h_gamma_ele, label="BW electrons")
        plt.plot(c_gamma_centered, h_gamma_pos, label="BW positrons")
        plt.legend()
        plt.savefig("case_{:d}".format(idx + 1))

    discr_ele = np.abs(h_gamma_ele - distrib)
    discr_pos = np.abs(h_gamma_pos - distrib)
    max_discr = 5.0 * np.sqrt(distrib)
    assert np.all(discr_ele < max_discr)
    assert np.all(discr_pos < max_discr)
    print("  [OK] energy distribution is within expectations")


# __________________


def check(dt, particle_data):
    for idx in range(4):
        phot_name = spec_names_phot[idx]
        ele_name = spec_names_ele[idx]
        pos_name = spec_names_pos[idx]
        p0 = initial_momenta[idx]

        p2_phot = p0[0] ** 2 + p0[1] ** 2 + p0[2] ** 2
        p_phot = np.sqrt(p2_phot)
        energy_phot = p_phot * c
        chi_phot = calc_chi_gamma(p0, E_f, B_f)
        gamma_phot = np.linalg.norm(p0) / mec

        print("** Case {:d} **".format(idx + 1))
        print("  initial momentum: ", p0)
        print("  quantum parameter: {:f}".format(chi_phot))
        print("  normalized photon energy: {:f}".format(gamma_phot))
        print("  timestep: {:f} fs".format(dt * 1e15))

        phot_data = particle_data[phot_name]
        ele_data = particle_data[ele_name]
        pos_data = particle_data[pos_name]

        p2_ele = ele_data["px"] ** 2 + ele_data["py"] ** 2 + ele_data["pz"] ** 2
        p_ele = np.sqrt(p2_ele)
        energy_ele = np.sqrt(1.0 + p2_ele / mec**2) * mec2
        p2_pos = pos_data["px"] ** 2 + pos_data["py"] ** 2 + pos_data["pz"] ** 2
        p_pos = np.sqrt(p2_pos)
        energy_pos = np.sqrt(1.0 + p2_pos / mec**2) * mec2

        n_lost = check_number_of_pairs(
            particle_data,
            phot_name,
            ele_name,
            pos_name,
            chi_phot,
            gamma_phot,
            dt,
            initial_particle_number,
        )

        check_weights(phot_data, ele_data, pos_data)

        check_momenta(phot_data, ele_data, pos_data, p0, p_ele, p_pos)

        check_energy(energy_phot, energy_ele, energy_pos)

        check_energy_distrib(
            energy_ele, energy_pos, gamma_phot, chi_phot, n_lost, NNS[idx], idx
        )

        check_opt_depths(phot_data, ele_data, pos_data)

        print("*************\n")
