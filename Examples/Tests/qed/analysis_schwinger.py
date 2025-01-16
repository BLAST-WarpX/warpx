#!/usr/bin/env python3

# Copyright 2020 Luca Fedeli, Neil Zaim
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

## This module is imported by each of the scripts used to analyze the Schwinger tests.

## The pair production rate is calculated using the formula described in
## Bulanov, S. S., et al. Physical review letters 104.22 (2010): 220404.

import os
import re
import sys

import numpy as np
import yt

# define some parameters

c = 299792458.0
m_e = 9.1093837015e-31
e = 1.602176634e-19
hbar = 1.054571817e-34
E_S = m_e**2 * c**3 / e / hbar  # Schwinger field

dV = (1.0e-6) ** 3  # total simulation volume
dt = 9.827726403e-17
filename = sys.argv[1]

Ex_test = 0.0
Ey_test = 0.0
Ez_test = 0.0
Bx_test = 0.0
By_test = 0.0
Bz_test = 0.0

# Find which test we are doing
test_name = os.path.split(os.getcwd())[1]
test_number = re.search("qed_schwinger_([1234])", test_name).group(1)
if test_number == "1":
    # First Schwinger test with "weak" EM field. No pair should be created.
    Ex_test = 1.0e16
    Bx_test = 16792888.570516706
    By_test = 5256650.141557486
    Bz_test = 18363530.799561853
elif test_number == "2":
    # Second Schwinger test with stronger EM field. Many pairs are created and a Gaussian
    # distribution is used to get the weights of the particles. This is the most sensitive test
    # because the relative std is extremely low.
    Ex_test = 1.0e18
    Bx_test = 1679288857.0516706
    By_test = 525665014.1557486
    Bz_test = 1836353079.9561853
    dV = dV / 2.0  # Schwinger is only activated in part of the simulation domain
elif test_number == "3":
    # Third Schwinger test with intermediate electric field such that average created pair per cell
    # is 1. A Poisson distribution is used to obtain the weights of the particles.
    Ey_test = 1.090934525450495e17
elif test_number == "4":
    # Fourth Schwinger test with extremely strong EM field but with E and B perpendicular and nearly
    # equal so that the pair production rate is fairly low. A Gaussian distribution is used in this
    # case.
    Ez_test = 2.5e20
    By_test = 833910140000.0
    dV = (
        dV * (3.0 / 4.0) ** 2.0
    )  # Schwinger is only activated in part of the simulation domain
else:
    assert False


def calculate_rate(Ex, Ey, Ez, Bx, By, Bz):
    ## Calculate theoretical pair production rate from EM field value

    E_squared = Ex**2 + Ey**2 + Ez**2
    H_squared = c**2 * (Bx**2 + By**2 + Bz**2)

    F = (E_squared - H_squared) / 2.0
    G = c * (Ex * Bx + Ey * By + Ez * Bz)

    epsilon = np.sqrt(np.sqrt(F**2 + G**2) + F) / E_S
    eta = np.sqrt(np.sqrt(F**2 + G**2) - F) / E_S

    if epsilon != 0.0 and eta != 0.0:
        return (
            e**2
            * E_S**2
            / 4.0
            / np.pi**2
            / c
            / hbar**2
            * epsilon
            * eta
            / np.tanh(np.pi * eta / epsilon)
            * np.exp(-np.pi / epsilon)
        )
    elif epsilon == 0.0:
        return 0.0
    else:
        return (
            e**2
            * E_S**2
            / 4.0
            / np.pi**2
            / c
            / hbar**2
            * epsilon**2
            / np.pi
            * np.exp(-np.pi / epsilon)
        )


def do_analysis(Ex, Ey, Ez, Bx, By, Bz):
    data_set = yt.load(filename)

    expected_total_physical_pairs_created = (
        dV * dt * calculate_rate(Ex, Ey, Ez, Bx, By, Bz)
    )
    if expected_total_physical_pairs_created < 0.01:
        np_ele = (
            data_set.particle_type_counts["ele_schwinger"]
            if "ele_schwinger" in data_set.particle_type_counts.keys()
            else 0
        )
        np_pos = (
            data_set.particle_type_counts["pos_schwinger"]
            if "pos_schwinger" in data_set.particle_type_counts.keys()
            else 0
        )
        assert np_ele == 0 and np_pos == 0
        ## Assert whether pairs are created or not.

    else:
        all_data = data_set.all_data()

        ele_data = all_data["ele_schwinger", "particle_weight"]
        pos_data = all_data["pos_schwinger", "particle_weight"]

        std_total_physical_pairs_created = np.sqrt(
            expected_total_physical_pairs_created
        )

        # Sorting the arrays is required because electrons and positrons are not necessarily
        # dumped in the same order.
        assert np.array_equal(np.sort(ele_data), np.sort(pos_data))
        # 5 sigma test that has an intrinsic probability to fail of 1 over ~2 millions
        error = np.abs(np.sum(ele_data) - expected_total_physical_pairs_created)
        print(
            "difference between expected and actual number of pairs created: "
            + str(error)
        )
        print("tolerance: " + str(5 * std_total_physical_pairs_created))
        assert error < 5 * std_total_physical_pairs_created


do_analysis(Ex_test, Ey_test, Ez_test, Bx_test, By_test, Bz_test)
