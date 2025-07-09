#!/usr/bin/env python3

# This test checks
# The box is filled with photons with same number density and opposite momenta.
# The photons are not pushed and collide via linear Breit-Wheeler.
# We estimate the electron-positron yield in time only from the input parameters.
# The number of electrons and positrons in time is N_+(t) = N_-(t) and follows this law:
# d^2 N_+ / (dt dV) = sigma * n_A * n_B * sqrt[ (v_A - v_B )^2 - ( v_A x v_B )^2 /c^2 ] = 2*c * sigma * n_A^2
# where dV is the differential volume, dt the differential time,
# n_A = n_B = N_A / V = N_B / V the photon (of species photonA and photonB) number density,
# sigma the linear Breit-Wheeler cross section, c the speed of light.
# Then we calculate dN_+/dt by integrating in space and compare with the simulation result.
# It also checks charge, energy, and momentum conservation.

import numpy as np
from analysis_base import (
    check_charge_conservation,
    check_energy_conservation,
    check_momentum_conservation,
    find_num_in_line,
)
from scipy.constants import c, m_e, physical_constants, pi
from scipy.integrate import cumulative_trapezoid

# constants
r_e = physical_constants["classical electron radius"][0]


# get input parameters from warpx_used_inputs
def get_input_parameters():
    with open("./warpx_used_inputs", "rt") as f:
        lines = f.readlines()
        for line in lines:
            if "warpx.cfl" in line:
                cfl = find_num_in_line(line)
            if "max_step" in line:
                num_steps = find_num_in_line(line)
            if "geometry.prob_lo" in line:
                xmin, ymin, zmin = find_num_in_line(line)
            if "geometry.prob_hi" in line:
                xmax, ymax, zmax = find_num_in_line(line)
            if "photonA.ux" in line:
                uAx = find_num_in_line(line)
            if "photonA.uy" in line:
                uAy = find_num_in_line(line)
            if "photonA.uz" in line:
                uAz = find_num_in_line(line)
            if "photonA.density" in line:
                dens = find_num_in_line(line)
            if "photonB.density" in line:
                dens = find_num_in_line(line)
            if "photonB.ux" in line:
                uBx = find_num_in_line(line)
            if "photonB.uy" in line:
                uBy = find_num_in_line(line)
            if "photonB.uz" in line:
                uBz = find_num_in_line(line)
            if "amr.n_cell" in line:
                nx, ny, nz = find_num_in_line(line)
    pAx, pAy, pAz = uAx * m_e * c, uAy * m_e * c, uAz * m_e * c
    pBx, pBy, pBz = uBx * m_e * c, uBy * m_e * c, uBz * m_e * c
    EA, EB = (
        np.sqrt(pAx**2 + pAy**2 + pAz**2) * c,
        np.sqrt(pBx**2 + pBy**2 + pBz**2) * c,
    )
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz
    dt = (
        cfl / c / np.sqrt(1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2)
    )  # works for Yee solver
    pA = np.sqrt(pAx**2 + pAy**2 + pAz**2)
    pB = np.sqrt(pBx**2 + pBy**2 + pBz**2)
    cos_ang = (pAx * pBx + pAy * pBy + pAz * pBz) / (pA * pB)
    theta = np.arccos(cos_ang)
    V = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    NA0 = dens * V
    NB0 = dens * V
    return (EA, EB, theta, dt, V, num_steps, NA0, NB0)


def cross_section(E1_lab, E2_lab, theta):
    s = E1_lab * E2_lab / (2.0 * m_e**2 * c**4) * (1.0 - np.cos(theta))
    beta = np.sqrt(1.0 - 1.0 / s)
    factor1 = 0.5 * pi * r_e**2 * (1.0 - beta**2)
    term1 = (3.0 - beta**4) * np.log((1.0 + beta) / (1.0 - beta))
    term2 = 2 * beta * (beta**2 - 2.0)
    factor2 = term1 + term2
    sigma = factor1 * factor2
    return sigma


def check_pair_rate():
    (EA_lab, EB_lab, theta, dt, V, num_steps, NA0, NB0) = get_input_parameters()

    t = np.arange(num_steps + 1) * dt
    sigma = cross_section(EA_lab, EB_lab, theta)

    # estimated number of real photons of species photonA in time
    NA_est = NA0 / (1.0 + 2.0 * sigma * c * t / V * NA0)
    # number of <<real>> photons of species photonA in time from simulation
    NA = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[:, 8]

    # estimated number of real photons of species photonB in time
    NB_est = NB0 / (1.0 + 2.0 * sigma * c * t / V * NB0)
    # number of <<real>> photons of species photonA in time from simulation
    NB = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[:, 9]
    # estimated number of real positrons in time
    Nplus_est = (
        2.0
        * sigma
        * c
        / V
        * cumulative_trapezoid(NA_est * NB_est, x=t, dx=dt, initial=0)
    )
    # number of <<real>> positrons in time from simulation
    Nplus = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[:, 11]

    assert np.all(np.isclose(Nplus_est, Nplus, rtol=1e-1, atol=0.0))
    assert np.all(np.isclose(NA_est, NA, rtol=1e-1, atol=0.0))
    assert np.all(np.isclose(NB_est, NB, rtol=1e-1, atol=0.0))


def main():
    check_energy_conservation()
    check_momentum_conservation()
    check_charge_conservation()
    check_pair_rate()


if __name__ == "__main__":
    main()
