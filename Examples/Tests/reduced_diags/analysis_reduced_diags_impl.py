#!/usr/bin/env python3

# Copyright 2019-2021 Luca Fedeli, Yinjian Zhao
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the reduced diagnostics.
# The setup is a uniform plasma with electrons, protons and photons.
# Various particle and field quantities are written to file using the reduced diagnostics
# and compared with the corresponding quantities computed from the data in the plotfiles.

import sys

import numpy as np
import yt
from scipy.constants import c, m_e, m_p
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0

# gamma threshold to switch between the relativistic expression of
# the kinetic energy and its Taylor expansion.
gamma_relativistic_threshold = 1.005


def do_analysis(single_precision=False):
    fn = sys.argv[1]

    ds = yt.load(fn)
    ad = ds.all_data()

    # --------------------------------------------------------------------------------------------------
    # Part 1: get results from plotfiles (label '_yt')
    # --------------------------------------------------------------------------------------------------

    # Quantities computed from plotfiles
    values_yt = dict()

    # Electrons
    px = ad["electrons", "particle_momentum_x"].to_ndarray()
    py = ad["electrons", "particle_momentum_y"].to_ndarray()
    pz = ad["electrons", "particle_momentum_z"].to_ndarray()
    w = ad["electrons", "particle_weight"].to_ndarray()
    p2 = px**2 + py**2 + pz**2

    # Accumulate particle energy, store number of particles and sum of weights
    e_u2 = p2 / (m_e**2 * c**2)
    e_gamma = np.sqrt(1 + e_u2)
    e_energy = (m_e * c**2) * np.where(
        e_gamma > gamma_relativistic_threshold,
        e_gamma - 1,
        (e_u2) / 2
        - (e_u2**2) / 8
        + (e_u2**3) / 16
        - (e_u2**4) * (5 / 128)
        + (e_u2**5) * (7 / 256),
    )
    values_yt["electrons: particle energy"] = np.sum(e_energy * w)
    values_yt["electrons: particle momentum in x"] = np.sum(px * w)
    values_yt["electrons: particle momentum in y"] = np.sum(py * w)
    values_yt["electrons: particle momentum in z"] = np.sum(pz * w)
    values_yt["electrons: number of particles"] = w.shape[0]
    values_yt["electrons: sum of weights"] = np.sum(w)

    # Protons
    px = ad["protons", "particle_momentum_x"].to_ndarray()
    py = ad["protons", "particle_momentum_y"].to_ndarray()
    pz = ad["protons", "particle_momentum_z"].to_ndarray()
    w = ad["protons", "particle_weight"].to_ndarray()
    p2 = px**2 + py**2 + pz**2

    # Accumulate particle energy, store number of particles and sum of weights
    p_u2 = p2 / (m_p**2 * c**2)
    p_gamma = np.sqrt(1 + p_u2)
    p_energy = (m_p * c**2) * np.where(
        p_gamma > gamma_relativistic_threshold,
        p_gamma - 1,
        (p_u2) / 2
        - (p_u2**2) / 8
        + (p_u2**3) / 16
        - (p_u2**4) * (5 / 128)
        + (p_u2**5) * (7 / 256),
    )
    values_yt["protons: particle energy"] = np.sum(p_energy * w)
    values_yt["protons: particle momentum in x"] = np.sum(px * w)
    values_yt["protons: particle momentum in y"] = np.sum(py * w)
    values_yt["protons: particle momentum in z"] = np.sum(pz * w)
    values_yt["protons: number of particles"] = w.shape[0]
    values_yt["protons: sum of weights"] = np.sum(w)

    # Photons
    px = ad["photons", "particle_momentum_x"].to_ndarray()
    py = ad["photons", "particle_momentum_y"].to_ndarray()
    pz = ad["photons", "particle_momentum_z"].to_ndarray()
    w = ad["photons", "particle_weight"].to_ndarray()
    p2 = px**2 + py**2 + pz**2

    # Accumulate particle energy, store number of particles and sum of weights
    values_yt["photons: particle energy"] = np.sum(np.sqrt(p2 * c**2) * w)
    values_yt["photons: particle momentum in x"] = np.sum(px * w)
    values_yt["photons: particle momentum in y"] = np.sum(py * w)
    values_yt["photons: particle momentum in z"] = np.sum(pz * w)
    values_yt["photons: number of particles"] = w.shape[0]
    values_yt["photons: sum of weights"] = np.sum(w)

    # Accumulate total particle diagnostics

    values_yt["particle energy"] = (
        values_yt["electrons: particle energy"]
        + values_yt["protons: particle energy"]
        + values_yt["photons: particle energy"]
    )

    values_yt["particle momentum in x"] = (
        values_yt["electrons: particle momentum in x"]
        + values_yt["protons: particle momentum in x"]
        + values_yt["photons: particle momentum in x"]
    )

    values_yt["particle momentum in y"] = (
        values_yt["electrons: particle momentum in y"]
        + values_yt["protons: particle momentum in y"]
        + values_yt["photons: particle momentum in y"]
    )

    values_yt["particle momentum in z"] = (
        values_yt["electrons: particle momentum in z"]
        + values_yt["protons: particle momentum in z"]
        + values_yt["photons: particle momentum in z"]
    )

    values_yt["number of particles"] = (
        values_yt["electrons: number of particles"]
        + values_yt["protons: number of particles"]
        + values_yt["photons: number of particles"]
    )

    values_yt["sum of weights"] = (
        values_yt["electrons: sum of weights"]
        + values_yt["protons: sum of weights"]
        + values_yt["photons: sum of weights"]
    )

    values_yt["mean particle energy"] = (
        values_yt["particle energy"] / values_yt["sum of weights"]
    )

    values_yt["mean particle momentum in x"] = (
        values_yt["particle momentum in x"] / values_yt["sum of weights"]
    )
    values_yt["mean particle momentum in y"] = (
        values_yt["particle momentum in y"] / values_yt["sum of weights"]
    )
    values_yt["mean particle momentum in z"] = (
        values_yt["particle momentum in z"] / values_yt["sum of weights"]
    )

    values_yt["electrons: mean particle energy"] = (
        values_yt["electrons: particle energy"] / values_yt["electrons: sum of weights"]
    )

    values_yt["electrons: mean particle momentum in x"] = (
        values_yt["electrons: particle momentum in x"]
        / values_yt["electrons: sum of weights"]
    )
    values_yt["electrons: mean particle momentum in y"] = (
        values_yt["electrons: particle momentum in y"]
        / values_yt["electrons: sum of weights"]
    )
    values_yt["electrons: mean particle momentum in z"] = (
        values_yt["electrons: particle momentum in z"]
        / values_yt["electrons: sum of weights"]
    )

    values_yt["protons: mean particle energy"] = (
        values_yt["protons: particle energy"] / values_yt["protons: sum of weights"]
    )

    values_yt["protons: mean particle momentum in x"] = (
        values_yt["protons: particle momentum in x"]
        / values_yt["protons: sum of weights"]
    )
    values_yt["protons: mean particle momentum in y"] = (
        values_yt["protons: particle momentum in y"]
        / values_yt["protons: sum of weights"]
    )
    values_yt["protons: mean particle momentum in z"] = (
        values_yt["protons: particle momentum in z"]
        / values_yt["protons: sum of weights"]
    )

    values_yt["photons: mean particle energy"] = (
        values_yt["photons: particle energy"] / values_yt["photons: sum of weights"]
    )

    values_yt["photons: mean particle momentum in x"] = (
        values_yt["photons: particle momentum in x"]
        / values_yt["photons: sum of weights"]
    )
    values_yt["photons: mean particle momentum in y"] = (
        values_yt["photons: particle momentum in y"]
        / values_yt["photons: sum of weights"]
    )
    values_yt["photons: mean particle momentum in z"] = (
        values_yt["photons: particle momentum in z"]
        / values_yt["photons: sum of weights"]
    )

    # Load 3D data from plotfiles
    ad = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    Ex = ad[("mesh", "Ex")].to_ndarray()
    Ey = ad[("mesh", "Ey")].to_ndarray()
    Ez = ad[("mesh", "Ez")].to_ndarray()
    Bx = ad[("mesh", "Bx")].to_ndarray()
    By = ad[("mesh", "By")].to_ndarray()
    Bz = ad[("mesh", "Bz")].to_ndarray()
    jx = ad[("mesh", "jx")].to_ndarray()
    jy = ad[("mesh", "jy")].to_ndarray()
    jz = ad[("mesh", "jz")].to_ndarray()
    rho = ad[("boxlib", "rho")].to_ndarray()
    rho_electrons = ad[("boxlib", "rho_electrons")].to_ndarray()
    rho_protons = ad[("boxlib", "rho_protons")].to_ndarray()
    x = ad[("boxlib", "x")].to_ndarray()
    y = ad[("boxlib", "y")].to_ndarray()
    z = ad[("boxlib", "z")].to_ndarray()

    # Field energy
    E2 = np.sum(Ex**2) + np.sum(Ey**2) + np.sum(Ez**2)
    B2 = np.sum(Bx**2) + np.sum(By**2) + np.sum(Bz**2)
    N = np.array(ds.domain_width / ds.domain_dimensions)
    dV = N[0] * N[1] * N[2]
    values_yt["field energy"] = 0.5 * dV * (E2 * eps0 + B2 / mu0)
    values_yt["field momentum in x"] = eps0 * np.sum(Ey * Bz - Ez * By) * dV
    values_yt["field momentum in y"] = eps0 * np.sum(Ez * Bx - Ex * Bz) * dV
    values_yt["field momentum in z"] = eps0 * np.sum(Ex * By - Ey * Bx) * dV

    # Field energy in quarter of simulation domain
    E2 = np.sum((Ex**2 + Ey**2 + Ez**2) * (y > 0) * (z < 0))
    B2 = np.sum((Bx**2 + By**2 + Bz**2) * (y > 0) * (z < 0))
    values_yt["field energy in quarter of simulation domain"] = (
        0.5 * dV * (E2 * eps0 + B2 / mu0)
    )

    # Max/min values of various grid quantities
    values_yt["maximum of |Ex|"] = np.amax(np.abs(Ex))
    values_yt["maximum of |Ey|"] = np.amax(np.abs(Ey))
    values_yt["maximum of |Ez|"] = np.amax(np.abs(Ez))
    values_yt["maximum of |Bx|"] = np.amax(np.abs(Bx))
    values_yt["maximum of |By|"] = np.amax(np.abs(By))
    values_yt["maximum of |Bz|"] = np.amax(np.abs(Bz))
    values_yt["maximum of |E|"] = np.amax(np.sqrt(Ex**2 + Ey**2 + Ez**2))
    values_yt["maximum of |B|"] = np.amax(np.sqrt(Bx**2 + By**2 + Bz**2))
    values_yt["maximum of rho"] = np.amax(rho)
    values_yt["minimum of rho"] = np.amin(rho)
    values_yt["electrons: maximum of |rho|"] = np.amax(np.abs(rho_electrons))
    values_yt["protons: maximum of |rho|"] = np.amax(np.abs(rho_protons))
    values_yt["maximum of |B| from generic field reduction"] = np.amax(
        np.sqrt(Bx**2 + By**2 + Bz**2)
    )
    values_yt["minimum of x*Ey*Bz"] = np.amin(x * Ey * Bz)
    values_yt["maximum of Edotj"] = np.amax(Ex * jx + Ey * jy + Ez * jz)

    # --------------------------------------------------------------------------------------------------
    # Part 2: get results from reduced diagnostics (label '_rd')
    # --------------------------------------------------------------------------------------------------

    # Quantities computed from reduced diagnostics
    values_rd = dict()

    # Load data from output files
    EFdata = np.genfromtxt("./diags/reducedfiles/EF.txt")  # Field energy
    EPdata = np.genfromtxt("./diags/reducedfiles/EP.txt")  # Particle energy
    PFdata = np.genfromtxt("./diags/reducedfiles/PF.txt")  # Field momentum
    PPdata = np.genfromtxt("./diags/reducedfiles/PP.txt")  # Particle momentum
    MFdata = np.genfromtxt("./diags/reducedfiles/MF.txt")  # Field maximum
    MRdata = np.genfromtxt("./diags/reducedfiles/MR.txt")  # Rho maximum
    NPdata = np.genfromtxt("./diags/reducedfiles/NP.txt")  # Particle number
    FR_Maxdata = np.genfromtxt(
        "./diags/reducedfiles/FR_Max.txt"
    )  # Field Reduction using maximum
    FR_Mindata = np.genfromtxt(
        "./diags/reducedfiles/FR_Min.txt"
    )  # Field Reduction using minimum
    FR_Integraldata = np.genfromtxt(
        "./diags/reducedfiles/FR_Integral.txt"
    )  # Field Reduction using integral
    Edotjdata = np.genfromtxt("./diags/reducedfiles/Edotj.txt")  # E dot j maximum

    # First index "1" points to the values written at the last time step
    values_rd["field energy"] = EFdata[1][2]
    values_rd["field energy in quarter of simulation domain"] = FR_Integraldata[1][2]
    values_rd["particle energy"] = EPdata[1][2]
    values_rd["electrons: particle energy"] = EPdata[1][3]
    values_rd["protons: particle energy"] = EPdata[1][4]
    values_rd["photons: particle energy"] = EPdata[1][5]
    values_rd["mean particle energy"] = EPdata[1][6]
    values_rd["electrons: mean particle energy"] = EPdata[1][7]
    values_rd["protons: mean particle energy"] = EPdata[1][8]
    values_rd["photons: mean particle energy"] = EPdata[1][9]
    values_rd["field momentum in x"] = PFdata[1][2]
    values_rd["field momentum in y"] = PFdata[1][3]
    values_rd["field momentum in z"] = PFdata[1][4]
    values_rd["particle momentum in x"] = PPdata[1][2]
    values_rd["particle momentum in y"] = PPdata[1][3]
    values_rd["particle momentum in z"] = PPdata[1][4]
    values_rd["electrons: particle momentum in x"] = PPdata[1][5]
    values_rd["electrons: particle momentum in y"] = PPdata[1][6]
    values_rd["electrons: particle momentum in z"] = PPdata[1][7]
    values_rd["protons: particle momentum in x"] = PPdata[1][8]
    values_rd["protons: particle momentum in y"] = PPdata[1][9]
    values_rd["protons: particle momentum in z"] = PPdata[1][10]
    values_rd["photons: particle momentum in x"] = PPdata[1][11]
    values_rd["photons: particle momentum in y"] = PPdata[1][12]
    values_rd["photons: particle momentum in z"] = PPdata[1][13]
    values_rd["mean particle momentum in x"] = PPdata[1][14]
    values_rd["mean particle momentum in y"] = PPdata[1][15]
    values_rd["mean particle momentum in z"] = PPdata[1][16]
    values_rd["electrons: mean particle momentum in x"] = PPdata[1][17]
    values_rd["electrons: mean particle momentum in y"] = PPdata[1][18]
    values_rd["electrons: mean particle momentum in z"] = PPdata[1][19]
    values_rd["protons: mean particle momentum in x"] = PPdata[1][20]
    values_rd["protons: mean particle momentum in y"] = PPdata[1][21]
    values_rd["protons: mean particle momentum in z"] = PPdata[1][22]
    values_rd["photons: mean particle momentum in x"] = PPdata[1][23]
    values_rd["photons: mean particle momentum in y"] = PPdata[1][24]
    values_rd["photons: mean particle momentum in z"] = PPdata[1][25]
    values_rd["maximum of |Ex|"] = MFdata[1][2]
    values_rd["maximum of |Ey|"] = MFdata[1][3]
    values_rd["maximum of |Ez|"] = MFdata[1][4]
    values_rd["maximum of |E|"] = MFdata[1][5]
    values_rd["maximum of |Bx|"] = MFdata[1][6]
    values_rd["maximum of |By|"] = MFdata[1][7]
    values_rd["maximum of |Bz|"] = MFdata[1][8]
    values_rd["maximum of |B|"] = MFdata[1][9]
    values_rd["maximum of rho"] = MRdata[1][2]
    values_rd["minimum of rho"] = MRdata[1][3]
    values_rd["electrons: maximum of |rho|"] = MRdata[1][4]
    values_rd["protons: maximum of |rho|"] = MRdata[1][5]
    values_rd["number of particles"] = NPdata[1][2]
    values_rd["electrons: number of particles"] = NPdata[1][3]
    values_rd["protons: number of particles"] = NPdata[1][4]
    values_rd["photons: number of particles"] = NPdata[1][5]
    values_rd["sum of weights"] = NPdata[1][6]
    values_rd["electrons: sum of weights"] = NPdata[1][7]
    values_rd["protons: sum of weights"] = NPdata[1][8]
    values_rd["photons: sum of weights"] = NPdata[1][9]
    values_rd["maximum of |B| from generic field reduction"] = FR_Maxdata[1][2]
    values_rd["minimum of x*Ey*Bz"] = FR_Mindata[1][2]
    values_rd["maximum of Edotj"] = Edotjdata[1][2]

    # --------------------------------------------------------------------------------------------------
    # Part 3: compare values from plotfiles and reduced diagnostics and print output
    # --------------------------------------------------------------------------------------------------

    error = dict()
    tolerance = 5e-3 if single_precision else 1e-12
    field_energy_tolerance = 0.3

    # The comparison of field energies requires a large tolerance,
    # because the field energy from the plotfiles is computed from cell-centered data,
    # while the field energy from the reduced diagnostics is computed from (Yee) staggered data.
    for k in values_yt.keys():
        print()
        print("values_yt[" + k + "] = ", values_yt[k])
        print("values_rd[" + k + "] = ", values_rd[k])
        error[k] = abs(values_yt[k] - values_rd[k]) / abs(values_yt[k])
        print("relative error = ", error[k])
        tol = field_energy_tolerance if (k == "field energy") else tolerance
        assert error[k] < tol
        print()
