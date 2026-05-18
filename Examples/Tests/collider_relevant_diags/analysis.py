#!/usr/bin/env python3

import sys

import numpy as np
import openpmd_api as io
import pandas as pd
from scipy.constants import c, e, hbar, m_e

sys.path.append("../../../../warpx/Tools/Parser/")
from input_file_parser import parse_input_file

E_crit = m_e**2 * c**3 / (e * hbar)
B_crit = m_e**2 * c**2 / (e * hbar)


def chi(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz):
    gamma = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
    vx = ux / gamma * c
    vy = uy / gamma * c
    vz = uz / gamma * c
    tmp1x = Ex + vy * Bz - vz * By
    tmp1y = Ey - vx * Bz + vz * Bx
    tmp1z = Ez + vx * By - vy * Bx
    tmp2 = (Ex * vx + Ey * vy + Ez * vz) / c
    chi = gamma / E_crit * np.sqrt(tmp1x**2 + tmp1y**2 + tmp1z**2 - tmp2**2)
    return chi


def dL_dt():
    series = io.Series("diags/diag2/openpmd_%T.h5", io.Access.read_only)
    iterations = np.asarray(series.iterations)
    lumi = []
    for n, ts in enumerate(iterations):
        it = series.iterations[ts]
        rho1 = it.meshes["rho_beam_e"]
        dV = np.prod(rho1.grid_spacing)
        rho1 = it.meshes["rho_beam_e"][io.Mesh_Record_Component.SCALAR].load_chunk()
        rho2 = it.meshes["rho_beam_p"][io.Mesh_Record_Component.SCALAR].load_chunk()
        beam_e_charge = it.particles["beam_e"]["charge"][
            io.Mesh_Record_Component.SCALAR
        ].load_chunk()
        beam_p_charge = it.particles["beam_p"]["charge"][
            io.Mesh_Record_Component.SCALAR
        ].load_chunk()
        q1 = beam_e_charge[0]
        if not np.all(beam_e_charge == q1):
            sys.exit("beam_e particles do not have the same charge")
        q2 = beam_p_charge[0]
        if not np.all(beam_p_charge == q2):
            sys.exit("beam_p particles do not have the same charge")
        series.flush()
        n1 = rho1 / q1
        n2 = rho2 / q2
        ln = 2 * np.sum(n1 * n2) * dV * c
        lumi.append(ln)
    return lumi


input_dict = parse_input_file("warpx_used_inputs")
Ex, Ey, Ez = [float(w) for w in input_dict["particles.E_external_particle"]]
Bx, By, Bz = [float(w) for w in input_dict["particles.B_external_particle"]]

CollDiagFname = "diags/reducedfiles/ColliderRelevant_beam_e_beam_p.txt"
df = pd.read_csv(CollDiagFname, sep=" ", header=0)

for species in ["beam_p", "beam_e"]:
    ux1, ux2, ux3 = [float(w) for w in input_dict[f"{species}.multiple_particles_ux"]]
    uy1, uy2, uy3 = [float(w) for w in input_dict[f"{species}.multiple_particles_uy"]]
    uz1, uz2, uz3 = [float(w) for w in input_dict[f"{species}.multiple_particles_uz"]]

    x = np.array([float(w) for w in input_dict[f"{species}.multiple_particles_pos_x"]])
    y = np.array([float(w) for w in input_dict[f"{species}.multiple_particles_pos_y"]])

    w = np.array([float(w) for w in input_dict[f"{species}.multiple_particles_weight"]])

    CHI_ANALYTICAL = np.array(
        [
            chi(ux1, uy1, uz1, Ex, Ey, Ez, Bx, By, Bz),
            chi(ux2, uy2, uz2, Ex, Ey, Ez, Bx, By, Bz),
            chi(ux3, uy3, uz3, Ex, Ey, Ez, Bx, By, Bz),
        ]
    )
    THETAX = np.array(
        [np.arctan2(ux1, uz1), np.arctan2(ux2, uz2), np.arctan2(ux3, uz3)]
    )
    THETAY = np.array(
        [np.arctan2(uy1, uz1), np.arctan2(uy2, uz2), np.arctan2(uy3, uz3)]
    )

    # CHI MAX
    fname = f"diags/reducedfiles/ParticleExtrema_{species}.txt"
    chimax_pe = np.loadtxt(fname)[:, 19]
    chimax_cr = df[
        [col for col in df.columns if f"chi_max_{species}" in col]
    ].to_numpy()
    assert np.allclose(np.max(CHI_ANALYTICAL), chimax_cr, rtol=1e-8)
    assert np.allclose(chimax_pe, chimax_cr, rtol=1e-8)

    # CHI MIN
    fname = f"diags/reducedfiles/ParticleExtrema_{species}.txt"
    chimin_pe = np.loadtxt(fname)[:, 18]
    chimin_cr = df[
        [col for col in df.columns if f"chi_min_{species}" in col]
    ].to_numpy()
    assert np.allclose(np.min(CHI_ANALYTICAL), chimin_cr, rtol=1e-8)
    assert np.allclose(chimin_pe, chimin_cr, rtol=1e-8)

    # CHI AVERAGE
    chiave_cr = df[
        [col for col in df.columns if f"chi_ave_{species}" in col]
    ].to_numpy()
    assert np.allclose(np.average(CHI_ANALYTICAL, weights=w), chiave_cr, rtol=1e-8)

    # X AVE STD
    x_ave_cr = df[[col for col in df.columns if f"]x_ave_{species}" in col]].to_numpy()
    x_std_cr = df[[col for col in df.columns if f"]x_std_{species}" in col]].to_numpy()
    x_ave = np.average(x, weights=w)
    x_std = np.sqrt(np.average((x - x_ave) ** 2, weights=w))
    assert np.allclose(x_ave, x_ave_cr, rtol=1e-8)
    assert np.allclose(x_std, x_std_cr, rtol=1e-8)

    # Y AVE STD
    y_ave_cr = df[[col for col in df.columns if f"]y_ave_{species}" in col]].to_numpy()
    y_std_cr = df[[col for col in df.columns if f"]y_std_{species}" in col]].to_numpy()
    y_ave = np.average(y, weights=w)
    y_std = np.sqrt(np.average((y - y_ave) ** 2, weights=w))
    assert np.allclose(y_ave, y_ave_cr, rtol=1e-8)
    assert np.allclose(y_std, y_std_cr, rtol=1e-8)

    # THETA X MIN AVE MAX STD
    thetax_min_cr = df[
        [col for col in df.columns if f"theta_x_min_{species}" in col]
    ].to_numpy()
    thetax_ave_cr = df[
        [col for col in df.columns if f"theta_x_ave_{species}" in col]
    ].to_numpy()
    thetax_max_cr = df[
        [col for col in df.columns if f"theta_x_max_{species}" in col]
    ].to_numpy()
    thetax_std_cr = df[
        [col for col in df.columns if f"theta_x_std_{species}" in col]
    ].to_numpy()
    thetax_min = np.min(THETAX)
    thetax_ave = np.average(THETAX, weights=w)
    thetax_max = np.max(THETAX)
    thetax_std = np.sqrt(np.average((THETAX - thetax_ave) ** 2, weights=w))
    assert np.allclose(thetax_min, thetax_min_cr, rtol=1e-8)
    assert np.allclose(thetax_ave, thetax_ave_cr, rtol=1e-8)
    assert np.allclose(thetax_max, thetax_max_cr, rtol=1e-8)
    assert np.allclose(thetax_std, thetax_std_cr, rtol=1e-8)

    # THETA Y MIN AVE MAX STD
    thetay_min_cr = df[
        [col for col in df.columns if f"theta_y_min_{species}" in col]
    ].to_numpy()
    thetay_ave_cr = df[
        [col for col in df.columns if f"theta_y_ave_{species}" in col]
    ].to_numpy()
    thetay_max_cr = df[
        [col for col in df.columns if f"theta_y_max_{species}" in col]
    ].to_numpy()
    thetay_std_cr = df[
        [col for col in df.columns if f"theta_y_std_{species}" in col]
    ].to_numpy()
    thetay_min = np.min(THETAY)
    thetay_ave = np.average(THETAY, weights=w)
    thetay_max = np.max(THETAY)
    thetay_std = np.sqrt(np.average((THETAY - thetay_ave) ** 2, weights=w))
    assert np.allclose(thetay_min, thetay_min_cr, rtol=1e-8)
    assert np.allclose(thetay_ave, thetay_ave_cr, rtol=1e-8)
    assert np.allclose(thetay_max, thetay_max_cr, rtol=1e-8)
    assert np.allclose(thetay_std, thetay_std_cr, rtol=1e-8)

    # dL/dt
    dL_dt_cr = df[[col for col in df.columns if "dL_dt" in col]].to_numpy()
    assert np.allclose(dL_dt_cr, dL_dt(), rtol=1e-8)
