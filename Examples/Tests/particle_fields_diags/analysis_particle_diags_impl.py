#!/usr/bin/env python3

# Copyright 2019-2022 Luca Fedeli, Yinjian Zhao, Hannah Klion
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# This script tests the reduced particle diagnostics.
# The setup is a uniform plasma with electrons, protons and photons.
# Various particle and field quantities are written to file using the reduced diagnostics
# and compared with the corresponding quantities computed from the data in the plotfiles.

import sys

import numpy as np
import openpmd_api as io
import yt
from scipy.constants import c, e, m_e, m_p


def do_analysis(single_precision=False):
    fn = sys.argv[1]

    ds = yt.load(fn)
    ad = ds.all_data()
    ad0 = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )

    opmd = io.Series("diags/openpmd/openpmd_%T.h5", io.Access.read_only)
    opmd_i = opmd.iterations[200]

    # --------------------------------------------------------------------------------------------------
    # Part 1: get results from plotfiles (label '_yt')
    # --------------------------------------------------------------------------------------------------

    # Quantities computed from plotfiles
    values_yt = dict()

    domain_size = ds.domain_right_edge.value - ds.domain_left_edge.value
    dx = domain_size / ds.domain_dimensions

    # Electrons
    x = ad["electrons", "particle_position_x"].to_ndarray()
    y = ad["electrons", "particle_position_y"].to_ndarray()
    z = ad["electrons", "particle_position_z"].to_ndarray()
    uz = ad["electrons", "particle_momentum_z"].to_ndarray() / m_e / c
    w = ad["electrons", "particle_weight"].to_ndarray()
    filt = uz < 0

    x_ind = ((x - ds.domain_left_edge[0].value) / dx[0]).astype(int)
    y_ind = ((y - ds.domain_left_edge[1].value) / dx[1]).astype(int)
    z_ind = ((z - ds.domain_left_edge[2].value) / dx[2]).astype(int)

    zavg = np.zeros(ds.domain_dimensions)
    uzavg = np.zeros(ds.domain_dimensions)
    zuzavg = np.zeros(ds.domain_dimensions)
    wavg = np.zeros(ds.domain_dimensions)
    uzavg_filt = np.zeros(ds.domain_dimensions)
    wavg_filt = np.zeros(ds.domain_dimensions)

    for i_p in range(len(x)):
        zavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += z[i_p] * w[i_p]
        uzavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += uz[i_p] * w[i_p]
        zuzavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += z[i_p] * uz[i_p] * w[i_p]
        wavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += w[i_p]
        uzavg_filt[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += uz[i_p] * w[i_p] * filt[i_p]
        wavg_filt[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += w[i_p] * filt[i_p]

    wavg_adj = np.where(wavg == 0, 1, wavg)
    wavg_filt_adj = np.where(wavg_filt == 0, 1, wavg_filt)
    values_yt["electrons: zavg"] = zavg / wavg_adj
    values_yt["electrons: uzavg"] = uzavg / wavg_adj
    values_yt["electrons: zuzavg"] = zuzavg / wavg_adj
    values_yt["electrons: uzavg_filt"] = uzavg_filt / wavg_filt_adj
    values_yt["electrons: jz"] = e * uzavg

    # protons
    x = ad["protons", "particle_position_x"].to_ndarray()
    y = ad["protons", "particle_position_y"].to_ndarray()
    z = ad["protons", "particle_position_z"].to_ndarray()
    uz = ad["protons", "particle_momentum_z"].to_ndarray() / m_p / c
    w = ad["protons", "particle_weight"].to_ndarray()
    filt = uz < 0

    x_ind = ((x - ds.domain_left_edge[0].value) / dx[0]).astype(int)
    y_ind = ((y - ds.domain_left_edge[1].value) / dx[1]).astype(int)
    z_ind = ((z - ds.domain_left_edge[2].value) / dx[2]).astype(int)

    zavg = np.zeros(ds.domain_dimensions)
    uzavg = np.zeros(ds.domain_dimensions)
    zuzavg = np.zeros(ds.domain_dimensions)
    wavg = np.zeros(ds.domain_dimensions)
    uzavg_filt = np.zeros(ds.domain_dimensions)
    wavg_filt = np.zeros(ds.domain_dimensions)

    for i_p in range(len(x)):
        zavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += z[i_p] * w[i_p]
        uzavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += uz[i_p] * w[i_p]
        zuzavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += z[i_p] * uz[i_p] * w[i_p]
        wavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += w[i_p]
        uzavg_filt[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += uz[i_p] * w[i_p] * filt[i_p]
        wavg_filt[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += w[i_p] * filt[i_p]

    wavg_adj = np.where(wavg == 0, 1, wavg)
    wavg_filt_adj = np.where(wavg_filt == 0, 1, wavg_filt)
    values_yt["protons: zavg"] = zavg / wavg_adj
    values_yt["protons: uzavg"] = uzavg / wavg_adj
    values_yt["protons: zuzavg"] = zuzavg / wavg_adj
    values_yt["protons: uzavg_filt"] = uzavg_filt / wavg_filt_adj
    values_yt["protons: jz"] = e * uzavg

    # Photons (momentum in units of m_e c)
    x = ad["photons", "particle_position_x"].to_ndarray()
    y = ad["photons", "particle_position_y"].to_ndarray()
    z = ad["photons", "particle_position_z"].to_ndarray()
    uz = ad["photons", "particle_momentum_z"].to_ndarray() / m_e / c
    w = ad["photons", "particle_weight"].to_ndarray()
    filt = uz < 0

    x_ind = ((x - ds.domain_left_edge[0].value) / dx[0]).astype(int)
    y_ind = ((y - ds.domain_left_edge[1].value) / dx[1]).astype(int)
    z_ind = ((z - ds.domain_left_edge[2].value) / dx[2]).astype(int)

    zavg = np.zeros(ds.domain_dimensions)
    uzavg = np.zeros(ds.domain_dimensions)
    zuzavg = np.zeros(ds.domain_dimensions)
    wavg = np.zeros(ds.domain_dimensions)
    uzavg_filt = np.zeros(ds.domain_dimensions)
    wavg_filt = np.zeros(ds.domain_dimensions)

    for i_p in range(len(x)):
        zavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += z[i_p] * w[i_p]
        uzavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += uz[i_p] * w[i_p]
        zuzavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += z[i_p] * uz[i_p] * w[i_p]
        wavg[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += w[i_p]
        uzavg_filt[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += uz[i_p] * w[i_p] * filt[i_p]
        wavg_filt[x_ind[i_p], y_ind[i_p], z_ind[i_p]] += w[i_p] * filt[i_p]

    wavg_adj = np.where(wavg == 0, 1, wavg)
    wavg_filt_adj = np.where(wavg_filt == 0, 1, wavg_filt)
    values_yt["photons: zavg"] = zavg / wavg_adj
    values_yt["photons: uzavg"] = uzavg / wavg_adj
    values_yt["photons: zuzavg"] = zuzavg / wavg_adj
    values_yt["photons: uzavg_filt"] = uzavg_filt / wavg_filt_adj
    values_yt["photons: jz"] = e * uzavg

    values_rd = dict()
    # Load reduced particle diagnostic data from plotfiles
    values_rd["electrons: zavg"] = ad0[("boxlib", "z_electrons")]
    values_rd["protons: zavg"] = ad0[("boxlib", "z_protons")]
    values_rd["photons: zavg"] = ad0[("boxlib", "z_photons")]

    values_rd["electrons: uzavg"] = ad0[("boxlib", "uz_electrons")]
    values_rd["protons: uzavg"] = ad0[("boxlib", "uz_protons")]
    values_rd["photons: uzavg"] = ad0[("boxlib", "uz_photons")]

    values_rd["electrons: zuzavg"] = ad0[("boxlib", "zuz_electrons")]
    values_rd["protons: zuzavg"] = ad0[("boxlib", "zuz_protons")]
    values_rd["photons: zuzavg"] = ad0[("boxlib", "zuz_photons")]

    values_rd["electrons: uzavg_filt"] = ad0[("boxlib", "uz_filt_electrons")]
    values_rd["protons: uzavg_filt"] = ad0[("boxlib", "uz_filt_protons")]
    values_rd["photons: uzavg_filt"] = ad0[("boxlib", "uz_filt_photons")]

    values_rd["electrons: jz"] = ad0[("boxlib", "jz_electrons")]
    values_rd["protons: jz"] = ad0[("boxlib", "jz_protons")]
    values_rd["photons: jz"] = ad0[("boxlib", "jz_photons")]

    values_opmd = dict()
    # Load reduced particle diagnostic data from OPMD output
    values_opmd["electrons: zavg"] = opmd_i.meshes["z_electrons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()
    values_opmd["protons: zavg"] = opmd_i.meshes["z_protons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()
    values_opmd["photons: zavg"] = opmd_i.meshes["z_photons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()

    values_opmd["electrons: uzavg"] = opmd_i.meshes["uz_electrons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()
    values_opmd["protons: uzavg"] = opmd_i.meshes["uz_protons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()
    values_opmd["photons: uzavg"] = opmd_i.meshes["uz_photons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()

    values_opmd["electrons: zuzavg"] = opmd_i.meshes["zuz_electrons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()
    values_opmd["protons: zuzavg"] = opmd_i.meshes["zuz_protons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()
    values_opmd["photons: zuzavg"] = opmd_i.meshes["zuz_photons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()

    values_opmd["electrons: uzavg_filt"] = opmd_i.meshes["uz_filt_electrons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()
    values_opmd["protons: uzavg_filt"] = opmd_i.meshes["uz_filt_protons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()
    values_opmd["photons: uzavg_filt"] = opmd_i.meshes["uz_filt_photons"][
        io.Mesh_Record_Component.SCALAR
    ].load_chunk()

    values_opmd["electrons: jz"] = opmd_i.meshes["j_electrons"]["z"].load_chunk()
    values_opmd["protons: jz"] = opmd_i.meshes["j_protons"]["z"].load_chunk()
    values_opmd["photons: jz"] = opmd_i.meshes["j_photons"]["z"].load_chunk()

    opmd.flush()
    del opmd

    # --------------------------------------------------------------------------------------------------
    # Part 3: compare values from plotfiles and diagnostics and print output
    # --------------------------------------------------------------------------------------------------

    error_plt = dict()
    error_opmd = dict()
    tolerance = 5e-3 if single_precision else 1e-12

    for k in values_yt.keys():
        # check that the zeros line up, since we'll be ignoring them in the error calculation
        assert np.all((values_yt[k] == 0) == (values_rd[k] == 0))
        error_plt[k] = np.max(
            abs(values_yt[k] - values_rd[k])[values_yt[k] != 0]
            / abs(values_yt[k])[values_yt[k] != 0]
        )
        print(k, "relative error plotfile = ", error_plt[k])
        assert error_plt[k] < tolerance
        assert np.all((values_yt[k] == 0) == (values_opmd[k].T == 0))
        error_opmd[k] = np.max(
            abs(values_yt[k] - values_opmd[k].T)[values_yt[k] != 0]
            / abs(values_yt[k])[values_yt[k] != 0]
        )
        assert error_opmd[k] < tolerance
        print(k, "relative error openPMD = ", error_opmd[k])
