#!/usr/bin/env python3

# Copyright 2019-2020 Luca Fedeli, Maxence Thevenet
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


'''
Analysis script of a WarpX simulation of rigid injection.

A Gaussian electron beam starts from -5 microns, propagates rigidly up to
20 microns after which it expands due to emittance only (the focal position is
20 microns). The beam width is measured after ~50 microns, and compared with
the theory (with a 5% error allowed).

As a help to the user, the script also compares beam width to the theory in
case rigid injection is OFF (i.e., the beam starts expanding from -5 microns),
in which case a warning is raised.

Additionally, this script tests that runtime attributes are correctly initialized
with the gaussian_beam injection style.
'''

import glob
import os

import numpy as np

#yt.funcs.mylog.setLevel(0)
#sys.path.insert(1, '../../../../warpx/Regression/Checksum/')
#import checksumAPI

#filename = sys.argv[1]


def do_analysis(filename):
    import openpmd_api as io

    s = io.Series(filename, io.Access.read_only)
    beam = s.iterations[280].particles["beam"]
    df = beam.to_df()
    print(df)

def compute_particles(n_particles):
    import pandas as pd
    from scipy.constants import e, m_e

    # Create physical quantities
    id = np.arange(1,n_particles+1)
    charge = np.ones(n_particles) * (-e)
    mass = np.ones(n_particles) * m_e
    momentum_x = np.random.normal(loc=0.0, scale=100., size=n_particles)
    momentum_y = np.random.normal(loc=0.0, scale=100., size=n_particles)
    momentum_z = np.random.normal(loc=1000., scale=0., size=n_particles)
    position_x = np.random.normal(loc=0.0, scale=1.e-6, size=n_particles)
    position_y = np.random.normal(loc=0.0, scale=1.e-6, size=n_particles)
    position_z = np.random.normal(loc=-5.e-6, scale=.5e-6, size=n_particles)
    positionOffset_x = np.zeros(n_particles)
    positionOffset_y = np.zeros(n_particles)
    positionOffset_z = np.zeros(n_particles)
    weighting = np.ones(n_particles) * 3.121e05

    # Store data in a pandas dataframe
    data = {'id': id,
            'charge': charge,
            'mass': mass,
            'momentum_x': momentum_x,
            'momentum_y': momentum_y,
            'momentum_z': momentum_z,
            'position_x': position_x,
            'position_y': position_y,
            'position_z': position_z,
            'positionOffset_x': positionOffset_x,
            'positionOffset_y': positionOffset_y,
            'positionOffset_z': positionOffset_z,
            'weighting': weighting
            }

    df = pd.DataFrame(data)
    return(df)

def h5_create(h5name, n_particles):
    from openpmd_api import Access, Dataset, Mesh_Record_Component, Series

    SCALAR = Mesh_Record_Component.SCALAR

    df = compute_particles(n_particles)

    f = Series(
        h5name,
        Access.create
    )

    f.particles_path = "particles"

    cur_it = f.iterations[0]

    # particles
    beam = cur_it.particles["beam"]

    n_particles = len(df)

    id = df["id"].values.astype(np.uint)
    charge = df["charge"].values.astype(np.float64)
    mass = df["mass"].values.astype(np.float64)
    momentum_x = df["momentum_x"].values.astype(np.float64)
    momentum_y = df["momentum_y"].values.astype(np.float64)
    momentum_z = df["momentum_z"].values.astype(np.float64)
    particlePos_x = df["position_x"].values.astype(np.float64)
    particlePos_y = df["position_y"].values.astype(np.float64)
    particlePos_z = df["position_z"].values.astype(np.float64)
    particleOff_x = df["positionOffset_x"].values.astype(np.float64)
    particleOff_y = df["positionOffset_y"].values.astype(np.float64)
    particleOff_z = df["positionOffset_z"].values.astype(np.float64)
    weighting = df["weighting"].values.astype(np.float64)

    d_id = Dataset(np.dtype("uint"), extent=[n_particles])
    beam["id"][SCALAR].reset_dataset(d_id)
    d = Dataset(np.dtype("float64"), extent=[n_particles])
    beam["charge"][SCALAR].reset_dataset(d)
    beam["mass"][SCALAR].reset_dataset(d)
    beam["momentum"]["x"].reset_dataset(d)
    beam["momentum"]["y"].reset_dataset(d)
    beam["momentum"]["z"].reset_dataset(d)
    beam["position"]["x"].reset_dataset(d)
    beam["position"]["y"].reset_dataset(d)
    beam["position"]["z"].reset_dataset(d)
    beam["positionOffset"]["x"].reset_dataset(d)
    beam["positionOffset"]["y"].reset_dataset(d)
    beam["positionOffset"]["z"].reset_dataset(d)
    beam["weighting"][SCALAR].reset_dataset(d)

    beam["id"][SCALAR].store_chunk(id)
    beam["charge"][SCALAR].store_chunk(charge)
    beam["mass"][SCALAR].store_chunk(mass)
    beam["momentum"]["x"].store_chunk(momentum_x)
    beam["momentum"]["y"].store_chunk(momentum_y)
    beam["momentum"]["z"].store_chunk(momentum_z)
    beam["position"]["x"].store_chunk(particlePos_x)
    beam["position"]["y"].store_chunk(particlePos_y)
    beam["position"]["z"].store_chunk(particlePos_z)
    beam["positionOffset"]["x"].store_chunk(particleOff_x)
    beam["positionOffset"]["y"].store_chunk(particleOff_y)
    beam["positionOffset"]["z"].store_chunk(particleOff_z)
    beam["weighting"][SCALAR].store_chunk(weighting)

    f.flush()
    cur_it.close()
    f.close()

def simulation_and_analysis(executable):
    os.system("./" + executable + " inputs_2d diag1.file_prefix=diags/openpmd_files/")
    do_analysis("./diags/openpmd_files/openpmd_000280.h5")

def main() :
    # Create a custom openPMD file for particles
    h5name = "beam_particle.h5"     #supported: '.h5', '.bp/'
    h5_create(h5name, n_particles = 2000)

    # Launch simulation & do the analysis
    executables = glob.glob("*.ex")
    if len(executables) == 1 :
        simulation_and_analysis(executables[0])
    else :
        assert(False)

    # Do the checksum test
    #filename_end = "diags/plotfiles/plt000251/"
    #test_name = os.path.split(os.getcwd())[1]
    #checksumAPI.evaluate_checksum(test_name, filename_end)
    #print('Passed')

if __name__ == "__main__":
    main()
